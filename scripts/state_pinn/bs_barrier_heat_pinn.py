"""Structured barrier PINN: driftless absorbing heat lift times a learned correction."""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
import torch
from torch import nn
from ..barrier_heat_kernel import knockout, vanilla_tensor, driftless_hit
from ..analytic_bs import barrier_price
from ..artifacts import RunRecorder

@dataclass(frozen=True)
class Market:
    s0:float=1.;r:float=.03;sigma:float=.4;T:float=1.

@dataclass(frozen=True)
class BarrierContract:
    direction:str='up';option_type:str='call';strike:float=1.;barrier:float=1.2

class MLP(nn.Module):
    def __init__(self,hidden=64,depth=3,price_parameterization='unbounded'):
        super().__init__()
        if price_parameterization not in ('unbounded','bounded_odds','bounded_softcap','bounded_calibrated_odds'):raise ValueError('Unknown price parameterization')
        self.price_parameterization=price_parameterization
        layers=[];width=4
        for _ in range(depth):layers.extend([nn.Linear(width,hidden),nn.Tanh()]);width=hidden
        last=nn.Linear(width,1)
        # Zero correction starts at the baseline of the selected parameterization.
        nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
        layers.append(last);self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x).squeeze(-1)


def model_value(model,market,contract,t,s,hit=False):
    tau=torch.clamp(market.T-t,min=0.)
    safe_tau=torch.where(tau>0,tau,torch.full_like(tau,market.T))
    sign=1. if contract.direction=='down' else -1.
    x=sign*torch.log(s/contract.barrier);z=torch.log(s/contract.strike)
    variance=market.sigma**2*safe_tau
    qb=x/torch.sqrt(x.square()+variance)
    qk=z/torch.sqrt(z.square()+variance)
    features=torch.stack([torch.sqrt(safe_tau/market.T),torch.log(s/market.s0),qb,qk],dim=1)
    H=knockout(s,tau,contract.strike,contract.barrier,market.r,market.sigma,contract.option_type,contract.direction,hit,driftless=True)
    scale=torch.where(tau>0,market.sigma*torch.sqrt(safe_tau),torch.zeros_like(tau))
    # Equivalent to H + H*(exp(scale*N)-1). Square-root time permits the
    # drift correction near strike/barrier layers; q inputs resolve both.
    raw=model(features)
    if getattr(model,'price_parameterization','unbounded')=='unbounded':
        return H*torch.exp(scale*raw)
    tiny=torch.finfo(s.dtype).tiny
    vanilla=vanilla_tensor(s,tau,contract.strike,market.r,market.sigma,contract.option_type).clamp_min(0.)
    if model.price_parameterization in ('bounded_odds','bounded_calibrated_odds'):
        hit_base=driftless_hit(s,tau,contract.strike,contract.barrier,market.r,market.sigma,contract.option_type,contract.direction)
        # Only representational-underflow guards, not a price projection.
        odds=torch.log(H.clamp_min(tiny))-torch.log(hit_base.clamp_min(tiny))
        if model.price_parameterization=='bounded_calibrated_odds':
            # Calibrate auxiliary out/in odds by C_aux/C. At zero network,
            # B = H/[1+(C_aux/C-1)*H/C_aux], which stays near H when the
            # auxiliary out share is small and approaches C when it is one.
            log_aux=torch.logaddexp(torch.log(H.clamp_min(tiny)),torch.log(hit_base.clamp_min(tiny)))
            odds=odds+log_aux-torch.log(vanilla.clamp_min(tiny))
    else:
        # B = 2*C*H/(C+H+sqrt((C-H)^2+4*epsilon*C*H)).
        # B approximates min(H,C); its relative deficit is at most
        # sqrt(epsilon)/(1+sqrt(epsilon)). Epsilon shrinks as tau^4,
        # so smoothing changes the terminal limit by O(tau^2).
        # Normalize before squaring, and rationalize the odds on H>C to
        # avoid cancellation. Both branches implement log(B/(C-B)).
        positive_h=H.clamp_min(0.)
        normalizer=(positive_h+vanilla).clamp_min(1e-150)
        h=positive_h/normalizer;c=vanilla/normalizer
        epsilon=1e-4*(safe_tau/market.T).pow(4)
        root=torch.sqrt(((c-h).square()+4*epsilon*c*h).clamp_min(tiny))
        below=torch.log((2*h).clamp_min(tiny))-torch.log((c-h+root).clamp_min(tiny))
        above=torch.log((h-c+root).clamp_min(tiny))-torch.log((2*epsilon*c).clamp_min(tiny))
        odds=torch.where(h<=c,below,above)
    fraction=torch.sigmoid(odds+scale*raw)
    value=vanilla*fraction
    payoff=torch.relu(s-contract.strike) if contract.option_type=='call' else torch.relu(contract.strike-s)
    value=torch.where(tau>0,value,payoff)
    alive=(x>0)&~torch.as_tensor(hit,dtype=torch.bool,device=s.device)
    return torch.where(alive,value,(s.square()+t.square())*0.)


def sample(batch,market,contract,device,dtype):
    n=batch//4
    tau=torch.cat([.001+.149*torch.rand(n,device=device,dtype=dtype),
                   .15+.7*torch.rand(batch-2*n,device=device,dtype=dtype),
                   .85+.15*torch.rand(n,device=device,dtype=dtype)])*market.T
    t=market.T-tau[torch.randperm(batch,device=device)]
    near=batch//3;eps=8*torch.finfo(dtype).eps
    distance=eps+.2*torch.rand(near,device=device,dtype=dtype)
    if contract.direction=='up':
        ratio=torch.cat([1-distance,.05+(1-eps-.05)*torch.rand(batch-near,device=device,dtype=dtype)])
    else:
        ratio=torch.cat([1+distance,1+eps+(2-eps)*torch.rand(batch-near,device=device,dtype=dtype)])
    s=contract.barrier*ratio[torch.randperm(batch,device=device)]
    return t,s


def loss_value(model,market,contract,batch,device,long_tau_weight=1.,pde_weight=1.):
    dtype=next(model.parameters()).dtype
    t,s=sample(batch,market,contract,device,dtype)
    t=t.detach().requires_grad_(True);s=s.detach().requires_grad_(True)
    value=model_value(model,market,contract,t,s)
    vt,vs=torch.autograd.grad(value.sum(),(t,s),create_graph=True)
    vss=torch.autograd.grad(vs.sum(),s,create_graph=True)[0]
    residual=vt+market.r*s*vs+.5*market.sigma**2*s.square()*vss-market.r*value
    squared=residual.square()
    pde=squared.mean()
    # Keep sampling and the unweighted PDE diagnostic unchanged. This optional
    # ablation emphasizes remaining maturities >= T/2 without price labels.
    weights=torch.where(market.T-t>=.5*market.T,long_tau_weight,1.)
    weighted_pde=(weights*squared).mean()
    vanilla=vanilla_tensor(s,market.T-t,contract.strike,market.r,market.sigma,contract.option_type)
    cap=torch.relu(value-vanilla).square().mean()
    nonnegative=torch.relu(-value).square().mean()
    pde_contribution=pde_weight*weighted_pde
    # Bounded outputs enforce both inequalities structurally. Their training
    # objective is PDE-only; retain bound diagnostics for independent checks.
    if model.price_parameterization=='unbounded':
        total=pde_contribution+.1*cap+.01*nonnegative
    else:
        total=pde_contribution
    return total,dict(pde=pde.item(),weighted_pde=weighted_pde.item(),pde_contribution=pde_contribution.item(),cap=cap.item(),nonnegative=nonnegative.item())


def verify_structure():
    torch.set_num_threads(1);torch.manual_seed(1);results=[]
    for direction in ['up','down']:
        for option in ['call','put']:
            market=Market();contract=BarrierContract(direction,option,1.,1.2 if direction=='up' else .8)
            model=MLP().double();t=torch.tensor([0.,.5,.99],dtype=torch.float64);s=torch.ones_like(t)
            H=knockout(s,1-t,1.,contract.barrier,.03,.4,option,direction,driftless=True)
            assert torch.equal(model_value(model,market,contract,t,s),H)
            with torch.no_grad():model.net[-1].weight.normal_(0,.1);model.net[-1].bias.fill_(.2)
            s=contract.barrier*torch.tensor([.5,.8,.9999] if direction=='up' else [1.0001,1.2,2.5],dtype=torch.float64)
            expiry=model_value(model,market,contract,torch.ones_like(s),s)
            payoff=torch.relu(s-1) if option=='call' else torch.relu(1-s)
            assert torch.equal(expiry,payoff)
            boundary=model_value(model,market,contract,t,torch.full_like(t,contract.barrier))
            assert torch.equal(boundary,torch.zeros_like(t))
            assert torch.equal(model_value(model,market,contract,t,torch.ones_like(t),hit=True),torch.zeros_like(t))
            loss,parts=loss_value(model,market,contract,128,torch.device('cpu'));loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            results.append(dict(direction=direction,option=option,terminal_exact=True,boundary_exact=True,sticky_hit=True,finite_gradients=True))
    return results


def train(args):
    recorder=RunRecorder(args,__spec__.name if __spec__ else __name__)
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    device=torch.device('cpu');market=Market(args.s0,args.r,args.sigma,args.T)
    contract=BarrierContract(args.direction,args.option_type,args.strike,args.barrier)
    model=MLP(args.hidden,args.depth,args.price_parameterization).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1000,args.lr_gamma)
    for epoch in range(1,args.epochs+1):
        optimizer.zero_grad(set_to_none=True)
        loss,parts=loss_value(model,market,contract,args.batch_size,device,args.long_tau_weight,args.pde_weight)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step();scheduler.step();recorder.log(epoch,loss,parts)
        if epoch==1 or epoch%args.print_every==0:print(f'epoch={epoch:05d} loss={loss.item():.9g} pde={parts["pde"]:.9g} cap={parts["cap"]:.9g}',flush=True)
    with torch.no_grad():pred=model_value(model,market,contract,torch.tensor([0.],dtype=torch.float64),torch.tensor([market.s0],dtype=torch.float64)).item()
    ref=barrier_price(market.s0,contract.strike,contract.barrier,market.T,market.r,market.sigma,contract.option_type,contract.direction+'-and-out')
    print(f'predicted_initial_price={pred:.10g} analytic={ref:.10g} relative_error={abs(pred-ref)/max(abs(ref),1e-12):.5%}',flush=True)
    recorder.finish(model,market,contract,pred,ref)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--price-parameterization',choices=['unbounded','bounded_odds','bounded_softcap','bounded_calibrated_odds'],default='unbounded')
    p.add_argument('--direction',choices=['up','down'],required=True)
    p.add_argument('--option-type',choices=['call','put'],required=True)
    p.add_argument('--barrier',type=float,required=True)
    p.add_argument('--s0',type=float,default=1.);p.add_argument('--strike',type=float,default=1.)
    p.add_argument('--r',type=float,default=.03);p.add_argument('--sigma',type=float,default=.4);p.add_argument('--T',type=float,default=1.)
    p.add_argument('--epochs',type=int,default=15000);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--depth',type=int,default=3)
    p.add_argument('--lr',type=float,default=5e-4);p.add_argument('--lr-gamma',type=float,default=.9)
    p.add_argument('--pde-weight',type=float,default=1.,help='Global coefficient multiplying the PDE loss at every state')
    p.add_argument('--long-tau-weight',type=float,default=1.,help='PDE squared-residual weight for remaining maturity >= T/2; default preserves original objective')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--print-every',type=int,default=1500);p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    if not math.isfinite(a.pde_weight) or a.pde_weight<=0:p.error('Positive finite PDE weight required')
    if not math.isfinite(a.long_tau_weight) or a.long_tau_weight<=0:p.error('Positive finite long-tau weight required')
    if min(a.s0,a.strike,a.barrier,a.sigma,a.T,a.epochs,a.batch_size,a.hidden,a.depth,a.threads)<=0:p.error('Positive parameters required')
    if (a.direction=='up' and a.s0>=a.barrier) or (a.direction=='down' and a.s0<=a.barrier):p.error('Initial spot must be alive')
    train(a)
