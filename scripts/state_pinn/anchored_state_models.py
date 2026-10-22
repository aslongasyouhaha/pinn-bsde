"""Revised hard structures and explicit optional MC anchors.

Old rational/tau-correction experiments stay reproducible in ten_contract_models.
"""
import math
import torch
from . import ten_contract_models as old
from ..barrier_heat_kernel import vanilla_tensor

CASES=old.CASES
MODES=old.MODES
contract=old.contract
lb=old.lb
bh=old.bh
make_model=old.make_model


def value(model,case,mode,t,s,y=None,hit=False):
    if mode!='hard':return old.value(model,case,mode,t,s,y,hit)
    tau=(1-t).clamp_min(0.);safe_tau=torch.where(tau>0,tau,torch.ones_like(tau))
    c=contract(case)
    if case.startswith('lookback'):
        x=torch.log(s/y) if c.option_type=='call' else torch.log(y/s)
        q=x.square()/(x.square()+.4**2*safe_tau)
        features=torch.stack([t,torch.log(s),q],1)
        radius=torch.sqrt(x.square()+.4**2*safe_tau)
        base=s*(-torch.expm1(-radius) if c.option_type=='call' else torch.expm1(radius))
        # sqrt(tau), not tau: the network may learn the leading corner term.
        v=base+s*torch.sqrt(safe_tau)*model(features)
        payoff=s-y if c.option_type=='call' else y-s
        return torch.where(tau>0,v,payoff)
    x=(1. if c.direction=='down' else -1.)*torch.log(s/c.barrier)
    z=torch.log(s);a=.4*torch.sqrt(safe_tau)
    features=torch.stack([torch.sqrt(tau),torch.log(s),
        x/torch.sqrt(x.square()+a.square()),z/torch.sqrt(z.square()+a.square())],1)
    # Positive learned shape scale; raw=0 gives scale 1. This smooth gate is
    # linear at the absorbing boundary and has no fixed sqrt(tau) tail at
    # a fixed interior spot. No auxiliary heat equation is solved.
    scale=(torch.nn.functional.softplus(model(features))/math.log(2.)).clamp_min(1e-30)
    fraction=torch.tanh(x.clamp_min(0.)/a*scale)
    vanilla=vanilla_tensor(s,tau,1.,.03,.4,c.option_type).clamp_min(0.)
    v=vanilla*fraction
    payoff=torch.relu(s-1) if c.option_type=='call' else torch.relu(1-s)
    v=torch.where(tau>0,v,payoff)
    alive=(x>0)&~torch.as_tensor(hit,dtype=torch.bool,device=s.device)
    return torch.where(alive,v,(s.square()+t.square())*0.)


def pde(model,case,mode,t,s,y=None,hit=False):
    t=t.detach().requires_grad_(True);s=s.detach().requires_grad_(True)
    v=value(model,case,mode,t,s,y,hit)
    vt,vs=torch.autograd.grad(v.sum(),(t,s),create_graph=True)
    vss=torch.autograd.grad(vs.sum(),s,create_graph=True)[0]
    return vt+.03*s*vs+.5*.4**2*s.square()*vss-.03*v


def sample(case,batch):
    c=contract(case)
    if case.startswith('lookback'):
        return lb.sample_interior(batch,lb.Market(),c,'cpu',.01,.001,'full')
    t,s=bh.sample(batch,bh.Market(),c,'cpu',torch.float64)
    return t,s,None


def loss(model,case,mode,batch=128,anchor=None,anchor_weight=0.):
    t,s,y=sample(case,batch);lp=pde(model,case,mode,t,s,y).square().mean()
    c=contract(case)
    # Draw exactly the same batches in all modes, including the two hard runs.
    if case.startswith('lookback'):
        st=lb.sample_spot(batch,lb.Market(),'cpu')
        yt=(st*torch.rand(batch)).clamp_min(1e-5) if c.option_type=='call' else st*(1+2*torch.rand(batch))
        payoff=st-yt if c.option_type=='call' else yt-st
        vt=value(model,case,mode,torch.ones_like(st),st,yt)
        tb=torch.rand(batch);sb=lb.sample_spot(batch,lb.Market(),'cpu')
        yy=sb.clone().requires_grad_(True)
        vb=value(model,case,mode,tb,sb,yy)
        boundary=torch.autograd.grad(vb.sum(),yy,create_graph=True)[0].square().mean()
        nonnegative=torch.relu(-value(model,case,mode,t,s,y)).square().mean()
        terminal=(vt-payoff).square().mean()
        total=lp+2*terminal+.1*boundary+.01*nonnegative
    else:
        _,st,_=sample(case,batch);payoff=torch.relu(st-1) if c.option_type=='call' else torch.relu(1-st)
        terminal=(value(model,case,mode,torch.ones_like(st),st)-payoff).square().mean()
        tb=torch.rand(batch);boundary=value(model,case,mode,tb,torch.full_like(tb,c.barrier)).square().mean()
        nonnegative=lp*0.
        total=lp+2*terminal+5*boundary if mode=='soft' else lp
    anchor_loss=lp*0.
    if anchor_weight:
        assert not case.startswith('lookback') and anchor is not None
        one=torch.ones(1);anchor_loss=(value(model,case,mode,torch.zeros(1),one)-anchor).square().mean()
        total=total+anchor_weight*anchor_loss
    return total,dict(pde=lp.item(),terminal=terminal.item(),boundary=boundary.item(),
        nonnegative=nonnegative.item(),anchor=anchor_loss.item(),anchor_weight=anchor_weight)


def checks():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);rows=[]
    for case in CASES:
        torch.manual_seed(912);m=make_model(case,'hard');c=contract(case)
        with torch.no_grad():
            for p in m.parameters():p.add_(.03*torch.randn_like(p))
        t=torch.tensor([0.,.5,.99]);s=torch.tensor([.9,1.,1.1])
        if case.startswith('lookback'):
            y=s*(.8 if c.option_type=='call' else 1.2);g=s-y if c.option_type=='call' else y-s
            term=value(m,case,'hard',torch.ones_like(t),s,y)
            yy=s.clone().requires_grad_(True);v=value(m,case,'hard',t,s,yy)
            error=torch.autograd.grad(v.sum(),yy)[0].abs().max().item()
        else:
            g=torch.relu(s-1) if c.option_type=='call' else torch.relu(1-s)
            term=value(m,case,'hard',torch.ones_like(t),s)
            error=value(m,case,'hard',t,torch.full_like(t,c.barrier)).abs().max().item()
            assert torch.equal(value(m,case,'hard',t,s,hit=True),torch.zeros_like(s))
            v=value(m,case,'hard',t,s);cap=vanilla_tensor(s,1-t,1.,.03,.4,c.option_type)
            assert (v>=0).all() and (v<=cap+1e-14).all()
        assert torch.equal(term,g) and error<1e-11
        l,_=loss(m,case,'hard',32);l.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        rows.append(dict(case=case,terminal_exact=True,boundary_error=error,finite_backward=True))
    # Show that the lookback corner coefficient is now trainable at arbitrarily
    # small positive maturity, unlike the old tau-scaled correction.
    m=make_model('lookback_call','hard');t=torch.tensor([1-1e-8]);one=torch.ones(1)
    v=value(m,'lookback_call','hard',t,one,one)
    grad=torch.autograd.grad(v.sum(),m.net[-1].bias)[0].item()
    assert abs(grad)>1e-6
    rows.append(dict(lookback_corner_coefficient_trainable=True,output_bias_gradient=grad))
    return rows
