"""Explicit-state models for the matched ten-contract experiment.

Barrier hard lift uses an algebraic survival *shape*, not a hitting probability
or an auxiliary PDE solution. Knock-ins are derived by parity only.
"""
from __future__ import annotations
import math
import torch
from . import bs_lookback_pinn as lb
from . import bs_barrier_heat_pinn as bh
from ..barrier_heat_kernel import vanilla_tensor

MODES=('soft','hard','heat')
CASES=('lookback_call','lookback_put','up_call','up_put','down_call','down_put')
LOOKBACK_MODES=dict(soft='direct',hard='hard_lift',heat='heat_lift')


def contract(case):
    direction,option=case.split('_')
    return (lb.LookbackContract(option) if direction=='lookback' else
            bh.BarrierContract(direction,option,1.,1.2 if direction=='up' else .8))


def make_model(case,mode):
    return (lb.MLP(64,3,LOOKBACK_MODES[mode],contract(case).option_type).double()
            if case.startswith('lookback') else bh.MLP(64,3,'bounded_odds').double())


def value(model,case,mode,t,s,y=None,hit=False):
    if case.startswith('lookback'):
        return lb.model_value(model,lb.Market(),t,s,y)
    m=bh.Market();c=contract(case)
    if mode=='heat':return bh.model_value(model,m,c,t,s,hit)
    tau=(m.T-t).clamp_min(0.)
    sign=1. if c.direction=='down' else -1.
    x=sign*torch.log(s/c.barrier);z=torch.log(s/c.strike)
    variance=m.sigma**2*tau.clamp_min(1e-14)
    features=torch.stack([torch.sqrt(tau/m.T),torch.log(s/m.s0),
        x/torch.sqrt(x.square()+variance),z/torch.sqrt(z.square()+variance)],1)
    raw=model(features)
    vanilla=vanilla_tensor(s,tau,c.strike,m.r,m.sigma,c.option_type).clamp_min(0.)
    if mode=='soft':
        # Same known vanilla bound as the H model; terminal/barrier are soft.
        v=vanilla*torch.sigmoid(raw)
    else:
        # p=x/(x+sigma*sqrt(tau)*exp(-sigma*sqrt(tau)*N)).
        # p=0 at barrier, tends to 1 at expiry for every fixed alive x>0.
        # Algebraic shape remains an explicit prior; no heat CDF/H/J is used.
        tiny=torch.finfo(s.dtype).tiny
        a=m.sigma*torch.sqrt(tau.clamp_min(1e-14))
        odds=torch.log(x.clamp_min(tiny))-torch.log(a)
        v=vanilla*torch.sigmoid(odds+a*raw)
        payoff=torch.relu(s-c.strike) if c.option_type=='call' else torch.relu(c.strike-s)
        v=torch.where(tau>0,v,payoff)
        v=torch.where(x>0,v,(s.square()+t.square())*0.)
    return torch.where(torch.as_tensor(hit,dtype=torch.bool,device=s.device),v*0.,v)


def pde(model,case,mode,t,s,y=None,hit=False):
    t=t.detach().requires_grad_(True);s=s.detach().requires_grad_(True)
    v=value(model,case,mode,t,s,y,hit)
    vt,vs=torch.autograd.grad(v.sum(),(t,s),create_graph=True)
    vss=torch.autograd.grad(vs.sum(),s,create_graph=True)[0]
    return vt+.03*s*vs+.5*.4**2*s.square()*vss-.03*v


def loss(model,case,mode,batch=128):
    if case.startswith('lookback'):
        return lb.lookback_loss(model,lb.Market(),contract(case),batch,torch.device('cpu'),
            .01,.01,.001,.1,.01,1.,2.,'full','autograd','autograd')
    m=bh.Market();c=contract(case)
    t,s=bh.sample(batch,m,c,'cpu',torch.float64)
    residual=pde(model,case,mode,t,s)
    lp=residual.square().mean()
    # Consume identical state draws in all three modes. Hard-condition errors
    # are checked here as diagnostics but are not added to their PDE objective.
    _,terminal_s=bh.sample(batch,m,c,'cpu',torch.float64)
    terminal=value(model,case,mode,torch.ones_like(terminal_s),terminal_s)
    payoff=torch.relu(terminal_s-c.strike) if c.option_type=='call' else torch.relu(c.strike-terminal_s)
    lt=(terminal-payoff).square().mean()
    tb=torch.rand(batch,dtype=torch.float64)
    boundary=value(model,case,mode,tb,torch.full_like(tb,c.barrier))
    lboundary=boundary.square().mean()
    total=lp+2*lt+5*lboundary if mode=='soft' else lp
    return total,dict(pde=lp.item(),terminal=lt.item(),boundary=lboundary.item(),
                      nonnegative=0.,cap=0.)


def structural_checks():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);rows=[]
    for case in CASES:
        for mode in MODES[1:]:
            torch.manual_seed(912);model=make_model(case,mode)
            # Arbitrary learned weights: checks are not restricted to zero head.
            with torch.no_grad():
                for parameter in model.parameters():parameter.add_(.03*torch.randn_like(parameter))
            t=torch.tensor([0.,.5,.99]);s=torch.tensor([.9,1.,1.1]);c=contract(case)
            if case.startswith('lookback'):
                y=(.8*s if c.option_type=='call' else 1.2*s)
                payoff=s-y if c.option_type=='call' else y-s
                term=value(model,case,mode,torch.ones_like(t),s,y)
                yy=s.clone().requires_grad_(True)
                v=value(model,case,mode,t,s,yy)
                bd=torch.autograd.grad(v.sum(),yy)[0].abs().max().item()
            else:
                payoff=torch.relu(s-1) if c.option_type=='call' else torch.relu(1-s)
                term=value(model,case,mode,torch.ones_like(t),s)
                bd=value(model,case,mode,t,torch.full_like(s,c.barrier)).abs().max().item()
                assert torch.equal(value(model,case,mode,t,s,hit=True),torch.zeros_like(s))
                interior=value(model,case,mode,t,s)
                cap=vanilla_tensor(s,1-t,1.,.03,.4,c.option_type)
                assert (interior>=-1e-14).all() and (interior<=cap+1e-14).all()
            assert torch.equal(term,payoff),(case,mode)
            assert bd<1e-11,(case,mode,bd)
            l,parts=loss(model,case,mode,32);l.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),(case,mode)
            if not case.startswith('lookback'):assert l.item()==parts['pde']
            rows.append(dict(case=case,mode=mode,terminal_exact=True,boundary_error=bd,
                             finite_backward=True,pde_weight=1.,long_tau_weight=1.))
    return rows
