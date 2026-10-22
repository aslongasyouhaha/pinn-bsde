"""Four BS knock-outs from one absorbing kernel; knock-ins only by in-out parity.

European, constant single barrier, continuously monitored, zero rebate, no dividends.
The driftless kernel is a candidate heat lift; the drifted kernel is the full price.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from .analytic_bs import barrier_price, bsm_price


def cdf(x):
    return .5*torch.erfc(-x/math.sqrt(2.))


def interval_exponential(k,mean,variance,lower,upper):
    """Integral exp(k*y) Normal(y; mean, variance) over the payoff-positive interval."""
    sd=torch.sqrt(variance);shift=mean+k*variance
    if math.isinf(lower):
        probability=torch.ones_like(mean) if math.isinf(upper) else cdf((upper-shift)/sd)
    elif math.isinf(upper):
        probability=cdf((shift-lower)/sd)
    else:
        lo=(lower-shift)/sd
        hi=(upper-shift)/sd
        probability=torch.where(lo>0,cdf(-lo)-cdf(-hi),cdf(hi)-cdf(lo))
    return torch.exp(k*mean+.5*k*k*variance)*probability


def vanilla_tensor(spot,tau,strike,r,sigma,option_type):
    safe_tau=torch.where(tau>0,tau,torch.ones_like(tau))
    d1=(torch.log(spot/strike)+(r+.5*sigma*sigma)*safe_tau)/(sigma*torch.sqrt(safe_tau))
    d2=d1-sigma*torch.sqrt(safe_tau);discount=torch.exp(-r*safe_tau)
    if option_type=='call':
        value=spot*cdf(d1)-strike*discount*cdf(d2);payoff=torch.relu(spot-strike)
    else:
        value=strike*discount*cdf(-d2)-spot*cdf(-d1);payoff=torch.relu(strike-spot)
    return torch.where(tau>0,value,payoff)


def knockout(spot,tau,strike,barrier,r,sigma,option_type,direction,hit=False,driftless=False):
    """Price conditional on sticky hit history; hit=True keeps the out contract dead.

    Inputs spot/tau are broadcastable tensors. Delta/Gamma at an alive barrier
    limit are inward derivatives; the exact touched state is already knocked out.
    """
    if option_type not in ('call','put') or direction not in ('up','down'):
        raise ValueError('Expected call/put and up/down')
    if min(strike,barrier,sigma)<=0 or not math.isfinite(r):raise ValueError('Invalid market/contract')
    if bool(torch.any(spot<=0)) or bool(torch.any(tau<0)):raise ValueError('Need positive spot and nonnegative tau')
    spot,tau=torch.broadcast_tensors(spot,tau)
    sign=1. if direction=='down' else -1.
    x=sign*torch.log(spot/barrier)
    alive=(x>0)&~torch.as_tensor(hit,dtype=torch.bool,device=spot.device)
    # Mask the inactive log coordinate so image terms cannot overflow on dead states.
    safe_x=torch.where(alive,x,torch.zeros_like(x))
    safe_tau=torch.where(tau>0,tau,torch.ones_like(tau))
    variance=sigma*sigma*safe_tau
    drift=0. if driftless else sign*(r-.5*sigma*sigma)
    mu1=safe_x+drift*safe_tau;mu2=-safe_x+drift*safe_tau
    image_weight=torch.exp(-2.*drift*safe_x/(sigma*sigma))
    if direction=='down':
        threshold=max(0.,math.log(strike/barrier))
        lower,upper=(threshold,float('inf')) if option_type=='call' else (0.,threshold)
    else:
        threshold=max(0.,math.log(barrier/strike))
        lower,upper=(0.,threshold) if option_type=='call' else (threshold,float('inf'))
    zero=(spot.square()+tau.square())*0.
    if upper<=lower:return zero
    mass=interval_exponential(0.,mu1,variance,lower,upper)-image_weight*interval_exponential(0.,mu2,variance,lower,upper)
    stock=barrier*(interval_exponential(sign,mu1,variance,lower,upper)-image_weight*interval_exponential(sign,mu2,variance,lower,upper))
    undiscounted=stock-strike*mass if option_type=='call' else strike*mass-stock
    value=torch.exp(-r*safe_tau)*undiscounted
    payoff=torch.relu(spot-strike) if option_type=='call' else torch.relu(strike-spot)
    value=torch.where(tau>0,value,payoff)
    return torch.where(alive,value,zero)


def driftless_hit(spot,tau,strike,barrier,r,sigma,option_type,direction):
    """Positive-integral auxiliary hit payoff, avoiding vanilla-out cancellation.

    In the zero-log-drift auxiliary model: hit contribution = payoff at y<0
    under mean x, plus the reflected positive-y image contribution (mean -x).
    Together with the driftless knockout this gives auxiliary vanilla.
    """
    spot,tau=torch.broadcast_tensors(spot,tau)
    sign=1. if direction=='down' else -1.
    raw_x=sign*torch.log(spot/barrier)
    x=torch.clamp(raw_x,min=0.)
    safe_tau=torch.where(tau>0,tau,torch.ones_like(tau));variance=sigma*sigma*safe_tau
    threshold=sign*math.log(strike/barrier)
    below=(direction=='up' and option_type=='call') or (direction=='down' and option_type=='put')
    if below:
        negative=(-float('inf'),min(0.,threshold))
        positive=(0.,max(0.,threshold))
    else:
        negative=(min(0.,threshold),0.)
        positive=(max(0.,threshold),float('inf'))
    def integral(mean,interval):
        lo,hi=interval
        if hi<=lo:return (mean.square()+variance.square())*0.
        mass=interval_exponential(0.,mean,variance,lo,hi)
        stock=barrier*interval_exponential(sign,mean,variance,lo,hi)
        return stock-strike*mass if option_type=='call' else strike*mass-stock
    value=torch.exp(-r*safe_tau)*(integral(x,negative)+integral(-x,positive))
    payoff=torch.relu(spot-strike) if option_type=='call' else torch.relu(strike-spot)
    terminal=torch.where(raw_x<=0,payoff,(spot.square()+tau.square())*0.)
    return torch.where(tau>0,value,terminal)


def price(spot,tau,strike,barrier,r,sigma,option_type,knock_type,hit=False):
    """Eight types, with every knock-in computed exclusively as vanilla minus out."""
    direction=knock_type.split('-')[0]
    if knock_type not in ('up-and-out','up-and-in','down-and-out','down-and-in'):
        raise ValueError('Unknown knock type')
    out=knockout(spot,tau,strike,barrier,r,sigma,option_type,direction,hit)
    if knock_type.endswith('-out'):return out
    return vanilla_tensor(spot,tau,strike,r,sigma,option_type)-out


def scalar(value):return float(value.detach())


def verify(output):
    from scipy.integrate import quad
    torch.set_num_threads(1);dtype=torch.float64
    rng=np.random.default_rng(20261024);checks=[];pde=[];quadrature=[]
    # Cover both sides of strike/barrier relation, negative/zero/positive rates,
    # small maturities and all eight combinations. No learned models involved.
    for direction in ['up','down']:
        for option in ['call','put']:
            for ratio in [.65,1.,1.4]:
                B=1.;K=ratio
                spots=np.array([.4,.8,.99]) if direction=='up' else np.array([1.01,1.2,2.])
                for r in [-.03,0.,.03]:
                    for tau in [.001,.1,1.]:
                        for S in spots:
                            for suffix in ['out','in']:
                                knock=f'{direction}-and-{suffix}'
                                got=scalar(price(torch.tensor(S,dtype=dtype),torch.tensor(tau,dtype=dtype),K,B,r,.4,option,knock))
                                ref=barrier_price(S,K,B,tau,r,.4,option,knock)
                                error=abs(got-ref)
                                assert error<2e-11,(direction,option,K,S,r,tau,got,ref)
                                checks.append(dict(direction=direction,option_type=option,strike=K,barrier=B,spot=S,tau=tau,r=r,knock_type=knock,
                                    kernel_price=got,legacy_analytic=ref,absolute_difference=error))
                # Independent killed-density quadrature for ordinary maturities.
                for S in spots:
                    tau=.6;r=.03;sig=.4;sign=1 if direction=='down' else -1
                    x=sign*math.log(S/B);a=sign*(r-.5*sig*sig);v=sig*sig*tau
                    density=lambda y:(math.exp(-(y-x-a*tau)**2/(2*v))-math.exp(-2*a*x/(sig*sig))*math.exp(-(y+x-a*tau)**2/(2*v)))/math.sqrt(2*math.pi*v)
                    def integrand(y):
                        st=B*math.exp(sign*y)
                        payoff=max(st-K,0) if option=='call' else max(K-st,0)
                        return math.exp(-r*tau)*payoff*density(y)
                    # Tail beyond this finite interval is negligible in these cases.
                    limit=max(12.,x+abs(a)*tau+12*math.sqrt(v))
                    kink=abs(math.log(K/B))
                    points=[kink] if 0<kink<limit else None
                    numerical=quad(integrand,0,limit,points=points,epsabs=1e-12,epsrel=1e-12)[0]
                    got=scalar(knockout(torch.tensor(S,dtype=dtype),torch.tensor(tau,dtype=dtype),K,B,r,sig,option,direction))
                    assert abs(got-numerical)<2e-11
                    quadrature.append(dict(direction=direction,option_type=option,strike=K,spot=float(S),kernel_price=got,quadrature=numerical,absolute_difference=abs(got-numerical)))
            # AD PDE check at strictly alive points, with nontrivial strike.
            B=1.2 if direction=='up' else .8;K=1.;r=.03;sig=.4
            S=(B*rng.uniform(.5,.99,128) if direction=='up' else B*rng.uniform(1.01,2.,128))
            spot=torch.tensor(S,dtype=dtype,requires_grad=True)
            tau=torch.tensor(rng.uniform(.01,1.,128),dtype=dtype,requires_grad=True)
            value=knockout(spot,tau,K,B,r,sig,option,direction)
            vt,vs=torch.autograd.grad(value.sum(),(tau,spot),create_graph=True)
            vss=torch.autograd.grad(vs.sum(),spot)[0]
            residual=-vt+r*spot*vs+.5*sig*sig*spot.square()*vss-r*value
            assert residual.abs().max()<1e-10
            H=knockout(spot,tau,K,B,r,sig,option,direction,driftless=True)
            ht,hs=torch.autograd.grad(H.sum(),(tau,spot),create_graph=True);hss=torch.autograd.grad(hs.sum(),spot)[0]
            # H obeys log-diffusion-only PDE with discounting: H_tau=D*(S^2 H_SS+S H_S)-rH.
            simplified=-ht+.5*sig*sig*(spot.square()*hss+spot*hs)-r*H
            assert simplified.abs().max()<1e-10
            full_H_residual=-ht+r*spot*hs+.5*sig*sig*spot.square()*hss-r*H
            pde.append(dict(direction=direction,option_type=option,full_kernel_pde_max=float(residual.abs().max()),
                driftless_H_own_pde_max=float(simplified.abs().max()),driftless_H_full_pde_rmse=float(full_H_residual.square().mean().sqrt())))
    # Sticky history including expiry, touching, and zero-value special contracts.
    history=[]
    for direction in ['up','down']:
        B=1.2 if direction=='up' else .8
        for option in ['call','put']:
            for tau in [0.,.5]:
                interior=1.05 if option=='call' else .95
                for S,hit in [(interior,False),(B,False),(interior,True)]:
                    s=torch.tensor(S,dtype=dtype);t=torch.tensor(tau,dtype=dtype)
                    out=scalar(price(s,t,1.,B,.03,.4,option,f'{direction}-and-out',hit))
                    inn=scalar(price(s,t,1.,B,.03,.4,option,f'{direction}-and-in',hit))
                    vanilla=bsm_price(S,1.,tau,.03,.4,option)
                    assert abs(out+inn-vanilla)<1e-12
                    if hit or S==B:assert out==0. and abs(inn-vanilla)<1e-12
                    extrema={'running_max':max(S,B) if hit else S} if direction=='up' else {'running_min':min(S,B) if hit else S}
                    for suffix,value in [('out',out),('in',inn)]:
                        old=barrier_price(S,1.,B,tau,.03,.4,option,f'{direction}-and-{suffix}',**extrema)
                        assert abs(old-value)<1e-12,(direction,option,tau,hit,S,old,value)
                    history.append(dict(direction=direction,option_type=option,tau=tau,spot=S,hit=hit,out=out,in_by_parity=inn,vanilla=vanilla))
    for name,rows in [('analytic_checks.csv',checks),('quadrature_checks.csv',quadrature),('pde_checks.csv',pde),('history_checks.csv',history)]:
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=dict(analytic_cases=len(checks),analytic_max_absolute_difference=max(x['absolute_difference'] for x in checks),
        quadrature_cases=len(quadrature),quadrature_max_absolute_difference=max(x['absolute_difference'] for x in quadrature),
        pde_checks=pde,history_cases=len(history),in_method='Vanilla minus out only',training_performed=False,
        contract='European; constant single barrier; continuous monitoring; zero rebate; no dividends',torch=torch.__version__)
    (output/'verification.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


def main(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    verify(out)
    rows=[];curves=[]
    for direction in ['up','down']:
        B=args.up_barrier if direction=='up' else args.down_barrier
        for option in ['call','put']:
            s=torch.tensor(args.s0,dtype=torch.float64);tau=torch.tensor(args.T,dtype=torch.float64)
            vo=scalar(knockout(s,tau,args.strike,B,args.r,args.sigma,option,direction))
            H=scalar(knockout(s,tau,args.strike,B,args.r,args.sigma,option,direction,driftless=True))
            vanilla=bsm_price(args.s0,args.strike,args.T,args.r,args.sigma,option)
            for suffix,val in [('out',vo),('in',vanilla-vo)]:
                rows.append(dict(option_type=option,knock_type=f'{direction}-and-{suffix}',spot=args.s0,strike=args.strike,barrier=B,
                    full_price=val,vanilla=vanilla,driftless_out_H=H if suffix=='out' else None,
                    reference=barrier_price(args.s0,args.strike,B,args.T,args.r,args.sigma,option,f'{direction}-and-{suffix}'),
                    method='Absorbing kernel' if suffix=='out' else 'Vanilla minus out (parity)'))
            for fraction in [0.,.9,.99]:
                alive_s=np.linspace(.25*B,.9999*B,301) if direction=='up' else np.linspace(1.0001*B,2.5*B,301)
                st=torch.tensor(alive_s,dtype=torch.float64);tt=torch.full_like(st,args.T*(1-fraction))
                full=knockout(st,tt,args.strike,B,args.r,args.sigma,option,direction).numpy()
                lift=knockout(st,tt,args.strike,B,args.r,args.sigma,option,direction,driftless=True).numpy()
                for si,vi,hi in zip(alive_s,full,lift):curves.append(dict(direction=direction,option_type=option,time_fraction=fraction,spot=float(si),full_price=float(vi),H=float(hi)))
    for name,data in [('eight_prices.csv',rows),('kernel_curves.csv',curves)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(direction,option) in zip(axes.ravel(),[(d,o) for d in ['up','down'] for o in ['call','put']]):
        for fraction,color in zip([0.,.9,.99],['C0','C1','C2']):
            data=[x for x in curves if x['direction']==direction and x['option_type']==option and x['time_fraction']==fraction]
            ax.plot([x['spot'] for x in data],[x['full_price'] for x in data],color=color,label=f'Full, t/T={fraction}')
            ax.plot([x['spot'] for x in data],[x['H'] for x in data],color=color,linestyle='--',label=f'Driftless H, t/T={fraction}')
        ax.set(title=f'{direction}-and-out {option}',xlabel='Spot S',ylabel='Price');ax.legend(fontsize=7)
    fig.savefig(out/'four_knockouts.png',dpi=160);plt.close(fig)
    print('Eight prices:',json.dumps(rows,indent=2));print('Saved to',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/barrier_heat_kernel')
    p.add_argument('--s0',type=float,default=1.);p.add_argument('--strike',type=float,default=1.)
    p.add_argument('--up-barrier',type=float,default=1.2);p.add_argument('--down-barrier',type=float,default=.8)
    p.add_argument('--r',type=float,default=.03);p.add_argument('--sigma',type=float,default=.4);p.add_argument('--T',type=float,default=1.)
    a=p.parse_args()
    if min(a.s0,a.strike,a.up_barrier,a.down_barrier,a.sigma,a.T)<=0 or not a.down_barrier<a.s0<a.up_barrier:
        p.error('Positive parameters and initially alive lower < spot < upper barriers are required')
    main(a)
