"""Independent state/path price, Greek, boundary and residual diagnostics."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from . import ten_contract_models as models
from .evaluate_paths import simulate_states
from ..barrier_heat_kernel import knockout,vanilla_tensor
from ..analytic_bs import floating_lookback_price,barrier_price

DATES=np.array([1,4,8,16,32,48,64,80,96,112,120,127])


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def lookback_exact(t,s,y,option):
    tau=1-t;sd=.4*torch.sqrt(tau);r=.03;sig=.4
    cdf=lambda x:.5*torch.erfc(-x/math.sqrt(2.))
    z=torch.log(s/y);discount=torch.exp(-r*tau);a=sig**2/(2*r);power=2*r/sig**2
    dp=(z+(r+.5*sig**2)*tau)/sd
    dm=(z+(r-.5*sig**2)*tau)/sd
    mirror=(-z+(r-.5*sig**2)*tau)/sd
    if option=='call':
        return s*cdf(dp)-y*discount*cdf(dm)+discount*s*a*(y/s)**power*cdf(mirror)-s*a*cdf(-dp)
    return y*discount*cdf(-dm)+s*(1+a)*cdf(dp)-s*discount*a*(y/s)**power*cdf(-mirror)-s


def reference(case,t,s,y,hit):
    c=models.contract(case)
    if case.startswith('lookback'):return lookback_exact(t,s,y,c.option_type)
    return knockout(s,1-t,1.,c.barrier,.03,.4,c.option_type,c.direction,hit)


def calculate(fn,arrays,batch=512):
    result={k:[] for k in ('price','delta','gamma')}
    for start in range(0,len(arrays['s']),batch):
        t=torch.tensor(arrays['t'][start:start+batch],dtype=torch.float64)
        s=torch.tensor(arrays['s'][start:start+batch],dtype=torch.float64,requires_grad=True)
        y=torch.tensor(arrays['y'][start:start+batch],dtype=torch.float64)
        h=torch.tensor(arrays['hit'][start:start+batch],dtype=torch.bool)
        v=fn(t,s,y,h);d=torch.autograd.grad(v.sum(),s,create_graph=True)[0]
        g=torch.autograd.grad(d.sum(),s)[0]
        for key,x in zip(result,(v,d,g)):result[key].append(x.detach().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def datasets(case,paths):
    option=models.contract(case).option_type;look=case.startswith('lookback')
    generator_option=option if look else ('put' if case.startswith('up') else 'call')
    times,spots,extrema=simulate_states(models.lb.Market(),generator_option,paths,128,20261202)
    hit=np.zeros_like(spots,dtype=bool) if look else (extrema>=1.2 if case.startswith('up') else extrema<=.8)
    path=dict(t=np.broadcast_to(times[DATES],spots[:,DATES].shape).ravel(),
              s=spots[:,DATES].ravel(),y=extrema[:,DATES].ravel(),hit=hit[:,DATES].ravel())
    ts=np.array([0.,.25,.5,.75,.9,.95,.99,.999])
    if look:
        ratios=np.linspace(.2,1.,31) if option=='call' else np.linspace(1.,2.5,31)
        t,s,ratio=np.meshgrid(ts,np.linspace(.6,1.8,31),ratios,indexing='ij')
        surface=dict(t=t.ravel(),s=s.ravel(),y=(s*ratio).ravel(),hit=np.zeros(t.size,dtype=bool))
    else:
        c=models.contract(case)
        ratios=np.unique(np.concatenate([np.linspace(.05,.999,81),[.99,.995,.9999,.99999]])) if c.direction=='up' else np.unique(np.concatenate([np.linspace(1.001,3.,81),[1.00001,1.0001,1.005,1.01]]))
        t,s=np.meshgrid(ts,c.barrier*ratios,indexing='ij')
        surface=dict(t=t.ravel(),s=s.ravel(),y=s.ravel(),hit=np.zeros(t.size,dtype=bool))
    return dict(path=path,surface=surface),dict(times=times,spots=spots,extrema=extrema,hit=hit)


def masks(case,a):
    tau=1-a['t'];alive=~a['hit'];c=models.contract(case)
    near=(a['y']/a['s']>=.96 if c.option_type=='call' else a['y']/a['s']<=1.04) if case.startswith('lookback') else np.abs(np.log(a['s']/c.barrier))<=.05
    return dict(all=np.ones(len(tau),dtype=bool),alive=alive,near_boundary=alive&near,
                near_expiry=alive&(tau<=.05),near_expiry_boundary=alive&(tau<=.05)&near,hit=a['hit'])


def error_stats(pred,ref,floor):
    e=pred-ref;absolute=np.abs(e);eligible=np.abs(ref)>=floor
    return dict(points=len(e),mae=float(absolute.mean()),rmse=float(np.sqrt(np.mean(e**2))),
        relative_l2=float(np.linalg.norm(e)/max(np.linalg.norm(ref),1e-30)),bias=float(e.mean()),
        max_absolute_error=float(absolute.max()),absolute_p95=float(np.quantile(absolute,.95)),
        relative_floor=floor,relative_eligible=int(eligible.sum()),relative_excluded=int((~eligible).sum()),
        point_relative_p95=float(np.quantile(absolute[eligible]/np.abs(ref[eligible]),.95)) if eligible.any() else None,
        point_relative_p99=float(np.quantile(absolute[eligible]/np.abs(ref[eligible]),.99)) if eligible.any() else None,
        wrong_sign_fraction=float(np.mean(pred[eligible]*ref[eligible]<0)) if eligible.any() else None)


def verify_reference():
    rows=[]
    for case in models.CASES:
        arrays,_=datasets(case,8)
        a=arrays['surface'];ids=np.linspace(0,len(a['s'])-1,24,dtype=int)
        a={k:v[ids].copy() for k,v in a.items()}
        # Keep FD probes strictly inside legal states and away from the barrier.
        if case.startswith('lookback'):
            a['y']=a['s']*(.9 if case.endswith('call') else 1.1)
        else:
            c=models.contract(case);a['s']=np.minimum(a['s'],.95*c.barrier) if c.direction=='up' else np.maximum(a['s'],1.05*c.barrier)
        got=calculate(lambda t,s,y,h:reference(case,t,s,y,h),a)
        c=models.contract(case)
        scalar=np.array([floating_lookback_price(s,1-t,.03,.4,c.option_type,
                    **({'running_min':y} if c.option_type=='call' else {'running_max':y}))
                    if case.startswith('lookback') else barrier_price(s,1.,c.barrier,1-t,.03,.4,c.option_type,c.direction+'-and-out')
                    for t,s,y in zip(a['t'],a['s'],a['y'])])
        assert np.max(abs(got['price']-scalar))<1e-10,(case,'scalar')
        # Numerical derivative sanity at the chosen fixed extremum/history.
        deriv=[]
        for relative_step in (1e-4,5e-5):
            step=relative_step*a['s'];plus=dict(a,s=a['s']+step);minus=dict(a,s=a['s']-step)
            def prices(v):
                with torch.no_grad():return reference(case,*[torch.tensor(v[k],dtype=torch.bool if k=='hit' else torch.float64) for k in ('t','s','y','hit')]).numpy()
            vp,vm=prices(plus),prices(minus)
            deriv.append(((vp-vm)/(2*step),(vp-2*got['price']+vm)/step**2))
        row=dict(case=case,max_scalar_difference=float(np.max(abs(got['price']-scalar))))
        for q,ref,coarse,fine in zip(('delta','gamma'),(got['delta'],got['gamma']),deriv[0],deriv[1]):
            row[q+'_fd_relative_l2']=float(np.linalg.norm(ref-fine)/max(np.linalg.norm(ref),1e-12))
            row[q+'_step_halving_relative_l2']=float(np.linalg.norm(fine-coarse)/max(np.linalg.norm(ref),1e-12))
            assert row[q+'_fd_relative_l2']<.01,(case,q,row)
        rows.append(row)
    return rows


def evaluate(root,paths=2048,model_api=None):
    # Optional API keeps historical candidates reproducible while allowing
    # explicitly versioned new structures to share the exact same audit.
    models_api=model_api
    if models_api is None:
        from . import ten_contract_models as models_api
    root=Path(root);out=root/'evaluation';out.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    write(out/'reference_checks.json',verify_reference())
    summaries=[];by_time=[];initials=[];conditions=[];residual_rows=[];timings=[]
    for case in models.CASES:
        arrays,full=datasets(case,paths);np.savez_compressed(out/(case+'_path_states.npz'),**full)
        refs={name:calculate(lambda t,s,y,h:reference(case,t,s,y,h),a) for name,a in arrays.items()}
        vanilla={name:calculate(lambda t,s,y,h:vanilla_tensor(s,1-t,1.,.03,.4,models.contract(case).option_type),a) for name,a in arrays.items()} if not case.startswith('lookback') else None
        predictions={}
        print('evaluating '+case,flush=True)
        for mode in models.MODES:
            run=root/case/mode;saved=torch.load(run/'model.pt',map_location='cpu')
            model=models_api.make_model(case,mode);model.load_state_dict(saved['state_dict']);model.eval()
            a=saved['args'];assert a.get('pde_weight',1.)==1. and a.get('long_tau_weight',1.)==1.
            with torch.no_grad():
                one=torch.ones(1);zero=torch.zeros(1)
                pred=models_api.value(model,case,mode,zero,one,one).item()
                exact=reference(case,zero,one,one,torch.zeros(1,dtype=torch.bool)).item()
            flavors=('value',) if case.startswith('lookback') else ('out','in')
            for flavor in flavors:
                is_in=flavor=='in'
                c0=vanilla_tensor(one,one,1.,.03,.4,models.contract(case).option_type).item() if is_in else 0.
                pp=c0-pred if is_in else pred;rr=c0-exact if is_in else exact
                initials.append(dict(contract=case+('' if flavor=='value' else '_'+flavor),mode=mode,
                    prediction=pp,analytic=rr,absolute_error=abs(pp-rr),relative_error=abs(pp-rr)/abs(rr),derived_by_parity=is_in))
            for name,a in arrays.items():
                actual=calculate(lambda t,s,y,h:models_api.value(model,case,mode,t,s,y,h),a)
                for q,v in actual.items():predictions[mode+'_'+name+'_'+q]=v
                # Preserve all sampled states and raw predictions for inspection.
                for flavor in flavors:
                    is_in=flavor=='in';label=case+('' if flavor=='value' else '_'+flavor)
                    for q in ('price','delta','gamma'):
                        pp=vanilla[name][q]-actual[q] if is_in else actual[q]
                        rr=vanilla[name][q]-refs[name][q] if is_in else refs[name][q]
                        for region,mask in masks(case,a).items():
                            if mask.any():summaries.append(dict(contract=label,mode=mode,dataset=name,quantity=q,region=region,
                                **error_stats(pp[mask],rr[mask],1e-6 if q=='price' else 1e-3)))
                        for t in np.unique(a['t']):
                            mask=a['t']==t
                            by_time.append(dict(contract=label,mode=mode,dataset=name,quantity=q,t=float(t),
                                **error_stats(pp[mask],rr[mask],1e-6 if q=='price' else 1e-3)))
            # Exact terminal, sticky history, and fixed-S extremum/barrier checks.
            c=models.contract(case);s=torch.linspace(.6,1.8,101);t=torch.linspace(0,.999,101)
            if case.startswith('lookback'):
                y=s*(.8 if c.option_type=='call' else 1.2);payoff=s-y if c.option_type=='call' else y-s
                terminal=models_api.value(model,case,mode,torch.ones_like(t),s,y)
                yy=s.clone().requires_grad_(True);bd=models_api.value(model,case,mode,t,s,yy)
                boundary=torch.autograd.grad(bd.sum(),yy)[0]
                dead_error=0.;cap_error=0.
            else:
                s=torch.linspace(.05*c.barrier,.9999*c.barrier,101) if c.direction=='up' else torch.linspace(1.0001*c.barrier,3*c.barrier,101)
                payoff=torch.relu(s-1) if c.option_type=='call' else torch.relu(1-s)
                terminal=models_api.value(model,case,mode,torch.ones_like(t),s)
                boundary=models_api.value(model,case,mode,t,torch.full_like(t,c.barrier))
                dead_error=models_api.value(model,case,mode,t,s,hit=True).abs().max().item()
                caps=vanilla_tensor(s,1-t,1.,.03,.4,c.option_type)
                values=models_api.value(model,case,mode,t,s)
                cap_error=max(torch.relu(values-caps).max().item(),torch.relu(-values).max().item())
            conditions.append(dict(case=case,mode=mode,terminal_rmse=(terminal-payoff).square().mean().sqrt().item(),
                boundary_rmse=boundary.square().mean().sqrt().item(),boundary_type='extremum_partial' if case.startswith('lookback') else 'value',
                sticky_hit_max_error=dead_error,max_price_bound_violation=cap_error))
            # Fresh, paired collocation states; saved seed is independent of training.
            torch.manual_seed(20261203)
            if case.startswith('lookback'):
                t,s,y=models.lb.sample_interior(4096,models.lb.Market(),c,'cpu',.01,.001,'full')
            else:
                t,s=models.bh.sample(4096,models.bh.Market(),c,'cpu',torch.float64);y=None
            residual=[]
            for start in range(0,len(t),256):
                r=models_api.pde(model,case,mode,t[start:start+256],s[start:start+256],None if y is None else y[start:start+256])
                residual.append(r.detach().numpy())
            r=np.concatenate(residual)
            for region,mask in [('all',np.ones(len(t),dtype=bool)),('near_expiry',(t.numpy()>=.95)),('long_remaining_time',(t.numpy()<=.5))]:
                if mask.any():residual_rows.append(dict(case=case,mode=mode,region=region,points=int(mask.sum()),mse=float(np.mean(r[mask]**2)),mean=float(r[mask].mean()),absolute_p95=float(np.quantile(abs(r[mask]),.95))))
            # Warmed CPU value inference on fixed 128 states; separate from training time.
            t=t[:128];s=s[:128];y=None if y is None else y[:128]
            with torch.no_grad():
                for _ in range(3):models_api.value(model,case,mode,t,s,y)
                durations=[]
                for _ in range(10):
                    started=time.perf_counter();models_api.value(model,case,mode,t,s,y);durations.append(time.perf_counter()-started)
            timings.append(dict(case=case,mode=mode,queries=128,median_batch_seconds=float(np.median(durations)),
                training_seconds=json.loads((run/'metrics.json').read_text())['elapsed_seconds']))
        for name,a in arrays.items():
            for key,v in a.items():predictions[name+'_state_'+key]=v
            for key,v in refs[name].items():predictions[name+'_reference_'+key]=v
        np.savez_compressed(out/(case+'_predictions.npz'),**predictions)
        # Write incrementally so completed cases survive an interrupted audit.
        for filename,rows in [('metrics.csv',summaries),('by_time.csv',by_time),('initial_prices.csv',initials),
                              ('conditions.csv',conditions),('pde_residuals.csv',residual_rows),('timings.csv',timings)]:csv_write(out/filename,rows)
    compact=[]
    for initial in initials:
        record=dict(initial)
        for dataset in ('surface','path'):
            for q in ('price','delta','gamma'):
                key=dict(contract=initial['contract'],mode=initial['mode'],dataset=dataset,quantity=q,region='all')
                row=next(x for x in summaries if all(x[k]==v for k,v in key.items()))
                record[dataset+'_'+q+'_relative_l2']=row['relative_l2']
                date_rows=[x for x in by_time if all(x[k]==v for k,v in key.items() if k!='region')]
                record[dataset+'_'+q+'_equal_date_rms']=float(np.sqrt(np.mean([x['relative_l2']**2 for x in date_rows])))
        compact.append(record)
    csv_write(out/'comparison.csv',compact);write(out/'comparison.json',compact)
    write(out/'evaluation_manifest.json',dict(status='complete',paths=paths,steps=128,path_dates=DATES.tolist(),
        path_seed=20261202,heldout_pde_seed=20261203,contracts=10,method_cells=len(compact),
        reference='Closed-form tensor prices and AD Greeks, cross-checked against scalar formulas and finite-difference step halving.',
        monitoring='Exact GBM endpoints plus continuous Brownian-bridge extrema; discrete observation dates do not discretize barrier monitoring.',
        limitations='One training seed. State grid includes hypothetical histories. Path dates are correlated. Relative Greek tails use a .001 floor with excluded counts. Hard boundary and hit checks do not certify interior prices. In prices/Greeks are vanilla minus out, never independently trained.'))
    plot_results(root,compact)
    return compact


def plot_results(root,compact):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'evaluation'
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,case in zip(axes.ravel(),models.CASES):
        for mode in models.MODES:
            with (root/case/mode/'history.csv').open() as f:rows=list(csv.DictReader(f))
            y=np.array([float(r['pde']) for r in rows]);w=min(200,len(y))
            ax.semilogy(np.arange(w,len(y)+1),np.convolve(y,np.ones(w)/w,'valid'),label=mode)
        ax.set(title=case,xlabel='Updates',ylabel='Raw PDE MSE (moving average)');ax.legend()
    fig.savefig(out/'pde_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    contracts=list(dict.fromkeys(r['contract'] for r in compact));x=np.arange(len(contracts))
    for ax,q in zip(axes,('price','delta','gamma')):
        for j,mode in enumerate(models.MODES):
            y=[next(r['path_'+q+'_equal_date_rms'] for r in compact if r['contract']==c and r['mode']==mode) for c in contracts]
            ax.bar(x+(j-1)*.25,np.maximum(y,1e-12),width=.25,label=mode)
        ax.set_yscale('log');ax.set_xticks(x);ax.set_xticklabels(contracts,rotation=80)
        ax.set(title=q,ylabel='Path equal-date RMS relative L2');ax.legend()
    fig.savefig(out/'path_comparison.png',dpi=160);plt.close(fig)
