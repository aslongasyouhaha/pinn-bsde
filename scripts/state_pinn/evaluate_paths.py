"""Out-of-sample pathwise pricing audit for saved floating-lookback state PINNs.

Exact GBM endpoints plus conditional Brownian-bridge extrema generate continuously
monitored running extrema at observation dates. No training, tuning or path selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import numpy as np
import torch
from . import bs_lookback_pinn as solver
from ..analytic_bs import floating_lookback_price


def _write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def simulate_states(market,option,paths,steps,seed):
    rng=np.random.default_rng(seed)
    dt=market.T/steps
    increments=(market.r-.5*market.sigma**2)*dt+market.sigma*math.sqrt(dt)*rng.standard_normal((paths,steps))
    log_s=np.concatenate([np.full((paths,1),math.log(market.s0)),math.log(market.s0)+np.cumsum(increments,axis=1)],axis=1)
    u=np.maximum(rng.random((paths,steps)),np.finfo(float).tiny)
    radius=np.sqrt(np.diff(log_s,axis=1)**2-2*market.sigma**2*dt*np.log(u))
    segment_extremum=.5*(log_s[:,:-1]+log_s[:,1:]+(-radius if option=="call" else radius))
    extrema=np.concatenate([log_s[:,:1],segment_extremum],axis=1)
    running=(np.minimum.accumulate(extrema,axis=1) if option=="call" else np.maximum.accumulate(extrema,axis=1))
    spots,extrema=np.exp(log_s),np.exp(running)
    if option=="call":
        assert np.all(segment_extremum<=np.minimum(log_s[:,:-1],log_s[:,1:])+1e-14)
        assert np.all(extrema<=spots+1e-14) and np.all(np.diff(extrema,axis=1)<=1e-14)
    else:
        assert np.all(segment_extremum>=np.maximum(log_s[:,:-1],log_s[:,1:])-1e-14)
        assert np.all(extrema>=spots-1e-14) and np.all(np.diff(extrema,axis=1)>=-1e-14)
    return np.linspace(0,market.T,steps+1),spots,extrema


def error_stats(prediction,truth,s0,floor):
    e=np.asarray(prediction)-truth
    if e.size == 0:
        template=error_stats(np.zeros(1),np.ones(1),s0,floor)
        return {key:(0 if key in ("points","relative_eligible_points","relative_excluded_points")
            else float(floor) if key=="relative_price_floor" else None) for key in template}
    absolute=np.abs(e)
    mask=truth>=floor
    relative=absolute[mask]/truth[mask]
    result=dict(points=int(e.size),mae=float(absolute.mean()),rmse=float(np.sqrt(np.mean(e**2))),
        relative_l2=float(np.sqrt(np.sum(e**2)/max(np.sum(truth**2),1e-30))),
        bias=float(e.mean()),max_absolute_error=float(absolute.max()),negative_fraction=float(np.mean(prediction<0)),
        absolute_error_q50=float(np.quantile(absolute,.5)),absolute_error_q95=float(np.quantile(absolute,.95)),
        absolute_error_q99=float(np.quantile(absolute,.99)),relative_price_floor=float(floor),
        relative_eligible_points=int(mask.sum()),relative_excluded_points=int((~mask).sum()))
    for quantile in [.5,.9,.95,.99]:
        result[f"point_relative_q{int(quantile*100)}"]=float(np.quantile(relative,quantile)) if relative.size else None
    for threshold in [.01,.05,.1]:
        result[f"fraction_relative_above_{int(threshold*100)}pct"]=float(np.mean(relative>threshold)) if relative.size else None
    result["mae_in_units_of_s0"]=result["mae"]/s0
    result["below_floor_mae"]=float(absolute[~mask].mean()) if (~mask).any() else None
    return result


@torch.no_grad()
def audit(run_dirs,output,paths=5000,steps=128,seed=20261018,batch_size=4096,relative_floor=1e-4):
    if paths<2 or steps<2 or batch_size<1 or relative_floor<=0:
        raise ValueError("Need paths>=2, steps>=2, batch-size>=1 and positive relative floor")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    saved_models=[];labels=[];checkpoint_hashes={}
    kwargs={"map_location":"cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:kwargs["weights_only"]=False
    for run_dir in run_dirs:
        run=Path(run_dir)
        saved=torch.load(run/"model.pt",**kwargs)
        if saved_models and (saved["market"]!=saved_models[0]["market"] or saved["contract"]!=saved_models[0]["contract"]):
            raise ValueError("All checkpoints must have the same market and contract")
        label=run.parents[2].name
        if label in labels:raise ValueError("Run labels must be unique")
        labels.append(label);saved_models.append(saved)
        checkpoint_hashes[label]=hashlib.sha256((run/"model.pt").read_bytes()).hexdigest()
    market=solver.Market(**saved_models[0]["market"])
    option=saved_models[0]["contract"]["option_type"]
    if abs(market.r)<1e-12:raise ValueError("Current analytic reference requires nonzero r")
    times,spots,extrema=simulate_states(market,option,paths,steps,seed)
    print(f"Simulated {paths} independent paths with continuous extrema",flush=True)
    truth=np.empty_like(spots)
    for j,t in enumerate(times):
        if j==steps:
            truth[:,j]=(spots[:,j]-extrema[:,j] if option=="call" else extrema[:,j]-spots[:,j])
        elif j==0:
            truth[:,j]=floating_lookback_price(market.s0,market.T,market.r,market.sigma,option,running_min=market.s0,running_max=market.s0)
        else:
            truth[:,j]=[floating_lookback_price(float(s),market.T-float(t),market.r,market.sigma,option,
                **{"running_min" if option=="call" else "running_max":float(y)}) for s,y in zip(spots[:,j],extrema[:,j])]
    if not np.isfinite(truth).all() or np.min(truth)<-1e-10:raise ValueError("Invalid analytic prices")
    true_initial=float(truth[0,0])
    payoff_samples=math.exp(-market.r*market.T)*truth[:,-1]
    mc_se=float(np.std(payoff_samples,ddof=1)/math.sqrt(paths))
    config=dict(paths=paths,steps=steps,seed=seed,generator="NumPy PCG64",market=saved_models[0]["market"],
        contract=saved_models[0]["contract"],checkpoint_hashes=checkpoint_hashes,run_dirs=[str(Path(x).resolve()) for x in run_dirs],
        monitoring="Exact GBM endpoints, independent conditional Brownian-bridge interval extrema, accumulated continuously",
        primary_region="All paths at observation dates strictly between 0 and T; uniform weight per path and date",
        relative_floor=relative_floor*market.s0,inference_batch=batch_size,
        note="Independent paths; observations on the same path are correlated. No preset acceptance tolerance, confidence interval, training or model selection.",
        simulation_sanity=dict(analytic_initial=true_initial,discounted_terminal_payoff_mc=float(payoff_samples.mean()),
            mc_standard_error=mc_se,z_score=(float(payoff_samples.mean())-true_initial)/mc_se if mc_se else None))
    (output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    np.savez_compressed(output/"path_states.npz",times=times,spots=spots,extrema=extrema,analytic=truth)
    t_grid=np.broadcast_to(times,spots.shape)
    flat=[x.ravel() for x in [t_grid,spots,extrema]]
    predictions={};summary=[];by_time=[];worst=[];worst_relative=[];by_path=[]
    for label,saved in zip(labels,saved_models):
        args=saved["args"];dtype=next(iter(saved["state_dict"].values())).dtype
        model=solver.MLP(args["hidden"],args["depth"],args.get("price_parameterization","direct"),args["option_type"]).to(dtype=dtype)
        model.load_state_dict(saved["state_dict"]);model.eval()
        prediction=np.empty(spots.size,dtype=np.float64)
        for offset in range(0,spots.size,batch_size):
            values=[torch.tensor(x[offset:offset+batch_size].copy(),dtype=dtype) for x in flat]
            prediction[offset:offset+batch_size]=solver.model_value(model,market,*values).numpy()
        prediction=prediction.reshape(spots.shape)
        if not np.isfinite(prediction).all():raise ValueError("Nonfinite model prices")
        predictions[label]=prediction
        regions={"intermediate":np.broadcast_to((times>0)&(times<market.T),spots.shape),
            "late_intermediate":np.broadcast_to((times>=.95*market.T)&(times<market.T),spots.shape),
            "expiry":np.broadcast_to(times==market.T,spots.shape)}
        ratio=extrema/spots
        regions["near_extremum_intermediate"]=regions["intermediate"]&((ratio>.95) if option=="call" else (ratio<1.05))
        regions["late_near_extremum"]=regions["late_intermediate"]&((ratio>.95) if option=="call" else (ratio<1.05))
        for region,mask in regions.items():
            summary.append(dict(model=label,region=region,**error_stats(prediction[mask],truth[mask],market.s0,relative_floor*market.s0)))
        for j,t in enumerate(times):
            by_time.append(dict(model=label,t=float(t),time_fraction=float(t/market.T),
                **error_stats(prediction[:,j],truth[:,j],market.s0,relative_floor*market.s0)))
        e=prediction[:,1:-1]-truth[:,1:-1]
        path_l2=np.sqrt(np.sum(e**2,axis=1)/np.maximum(np.sum(truth[:,1:-1]**2,axis=1),1e-30))
        for i in range(paths):
            by_path.append(dict(model=label,path_id=i,relative_l2=float(path_l2[i]),max_absolute_error=float(np.abs(e[i]).max())))
        indices=np.argsort(np.abs(e).ravel())[-20:][::-1]
        for k in indices:
            i,j=np.unravel_index(k,e.shape);j+=1
            worst.append(dict(model=label,path_id=int(i),t=float(times[j]),spot=float(spots[i,j]),extremum=float(extrema[i,j]),
                extremum_spot_ratio=float(ratio[i,j]),prediction=float(prediction[i,j]),analytic=float(truth[i,j]),
                absolute_error=float(abs(prediction[i,j]-truth[i,j]))))
        eligible=truth[:,1:-1]>=relative_floor*market.s0
        relative=np.full_like(e,-np.inf)
        relative[eligible]=np.abs(e[eligible])/truth[:,1:-1][eligible]
        for k in np.argsort(relative.ravel())[-min(20,int(eligible.sum())):][::-1] if eligible.any() else []:
            i,j=np.unravel_index(k,e.shape);j+=1
            worst_relative.append(dict(model=label,path_id=int(i),t=float(times[j]),spot=float(spots[i,j]),
                extremum=float(extrema[i,j]),extremum_spot_ratio=float(ratio[i,j]),
                prediction=float(prediction[i,j]),analytic=float(truth[i,j]),
                absolute_error=float(abs(prediction[i,j]-truth[i,j])),relative_error=float(relative[i,j-1])))
        print(label,json.dumps(next(row for row in summary if row["model"]==label and row["region"]=="intermediate")),flush=True)
    _write_csv(output/"pathwise_summary.csv",summary)
    _write_csv(output/"by_time.csv",by_time)
    _write_csv(output/"by_path.csv",by_path)
    _write_csv(output/"worst_states.csv",worst)
    if worst_relative:_write_csv(output/"worst_relative_states.csv",worst_relative)
    np.savez_compressed(output/"path_predictions.npz",**predictions)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,4),constrained_layout=True)
    for label in labels:
        data=[row for row in by_time if row["model"]==label and 0<row["time_fraction"]<1]
        x=[row["time_fraction"] for row in data]
        for ax,key in zip(axes,["relative_l2","mae","point_relative_q95"]):
            ax.plot(x,[row[key] for row in data],label=label.replace("state_pinn_lookback_",""))
    for ax,title in zip(axes,["Cross-path relative L2","Mean absolute error","Pointwise relative error: 95th percentile"]):
        ax.set(xlabel="Time t/T (excludes expiry)",title=title);ax.legend(fontsize=7)
    fig.savefig(output/"pathwise_errors.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    for ax,i in zip(axes,[0,1,2]):
        ax.plot(times,truth[i],color="black",label="Analytic")
        for label in labels:ax.plot(times,predictions[label][i],label=label.replace("state_pinn_lookback_",""))
        ax.set(title=f"Path {i} (preselected)",xlabel="Time t/T",ylabel="Price")
        ax.legend(fontsize=7)
    fig.savefig(output/"example_path_prices.png",dpi=160);plt.close(fig)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs",nargs="+",help="Saved lookback run directories")
    parser.add_argument("--output-dir",default="outputs/lookback_pathwise_audit")
    parser.add_argument("--paths",type=int,default=5000)
    parser.add_argument("--steps",type=int,default=128)
    parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--batch-size",type=int,default=4096,help="Inference batch, not training batch")
    parser.add_argument("--relative-floor",type=float,default=1e-4,help="Price floor in S0 units for pointwise relative statistics")
    args=parser.parse_args();torch.set_num_threads(1)
    audit(args.run_dirs,args.output_dir,args.paths,args.steps,args.seed,args.batch_size,args.relative_floor)
