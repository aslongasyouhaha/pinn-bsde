"""Single-asset discrete barrier: author independent Y/Z LSTMs and native loss.

Only the project exact-GBM sampler and up-and-out payoff are adapters. The
verified author library, network initializers and fbsdeint remain unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import torch

from .reproduce_author_lookback import load_author, verify_source, write_json


class DiscreteUpOutCall:
    def __init__(self, strike, barrier):
        self.strike, self.barrier = strike, barrier

    def payoff(self, paths):
        prices = paths[:, :, 0]
        return ((prices.max(1).values < self.barrier)*
            torch.relu(prices[:, -1]-self.strike)).unsqueeze(1)


def make_model(author_cls, args):
    class ExactGBM(author_cls):
        def sdeint(self, ts, x0):
            h = ts[1:]-ts[:-1]
            dw = torch.randn(len(x0), len(h), 1, device=x0.device, dtype=x0.dtype)*h.sqrt()[None, :, None]
            log_return = (self.mu-.5*self.sigma**2)*h[None, :, None]+self.sigma*dw
            future = x0[:, None]*torch.exp(log_return.cumsum(1))
            return torch.cat([x0[:, None], future], 1), dw
    return ExactGBM(1, args.r, args.sigma, args.depth, args.hidden,
        [args.hidden, args.hidden]).to(args.device)


def saved_rng():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:torch.cuda.set_rng_state_all(state[1])


def preflight(model, option, ts, args):
    state = saved_rng()
    try:
        torch.manual_seed(20261620)
        x0 = torch.full((4, 1), args.s0, device=args.device)
        before = saved_rng()
        native, _, _ = model.fbsdeint(ts, x0, option, args.lag)
        grads = torch.autograd.grad(native, tuple(model.parameters()))
        restore_rng(before)
        paths, f, dw = model.prepare_data(ts, x0, args.lag)
        y, z, g = model.f(f), model.dfdx(f), option.payoff(paths)
        coarse = ts[::args.lag]
        residual = y[:, :-1]+(z*dw).sum(-1, keepdim=True)[:, :-1]-torch.exp(
            -args.r*(coarse[1:]-coarse[:-1]))[None, :, None]*y[:, 1:].detach()
        replay = residual.square().mean(0).sum()+(y[:, -1]-g).square().mean()
        other = torch.autograd.grad(replay, tuple(model.parameters()))
        error = max(float((a-b).abs().max()) for a,b in zip(grads,other))
        assert abs(float(native)-float(replay)) < 1e-6 and error < 1e-5
        assert all(torch.isfinite(v).all() for v in grads)
        assert sum(isinstance(m, torch.nn.LSTM) for m in model.modules()) == 2
        assert {p.data_ptr() for p in model.f.parameters()}.isdisjoint(
            {p.data_ptr() for p in model.dfdx.parameters()})
        assert torch.equal(g.squeeze(1), torch.relu(paths[:, -1, 0]-args.strike)*
            (paths[:, :, 0].max(1).values<args.barrier))
        assert (paths>0).all() and y.shape==(4,args.steps//args.lag+1,1)
        model.train(False)
        causal = 0.
        with torch.no_grad():
            full_y, full_z = model.f(f), model.dfdx(f)
            for j in [0, args.steps//args.lag//2, args.steps//args.lag]:
                k=j*args.lag;prefix=model.get_stream_signatures(ts[:k+1],paths[:,:k+1],args.lag)
                causal=max(causal,float((model.f(prefix)[:,-1]-full_y[:,j]).abs().max()),
                    float((model.dfdx(prefix)[:,-1]-full_z[:,j]).abs().max()))
        assert causal < 1e-5
        return dict(dimension=1, independent_Y_Z_LSTMs=True, native_author_loss_replay_error=abs(float(native)-float(replay)),
            native_author_gradient_replay_max_error=error, finite_gradients=True,
            prefix_causality_max_error=causal, native_terminal_retained=True,
            output_shape=list(y.shape), exact_GBM_sampler=True, discrete_payoff_verified=True)
    finally:
        model.train(True);restore_rng(state)


@torch.no_grad()
def mean_audit(model, option, ts, args, seed):
    torch.manual_seed(seed)
    coarse=ts[::args.lag];d=torch.exp(-args.r*coarse)
    rows=[];defects=[];terminals=[]
    for start in range(0,args.mc_paths,512):
        n=min(512,args.mc_paths-start)
        x,f,dw=model.prepare_data(ts,torch.full((n,1),args.s0,device=args.device),args.lag)
        y,z=model.f(f).squeeze(-1),model.dfdx(f)
        g=(math.exp(-args.r*args.T)*option.payoff(x).squeeze(-1)).double()
        integral=(d[None,:-1,None]*z[:,:-1]*dw[:,:-1]).sum((1,2)).double()
        discounted=(y*d).double()
        rows.append(torch.stack([g,integral,discounted[:,0]],1).cpu())
        defects.append((discounted-g[:,None]).cpu())
        terminals.append((y[:,-1]-option.payoff(x).squeeze(-1)).double().cpu())
    rows=torch.cat(rows);defects=torch.cat(defects);terminals=torch.cat(terminals)
    g,i,y0=rows.T;cv=g-i;n=len(g)
    se=lambda v:float(v.std(unbiased=True)/math.sqrt(n))
    vg,vi=g.var(unbiased=True),i.var(unbiased=True)
    covariance=((g-g.mean())*(i-i.mean())).sum()/(n-1)
    return dict(paths=n,seed=seed,initial_price=float(y0.mean()),grid_mc=float(g.mean()),grid_mc_se=se(g),
        cv_mc=float(cv.mean()),cv_se=se(cv),integral_mean=float(i.mean()),integral_mean_se=se(i),
        payoff_integral_correlation=float(covariance/(vg*vi).sqrt()) if vg>0 and vi>0 else None,
        variance_reduction=float(vg/cv.var(unbiased=True)),times=coarse.cpu().tolist(),
        discounted_y_mean=(defects.mean(0)+g.mean()).tolist(),mean_defect=defects.mean(0).tolist(),
        mean_defect_se=(defects.std(0,unbiased=True)/math.sqrt(n)).tolist(),
        terminal_rmse=float(terminals.square().mean().sqrt()),terminal_mae=float(terminals.abs().mean()),
        terminal_is_learned=True)


@torch.no_grad()
def conditional_reference(prefix, k, args, seed):
    if k==args.steps:
        p=prefix[0,:,0]
        return float((p.max()<args.barrier)*torch.relu(p[-1]-args.strike)),0.
    if float(prefix.max())>=args.barrier:return 0.,0.
    torch.manual_seed(seed)
    remaining=args.steps-k;dt=args.T/args.steps;spot=prefix[0,-1,0].double()
    total=0.;square=0.
    for start in range(0,args.nested_paths,2048):
        n=min(2048,args.nested_paths-start)
        dw=torch.randn(n,remaining,device=args.device,dtype=torch.float64)*math.sqrt(dt)
        future=spot*torch.exp(((args.r-.5*args.sigma**2)*dt+args.sigma*dw).cumsum(1))
        g=math.exp(-args.r*remaining*dt)*torch.relu(future[:,-1]-args.strike)*(future.max(1).values<args.barrier)
        total+=float(g.sum());square+=float(g.square().sum())
    mean=total/args.nested_paths
    return mean,math.sqrt(max(0.,(square-args.nested_paths*mean**2)/(args.nested_paths-1)/args.nested_paths))


def evaluate(model,option,ts,args,root):
    model.train(False)
    for label,seed in [('primary',20261621),('confirmation',20261622)]:
        write_json(root/'progress.json',dict(status='evaluating',phase=label))
        write_json(root/(label+'_mean_audit.json'),mean_audit(model,option,ts,args,seed))
    with torch.no_grad():
        torch.manual_seed(20261623)
        paths,f,_=model.prepare_data(ts,torch.full((args.eval_paths,1),args.s0,device=args.device),args.lag)
        y=model.f(f).squeeze(-1).cpu()
    query=sorted(set([0,args.steps//4,args.steps//2,3*args.steps//4,args.steps-args.lag,args.steps]))
    assert all(k%args.lag==0 for k in query)
    rows=[]
    for k in query:
        for i in range(args.eval_paths):
            prefix=paths[i:i+1,:k+1]
            ref,se=conditional_reference(prefix,k,args,20304000+i*len(query)+query.index(k))
            rows.append(dict(path=i,fine_index=k,time=args.T*k/args.steps,
                dead=bool(float(prefix.max())>=args.barrier),prediction=float(y[i,k//args.lag]),
                reference=ref,reference_se=se,terminal_is_learned=k==args.steps))
        with (root/'conditional_prices.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write_json(root/'progress.json',dict(status='evaluating',phase='conditional_prices',completed_dates=query.index(k)+1,total_dates=len(query)))
    summaries=[]
    for k in query:
        rr=[r for r in rows if r['fine_index']==k]
        denominator=sum(r['reference']**2 for r in rr)
        dead=[r for r in rr if r['dead']];alive=[r for r in rr if not r['dead']]
        summaries.append(dict(fine_index=k,time=args.T*k/args.steps,
            relative_l2=math.sqrt(sum((r['prediction']-r['reference'])**2 for r in rr)/denominator) if denominator else None,
            reference_noise_relative_l2=math.sqrt(sum(r['reference_se']**2 for r in rr)/denominator) if denominator else None,
            dead_price_rmse=math.sqrt(sum(r['prediction']**2 for r in dead)/len(dead)) if dead else None,
            alive_relative_l2=math.sqrt(sum((r['prediction']-r['reference'])**2 for r in alive)/denominator) if denominator else None,
            terminal_is_learned=k==args.steps))
    write_json(root/'conditional_summary.json',dict(paths=args.eval_paths,nested_paths=args.nested_paths,
        by_time=summaries,no_hit_mask=True,terminal_overwritten=False))
    torch.save(dict(paths=paths.cpu(),predictions=y,history_seed=20261623),root/'evaluation_paths.pt')
    primary=json.loads((root/'primary_mean_audit.json').read_text())
    lines=['# Single-asset author-structure barrier BSDE','',
        'One asset, discrete up-and-out call. Original author independent price/Z LSTMs and '
        'unchanged stopped-target fbsdeint are used. Only exact-GBM simulation and payoff are '
        'project adapters. No supplied hit state/gate, hard terminal substitution, terminal pretraining, '
        'MC anchor or analytic/conditional-price training label. Old experiments remain unchanged.','',
        f"Initial network price: {primary['initial_price']:.9f}; matched-grid MC: {primary['grid_mc']:.9f} "
        f"(SE {primary['grid_mc_se']:.9f}); correlation: {primary['payoff_integral_correlation']}; "
        f"native terminal RMSE: {primary['terminal_rmse']:.9f}.",'',
        '| Fine index | Price relative L2 | Reference noise L2 | Dead-price RMSE |',
        '|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['fine_index']} | {s['relative_l2']} | {s['reference_noise_relative_l2']} | {s['dead_price_rmse']} |")
    lines+=['','These are one-training-seed results; relative point gaps are not accuracy bounds. '
        'Two independent unconditional audits and per-prefix reference noise are retained. '
        'The terminal predictions are native network outputs and are evaluated rather than replaced. '
        'Architecture, objective, optimizer and width differ from the earlier shared-encoder '
        'multi-step version; this is a clean author-structure baseline, not an isolated architecture ablation.']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def run(args):
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise FileExistsError('Use a fresh output directory')
    source=root/'source';verify_source(Path(args.source_dir));shutil.copytree(args.source_dir,source)
    shutil.copy2(__file__,source/Path(__file__).name)
    cls,_=load_author(source);torch.set_num_threads(1);torch.manual_seed(args.seed)
    model=make_model(cls,args);option=DiscreteUpOutCall(args.strike,args.barrier)
    ts=torch.linspace(0,args.T,args.steps+1,device=args.device)
    checks=preflight(model,option,ts,args);write_json(root/'validation.json',checks)
    policy=dict(vars(args),dimension=1,independent_Y_Z_LSTMs=True,native_author_fbsdeint=True,
        source_commit=json.loads((source/'source_manifest.json').read_text())['commit'],
        sampler='exact GBM; 128-step project discrete monitoring by default',
        training_initial_spot='fixed project s0',no_hit_mask=True,terminal='native network output plus payoff MSE',
        terminal_weight=1,bsde_date_reduction='sum',warmup_steps=0,mc_anchor=False,
        optimizer='RMSprop',lr=.0005,lr_step=5000,lr_gamma=.2,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='training')
    write_json(root/'config.json',policy)
    (root/'report.md').write_text('# Single-asset author-structure BSDE: running\n\nSee config.json, validation.json and progress.json. Final pricing/consistency results are not available yet.\n',encoding='utf-8')
    optimizer=torch.optim.RMSprop(model.parameters(),lr=.0005)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,5000,.2)
    started=time.perf_counter()
    with (root/'history.csv').open('w',newline='',buffering=1) as f:
        w=csv.DictWriter(f,fieldnames=['epoch','loss','bsde','terminal','initial_batch_mean','payoff_mean','elapsed_seconds']);w.writeheader()
        for epoch in range(1,args.epochs+1):
            optimizer.zero_grad()
            loss,y,g=model.fbsdeint(ts,torch.full((args.batch_size,1),args.s0,device=args.device),option,args.lag)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native loss')
            loss.backward();optimizer.step();scheduler.step()
            terminal=float((y[:,-1]-g).square().mean())
            row=dict(epoch=epoch,loss=float(loss),bsde=float(loss)-terminal,terminal=terminal,
                initial_batch_mean=float(y[:,0].mean()),payoff_mean=float(g.mean()),elapsed_seconds=time.perf_counter()-started)
            w.writerow(row)
            if epoch==1 or epoch%100==0 or epoch==args.epochs:
                write_json(root/'progress.json',dict(status='training',total_epochs=args.epochs,**row));print(row,flush=True)
            if epoch%1000==0 or epoch==args.epochs:
                torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    rng=saved_rng(),epoch=epoch,config=policy),root/'checkpoint.pt')
    torch.save(dict(state_dict=model.state_dict(),config=policy,completed_epochs=args.epochs),root/'model.pt')
    write_json(root/'progress.json',dict(status='evaluating',epoch=args.epochs))
    evaluate(model,option,ts,args,root)
    verify_source(source);policy.update(status='complete',training_seconds=row['elapsed_seconds'])
    write_json(root/'config.json',policy)
    write_json(root/'progress.json',dict(status='complete',epoch=args.epochs,total_elapsed_seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/single_asset_author_bsde_barrier_15000')
    p.add_argument('--source-dir',default='vendor/deep_ppde')
    p.add_argument('--device',default='cuda')
    for name,value in [('s0',1.),('strike',1.),('barrier',1.2),('r',.03),('sigma',.4),('T',1.)]:p.add_argument('--'+name,type=float,default=value)
    for name,value in [('steps',128),('lag',8),('depth',3),('hidden',20),('epochs',15000),('batch-size',128),('seed',1),('mc-paths',131072),('eval-paths',128),('nested-paths',65536)]:p.add_argument('--'+name,type=int,default=value)
    a=p.parse_args()
    if min(a.steps,a.lag,a.depth,a.hidden,a.epochs,a.batch_size,a.mc_paths,a.eval_paths,a.nested_paths)<2 or a.steps%a.lag or any(k%a.lag for k in [a.steps//4,a.steps//2,3*a.steps//4]) or a.s0<=0 or a.s0>=a.barrier or a.sigma<=0 or a.T<=0:p.error('Invalid grid/positive counts or market')
    try:run(a)
    except Exception:
        root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True);write_json(root/'failure.json',dict(error=traceback.format_exc()));raise
