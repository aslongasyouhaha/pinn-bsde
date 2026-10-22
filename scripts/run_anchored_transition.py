"""Soft+MC -> revised hard +/- MC -> H before/after unit-weight L-BFGS."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .state_pinn import anchored_state_models as api
from .state_pinn.evaluate_ten_contracts import evaluate,reference

OLD=Path('outputs/ten_contract_pinn_complete_15000')
SOURCES=('scripts/run_anchored_transition.py','scripts/state_pinn/anchored_state_models.py',
    'scripts/report_anchored_transition.py',
    'scripts/state_pinn/ten_contract_models.py','scripts/state_pinn/evaluate_ten_contracts.py',
    'scripts/state_pinn/bs_lookback_pinn.py','scripts/state_pinn/bs_barrier_heat_pinn.py',
    'scripts/barrier_heat_kernel.py','scripts/analytic_bs.py','scripts/state_pinn/evaluate_paths.py')


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def prepare_anchors(root,count):
    path=root/'mc_anchors.json'
    if path.exists():return json.loads(path.read_text())
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    generator=torch.Generator().manual_seed(20261210);totals={c:np.zeros(2) for c in api.CASES[2:]}
    started=time.perf_counter();dt=1/128
    for start in range(0,count,2048):
        n=min(2048,count-start)
        z=torch.randn(n,128,generator=generator)
        logs=torch.cat([torch.zeros(n,1),((.03-.5*.4**2)*dt+.4*math.sqrt(dt)*z).cumsum(1)],1)
        terminal=logs[:,-1].exp()
        for direction in ('up','down'):
            b=1.2 if direction=='up' else .8
            distance=math.log(b)-logs if direction=='up' else logs-math.log(b)
            alive=(distance[:,:-1]>0)&(distance[:,1:]>0)
            product=(distance[:,:-1]*distance[:,1:]).clamp_min(0.)
            interval=torch.where(alive,-torch.expm1(-2*product/(.4**2*dt)),torch.zeros_like(product))
            survival=interval.prod(1)
            for option in ('call','put'):
                payoff=torch.relu(terminal-1) if option=='call' else torch.relu(1-terminal)
                samples=math.exp(-.03)*survival*payoff
                totals[direction+'_'+option]+=np.array([samples.sum().item(),samples.square().sum().item()])
    anchors={}
    for case,(total,squared) in totals.items():
        mean=total/count;variance=max(0.,(squared-count*mean**2)/(count-1))
        anchors[case]=dict(value=mean,standard_error=math.sqrt(variance/count),paths=count,
            seed=20261210,steps=128,weight=100.,monitoring='Exact GBM endpoints with Brownian-bridge conditional survival; continuous zero-rebate barrier.',analytic_label=False)
    data=dict(anchors=anchors,elapsed_seconds=time.perf_counter()-started,
        shared_targets='The identical saved MC value is used by soft_mc and hard_mc; no reseeding or selection using analytic error.')
    write(path,data);return data


def save_model(root,case,variant,model,config,elapsed,extra=None):
    out=root/'training'/case/variant;out.mkdir(parents=True,exist_ok=True)
    mode='soft' if variant=='soft_mc' else 'heat' if variant.startswith('heat') else 'hard'
    one=torch.ones(1);zero=torch.zeros(1)
    with torch.no_grad():pred=api.value(model,case,mode,zero,one,one).item()
    exact=reference(case,zero,one,one,torch.zeros(1,dtype=torch.bool)).item()
    saved=dict(state_dict=model.state_dict(),args=config,market=asdict(api.lb.Market()),contract=asdict(api.contract(case)))
    if extra:saved['refinement']=extra
    torch.save(saved,out/'model.pt')
    write(out/'metrics.json',dict(prediction=pred,continuous_analytic=exact,relative_error=abs(pred-exact)/abs(exact),
        elapsed_seconds=elapsed,args=config,market=saved['market'],contract=saved['contract']))


def train_worker(args):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1)
    root=Path(args.output_dir);out=root/'training'/args.case/args.variant;out.mkdir(parents=True,exist_ok=True)
    mode='soft' if args.variant=='soft_mc' else 'heat' if args.variant=='heat_before' else 'hard'
    anchored=args.variant in ('soft_mc','hard_mc') and not args.case.startswith('lookback')
    anchors=json.loads((root/'mc_anchors.json').read_text())['anchors']
    target=anchors[args.case]['value'] if anchored else None;weight=100. if anchored else 0.
    model=api.make_model(args.case,mode);opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    schedule=torch.optim.lr_scheduler.StepLR(opt,1000,.9)
    config=dict(case=args.case,variant=args.variant,mode=mode,epochs=args.epochs,batch_size=128,
        hidden=64,depth=3,seed=1,precision='float64',pde_weight=1.,long_tau_weight=1.,
        mc_anchor_weight=weight,mc_anchor_value=target,mc_anchor_source=str(root/'mc_anchors.json') if anchored else None,
        structure_version='learnable_sqrt_corner_and_tanh_gate_v2',lr=5e-4,lr_step=1000,lr_gamma=.9,grad_clip=10.,status='running')
    write(out/'run_config.json',config);started=time.perf_counter()
    fields=['epoch','loss','pde','terminal','boundary','nonnegative','anchor','anchor_weight','gradient_norm','elapsed_seconds']
    with (out/'history.csv').open('w',newline='',encoding='utf-8',buffering=1) as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for epoch in range(1,args.epochs+1):
            opt.zero_grad(set_to_none=True);loss,parts=api.loss(model,args.case,mode,128,target,weight)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.))
            opt.step();schedule.step()
            row=dict(epoch=epoch,loss=loss.item(),**parts,gradient_norm=norm,elapsed_seconds=time.perf_counter()-started)
            writer.writerow(row)
            if epoch==1 or epoch%100==0:write(out/'progress.json',dict(status='training',**row,total_epochs=args.epochs))
            if epoch==1 or epoch%1000==0:
                torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),
                    rng_state=torch.get_rng_state(),epoch=epoch,args=config),out/'checkpoint.pt')
                print(args.case,args.variant,epoch,loss.item(),flush=True)
    config['status']='complete';save_model(root,args.case,args.variant,model,config,time.perf_counter()-started)
    write(out/'run_config.json',config);write(out/'progress.json',dict(status='complete',epoch=args.epochs,total_epochs=args.epochs))


def fixed_objective(model,case,t,s,y,chunk,backward):
    total=0.;pde_total=0.;negative_total=0.
    for start in range(0,len(t),chunk):
        tt=t[start:start+chunk];ss=s[start:start+chunk];yy=None if y is None else y[start:start+chunk]
        r=api.pde(model,case,'heat',tt,ss,yy);pde=r.square().sum()/len(t)
        negative=(torch.relu(-api.value(model,case,'heat',tt,ss,yy)).square().sum()/len(t)
                  if case.startswith('lookback') else pde*0.)
        loss=pde+.01*negative
        if backward:loss.backward()
        total+=loss.item();pde_total+=pde.item();negative_total+=negative.item()
    if not math.isfinite(total):raise FloatingPointError('Nonfinite LBFGS objective')
    return dict(loss=total,pde=pde_total,nonnegative=negative_total)


def refine_worker(args):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    root=Path(args.output_dir);source=root/'training'/args.case/'heat_before';out=root/'training'/args.case/'heat_after'
    out.mkdir(parents=True,exist_ok=True);saved=torch.load(source/'model.pt',map_location='cpu')
    assert saved['args'].get('pde_weight',1.)==1. and saved['args'].get('long_tau_weight',1.)==1.
    model=api.make_model(args.case,'heat');model.load_state_dict(saved['state_dict'])
    # Check full original unit-weight gradient before changing the optimizer.
    torch.manual_seed(20261220);rng=torch.get_rng_state()
    loss,_=api.loss(model,args.case,'heat',128);loss.backward()
    grad=torch.cat([p.grad.flatten() for p in model.parameters()]).detach().clone()
    model.zero_grad(set_to_none=True);torch.set_rng_state(rng);et,es,ey=api.sample(args.case,128)
    same=fixed_objective(model,args.case,et,es,ey,32,True)
    diff=(torch.cat([p.grad.flatten() for p in model.parameters()])-grad).abs().max().item()
    assert abs(same['loss']-loss.item())<1e-12 and diff<1e-9
    write(out/'objective_equivalence.json',dict(loss_difference=abs(same['loss']-loss.item()),max_gradient_difference=diff,pde_weight=1.))
    torch.manual_seed(20261221+api.CASES.index(args.case));t,s,y=api.sample(args.case,args.collocation)
    np.savez_compressed(out/'training_states.npz',t=t.numpy(),s=s.numpy(),y=np.array([]) if y is None else y.numpy())
    initial=fixed_objective(model,args.case,t,s,y,256,False);calls=0;history=[];started=time.perf_counter()
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=10,max_eval=15,history_size=50,
        tolerance_grad=1e-11,tolerance_change=1e-16,line_search_fn='strong_wolfe')
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True);p=fixed_objective(model,args.case,t,s,y,256,True);calls+=1
        return torch.tensor(p['loss'],dtype=torch.float64)
    for step in range(1,args.outer_steps+1):
        opt.step(closure);p=fixed_objective(model,args.case,t,s,y,256,False)
        history.append(dict(epoch=step,gradient_evaluations=calls,**p))
        with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
        write(out/'progress.json',dict(status='refining',step=step,total_steps=args.outer_steps,gradient_evaluations=calls,**p))
        torch.save(dict(state_dict=model.state_dict(),step=step,gradient_evaluations=calls),out/'checkpoint.pt')
        print(args.case,'LBFGS',step,calls,p['loss'],flush=True)
    elapsed=time.perf_counter()-started;config=dict(saved['args'],variant='heat_after',pde_weight=1.,long_tau_weight=1.,mc_anchor_weight=0.,
        lbfgs_outer_steps=args.outer_steps,collocation=args.collocation,gradient_evaluations=calls,
        source_sha256=hashlib.sha256((source/'model.pt').read_bytes()).hexdigest(),status='complete')
    save_model(root,args.case,'heat_after',model,config,json.loads((source/'metrics.json').read_text())['elapsed_seconds']+elapsed,
        dict(method='LBFGS',seconds=elapsed,gradient_evaluations=calls,initial=initial,final=history[-1],
             objective='PDE MSE coefficient 1 plus original lookback nonnegative penalty; no MC/analytic labels.'))
    write(out/'run_config.json',config);write(out/'progress.json',dict(status='complete',step=args.outer_steps,gradient_evaluations=calls))


def copy_run(source,dest):
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('model.pt','metrics.json','history.csv'):
        shutil.copy2(source/name,dest/name)
    write(dest/'reuse.json',dict(source=str(source),sha256=hashlib.sha256((source/'model.pt').read_bytes()).hexdigest()))


def run(args):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}
    if (root/'experiment.json').exists():
        if not args.resume:raise FileExistsError('Choose fresh output or --resume')
        config=json.loads((root/'experiment.json').read_text());assert config['sources']==hashes
    else:
        write(root/'structure_checks.json',api.checks());prepare_anchors(root,args.mc_paths)
        config=dict(status='training',epochs=args.epochs,mc_paths=args.mc_paths,paths=args.paths,
            collocation=args.collocation,outer_steps=args.outer_steps,pde_weight=1.,long_tau_weight=1.,
            sources=hashes,scope='Barrier soft+MC, revised hard +/- identical MC, H before/after unit-weight LBFGS. No MC for lookbacks or H. No analytic training labels.',
            limitation='One Adam training seed; model structure changes are explicit. L-BFGS is an extra-budget phase with fixed training collocation and independent evaluation.')
        write(root/'experiment.json',config)
        for name in SOURCES:
            dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    def done(case,variant):return (root/'training'/case/variant/'model.pt').exists()
    def launch(item,refine=False):
        case,variant=item
        if done(case,variant):return
        out=root/'training'/case/variant;out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u','-m','scripts.run_anchored_transition','--worker','refine' if refine else 'train',
            '--case',case,'--variant',variant,'--output-dir',str(root),'--epochs',str(args.epochs),
            '--collocation',str(args.collocation),'--outer-steps',str(args.outer_steps)]
        with (out/'train.log').open('w',encoding='utf-8') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        jobs=[]
        for case in api.CASES:
            for variant in ('soft_mc','heat_before'):
                native='soft' if variant=='soft_mc' else 'heat'
                # Reuse historical artifacts only when explicitly requested.
                # A clean checkout trains every required model from scratch.
                can_reuse=args.reuse_existing and args.epochs==15000 and (native=='heat' or case.startswith('lookback'))
                if can_reuse and not done(case,variant):
                    saved=torch.load(OLD/case/native/'model.pt',map_location='cpu')
                    assert saved['args']['epochs']==15000 and saved['args'].get('pde_weight',1.)==1.
                    assert saved['args'].get('mc_anchor_weight',0.)==0.
                    copy_run(OLD/case/native,root/'training'/case/variant)
                elif not done(case,variant):jobs.append((case,variant))
            jobs.append((case,'hard_no_mc'))
            if not case.startswith('lookback'):jobs.append((case,'hard_mc'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(launch,jobs))
        config['status']='refining';write(root/'experiment.json',config)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda c:launch((c,'heat_after'),True),api.CASES))
        assert hashes=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}
        config['status']='evaluating';write(root/'experiment.json',config)
        stages={'hard_without_mc':dict(soft='soft_mc',hard='hard_no_mc',heat='heat_before'),
                'hard_with_mc':dict(soft='soft_mc',hard='hard_mc',heat='heat_before'),
                'heat_refined':dict(soft='soft_mc',hard='hard_mc',heat='heat_after')}
        for stage,mapping in stages.items():
            folder=root/'stages'/stage;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'evaluation/evaluation_manifest.json').exists():continue
            for case in api.CASES:
                for mode,variant in mapping.items():
                    if case.startswith('lookback') and variant=='hard_mc':variant='hard_no_mc'
                    copy_run(root/'training'/case/variant,folder/case/mode)
            evaluate(folder,args.paths,model_api=api)
        from .report_anchored_transition import report
        report(root)
        config['status']='complete';write(root/'experiment.json',config)
    except BaseException:
        config['status']='failed';config['error']=traceback.format_exc();write(root/'experiment.json',config);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/anchored_transition_15000')
    p.add_argument('--epochs',type=int,default=15000);p.add_argument('--mc-paths',type=int,default=200000)
    p.add_argument('--paths',type=int,default=2048);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--collocation',type=int,default=8192);p.add_argument('--outer-steps',type=int,default=20)
    p.add_argument('--worker',choices=['train','refine']);p.add_argument('--case',choices=api.CASES)
    p.add_argument('--reuse-existing',action='store_true',help='Reuse historical seed-1 checkpoints from OLD instead of training those variants.')
    p.add_argument('--variant');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if min(args.epochs,args.mc_paths,args.paths,args.workers,args.collocation,args.outer_steps)<1:p.error('Positive counts required')
    if args.worker=='train':train_worker(args)
    elif args.worker=='refine':refine_worker(args)
    else:run(args)
