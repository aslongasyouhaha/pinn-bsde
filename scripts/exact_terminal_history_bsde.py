"""Discrete single-asset barrier BSDE: exact payoff boundary, backward dates.

The original learned-terminal backward experiment is preserved. Fine-grid
payoff is used in TRAINING the final interval, never as an interior input.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch
from .history_models import DateValues, DateZ, freeze, tensor_hash
from .pure_history_h import HistoryValue, BrownianCoefficient, draw, payoff
from .single_asset_signature_bsde import make_model, conditional_reference
from .reproduce_author_lookback import verify_source, load_author, write_json


def next_value(target,x,f,ct,s,a,i):
    if target is None:
        assert i==a.steps//a.lag-1
        return payoff(x,a)
    return target(f[:,:i+2],ct[:i+2],s[:,:i+2],a)[:,-1]


@torch.no_grad()
def outputs(values,zs,sampler,ts,a,x,features=None):
    f=sampler.get_stream_signatures(ts,x,a.lag) if features is None else features
    ct=ts[::a.lag];s=x[:,::a.lag,0]
    interior=values(f[:,:-1],ct[:-1],s[:,:-1],a)
    # This boundary is also used in next_value during the final training stage.
    y=torch.cat([interior,payoff(x,a)[:,None]],1)
    return y,zs(f,ct,s,a)


@torch.no_grad()
def stage_audit(y,z,target,sampler,ts,a,i):
    torch.manual_seed(91000+i);ct=ts[::a.lag];disc=math.exp(-a.r*float(ct[i+1]-ct[i]));parts=[]
    for start in range(0,a.validation_paths,512):
        x,f,dw=draw(sampler,ts,a,min(512,a.validation_paths-start));s=x[:,::a.lag,0]
        now=y(f[:,:i+1],ct[:i+1],s[:,:i+1],a)[:,-1]
        zz=z(f[:,:i+1],ct[:i+1],s[:,:i+1],a)[:,-1]
        nxt=disc*next_value(target,x,f,ct,s,a,i)
        residual=now+zz*dw[:,i,0]-nxt
        parts.append(torch.stack([now,nxt,residual],1).double().cpu())
    v=torch.cat(parts);n=len(v)
    return dict(index=i,time=float(ct[i]),paths=n,prediction_mean=float(v[:,0].mean()),
        discounted_next_mean=float(v[:,1].mean()),bellman_mean_residual=float(v[:,2].mean()),
        bellman_mean_residual_se=float(v[:,2].std()/n**.5),residual_rmse=float(v[:,2].square().mean().sqrt()),
        target_is_exact_payoff=target is None,next_date_target_unchanged=True)


@torch.no_grad()
def validate(values,zs,sampler,ts,a):
    torch.manual_seed(996);x,f,_=draw(sampler,ts,a,512)
    y,z=outputs(values,zs,sampler,ts,a,x,f)
    assert torch.equal(y[:,-1],payoff(x,a))
    assert int((y[:,-1]>0).sum())>0
    k=a.steps//2;changed=x.clone();changed[:,k+1:]*=1.7
    cy,cz=outputs(values,zs,sampler,ts,a,changed)
    error=max(float((cy[:,:k//a.lag+1]-y[:,:k//a.lag+1]).abs().max()),
              float((cz[:,:k//a.lag+1]-z[:,:k//a.lag+1]).abs().max()))
    assert error<1e-6
    ids=[{p.data_ptr() for p in m.parameters()} for m in list(values.models)+list(zs.models)]
    assert all(a1.isdisjoint(a2) for j,a1 in enumerate(ids) for a2 in ids[j+1:])
    return dict(prefix_future_perturbation_error=error,terminal_exact=True,
        terminal_positive_examples=int((y[:,-1]>0).sum()),independent_date_and_Y_Z_parameters=True,
        terminal_uses_full_fine_grid=True,no_explicit_history_state_inputs=True,
        terminal_zero_error_is_structural_not_evidence_of_interior_accuracy=True)


@torch.no_grad()
def evaluate(values,zs,sampler,ts,a,root):
    torch.manual_seed(992);ct=ts[::a.lag];discount=torch.exp(-a.r*ct)
    ys=[];gs=[];integrals=[]
    for start in range(0,a.mc_paths,512):
        x,f,dw=draw(sampler,ts,a,min(512,a.mc_paths-start))
        y,z=outputs(values,zs,sampler,ts,a,x,f)
        ys.append((y*discount).double().cpu());gs.append((discount[-1]*payoff(x,a)).double().cpu())
        integrals.append((z[:,:-1]*dw[:,:-1,0]*discount[:-1]).sum(1).double().cpu())
    y=torch.cat(ys);g=torch.cat(gs);integral=torch.cat(integrals);n=len(g);defect=y-g[:,None]
    cv=g-integral;cov=((g-g.mean())*(integral-integral.mean())).mean()
    audit=dict(paths=n,initial_price=float(y[:,0].mean()),grid_mc=float(g.mean()),grid_mc_se=float(g.std()/n**.5),
        discounted_y_mean=y.mean(0).tolist(),mean_defect=defect.mean(0).tolist(),mean_defect_se=(defect.std(0)/n**.5).tolist(),
        terminal_rmse=0.,terminal_is_learned=False,terminal_exact_by_construction=True,
        initial_relative_gap=float((y[:,0].mean()-g.mean()).abs()/g.mean()),
        payoff_integral_correlation=float(cov/(g.std(unbiased=False)*integral.std(unbiased=False)).clamp_min(1e-30)),
        variance_reduction=float(g.var()/cv.var()),integral_mean=float(integral.mean()),integral_mean_se=float(integral.std()/n**.5))
    write_json(root/'mean_audit.json',audit)
    torch.manual_seed(993);x,f,_=draw(sampler,ts,a,a.eval_paths);prediction,_=outputs(values,zs,sampler,ts,a,x,f)
    rows=[];summaries=[]
    for k in [0,a.steps//4,a.steps//2,3*a.steps//4,a.steps-a.lag,a.steps]:
        for j in range(a.eval_paths):
            ref,se=conditional_reference(x[j:j+1,:k+1],k,a,80000+k*a.eval_paths+j)
            rows.append(dict(path=j,index=k,time=a.T*k/a.steps,dead=bool(x[j,:k+1,0].max()>=a.barrier),
                prediction=float(prediction[j,k//a.lag]),reference=ref,reference_se=se,terminal_exact=k==a.steps))
        rr=[r for r in rows if r['index']==k];den=sum(r['reference']**2 for r in rr);dead=[r for r in rr if r['dead']]
        summaries.append(dict(time=a.T*k/a.steps,relative_l2=math.sqrt(sum((r['prediction']-r['reference'])**2 for r in rr)/den) if den else None,
            reference_noise_l2=math.sqrt(sum(r['reference_se']**2 for r in rr)/den) if den else None,
            dead_count=len(dead),dead_price_rmse=math.sqrt(sum(r['prediction']**2 for r in dead)/len(dead)) if dead else None,
            terminal_exact_by_construction=k==a.steps))
    with (root/'conditional_prices.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/'conditional_summary.json',dict(paths=a.eval_paths,nested_paths=a.nested_paths,by_time=summaries))
    torch.save(dict(paths=x.cpu(),predictions=prediction.cpu()),root/'evaluation_paths.pt')
    return audit


def run(cli):
    root=Path(cli.output_dir);root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise FileExistsError('Use a fresh directory')
    torch.set_num_threads(1);torch.manual_seed(cli.seed)
    checkpoint=torch.load(cli.initial_value,map_location=cli.device)
    a=argparse.Namespace(**dict(checkpoint['args'],**{}))
    for k,v in vars(cli).items():setattr(a,k,v)
    a.method='bsde'
    assert checkpoint['args']['method']=='bsde' and checkpoint.get('terminal_only')
    verify_source(Path(a.source_dir));cls,_=load_author(Path(a.source_dir));sampler=make_model(cls,a)
    ts=torch.linspace(0,a.T,a.steps+1,device=a.device);ct=ts[::a.lag];dates=len(ct)-1
    budget=dates*a.stage_epochs if cli.total_training_updates is None else cli.total_training_updates-checkpoint['epochs']
    if budget<dates or not 0<cli.lr_final_factor<=1:raise ValueError('Invalid total budget or learning-rate factor')
    stage_counts=[budget//dates+int(j<budget%dates) for j in range(dates)]
    # Matched initialization only. This network is NEVER a terminal target.
    initializer=HistoryValue(sampler.sig_channels,a).to(a.device);initializer.load_state_dict(checkpoint['value'])
    values=[None]*dates;zs=[None]*dates
    write_json(root/'config.json',dict(vars(a),terminal='exact fine-grid path payoff in final-interval TRAINING',
        learned_terminal_network=False,initial_checkpoint_role='initialization only, matched to learned-terminal backward control',
        initial_checkpoint_sha256=hashlib.sha256(Path(cli.initial_value).read_bytes()).hexdigest(),
        initial_pretraining_updates=checkpoint['epochs'],backward_updates=budget,
        total_training_updates=checkpoint['epochs']+budget,updates_per_date_in_backward_order=stage_counts,
        lr_schedule='per-date cosine decay' if cli.lr_final_factor!=1 else 'constant',lr_final_factor=cli.lr_final_factor,
        no_explicit_history_state_inputs=True,mc_anchor=False,loss_weight=1,
        optimizer='fresh RMSprop per date',Y_output='same H*sigmoid price head as backward control'))
    shutil.copy2(__file__,root/'implementation.py')
    start=time.perf_counter();stages=[]
    with (root/'history.csv').open('w',newline='',buffering=1) as f:
        writer=csv.DictWriter(f,fieldnames=['index','time','update','lr','loss','prediction_mean','target_mean','seconds']);writer.writeheader()
        for i in range(dates-1,-1,-1):
            target=None if i==dates-1 else values[i+1]
            target_hash=None if target is None else tensor_hash(target)
            y=copy.deepcopy(initializer if target is None else target)
            z=BrownianCoefficient(sampler.sig_channels,a).to(a.device) if i==dates-1 else copy.deepcopy(zs[i+1])
            for model in [y,z]:
                model.train(True)
                for p in model.parameters():p.requires_grad_(True)
            params=list(y.parameters())+list(z.parameters());opt=torch.optim.RMSprop(params,lr=a.lr)
            if target is not None:assert {id(p) for p in params}.isdisjoint({id(p) for p in target.parameters()})
            torch.manual_seed(a.seed+1000+i);disc=torch.exp(-a.r*(ct[i+1]-ct[i]))
            stage_updates=stage_counts[dates-1-i]
            for update in range(1,stage_updates+1):
                lr=a.lr*(cli.lr_final_factor+(1-cli.lr_final_factor)*.5*(1+math.cos(math.pi*(update-1)/max(stage_updates-1,1))))
                for group in opt.param_groups:group['lr']=lr
                x,features,dw=draw(sampler,ts,a,a.batch_size);s=x[:,::a.lag,0]
                with torch.no_grad():nxt=disc*next_value(target,x,features,ct,s,a,i)
                now=y(features[:,:i+1],ct[:i+1],s[:,:i+1],a)[:,-1]
                zz=z(features[:,:i+1],ct[:i+1],s[:,:i+1],a)[:,-1]
                loss=(now+zz*dw[:,i,0]-nxt).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                opt.zero_grad();loss.backward()
                if update==1:
                    assert not nxt.requires_grad
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    if target is not None:assert all(p.grad is None for p in target.parameters())
                    else:assert torch.equal(nxt,disc*payoff(x,a))
                opt.step()
                row=dict(index=i,time=float(ct[i]),update=update,lr=lr,loss=float(loss),prediction_mean=float(now.mean()),target_mean=float(nxt.mean()),seconds=time.perf_counter()-start)
                writer.writerow(row)
                if update==1 or update%250==0 or update==stage_updates:
                    write_json(root/'progress.json',dict(status='training',completed_dates=dates-1-i,dates=dates,**row));print(row,flush=True)
            if target is not None:assert tensor_hash(target)==target_hash
            freeze(y);freeze(z);values[i]=y;zs[i]=z
            audit=stage_audit(y,z,target,sampler,ts,a,i);audit['target_hash']=target_hash
            stages.append(audit);write_json(root/'stage_audits.json',stages)
            torch.save(dict(value=y.state_dict(),z=z.state_dict(),index=i,args=vars(a),optimizer=opt.state_dict(),
                completed_date_updates=stage_updates,cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),root/('date_'+str(i)+'.pt'))
    value_net,z_net=DateValues(values),DateZ(zs)
    torch.save(dict(values=value_net.state_dict(),zs=z_net.state_dict(),args=vars(a),terminal_exact=True),root/'model.pt')
    write_json(root/'validation.json',validate(value_net,z_net,sampler,ts,a))
    write_json(root/'progress.json',dict(status='evaluating'))
    audit=evaluate(value_net,z_net,sampler,ts,a,root)
    write_json(root/'progress.json',dict(status='complete',seconds=time.perf_counter()-start,initial_price=audit['initial_price']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--initial-value',default='outputs/pure_history_h_followup/bsde_diagnosis/terminal_control.pt')
    p.add_argument('--device',default='cuda');p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--total-training-updates',type=int,help='Total case budget including initialization pretraining, distributed across all dates.')
    p.add_argument('--lr-final-factor',type=float,default=1.,help='Per-date cosine final rate / initial rate.')
    for key,value in [('stage-epochs',750),('batch-size',128),('seed',1),('validation-paths',4096),('mc-paths',16384),('eval-paths',64),('nested-paths',8192)]:p.add_argument('--'+key,type=int,default=value)
    cli=p.parse_args()
    try:run(cli)
    except Exception:
        import traceback
        root=Path(cli.output_dir);root.mkdir(parents=True,exist_ok=True)
        write_json(root/'failure.json',dict(error=traceback.format_exc()));raise
