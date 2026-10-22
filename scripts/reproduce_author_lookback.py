"""Run the unmodified Deep-PPDE README lookback example at a pinned commit.

The pinned library is bundled under vendor with its license; this wrapper adds recording and
independent evaluation, not changes to the author's optimizer or loss.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import torch

COMMIT='b704b32d3a5b064586893bdb15a00f1aa12d2fc9'
FILES=('lib/__init__.py','lib/augmentations.py','lib/bsde.py','lib/networks.py','lib/options.py','README.md','LICENSE')
DEFAULT_OUTPUT='outputs/author_lookback_readme_10000'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def prepare(root):
    source=root/'source'
    if (source/'source_manifest.json').exists():return source
    source.mkdir(parents=True,exist_ok=True);manifest={}
    for filename in FILES:
        url=f'https://raw.githubusercontent.com/msabvid/Deep-PPDE/{COMMIT}/{filename}'
        data=urllib.request.urlopen(url,timeout=30).read()
        dest=source/filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        manifest[filename]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
    write_json(source/'source_manifest.json',dict(repository='https://github.com/msabvid/Deep-PPDE',commit=COMMIT,files=manifest,modified=False))
    return source


def verify_source(source):
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if manifest['commit']!=COMMIT:raise ValueError('Unexpected author version')
    for filename,entry in manifest['files'].items():
        if hashlib.sha256((source/filename).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Author source changed: '+filename)


def load_author(source):
    sys.path.insert(0,str(source.resolve()))
    from lib.bsde import PPDE_BlackScholes
    from lib.options import Lookback
    return PPDE_BlackScholes,Lookback


def sample_x0(batch,device='cpu'):
    # Exactly the distribution written in the author's README.
    z=torch.randn(batch,1,device=device)
    return torch.exp((.08-.5*.3**2)*.1+.3*math.sqrt(.1)*z)


def make_model(cls):return cls(1,.05,.3,3,20,[20,20])


@torch.no_grad()
def evaluate(model,option,ts,root,count,nested_count):
    # PPDE.eval is an author-defined pricing method, so use train(False) to
    # change module mode without invoking that unrelated convenience API.
    model.train(False);torch.manual_seed(20261103)
    sums={k:0. for k in ['payoff','payoff2','cv','cv2','integral','integral2','cross']}
    initial=None;coarse=ts[::20];discount=torch.exp(-.05*coarse[:-1]).reshape(1,-1,1)
    mean_sums=torch.zeros(26,device=ts.device,dtype=torch.float64)
    defect_sums=torch.zeros_like(mean_sums);defect_squares=torch.zeros_like(mean_sums)
    for start in range(0,count,512):
        size=min(512,count-start)
        x,features,dw=model.prepare_data(ts,torch.ones(size,1,device=ts.device),20)
        y=model.f(features);z=model.dfdx(features)
        if initial is None:initial=float(y[0,0,0])
        payoff=math.exp(-.05)*option.payoff(x).squeeze(-1).double()
        # Native author Z is trained directly against Brownian increments.
        # Do not multiply it by sigma*S a second time.
        integral=(discount*z[:,:-1,:]*dw[:,:-1,:]).sum((1,2)).double()
        cv=payoff-integral
        discounted_y=(y.squeeze(-1)*torch.exp(-.05*coarse)).double()
        defects=discounted_y-payoff[:,None]
        mean_sums+=discounted_y.sum(0);defect_sums+=defects.sum(0);defect_squares+=defects.square().sum(0)
        for key,value in [('payoff',payoff),('payoff2',payoff.square()),('cv',cv),('cv2',cv.square()),('integral',integral),('integral2',integral.square()),('cross',payoff*integral)]:sums[key]+=value.sum().item()
    mc=sums['payoff']/count;cv=sums['cv']/count;integral=sums['integral']/count
    var_mc=(sums['payoff2']-count*mc*mc)/(count-1)
    var_cv=(sums['cv2']-count*cv*cv)/(count-1)
    var_int=(sums['integral2']-count*integral*integral)/(count-1)
    covariance=(sums['cross']-count*mc*integral)/(count-1)
    from scripts.analytic_bs import floating_lookback_price
    exact=floating_lookback_price(1.,1.,.05,.3,'put',running_max=1.)
    metrics=dict(network_initial_price=initial,euler_grid_mc=mc,euler_grid_mc_se=math.sqrt(max(0.,var_mc)/count),
        control_variate_mc=cv,control_variate_se=math.sqrt(max(0.,var_cv)/count),control_integral_mean=integral,
        variance_reduction_factor=var_mc/var_cv if var_cv>0 else None,
        payoff_integral_correlation=covariance/math.sqrt(var_mc*var_int) if var_mc*var_int>0 else None,
        network_relative_error_vs_euler_mc=abs(initial-mc)/abs(mc),continuous_analytic=exact,
        network_relative_error_vs_continuous=abs(initial-exact)/abs(exact),mc_paths=count,mc_seed=20261103,
        monitoring='Author Euler SDE and grid maximum, 500 fine steps; CV removes network pricing bias for this discrete model, not monitoring/SDE bias.',
        z_interpretation='Native author dfdx output is used as trained Brownian-integral coefficient, not claimed to be an AD derivative.')
    write_json(root/'initial_metrics.json',metrics);print('initial_metrics='+json.dumps(metrics),flush=True)
    mean_defect=defect_sums/count
    defect_se=((defect_squares-count*mean_defect.square()).clamp_min(0)/(count-1)/count).sqrt()
    write_json(root/'time_consistency.json',dict(times=coarse.cpu().tolist(),paths=count,
        discounted_y_mean=(mean_sums/count).cpu().tolist(),terminal_anchored_mean_defect=mean_defect.cpu().tolist(),
        paired_standard_error=defect_se.cpu().tolist(),
        scope='Unconditional necessary martingale check only; passing does not certify conditional prices. Terminal output is learned, not replaced by payoff.'))
    torch.manual_seed(20261104)
    paths,features,_=model.prepare_data(ts,torch.ones(4,1,device=ts.device),20);predictions=model.f(features)
    rows=[]
    for path_id in range(4):
        for coarse_id in [5,12,20,25]:
            fine_id=coarse_id*20;t=float(ts[fine_id]);prefix=paths[path_id:path_id+1,:fine_id+1,:]
            prediction=float(predictions[path_id,coarse_id,0]);total=squared=0.
            if fine_id==500:
                truth=float(option.payoff(prefix));se=0.
            else:
                for start in range(0,nested_count,1024):
                    size=min(1024,nested_count-start)
                    past=prefix.repeat(size,1,1);future,_=model.sdeint(ts[fine_id:],past[:,-1,:])
                    full=torch.cat([past,future[:,1:,:]],dim=1)
                    values=math.exp(-.05*(1-t))*option.payoff(full).double()
                    total+=values.sum().item();squared+=values.square().sum().item()
                truth=total/nested_count;variance=max(0.,(squared-nested_count*truth*truth)/(nested_count-1));se=math.sqrt(variance/nested_count)
            rows.append(dict(path=path_id,t=t,spot=float(prefix[0,-1,0]),running_max=float(prefix.max()),prediction=prediction,
                euler_grid_mc=truth,mc_se=se,absolute_error=abs(prediction-truth),relative_error=abs(prediction-truth)/max(abs(truth),1e-6)))
    with (root/'pathwise.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    torch.save(dict(paths=paths,predictions=predictions,times=coarse),root/'pathwise_samples.pt')
    write_json(root/'evaluation_manifest.json',dict(initial_mc_seed=20261103,pathwise_seed=20261104,paths=4,
        nested_mc_paths=nested_count,coarse_indices=[5,12,20,25],source_unmodified=True,
        control_variate='Evaluation wrapper implements discounted adapted stochastic-integral CV using native author Z; no training labels or anchors added.'))


def train(args,root,source):
    if (root/'config.json').exists():raise ValueError('Existing experiment: choose a new output directory')
    verify_source(source);cls,option_cls=load_author(source)
    config=dict(status='running',source_commit=COMMIT,source_dir=str(source.resolve()),
        scope='Author README 1D floating-lookback put example, not the paper main 2D experiment and not our barrier variant.',
        epochs=10000,batch_size=500,steps=500,lag=20,bsde_intervals=25,depth=3,hidden=20,ffn_hidden=[20,20],
        T=1.,r=.05,sigma=.3,dimension=1,optimizer='RMSprop',lr=.0005,lr_step=5000,lr_gamma=.2,
        seed=1,threads=1,torch=torch.__version__,device=args.device,precision='float32',
        changes='Recording, seed/thread selection, checkpoints and independent evaluation only; author source and training objective unchanged.')
    write_json(root/'config.json',config)
    torch.manual_seed(1);model=make_model(cls).to(args.device);option=option_cls();ts=torch.linspace(0,1,501,device=args.device)
    optimizer=torch.optim.RMSprop(model.parameters(),lr=.0005)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5000,gamma=.2)
    started=time.perf_counter();recent=[]
    try:
        with (root/'history.csv').open('w',newline='',encoding='utf-8',buffering=1) as stream:
            writer=csv.DictWriter(stream,fieldnames=['epoch','loss','bsde','terminal','elapsed_seconds']);writer.writeheader()
            for epoch in range(1,10001):
                optimizer.zero_grad();x0=sample_x0(500,args.device)
                loss,y,payoff=model.fbsdeint(ts=ts,x0=x0,option=option,lag=20)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite author training loss')
                loss.backward();optimizer.step();scheduler.step()
                terminal=(y[:,-1,:]-payoff).square().mean().item();value=loss.item();elapsed=time.perf_counter()-started
                writer.writerow(dict(epoch=epoch,loss=value,bsde=value-terminal,terminal=terminal,elapsed_seconds=elapsed))
                recent.append(value);recent=recent[-100:]
                if epoch==1 or epoch%100==0:
                    write_json(root/'progress.json',dict(epoch=epoch,total_epochs=10000,loss=value,mean_last_100=sum(recent)/len(recent),
                        elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(10000-epoch)/epoch))
                    print(f'epoch={epoch:05d} loss={value:.9g} mean100={sum(recent)/len(recent):.9g} elapsed={elapsed:.1f}s',flush=True)
                if epoch%1000==0:
                    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,rng_state=torch.get_rng_state(),config=config),root/'checkpoint_last.pt')
        config['training_elapsed_seconds']=time.perf_counter()-started;config['status']='evaluating';write_json(root/'config.json',config)
        torch.save(dict(state_dict=model.state_dict(),config=config),root/'model.pt')
        evaluate(model,option,ts,root,args.mc_paths,args.nested_mc_paths)
        verify_source(source);config['status']='complete';write_json(root/'config.json',config)
        print('Author README reproduction and evaluation complete',flush=True)
    except BaseException:
        config['status']='failed';write_json(root/'config.json',config)
        (root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--mc-paths',type=int,default=131072)
    parser.add_argument('--nested-mc-paths',type=int,default=8192)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    if min(args.mc_paths,args.nested_mc_paths)<2:parser.error('MC counts must be >=2')
    source=prepare(root);verify_source(source)
    if args.prepare_only:print('Prepared pinned, unmodified author source at',source);return
    torch.set_num_threads(1)
    if args.check_only:
        cls,option_cls=load_author(source);torch.manual_seed(1);model=make_model(cls).to(args.device);option=option_cls();ts=torch.linspace(0,1,501,device=args.device)
        loss,y,payoff=model.fbsdeint(ts,torch.ones(4,1,device=args.device),option,20);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(isinstance(x,torch.nn.LSTM) for x in model.modules())==2
        with torch.no_grad():
            model.train(False);paths,features,dw=model.prepare_data(ts,torch.ones(4,1,device=args.device),20);z=model.dfdx(features)
            assert z.shape==dw.shape and model.f(features).shape==(4,26,1)
            assert torch.equal(option.payoff(paths).squeeze(-1),paths[:,:,0].max(1).values-paths[:,-1,0])
        print('Author loss/backward, two LSTMs, Euler paths, 26 outputs and lookback payoff verified');return
    train(args,root,source)


if __name__=='__main__':main()
