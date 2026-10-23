"""Single-asset discrete barrier pilots without supplied historical state.

H is a Gaussian heat convolution of the terminal-only capped call, not a
continuous absorbing-barrier solution. Y=H*sigmoid(A(history,t,S)). This bounds
prices but DOES NOT impose the historical terminal condition. The latter is
learned from payoff MSE. No hit labels, MC anchors or extrema enter either net.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from .barrier_heat_kernel import interval_exponential
from .reproduce_author_lookback import load_author, verify_source, write_json
from .single_asset_signature_bsde import make_model, conditional_reference


def heat_lift(t, spot, a):
    """e^-r*tau E[(S_T-K)+ 1(S_T<B)|S_t]; no past information."""
    tau = (a.T-t).clamp_min(0)
    safe = torch.where(tau > 0, tau, torch.ones_like(tau))
    mean = spot.log()+(a.r-.5*a.sigma**2)*safe
    var = a.sigma**2*safe
    lo, hi = math.log(a.strike), math.log(a.barrier)
    stock = interval_exponential(1, mean, var, lo, hi)
    mass = interval_exponential(0, mean, var, lo, hi)
    h = torch.exp(-a.r*safe)*(stock-a.strike*mass)
    local_payoff = torch.relu(spot-a.strike)*(spot < a.barrier)
    return torch.where(tau > 0, h.clamp_min(0), local_payoff)


def payoff(paths, a):
    # Only a target/reference. Never passed to the predictor.
    return torch.relu(paths[:, -1, 0]-a.strike)*(paths[:, :, 0].max(1).values < a.barrier)


class HistoryValue(nn.Module):
    def __init__(self, channels, a):
        super().__init__()
        self.encoder = nn.LSTM(channels, a.hidden, batch_first=True)
        activation = nn.SiLU() if getattr(a, 'head_activation', 'tanh') == 'silu' else nn.Tanh()
        self.head = nn.Sequential(nn.Linear(a.hidden+2, a.hidden), activation, nn.Linear(a.hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def value(self, context, t, s, a):
        features = torch.cat([context, (t/a.T).unsqueeze(-1), (s/a.s0).log().unsqueeze(-1)], -1)
        fraction = torch.sigmoid(self.head(features).squeeze(-1))
        return heat_lift(t, s, a)*fraction

    def forward(self, features, ts, spots, a):
        context, _ = self.encoder(features)
        return self.value(context, ts.expand_as(spots), spots, a)


class BrownianCoefficient(nn.Module):
    def __init__(self, channels, a):
        super().__init__()
        self.encoder = nn.LSTM(channels, a.hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(a.hidden+2, a.hidden), nn.Tanh(), nn.Linear(a.hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, ts, spots, a):
        h, _ = self.encoder(features)
        x = torch.cat([h, (ts/a.T).expand_as(spots).unsqueeze(-1), (spots/a.s0).log().unsqueeze(-1)], -1)
        return self.head(x).squeeze(-1)  # Z, not Delta; no extra sigma*S


def raw_features(paths, ts, a):
    return torch.stack([(ts/a.T).expand_as(paths[:, :, 0]), (paths[:, :, 0]/a.s0).log()], -1)


def draw(sampler, ts, a, n):
    x0 = torch.full((n, 1), a.s0, device=a.device)
    if a.method == 'bsde':
        return sampler.prepare_data(ts, x0, a.lag)
    x, dw = sampler.sdeint(ts, x0)
    return x, raw_features(x, ts, a), dw[:, :, 0]


def bsde_loss(net, znet, sampler, ts, a):
    x, features, dw = draw(sampler, ts, a, a.batch_size)
    ct = ts[::a.lag];spots = x[:, ::a.lag, 0]
    y, z = net(features, ct, spots, a), znet(features, ct, spots, a)
    residual = y[:, :-1]+z[:, :-1]*dw[:, :-1, 0]-torch.exp(-a.r*(ct[1:]-ct[:-1]))*y[:, 1:].detach()
    dynamics = residual.square().mean(0).sum()
    terminal = (y[:, -1]-payoff(x, a)).square().mean()
    zero = dynamics*0
    return dynamics+terminal, dict(dynamics=dynamics, terminal=terminal, pde=zero,
        initial=y[:, 0].mean(), terminal_mean=y[:, -1].mean(), payoff_mean=payoff(x, a).mean())


def terminal_loss(net, sampler, ts, a):
    """Known payoff only; preserve the joint run's path RNG stream."""
    x, features, _ = draw(sampler, ts, a, a.batch_size)
    if a.method == 'pinn':
        torch.randint(a.steps, (len(x),), device=a.device)
        torch.rand(len(x), device=a.device)
        torch.randn(len(x), device=a.device)
    ct = ts[::a.lag] if a.method == 'bsde' else ts
    spots = x[:, ::a.lag, 0] if a.method == 'bsde' else x[:, :, 0]
    with torch.backends.cudnn.flags(enabled=a.method == 'bsde'):
        y = net(features, ct, spots, a)
    g = payoff(x, a)
    terminal = (y[:, -1]-g).square().mean()
    zero = terminal*0
    return terminal, dict(dynamics=zero, terminal=terminal, pde=zero,
        initial=y[:, 0].mean(), terminal_mean=y[:, -1].mean(), payoff_mean=g.mean())


def pinn_loss(net, sampler, ts, a):
    x, features, _ = draw(sampler, ts, a, a.batch_size)
    # Native LSTM avoids cuDNN's unsupported second derivatives; no detach of
    # contexts with respect to network parameters. Query t/S are separate.
    with torch.backends.cudnn.flags(enabled=False):
        context, _ = net.encoder(features)
    b = torch.arange(len(x), device=a.device)
    k = torch.randint(a.steps, (len(x),), device=a.device)
    dt = a.T/a.steps
    u = (.001+.998*torch.rand(len(x), device=a.device))*dt
    t = (ts[k]+u).detach().requires_grad_(True)
    s = (x[b, k, 0]*torch.exp((a.r-.5*a.sigma**2)*u+
         a.sigma*u.sqrt()*torch.randn_like(u))).detach().requires_grad_(True)
    v = net.value(context[b, k], t, s, a)
    vt, vs = torch.autograd.grad(v.sum(), (t, s), create_graph=True)
    vss = torch.autograd.grad(vs.sum(), s, create_graph=True)[0]
    residual = vt+a.r*s*vs+.5*a.sigma**2*s.square()*vss-a.r*v
    pde = residual.square().mean()
    # Left limit at each observation uses old memory; right limit consumes
    # that observation. Neither side receives a true hit flag. Full-gradient
    # mode differentiates the actual PINN interface condition on both sides;
    # stopped mode preserves the historical one-sided update for comparison.
    left = net.value(context[:, :-1], ts[1:].expand_as(x[:, 1:, 0]), x[:, 1:, 0], a)
    right = net.value(context[:, 1:], ts[1:].expand_as(x[:, 1:, 0]), x[:, 1:, 0], a)
    target = right if getattr(a,'interface_gradient','stopped')=='full' else right.detach()
    dynamics = (left-target).square().mean(0).sum()
    terminal = (right[:, -1]-payoff(x, a)).square().mean()
    initial = net.value(context[:, 0], ts[0].expand(len(x)), x[:, 0, 0], a)
    return pde+dynamics+terminal, dict(dynamics=dynamics, terminal=terminal, pde=pde,
        initial=initial.mean(), terminal_mean=right[:, -1].mean(), payoff_mean=payoff(x, a).mean())


@torch.no_grad()
def predictions(net, znet, sampler, ts, x, a):
    if a.method == 'bsde':
        f = sampler.get_stream_signatures(ts, x, a.lag)
        ct, s = ts[::a.lag], x[:, ::a.lag, 0]
        return net(f, ct, s, a), znet(f, ct, s, a)
    return net(raw_features(x, ts, a), ts, x[:, :, 0], a), None


def preflight(net, znet, sampler, ts, a):
    # Future perturbation must not change an earlier query. Also verify
    # bound, native learned terminal, finite train gradients and H at T.
    torch.manual_seed(991)
    x, _, _ = draw(sampler, ts, a, 8)
    j = a.steps//2
    before, _ = predictions(net, znet, sampler, ts, x, a)
    changed = x.clone();changed[:, j+1:] *= 1.7
    after, _ = predictions(net, znet, sampler, ts, changed, a)
    stride = a.lag if a.method == 'bsde' else 1
    causal = float((before[:, :j//stride+1]-after[:, :j//stride+1]).abs().max())
    assert causal < 1e-6
    h = heat_lift(ts[::stride].expand_as(before), x[:, ::stride, 0], a)
    assert torch.all(before >= 0) and torch.all(before <= h+1e-7)
    terminal_h = heat_lift(ts[-1].expand(len(x)), x[:, -1, 0], a)
    assert torch.equal(terminal_h, torch.relu(x[:, -1, 0]-a.strike)*(x[:, -1, 0]<a.barrier))
    loss, _ = bsde_loss(net,znet,sampler,ts,a) if a.method=='bsde' else pinn_loss(net,sampler,ts,a)
    loss.backward()
    parameters=[p for p in net.parameters() if p.requires_grad]+(list(znet.parameters()) if znet else [])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    for p in parameters:p.grad=None
    if znet:assert {id(p) for p in net.parameters()}.isdisjoint({id(p) for p in znet.parameters()})
    return dict(prefix_future_perturbation_max_error=causal,finite_gradients=True,
        price_bounds_verified=True,terminal_local_H_verified=True,
        true_history_terminal_hard=False,explicit_history_state_inputs=False)


@torch.no_grad()
def evaluate(net, znet, sampler, ts, a, root):
    net.train(False)
    if znet:znet.train(False)
    torch.manual_seed(992)
    ys=[];gs=[];ints=[]
    for start in range(0,a.mc_paths,512):
        x, _, dw=draw(sampler,ts,a,min(512,a.mc_paths-start))
        y,z=predictions(net,znet,sampler,ts,x,a)
        stride=a.lag if a.method=='bsde' else 1
        ys.append((y*torch.exp(-a.r*ts[::stride])).double().cpu())
        gs.append((math.exp(-a.r*a.T)*payoff(x,a)).double().cpu())
        if z is not None:
            ints.append((z[:,:-1]*dw[:,:-1,0]*torch.exp(-a.r*ts[::a.lag][:-1])).sum(1).double().cpu())
    y=torch.cat(ys);g=torch.cat(gs);n=len(g)
    defect=y-g[:,None]
    audit=dict(paths=n,initial_price=float(y[:,0].mean()),grid_mc=float(g.mean()),
        grid_mc_se=float(g.std()/math.sqrt(n)),terminal_rmse=float((math.exp(a.r*a.T)*defect[:,-1]).square().mean().sqrt()),
        discounted_y_mean=y.mean(0).tolist(),mean_defect=defect.mean(0).tolist(),
        mean_defect_se=(defect.std(0)/math.sqrt(n)).tolist(),
        initial_relative_gap=float((y[:,0].mean()-g.mean()).abs()/g.mean()),
        terminal_is_learned=True)
    if ints:
        integral=torch.cat(ints);cv=g-integral
        audit.update(payoff_integral_correlation=float(((g-g.mean())*(integral-integral.mean())).mean()/(g.std(unbiased=False)*integral.std(unbiased=False)).clamp_min(1e-30)),
            variance_reduction=float(g.var()/cv.var()),integral_mean=float(integral.mean()),
            integral_mean_se=float(integral.std()/math.sqrt(n)))
    write_json(root/'mean_audit.json',audit)
    torch.manual_seed(993)
    x,_,_=draw(sampler,ts,a,a.eval_paths);pred,_=predictions(net,znet,sampler,ts,x,a)
    rows=[];summaries=[]
    for k in [0,a.steps//4,a.steps//2,3*a.steps//4,a.steps-a.lag,a.steps]:
        for i in range(a.eval_paths):
            ref,se=conditional_reference(x[i:i+1,:k+1],k,a,80000+k*a.eval_paths+i)
            rows.append(dict(path=i,index=k,time=a.T*k/a.steps,dead=bool(x[i,:k+1,0].max()>=a.barrier),
                prediction=float(pred[i,k//stride]),reference=ref,reference_se=se))
        rr=[r for r in rows if r['index']==k];den=sum(r['reference']**2 for r in rr)
        dead=[r for r in rr if r['dead']]
        summaries.append(dict(time=a.T*k/a.steps,
            relative_l2=math.sqrt(sum((r['prediction']-r['reference'])**2 for r in rr)/den) if den else None,
            reference_noise_l2=math.sqrt(sum(r['reference_se']**2 for r in rr)/den) if den else None,
            dead_count=len(dead),dead_price_rmse=math.sqrt(sum(r['prediction']**2 for r in dead)/len(dead)) if dead else None))
    with (root/'conditional_prices.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(root/'conditional_summary.json',dict(paths=a.eval_paths,nested_paths=a.nested_paths,by_time=summaries))
    return audit


def run(a):
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise FileExistsError('Use a new output directory')
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    source=Path(a.source_dir);verify_source(source);cls,_=load_author(source)
    sampler=make_model(cls,a)
    channels=sampler.sig_channels if a.method=='bsde' else 2
    net=HistoryValue(channels,a).to(a.device)
    znet=BrownianCoefficient(channels,a).to(a.device) if a.method=='bsde' else None
    initial_sha=None
    if a.initial_value:
        initial=torch.load(a.initial_value,map_location=a.device)
        if a.method!='pinn' or initial['args']['method']!='pinn' or not initial.get('terminal_only'):
            raise ValueError('Initial value must be a PINN terminal-only checkpoint')
        for key in ['s0','strike','barrier','r','sigma','T','steps','hidden']:
            if initial['args'][key]!=getattr(a,key):raise ValueError('Initial checkpoint mismatch: '+key)
        if initial['args'].get('head_activation','tanh')!=a.head_activation:
            raise ValueError('Initial checkpoint activation mismatch')
        net.load_state_dict(initial['value'])
        initial_sha=hashlib.sha256(Path(a.initial_value).read_bytes()).hexdigest()
    encoder_initial=None
    if a.freeze_encoder:
        if a.method!='pinn' or not a.initial_value:raise ValueError('Freezing requires a pretrained PINN encoder')
        encoder_initial={k:v.detach().clone() for k,v in net.encoder.state_dict().items()}
        for p in net.encoder.parameters():p.requires_grad_(False)
    ts=torch.linspace(0,a.T,a.steps+1,device=a.device)
    write_json(root/'config.json',dict(vars(a),history_inputs='observed prices/times only; BSDE segment logsignatures',
        H='discounted Gaussian convolution of terminal-only capped call; no future intermediate monitoring',
        output='H*sigmoid(history readout)',full_terminal_hard=False,
        hit_or_extrema_supervision=False,mc_anchor=False,
        all_loss_weights=1 if a.terminal_warmup+a.dynamics_ramp==0 else 'epoch-only curriculum; final weights all 1',
        final_loss_weights=1,
        clipping=None,independent_Y_Z=a.method=='bsde',
        initial_checkpoint_sha256=initial_sha,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    shutil.copy2(__file__,root/'implementation.py')
    write_json(root/'validation.json',preflight(net,znet,sampler,ts,a))
    # Reset training RNG after validation for reproducibility.
    torch.manual_seed(a.seed)
    params=[p for p in net.parameters() if p.requires_grad]+(list(znet.parameters()) if znet else [])
    optimizer=torch.optim.RMSprop(params,lr=a.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,5000,.2)
    start=time.perf_counter()
    with (root/'history.csv').open('w',newline='',buffering=1) as f:
        keys=['epoch','loss','dynamics','terminal','pde','initial','terminal_mean','payoff_mean','seconds','phase','dynamics_factor']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for epoch in range(1,a.epochs+1):
            optimizer.zero_grad()
            if epoch <= a.terminal_warmup:
                loss,parts=terminal_loss(net,sampler,ts,a)
                factor,phase=0.,'terminal_only'
            else:
                loss,parts=bsde_loss(net,znet,sampler,ts,a) if a.method=='bsde' else pinn_loss(net,sampler,ts,a)
                factor=min(1.,(epoch-a.terminal_warmup)/a.dynamics_ramp) if a.dynamics_ramp else 1.
                phase='ramp' if factor<1 else 'joint'
                if factor != 1.:
                    loss=parts['terminal']+factor*(parts['dynamics']+parts['pde'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();optimizer.step();scheduler.step()
            row=dict(epoch=epoch,loss=float(loss),**{k:float(v) for k,v in parts.items()},seconds=time.perf_counter()-start,
                phase=phase,dynamics_factor=factor)
            w.writerow(row)
            if epoch==1 or epoch%100==0 or epoch==a.epochs:
                write_json(root/'progress.json',dict(status='training',**row));print(row,flush=True)
            if a.terminal_warmup and epoch in (a.terminal_warmup,a.terminal_warmup+a.dynamics_ramp):
                torch.save(dict(value=net.state_dict(),z=znet.state_dict() if znet else None,args=vars(a),epoch=epoch),root/('phase_'+str(epoch)+'.pt'))
    if encoder_initial is not None:
        unchanged=all(torch.equal(v,net.encoder.state_dict()[k]) for k,v in encoder_initial.items())
        assert unchanged
        write_json(root/'frozen_encoder_validation.json',dict(encoder_parameters_unchanged=True,trainable_parameter_count=sum(p.numel() for p in params)))
    torch.save(dict(value=net.state_dict(),z=znet.state_dict() if znet else None,args=vars(a)),root/'model.pt')
    write_json(root/'progress.json',dict(status='evaluating',epoch=a.epochs,training_seconds=time.perf_counter()-start))
    audit=evaluate(net,znet,sampler,ts,a,root)
    write_json(root/'progress.json',dict(status='complete',epoch=a.epochs,total_seconds=time.perf_counter()-start,initial_price=audit['initial_price']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['bsde','pinn'],default='bsde')
    p.add_argument('--head-activation',choices=['tanh','silu'],default='tanh',
        help='Price head only; Z head remains unchanged for a controlled ablation.')
    p.add_argument('--terminal-warmup',type=int,default=0)
    p.add_argument('--dynamics-ramp',type=int,default=0)
    p.add_argument('--interface-gradient',choices=['stopped','full'],default='stopped',
        help='PINN observation interface: historical one-sided or full residual gradient.')
    p.add_argument('--initial-value',help='Optional PINN terminal-only checkpoint; optimizer starts fresh.')
    p.add_argument('--freeze-encoder',action='store_true',help='Keep the pretrained PINN history encoder fixed; train price head only.')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--source-dir',default='deep_ppde')
    p.add_argument('--device',default='cuda')
    for key,value in [('s0',1.),('strike',1.),('barrier',1.2),('r',.03),('sigma',.4),('T',1.),('lr',.0005)]:p.add_argument('--'+key,type=float,default=value)
    for key,value in [('steps',128),('lag',8),('depth',3),('hidden',20),('epochs',3000),('batch-size',128),('seed',1),('mc-paths',16384),('eval-paths',64),('nested-paths',8192)]:p.add_argument('--'+key,type=int,default=value)
    a=p.parse_args()
    if a.steps%a.lag or a.steps%4 or min(a.steps,a.lag,a.hidden,a.epochs,a.batch_size,a.mc_paths,a.eval_paths,a.nested_paths)<=0 or not (0<a.strike<a.barrier and 0<a.s0<a.barrier) or a.T<=0 or a.sigma<=0 or min(a.terminal_warmup,a.dynamics_ramp)<0 or a.terminal_warmup+a.dynamics_ramp>a.epochs:p.error('Invalid market/grid/counts/curriculum')
    try:run(a)
    except Exception:
        import traceback
        root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
        write_json(root/'failure.json',dict(error=traceback.format_exc()))
        raise
