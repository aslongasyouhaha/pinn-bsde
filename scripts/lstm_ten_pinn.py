"""Pure-history LSTM-PINN for the ten-contract study (six trained cases, parity knock-ins).

Same network and loss as sharp_history_pinn.py / sharp_history_pinn_lookback.py
(observed-price LSTM with learnable level features, query level features, chrono
init; PDE between observations scaled by the history-free lift H; stopped-gradient
canonical observation interface summed over dates; canonical terminal loss).
Budget: 3000 terminal-only updates + 47000 joint updates (total 50000) with the
joint learning rate decayed by cosine to lr_final_factor*lr. 128-date discrete
monitoring; no running extremum, hit flag or alive gate is ever an input.
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

from .lstm_ten_contracts import CASES, lift, payoff, score
from .reproduce_author_lookback import load_author, verify_source, write_json
from .sharp_history_pinn import SharpHistoryValue, simulate
from .single_asset_signature_bsde import make_model


class TenValue(SharpHistoryValue):
    def value(self, ctx, t, s, a):
        return lift(a.case, t, s, a)*torch.sigmoid(self.logit(ctx, t, s, a))


def pinn_loss(net, prices, ts, a, terminal_only=False):
    n = len(prices)
    with torch.backends.cudnn.flags(enabled=False):
        ctx = net.context(prices, ts, a)
    tt, sp = ts[1:].expand(n, -1), prices[:, 1:]
    H = lift(a.case, tt, sp, a)
    a_right = net.logit(ctx[:, 1:], tt, sp, a)
    g = payoff(a.case, prices)
    terminal = (H[:, -1]*nn.functional.softplus(a_right[:, -1])-g*a_right[:, -1]).mean()
    if terminal_only:
        zero = terminal*0
        return terminal, dict(pde=zero, interface=zero, terminal=terminal)
    b = torch.arange(n, device=a.device)
    k = torch.randint(a.steps, (n,), device=a.device)
    dt = a.T/a.steps
    u = (.001+.998*torch.rand(n, device=a.device))*dt
    t = (ts[k]+u).detach().requires_grad_(True)
    s = (prices[b, k]*torch.exp((a.r-.5*a.sigma**2)*u+a.sigma*u.sqrt()*torch.randn_like(u))).detach().requires_grad_(True)
    v = net.value(ctx[b, k], t, s, a)
    vt, vs = torch.autograd.grad(v.sum(), (t, s), create_graph=True)
    vss = torch.autograd.grad(vs.sum(), s, create_graph=True)[0]
    scale = lift(a.case, t, s, a).detach()
    if getattr(a, 'pde_scale', 'lift') == 'uncapped' and a.case == 'lookback_put':
        scale = scale/a.put_cap          # normalise by S, not cap*S (the cap would down-weight the PDE by cap^2)
    residual = (vt+a.r*s*vs+.5*a.sigma**2*s.square()*vss-a.r*v)/scale.clamp_min(a.q_floor)
    pde = residual.square().mean()
    right = (H*torch.sigmoid(a_right)).detach()
    a_left = net.logit(ctx[:, :-1], tt, sp, a)
    interface = (H*nn.functional.softplus(a_left)-right*a_left).mean(0).sum()
    return pde+interface+terminal, dict(pde=pde, interface=interface, terminal=terminal)


@torch.no_grad()
def predict(net, prices, ts, a):
    ctx = net.context(prices, ts, a)
    return net.value(ctx, ts.expand_as(prices), prices, a)


def run(a):
    root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
    if (root/'config.json').exists():
        raise FileExistsError('Use a fresh directory')
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    verify_source(Path(a.source_dir)); cls, _ = load_author(Path(a.source_dir))
    sampler = make_model(cls, a)
    ts = torch.linspace(0, a.T, a.steps+1, device=a.device)
    write_json(root/'config.json', dict(vars(a), method='LSTM-PINN', monitoring='128-date discrete',
        explicit_history_state_inputs=False, mc_anchor=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    shutil.copy2(__file__, root/'implementation.py')
    net = TenValue(a).to(a.device)
    start = time.perf_counter()
    with (root/'history.csv').open('w', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=['phase', 'update', 'lr', 'loss', 'pde', 'interface', 'terminal', 'seconds'])
        writer.writeheader()
        for phase, updates in [('terminal_only', a.pretrain_updates), ('joint', a.total_training_updates-a.pretrain_updates)]:
            torch.manual_seed(a.seed if phase == 'terminal_only' else a.seed+1)
            opt = torch.optim.RMSprop(net.parameters(), lr=a.lr)
            for update in range(1, updates+1):
                lr = a.lr if phase == 'terminal_only' else a.lr*(a.lr_final_factor+(1-a.lr_final_factor)*.5*(1+math.cos(math.pi*(update-1)/max(updates-1, 1))))
                for group in opt.param_groups:
                    group['lr'] = lr
                loss, parts = pinn_loss(net, simulate(sampler, ts, a, a.batch_size), ts, a, terminal_only=phase == 'terminal_only')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                opt.zero_grad(); loss.backward(); opt.step()
                if update == 1 or update % 500 == 0 or update == updates:
                    row = dict(phase=phase, update=update, lr=lr, loss=float(loss), **{k: float(v) for k, v in parts.items()},
                               seconds=time.perf_counter()-start)
                    writer.writerow(row); write_json(root/'progress.json', dict(status='training', **row)); print(row, flush=True)
            torch.save(dict(value=net.state_dict(), args=vars(a), phase=phase), root/(phase+'.pt'))
    write_json(root/'progress.json', dict(status='evaluating', training_seconds=time.perf_counter()-start))
    net.train(False)
    torch.manual_seed(994)
    prices = torch.cat([simulate(sampler, ts, a, 1024).cpu() for _ in range(a.exact_paths//1024)])
    pred = torch.cat([predict(net, prices[j:j+1024].to(a.device), ts, a).cpu() for j in range(0, len(prices), 1024)])
    nodes = list(range(0, a.steps+1, a.lag))
    result = score(a.case, a, prices, pred[:, nodes], nodes)
    result.update(method='LSTM-PINN', seed=a.seed, total_training_updates=a.total_training_updates)
    write_json(root/'scores.json', result)
    write_json(root/'progress.json', dict(status='complete', seconds=time.perf_counter()-start,
                                          t0=result['out']['t0_prediction'], t0_exact=result['out']['t0_exact']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source-dir', default='deep_ppde')
    p.add_argument('--device', default='cuda')
    for flag in ['level-features', 'query-level', 'chrono']:
        p.add_argument('--'+flag, action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pde-scale', choices=['lift', 'uncapped'], default='lift',
                   help='Divide the PDE residual by the lift (original) or by the lift without the put cap.')
    for key, value in [('s0', 1.), ('r', .03), ('sigma', .4), ('T', 1.), ('lr', .0005), ('lr_final_factor', .05),
                       ('q_floor', 1e-3), ('level_gamma', 50.), ('put_cap', 6.)]:
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=float, default=value)
    for key, value in [('steps', 128), ('lag', 8), ('depth', 3), ('hidden', 20), ('batch_size', 128), ('seed', 1),
                       ('level_count', 32), ('pretrain_updates', 3000), ('total_training_updates', 50000), ('exact_paths', 8192)]:
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=int, default=value)
    a = p.parse_args()
    try:
        run(a)
    except Exception:
        import traceback
        root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
        write_json(root/'failure.json', dict(error=traceback.format_exc()))
        raise
