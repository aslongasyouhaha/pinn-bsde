"""Pure-history LSTM-PINN with a sharp-readout history encoder.

Mirrors pure_history_h.py --method pinn: the LSTM reads all 129 observed
(t, log S); between observations the memory is frozen and the BS PDE residual is
taken with autograd in the query (t, S); at each observation the left limit
(old memory) is matched to the right value (new memory) with a stopped gradient;
the terminal value is matched to the known payoff. Y = H*sigmoid(A) with the same
barrier-free capped heat lift H. No running maximum, hit flag, alive gate or MC
anchor is supplied. Protocol: terminal-only initialisation, then joint training.

Changes (each switchable):
* --level-features: learnable level-crossing features of every observed log S
  (centres on a grid not aligned with the barrier; centres/sharpness trained);
* --query-level: the same kind of features of the query log S in the head;
* --chrono: chrono gate-bias initialisation;
* --loss canon: interface and terminal use H*softplus(A) - target*A (same
  minimiser as the squared error, logit gradient = residual);
* --pde-space q: the PDE residual is divided by max(H, floor) (Doob h-transform
  scale), so it is not suppressed by H^2 where prices are small.
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

from .exact_barrier_reference import DiscreteUpOutReference, score
from .pure_history_h import heat_lift, payoff
from .reproduce_author_lookback import load_author, verify_source, write_json
from .sharp_history_bsde import LevelFeatures, chrono_
from .single_asset_signature_bsde import conditional_reference, make_model


class SharpHistoryValue(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.level = LevelFeatures(a.level_count, a.level_gamma) if a.level_features else None
        self.qlevel = LevelFeatures(a.level_count, a.level_gamma) if a.query_level else None
        self.encoder = nn.LSTM(2+(a.level_count if a.level_features else 0), a.hidden, batch_first=True)
        if a.chrono:
            chrono_(self.encoder, a.steps)
        self.head = nn.Sequential(nn.Linear(a.hidden+2+(a.level_count if a.query_level else 0), a.hidden),
                                  nn.Tanh(), nn.Linear(a.hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def context(self, prices, ts, a):
        logs = torch.log(prices/a.s0)
        tokens = torch.stack([(ts/a.T).expand_as(logs), logs], -1)
        if self.level is not None:
            tokens = torch.cat([tokens, self.level(logs)], -1)
        return self.encoder(tokens)[0]

    def logit(self, ctx, t, s, a):
        logs = torch.log(s/a.s0)
        parts = [ctx, (t/a.T).unsqueeze(-1), logs.unsqueeze(-1)]
        if self.qlevel is not None:
            parts.append(self.qlevel(logs))
        return self.head(torch.cat(parts, -1)).squeeze(-1)

    def value(self, ctx, t, s, a):
        return heat_lift(t, s, a)*torch.sigmoid(self.logit(ctx, t, s, a))


def simulate(sampler, ts, a, n):
    x, _ = sampler.sdeint(ts, torch.full((n, 1), a.s0, device=a.device))
    return x[:, :, 0]


def pinn_loss(net, prices, ts, a, terminal_only=False):
    n = len(prices)
    with torch.backends.cudnn.flags(enabled=False):
        ctx = net.context(prices, ts, a)
    tt = ts[1:].expand(n, -1)
    sp = prices[:, 1:]
    lift = heat_lift(tt, sp, a)
    a_right = net.logit(ctx[:, 1:], tt, sp, a)
    g = payoff(prices.unsqueeze(-1), a)
    if a.loss == 'mse':
        terminal = (lift[:, -1]*torch.sigmoid(a_right[:, -1])-g).square().mean()
    else:
        terminal = (lift[:, -1]*nn.functional.softplus(a_right[:, -1])-g*a_right[:, -1]).mean()
    zero = terminal*0
    if terminal_only:
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
    residual = vt+a.r*s*vs+.5*a.sigma**2*s.square()*vss-a.r*v
    if a.pde_space == 'q':
        residual = residual/heat_lift(t, s, a).detach().clamp_min(a.q_floor)
    pde = residual.square().mean()
    right = (lift*torch.sigmoid(a_right)).detach()
    a_left = net.logit(ctx[:, :-1], tt, sp, a)
    if a.loss == 'mse':
        interface = (lift*torch.sigmoid(a_left)-right).square().mean(0).sum()
    else:
        interface = (lift*nn.functional.softplus(a_left)-right*a_left).mean(0).sum()
    if getattr(a, 'interface_reduction', 'sum') == 'mean':
        interface = interface/a.steps  # average over the 128 observation dates instead of summing
    return pde+interface+terminal, dict(pde=pde, interface=interface, terminal=terminal)


@torch.no_grad()
def predict(net, prices, ts, a):
    ctx = net.context(prices, ts, a)
    return net.value(ctx, ts.expand_as(prices), prices, a)


@torch.no_grad()
def evaluate(net, sampler, ts, a, root):
    net.train(False)
    discount = torch.exp(-a.r*ts)
    torch.manual_seed(992)
    ys, gs = [], []
    for start in range(0, a.mc_paths, 512):
        prices = simulate(sampler, ts, a, min(512, a.mc_paths-start))
        ys.append((predict(net, prices, ts, a)*discount).double().cpu())
        gs.append((discount[-1]*payoff(prices.unsqueeze(-1), a)).double().cpu())
    y, g = torch.cat(ys), torch.cat(gs)
    n = len(g)
    defect = y-g[:, None]
    audit = dict(paths=n, initial_price=float(y[:, 0].mean()), grid_mc=float(g.mean()), grid_mc_se=float(g.std()/n**.5),
                 terminal_rmse=float((math.exp(a.r*a.T)*defect[:, -1]).square().mean().sqrt()),
                 discounted_y_mean=y.mean(0).tolist(), initial_relative_gap=float((y[:, 0].mean()-g.mean()).abs()/g.mean()),
                 terminal_is_learned=True)
    write_json(root/'mean_audit.json', audit)
    torch.manual_seed(993)
    prices = simulate(sampler, ts, a, a.eval_paths)
    pred = predict(net, prices, ts, a)
    reference = DiscreteUpOutReference(a)
    exact = reference.node_values(prices)
    rows, summaries = [], []
    for k in [0, a.steps//4, a.steps//2, 3*a.steps//4, a.steps-a.lag, a.steps]:
        for i in range(a.eval_paths):
            ref, se = conditional_reference(prices[i:i+1, :k+1].unsqueeze(-1), k, a, 80000+k*a.eval_paths+i)
            rows.append(dict(path=i, index=k, time=a.T*k/a.steps, dead=bool(prices[i, :k+1].max() >= a.barrier),
                             prediction=float(pred[i, k]), reference=ref, reference_se=se, exact_reference=float(exact[i, k])))
        rr = [r for r in rows if r['index'] == k]
        den = sum(r['reference']**2 for r in rr)
        exact_den = sum(r['exact_reference']**2 for r in rr)
        dead = [r for r in rr if r['dead']]
        summaries.append(dict(time=a.T*k/a.steps,
            relative_l2=math.sqrt(sum((r['prediction']-r['reference'])**2 for r in rr)/den) if den else None,
            reference_noise_l2=math.sqrt(sum(r['reference_se']**2 for r in rr)/den) if den else None,
            relative_l2_vs_exact=math.sqrt(sum((r['prediction']-r['exact_reference'])**2 for r in rr)/exact_den) if exact_den else None,
            dead_count=len(dead), dead_price_rmse=math.sqrt(sum(r['prediction']**2 for r in dead)/len(dead)) if dead else None))
    with (root/'conditional_prices.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(root/'conditional_summary.json', dict(paths=a.eval_paths, nested_paths=a.nested_paths, by_time=summaries))
    torch.manual_seed(994)
    prices = torch.cat([simulate(sampler, ts, a, 1024).cpu() for _ in range(a.exact_paths//1024)])
    pred = torch.cat([predict(net, prices[j:j+1024].to(a.device), ts, a).cpu() for j in range(0, len(prices), 1024)])
    exact = reference.node_values(prices)
    scores = [score(pred[:, k], exact[:, k], prices, a, k) for k in range(a.lag, a.steps+1, a.lag)]
    s_t = prices[:, -1]
    band = (s_t > a.strike) & (s_t < a.barrier)
    alive = prices.max(1).values < a.barrier
    lift_t = torch.relu(s_t-a.strike)*band
    frac = pred[:, -1].double()/lift_t.double().clamp_min(1e-12)
    write_json(root/'exact_reference_scores.json', dict(histories=len(prices), seed=994, by_date=scores,
        t0_prediction=float(pred[0, 0]), t0_exact=float(exact[0, 0]),
        terminal_fraction_alive=float(frac[band & alive].mean()), terminal_fraction_dead=float(frac[band & ~alive].mean())))
    return audit


def run(a):
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'config.json').exists():
        raise FileExistsError('Use a fresh directory')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    verify_source(Path(a.source_dir))
    cls, _ = load_author(Path(a.source_dir))
    sampler = make_model(cls, a)
    ts = torch.linspace(0, a.T, a.steps+1, device=a.device)
    write_json(root/'config.json', dict(vars(a), explicit_history_state_inputs=False, mc_anchor=False,
        interface_gradient='stopped', all_loss_weights=1,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    shutil.copy2(__file__, root/'implementation.py')
    net = SharpHistoryValue(a).to(a.device)
    start = time.perf_counter()
    with (root/'history.csv').open('w', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=['phase', 'update', 'loss', 'pde', 'interface', 'terminal', 'seconds'])
        writer.writeheader()
        for phase, updates in [('terminal_only', a.pretrain_updates), ('joint', a.joint_updates)]:
            torch.manual_seed(a.seed if phase == 'terminal_only' else a.seed+1)
            opt = torch.optim.RMSprop(net.parameters(), lr=a.lr)
            for update in range(1, updates+1):
                prices = simulate(sampler, ts, a, a.batch_size)
                loss, parts = pinn_loss(net, prices, ts, a, terminal_only=phase == 'terminal_only')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                opt.zero_grad(); loss.backward(); opt.step()
                if update == 1 or update % 250 == 0 or update == updates:
                    row = dict(phase=phase, update=update, loss=float(loss), **{k: float(v) for k, v in parts.items()},
                               seconds=time.perf_counter()-start)
                    writer.writerow(row)
                    write_json(root/'progress.json', dict(status='training', **row))
                    print(row, flush=True)
            torch.save(dict(value=net.state_dict(), args=vars(a), phase=phase), root/(phase+'.pt'))
    write_json(root/'progress.json', dict(status='evaluating', training_seconds=time.perf_counter()-start))
    audit = evaluate(net, sampler, ts, a, root)
    write_json(root/'progress.json', dict(status='complete', seconds=time.perf_counter()-start, initial_price=audit['initial_price']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source-dir', default='deep_ppde')
    p.add_argument('--device', default='cuda')
    p.add_argument('--loss', choices=['mse', 'canon'], default='canon')
    p.add_argument('--pde-space', choices=['v', 'q'], default='q')
    p.add_argument('--q-floor', type=float, default=1e-3)
    p.add_argument('--interface-reduction', choices=['sum', 'mean'], default='sum',
                   help='Reduce the observation-interface term over dates by sum (original) or mean.')
    for flag in ['level-features', 'query-level', 'chrono']:
        p.add_argument('--'+flag, action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--level-count', type=int, default=32)
    p.add_argument('--level-gamma', type=float, default=50.)
    for key, value in [('s0', 1.), ('strike', 1.), ('barrier', 1.2), ('r', .03), ('sigma', .4), ('T', 1.), ('lr', .0005)]:
        p.add_argument('--'+key, type=float, default=value)
    for key, value in [('steps', 128), ('lag', 8), ('depth', 3), ('hidden', 20), ('batch-size', 128), ('seed', 1),
                       ('pretrain-updates', 3000), ('joint-updates', 3000), ('mc-paths', 16384),
                       ('eval-paths', 64), ('nested-paths', 8192), ('exact-paths', 8192)]:
        p.add_argument('--'+key, type=int, default=value)
    a = p.parse_args()
    try:
        run(a)
    except Exception:
        import traceback
        root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
        write_json(root/'failure.json', dict(error=traceback.format_exc()))
        raise
