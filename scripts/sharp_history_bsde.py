"""Pure-history backward-date BSDE with a sharp-readout history encoder.

Same contract, exact-GBM sampler, exact final-interval payoff, frozen next-date
targets, per-date fresh RMSprop, seeds and budget accounting as
exact_terminal_history_bsde.py. What changes (each switch is explicit):

* encoder: reads every observed (t, log S) up to the current BSDE date instead
  of lag-8 logsignature segments. With --level-features it also reads learnable
  level-crossing features sigmoid(gamma_m (log S - c_m)); the centres start on a
  uniform grid deliberately NOT aligned with log(B/S0) and both centres and
  sharpness are trained. No running maximum, hit flag or alive gate is given.
* --chrono: chrono forget/input-gate bias initialisation (horizon = fine steps).
* --loss canon: Y minimises H*softplus(A) - (target - Zbar dW)*A, whose
  conditional minimiser is the same conditional expectation as the squared
  residual but whose logit gradient is the residual itself; Z keeps the squared
  residual with Ybar. --loss mse is the original joint squared residual.
* --absorb: the regression target is exactly zero for paths knocked out by the
  next BSDE date. It uses the contract's knock-out rule in the TARGET only.

Evaluation repeats the original protocol (16384-path mean audit, seed 992; 64
histories with 8192 nested paths, seed 993) and adds the noise-free quadrature
reference on 8192 further histories (seed 994).
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
from torch import nn

from .history_models import freeze, tensor_hash
from .exact_barrier_reference import DiscreteUpOutReference, score
from .pure_history_h import heat_lift, payoff
from .reproduce_author_lookback import load_author, verify_source, write_json
from .single_asset_signature_bsde import conditional_reference, make_model


class LevelFeatures(nn.Module):
    def __init__(self, count, gamma, lo=-.6, hi=.6):
        super().__init__()
        centre = torch.linspace(lo, hi, count)
        self.centre = nn.Parameter(centre+.5*(centre[1]-centre[0]))
        self.log_gamma = nn.Parameter(torch.full((count,), math.log(gamma)))

    def forward(self, logs):
        return torch.sigmoid(self.log_gamma.exp()*(logs.unsqueeze(-1)-self.centre))


def chrono_(lstm, horizon):
    hidden = lstm.hidden_size
    with torch.no_grad():
        bias = torch.log(torch.empty(hidden, device=lstm.bias_ih_l0.device).uniform_(1, horizon-1))
        lstm.bias_ih_l0[hidden:2*hidden] = bias
        lstm.bias_hh_l0[hidden:2*hidden] = 0
        lstm.bias_ih_l0[:hidden] = -bias
        lstm.bias_hh_l0[:hidden] = 0


class FineHistoryNet(nn.Module):
    """Reads observed fine-grid prices up to BSDE date i; one scalar at date i."""
    def __init__(self, a):
        super().__init__()
        self.level = LevelFeatures(a.level_count, a.level_gamma) if a.level_features else None
        self.encoder = nn.LSTM(2+(a.level_count if a.level_features else 0), a.hidden, batch_first=True)
        if a.chrono:
            chrono_(self.encoder, a.steps)
        self.head = nn.Sequential(nn.Linear(a.hidden+2, a.hidden), nn.Tanh(), nn.Linear(a.hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, prices, ts, i, a):
        k = i*a.lag
        logs = torch.log(prices[:, :k+1]/a.s0)
        tokens = torch.stack([(ts[:k+1]/a.T).expand_as(logs), logs], -1)
        if self.level is not None:
            tokens = torch.cat([tokens, self.level(logs)], -1)
        h, _ = self.encoder(tokens)
        query = torch.cat([h[:, -1], (ts[k]/a.T).expand(len(logs), 1), logs[:, -1:]], -1)
        return self.head(query).squeeze(-1)


def price(net, prices, ts, i, a):
    k = i*a.lag
    logit = net(prices, ts, i, a)
    lift = heat_lift(ts[k].expand(len(prices)), prices[:, k], a)
    return lift*torch.sigmoid(logit), logit, lift


def simulate(sampler, ts, a, n):
    x, dw = sampler.sdeint(ts, torch.full((n, 1), a.s0, device=a.device))
    return x[:, :, 0], dw[:, :, 0]


def loss_terms(y, z, prices, dw, target, ts, i, a):
    value, logit, lift = price(y, prices, ts, i, a)
    dW = dw[:, i*a.lag:(i+1)*a.lag].sum(1)
    zz = z(prices, ts, i, a)
    residual = value+zz*dW-target
    if a.loss == 'mse':
        return residual.square().mean(), value
    canonical = (lift*nn.functional.softplus(logit)-(target-zz.detach()*dW)*logit).mean()
    return canonical+(value.detach()+zz*dW-target).square().mean(), value


def next_target(target, prices, ts, i, a, dates):
    with torch.no_grad():
        if target is None:
            assert i == dates-1
            return payoff(prices.unsqueeze(-1), a)
        nxt = price(target, prices, ts, i+1, a)[0]
        if a.absorb:
            nxt = nxt*(prices[:, :(i+1)*a.lag+1].max(1).values < a.barrier)
        return nxt


def pretrain_terminal(a, sampler, ts, dates, root):
    """Initialisation only (as terminal_control.pt): readout at T fitted to the payoff."""
    torch.manual_seed(a.seed)
    net = FineHistoryNet(a).to(a.device)
    opt = torch.optim.RMSprop(net.parameters(), lr=a.lr)
    rows = []
    for update in range(1, a.pretrain_updates+1):
        prices, _ = simulate(sampler, ts, a, a.batch_size)
        g = payoff(prices.unsqueeze(-1), a)
        value, logit, lift = price(net, prices, ts, dates, a)
        loss = ((value-g).square().mean() if a.loss == 'mse' else
                (lift*nn.functional.softplus(logit)-g*logit).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if update % 250 == 0 or update == a.pretrain_updates:
            rows.append(dict(update=update, loss=float(loss), terminal_mean=float(value.mean()), payoff_mean=float(g.mean())))
            print(dict(stage='terminal_pretraining', **rows[-1]), flush=True)
    write_json(root/'terminal_pretraining.json', rows)
    return net


@torch.no_grad()
def predict(values, prices, ts, a, dates):
    """(n, dates+1) predictions; final column is the exact payoff (structural)."""
    out = [price(values[i], prices, ts, i, a)[0] for i in range(dates)]
    return torch.stack(out+[payoff(prices.unsqueeze(-1), a)], 1)


@torch.no_grad()
def evaluate(values, zs, sampler, ts, a, root, dates):
    for m in values+zs:
        m.train(False)
    ct = ts[::a.lag]
    discount = torch.exp(-a.r*ct)
    torch.manual_seed(992)
    ys, gs, integrals = [], [], []
    for start in range(0, a.mc_paths, 512):
        prices, dw = simulate(sampler, ts, a, min(512, a.mc_paths-start))
        y = predict(values, prices, ts, a, dates)
        zz = torch.stack([zs[i](prices, ts, i, a) for i in range(dates)], 1)
        dW = dw.reshape(len(dw), dates, a.lag).sum(-1)
        ys.append((y*discount).double().cpu())
        gs.append((discount[-1]*payoff(prices.unsqueeze(-1), a)).double().cpu())
        integrals.append((zz*dW*discount[:-1]).sum(1).double().cpu())
    y, g, integral = torch.cat(ys), torch.cat(gs), torch.cat(integrals)
    n = len(g)
    cv = g-integral
    cov = ((g-g.mean())*(integral-integral.mean())).mean()
    audit = dict(paths=n, initial_price=float(y[:, 0].mean()), grid_mc=float(g.mean()), grid_mc_se=float(g.std()/n**.5),
                 discounted_y_mean=y.mean(0).tolist(), terminal_exact_by_construction=True,
                 initial_relative_gap=float((y[:, 0].mean()-g.mean()).abs()/g.mean()),
                 payoff_integral_correlation=float(cov/(g.std(unbiased=False)*integral.std(unbiased=False)).clamp_min(1e-30)),
                 variance_reduction=float(g.var()/cv.var()))
    write_json(root/'mean_audit.json', audit)
    # Same 64 histories and nested-MC seeds as exact_terminal_history_bsde.evaluate.
    torch.manual_seed(993)
    prices, _ = simulate(sampler, ts, a, a.eval_paths)
    prediction = predict(values, prices, ts, a, dates)
    reference = DiscreteUpOutReference(a)
    exact = reference.node_values(prices)
    rows, summaries = [], []
    for k in [0, a.steps//4, a.steps//2, 3*a.steps//4, a.steps-a.lag, a.steps]:
        for j in range(a.eval_paths):
            ref, se = conditional_reference(prices[j:j+1, :k+1].unsqueeze(-1), k, a, 80000+k*a.eval_paths+j)
            rows.append(dict(path=j, index=k, time=a.T*k/a.steps, dead=bool(prices[j, :k+1].max() >= a.barrier),
                             prediction=float(prediction[j, k//a.lag]), reference=ref, reference_se=se,
                             exact_reference=float(exact[j, k])))
        rr = [r for r in rows if r['index'] == k]
        den = sum(r['reference']**2 for r in rr)
        dead = [r for r in rr if r['dead']]
        exact_den = sum(r['exact_reference']**2 for r in rr)
        summaries.append(dict(time=a.T*k/a.steps,
            relative_l2=math.sqrt(sum((r['prediction']-r['reference'])**2 for r in rr)/den) if den else None,
            reference_noise_l2=math.sqrt(sum(r['reference_se']**2 for r in rr)/den) if den else None,
            relative_l2_vs_exact=math.sqrt(sum((r['prediction']-r['exact_reference'])**2 for r in rr)/exact_den) if exact_den else None,
            dead_count=len(dead),
            dead_price_rmse=math.sqrt(sum(r['prediction']**2 for r in dead)/len(dead)) if dead else None))
    with (root/'conditional_prices.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(root/'conditional_summary.json', dict(paths=a.eval_paths, nested_paths=a.nested_paths, by_time=summaries))
    # Larger noise-free scoring set.
    torch.manual_seed(994)
    scores = []
    prices_all, pred_all = [], []
    for start in range(0, a.exact_paths, 1024):
        prices, _ = simulate(sampler, ts, a, min(1024, a.exact_paths-start))
        prices_all.append(prices.cpu()); pred_all.append(predict(values, prices, ts, a, dates).cpu())
    prices, prediction = torch.cat(prices_all), torch.cat(pred_all)
    exact = reference.node_values(prices)
    for i in range(1, dates):
        scores.append(score(prediction[:, i], exact[:, i*a.lag], prices, a, i*a.lag))
    write_json(root/'exact_reference_scores.json', dict(histories=a.exact_paths, seed=994, by_date=scores,
               t0_prediction=float(prediction[0, 0]), t0_exact=float(exact[0, 0])))
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
    dates = a.steps//a.lag
    budget = a.total_training_updates-a.pretrain_updates
    stage_counts = [budget//dates+int(j < budget % dates) for j in range(dates)]
    write_json(root/'config.json', dict(vars(a), dates=dates, backward_updates=budget,
        updates_per_date_in_backward_order=stage_counts,
        encoder='fine-grid observed (t, log S)' + (' + learnable level-crossing features' if a.level_features else ''),
        explicit_history_state_inputs=False, mc_anchor=False, terminal='exact fine-grid payoff in final-interval TRAINING',
        absorbing_target=a.absorb, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    shutil.copy2(__file__, root/'implementation.py')
    start = time.perf_counter()
    initializer = pretrain_terminal(a, sampler, ts, dates, root) if a.pretrain_updates else FineHistoryNet(a).to(a.device)
    values, zs = [None]*dates, [None]*dates
    with (root/'history.csv').open('w', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'time', 'update', 'lr', 'loss', 'prediction_mean', 'target_mean', 'seconds'])
        writer.writeheader()
        for i in range(dates-1, -1, -1):
            target = None if i == dates-1 else values[i+1]
            target_hash = None if target is None else tensor_hash(target)
            y = copy.deepcopy(initializer if target is None else target)
            if target is None:
                torch.manual_seed(a.seed+500)
                z = FineHistoryNet(a).to(a.device)
            else:
                z = copy.deepcopy(zs[i+1])
            for model in [y, z]:
                model.train(True)
                for p in model.parameters():
                    p.requires_grad_(True)
            params = list(y.parameters())+list(z.parameters())
            opt = torch.optim.RMSprop(params, lr=a.lr)
            torch.manual_seed(a.seed+1000+i)
            disc = math.exp(-a.r*float(ts[(i+1)*a.lag]-ts[i*a.lag]))
            updates = stage_counts[dates-1-i]
            for update in range(1, updates+1):
                lr = a.lr*(a.lr_final_factor+(1-a.lr_final_factor)*.5*(1+math.cos(math.pi*(update-1)/max(updates-1, 1))))
                for group in opt.param_groups:
                    group['lr'] = lr
                prices, dw = simulate(sampler, ts, a, a.batch_size)
                nxt = disc*next_target(target, prices, ts, i, a, dates)
                loss, now = loss_terms(y, z, prices, dw, nxt, ts, i, a)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                opt.zero_grad(); loss.backward(); opt.step()
                if update == 1 or update % 250 == 0 or update == updates:
                    row = dict(index=i, time=float(ts[i*a.lag]), update=update, lr=lr, loss=float(loss),
                               prediction_mean=float(now.mean()), target_mean=float(nxt.mean()), seconds=time.perf_counter()-start)
                    writer.writerow(row)
                    write_json(root/'progress.json', dict(status='training', completed_dates=dates-1-i, dates=dates, **row))
                    print(row, flush=True)
            if target is not None:
                assert tensor_hash(target) == target_hash
            freeze(y); freeze(z)
            values[i], zs[i] = y, z
            torch.save(dict(value=y.state_dict(), z=z.state_dict(), index=i, args=vars(a)), root/('date_'+str(i)+'.pt'))
    torch.save(dict(values=[m.state_dict() for m in values], zs=[m.state_dict() for m in zs], args=vars(a)), root/'model.pt')
    write_json(root/'progress.json', dict(status='evaluating', training_seconds=time.perf_counter()-start))
    audit = evaluate(values, zs, sampler, ts, a, root, dates)
    write_json(root/'progress.json', dict(status='complete', seconds=time.perf_counter()-start, initial_price=audit['initial_price']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source-dir', default='vendor/deep_ppde')
    p.add_argument('--device', default='cuda')
    p.add_argument('--loss', choices=['mse', 'canon'], default='canon')
    p.add_argument('--level-features', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--chrono', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--absorb', action='store_true')
    p.add_argument('--level-count', type=int, default=32)
    p.add_argument('--level-gamma', type=float, default=50.)
    p.add_argument('--lr-final-factor', type=float, default=1.)
    for key, value in [('s0', 1.), ('strike', 1.), ('barrier', 1.2), ('r', .03), ('sigma', .4), ('T', 1.), ('lr', .0005)]:
        p.add_argument('--'+key, type=float, default=value)
    for key, value in [('steps', 128), ('lag', 8), ('depth', 3), ('hidden', 20), ('batch-size', 128), ('seed', 1),
                       ('pretrain-updates', 3000), ('total-training-updates', 15000), ('mc-paths', 16384),
                       ('eval-paths', 64), ('nested-paths', 8192), ('exact-paths', 8192)]:
        p.add_argument('--'+key, type=int, default=value)
    a = p.parse_args()
    if a.steps % a.lag or a.total_training_updates-a.pretrain_updates < a.steps//a.lag:
        p.error('Invalid grid or budget')
    try:
        run(a)
    except Exception:
        import traceback
        root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
        write_json(root/'failure.json', dict(error=traceback.format_exc()))
        raise
