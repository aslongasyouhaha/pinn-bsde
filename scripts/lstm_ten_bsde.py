"""Pure-history backward-date LSTM-BSDE for the ten-contract study (six trained cases, parity knock-ins).

Same protocol as sharp_history_bsde.py: fine-observation encoder with learnable
level features and chrono init, Y = H*sigmoid(A) with the history-free lift H,
canonical Y loss with a detached Z control variate, Z by squared residual,
terminal-only initialisation (3000) then 47000 backward updates spread over the
16 dates with per-date cosine decay to lr_final_factor*lr, frozen next-date
targets and the exact payoff in the final interval. No absorbing targets.
"""
import argparse
import copy
import csv
import hashlib
import math
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from .history_models import freeze, tensor_hash
from .lstm_ten_contracts import CASES, lift, payoff, score
from .reproduce_author_lookback import load_author, verify_source, write_json
from .sharp_history_bsde import FineHistoryNet, simulate
from .single_asset_signature_bsde import make_model


def price(net, prices, ts, i, a):
    k = i*a.lag
    H = lift(a.case, ts[k].expand(len(prices)), prices[:, k], a)
    logit = net(prices, ts, i, a)
    return H*torch.sigmoid(logit), logit, H


def loss_terms(y, z, prices, dw, target, ts, i, a):
    value, logit, H = price(y, prices, ts, i, a)
    dW = dw[:, i*a.lag:(i+1)*a.lag].sum(1)
    zz = z(prices, ts, i, a)
    canonical = (H*nn.functional.softplus(logit)-(target-zz.detach()*dW)*logit).mean()
    return canonical+(value.detach()+zz*dW-target).square().mean(), value


@torch.no_grad()
def predict(values, prices, ts, a, dates):
    return torch.stack([price(values[i], prices, ts, i, a)[0] for i in range(dates)]+[payoff(a.case, prices)], 1)


def run(a):
    root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
    if (root/'config.json').exists():
        raise FileExistsError('Use a fresh directory')
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    verify_source(Path(a.source_dir)); cls, _ = load_author(Path(a.source_dir))
    sampler = make_model(cls, a)
    ts = torch.linspace(0, a.T, a.steps+1, device=a.device)
    dates = a.steps//a.lag
    budget = a.total_training_updates-a.pretrain_updates
    stage_counts = [budget//dates+int(j < budget % dates) for j in range(dates)]
    write_json(root/'config.json', dict(vars(a), method='LSTM-BSDE', monitoring='128-date discrete', dates=dates,
        backward_updates=budget, updates_per_date_in_backward_order=stage_counts, explicit_history_state_inputs=False,
        mc_anchor=False, absorbing_target=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    shutil.copy2(__file__, root/'implementation.py')
    start = time.perf_counter()
    torch.manual_seed(a.seed)
    init = FineHistoryNet(a).to(a.device)
    opt = torch.optim.RMSprop(init.parameters(), lr=a.lr)
    for update in range(1, a.pretrain_updates+1):          # terminal-only initialisation (readout at T)
        prices, _ = simulate(sampler, ts, a, a.batch_size)
        g = payoff(a.case, prices)
        value, logit, H = price(init, prices, ts, dates, a)
        loss = (H*nn.functional.softplus(logit)-g*logit).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    values, zs = [None]*dates, [None]*dates
    with (root/'history.csv').open('w', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'update', 'lr', 'loss', 'prediction_mean', 'target_mean', 'seconds'])
        writer.writeheader()
        for i in range(dates-1, -1, -1):
            target = None if i == dates-1 else values[i+1]
            target_hash = None if target is None else tensor_hash(target)
            y = copy.deepcopy(init if target is None else target)
            if target is None:
                torch.manual_seed(a.seed+500)
                z = FineHistoryNet(a).to(a.device)
            else:
                z = copy.deepcopy(zs[i+1])
            for model in [y, z]:
                model.train(True)
                for p in model.parameters():
                    p.requires_grad_(True)
            opt = torch.optim.RMSprop(list(y.parameters())+list(z.parameters()), lr=a.lr)
            torch.manual_seed(a.seed+1000+i)
            disc = math.exp(-a.r*float(ts[(i+1)*a.lag]-ts[i*a.lag]))
            updates = stage_counts[dates-1-i]
            for update in range(1, updates+1):
                lr = a.lr*(a.lr_final_factor+(1-a.lr_final_factor)*.5*(1+math.cos(math.pi*(update-1)/max(updates-1, 1))))
                for group in opt.param_groups:
                    group['lr'] = lr
                prices, dw = simulate(sampler, ts, a, a.batch_size)
                with torch.no_grad():
                    nxt = disc*(payoff(a.case, prices) if target is None else price(target, prices, ts, i+1, a)[0])
                loss, now = loss_terms(y, z, prices, dw, nxt, ts, i, a)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                opt.zero_grad(); loss.backward(); opt.step()
                if update == 1 or update % 500 == 0 or update == updates:
                    row = dict(index=i, update=update, lr=lr, loss=float(loss), prediction_mean=float(now.mean()),
                               target_mean=float(nxt.mean()), seconds=time.perf_counter()-start)
                    writer.writerow(row)
                    write_json(root/'progress.json', dict(status='training', completed_dates=dates-1-i, dates=dates, **row))
                    print(row, flush=True)
            if target is not None:
                assert tensor_hash(target) == target_hash
            freeze(y); freeze(z)
            values[i], zs[i] = y, z
    torch.save(dict(values=[m.state_dict() for m in values], zs=[m.state_dict() for m in zs], args=vars(a)), root/'model.pt')
    write_json(root/'progress.json', dict(status='evaluating', training_seconds=time.perf_counter()-start))
    # hedge audit: payoff vs native stochastic integral (16 intervals)
    torch.manual_seed(992)
    disc_t = torch.exp(-a.r*ts[::a.lag])
    G, I = [], []
    with torch.no_grad():
        for _ in range(a.mc_paths//512):
            prices, dw = simulate(sampler, ts, a, 512)
            zz = torch.stack([zs[i](prices, ts, i, a) for i in range(dates)], 1)
            G.append((disc_t[-1]*payoff(a.case, prices)).double().cpu())
            I.append((zz*dw.reshape(len(dw), dates, a.lag).sum(-1)*disc_t[:-1]).sum(1).double().cpu())
    G, I = torch.cat(G), torch.cat(I)
    corr = float(((G-G.mean())*(I-I.mean())).mean()/(G.std(unbiased=False)*I.std(unbiased=False)))
    torch.manual_seed(994)
    prices = torch.cat([simulate(sampler, ts, a, 1024)[0].cpu() for _ in range(a.exact_paths//1024)])
    pred = torch.cat([predict(values, prices[j:j+1024].to(a.device), ts, a, dates).cpu() for j in range(0, len(prices), 1024)])
    result = score(a.case, a, prices, pred, list(range(0, a.steps+1, a.lag)))
    result.update(method='LSTM-BSDE', seed=a.seed, total_training_updates=a.total_training_updates,
                  payoff_integral_correlation=corr, variance_reduction=float(G.var()/(G-I).var()))
    write_json(root/'scores.json', result)
    write_json(root/'progress.json', dict(status='complete', seconds=time.perf_counter()-start,
                                          t0=result['out']['t0_prediction'], t0_exact=result['out']['t0_exact']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source-dir', default='vendor/deep_ppde')
    p.add_argument('--device', default='cuda')
    for flag in ['level-features', 'chrono']:
        p.add_argument('--'+flag, action=argparse.BooleanOptionalAction, default=True)
    for key, value in [('s0', 1.), ('r', .03), ('sigma', .4), ('T', 1.), ('lr', .0005), ('lr_final_factor', .05),
                       ('level_gamma', 50.), ('put_cap', 6.)]:
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=float, default=value)
    for key, value in [('steps', 128), ('lag', 8), ('depth', 3), ('hidden', 20), ('batch_size', 128), ('seed', 1),
                       ('level_count', 32), ('pretrain_updates', 3000), ('total_training_updates', 50000),
                       ('mc_paths', 16384), ('exact_paths', 8192)]:
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=int, default=value)
    a = p.parse_args()
    try:
        run(a)
    except Exception:
        import traceback
        root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
        write_json(root/'failure.json', dict(error=traceback.format_exc()))
        raise
