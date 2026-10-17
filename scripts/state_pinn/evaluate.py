"""Evaluate a saved sufficient-state barrier PINN away from its initial anchor."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path

import torch

from ..analytic_bs import barrier_price


@torch.no_grad()
def evaluate(run_dir):
    root = Path(run_dir)
    metrics = json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
    module = importlib.import_module(metrics['module'])
    saved = torch.load(root / 'model.pt', map_location='cpu')
    args = saved['args']
    dtype = next(iter(saved['state_dict'].values())).dtype
    market = module.Market(**saved['market'])
    contract = module.BarrierContract(**saved['contract'])
    model = module.MLP(hidden=args['hidden'], depth=args['depth']).to(dtype=dtype)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    def value(t, s):
        return module.model_value(model, market, contract, t, s)
    up = 'upout' in metrics['module']
    ratios = [.5, .65, .8, .9, .95, .98] if up else [1.02, 1.1, 1.25, 1.5, 2., 3.]
    spots = [contract.barrier * ratio for ratio in ratios]
    pairs = [(fraction * market.T, spot) for fraction in [0., .25, .5, .75, .9] for spot in spots
             if fraction != 0 or abs(spot-market.s0) > 1e-10]
    ts = torch.tensor([pair[0] for pair in pairs], dtype=dtype)
    ss = torch.tensor([pair[1] for pair in pairs], dtype=dtype)
    prediction = value(ts, ss).tolist()
    rows = []
    for (t, spot), pred in zip(pairs, prediction):
        analytic = barrier_price(spot, contract.strike, contract.barrier, market.T-t,
                                 market.r, market.sigma, contract.option_type,
                                 'up-and-out' if up else 'down-and-out')
        rows.append(dict(t=t, spot=spot, prediction=pred, analytic=analytic,
                         absolute_error=abs(pred-analytic)))
    with (root / 'surface_validation.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = [row['prediction']-row['analytic'] for row in rows]
    truth = [row['analytic'] for row in rows]
    boundary_t = torch.tensor([fraction * market.T for fraction in [0., .25, .5, .75, .9, .99]], dtype=dtype)
    boundary = value(boundary_t, torch.full_like(boundary_t,contract.barrier))
    terminal = value(torch.full_like(ss,market.T),ss)
    terminal_error = terminal-module.vanilla_payoff(ss,contract)
    result = dict(points=len(rows), initial_price=metrics['prediction'],
                  initial_analytic=metrics['continuous_analytic'],
                  initial_relative_error=metrics['relative_error'],
                  surface_rmse=math.sqrt(sum(e*e for e in errors)/len(errors)),
                  surface_relative_l2=math.sqrt(sum(e*e for e in errors)/max(sum(v*v for v in truth),1e-30)),
                  surface_max_absolute_error=max(abs(e) for e in errors),
                  boundary_rmse=boundary.square().mean().sqrt().item(),
                  boundary_max_absolute_error=boundary.abs().max().item(),
                  terminal_rmse=terminal_error.square().mean().sqrt().item(),
                  mc_anchor_weight=args['mc_anchor_weight'],
                  note='Deterministic holdout grid excludes the initial anchor. Relative L2 is a price-surface diagnostic, not a convergence guarantee.')
    (root / 'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(evaluate(args.run_dir),indent=2))
