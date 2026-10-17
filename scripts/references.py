"""Independent, batched MC baselines; invoked using the training interpreter."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .analytic_bs import barrier_price, floating_lookback_price


@torch.no_grad()
def reference(config, product):
    s0, r, sigma, maturity = (config[key] for key in ('s0', 'r', 'sigma', 'T'))
    steps, count = config['steps'], config['mc_paths']
    barrier = config['up_barrier'] if product == 'barrier_upout' else config['down_barrier']
    option = config['option_type']
    if product == 'lookback':
        analytic = floating_lookback_price(s0, maturity, r, sigma, option, running_min=s0, running_max=s0)
    else:
        analytic = barrier_price(s0, config['strike'], barrier, maturity, r, sigma, option,
                                 'up-and-out' if product == 'barrier_upout' else 'down-and-out')
    torch.manual_seed(config['reference_seed'])
    totals = {key: [0.0, 0.0] for key in ['discrete', 'bridge'] if key != 'bridge' or product != 'lookback'}
    dt = maturity / steps
    discount = math.exp(-r * maturity)
    for offset in range(0, count, 2048):
        size = min(2048, count - offset)
        increments = (r - sigma**2 / 2) * dt + sigma * math.sqrt(dt) * torch.randn(size, steps, dtype=torch.float64)
        log_path = math.log(s0) + torch.cat([torch.zeros(size, 1, dtype=torch.float64), increments.cumsum(1)], dim=1)
        paths = log_path.exp()
        terminal = paths[:, -1]
        if product == 'lookback':
            payoff = terminal - paths.min(1).values if option == 'call' else paths.max(1).values - terminal
            samples = {'discrete': discount * payoff}
        else:
            payoff = (terminal - config['strike']).clamp_min(0) if option == 'call' else (config['strike'] - terminal).clamp_min(0)
            gap = math.log(barrier) - log_path if product == 'barrier_upout' else log_path - math.log(barrier)
            alive = (gap > 0).all(1)
            exponent = (-2 * gap[:, :-1] * gap[:, 1:] / (sigma**2 * dt)).clamp_max(0)
            interval = torch.where((gap[:, :-1] > 0) & (gap[:, 1:] > 0), -torch.expm1(exponent), torch.zeros_like(exponent))
            samples = {'discrete': discount * alive * payoff, 'bridge': discount * interval.prod(1) * payoff}
        for key, values in samples.items():
            totals[key][0] += values.sum().item()
            totals[key][1] += values.square().sum().item()
    result = {'continuous_analytic': analytic, 'paths': count, 'steps': steps,
              'seed': config['reference_seed'], 'monitoring': 'grid; bridge additionally available for barriers'}
    for key, (total, squared) in totals.items():
        result[key + '_mc'] = total / count
        variance = max(0.0, (squared - total * total / count) / (count - 1))
        result[key + '_mc_se'] = math.sqrt(variance / count)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    path = Path(args.config)
    config = json.loads(path.read_text(encoding='utf-8'))
    results = {product: reference(config, product) for product in config['products']}
    (path.parent / 'references.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
