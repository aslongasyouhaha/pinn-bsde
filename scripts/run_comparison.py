"""Train the three solver families with a shared market and save a comparison."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('state_pinn', 'lstm_pinn', 'signature_bsde')
PRODUCTS = ('lookback', 'barrier_upout', 'barrier_downout')
SUFFIX = dict(zip(METHODS, ('pinn', 'lstm_pinn', 'signature_bsde')))


def run_command(command, log_path):
    with log_path.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}). See {log_path}\n{log_path.read_text(encoding='utf-8', errors='replace')[-4000:]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--products', nargs='+', choices=PRODUCTS, default=['barrier_upout'])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--lag', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mc-paths', type=int, default=50000)
    parser.add_argument('--eval-paths', type=int, default=512)
    parser.add_argument('--reference-seed', type=int, default=1729)
    parser.add_argument('--option-type', choices=['call', 'put'], default='call')
    parser.add_argument('--s0', type=float, default=1.0)
    parser.add_argument('--r', type=float, default=.03)
    parser.add_argument('--sigma', type=float, default=.4)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--strike', type=float, default=1.0)
    parser.add_argument('--up-barrier', type=float, default=1.2)
    parser.add_argument('--down-barrier', type=float, default=.8)
    parser.add_argument('--mc-anchor-weight', type=float, default=None,
                        help='Override anchor weight; default: state PINN=100, LSTM PINN=0')
    parser.add_argument('--terminal-sampling', choices=['legacy', 'full'], default='legacy',
                        help='State up-and-out PINN only; keep legacy as the default control')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--python', default=sys.executable, help='Interpreter with torch and signatory installed')
    parser.add_argument('--output-dir', default='outputs/comparison')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.steps, args.lag, args.eval_paths, args.threads) <= 0 or args.mc_paths < 2:
        parser.error('Counts must be positive; --mc-paths must be at least 2.')
    if args.steps < 3 or args.steps % args.lag:
        parser.error('--steps must be >=3 and divisible by --lag.')
    if min(args.s0, args.sigma, args.T, args.strike, args.up_barrier, args.down_barrier) <= 0:
        parser.error('Market levels, sigma and T must be positive.')
    if 'lookback' in args.products and abs(args.r) < 1e-12:
        parser.error('The current lookback analytical formula requires nonzero r.')
    if 'barrier_upout' in args.products and args.s0 >= args.up_barrier:
        parser.error('Use an initially alive up-and-out contract: s0 < up-barrier.')
    if 'barrier_downout' in args.products and args.s0 <= args.down_barrier:
        parser.error('Use an initially alive down-and-out contract: s0 > down-barrier.')
    if len(set(args.methods)) != len(args.methods) or len(set(args.products)) != len(args.products) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Methods, products and seeds must not contain duplicates.')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        parser.error(f'{output} already contains a run. Choose a fresh --output-dir.')
    config = vars(args).copy()
    config['python'] = str(Path(args.python).resolve())
    config['status'] = 'running'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    # Isolate workers so RNG state and module globals cannot leak between methods.
    prefix = [config['python'], '-u', '-c',
              f"import runpy,sys,torch; torch.set_num_threads({args.threads}); module=sys.argv.pop(1); runpy.run_module(module,run_name='__main__')"]
    rows = []
    try:
        run_command(prefix + ['scripts.references', str(config_path)], output / 'references.log')
        references = json.loads((output / 'references.json').read_text(encoding='utf-8'))
        for product in args.products:
            for method in args.methods:
                for seed in args.seeds:
                    run_dir = output / product / method / f'seed_{seed}'
                    run_dir.mkdir(parents=True, exist_ok=True)
                    module = f'scripts.{method}.bs_{product}_{SUFFIX[method]}'
                    native = dict(epochs=args.epochs, batch_size=args.batch_size, seed=seed, device=args.device,
                                  s0=args.s0, r=args.r, sigma=args.sigma, T=args.T, lr=args.lr,
                                  option_type=args.option_type, output_dir=str(run_dir), print_every=max(1, args.epochs // 10))
                    if method == 'state_pinn' and product == 'barrier_upout':
                        native['terminal_sampling'] = args.terminal_sampling
                    if product != 'lookback':
                        native.update(strike=args.strike, barrier=args.up_barrier if product == 'barrier_upout' else args.down_barrier)
                    if method != 'state_pinn':
                        native.update(steps=args.steps, eval_paths=args.eval_paths, mc_paths=args.mc_paths)
                    elif product != 'lookback':
                        native.update(mc_steps=args.steps, mc_paths=args.mc_paths)
                    if method == 'signature_bsde':
                        native.update(lag=args.lag)
                    elif product != 'lookback':
                        anchor_weight = args.mc_anchor_weight if args.mc_anchor_weight is not None else (100.0 if method == 'state_pinn' else 0.0)
                        native.update(mc_anchor_weight=anchor_weight, mc_anchor_paths=args.mc_paths)
                        if method == 'state_pinn':
                            native.update(mc_anchor_steps=args.steps)
                    command = prefix + [module]
                    for key, value in native.items():
                        command += ['--' + key.replace('_', '-'), str(value)]
                    (run_dir / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
                    print(f'Training {product} / {method} / seed {seed}', flush=True)
                    run_command(command, run_dir / 'train.log')
                    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
                    row = {key: metrics[key] for key in ['prediction', 'continuous_analytic', 'absolute_error', 'relative_error', 'elapsed_seconds']}
                    row.update(product=product, method=method, seed=seed,
                               mc_anchor_weight=native.get('mc_anchor_weight', 0.0),
                               training_monitoring={
                                   'state_pinn': 'continuous PDE',
                                   'lstm_pinn': 'experimental path residual; grid terminal states',
                                   'signature_bsde': 'discrete grid',
                               }[method])
                    row.update({key: value for key, value in references[product].items() if key.endswith(('_mc', '_mc_se'))})
                    row['run_dir'] = run_dir.relative_to(output).as_posix()
                    rows.append(row)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        config['status'] = 'complete'
    except Exception:
        config['status'] = 'failed'
        raise
    finally:
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Comparison saved to {output / "comparison.csv"}')


if __name__ == '__main__':
    main()
