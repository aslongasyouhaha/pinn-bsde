"""Three-seed summary of the anchored-transition sufficient-state PINNs (non-LSTM methods).

Seed 1: outputs/anchored_transition_15000; seeds 2-3: outputs/anchored_transition_15000_seed{2,3}
(scripts/run_anchored_seeds.py). Same stages, evaluator and continuous-monitoring analytic
references. Writes outputs/anchored_transition_seeds/{report.md,summary.json}.
"""
import csv
import json
from pathlib import Path

ROOTS = {1: Path('outputs/anchored_transition_15000'), 2: Path('outputs/anchored_transition_15000_seed2'),
         3: Path('outputs/anchored_transition_15000_seed3')}
OUT = Path('outputs/anchored_transition_seeds')
STAGES = [('hard_without_mc', 'Stage 1: soft + barrier MC / hard without MC / H before refinement'),
          ('hard_with_mc', 'Stage 2: soft + barrier MC / hard + barrier MC / H before refinement'),
          ('heat_refined', 'Stage 3: soft + barrier MC / hard + barrier MC / H after L-BFGS')]
CONTRACTS = ['lookback_call', 'lookback_put', 'up_call_out', 'up_put_out', 'down_call_out', 'down_put_out',
             'up_call_in', 'up_put_in', 'down_call_in', 'down_put_in']
NAMES = dict(lookback_call='Lookback call', lookback_put='Lookback put', up_call_out='Up-and-out call', up_put_out='Up-and-out put',
             down_call_out='Down-and-out call', down_put_out='Down-and-out put', up_call_in='Up-and-in call (parity)', up_put_in='Up-and-in put (parity)',
             down_call_in='Down-and-in call (parity)', down_put_in='Down-and-in put (parity)')
MODES = [('soft', 'Soft'), ('hard', 'Hard'), ('heat', 'H')]


def load(seed, stage):
    path = ROOTS[seed]/'stages'/stage/'evaluation'/'comparison.csv'
    if not path.exists():
        return None
    with path.open(encoding='utf-8-sig') as f:
        return {(r['contract'], r['mode']): r for r in csv.DictReader(f)}


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return '—'
    m = sum(values)/len(values)
    return f'{100*m:.2f}% [{100*min(values):.2f}, {100*max(values):.2f}]' if len(values) > 1 else f'{100*m:.2f}%'


def main():
    summary, lines = {}, ['# Sufficient-state PINNs: three training seeds', '',
        'Ten contracts, including parity-derived knock-ins; continuous-monitoring analytical references. '
        'Seed 1 uses anchored_transition_15000; seeds 2 and 3 follow the same protocol through '
        'scripts/run_anchored_seeds.py. MC anchors, L-BFGS collocation seeds, and evaluators are shared. '
        'Entries report mean [minimum, maximum] across available training seeds; ranges are not confidence intervals.', '']
    for stage, title in STAGES:
        tables = {k: load(k, stage) for k in ROOTS}
        seeds = [k for k, t in tables.items() if t]
        lines += [f'## {title}', '', f'Completed seeds: {seeds}', '',
                  '| Contract | ' + ' | '.join(f'{n} initial error | {n} path-price L2' for _, n in MODES) + ' |',
                  '|---|' + '---:|'*(2*len(MODES))]
        for contract in CONTRACTS:
            row = [NAMES[contract]]
            for mode, _ in MODES:
                rs = [tables[k].get((contract, mode)) for k in seeds]
                rs = [r for r in rs if r]
                t0 = [float(r['relative_error']) for r in rs]
                path = [float(r['path_price_relative_l2']) for r in rs]
                summary[f'{stage}/{contract}/{mode}'] = dict(seeds=seeds, relative_error=t0, path_price_relative_l2=path,
                    path_delta_relative_l2=[float(r['path_delta_relative_l2']) for r in rs])
                row += [fmt(t0), fmt(path)]
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
