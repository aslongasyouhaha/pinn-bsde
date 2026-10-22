"""Ten-contract table for the pure-history LSTM-PINN / LSTM-BSDE runs (50000 updates, seeds 1-3).

Reads outputs/lstm_ten_50000/{pinn,bsde}/<case>/seed<k>/scores.json. Knock-ins are the
exact discrete in-out parity of the trained knock-outs. Writes report.md and summary.json.
"""
import json
from pathlib import Path

ROOT = Path('outputs/lstm_ten_50000')
TIMES = [.25, .5, .75, .9375]
ROWS = [('lookback_call', 'out', 'Lookback call'), ('lookback_put', 'out', 'Lookback put'),
        ('up_call', 'out', 'Up-and-out call'), ('up_put', 'out', 'Up-and-out put'),
        ('down_call', 'out', 'Down-and-out call'), ('down_put', 'out', 'Down-and-out put'),
        ('up_call', 'in', 'Up-and-in call (parity)'), ('up_put', 'in', 'Up-and-in put (parity)'),
        ('down_call', 'in', 'Down-and-in call (parity)'), ('down_put', 'in', 'Down-and-in put (parity)')]


def run_dir(method, case, seed):
    # Uniform PINN rule: PDE residual divided by the lift WITHOUT the put cap. It only changes the
    # lookback put (the other lifts have no cap), whose rerun lives in pinn_put_uncapped/.
    if method == 'pinn' and case == 'lookback_put':
        return ROOT/'pinn_put_uncapped'/case/f'seed{seed}'
    return ROOT/method/case/f'seed{seed}'


def load(method, case, seed, name='scores.json'):
    p = run_dir(method, case, seed)/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def metrics(block):
    by = {round(r['time'], 4): r['relative_l2'] for r in block['by_date']}
    return dict(t0_error=block['t0_prediction']/block['t0_exact']-1, t0_exact=block['t0_exact'],
                l2=[by.get(t) for t in TIMES])


def fmt(values, signed=False):
    values = [v for v in values if v is not None]
    if not values:
        return '—'
    m = sum(values)/len(values)
    f = (lambda v: f'{100*v:+.1f}%') if signed else (lambda v: f'{100*v:.1f}%')
    return f'{f(m)} [{f(min(values))}, {f(max(values))}]' if len(values) > 1 else f(m)


def main():
    summary, lines = {}, ['# History-based LSTM-PINN / LSTM-BSDE: ten contracts, 50000 updates, seeds 1-3', '',
        'Single asset: S0=K=1, r=0.03, sigma=0.4, T=1; barriers 1.2 (up) and 0.8 (down), with 128 monitoring steps. '
        'Inputs contain observed prices and times, without explicit extrema or hit flags. Each run has 3000 terminal '
        'updates and 47000 main updates; the main learning rate decays to 0.05 of its initial value. '
        'Knock-ins use in-out parity rather than separate training. Evaluation uses numerical quadrature on '
        '8192 held-out histories (seed 994). Entries are mean [minimum, maximum], not confidence intervals. '
        'The reported PINNs ran on GPU and BSDEs on CPU.', '',
        '| Contract | Reference initial | PINN initial bias | PINN t=0.5 L2 | PINN four-date average L2 | BSDE initial bias | BSDE t=0.5 L2 | BSDE four-date average L2 | Completed seeds (PINN/BSDE) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case, flavor, label in ROWS:
        cells, t0_exact = {}, None
        for method in ['pinn', 'bsde']:
            ms = []
            for seed in [1, 2, 3]:
                s = load(method, case, seed)
                if s and flavor in s:
                    ms.append(metrics(s[flavor])); t0_exact = ms[-1]['t0_exact']
            cells[method] = ms
        summary[f'{case}/{flavor}'] = {m: v for m, v in cells.items()}
        row = [label, '—' if t0_exact is None else f'{t0_exact:.6f}']
        for method in ['pinn', 'bsde']:
            ms = cells[method]
            row += [fmt([m['t0_error'] for m in ms], True), fmt([m['l2'][1] for m in ms]),
                    fmt([sum(m['l2'])/4 for m in ms if None not in m['l2']])]
        row.append(f"{len(cells['pinn'])}/{len(cells['bsde'])}")
        lines.append('| '+' | '.join(row)+' |')
    corr = []
    for case in ['lookback_call', 'lookback_put', 'up_call', 'up_put', 'down_call', 'down_put']:
        vals = [load('bsde', case, k)['payoff_integral_correlation'] for k in [1, 2, 3] if load('bsde', case, k)]
        if vals:
            corr.append(f'{case} {sum(vals)/len(vals):.3f}')
    lines += ['', 'Native BSDE Z correlation with payoff (16-interval stochastic integral, seed mean): '+('; '.join(corr) if corr else '—'), '']
    old = [json.loads(p.read_text(encoding='utf-8'))['out'] for p in sorted((ROOT/'pinn'/'lookback_put').glob('seed*/scores.json'))]
    if old:
        lines += ['Note: PDE normalization removes the lookback-put envelope factor. Dividing by 6S instead of S '
                  'reduces the squared PDE term by 36. The earlier initial bias was ' + fmt([o['t0_prediction']/o['t0_exact']-1 for o in old], True)
                  + ', with t=0.5 L2 ' + fmt([[r['relative_l2'] for r in o['by_date'] if abs(r['time']-.5) < 1e-9][0] for o in old])
                  + '. These earlier results remain under pinn/lookback_put/. Other contracts are unaffected.', '']
    lines += ['## PINN initial-bias decomposition: 16384 paths, seed 995, training-seed means', '',
              'Identity: discounted expected payoff minus V0 = terminal defect + monitoring jumps + interval drift. '
              'Positive values lower the initial prediction; all three are zero for an exact solution.', '',
              '| Contract | Terminal defect | Net jump | Jump components | Interval drift |', '|---|---:|---:|---|---:|']
    for case in ['lookback_call', 'lookback_put', 'up_call', 'up_put', 'down_call', 'down_put']:
        ds = [d for d in (load('pinn', case, k, 'drift_decomposition.json') for k in [1, 2, 3]) if d]
        if not ds:
            continue
        avg = lambda key: sum(d[key]['mean'] for d in ds)/len(ds)
        parts = ', '.join(f"{k[5:]} {avg(k):+.5f}" for k in ds[0] if k.startswith('jump_'))
        lines.append(f"| {case} | {avg('terminal'):+.5f} | {avg('jump'):+.5f} | {parts} | {avg('inside'):+.5f} |")
    lines.append('')
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
