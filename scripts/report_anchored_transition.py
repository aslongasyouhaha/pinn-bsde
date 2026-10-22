"""Current anchored baseline + hard control + unit-weight H refinement tables."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .state_pinn.evaluate_ten_contracts import csv_write
from .state_pinn import anchored_state_models as api

LABELS = {
    'lookback_call': 'Lookback call', 'lookback_put': 'Lookback put',
    'up_call_out': 'Up-and-out call', 'up_call_in': 'Up-and-in call',
    'up_put_out': 'Up-and-out put', 'up_put_in': 'Up-and-in put',
    'down_call_out': 'Down-and-out call', 'down_call_in': 'Down-and-in call',
    'down_put_out': 'Down-and-out put', 'down_put_in': 'Down-and-in put',
}
VARIANTS=('soft_mc','hard_no_mc','hard_mc','heat_before','heat_after')


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def report(root,update_docs=False):
    root=Path(root);out=root/'comparison';out.mkdir(exist_ok=True)
    # Generated reports stay in outputs; the research summary is maintained separately.
    stages={name:json.loads((root/'stages'/name/'evaluation/comparison.json').read_text())
            for name in ('hard_without_mc','hard_with_mc','heat_refined')}
    selections=(('hard_without_mc','soft','soft_mc'),('hard_without_mc','hard','hard_no_mc'),
        ('hard_with_mc','hard','hard_mc'),('hard_without_mc','heat','heat_before'),('heat_refined','heat','heat_after'))
    combined=[];extra={name:[] for name in ('metrics','by_time','conditions','pde_residuals','timings')}
    for stage,mode,variant in selections:
        for r in stages[stage]:
            if r['mode']!=mode or (variant=='hard_mc' and r['contract'].startswith('lookback')):continue
            combined.append(dict(r,variant=variant,evaluation_stage=stage))
        for filename in extra:
            for r in read(root/'stages'/stage/'evaluation'/(filename+'.csv')):
                if r['mode']!=mode:continue
                case=r.get('contract',r.get('case',''))
                if variant=='hard_mc' and case.startswith('lookback'):continue
                extra[filename].append(dict(r,variant=variant,evaluation_stage=stage))
    assert len(combined)==48 and len({(r['contract'],r['variant']) for r in combined})==48
    csv_write(out/'results.csv',combined)
    for name,rows in extra.items():csv_write(out/(name+'.csv'),rows)
    (out/'results.json').write_text(json.dumps(combined,indent=2,allow_nan=False),encoding='utf-8')
    anchors=json.loads((root/'mc_anchors.json').read_text())['anchors'];anchor_rows=[];h_rows=[]
    for case in api.CASES:
        if not case.startswith('lookback'):
            for variant in ('soft_mc','hard_mc'):
                a=json.loads((root/'training'/case/variant/'run_config.json').read_text())
                assert a['mc_anchor_weight']==100. and a['mc_anchor_value']==anchors[case]['value']
                anchor_rows.append(dict(case=case,variant=variant,mc_value=anchors[case]['value'],
                    standard_error=anchors[case]['standard_error'],paths=anchors[case]['paths'],weight=100.))
        for variant in ('heat_before','heat_after'):
            folder=root/'training'/case/variant;saved=torch.load(folder/'model.pt',map_location='cpu')
            a=saved['args'];assert a.get('pde_weight',1.)==1. and a.get('long_tau_weight',1.)==1. and a.get('mc_anchor_weight',0.)==0.
            rows=read(folder/'history.csv')
            discrepancy=max(abs(float(r['loss'])-float(r['pde'])-.01*float(r.get('nonnegative',0.))) for r in rows)
            assert discrepancy<1e-12
            h_rows.append(dict(case=case,variant=variant,pde_weight=1.,long_tau_weight=1.,mc_anchor_weight=0.,
                objective_check_error=discrepancy,checkpoint_sha256=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),
                lbfgs_gradient_evaluations=a.get('gradient_evaluations',0),lbfgs_collocation=a.get('collocation',0)))
    csv_write(out/'anchor_audit.csv',anchor_rows);csv_write(out/'h_audit.csv',h_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,case in zip(axes.ravel(),api.CASES):
        for variant in VARIANTS[:-1]:
            if case.startswith('lookback') and variant=='hard_mc':continue
            rows=read(root/'training'/case/variant/'history.csv');x=np.array([float(r['pde']) for r in rows]);w=min(200,len(x))
            ax.semilogy(np.arange(w,len(x)+1),np.convolve(x,np.ones(w)/w,'valid'),label=variant)
        ax.set(title=case,xlabel='Adam updates',ylabel='Raw PDE MSE');ax.legend()
    fig.savefig(out/'adam_pde_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,case in zip(axes.ravel(),api.CASES):
        rows=read(root/'training'/case/'heat_after/history.csv')
        ax.semilogy([float(r['gradient_evaluations']) for r in rows],[float(r['loss']) for r in rows])
        ax.set(title=case,xlabel='LBFGS full-gradient evaluations',ylabel='Fixed-collocation unit-weight objective')
    fig.savefig(out/'h_lbfgs_curves.png',dpi=160);plt.close(fig)
    text = '# Anchored soft constraints, hard constraints, and H refinement\n\n'
    text += ('Barrier soft constraints use MC anchors. Hard constraints are compared with and without '
             'the identical anchor. Lookbacks and H models use no MC anchors. Knock-ins and their '
             'Greeks follow from parity. Analytical values are used for evaluation only.\n\n')
    text += ('Hard lookbacks use a learnable square-root-time correction; hard barriers use a tanh '
             'gate with a positive learned scale. These structural priors do not guarantee successful fitting. '
             'Anchors use continuous-monitoring Brownian-bridge survival estimates; their standard errors '
             'and sample counts are saved in mc_anchors.json. Soft and hard anchors share coefficient 100. '
             'L-BFGS is an additional optimization phase, not an equal-compute comparison.\n')
    contracts = list(dict.fromkeys(row['contract'] for row in combined))

    def table(key, title, price=False):
        text = ('\n\n## ' + title + '\n\n'
                '| Contract | Analytical reference | Soft + barrier MC | Hard without MC | Hard + MC | H before | H + L-BFGS |\n'
                '|---|---:|---:|---:|---:|---:|---:|\n')
        for contract in contracts:
            subset = {row['variant']: row for row in combined if row['contract'] == contract}
            cells = []
            for variant in VARIANTS:
                row = subset.get(variant)
                cells.append('N/A' if row is None else
                             f"{row['prediction']:.8f} ({100*row['relative_error']:.4f}%)" if price else
                             f"{100*row[key]:.4f}%")
            text += f"| {LABELS[contract]} | {subset['soft_mc']['analytic']:.8f} | " + ' | '.join(cells) + ' |\n'
        return text

    text += table('relative_error', 'Initial prices (relative error in parentheses)', True)
    for key, title in [
        ('surface_price_relative_l2', 'Interior-state price relative L2'),
        ('path_price_equal_date_rms', 'Path prices: RMS of datewise relative L2'),
        ('path_delta_equal_date_rms', 'Path Delta: RMS of datewise relative L2'),
        ('path_gamma_equal_date_rms', 'Path Gamma: RMS of datewise relative L2'),
    ]:
        text += table(key, title)
    text += ('\nFull regional, datewise, terminal/boundary, independent PDE residual, and timing '
             'metrics are saved as CSV files. An initial anchor does not guarantee intermediate '
             'accuracy. A finite evaluation grid and one training seed do not establish uniform '
             'accuracy or universal method superiority.\n')
    (out/'report.md').write_text(text, encoding='utf-8')
    (out/'report_manifest.json').write_text(json.dumps(dict(status='complete', result_cells=48, anchor_pairs=4,
        h_before_after_pairs=6, lookback_mc_not_applicable=True, pde_weight=1.), indent=2), encoding='utf-8')
