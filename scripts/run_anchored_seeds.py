"""Additional Adam training seeds for the anchored-transition sufficient-state PINNs.

Repeats every trained variant of run_anchored_transition.py with another training
seed, without touching the seed-1 experiment:
  soft_mc (barriers anchored to the identical saved MC value; lookbacks unanchored),
  heat_before, hard_no_mc, hard_mc (barriers), then heat_after via the ORIGINAL
  refine_worker (unit-weight L-BFGS, original fixed collocation seeds).
The three evaluation stages and the evaluator are the original ones.

Differences from seed 1 are only the Adam seed. In the seed-1 run, heat_before and
the lookback soft models were reused from earlier runs trained through
ten_contract_models.loss; here they are trained through anchored_state_models.loss,
which was checked to give bitwise-identical losses and gradients for those modes.
The MC anchor file is copied from seed 1 (it is a problem input, not training noise).
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from . import run_anchored_transition as rat
from .state_pinn import anchored_state_models as api
from .state_pinn.evaluate_ten_contracts import evaluate

SEED1 = Path('outputs/anchored_transition_15000')


def train_worker(args):
    """run_anchored_transition.train_worker with the Adam seed as a parameter (otherwise identical)."""
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(args.seed)
    root = Path(args.output_dir); out = root/'training'/args.case/args.variant; out.mkdir(parents=True, exist_ok=True)
    mode = 'soft' if args.variant == 'soft_mc' else 'heat' if args.variant == 'heat_before' else 'hard'
    anchored = args.variant in ('soft_mc', 'hard_mc') and not args.case.startswith('lookback')
    anchors = json.loads((root/'mc_anchors.json').read_text())['anchors']
    target = anchors[args.case]['value'] if anchored else None; weight = 100. if anchored else 0.
    model = api.make_model(args.case, mode); opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 1000, .9)
    config = dict(case=args.case, variant=args.variant, mode=mode, epochs=args.epochs, batch_size=128,
        hidden=64, depth=3, seed=args.seed, precision='float64', pde_weight=1., long_tau_weight=1.,
        mc_anchor_weight=weight, mc_anchor_value=target, mc_anchor_source=str(root/'mc_anchors.json') if anchored else None,
        structure_version='learnable_sqrt_corner_and_tanh_gate_v2', lr=5e-4, lr_step=1000, lr_gamma=.9, grad_clip=10., status='running')
    rat.write(out/'run_config.json', config); started = time.perf_counter()
    fields = ['epoch', 'loss', 'pde', 'terminal', 'boundary', 'nonnegative', 'anchor', 'anchor_weight', 'gradient_norm', 'elapsed_seconds']
    with (out/'history.csv').open('w', newline='', encoding='utf-8', buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for epoch in range(1, args.epochs+1):
            opt.zero_grad(set_to_none=True); loss, parts = api.loss(model, args.case, mode, 128, target, weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
            opt.step(); schedule.step()
            row = dict(epoch=epoch, loss=loss.item(), **parts, gradient_norm=norm, elapsed_seconds=time.perf_counter()-started)
            writer.writerow(row)
            if epoch == 1 or epoch % 100 == 0:
                rat.write(out/'progress.json', dict(status='training', **row, total_epochs=args.epochs))
            if epoch == 1 or epoch % 1000 == 0:
                print(args.case, args.variant, epoch, loss.item(), flush=True)
    config['status'] = 'complete'; rat.save_model(root, args.case, args.variant, model, config, time.perf_counter()-started)
    rat.write(out/'run_config.json', config); rat.write(out/'progress.json', dict(status='complete', epoch=args.epochs, total_epochs=args.epochs))


def run(args):
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    if not (root/'mc_anchors.json').exists():
        shutil.copy2(SEED1/'mc_anchors.json', root/'mc_anchors.json')
    rat.write(root/'experiment.json', dict(status='training', seed=args.seed, epochs=args.epochs, paths=args.paths,
        collocation=args.collocation, outer_steps=args.outer_steps, seed1_root=str(SEED1),
        mc_anchor_sha256=hashlib.sha256((root/'mc_anchors.json').read_bytes()).hexdigest(),
        note=__doc__))
    def launch(item, refine=False):
        case, variant = item
        out = root/'training'/case/variant
        if (out/'model.pt').exists():
            return
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', '-m', 'scripts.run_anchored_seeds', '--worker', 'refine' if refine else 'train',
               '--case', case, '--variant', variant, '--seed', str(args.seed), '--output-dir', str(root),
               '--epochs', str(args.epochs), '--collocation', str(args.collocation), '--outer-steps', str(args.outer_steps)]
        with (out/'train.log').open('w', encoding='utf-8') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        jobs = []
        for case in api.CASES:
            jobs += [(case, 'soft_mc'), (case, 'heat_before'), (case, 'hard_no_mc')]
            if not case.startswith('lookback'):
                jobs.append((case, 'hard_mc'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch, jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda c: launch((c, 'heat_after'), True), api.CASES))
        stages = {'hard_without_mc': dict(soft='soft_mc', hard='hard_no_mc', heat='heat_before'),
                  'hard_with_mc': dict(soft='soft_mc', hard='hard_mc', heat='heat_before'),
                  'heat_refined': dict(soft='soft_mc', hard='hard_mc', heat='heat_after')}
        def stage_eval(item):
            stage, mapping = item
            folder = root/'stages'/stage; folder.mkdir(parents=True, exist_ok=True)
            if (folder/'evaluation'/'evaluation_manifest.json').exists():
                return
            for case in api.CASES:
                for mode, variant in mapping.items():
                    if case.startswith('lookback') and variant == 'hard_mc':
                        variant = 'hard_no_mc'
                    rat.copy_run(root/'training'/case/variant, folder/case/mode)
            evaluate(folder, args.paths, model_api=api)
        rat.write(root/'experiment.json', dict(json.loads((root/'experiment.json').read_text()), status='evaluating'))
        for item in stages.items():
            stage_eval(item)
        rat.write(root/'experiment.json', dict(json.loads((root/'experiment.json').read_text()), status='complete'))
    except BaseException:
        rat.write(root/'experiment.json', dict(json.loads((root/'experiment.json').read_text()), status='failed', error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=15000); p.add_argument('--paths', type=int, default=2048)
    p.add_argument('--workers', type=int, default=22); p.add_argument('--collocation', type=int, default=8192)
    p.add_argument('--outer-steps', type=int, default=20)
    p.add_argument('--worker', choices=['train', 'refine']); p.add_argument('--case', choices=api.CASES)
    p.add_argument('--variant')
    args = p.parse_args()
    if args.seed == 1:
        p.error('Seed 1 already exists in outputs/anchored_transition_15000')
    if args.worker == 'train':
        train_worker(args)
    elif args.worker == 'refine':
        rat.refine_worker(args)
    else:
        run(args)
