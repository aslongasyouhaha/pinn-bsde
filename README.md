# Single-Asset Path-Dependent Option Pricing

PINN and BSDE implementations for ten Black-Scholes option types: floating-strike lookback calls and puts, and up/down knock-in/knock-out calls and puts. Knock-ins are obtained through exact in-out parity.

Read [summary.md](summary.md) for the mathematical derivations, investigations, result tables, three-seed ranges, and conclusions. This README covers installation, code organization, execution, and output locations.

The repository includes necessary source code, the original paper, its accompanying library and license, and these two Markdown documents. Training outputs, weights, notebooks, course assignments, and local environments are not distributed. Reported results are recorded in the summary; reproducing the underlying artifacts requires running the relevant experiments.

## Methods

| Method | Historical information | Main output |
|---|---|---|
| Sufficient-state PINN | Explicit extremum or survival state; direct state-space sampling | Differentiable prices and Greeks |
| Raw-history LSTM-PINN | Chronological observed prices and times | History-conditioned differentiable prices |
| Signature + LSTM-BSDE | Truncated signatures of successive path segments | Prices and Brownian-integral coefficients |
| Raw-history LSTM-BSDE | Fine-grid observations and learned price-level features | Prices and an adapted hedge-coefficient sequence |

Sufficient-state experiments use continuous-monitoring analytical references. History-based experiments use 128 discrete monitoring steps and matched numerical references. The original Signature barrier configurations did not meet pricing requirements; improved raw-history configurations are a separate method configuration, not a successful reproduction of the original method. See the summary for the evidence and its scope.

## Installation

Use Python 3.9. Signatory is a compiled extension tied to a particular PyTorch build. The tested CPU environment uses PyTorch 1.9.0 and Signatory 1.2.6.1.9.0.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, signatory; print(torch.__version__, signatory.__version__)"
```

For GPU training, use a matching PyTorch 1.9.0 CUDA build and compatible Signatory wheel rather than the CPU torch pin. The experiments used a separate Linux environment with PyTorch 1.9.0+cu111. CUDA must be available before selecting `--device cuda`. The sufficient-state suite runs on CPU; the history-based programs accept a device option.

## Project Structure

```text
README.md                         Installation and execution
summary.md                        Consolidated research summary
requirements.txt                  Runtime dependencies
docs/
  2011.10630_...pdf                Original paper
  Deep-PPDE-LICENSE.txt            Upstream license
scripts/
  analytic_bs.py                  Analytical prices
  binomial_bs.py                  CRR lattice reference
  barrier_heat_kernel.py          Auxiliary and exact barrier kernels
  state_pinn/                    Sufficient-state models and evaluators
  run_anchored_transition.py      Soft/hard/H training and refinement
  run_anchored_seeds.py           Additional sufficient-state seeds
  lstm_ten_pinn.py                Current raw-history LSTM-PINN
  lstm_ten_bsde.py                Current raw-history LSTM-BSDE
  lstm_ten_contracts.py           Discrete payoffs and quadrature references
  single_asset_signature_bsde.py  Original Signature barrier baseline
  exact_terminal_history_bsde.py  Historical exact-terminal Signature variant
  reproduce_author_lookback.py    Author's single-asset lookback example
  report_*.py                    Report generation from saved results
vendor/deep_ppde/
  lib/                           Required pinned author library
  source_manifest.json           Source URLs, commit, and SHA256 hashes
  LICENSE                        Upstream GPL-3.0 license
```

All commands below run from the repository root. Use a fresh output directory for a new experiment. Training programs create their output directories; no pretrained weights are required by the current ten-option entry points.

## Sufficient-State PINNs

Run the soft-constraint, hard-constraint, and H-based comparison:

```bash
python -m scripts.run_anchored_transition --output-dir outputs/anchored_transition_15000
```

This trains the required variants from scratch, creates shared barrier MC anchors, refines H models using L-BFGS, and evaluates prices and Greeks. Existing historical seed-1 checkpoints are reused only with the explicit `--reuse-existing` option. The default run uses 15,000 Adam updates per trained variant; refinement adds computation.

After seed 1 has generated `mc_anchors.json`, run the other seeds:

```bash
python -m scripts.run_anchored_seeds --seed 2 --workers 4 --output-dir outputs/anchored_transition_15000_seed2
python -m scripts.run_anchored_seeds --seed 3 --workers 4 --output-dir outputs/anchored_transition_15000_seed3
python -m scripts.report_state_pinn_seeds
```

Outputs:

- `outputs/anchored_transition_15000*/training/<case>/<variant>/`: checkpoints, per-update loss CSV, and configurations.
- `outputs/anchored_transition_15000*/stages/<stage>/evaluation/`: prices, Delta/Gamma errors, state/path metrics, and residual audits.
- `outputs/anchored_transition_15000/comparison/`: comparison CSV/JSON, plots, and the generated English report.
- `outputs/anchored_transition_seeds/`: three-seed aggregate data and report.

## Raw-History LSTM-PINN and LSTM-BSDE

Available cases are `lookback_call`, `lookback_put`, `up_call`, `up_put`, `down_call`, and `down_put`. Run each case with seeds 1, 2, and 3 to reproduce the ten-option tables; knock-ins are derived during evaluation.

Examples:

```bash
python -m scripts.lstm_ten_pinn --case up_call --seed 1 --device cuda --output-dir outputs/lstm_ten_50000/pinn/up_call/seed1
python -m scripts.lstm_ten_bsde --case up_call --seed 1 --device cpu --output-dir outputs/lstm_ten_50000/bsde/up_call/seed1
```

Both default to 50,000 total updates, including 3,000 terminal-initialization updates, with a cosine main-stage learning-rate schedule. BSDE main updates are split across 16 backward intervals, while the contract still monitors all 128 steps.

For the final lookback-put PINN configuration, normalize the PDE without the envelope factor:

```bash
python -m scripts.lstm_ten_pinn --case lookback_put --seed 1 --pde-scale uncapped --device cuda --output-dir outputs/lstm_ten_50000/pinn_put_uncapped/lookback_put/seed1
```

Use the corresponding `seed2` and `seed3` paths for the other seeds. Once scores exist, regenerate the aggregate report with:

```bash
python -m scripts.report_lstm_ten
```

Each run writes `config.json`, `progress.json`, `history.csv`, model checkpoints, and `scores.json` under its specified output directory. The aggregate report and `summary.json` are written under `outputs/lstm_ten_50000/`. Report commands read saved scores; they do not train models or overwrite the research summary.

## Signature Baseline and Author Example

Run the project-adapted single-asset Signature barrier baseline:

```bash
python -m scripts.single_asset_signature_bsde --device cpu --output-dir outputs/single_asset_author_bsde_barrier_15000
```

This is the baseline whose collapse is documented in the summary, not the improved raw-history solver. It uses the bundled, verified author library by default. The program records its checkpoints, loss history, validation, conditional prices, and hedging diagnostics under the selected directory.

The public author's single-asset lookback example is available separately:

```bash
python -m scripts.reproduce_author_lookback --device cpu --output-dir outputs/author_lookback_readme_10000
```

This wrapper downloads and verifies the pinned source under its output directory and uses the author's example parameters. It requires network access and is not the common-market ten-option experiment.

The retained `exact_terminal_history_bsde.py` is a historical matched-initialization comparison: it requires an explicitly supplied terminal-only BSDE checkpoint through `--initial-value`. That historical checkpoint is not distributed. The current ten-option programs do not depend on it.

## Paper and Source Attribution

[Solving path dependent PDEs with LSTM networks and path signatures](https://arxiv.org/abs/2011.10630), Marc Sabate-Vidales, David Siska, and Lukasz Szpruch (2020). A [PDF copy](docs/2011.10630_Solving_path_dependent_PDEs_with_LSTM_networks_and_path_signatures.pdf) is included.

The necessary author library is pinned to commit `b704b32d3a5b064586893bdb15a00f1aa12d2fc9` of [msabvid/Deep-PPDE](https://github.com/msabvid/Deep-PPDE). Its files are unchanged and accompanied by their original GPL-3.0 license and source manifest. Project wrappers, state models, and raw-history variants are identified separately from the original author method.
