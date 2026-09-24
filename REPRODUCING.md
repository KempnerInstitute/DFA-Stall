# Common-Mode Collapse and Recovery in Direct Feedback Alignment

This repository contains the experimental code, configurations, tests and analysis scripts for the paper.
The manuscript gives the protocol in Appendix A, derivations in B, mechanism and model tests in C,
decodability and gate recovery in D, and interventions and boundary cases in E.

## Contents

- `src/cmc/`: data loading, networks, learning rules, diagnostics, training, and the reduced model.
- `configs/`: complete command-line configurations for the experiment families.
- `scripts/`: runners, analysis, figure and table generation, and reproducibility checks.
- `tests/`: numerical identity, training-rule, diagnostic and model checks.
- `results/residual_ratio.*`: saved reference outputs for the residual diagnostic.
- `results/review_20260919/`: frozen-feature, readout, drift, recovery-ablation and reporting checks.
- `results/review_resolution/`: saved initial states and quadrature/Euler resolution checks for three selected settings.
- `results/generality_corr_rows.csv`: deduplicated trajectories and physical-condition keys for the generality correlation.
- `results/cifar_reporting/`: shared CIFAR-10 figure/table trajectories, run metadata and source hashes, per-run extrema, and the layer-selection audit.
- `results/E18_fresh/`: metadata and predictions saved before the nine architecture-validation runs.
- `results/paper_summary_sources/`: complete E1–E12 condition and per-run summaries, source hashes, and a mapping from each focused manuscript table to its source rows.

This code release contains the code, configurations, tests, and compact numerical
validation records. `python scripts/reproduce_submission.py` verifies its file hashes and runs
the regression tests without image data or a GPU after dependencies are installed.
Full training trajectories and unit snapshots are omitted, as are rendered figures and all
manuscript files. The root README and manifest describe its coverage. The commands
below explain training and analyses; regenerate the required training records before plotting
or running analyses that consume trajectories or unit snapshots.
## Environment and data

The experiments used Python 3.10 and PyTorch 2.9. Required packages are `torch`, `torchvision`,
`numpy`, `scipy`, `pandas`, `matplotlib`, and `pytest`. A CUDA device is optional.
Set `CMC_DATA_ROOT` to the data directory before running. The code expects torchvision's downloaded
MNIST and FashionMNIST trees there, and CIFAR-10/CIFAR-100 under `$CMC_DATA_ROOT/torchvision/`.
The loader uses `download=False`; download those public datasets with torchvision first.

## Reproduce a training run

The optimizer, drift, sampling and width–depth comparisons can be reproduced with:

```bash
PYTHONPATH=src python scripts/run_revision_suite.py --device cuda
PYTHONPATH=src python scripts/measure_revision_dynamics.py --device cuda
PYTHONPATH=src python scripts/measure_batch_noise.py --device cuda
PYTHONPATH=src python scripts/run_revision_models.py --grid
PYTHONPATH=src python scripts/reduced_crosscheck.py --grid
PYTHONPATH=src python scripts/summarize_revision.py
python scripts/revision_tables.py
PYTHONPATH=src python scripts/figures/fig_revision_controls.py
```

The suite records 129 training runs: 27 optimizer comparisons, 12 activation/feedback
controls, 12 CIFAR-100 runs (DFA and matched BP for each readout), 30 batch-size controls and
48 width-depth runs. Three paired seeds are used per condition. `configs/revision_20260918.json`
gives every setting; fields added later default to the earlier behaviour.
The runner skips a completed matching run and refuses to overwrite a different one.
The diagnostic scripts use copied networks for virtual steps and never change training.
The 48-condition reduced-model comparison is replicated across seeds with
`scripts/run_revision_models.py`, then audited by `scripts/reduced_crosscheck.py --all-seeds`.
The shipped coverage manifest identifies all 144 primary model integrations, including seed 0.
The comparison tables retain censored exits and distinguish condition labels from
distinct physical settings. The corresponding training records must first be regenerated for
that replication; saved comparison outputs are included as references.

## The original DFA protocol

```bash
PYTHONPATH=src python scripts/run_nokland_protocol.py --device cuda
PYTHONPATH=src python scripts/submission_reporting.py
```

The runner trains the 3x800 tanh MNIST network of Nokland (2016) with logistic outputs, binary
cross-entropy, inputs in [0, 1], RMSprop and minibatches of 64: DFA from zero initialization with
uniform feedback scaled by fan-out, BP from fan-in initialization, and DFA with this paper's Xavier
initialization and Gaussian feedback, each at RMSprop rates 1e-3 and 1e-4 with three paired seeds
and 6,000 updates (18 runs; `configs/nokland_20260923.json`). The reporting script writes
`results/submission_20260923/nokland_runs.csv` and the corresponding supplement table.

`scripts/run_sign_feedback.py` trains the sign-error feedback test (E20: plain, error-centered and prior-bias arms,
three paired seeds, 3,000 updates; `configs/sign_20260923.json`), which the same reporting script summarizes.

`configs/E21_lr.txt` (plain-SGD learning rates up to 0.1 for DFA, DFA with prior bias and BP; 36 runs) and
`configs/E22_imbcalib.txt` (class-imbalanced softmax with a readout calibrated to the sampling prior; 6 runs) are
run with `scripts/run_local.sh`; `scripts/submission_reporting.py` also summarizes them, together with the centered
activity geometry of E1 and the mean-activity growth under error centering in E10.

To run an individual baseline and the numerical checks:

```bash
PYTHONPATH=src python -m cmc.run --exp smoke --tag base --seed 0 --fb_seed 0 --steps 300
PYTHONPATH=src python -m pytest tests -q
```

Each line in `configs/*.txt` contains arguments for `python -m cmc.run`; files for the
teacher--student experiment use its dedicated runner, `scripts/run_ladder.sh`.
A run writes a CSV trajectory, a JSON metadata record and (when enabled) NPZ unit snapshots to
`results/<experiment>/<condition>/`. Use `scripts/run_local.sh configs/<family>.txt N` to run a configuration file with `N` local workers.
The Python entry points also run individually; cluster scheduler templates are omitted from this archive.

## Reproduce analyses and figures

After the required training runs, create `paper/figures/` and `paper/tables/`, then run the scripts in this order:

```bash
PYTHONPATH=src python scripts/summarize_all.py
PYTHONPATH=src python scripts/derived_laws.py
PYTHONPATH=src python scripts/run_revision_models.py
PYTHONPATH=src python scripts/reduced_crosscheck.py --all-seeds
PYTHONPATH=src python scripts/analyze_snapshots.py
PYTHONPATH=src python scripts/gauss_reconstruction.py
python scripts/review_reporting.py
PYTHONPATH=src python scripts/submission_reporting.py
python scripts/make_numbers.py
python scripts/make_tables.py
python scripts/revision_tables.py
python scripts/supplement_tables.py
```

`scripts/submission_reporting.py` computes the learning cost of collapse (Table 5 and Figure 3c), the closed-form
null for the reduced model, the shared-to-input-dependent error ratios and the frozen-readout reference from
existing records. It also reports learning-rate robustness, centered geometry, mean-activity growth and
imbalanced readout calibration. It writes compact numerical records in `results/submission_20260923/`
and the corresponding `paper/tables/submission_*.tex` tables, and trains nothing.

The supplement tables can be regenerated directly from the included summaries with
`python scripts/supplement_tables.py`; no training or access to image data is needed.
`paper_rows.json` identifies every displayed row's source. The complete CSVs retain
sweeps and ancillary controls omitted from the printed supplement. Earlier generic
table generators remain available for inspecting full results.

Figure numbering follows the compiled manuscript, not the script filenames:

| Figures | Script in `scripts/figures/` | Content |
|---|---|---|
| 1–3 | `fig1_phenomenon.py`, `fig2_reduced_model.py`, `fig3_master_curve.py` | Baseline, reduced model and scaling |
| 4, 6 | `fig_review_validation.py` | Frozen-feature learning and closure checks |
| 5 | `fig4_dissection.py` | Mean-drive and alignment controls |
| 7, 11, 14 | `fig_revision_controls.py` | Model grid, Adam and batch fluctuations |
| 8 | `fig8b_gaussian_ablations.py` | Gaussian reconstruction and moment substitutions |
| 9 | `fig8_gates.py` | Gate turnover, masked descent and gradient direction/magnitude |
| 10, 12, 13 | `fig5_cures.py`, `fig6_generality.py`, `fig7_cifar.py` | Interventions and generality |
| 15 | `fig9_ladder.py` | Teacher–student comparison |

The Gaussian ablation script reads the reconstruction CSV; the gate figure also reads baseline
trajectories and the norm-matched virtual-step records. Legacy exploratory figures can still
be regenerated, but are not included in the manuscript or arXiv source bundle.
The shared-spread and zero-mean curves substitute measured moments in a Gaussian
reconstruction; they are not interventions on training. Figure 1 combines an
illustrative schematic with measured trajectories.
The current teacher--student figure is generated by `scripts/figures/fig9_ladder.py`;
`scripts/ladder_report.py` generates the detailed report and an earlier diagnostic grid.
The figure scripts create their output directories. Number and table generation create their
respective manuscript artifacts; these are intended to be used with the accompanying paper sources.

## CIFAR-10 reporting

The CIFAR-10 figure and geometry table can both be regenerated from the included
`results/cifar_reporting/` records, without image data or the full E11/E12 logs:

```bash
mkdir -p paper/tables
python scripts/supplement_tables.py
PYTHONPATH=src python scripts/figures/fig7_cifar.py
```

CNN geometry uses hidden block 3 (the 128-unit fully connected layer after two
convolutions); MLP geometry uses layer 1. Table extrema are computed within each
run before averaging. The manifest records source run paths, hashes, metric columns,
windows and reduction conventions. `python scripts/cifar_reporting.py` rebuilds this
archive from the original E11/E12 logs and checks it against the preserved summaries.

## Frozen features and recovery validation

```bash
PYTHONPATH=src python scripts/review_experiments.py --device cuda
PYTHONPATH=src python scripts/review_fresh_architectures.py
PYTHONPATH=src python scripts/review_model_ablation.py
python scripts/review_reporting.py
PYTHONPATH=src python scripts/figures/fig_review_validation.py
```

The frozen-feature script replays three baseline runs and checks their participation against
`E13_adam/sgd_base` records. It fits on 10,000 unique training examples and evaluates on 5,000
separate validation examples, with a fixed ridge and no selection on evaluation accuracy.
Readout continuation uses identical initial predictions and minibatches in raw, centered and
centered/RMS-scaled coordinates. Its metadata records the measurement source hashes.

The fresh-architecture runner saves four model predictions before training each width-450
network at depth 2, 4 or 6 (paired seeds 10, 11 and 12). It currently requires CUDA; the
remaining analysis scripts run on CPU. The ablation script also reads the original parameter
sweeps and width–depth records, so regenerate those runs before recomputing the complete
comparison. Saved ablation outputs are included, with no omitted failure cases. Reporting
requires the original training trajectories and baseline unit snapshots; regenerate them
before running `scripts/review_reporting.py`. Compact derived outputs are included for inspection.
The supplied reference outputs and protocol JSON files distinguish measurements from fits,
conditional timing errors, and predictions made before the new training runs.

## Reduced model

The numerical-resolution table can be regenerated without image data or training logs:

```bash
mkdir -p paper/tables
python scripts/review_resolution.py
```

The saved initial states include all model inputs and feedback matrices. With image data and
the original run metadata available, `--rebuild-init` reconstructs them before checking
32/64/128 quadrature nodes and 1/2/4 Euler substeps. These are three selected settings,
not an exhaustive convergence study. The generality interval resamples complete physical
conditions using `cmc.reporting.condition_correlation`; its input rows are included above.

```bash
PYTHONPATH=src python -m cmc.reduced --from_run results/E1_headline/base/seed0_fb0.meta.json --compare
```

The model reads its initial state from the network and data. Its mean-drive dynamics and additional
phenomenological covariance cascade have no fitted trajectory parameters. It predicts collapse
and an approximate exit, but does not model the participation rebound.

## Gate--error residual diagnostic

```bash
python scripts/measure_residual.py --output-dir reproduced_residual
```

This runs three fresh 300-step trajectories: MNIST tanh, MNIST ReLU and a CIFAR-10 tanh MLP,
with initialization seed 0 and feedback seed 0. These are single diagnostic trajectories, separate
from the fifteen-run baseline ensemble. Use `--seed`, `--fb-seed`, `--configs` or `--device` to change settings.

The CSV contains absolute residual and leading-term norms, their ratio, and the numerical error
in the exact decomposition. The JSON summary feeds manuscript numbers; the metadata JSON records
configurations, software versions and source/probe/feedback hashes. The reference source hashes
describe the original measurement code; portability edits to the archive's default data path can
change source hashes without changing a run's configuration or data.
