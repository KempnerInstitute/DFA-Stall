# Common-Mode Collapse and Recovery in Direct Feedback Alignment

Code accompanying the paper by Varun Reddy, Houman Safaai and Bernardo L. Sabatini.

The experiments examine how a shared output error can drive hidden units toward
saturation in direct feedback alignment (DFA), and how the readout, optimizer and
input statistics affect collapse and subsequent learning. This repository contains
the implementation used for the revised paper, experiment configurations,
numerical tests, analysis scripts and compact reference results.

## Install

Use Python 3.10. The dependency versions in `requirements.txt` were used for the
release checks. Most experiments support CPU or CUDA; the fresh-architecture
validation suite requires CUDA.

```bash
git clone https://github.com/KempnerInstitute/DFA-Stall.git
cd DFA-Stall
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Check the release

```bash
python scripts/reproduce_submission.py
```

This verifies the release file hashes and runs 50 numerical and reporting tests.
After dependencies are installed, the checks need no image datasets, network
access or GPU. They cover the update decomposition, learning rules, diagnostics,
reduced model, and the consistency of the CIFAR figure and table measurements.
Run the hash check before modifying files or regenerating results. For subsequent
code changes, run `PYTHONPATH=src python -m pytest tests -q` directly.

## Run a baseline experiment

```bash
export CMC_DATA_ROOT="$PWD/data"
python scripts/download_data.py --datasets mnist
PYTHONPATH=src python -m cmc.run \
  --exp example --tag dfa --seed 0 --fb_seed 0 --steps 3000 --device cpu
```

This trains a three-layer tanh network with sigmoid outputs on MNIST. Use
`--rule bp` for the backpropagation control, or `--out_bias_q 0.1` for DFA with a
readout initially calibrated to the balanced class prior. Run each control under
a different `--tag`. Trajectories, metadata and optional unit snapshots are written
to `results/<experiment>/<tag>/`.

The default data directory is `./data`; `CMC_DATA_ROOT` overrides it. To obtain all
image datasets used in the paper, run:

```bash
python scripts/download_data.py --datasets mnist fashion cifar10 cifar100
```

## Reproduce the paper

[REPRODUCING.md](REPRODUCING.md) lists the experiment families, analysis order,
figure-to-script mapping and dependencies. It distinguishes checks that use the
included reference records from analyses that require new training runs.

For example, the focused summary tables and the CIFAR geometry figure can be
generated directly from the included records:

```bash
mkdir -p paper/tables
python scripts/supplement_tables.py
PYTHONPATH=src python scripts/figures/fig7_cifar.py
```

Generated tables and figures appear under `paper/`. This directory is an output
location; the repository does not contain the manuscript or pre-rendered figures.

| Directory | Contents |
| --- | --- |
| `src/cmc/` | Networks, learning rules, datasets, diagnostics and reduced model |
| `configs/` | Experiment configurations and seed choices |
| `scripts/` | Training, data download, analysis, plotting and validation |
| `tests/` | Numerical and reporting regression tests |
| `results/` | Run metadata, numerical summaries, source mappings and validation inputs |

The reference records cover the E1–E22 experiments, reduced-model comparisons,
gate and residual measurements, and selected CIFAR metric trajectories. They
make numerical reporting inspectable without rerunning every experiment. Full
training trajectories, unit snapshots and image datasets are not included;
regenerate them before running analyses that require those records. Floating-point
results can vary across hardware.

## Project history

[Varun Reddy](https://github.com/varun04reddy) developed the original DFA stall
implementation. His seven commits remain in this repository's `main` history
with Varun as their sole credited author. His authorship, dates and code changes
are preserved; removing automated co-author trailers changed the commit IDs.
The [original implementation](https://github.com/KempnerInstitute/DFA-Stall/tree/8c21518db68b6beec360a196fa7e12d581af827d)
and its figures can be inspected at that revision. The current checkout contains
the implementation and reference records used for the revised paper.

## Release provenance and citation

This release uses the same training, analysis and test code as the paper's
anonymous code supplement. `release.json` identifies the source revision and
supplement archive, and `manifest.json` records the released file hashes. Public
documentation replaces the anonymous package's introduction; scientific code
and numerical reference records are preserved.

Use [CITATION.cff](CITATION.cff) to cite the code and the paper title above to cite
the manuscript. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the release scope.
