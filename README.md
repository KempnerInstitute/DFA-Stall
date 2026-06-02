# DFA-STALL

Self-contained experiment: Direct Feedback Alignment stall on MNIST.

## Contents

```
DFA-STALL/
├── train.py          ← single training + figure script (run this)
├── run_slurm.sh      ← SLURM submission for GPU node
├── README.md
├── data/             ← symlink to shared MNIST (./data/MNIST/)
├── logs/             ← SLURM stdout/stderr
├── figures/          ← generated PNGs (written after training)
│   ├── fig1_training_curves.png
│   ├── fig2_order_params.png
│   └── fig3_phase_portraits.png
└── metrics.csv       ← per-step data (written after training)
```

## Quick start

### Login node (CPU, ~10 min)
```bash
cd DFA-STALL
/n/home00/varunreddy/dynamics/venv/bin/python train.py
```

### GPU via SLURM (~3 min)
```bash
cd DFA-STALL
sbatch run_slurm.sh
```

## Model

| | Value |
|---|---|
| Architecture | 3 × 300 tanh MLP |
| Output | sigmoid, binary log-loss |
| Optimizer | SGD, lr = 1e-3 |
| Batch size | 128 |
| Steps | 3 000 |
| Dataset | MNIST (10-class) |

## Order parameters tracked (every step)

| Column | Equation | Meaning |
|---|---|---|
| `gate_participation_lN` | $(Σ u_i)^2 / (d · Σ u_i^2)$ | How evenly DFA teaching energy is distributed |
| `gate_strength_sq_lN` | $(1/d) Σ u_i$, $u_i = E[φ'(a_i)^2]$ | Mean squared gate value |
| `saturation_frac_lN` | $P(\|φ'(a)\| < 0.01)$ | Fraction of deeply saturated units |
| `feedback_signal_norm_lN` | $\|D_\ell B_\ell e\|_F / \sqrt{Bd}$ | Normalised teaching signal magnitude |
| `angular_update_lN` | $Σ_i \|(I-\hat w_i \hat w_i^T)G_i\|^2 / \|w_i\|^2$ | Rotational component of DFA weight update |
| `feature_movement_lN` | $Σ_s E[\|h_\ell(t)-h_\ell(t-1)\|^2] / E[\|h_\ell(0)\|^2]$ | Cumulative hidden representation travel |
| `weight_alignment_lN` | $\cos(M_\ell, B_\ell)$ | Forward-feedback matrix alignment |
| `grad_alignment` | $\cos(g_{\rm DFA}, g_{\rm BP})$ | Global DFA vs BP gradient alignment |
| `dfa_loss` / `bp_loss` | binary log-loss | Training losses |

## Figures

**fig1_training_curves.png** — DFA vs BP loss (log), weight alignment, gradient alignment.
Stall region shaded in orange.

**fig2_order_params.png** — All 6 order parameters per layer over training.
Stall region shaded in orange.

**fig3_phase_portraits.png** — Phase-portrait scatter (participation, feature movement, weight alignment vs gradient alignment).
Viridis = training step; **orange squares = stall steps**.
