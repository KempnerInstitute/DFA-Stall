#!/usr/bin/env python3
"""
make_landscape.py — 2D loss landscape from real parameter checkpoints.

Uses the checkpoints saved by train.py to project the full loss surface
into a 2D plane defined by the actual DFA and BP optimization trajectories.

Coordinate system
-----------------
  Reference point θ_ref : DFA parameters at the stall start (step 118).

  Direction d1  : DFA(final) − DFA(stall_start)
                  The direction DFA eventually travels during recovery.

  Direction d2  : BP(final) − DFA(stall_start), Gram-Schmidt ⊥ d1.
                  The direction BP has moved relative to the stall point.
                  In this coordinate, BP ends up at β>0 while DFA stays near β≈0
                  during the stall — this makes the "different paths" visible.

Both directions are filter-normalised (Li et al. 2018): each layer's component
is scaled so that α=1 corresponds to moving by one weight-norm unit, making the
landscape scale-interpretable.

Grid: 50×50 evaluations of real MNIST loss (2000-sample validation batch).

Output: figures/fig7_loss_landscape.png / .svg / .pdf
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

HERE  = Path(__file__).resolve().parent
CKPT  = HERE / "checkpoints"
OUTDIR = HERE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── minimal model re-definition (self-contained) ──────────────────────────────

class TanhMLP(nn.Module):
    def __init__(self, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        dims = [784, 300, 300, 300, 10]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        for l in self.layers:
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
        self.preacts: List[torch.Tensor] = []
        self.acts:    List[torch.Tensor] = []

    def forward(self, x):
        self.preacts, self.acts = [], []
        h = x
        for l in self.layers[:-1]:
            a = l(h); h = torch.tanh(a)
            self.preacts.append(a); self.acts.append(h)
        return torch.sigmoid(self.layers[-1](h))

def binary_log_loss(t, p, eps=1e-12):
    p = p.clamp(eps, 1-eps)
    return -(t*p.log() + (1-t)*(1-p).log()).sum(1).mean()

def to_one_hot(y, n):
    return F.one_hot(y.long(), n).float()

# ── parameter vector utilities ────────────────────────────────────────────────

def param_vec(model: TanhMLP) -> torch.Tensor:
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])

def load_vec(model: TanhMLP, vec: torch.Tensor, device) -> None:
    vec = vec.to(device)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[offset:offset+n].reshape(p.shape))
            offset += n

def filter_normalise(d: torch.Tensor, ref: TanhMLP) -> torch.Tensor:
    """Scale each layer's chunk of d to have the same Frobenius norm as the
    corresponding layer in ref.  This makes α/β interpretable as weight-norm units."""
    out = d.clone()
    offset = 0
    for p in ref.parameters():
        n = p.numel()
        chunk = d[offset:offset+n]
        ref_norm  = p.detach().cpu().norm().item()
        chunk_norm = chunk.norm().item()
        if chunk_norm > 1e-12:
            out[offset:offset+n] = chunk * (ref_norm / chunk_norm)
        offset += n
    return out

# ── load MNIST ────────────────────────────────────────────────────────────────

def load_mnist(data_dir: Path):
    from torchvision import datasets
    te = datasets.MNIST(root=str(data_dir), train=False, download=False)
    X  = te.data.float().div(255.0).view(-1, 784)
    y  = torch.as_tensor(te.targets, dtype=torch.long)
    return X[:2000], y[:2000]     # fixed 2000-sample validation slice

# ── evaluate loss at a parameter vector ───────────────────────────────────────

@torch.no_grad()
def eval_loss(model: TanhMLP, vec: torch.Tensor, X, y, device) -> float:
    load_vec(model, vec, device)
    preds = model(X.to(device).float())
    tgts  = to_one_hot(y.to(device), 10)
    return float(binary_log_loss(tgts, preds).item())

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # check checkpoints exist
    required = ["dfa_00001.pt", "dfa_00118.pt", "dfa_03000.pt",
                "bp_00001.pt",  "bp_00118.pt",  "bp_03000.pt"]
    missing  = [f for f in required if not (CKPT / f).exists()]
    if missing:
        print(f"Missing checkpoints: {missing}")
        print("Re-run train.py first.")
        return

    # available checkpoint steps
    dfa_ckpts = sorted(CKPT.glob("dfa_*.pt"))
    bp_ckpts  = sorted(CKPT.glob("bp_*.pt"))
    dfa_steps = [int(p.stem.split("_")[1]) for p in dfa_ckpts]
    bp_steps  = [int(p.stem.split("_")[1]) for p in bp_ckpts]
    print(f"DFA checkpoints: {dfa_steps}")

    # load validation data
    X_val, y_val = load_mnist(HERE / "data")
    print(f"Validation: {tuple(X_val.shape)}")

    # load models
    seed = 42
    m_tmp = TanhMLP(seed).to(device)   # reusable workspace model

    def load_ckpt(path):
        sd = torch.load(path, map_location="cpu")
        m_tmp2 = TanhMLP(seed)
        m_tmp2.load_state_dict(sd)
        return param_vec(m_tmp2)

    # key parameter vectors
    dfa_vecs: Dict[int, torch.Tensor] = {}
    bp_vecs:  Dict[int, torch.Tensor] = {}
    for p in dfa_ckpts: dfa_vecs[int(p.stem.split("_")[1])] = load_ckpt(p)
    for p in bp_ckpts:  bp_vecs[int(p.stem.split("_")[1])]  = load_ckpt(p)

    # ── define 2D coordinate system ───────────────────────────────────────────
    # Stall reference: DFA at stall start (step 118, closest available)
    stall_step = min(dfa_vecs.keys(), key=lambda s: abs(s - 118))
    final_step = max(dfa_vecs.keys())
    print(f"Using stall step: {stall_step},  final step: {final_step}")

    theta_ref = dfa_vecs[stall_step]

    # d1: DFA recovery direction (stall→final)
    d1_raw = dfa_vecs[final_step] - theta_ref
    m_ref  = TanhMLP(seed); m_ref.load_state_dict(
        torch.load(CKPT / f"dfa_{stall_step:05d}.pt", map_location="cpu"))
    d1     = filter_normalise(d1_raw, m_ref)
    d1_hat = d1 / d1.norm().clamp_min(1e-12)

    # d2: BP direction (BP_final − DFA_stall), Gram-Schmidt ⊥ d1
    d2_raw  = bp_vecs[final_step] - theta_ref
    d2_raw  = d2_raw - torch.dot(d2_raw, d1_hat) * d1_hat  # orthogonalise
    d2      = filter_normalise(d2_raw, m_ref)
    d2_hat  = d2 / d2.norm().clamp_min(1e-12)

    scale1 = d1.norm().item()
    scale2 = d2.norm().item()
    print(f"d1 norm (DFA recovery): {scale1:.2f}")
    print(f"d2 norm (BP direction): {scale2:.2f}")

    # ── project all checkpoints onto (d1̂, d2̂) ────────────────────────────────
    def proj(vec):
        diff = vec - theta_ref
        a = torch.dot(diff, d1_hat).item() / scale1
        b = torch.dot(diff, d2_hat).item() / scale2
        return a, b

    dfa_proj = [(s, *proj(v)) for s, v in sorted(dfa_vecs.items())]
    bp_proj  = [(s, *proj(v)) for s, v in sorted(bp_vecs.items())]

    dfa_proj_arr = np.array([(a, b) for _, a, b in dfa_proj])
    bp_proj_arr  = np.array([(a, b) for _, a, b in bp_proj])
    dfa_proj_steps = [s for s, _, _ in dfa_proj]
    bp_proj_steps  = [s for s, _, _ in bp_proj]

    # ── compute loss landscape on 50×50 grid ──────────────────────────────────
    # Range: cover both trajectories with a margin
    all_a = np.concatenate([dfa_proj_arr[:, 0], bp_proj_arr[:, 0]])
    all_b = np.concatenate([dfa_proj_arr[:, 1], bp_proj_arr[:, 1]])
    margin = 0.25
    a_lo, a_hi = all_a.min() - margin, all_a.max() + margin
    b_lo, b_hi = all_b.min() - margin, all_b.max() + margin

    N_GRID = 50
    a_grid = np.linspace(a_lo, a_hi, N_GRID)
    b_grid = np.linspace(b_lo, b_hi, N_GRID)
    AG, BG = np.meshgrid(a_grid, b_grid)
    LG     = np.zeros_like(AG)

    print(f"Computing {N_GRID}×{N_GRID} = {N_GRID**2} loss evaluations …")
    for i in range(N_GRID):
        for j in range(N_GRID):
            theta = theta_ref + AG[i, j] * scale1 * d1_hat \
                              + BG[i, j] * scale2 * d2_hat
            LG[i, j] = eval_loss(m_tmp, theta, X_val, y_val, device)
        if i % 10 == 0:
            print(f"  row {i}/{N_GRID}  loss range so far: "
                  f"[{LG[:i+1].min():.3f}, {LG[:i+1].max():.3f}]")

    # restore ref for labelling
    load_vec(m_tmp, theta_ref, device)

    # loss at DFA stall = ref point
    loss_at_stall = eval_loss(m_tmp, theta_ref, X_val, y_val, device)
    print(f"Loss at stall reference: {loss_at_stall:.4f}")

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.75, "lines.linewidth": 1.3,
    })
    def clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    STALL_S, STALL_E = 118, 451
    DFA_COL  = "#1f77b4"
    BP_COL   = "#d62728"
    STALL_COL = "#e8501a"

    # stall steps mask
    dfa_stall_mask = np.array([STALL_S <= s <= STALL_E for s in dfa_proj_steps])
    bp_stall_mask  = np.array([STALL_S <= s <= STALL_E for s in bp_proj_steps])

    fig = plt.figure(figsize=(12, 5.2))
    gs  = GridSpec(1, 2, figure=fig, left=0.08, right=0.96,
                   bottom=0.13, top=0.88, wspace=0.38)

    # ── panel A: full landscape ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    LG_clipped = np.clip(LG, 0, np.percentile(LG, 97))
    lvls = np.linspace(LG_clipped.min(), LG_clipped.max(), 30)
    cf = ax.contourf(AG, BG, LG_clipped, levels=lvls, cmap="Blues_r", alpha=0.70)
    ax.contour(AG, BG, LG_clipped, levels=lvls, colors="white",
               linewidths=0.25, alpha=0.5)
    plt.colorbar(cf, ax=ax, pad=0.02, fraction=0.05,
                 label="Validation loss")

    # full trajectories (thin)
    ax.plot(dfa_proj_arr[:, 0], dfa_proj_arr[:, 1], color=DFA_COL,
            lw=0.8, alpha=0.5)
    ax.plot(bp_proj_arr[:, 0],  bp_proj_arr[:, 1],  color=BP_COL,
            lw=0.8, alpha=0.5)

    # non-stall scatter
    ax.scatter(dfa_proj_arr[~dfa_stall_mask, 0], dfa_proj_arr[~dfa_stall_mask, 1],
               c=np.array(dfa_proj_steps)[~dfa_stall_mask], cmap="Blues",
               vmin=0, vmax=3000, s=22, linewidths=0.4,
               edgecolors="k", zorder=4, label="DFA")
    ax.scatter(bp_proj_arr[~bp_stall_mask, 0],  bp_proj_arr[~bp_stall_mask, 1],
               c=np.array(bp_proj_steps)[~bp_stall_mask], cmap="Reds",
               vmin=0, vmax=3000, s=22, linewidths=0.4,
               edgecolors="k", zorder=4, label="BP")

    # stall squares (orange)
    if dfa_stall_mask.any():
        ax.scatter(dfa_proj_arr[dfa_stall_mask, 0],
                   dfa_proj_arr[dfa_stall_mask, 1],
                   color=STALL_COL, marker="s", s=35,
                   linewidths=0.5, edgecolors="k",
                   zorder=5, label="DFA stall")
    if bp_stall_mask.any():
        ax.scatter(bp_proj_arr[bp_stall_mask, 0],
                   bp_proj_arr[bp_stall_mask, 1],
                   color="#ff9900", marker="s", s=35,
                   linewidths=0.5, edgecolors="k",
                   zorder=5, label="BP at stall steps")

    # start / end markers
    ax.scatter(*dfa_proj_arr[0],  marker="x", color=DFA_COL, s=80, lw=1.5, zorder=6)
    ax.scatter(*dfa_proj_arr[-1], marker="o", facecolor="none",
               edgecolor=DFA_COL, s=80, lw=1.5, zorder=6)
    ax.scatter(*bp_proj_arr[0],   marker="x", color=BP_COL,  s=80, lw=1.5, zorder=6)
    ax.scatter(*bp_proj_arr[-1],  marker="o", facecolor="none",
               edgecolor=BP_COL,  s=80, lw=1.5, zorder=6)

    # reference point annotation
    ax.scatter(0, 0, marker="*", color="gold", edgecolor="k", s=200,
               zorder=7, linewidths=0.8)
    ax.annotate("DFA stall\nreference", xy=(0, 0),
                xytext=(0.15, -0.12),
                fontsize=6.5, color=STALL_COL,
                arrowprops=dict(arrowstyle="->", color=STALL_COL, lw=0.8))

    ax.set_xlabel(r"$\alpha$  (DFA recovery direction,  filter-normalised)")
    ax.set_ylabel(r"$\beta$  (BP direction,  filter-normalised)")
    ax.set_title("2D loss landscape slice\n"
                 r"Origin = DFA stall start  |  $\alpha$=DFA recovery  |  $\beta$=BP path")
    ax.legend(frameon=False, loc="upper right", fontsize=6.5, ncol=2)
    clean(ax)
    ax.text(0.02, 0.97, "a", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")

    # ── panel B: zoomed to stall region ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    # zoom to within ±0.4 of origin
    zmarg = 0.45
    z_mask_a = (AG >= -zmarg) & (AG <= zmarg)
    z_mask_b = (BG >= -zmarg) & (BG <= zmarg)
    z_mask   = z_mask_a & z_mask_b
    if z_mask.any():
        AG_z = np.where(z_mask, AG, np.nan)
        BG_z = np.where(z_mask, BG, np.nan)
        LG_z = np.where(z_mask, LG_clipped, np.nan)
        lvls_z = np.linspace(np.nanmin(LG_z), np.nanmax(LG_z), 25)
        ax2.contourf(AG, BG, LG_clipped, levels=lvls_z, cmap="Blues_r", alpha=0.70)
        ax2.contour(AG,  BG, LG_clipped, levels=lvls_z, colors="white",
                    linewidths=0.3, alpha=0.5)

    # same trajectory in zoomed view
    ax2.plot(dfa_proj_arr[:, 0], dfa_proj_arr[:, 1], color=DFA_COL, lw=0.8, alpha=0.4)
    ax2.plot(bp_proj_arr[:, 0],  bp_proj_arr[:, 1],  color=BP_COL,  lw=0.8, alpha=0.4)

    ax2.scatter(dfa_proj_arr[~dfa_stall_mask, 0], dfa_proj_arr[~dfa_stall_mask, 1],
                c=np.array(dfa_proj_steps)[~dfa_stall_mask], cmap="Blues",
                vmin=0, vmax=3000, s=25, linewidths=0.4, edgecolors="k", zorder=4)
    ax2.scatter(bp_proj_arr[~bp_stall_mask, 0],  bp_proj_arr[~bp_stall_mask, 1],
                c=np.array(bp_proj_steps)[~bp_stall_mask], cmap="Reds",
                vmin=0, vmax=3000, s=25, linewidths=0.4, edgecolors="k", zorder=4)
    if dfa_stall_mask.any():
        ax2.scatter(dfa_proj_arr[dfa_stall_mask, 0],
                    dfa_proj_arr[dfa_stall_mask, 1],
                    color=STALL_COL, marker="s", s=45,
                    linewidths=0.5, edgecolors="k", zorder=5)
    if bp_stall_mask.any():
        ax2.scatter(bp_proj_arr[bp_stall_mask, 0],
                    bp_proj_arr[bp_stall_mask, 1],
                    color="#ff9900", marker="s", s=45,
                    linewidths=0.5, edgecolors="k", zorder=5)

    ax2.scatter(0, 0, marker="*", color="gold", edgecolor="k", s=200, zorder=7, lw=0.8)
    ax2.set_xlim(-zmarg, zmarg); ax2.set_ylim(-zmarg, zmarg)
    ax2.set_xlabel(r"$\alpha$  (DFA recovery direction)")
    ax2.set_ylabel(r"$\beta$  (BP direction)")
    ax2.set_title("Stall region — zoomed\n"
                  "Orange ■ = DFA stall steps  |  Yellow ■ = BP same steps")
    clean(ax2)
    ax2.text(0.02, 0.97, "b", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top")

    fig.suptitle(
        "Real loss landscape: 2D projection through DFA stall and BP trajectories\n"
        "Loss evaluated on 2000 MNIST samples at each of the 2500 grid points  "
        f"(ref loss at stall: {loss_at_stall:.3f})",
        fontsize=8.5,
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUTDIR / f"fig7_loss_landscape.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/fig7_loss_landscape.png/svg/pdf")

    # save projection data alongside figure
    np.savez(OUTDIR / "landscape_data.npz",
             AG=AG, BG=BG, LG=LG,
             dfa_proj=dfa_proj_arr, dfa_steps=np.array(dfa_proj_steps),
             bp_proj=bp_proj_arr,   bp_steps=np.array(bp_proj_steps))
    print("Saved: figures/landscape_data.npz")


if __name__ == "__main__":
    main()
