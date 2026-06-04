#!/usr/bin/env python3
"""
make_landscape.py — 2D loss landscape showing DFA vs BP trajectories.

The key insight from the data: DFA and BP start at the SAME point but travel
in nearly ORTHOGONAL directions through parameter space.  A coordinate system
that captures only one path hides the other entirely.

Coordinate system (this version)
---------------------------------
  Origin   : initial parameters θ₀ (same for both DFA and BP, same seed).
  α-axis   : direction DFA travels  =  DFA_final − θ₀  (filter-normalised).
  β-axis   : direction BP  travels  =  BP_final  − θ₀,  Gram-Schmidt ⊥ α.

In this system:
  • DFA trajectory sweeps along α (DFA stall = α barely changes for ~350 steps).
  • BP  trajectory sweeps along β (BP converges steadily while DFA stalls).
  • Both start at (0, 0) and diverge to (≈1, ≈0) and (≈0, ≈1) respectively.

The loss contours show the surface DFA and BP are actually navigating — you can
directly read off why DFA stalls: it is on a flatter part of the landscape while
BP is on a steeper ridge.

Output: figures/fig7_loss_landscape.png / .svg
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import List

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
import matplotlib.patheffects as pe

HERE   = Path(__file__).resolve().parent
CKPT   = HERE / "checkpoints"
OUTDIR = HERE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── minimal model ─────────────────────────────────────────────────────────────
class TanhMLP(nn.Module):
    def __init__(self, seed=42):
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

def to_one_hot(y, n): return F.one_hot(y.long(), n).float()

# ── param utils ───────────────────────────────────────────────────────────────
def param_vec(model):
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])

def load_vec(model, vec, device):
    vec = vec.to(device); offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[offset:offset+n].reshape(p.shape)); offset += n

def filter_norm(d, ref_model):
    """Li et al. filter normalisation: scale each layer chunk to ref layer norm."""
    out = d.clone(); offset = 0
    for p in ref_model.parameters():
        n = p.numel(); chunk = d[offset:offset+n]
        rn = p.detach().cpu().norm().item(); cn = chunk.norm().item()
        if cn > 1e-12: out[offset:offset+n] = chunk * (rn / cn)
        offset += n
    return out

# ── data ──────────────────────────────────────────────────────────────────────
def load_mnist(data_dir):
    from torchvision import datasets
    te = datasets.MNIST(root=str(data_dir), train=False, download=False)
    X = te.data.float().div(255.0).view(-1, 784)
    y = torch.as_tensor(te.targets, dtype=torch.long)
    return X[:2000], y[:2000]

@torch.no_grad()
def eval_loss(model, vec, X, y, device):
    load_vec(model, vec, device)
    preds = model(X.to(device).float())
    return float(binary_log_loss(to_one_hot(y.to(device), 10), preds).item())

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # verify checkpoints
    for f in ["dfa_00001.pt", "dfa_03000.pt", "bp_00001.pt", "bp_03000.pt"]:
        if not (CKPT / f).exists():
            print(f"Missing: {f} — run train.py first"); return

    X_val, y_val = load_mnist(HERE / "data")
    seed = 42
    m_work = TanhMLP(seed).to(device)

    def load_ckpt(path):
        m = TanhMLP(seed)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return param_vec(m)

    # load all checkpoints
    dfa_files = sorted(CKPT.glob("dfa_*.pt"))
    bp_files  = sorted(CKPT.glob("bp_*.pt"))
    dfa_steps = [int(p.stem.split("_")[1]) for p in dfa_files]
    bp_steps  = [int(p.stem.split("_")[1]) for p in bp_files]
    dfa_vecs  = {s: load_ckpt(p) for s, p in zip(dfa_steps, dfa_files)}
    bp_vecs   = {s: load_ckpt(p) for s, p in zip(bp_steps,  bp_files)}

    # ── coordinate system: SHARED ORIGIN = initial params ─────────────────────
    theta0 = dfa_vecs[1]          # step 1 = same for DFA and BP (same seed)
    m_ref  = TanhMLP(seed)        # reference model for filter normalisation
    m_ref.load_state_dict(torch.load(CKPT / "dfa_00001.pt", map_location="cpu"))

    # d1: DFA direction  (DFA_final − θ₀)
    d1_raw = dfa_vecs[3000] - theta0
    d1     = filter_norm(d1_raw, m_ref)
    d1_hat = d1 / d1.norm().clamp_min(1e-12)
    d1_scale = d1.norm().item()

    # d2: BP direction  (BP_final − θ₀),  Gram-Schmidt ⊥ d1
    d2_raw  = bp_vecs[3000] - theta0
    d2_raw  = d2_raw - torch.dot(d2_raw, d1_hat) * d1_hat   # orthogonalise
    d2      = filter_norm(d2_raw, m_ref)
    d2_hat  = d2 / d2.norm().clamp_min(1e-12)
    d2_scale = d2.norm().item()
    print(f"d1 (DFA direction) norm: {d1_scale:.2f}")
    print(f"d2 (BP  direction) norm: {d2_scale:.2f}")

    # project all checkpoints
    def proj(vec):
        diff = vec - theta0
        return (torch.dot(diff, d1_hat).item() / d1_scale,
                torch.dot(diff, d2_hat).item() / d2_scale)

    dfa_proj = np.array([proj(dfa_vecs[s]) for s in dfa_steps])
    bp_proj  = np.array([proj(bp_vecs[s])  for s in bp_steps])

    print("DFA trajectory range: "
          f"α [{dfa_proj[:,0].min():.3f}, {dfa_proj[:,0].max():.3f}]  "
          f"β [{dfa_proj[:,1].min():.3f}, {dfa_proj[:,1].max():.3f}]")
    print("BP  trajectory range: "
          f"α [{bp_proj[:,0].min():.3f}, {bp_proj[:,0].max():.3f}]  "
          f"β [{bp_proj[:,1].min():.3f}, {bp_proj[:,1].max():.3f}]")

    # ── loss grid ──────────────────────────────────────────────────────────────
    all_a = np.concatenate([dfa_proj[:,0], bp_proj[:,0]])
    all_b = np.concatenate([dfa_proj[:,1], bp_proj[:,1]])
    mg = 0.12
    a_lo, a_hi = all_a.min() - mg, all_a.max() + mg
    b_lo, b_hi = all_b.min() - mg, all_b.max() + mg

    N = 50
    ag = np.linspace(a_lo, a_hi, N)
    bg = np.linspace(b_lo, b_hi, N)
    AG, BG = np.meshgrid(ag, bg)
    LG = np.zeros_like(AG)

    print(f"Computing {N*N} loss evaluations …")
    for i in range(N):
        for j in range(N):
            theta = theta0 + AG[i,j]*d1_scale*d1_hat + BG[i,j]*d2_scale*d2_hat
            LG[i,j] = eval_loss(m_work, theta, X_val, y_val, device)
        if i % 10 == 0:
            valid = LG[:i+1][LG[:i+1] < 50]
            if len(valid): print(f"  row {i}: loss [{valid.min():.2f}, {valid.max():.2f}]")

    # clip outliers for display
    LG_disp = np.clip(LG, 0, np.percentile(LG, 96))

    # loss along each trajectory
    dfa_losses = np.array([eval_loss(m_work, dfa_vecs[s], X_val, y_val, device)
                           for s in dfa_steps])
    bp_losses  = np.array([eval_loss(m_work, bp_vecs[s],  X_val, y_val, device)
                           for s in bp_steps])
    print(f"DFA final loss: {dfa_losses[-1]:.4f}")
    print(f"BP  final loss: {bp_losses[-1]:.4f}")

    # stall masks
    STALL_S, STALL_E = 118, 451
    dfa_stall = np.array([STALL_S <= s <= STALL_E for s in dfa_steps])
    bp_stall  = np.array([STALL_S <= s <= STALL_E for s in bp_steps])

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
        "axes.linewidth":0.75,"lines.linewidth":1.3,
    })
    def clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    DFA_COL  = "#1f77b4"
    BP_COL   = "#d62728"
    STALL_COL = "#e8501a"

    fig = plt.figure(figsize=(13, 9))
    gs  = GridSpec(2, 2, figure=fig, left=0.09, right=0.97,
                   bottom=0.09, top=0.91, wspace=0.38, hspace=0.42)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel A: main landscape with both trajectories
    # ─────────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[:, 0])   # left half, full height

    lvls = np.linspace(LG_disp.min(), LG_disp.max(), 28)
    cf = ax.contourf(AG, BG, LG_disp, levels=lvls, cmap="RdYlGn_r", alpha=0.72)
    cs = ax.contour(AG, BG, LG_disp, levels=lvls[::3],
                    colors="k", linewidths=0.35, alpha=0.55)
    plt.colorbar(cf, ax=ax, pad=0.02, fraction=0.04, label="Validation loss")
    ax.clabel(cs, fmt="%.1f", fontsize=5.5, inline=True)

    # draw trajectories as thick lines
    ax.plot(dfa_proj[:,0], dfa_proj[:,1], color=DFA_COL, lw=2.0,
            alpha=0.85, zorder=4, label="DFA path")
    ax.plot(bp_proj[:,0],  bp_proj[:,1],  color=BP_COL,  lw=2.0,
            alpha=0.85, zorder=4, label="BP path")

    # non-stall circles
    ax.scatter(dfa_proj[~dfa_stall,0], dfa_proj[~dfa_stall,1],
               c=np.array(dfa_steps)[~dfa_stall], cmap="Blues",
               vmin=0, vmax=3000, s=30, edgecolors="k", lw=0.4, zorder=5)
    ax.scatter(bp_proj[~bp_stall,0],  bp_proj[~bp_stall,1],
               c=np.array(bp_steps)[~bp_stall],  cmap="Reds",
               vmin=0, vmax=3000, s=30, edgecolors="k", lw=0.4, zorder=5)

    # stall squares  ─── the key visual ───────────────────────────────────────
    if dfa_stall.any():
        ax.scatter(dfa_proj[dfa_stall,0], dfa_proj[dfa_stall,1],
                   color=STALL_COL, marker="s", s=55, edgecolors="k",
                   lw=0.5, zorder=6, label="DFA stall steps")
    if bp_stall.any():
        ax.scatter(bp_proj[bp_stall,0],  bp_proj[bp_stall,1],
                   color="#ff9900", marker="^", s=55, edgecolors="k",
                   lw=0.5, zorder=6, label="BP at same steps")

    # arrows showing direction of movement
    for arr, col in [(dfa_proj, DFA_COL), (bp_proj, BP_COL)]:
        for k in range(len(arr)-1):
            da = arr[k+1,0]-arr[k,0]; db = arr[k+1,1]-arr[k,1]
            if np.sqrt(da**2+db**2) > 0.008:
                ax.annotate("", xy=(arr[k+1,0], arr[k+1,1]),
                            xytext=(arr[k,0], arr[k,1]),
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=1.2, mutation_scale=8))

    # shared start marker
    ax.scatter(0, 0, marker="*", color="gold", edgecolor="k", s=250,
               zorder=8, lw=0.9)
    ax.annotate("shared\nstart (same init)", xy=(0, 0), xytext=(0.04, 0.04),
                fontsize=6.5, color="k",
                arrowprops=dict(arrowstyle="->", color="k", lw=0.7))

    # axis labels and annotations
    ax.set_xlabel(r"$\alpha$  — DFA's direction in weight space  (normalised)",
                  fontsize=8)
    ax.set_ylabel(r"$\beta$  — BP's direction in weight space  (normalised)",
                  fontsize=8)
    ax.set_title("Real loss landscape: DFA vs BP take\ndifferent paths in weight space",
                 fontsize=8.5)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5, ncol=1)

    # annotation: DFA stall region on the plot
    stall_pts = dfa_proj[dfa_stall]
    if len(stall_pts) > 1:
        cx, cy = stall_pts[:,0].mean(), stall_pts[:,1].mean()
        ax.annotate("DFA stalled here\n(~350 steps, loss flat)",
                    xy=(cx, cy), xytext=(cx+0.07, cy-0.08),
                    fontsize=6.5, color=STALL_COL,
                    arrowprops=dict(arrowstyle="->", color=STALL_COL, lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=STALL_COL, alpha=0.85, lw=0.8))

    # annotation: BP at same time
    bp_stall_pts = bp_proj[bp_stall]
    if len(bp_stall_pts) > 1:
        bx, by = bp_stall_pts[:,0].mean(), bp_stall_pts[:,1].mean()
        ax.annotate("BP here at\nthe same steps",
                    xy=(bx, by), xytext=(bx-0.12, by+0.06),
                    fontsize=6.5, color="#cc4400",
                    arrowprops=dict(arrowstyle="->", color="#cc4400", lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#cc4400", alpha=0.85, lw=0.8))
    clean(ax)
    ax.text(0.02, 0.97, "a", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel B: loss vs step for both models (from actual checkpoint evaluations)
    # ─────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.axvspan(STALL_S, STALL_E, color=STALL_COL, alpha=0.12, lw=0,
                label="DFA stall window")
    ax2.semilogy(dfa_steps, dfa_losses, color=DFA_COL, lw=1.8,
                 marker="o", ms=4, label="DFA loss")
    ax2.semilogy(bp_steps,  bp_losses,  color=BP_COL,  lw=1.8,
                 marker="s", ms=4, label="BP loss")

    ax2.set_xlabel("Training step"); ax2.set_ylabel("Validation loss (log)")
    ax2.set_title("Loss at checkpoints\n(same steps on landscape panel A)")
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_xlim(0, 3100)
    clean(ax2)
    ax2.text(0.02, 0.97, "b", transform=ax2.transAxes,
             fontsize=11, fontweight="bold", va="top")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel C: distance traveled in each direction (shows DFA = mostly α, BP = mostly β)
    # ─────────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    ax3.axvspan(STALL_S, STALL_E, color=STALL_COL, alpha=0.12, lw=0)
    ax3.plot(dfa_steps, dfa_proj[:,0], color=DFA_COL, lw=1.6,
             label=r"DFA  $\alpha$ (its own direction)")
    ax3.plot(dfa_steps, dfa_proj[:,1], color=DFA_COL, lw=1.0, ls="--", alpha=0.55,
             label=r"DFA  $\beta$ (BP direction)")
    ax3.plot(bp_steps,  bp_proj[:,0],  color=BP_COL,  lw=1.0, ls="--", alpha=0.55,
             label=r"BP   $\alpha$ (DFA direction)")
    ax3.plot(bp_steps,  bp_proj[:,1],  color=BP_COL,  lw=1.6,
             label=r"BP   $\beta$ (its own direction)")
    ax3.axhline(0, color="0.7", lw=0.6, ls=":")

    ax3.set_xlabel("Training step")
    ax3.set_ylabel("Normalised displacement")
    ax3.set_title("Paths are nearly orthogonal\n"
                  "DFA travels along α,  BP travels along β")
    ax3.legend(frameon=False, fontsize=6, ncol=2, loc="center right")
    ax3.set_xlim(0, 3100)
    clean(ax3)
    ax3.text(0.02, 0.97, "c", transform=ax3.transAxes,
             fontsize=11, fontweight="bold", va="top")

    fig.suptitle(
        "Real loss landscape  —  DFA and BP navigate to different parts of weight space\n"
        "Same initialisation, same data, but DFA (random feedback) finds a different path than BP (true gradient)",
        fontsize=9,
    )

    for ext in ("png", "svg"):
        fig.savefig(OUTDIR / f"fig7_loss_landscape.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/fig7_loss_landscape.png / .svg")

    # save updated npz
    np.savez(OUTDIR / "landscape_data.npz",
             AG=AG, BG=BG, LG=LG,
             dfa_proj=dfa_proj, dfa_steps=np.array(dfa_steps),
             bp_proj=bp_proj,   bp_steps=np.array(bp_steps),
             dfa_losses=dfa_losses, bp_losses=bp_losses)


if __name__ == "__main__":
    main()
