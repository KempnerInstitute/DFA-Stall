#!/usr/bin/env python3
"""
animate_training.py — Real-data animated visualization of the DFA stall.

ALL panels use metrics.csv from the actual MNIST 3×300 training run.
Nothing is simulated or designed.

Six panels, all drawing progressively as training steps advance:

  A. (Loss, Activation rank) phase scatter — accumulating dots coloured by step.
     Shows the lazy-regime collapse (rank→1, loss stays high) then the escape
     (rank explodes, loss drops).  Most intuitive summary of the whole story.

  B. Loss curves (DFA + BP) — drawn step by step.

  C. Activation rank per layer — drawn step by step.

  D. Phase portrait (act_eff_rank_l2 vs grad_alignment) — accumulating scatter.
     Shows the orbit during the stall, then the escape trajectory.

  E. Teacher effective rank (DFA solid, BP dashed) — drawn step by step.
     DFA rank stays flat during stall — confirms it is NOT a rank-collapse story.

  F. Gate anisotropy CV_D per layer — drawn step by step.
     Rises during stall, showing the gate bottleneck.

Output
------
  figures/fig6_training_animation.gif  (300 frames, 8 fps ≈ 37 s)
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.animation as animation
import matplotlib.colors as mcolors

HERE   = Path(__file__).resolve().parent
OUTDIR = HERE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

STALL_S   = 118
STALL_E   = 451
LC        = ["#1f77b4", "#ff7f0e", "#2ca02c"]
STALL_COL = "#e8501a"
DFA_COL   = "#1f77b4"
BP_COL    = "#d62728"

# ── load & smooth ─────────────────────────────────────────────────────────────
df = pd.read_csv(HERE / "metrics.csv").sort_values("step").reset_index(drop=True)
print(f"Loaded {len(df)} steps × {len(df.columns)} columns  (real MNIST data)")

def ema(x, a=0.06):
    o = np.empty_like(np.asarray(x, float)); o[0] = x[0]
    for i in range(1, len(x)): o[i] = a*x[i] + (1-a)*o[i-1]
    return o

steps    = df["step"].to_numpy()
SUBSAMP  = 10
n_frames = len(steps[::SUBSAMP])
print(f"Frames: {n_frames}  @8 fps ≈ {n_frames/8:.0f}s")

dfa_loss = ema(df["dfa_loss"])
bp_loss  = ema(df["bp_loss"])
grad_al  = ema(df["grad_alignment"])
ga_l2    = ema(df["param_grad_alignment_l2"])

act_r  = {li+1: ema(df[f"act_eff_rank_l{li+1}"])       for li in range(3)}
dfa_r  = {li+1: ema(df[f"teach_eff_rank_l{li+1}"])     for li in range(3)}
bp_r   = {li+1: ema(df[f"bp_teach_eff_rank_l{li+1}"]) for li in range(3)}
gate_cv = {li+1: ema(df[f"gate_cv_l{li+1}"])           for li in range(3)}

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2,
})

fig = plt.figure(figsize=(14, 9))
gs  = mgridspec.GridSpec(2, 3, figure=fig,
                          left=0.07, right=0.97, bottom=0.09, top=0.90,
                          wspace=0.40, hspace=0.50)

axA = fig.add_subplot(gs[0, 0])   # (loss, act_rank) phase scatter
axB = fig.add_subplot(gs[0, 1])   # loss curves
axC = fig.add_subplot(gs[0, 2])   # activation rank
axD = fig.add_subplot(gs[1, 0])   # phase portrait (rank vs alignment)
axE = fig.add_subplot(gs[1, 1])   # teacher rank
axF = fig.add_subplot(gs[1, 2])   # gate anisotropy

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# axis limits (pre-compute from full data)
loss_range = (min(dfa_loss.min(), bp_loss.min()) * 0.8,
              dfa_loss.max() * 1.05)
act_range  = (0, max(act_r[li+1].max() for li in range(3)) * 1.05)
dfa_r_range = (0, max(max(dfa_r[li+1].max(), bp_r[li+1].max()) for li in range(3)) * 1.05)
cv_range   = (0, max(gate_cv[li+1].max() for li in range(3)) * 1.05)

# stall shading helper (for step-axis panels)
def shade(ax):
    ax.axvspan(STALL_S, STALL_E, color=STALL_COL, alpha=0.10, lw=0)

# ── panel A setup: (loss × act_rank) phase scatter ───────────────────────────
axA.set_xlabel("Activation eff. rank  L2")
axA.set_ylabel("DFA loss")
axA.set_title("Loss vs activation rank\n(rank collapses → stall; rank expands → recovery)")
axA.set_xlim(act_range[0], act_r[2].max() * 1.08)
axA.set_ylim(loss_range[0] * 0.9, loss_range[1])
axA.set_yscale("log")
# reference labels
axA.text(0.5, 0.95, "← rank-1 collapse (stall)", transform=axA.transAxes,
         ha="center", va="top", fontsize=6, color=STALL_COL)
axA.text(0.70, 0.20, "recovery →", transform=axA.transAxes,
         ha="left", va="bottom", fontsize=6, color=DFA_COL)
clean(axA)

# ── panel B setup: loss curves ────────────────────────────────────────────────
shade(axB)
axB.set_xlabel("Training step"); axB.set_ylabel("Loss")
axB.set_title("Loss curves  (DFA vs BP)\nstall = orange window")
axB.set_xlim(0, steps[-1]); axB.set_ylim(*loss_range)
axB.set_yscale("log")
# legend proxies
from matplotlib.lines import Line2D
axB.legend(handles=[Line2D([],[],color=DFA_COL,label="DFA"),
                    Line2D([],[],color=BP_COL,ls="--",label="BP")],
           frameon=False, loc="upper right")
clean(axB)

# ── panel C setup: activation rank ────────────────────────────────────────────
shade(axC)
axC.set_xlabel("Training step"); axC.set_ylabel(r"Activation eff. rank  $r^h_\ell$")
axC.set_title("Activation rank\nrank-1 collapse at step 50 → +266% at recovery")
axC.set_xlim(0, steps[-1]); axC.set_ylim(*act_range)
axC.legend(handles=[Line2D([],[],color=c,label=f"L{i+1}") for i,c in enumerate(LC)],
           frameon=False, loc="upper right")
clean(axC)

# ── panel D setup: (act_rank, grad_align) phase portrait ─────────────────────
axD.axhline(0, color="0.65", lw=0.6, ls=":")
axD.set_xlabel(r"Activation eff. rank  $r^h_{L2}$")
axD.set_ylabel("Grad alignment  L2")
axD.set_title("Phase portrait: rank vs alignment\n(orbit = stall; diagonal sweep = recovery)")
axD.set_xlim(act_r[2].min() - 0.3, act_r[2].max() * 1.05)
axD.set_ylim(ga_l2.min() - 0.05, ga_l2.max() + 0.05)
clean(axD)

# ── panel E setup: teacher rank ───────────────────────────────────────────────
shade(axE)
axE.set_xlabel("Training step"); axE.set_ylabel(r"Teacher eff. rank  $r^\delta_\ell$")
axE.set_title("Teacher rank  (solid=DFA, dashed=BP)\nDFA rank FLAT during stall — not a rank-collapse")
axE.set_xlim(0, steps[-1]); axE.set_ylim(*dfa_r_range)
axE.legend(handles=[Line2D([],[],color=c,label=f"L{i+1}") for i,c in enumerate(LC)],
           frameon=False, loc="upper left")
clean(axE)

# ── panel F setup: gate anisotropy ────────────────────────────────────────────
shade(axF)
axF.set_xlabel("Training step"); axF.set_ylabel(r"Gate anisotropy  $\mathrm{CV}_D$")
axF.set_title("Gate anisotropy rises during stall\n(derivative gate becomes unequal across units)")
axF.set_xlim(0, steps[-1]); axF.set_ylim(*cv_range)
axF.legend(handles=[Line2D([],[],color=c,label=f"L{i+1}") for i,c in enumerate(LC)],
           frameon=False, loc="upper right")
clean(axF)

# ── animated line artists (one per series per panel) ─────────────────────────
line_dfa_B, = axB.plot([], [], color=DFA_COL, lw=1.3)
line_bp_B,  = axB.plot([], [], color=BP_COL,  lw=1.0, ls="--")

lines_C = [axC.plot([], [], color=LC[li], lw=1.1)[0] for li in range(3)]

lines_dfa_E = [axE.plot([], [], color=LC[li], lw=1.1)[0]       for li in range(3)]
lines_bp_E  = [axE.plot([], [], color=LC[li], lw=0.8, ls="--", alpha=0.5)[0] for li in range(3)]

lines_F = [axF.plot([], [], color=LC[li], lw=1.1)[0] for li in range(3)]

# scatters
cmap_v = plt.get_cmap("viridis")
norm_v = mcolors.Normalize(vmin=0, vmax=steps[-1])

scat_A = axA.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=steps[-1],
                     s=10, linewidths=0, alpha=0.80, zorder=3)
dot_A,  = axA.plot([], [], "o", ms=8, zorder=5)

scat_D = axD.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=steps[-1],
                     s=10, linewidths=0, alpha=0.80, zorder=3)
dot_D,  = axD.plot([], [], "o", ms=8, zorder=5)

# colourbar shared for both scatters
sm_cb = plt.cm.ScalarMappable(cmap="viridis", norm=norm_v); sm_cb.set_array([])
cb = fig.colorbar(sm_cb, ax=[axA, axD], pad=0.01, fraction=0.018, label="Training step")

# header text
step_txt  = fig.text(0.50, 0.960, "Step 0", ha="center", va="top",
                     fontsize=12, fontweight="bold")
phase_txt = fig.text(0.50, 0.938, "", ha="center", va="top", fontsize=9)
fig.text(0.50, 0.982, "DFA Stall on MNIST  —  Real Training Dynamics",
         ha="center", va="top", fontsize=11.5, fontweight="bold")

# ── init ──────────────────────────────────────────────────────────────────────
def init():
    for ln in ([line_dfa_B, line_bp_B] + lines_C +
               lines_dfa_E + lines_bp_E + lines_F + [dot_A, dot_D]):
        ln.set_data([], [])
    scat_A.set_offsets(np.empty((0, 2)))
    scat_A.set_array(np.empty(0))
    scat_D.set_offsets(np.empty((0, 2)))
    scat_D.set_array(np.empty(0))
    return []

# ── update ────────────────────────────────────────────────────────────────────
def update(frame_idx):
    idx  = frame_idx * SUBSAMP
    s_now = steps[idx]
    s_sl  = slice(0, idx + 1)
    x_sl  = steps[s_sl]

    # phase
    if s_now < STALL_S:
        phase_str = "PHASE 1 — Rapid descent  (high rank, loss drops fast)"
        pcol = "#2ecc71"
    elif s_now <= STALL_E:
        phase_str = f"PHASE 2 — STALL  (rank-1, grad-alignment ≈ 0, loss plateau)"
        pcol = STALL_COL
    else:
        phase_str = "PHASE 3 — Recovery  (rank expands, alignment rises, loss resumes)"
        pcol = DFA_COL

    step_txt.set_text(f"Step  {s_now:4d} / {steps[-1]}")
    phase_txt.set_text(phase_str); phase_txt.set_color(pcol)

    # B: loss curves
    line_dfa_B.set_data(x_sl, dfa_loss[s_sl])
    line_bp_B.set_data(x_sl,  bp_loss[s_sl])

    # C: activation rank
    for li in range(3):
        lines_C[li].set_data(x_sl, act_r[li+1][s_sl])

    # E: teacher rank
    for li in range(3):
        lines_dfa_E[li].set_data(x_sl, dfa_r[li+1][s_sl])
        lines_bp_E[li].set_data(x_sl,  bp_r[li+1][s_sl])

    # F: gate anisotropy
    for li in range(3):
        lines_F[li].set_data(x_sl, gate_cv[li+1][s_sl])

    # A: (loss × act_rank_l2) scatter
    x_a = act_r[2][s_sl]; y_a = dfa_loss[s_sl]; c_a = x_sl
    scat_A.set_offsets(np.column_stack([x_a, y_a]))
    scat_A.set_array(c_a)
    dot_A.set_data([act_r[2][idx]], [dfa_loss[idx]])
    dot_A.set_color(pcol); dot_A.set_markeredgecolor("k"); dot_A.set_markeredgewidth(0.5)

    # D: (act_rank_l2 × grad_align) scatter
    x_d = act_r[2][s_sl]; y_d = ga_l2[s_sl]; c_d = x_sl
    scat_D.set_offsets(np.column_stack([x_d, y_d]))
    scat_D.set_array(c_d)
    dot_D.set_data([act_r[2][idx]], [ga_l2[idx]])
    dot_D.set_color(pcol); dot_D.set_markeredgecolor("k"); dot_D.set_markeredgewidth(0.5)

    return []

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=125,   # 8 fps
    blit=False,
)

out = OUTDIR / "fig6_training_animation.gif"
print(f"Rendering {n_frames} frames at 8 fps …")
ani.save(str(out), writer=animation.PillowWriter(fps=8), dpi=110)
plt.close(fig)
print(f"Saved: {out}  ({out.stat().st_size/1024:.0f} KB)")
