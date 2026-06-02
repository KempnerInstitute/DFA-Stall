#!/usr/bin/env python3
"""
Generate the same figures as:
  final_clean_story_figures/figure_1_gate_participation_diagnostic.*
  final_clean_story_figures/figure_2_participation_state_variable.*
  order_params_study_paper_standard/paper_figures/candidate_axes/candidate_*.*

...but using DFA-STALL/metrics.csv from a single training run.

Column mapping from DFA-STALL → original figure scripts:
  gate_participation_l{N}  →  gate_participation_unit_l{N}
  saturation_frac_l{N}     →  gate_saturation_frac_low_g_l{N}
  dfa_loss                 →  train_loss
  grad_alignment           →  param_grad_alignment_l{N}  (global; used for all layers)
  feature_movement_l{N}    →  feature_path_l{N}
  gate_strength_sq_l{N}    →  same
  feedback_signal_norm_l{N}→  same
  angular_update_l{N}      →  same  (used as "angular update strength" proxy)

Saves all figures to ./figures/ as PNG + SVG + PDF.
"""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE   = Path(__file__).resolve().parent
OUTDIR = HERE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── inline research-style helpers (no external dependencies) ─────────────────

def set_research_style() -> None:
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         8,
        "axes.labelsize":    8,
        "axes.titlesize":    8.5,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "axes.linewidth":    0.75,
        "lines.linewidth":   1.35,
        "figure.dpi":        150,
    })

def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def add_panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.02, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")

def smooth_ema(x: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[0] = float(x[0])
    for i in range(1, len(x)):
        out[i] = alpha * float(x[i]) + (1.0 - alpha) * out[i - 1]
    return out

STALL_COLOR = "#e8501a"
LI = 1   # representative layer (L1)


# ── load & normalise ──────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(HERE / "metrics.csv").sort_values("step")
    # canonical alias: train_loss = dfa_loss
    df["train_loss"] = df["dfa_loss"]
    # interpolate sparse columns (logged every 50 steps)
    for col in df.columns:
        if col.startswith("selected_loss") or col.startswith("residual_loss") \
                or col.startswith("feature_path") or col == "val_loss":
            df[col] = df[col].interpolate(limit_direction="both")
    return df


# ── helpers ───────────────────────────────────────────────────────────────────

def sm(x, a=0.06): return smooth_ema(np.asarray(x, float), a)

STALL_LO, STALL_HI = None, None   # set after detect_stall

def detect_stall(loss: np.ndarray) -> tuple[int, int]:
    n = len(loss)
    w = min(100, n)
    sv = np.convolve(loss, np.ones(w)/w, mode="same")
    vel = -np.gradient(sv)
    skip = min(50, n//4)
    early = vel[skip:skip+min(200, n-skip)]
    if not len(early) or np.nanmax(early) <= 0:
        return n//4, 3*n//4
    thresh = 0.05 * np.nanmax(early)
    idxs = np.where((vel < thresh) & (np.arange(n) >= skip))[0]
    if not len(idxs): return n//4, 3*n//4
    s = int(idxs[0])
    rec = np.where((vel > thresh) & (np.arange(n) > s + min(50, n//10)))[0]
    e = int(rec[0]) if len(rec) else n-1
    return s, e

def stall_mask(steps): return (steps >= STALL_LO) & (steps <= STALL_HI)

def step_colors(steps):
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=steps.min(), vmax=steps.max())
    return cmap(norm(steps)), cmap, norm

def scatter_stall(ax, x, y, steps, colors, s=9, **kw):
    """Viridis circles outside stall; orange-red squares inside stall."""
    m = stall_mask(steps)
    kw.pop("c", None); kw.pop("color", None)
    if (~m).any():
        ax.scatter(x[~m], y[~m], c=colors[~m], s=s, marker="o", linewidths=0, **kw)
    if m.any():
        kw.pop("alpha", None)
        ax.scatter(x[m], y[m], color=STALL_COLOR, s=s*3.5,
                   marker="s", linewidths=0, alpha=0.90, zorder=4, **kw)

def ends(ax, x, y):
    ax.scatter(x[0],  y[0],  marker="x", color="#222", s=22, linewidths=1.0, zorder=5)
    ax.scatter(x[-1], y[-1], marker="o", facecolor="none",
               edgecolor="#222", s=22, linewidths=1.0, zorder=5)

def add_cb(fig, cmap, norm):
    sm2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm2.set_array([])
    cax = fig.add_axes([0.915, 0.24, 0.014, 0.58])
    fig.colorbar(sm2, cax=cax).set_label("Training step")

def save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {stem}.png/svg/pdf")


# ── figure 1: gate participation diagnostic ───────────────────────────────────
# (identical layout to final_clean_story_figures/figure_1_gate_participation_diagnostic)
#  x-axis = gate participation;  y-axes = gate strength, saturation, feedback norm

def figure_1_gate_diagnostic(df: pd.DataFrame) -> None:
    steps  = df["step"].to_numpy()
    colors, cmap, norm = step_colors(steps)
    layer_id = LI

    specs = [
        (f"gate_strength_sq_l{layer_id}",        "Gate strength",       "linear"),
        (f"saturation_frac_l{layer_id}",         "Saturation fraction", "symlog"),
        (f"feedback_signal_norm_l{layer_id}",    "Feedback signal norm","linear"),
    ]
    p_col = f"gate_participation_unit_l{layer_id}"
    p     = df[p_col].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for ax, (col, ylabel, scale), lbl in zip(axes, specs, ["a", "b", "c"]):
        y = df[col].to_numpy()
        scatter_stall(ax, p, y, steps, colors, s=5, alpha=0.9)
        ends(ax, p, y)
        ax.set_xlabel("Gate participation")
        ax.set_ylabel(ylabel)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylim(-2e-5, 5e-2)
        add_panel_label(ax, lbl)
        clean_axis(ax)

    fig.subplots_adjust(left=0.08, right=0.875, bottom=0.24, top=0.92, wspace=0.58)
    add_cb(fig, cmap, norm)
    save(fig, "figure_1_gate_participation_diagnostic")


# ── figure 2: participation state variable ────────────────────────────────────
# (identical layout to final_clean_story_figures/figure_2_participation_state_variable)
#  x-axis = gate participation;  y-axes = train loss (log), grad alignment

def figure_2_participation_state(df: pd.DataFrame) -> None:
    steps  = df["step"].to_numpy()
    colors, cmap, norm = step_colors(steps)
    layer_id = LI

    p_col = f"gate_participation_unit_l{layer_id}"
    p     = df[p_col].to_numpy()
    loss  = df["train_loss"].to_numpy()
    ga    = df[f"param_grad_alignment_l{layer_id}"].fillna(df["grad_alignment"]).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45))
    scatter_stall(axes[0], p, loss, steps, colors, s=5)
    ends(axes[0], p, loss)
    axes[0].set_xlabel("Gate participation")
    axes[0].set_ylabel("Train loss")
    axes[0].set_yscale("log")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])

    scatter_stall(axes[1], p, ga, steps, colors, s=5)
    ends(axes[1], p, ga)
    axes[1].set_xlabel("Gate participation")
    axes[1].set_ylabel("Gradient alignment")
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    fig.subplots_adjust(left=0.09, right=0.87, bottom=0.24, top=0.92, wspace=0.35)
    add_cb(fig, cmap, norm)
    save(fig, "figure_2_participation_state_variable")


# ── candidate axes ────────────────────────────────────────────────────────────

def _state(df: pd.DataFrame) -> dict:
    """Compute run-relative state variables for candidate_axes figures."""
    step = df["step"].to_numpy()

    # gate participation
    p = sm(df[f"gate_participation_unit_l{LI}"])

    # feedback selectivity: normalised drop in teacher_reff from early baseline
    reff      = sm(df[f"teacher_reff_l{LI}"])
    reff_early = float(np.nanmedian(reff[step <= 80]))
    sel = np.clip(1.0 - reff / (reff_early + 1e-12), 0.0, 1.0)

    # feature movement: normalised cumulative feature path
    fp    = sm(df[f"feature_path_l{LI}"])
    fp_n  = fp / (np.nanmax(fp) + 1e-12)

    # feature catch-up: post-rapid-phase increase, normalised to [0,1]
    idx100  = int(np.searchsorted(step, 100))
    fp_base = float(fp[idx100]) if idx100 < len(fp) else 0.0
    fp_max  = float(np.nanmax(fp))
    catchup = np.clip((fp - fp_base) / (fp_max - fp_base + 1e-12), 0.0, 1.05)

    # bottleneck gap: selected − residual virtual-step loss response
    sel_resp = sm(df[f"selected_loss_response_l{LI}"].to_numpy())
    res_resp = sm(df[f"residual_loss_response_l{LI}"].to_numpy())
    gap = sel_resp - res_resp

    # per-layer gradient alignment
    ga   = sm(df[f"param_grad_alignment_l{LI}"])

    # val_loss (interpolated)
    loss = np.clip(sm(df["val_loss"].to_numpy()), 0.05, None)

    # gate anisotropy
    gate_cv = sm(df[f"gate_cv_l{LI}"])

    return dict(step=step, participation=p, selectivity=sel,
                feature=fp_n, catchup=catchup, gap=gap, ga=ga,
                loss=loss, gate_cv=gate_cv)


def _scatter_candidate(ax, x, y, steps, colors, log_y=False):
    scatter_stall(ax, x, y, steps, colors, s=9, alpha=0.92)
    ends(ax, x, y)
    if log_y:
        ax.set_yscale("log")


# ── candidate_bottleneck_gap ──────────────────────────────────────────────────

def candidate_bottleneck_gap(df: pd.DataFrame) -> None:
    """
    Panel a: training step vs normalised selectivity + feature movement.
    Panel b: bottleneck proxy (angular update) vs gradient alignment.
    Panel c: bottleneck proxy vs training loss.
    Note: 'bottleneck gap' here uses normalised angular update strength as a
          proxy for selected_loss_response − residual_loss_response (not tracked
          in this run).
    """
    steps  = df["step"].to_numpy()
    colors, cmap, norm = step_colors(steps)
    st = _state(df)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    # (a) time series
    ax = axes[0]
    ax.plot(st["step"], st["selectivity"], color="#d62728", lw=1.2,
            label="feedback selectivity")
    ax.plot(st["step"], st["feature"],    color="#1f77b4", lw=1.2,
            label="feature movement")
    ax.axvspan(STALL_LO, STALL_HI, color=STALL_COLOR, alpha=0.10, lw=0)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalized state")
    ax.legend(frameon=False, fontsize=6)
    add_panel_label(ax, "a")
    clean_axis(ax)

    # (b) bottleneck gap vs gradient alignment
    _scatter_candidate(axes[1], st["gap"], st["ga"], steps, colors)
    axes[1].set_xlabel("Bottleneck gap\n(norm. angular update)")
    axes[1].set_ylabel("Gradient alignment")
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    # (c) bottleneck gap vs loss
    _scatter_candidate(axes[2], st["gap"], st["loss"], steps, colors, log_y=True)
    axes[2].set_xlabel("Bottleneck gap\n(norm. angular update)")
    axes[2].set_ylabel("Validation loss")
    add_panel_label(axes[2], "c")
    clean_axis(axes[2])

    fig.subplots_adjust(left=0.08, right=0.875, bottom=0.24, top=0.92, wspace=0.58)
    add_cb(fig, cmap, norm)
    save(fig, "candidate_bottleneck_gap")


# ── candidate_feature_loss_alignment ─────────────────────────────────────────

def candidate_feature_loss_alignment(df: pd.DataFrame) -> None:
    steps  = df["step"].to_numpy()
    colors, cmap, norm = step_colors(steps)
    st = _state(df)

    specs = [
        (st["feature"], st["ga"],      r"Feature movement $F_\ell(t)$", "Gradient alignment",      False, "a"),
        (st["feature"], st["loss"],    r"Feature movement $F_\ell(t)$", "Validation loss",          True,  "b"),
        (st["feature"], st["gate_cv"], r"Feature movement $F_\ell(t)$", "Angular update strength",  False, "c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for ax, (x, y, xl, yl, log_y, lbl) in zip(axes, specs):
        _scatter_candidate(ax, x, y, steps, colors, log_y=log_y)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        add_panel_label(ax, lbl); clean_axis(ax)

    fig.subplots_adjust(left=0.08, right=0.875, bottom=0.24, top=0.92, wspace=0.54)
    add_cb(fig, cmap, norm)
    save(fig, "candidate_feature_loss_alignment")


# ── candidate_selectivity_feature_catchup ────────────────────────────────────

def candidate_selectivity_feature_catchup(df: pd.DataFrame) -> None:
    steps  = df["step"].to_numpy()
    colors, cmap, norm = step_colors(steps)
    st = _state(df)

    specs = [
        (st["selectivity"], st["feature"], "Feedback selectivity", "Feature movement",   False, "a"),
        (st["catchup"],     st["ga"],      "Feature catch-up",     "Gradient alignment", False, "b"),
        (st["catchup"],     st["loss"],    "Feature catch-up",     "Validation loss",    True,  "c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for ax, (x, y, xl, yl, log_y, lbl) in zip(axes, specs):
        _scatter_candidate(ax, x, y, steps, colors, log_y=log_y)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        add_panel_label(ax, lbl); clean_axis(ax)

    fig.subplots_adjust(left=0.08, right=0.875, bottom=0.24, top=0.92, wspace=0.54)
    add_cb(fig, cmap, norm)
    save(fig, "candidate_selectivity_feature_catchup")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_research_style()
    df = load()

    # detect stall
    global STALL_LO, STALL_HI
    steps = df["step"].to_numpy()
    si, ei = detect_stall(df["dfa_loss"].to_numpy())
    STALL_LO = float(steps[si])
    STALL_HI = float(steps[min(ei, len(steps)-1)])
    print(f"Stall: steps {STALL_LO:.0f}–{STALL_HI:.0f}")

    print("Generating figures …")
    figure_1_gate_diagnostic(df)
    figure_2_participation_state(df)
    candidate_bottleneck_gap(df)
    candidate_feature_loss_alignment(df)
    candidate_selectivity_feature_catchup(df)

    print(f"\nAll figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
