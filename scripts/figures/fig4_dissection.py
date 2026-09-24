#!/usr/bin/env python3
"""Figure 4, the causal dissection: the 2x2 of input centering against frozen hidden biases, the three controls that
separate the common-mode term from the feedback gain (centered teaching signal, pre-aligned feedback, BP with output
weights scaled 12x), and the per-unit drift against each unit's fixed common-mode coefficient (B_l 1)_i.
Reads results/E1_headline (trajectories and snapshots).

Usage: PYTHONPATH=src python3 scripts/figures/fig4_dissection.py [--results results]
"""
from __future__ import annotations
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import A, S

EXP = "E1_headline"
NAME = "fig4_dissection"
DISSECTION = [("base", "baseline"), ("pixcenter", "centered inputs"), ("frzbias", "frozen biases"),
              ("pixcenter_frzbias", "both interventions")]
CONTROLS = [("center_e", "centered error $e$"), ("aligned", "pre-aligned DFA"),
            ("bp_outx12", r"BP, $W_{\rm out}\times 12$")]
XMAX = 800


def panel_cos(ax, d, tag, title, letter):
    """Hidden cosine per layer for one condition, over the collapse window."""
    n = d.layers(EXP, tag)
    if not n:
        S.skip(f"fig4 {tag}", f"{EXP}/{tag} has no summarized runs"); S.empty(ax, f"{tag}\nmissing")
        S.panel_title(ax, letter, title); return
    for l in range(1, n + 1):
        t = d.traj(EXP, tag, f"cos_l{l}")
        if t is None: continue
        s, m, sd = t
        keep = s <= XMAX
        ax.plot(s[keep], m[keep], color=S.color(l - 1), zorder=3)
        S.band(ax, s[keep], m[keep], sd[keep], S.color(l - 1))
    ax.set_ylim(0, 1.05); ax.set_xlim(0, XMAX); ax.set_xlabel("step")
    S.panel_title(ax, letter, title)


def panel_drift(ax, d, letter):
    """Per-unit mean-preactivation drift against the fixed common-mode coefficient (B_l 1)_i."""
    if not d.has(EXP, "base"):
        S.skip("fig4h drift", f"{EXP}/base has no runs"); S.empty(ax, "no snapshots"); S.panel_title(ax, letter, "per-unit drift"); return
    n = d.layers(EXP, "base") or 3
    rhos, drawn = [], 0
    for l in range(1, n + 1):
        r = A.drift_scatter(d.exp_dir(EXP), "base", l, step=120)
        if r is None: continue
        b1, dmu, step = r
        ax.plot(b1, dmu, ".", color=S.color(l - 1), ms=2.0, alpha=0.6, ls="none")
        rho = A.drift_spearman(d.exp_dir(EXP), "base", l, step=120)
        rhos.append(f"L{l} {rho:+.2f}"); drawn += 1
    if not drawn:
        S.skip("fig4h drift", "no snapshots.npz with mu and B1 arrays"); S.empty(ax, "no snapshots")
        S.panel_title(ax, letter, "per-unit drift"); return
    ax.axhline(0, color=S.MUTED, lw=0.7, zorder=1); ax.axvline(0, color=S.MUTED, lw=0.7, zorder=1)
    ax.set_xlabel(r"$(B_\ell \mathbf{1})_i$"); ax.set_ylabel(r"$\Delta\mu_i$ to step %d" % step)
    S.panel_title(ax, letter, "per-unit drift")


def main():
    args = F.parse(__doc__)
    S.use_style()
    d = F.Data(args.results)
    if not d.has(EXP):
        S.skip("figure 4", f"{os.path.join(args.results, EXP)} does not exist")
    fig, axes = S.plt.subplots(2, 4, figsize=(S.FULL_WIDTH, 3.5))
    letters = iter("abcdefgh")
    for ax, (tag, title) in zip(axes[0], DISSECTION):
        panel_cos(ax, d, tag, title, next(letters))
    axes[0, 0].set_ylabel("hidden cosine")
    for ax, (tag, title) in zip(axes[1], CONTROLS):
        panel_cos(ax, d, tag, title, next(letters))
    axes[1, 0].set_ylabel("hidden cosine")
    panel_drift(axes[1, 3], d, next(letters))
    nl = d.layers(EXP, "base") or 3
    S.rule_legend(fig, [(f"layer {l}", dict(color=S.color(l - 1), lw=S.LW)) for l in range(1, nl + 1)], y=-0.04)
    fig.subplots_adjust(wspace=0.42, hspace=0.62)
    S.save(fig, NAME, root=args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
