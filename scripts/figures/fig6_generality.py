#!/usr/bin/env python3
"""Figure 6, generality and demarcation: for every setting of results/E3_generality, the hidden cosine of the deepest
layer under DFA, under DFA with a centered teaching signal, and under the matched BP twin. Tags are '<setting>',
'<setting>_centere' (output error centered before projection; '<setting>_center' centers the teaching signal and is not drawn) and '<setting>_bp'; settings with no DFA run are skipped.

Usage: PYTHONPATH=src python3 scripts/figures/fig6_generality.py [--results results]
"""
from __future__ import annotations
import math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import A, S

EXP = "E3_generality"
NAME = "fig6_generality"
SUFFIXES = ("_bp", "_center", "_centere")
VARIANTS = [("", "plain DFA / FA", 0, "-"), ("_centere", r"centered error $e$", 2, "-"), ("_bp", "matched BP", 1, (0, (3, 2)))]
NICE = {"depth6": "depth 6", "cnn": "small CNN", "relu": "ReLU", "gelu": "GELU", "linear": "linear units",
        "mse_s1": "MSE, targets 1", "mse_s5": "MSE, targets 5", "mse_s1_ctr": r"MSE, centered $y$",
        "multilabel": "multi-label head", "fashion": "Fashion-MNIST", "fashion_softmax": "Fashion, softmax",
        "fa": "layerwise FA", "fa_depth6": "FA, depth 6", "relu_lr3e4": r"ReLU, $\eta=3\cdot10^{-4}$",
        "w900_depth6": "width 900, depth 6"}


def settings(d):
    """Base setting names: tags that are not themselves a _bp or _center variant."""
    return [t for t in d.tags(EXP) if not t.endswith(SUFFIXES)]


def panel(ax, d, setting, letter):
    """Deepest-layer hidden cosine for the three variants of one setting."""
    drawn = 0
    for suffix, _, ci, ls in VARIANTS:
        tag = setting + suffix
        n = d.layers(EXP, tag)
        if not n: continue
        t = d.traj(EXP, tag, f"cos_l{n}")
        if t is None: continue
        s, m, sd = t
        ax.plot(s, m, color=S.color(ci), ls=ls, lw=S.LW if suffix != "_bp" else 1.1, zorder=3)
        if suffix == "": S.band(ax, s, m, sd, S.color(ci))
        drawn += 1
    if not drawn:
        S.skip(f"fig6 {setting}", "no runs for any variant"); S.empty(ax, "no runs")
        S.panel_title(ax, letter, NICE.get(setting, setting)); return 0
    ax.set_ylim(0, 1.05)
    S.panel_title(ax, letter, NICE.get(setting, setting))
    return drawn


def main():
    args = F.parse(__doc__)
    S.use_style()
    d = F.Data(args.results)
    if not d.has(EXP):
        S.skip("figure 6", f"{os.path.join(args.results, EXP)} does not exist")
    names = settings(d)
    if not names:
        S.skip("figure 6", f"{EXP} has no summarized settings"); names = []
    ncol = 4
    nrow = max(1, math.ceil(max(len(names), 1) / ncol))
    fig, axes = S.plt.subplots(nrow, ncol, figsize=(S.FULL_WIDTH, 1.35 * nrow + 0.5), squeeze=False)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, ax in enumerate(axes.flat):
        if i >= len(names): ax.set_axis_off(); continue
        panel(ax, d, names[i], letters[i % len(letters)])
        if i % ncol == 0: ax.set_ylabel("hidden cosine")
        if i >= len(names) - ncol: ax.set_xlabel("step")
    S.rule_legend(fig, [(lab, dict(color=S.color(ci), ls=ls, lw=S.LW if suf != "_bp" else 1.1))
                        for suf, lab, ci, ls in VARIANTS], y=-0.03)
    fig.subplots_adjust(wspace=0.35, hspace=0.75)
    S.save(fig, NAME, root=args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
