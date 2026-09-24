#!/usr/bin/env python3
"""Figure 7 (appendix), CIFAR-10: a 2-by-2 comparison of hidden cosine under DFA, matched BP and the interventions, for the small convolutional
network under the sigmoid readout, under a balanced and an imbalanced softmax, and for the flattened MLP. Reads the shared
results/cifar_reporting archive produced from E11/E12 logs by scripts/cifar_reporting.py. Usage: PYTHONPATH=src python3 scripts/figures/fig7_cifar.py [--results results]"""
from __future__ import annotations
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import S
from cmc.cifar_reporting import load_records, trajectory

NAME = "fig7_cifar"
PANELS = [  # (title, [(exp, tag, label, colour index, linestyle)])
    ("CNN, sigmoid readout", [("E11_cifar", "cnn", "DFA", 0, "-"), ("E11_cifar", "cnn_bp", "matched BP", 1, (0, (3, 2))), ("E11_cifar", "cnn_centere", r"centered $e$", 2, "-"),
                           ("E11_cifar", "cnn_centere_pixcenter", r"centered $e$ + inputs", 3, "-"), ("E11_cifar", "cnn_priorbias", "prior bias", 4, "-"), ("E12_cifar2", "cnn_calib", "calibrated readout", 5, "-")]),
    ("CNN, balanced softmax", [("E11_cifar", "cnn_softmax", "DFA", 0, "-"), ("E11_cifar", "cnn_softmax_bp", "matched BP", 1, (0, (3, 2)))]),
    ("CNN, softmax, $p_0=0.7$", [("E11_cifar", "cnn_softmax_imb07", "DFA", 0, "-"), ("E11_cifar", "cnn_softmax_imb07_bp", "matched BP", 1, (0, (3, 2))), ("E11_cifar", "cnn_softmax_imb07_centere", r"centered $e$", 2, "-")]),
    ("MLP, sigmoid readout (layer 1)", [("E11_cifar", "mlp", "DFA", 0, "-"), ("E11_cifar", "mlp_bp", "matched BP", 1, (0, (3, 2))), ("E11_cifar", "mlp_centere", r"centered $e$", 2, "-"),
                                     ("E11_cifar", "mlp_pixcenter", "centered inputs", 3, "-"), ("E11_cifar", "mlp_std", "standardized", 6, "-"), ("E12_cifar2", "mlp_calib", "calibrated readout", 5, "-")]),
]
XMAX = 2000


def main():
    args = F.parse(__doc__); S.use_style(); records = load_records(args.results)
    fig, axes = S.plt.subplots(2, 2, figsize=(S.FULL_WIDTH, 4.9))
    for ax, (title, series), letter in zip(axes.flat, PANELS, "abcd"):
        drew = False
        for exp, tag, label, ci, ls in series:
            s, m, sd = trajectory(records, exp, tag); keep = s <= XMAX
            ax.plot(s[keep], m[keep], color=S.color(ci), lw=1.4, ls=ls, label=label); S.band(ax, s[keep], m[keep], sd[keep], S.color(ci), alpha=0.12); drew = True
        if not drew: S.empty(ax, "no runs")
        ax.set_ylim(0, 1.02); ax.set_xlim(0, XMAX); ax.set_xticks([0, 500, 1000, 1500, 2000]); ax.set_xlabel("step")
        ax.legend(fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2,
                  labelspacing=0.25, handlelength=1.4, columnspacing=0.8, borderaxespad=0)
        S.panel_title(ax, letter, title)
    for ax in axes[:, 0]: ax.set_ylabel("hidden cosine")
    fig.subplots_adjust(wspace=0.25, hspace=0.9, bottom=0.2, top=0.95, left=0.08, right=0.99)
    S.save(fig, NAME, root=args.out_root)


if __name__ == "__main__":
    main()
