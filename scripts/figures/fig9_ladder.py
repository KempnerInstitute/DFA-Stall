#!/usr/bin/env python3
"""Compact teacher-student ladder (appendix): test loss relative to the constant predictor, top-layer hidden cosine and DFA weight alignment against
t = step / D, for six rungs from the Refinetti setting to ours, DFA and its BP twin, three seeds each. Reads results/E6_ladder.
Writes paper/figures/fig6b_ladder.{pdf,png}. Usage: PYTHONPATH=src python3 scripts/figures/fig9_ladder.py"""
from __future__ import annotations
import glob, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import S

NAME = "fig6b_ladder"
RUNGS = [("rung0_vector_K4M4", "R0 zero-mean"), ("rung1_nonzero_mean_targets", "R1 one-hot"), ("rung2_hidden_biases", "R2 +biases"),
         ("rung3_depth3_tanh", "R3 +depth"), ("rung4_shifted_inputs", "R4 +x shift"), ("rung5_gain3", "R5 +gain 3")]


def runs(root, rung, rule):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "E6_ladder", f"{rung}_{rule}", "seed*.csv"))):
        meta = json.load(open(f.replace(".csv", ".meta.json"))); out.append((pd.read_csv(f), meta))
    return out


def main():
    args = F.parse(__doc__); S.use_style()
    fig, axes = S.plt.subplots(3, len(RUNGS), figsize=(S.FULL_WIDTH, 3.7), sharex="col")
    for j, (rung, title) in enumerate(RUNGS):
        top, mid, bot = axes[0, j], axes[1, j], axes[2, j]; drew = False
        for rule, col, ls in (("bp", S.color(1), (0, (3, 2))), ("dfa", S.color(0), "-")):
            for si, (df, meta) in enumerate(runs(args.results, rung, rule)):
                nl = len(meta["widths_list"]); d = df[df.step > 0]; Lc = meta["const_loss"]
                kw = dict(color=col, ls=ls, lw=0.85, alpha=0.7, label=rule.upper() if si == 0 else None)
                top.plot(d.t, d.test_loss / Lc, **kw); mid.plot(d.t, d[f"hcos_l{nl}"], **kw)
                if rule == "dfa": bot.plot(d.t, d[f"wa_vec_l{nl}"], **kw)
                drew = True
        if not drew: S.empty(top, "no runs"); S.empty(mid, ""); S.empty(bot, ""); continue
        top.axhline(1.0, color=S.MUTED, lw=0.8, ls=":"); top.set_xscale("log"); top.set_yscale("log"); top.set_ylim(2e-2, 2.0)
        mid.set_xscale("log"); mid.set_ylim(-0.05, 1.05)
        bot.set_xscale("log"); bot.set_ylim(-0.15, 1.05); bot.axhline(0, color=S.MUTED, lw=0.6); bot.set_xlabel(r"$t=\mathrm{step}/D$")
        S.panel_title(top, "abcdef"[j], title.replace(" ", ": ", 1))
        if j: top.set_yticklabels([]); mid.set_yticklabels([]); bot.set_yticklabels([])
    axes[0, 0].set_ylabel("relative loss"); axes[1, 0].set_ylabel("hidden cosine"); axes[2, 0].set_ylabel("matrix alignment (DFA)")
    axes[0, 0].legend(fontsize=5.6, loc="lower left", borderpad=0.2, labelspacing=0.2)
    fig.subplots_adjust(wspace=0.18, hspace=0.28, left=0.07, right=0.99, top=0.93, bottom=0.12)
    S.save(fig, NAME, root=args.out_root)


if __name__ == "__main__":
    main()
