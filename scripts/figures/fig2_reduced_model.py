#!/usr/bin/env python3
"""Figure 2, the reduced model against the network. Upper three panels overlay the integrated system (dashed) on the run it
was initialized from (solid): probe loss with the constant-predictor line, gate participation per layer, and mean-activity
energy per layer. The lower two panels compare model and network across every condition the model was run on: participation
minima per layer, and plateau onset and exit.

Reads results/<exp>/<tag>/reduced_<stem>.csv (written by `python -m cmc.reduced --from_run <meta.json>`), the matching run
CSV, and results/reduced_validation.csv (scripts/validate_reduced.py).
Usage: PYTHONPATH=src python3 scripts/figures/fig2_reduced_model.py [--results results]
"""
from __future__ import annotations
import glob, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import S

NAME = "fig2_reduced_model"
RUN = ("E1_headline", "base", "seed0_fb0")
XMAX = 800


def load_pair(results):
    exp, tag, stem = RUN
    fm = os.path.join(results, exp, tag, f"reduced_{stem}.csv"); fr = os.path.join(results, exp, tag, f"{stem}.csv")
    if not (os.path.exists(fm) and os.path.exists(fr)): return None, None, None
    meta = json.load(open(os.path.join(results, exp, tag, f"{stem}.meta.json")))
    return pd.read_csv(fm), pd.read_csv(fr), meta


def panel_loss(ax, model, run, meta, letter):
    if model is None: S.skip("fig2a", "reduced model not run"); S.empty(ax, "reduced model not run"); return
    ax.axhline(meta["prior_loss"], color=S.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate("class-prior loss", xy=(XMAX, meta["prior_loss"]), xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom", fontsize=6, color=S.MUTED)
    r = run[run.step <= XMAX]; m = model[model.step <= XMAX]
    ax.plot(r.step, r.probe_loss, color=S.color(0), lw=S.LW, label="network", zorder=3)
    ax.plot(m.step, m.loss_model, color=S.color(0), lw=1.3, ls=(0, (3, 1.5)), label="model", zorder=4)
    if "onset_pred" in m and np.isfinite(m.onset_pred.iloc[0]):
        ax.axvline(float(m.onset_pred.iloc[0]), color=S.color(1), lw=0.8, ls=":", zorder=2)
        ax.annotate("onset", xy=(float(m.onset_pred.iloc[0]), 0.7), xytext=(3, 0), textcoords="offset points", fontsize=6, color=S.color(1), va="bottom")
    if "exit_pred" in m and np.isfinite(m.exit_pred.iloc[0]) and m.exit_pred.iloc[0] <= XMAX:
        ax.axvline(float(m.exit_pred.iloc[0]), color=S.color(1), lw=0.8, ls=":", zorder=2)
        ax.annotate("exit", xy=(float(m.exit_pred.iloc[0]), 0.7), xytext=(3, 0), textcoords="offset points", fontsize=6, color=S.color(1), va="bottom")
    ax.set_xlim(0, XMAX); ax.set_xlabel("step"); ax.set_ylabel("probe loss"); ax.set_ylim(0.5, 7.2)
    ax.legend(fontsize=6, loc="upper right", handlelength=1.8, borderpad=0.2, labelspacing=0.25)
    S.panel_title(ax, letter, "loss")


def panel_layers(ax, model, run, stem, ylabel, letter, title, log=False):
    if model is None: S.skip(f"fig2 {stem}", "reduced model not run"); S.empty(ax, "reduced model not run"); return
    r = run[run.step <= XMAX]; m = model[model.step <= XMAX]; L = sum(1 for c in run.columns if c.startswith(f"{stem}_l"))
    for l in range(1, L + 1):
        c = f"{stem}_l{l}"
        if c in r: ax.plot(r.step, r[c], color=S.color(l - 1), lw=S.LW, zorder=3)
        if c in m: ax.plot(m.step, m[c], color=S.color(l - 1), lw=1.3, ls=(0, (3, 1.5)), zorder=4)
    ax.set_xlim(0, XMAX); ax.set_xlabel("step"); ax.set_ylabel(ylabel)
    if log: ax.set_yscale("log")
    else: ax.set_ylim(0, 1.05)
    S.panel_title(ax, letter, title)


def panel_minima(ax, results, letter):
    f = os.path.join(results, "reduced_crossseed.csv")
    if not os.path.exists(f): S.skip("fig2d", "reduced_crosscheck.csv missing"); S.empty(ax, "cross-check not run"); return
    v = pd.read_csv(f); v = v[np.isfinite(v.p_meas) & np.isfinite(v.p_model)]
    if not len(v): S.empty(ax, "no minima"); return
    for l in sorted(v.layer.unique()):
        sub = v[v.layer == l]
        ax.scatter(sub.p_meas, sub.p_model, s=7, color=S.color(int(l) - 1), lw=0, alpha=0.6, label=f"L{int(l)}", zorder=3)
    r = np.corrcoef(v.p_meas, v.p_model)[0, 1]; mae = np.mean(np.abs(v.p_meas - v.p_model))
    ax.plot([0, 1], [0, 1], color=S.MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xlabel(r"measured $\min p_\ell$"); ax.set_ylabel(r"modelled $\min p_\ell$")
    S.panel_title(ax, letter, "all conditions")


def panel_onset_exit(ax, results, letter):
    f = os.path.join(results, "reduced_crossseed_plateaus.csv")
    if not os.path.exists(f): S.skip("fig2e", "reduced_crosscheck_plateaus.csv missing"); S.empty(ax, "cross-check not run"); return
    v = pd.read_csv(f); drew = False
    # Neutral colours: the figure's categorical colours denote layers.
    for a, b, lab, mk, col in (("onset_meas", "onset_model", "onset", "o", "#9a9a9a"), ("exit_meas", "exit_model", "exit", "^", S.INK)):
        x = pd.to_numeric(v[a], errors="coerce").values; y = pd.to_numeric(v[b], errors="coerce").values; m = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        if not m.any(): continue
        ax.scatter(x[m], y[m], s=10, marker=mk, color=col, lw=0, alpha=0.85, label=lab, zorder=3); drew = True
    if not drew: S.empty(ax, "no plateaus"); return
    if "model_exit_censored" in v:
        censored=v[v.model_exit_censored.astype(bool) & v.exit_meas.notna()]
        if len(censored):
            ax.scatter(censored.exit_meas,censored.model_horizon,marker="^",s=22,
                       facecolors="none",edgecolors=S.INK,lw=.8,label="exit censored",zorder=4)
    lo, hi = 0, 3000; ax.plot([lo, hi], [lo, hi], color=S.MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xscale("symlog",linthresh=10); ax.set_yscale("symlog",linthresh=10); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured step"); ax.set_ylabel("modelled step")
    ax.legend(fontsize=6.2, loc="lower right", handletextpad=0.2, borderpad=0.2, labelspacing=0.2)
    S.panel_title(ax, letter, "plateau timing")


def main():
    args = F.parse(__doc__); S.use_style()
    model, run, meta = load_pair(args.results)
    fig=S.plt.figure(figsize=(S.FULL_WIDTH,3.2))
    grid=fig.add_gridspec(2,6,left=.075,right=.985,bottom=.19,top=.91,wspace=1.35,hspace=.75)
    ax=[fig.add_subplot(grid[0,i:i+2]) for i in (0,2,4)]+[fig.add_subplot(grid[1,:3]),fig.add_subplot(grid[1,3:])]
    panel_loss(ax[0], model, run, meta, "a")
    panel_layers(ax[1], model, run, "p", r"participation $p_\ell$", "b", "gate participation")
    ax[1].annotate('rebound not predicted',xy=(650,.7),xytext=(250,.97),fontsize=5.7,
                   arrowprops=dict(arrowstyle='->',lw=.55,color=S.MUTED),color=S.INK)
    panel_layers(ax[2], model, run, "H", r"$\|\bar h_\ell\|^2$", "c", "activity energy", log=True)
    panel_minima(ax[3], args.results, "d"); panel_onset_exit(ax[4], args.results, "e")
    # The validation scatter includes up to six layers; share its layer key
    # with the trajectories rather than drawing over the measured points.
    S.rule_legend(fig, [(f"L{l}", dict(color=S.color(l - 1), lw=S.LW)) for l in range(1, 7)] + [("network", dict(color=S.INK, lw=S.LW)), ("model", dict(color=S.INK, lw=1.3, ls=(0, (3, 1.5))))], y=0.08)
    S.save(fig, NAME, root=args.out_root)


if __name__ == "__main__":
    main()
