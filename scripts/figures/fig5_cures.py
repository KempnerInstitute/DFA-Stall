#!/usr/bin/env python3
"""Interventions: matched-step accuracy, early participation and conditioner damping. Reads results/E4_cures, whose tags are '<setting>__<cure>'.

Usage: PYTHONPATH=src python3 scripts/figures/fig5_cures.py [--results results]
"""
from __future__ import annotations
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import A, S

EXP = "E4_cures"
NAME = "fig5_cures"
CONTROL = "none"
ACC_STEPS = (1000, 1500, 3000)
CURE_LABEL = {"none": "plain DFA", "center_delta": r"center $\delta$", "center_e": r"center $e$",
              "gated": "gated centering", "gated03": "gated centering 0.3", "muon": "orthogonalize", "bn": "batch norm.",
              "whiten": "conditioner", "aligned": "aligned feedback", "priorbias": "prior bias",
              "outlr10": r"$10\times\eta_{\rm out}$", "center_outlr10": r"center $e$ + $10\times\eta_{\rm out}$",
              "center_priorbias": "center $e$ + prior bias"}
CURE_ORDER = ["none", "center_delta", "center_e", "priorbias", "outlr10",
              "center_outlr10", "center_priorbias", "aligned", "bn", "muon", "whiten"]
HIDDEN = ("gated", "gated03")          # energy-gated centering never released its gate, so it duplicates centre e
CURE_LABEL['whiten']='conditioner, default'
CURE_LABEL['whiten_lam10.0']='conditioner, selected'
CURE_ORDER.append('whiten_lam10.0')


def split(tag):
    """'<setting>__<cure>' -> (setting, cure); a tag without '__' counts as a cure of the base setting."""
    return tuple(tag.split("__", 1)) if "__" in tag else ("base", tag)


def inventory(d):
    """(settings, cures) present in the experiment, in a stable order."""
    tags = d.tags(EXP)
    settings, cures = [], []
    for t in tags:
        s, c = split(t)
        if s not in settings: settings.append(s)
        if c not in cures: cures.append(c)
    cures = [c for c in CURE_ORDER if c in cures] + [c for c in cures if c not in CURE_ORDER and c not in HIDDEN and "lam" not in c]
    return sorted(settings), cures


def label(cure):
    return CURE_LABEL.get(cure, cure.replace("_", " "))


def grouped_bars(ax, groups, series, values, colors, ylabel, letter, title, ylim=None, annotate_series=False, errors=None):
    """Grouped means and standard deviations; crosses denote unmeasured cells."""
    n, k = len(groups), max(len(series), 1)
    width = 0.8 / k
    x = np.arange(n)
    for j, name in enumerate(series):
        v = np.asarray([values[(g, name)] for g in groups], float)
        off = (j - (k - 1) / 2) * width
        finite = np.isfinite(v)
        sd=np.asarray([errors.get((g,name),np.nan) for g in groups]) if errors is not None else np.zeros(n)
        ax.errorbar((x+off)[finite],v[finite],yerr=sd[finite],fmt='o',ms=2.6,lw=.7,capsize=1.2,color=colors[j],label=str(name))
        if (~finite).any():
            ax.scatter((x + off)[~finite], np.full((~finite).sum(), 0.025), marker="x", s=9, lw=.65, color=colors[j], zorder=4)
    if annotate_series:
        ax.legend([f"{s} steps" for s in series], fontsize=6, loc="lower right", ncol=len(series), handlelength=1.0,
                  columnspacing=0.8, borderpad=0.2, handletextpad=0.4, framealpha=0.9, frameon=True, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([label(g) for g in groups], rotation=40, ha="right")
    ax.tick_params(axis="x", labelsize=6 if n > 8 else 7)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    S.panel_title(ax, letter, title)


def panel_acc_steps(ax, d, cures, setting, letter):
    """Accuracy at 1000/1500/3000 steps for each cure, in one setting."""
    title = f"accuracy, {setting} setting"
    if not cures:
        S.skip("fig5a accuracy at steps", f"{EXP} has no tags"); S.empty(ax, "no cure runs")
        S.panel_title(ax, letter, title); return
    vals = {(c, s): d.stat(EXP, f"{setting}__{c}", f"acc_{s}") for c in cures for s in ACC_STEPS}
    errors={(c,s):d.stat(EXP,f'{setting}__{c}',f'acc_{s}',how='sd') for c in cures for s in ACC_STEPS}
    if not np.isfinite(list(vals.values())).any():
        S.skip("fig5a accuracy at steps", f"no acc_{ACC_STEPS} in setting {setting} (runs too short)")
        S.empty(ax, "runs shorter than 1000 steps"); S.panel_title(ax, letter, title); return
    grouped_bars(ax, cures, list(ACC_STEPS), vals, [S.color(i) for i in range(3)], "probe accuracy", letter,
                 title, ylim=(0, 1.0), annotate_series=True,errors=errors)


def panel_by_setting(ax, d, cures, settings, col, ylabel, letter, title, ylim=None, omit_empty=False):
    """One column of the summary (accuracy or collapse depth) for every cure in every setting."""
    if not cures or not settings:
        S.skip(f"fig5 {col}", f"{EXP} has no tags"); S.empty(ax, "no cure runs"); S.panel_title(ax, letter, title); return
    vals = {};errors={}
    for c in cures:
        for st in settings:
            tag = f"{st}__{c}"
            name = col.replace("{L}", str(d.layers(EXP, tag) or 3))
            vals[(c, st)] = d.stat(EXP, tag, name)
            errors[(c,st)]=d.stat(EXP,tag,name,how='sd')
    if not np.isfinite(list(vals.values())).any():
        S.skip(f"fig5 {col}", f"no finite {col} in any setting")
        S.empty(ax, "column not available"); S.panel_title(ax, letter, title); return
    colors={setting:S.color(i+3) for i,setting in enumerate(settings)}
    if omit_empty:
        settings=[setting for setting in settings if any(np.isfinite(vals[(c,setting)]) for c in cures)]
    grouped_bars(ax, cures, settings, vals, [colors[setting] for setting in settings], ylabel, letter, title,
                 ylim=ylim,errors=errors)


def panel_ridge(ax, d, letter):
    """The error-side conditioner: accuracy at 3000 steps and peak cosine versus ridge."""
    tags = {0.1: "base__whiten", 0.3: "base__whiten_lam0.3", 1.0: "base__whiten_lam1.0", 3.0: "base__whiten_lam3.0", 10.0: "base__whiten_lam10.0"}
    lam = sorted(l for l, t in tags.items() if d.has(EXP, t))
    if len(lam) < 2: S.skip("fig5d ridge", "no ridge sweep"); S.empty(ax, "no ridge sweep"); S.panel_title(ax, letter, "conditioner ridge"); return
    L = d.layers(EXP, "base__none") or 3
    acc = [d.stat(EXP, tags[l], "acc_3000") for l in lam]; cos = [d.stat(EXP, tags[l], f"maxcos_l{L}") for l in lam]
    base_acc = d.stat(EXP, f"base__{CONTROL}", "acc_3000"); base_cos = d.stat(EXP, f"base__{CONTROL}", f"maxcos_l{L}")
    ax.errorbar(lam,acc,yerr=[d.stat(EXP,tags[l],'acc_3000',how='sd') for l in lam],fmt='o-',color=S.color(0),lw=1.2,ms=3,capsize=2,label='accuracy at 3000')
    ax.errorbar(lam,cos,yerr=[d.stat(EXP,tags[l],f'maxcos_l{L}',how='sd') for l in lam],fmt='s-',color=S.color(1),lw=1.2,ms=3,capsize=2,label=r'peak $\cos_L$')
    if np.isfinite(base_acc): ax.axhline(base_acc, color=S.color(0), lw=0.8, ls=(0, (4, 3)), label="plain DFA, accuracy")
    if np.isfinite(base_cos): ax.axhline(base_cos, color=S.color(1), lw=0.8, ls=(0, (4, 3)), label=r"plain DFA, peak $\cos_L$")
    ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlabel(r"relative ridge $\lambda_{\rm rel}$"); ax.set_ylabel("value")
    ax.legend(fontsize=6, loc="lower right", borderpad=0.2, labelspacing=0.25)
    S.panel_title(ax, letter, "error-side conditioner")


def main():
    args = F.parse(__doc__)
    S.use_style()
    d = F.Data(args.results)
    if not d.has(EXP):
        S.skip("figure 5", f"{os.path.join(args.results, EXP)} does not exist")
    settings, cures = inventory(d)
    fig, axes = S.plt.subplots(2, 2, figsize=(S.FULL_WIDTH, 4.2))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    panel_acc_steps(ax_a, d, cures, settings[0] if settings else "base", "a")
    panel_by_setting(ax_b, d, cures, settings, "acc_3000", "probe accuracy at 3000", "b",
                     "accuracy per setting", ylim=(0, 1.0), omit_empty=True)
    panel_by_setting(ax_c, d, cures, settings, f"minp_w{A.P_WINDOW}_l{{L}}",
                     rf"$\min_{{t\leq{A.P_WINDOW}}} p_L$", "c", "minimum gate participation", ylim=(0, 1.05))
    panel_ridge(ax_d, d, "d")
    if settings:
        S.rule_legend(fig, [({"base":"base MLP","cnn":"CNN","depth6":"six-layer MLP"}.get(st,st), dict(color=S.color(i + 3), lw=4)) for i, st in enumerate(settings)], y=-0.075)
    fig.subplots_adjust(wspace=0.28, hspace=0.95, left=0.07, right=0.99, top=0.94, bottom=0.14)
    S.save(fig, NAME, root=args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
