#!/usr/bin/env python3
"""Figure 1: a conceptual schematic above the four measured phenomenon panels.
The schematic distinguishes approximate mean-driven collapse from observed recovery.
Data panels show probe loss, hidden cosine, gate participation, and the shared and input-dependent output error.
Reads results/E1_headline. The schematic's illustrative bars are not measurements.

Usage: PYTHONPATH=src python3 scripts/figures/fig1_phenomenon.py [--results results]
"""
from __future__ import annotations
import os, sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import S

EXP = "E1_headline"
DFA_TAG, BP_TAG = "base", "bp"
NAME = "fig1_phenomenon"


def panel_loss(ax, d):
    """Probe loss of DFA and its BP twin, with the constant-predictor loss marked."""
    dfa = d.traj(EXP, DFA_TAG, "probe_loss"); bp = d.traj(EXP, BP_TAG, "probe_loss")
    if dfa is None:
        S.skip("fig1a loss", f"{EXP}/{DFA_TAG} has no runs"); S.empty(ax, "no DFA runs"); return
    prior = d.stat(EXP, DFA_TAG, "prior_loss")
    if np.isfinite(prior):
        ax.axhline(prior, color=S.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate("class prior", xy=(0.97, prior), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=S.MUTED)
    s, m, sd = dfa
    # DFA and BP follow the shared legend (solid ink and dashed grey); layer colours are reserved for e and f.
    ax.plot(s, m, color=S.INK, zorder=3); S.band(ax, s, m, sd, S.INK, alpha=0.12)
    S.label_line(ax, s[-1], m[-1], " DFA", S.INK)
    if bp is None:
        S.skip("fig1a BP twin", f"{EXP}/{BP_TAG} has no runs")
    else:
        sb, mb, sdb = bp
        ax.plot(sb, mb, color=S.MUTED, ls=(0, (3, 2)), zorder=3); S.band(ax, sb, mb, sdb, S.MUTED, alpha=0.12)
        S.label_line(ax, sb[-1], mb[-1], " BP", S.MUTED)
    on = d.stat(EXP, DFA_TAG, "plateau_onset"); ex = d.stat(EXP, DFA_TAG, "plateau_exit")
    if np.isfinite(on) and np.isfinite(ex):
        ax.axvspan(on, ex, color=S.color(0), alpha=0.09, lw=0, zorder=0)
    ax.set_xlabel("step"); ax.set_ylabel("probe loss"); ax.set_xlim(0, s[-1]); S.extend_right(ax)
    S.panel_tag(ax, "d")


def schematic(fig, axes):
    """An explanatory cartoon; unit positions denote identity, not spectral modes."""
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    a, b, c = axes
    for ax, letter, heading in zip(axes, "abc", ("Mean drive", "Collapse and plateau", "Recovery")):
        S.panel_title(ax, letter, heading)
    a.text(0.5, 0.86, "shared output error", ha="center", fontsize=7.2)
    a.add_patch(FancyBboxPatch((0.19, 0.51), 0.62, 0.19, boxstyle="round,pad=0.025",
                             fc="#edf3f8", ec=S.color(0), lw=0.7))
    a.text(0.5, 0.605, "fixed feedback", ha="center", va="center", fontsize=7.1)
    a.annotate("", (0.5, 0.715), (0.5, 0.83), arrowprops=dict(arrowstyle="->", lw=0.8, color=S.INK))
    xs = np.linspace(0.12, 0.88, 5)
    for i, x in enumerate(xs):
        a.annotate("", (x, 0.34), (0.23 + i * 0.135, 0.50),
                   arrowprops=dict(arrowstyle="->", lw=0.8, color=S.color(1)))
        a.add_patch(Ellipse((x, 0.24), 0.06, 0.14, facecolor="white", edgecolor=S.INK, lw=0.7))
        a.text(x, 0.24, str(i + 1), ha="center", va="center", fontsize=6)
    a.text(0.5, -0.05, "Unequal shifts in unit means", ha="center", va="top", fontsize=7.0)
    for ax, heights in [(b, [0.06, 0.1, 0.60, 0.04, 0.08]),
                        (c, [0.35, 0.55, 0.12, 0.48, 0.20])]:
        centers=0.52+xs*.48 if ax is b else xs
        ax.text(0.76 if ax is b else 0.5, 0.87, "gate energy", ha="center", fontsize=7.1, color=S.INK)
        ax.plot([centers[0]-.04, centers[-1]+.04], [0.17, 0.17], color=S.MUTED, lw=0.6)
        ax.bar(centers, heights, width=0.048 if ax is b else 0.10, bottom=0.17, color=S.color(0), alpha=0.8)
        for i, x in enumerate(centers):
            ax.text(x, 0.12, str(i + 1), ha="center", va="top", fontsize=6)
    curve=b.inset_axes([.01,.24,.43,.53]);v=np.linspace(-3,3,100)
    curve.plot(v,np.tanh(v),color=S.INK,lw=.9)
    curve.axhline(0,color="#d5d8dc",lw=.5);curve.axvline(0,color="#d5d8dc",lw=.5)
    curve.scatter([0],[0],s=14,facecolors="white",edgecolors=S.color(0),zorder=4)
    curve.scatter([-2.4,2.4],np.tanh([-2.4,2.4]),s=13,color=S.color(1),zorder=4)
    curve.set(xlim=(-3.2,3.2),ylim=(-1.2,1.2),xticks=[],yticks=[])
    for spine in curve.spines.values():spine.set_visible(False)
    b.text(.22,.87,r"$\tanh(\mu_i)$",ha="center",fontsize=7)
    b.text(.22,.13,"unit means",ha="center",va="top",fontsize=6)
    b.text(0.5, -0.05, "Saturation; readout fits the prior", ha="center", va="top", fontsize=7.0)
    c.text(0.5, -0.05, "More units participate; the set changes", ha="center", va="top", fontsize=7.0)
    for left, right, style in [(a, b, "-"), (b, c, (0, (3, 2)))]:
        lbox, rbox = left.get_position(), right.get_position()
        y = lbox.y0 + 0.5 * lbox.height
        fig.add_artist(FancyArrowPatch((lbox.x1 + 0.004, y), (rbox.x0 - 0.01, y),
                       transform=fig.transFigure, arrowstyle="->", mutation_scale=9,
                       lw=1.0, linestyle=style, color=S.INK))


def panel_layers(ax, d, stem, ylabel, letter, tag=DFA_TAG, ref_tag=BP_TAG, ylim=None):
    """One colour per layer, with solid DFA and dashed matched BP trajectories."""
    n = d.layers(EXP, tag)
    if not n:
        S.skip(f"fig1 {stem}", f"{EXP}/{tag} has no summarized runs"); S.empty(ax, "no runs"); return
    cols = F.layer_colors(n)
    for l in range(1, n + 1):
        t = d.traj(EXP, tag, f"{stem}_l{l}")
        if t is None: continue
        s, m, _ = t
        ax.plot(s, m, color=cols[l - 1], zorder=3)
        if ref_tag is not None and d.has(EXP, ref_tag):
            r = d.traj(EXP, ref_tag, f"{stem}_l{l}")
            if r is not None: ax.plot(r[0], r[1], color=cols[l - 1], lw=0.9, alpha=0.45, ls=(0, (3, 2)), zorder=2)
    ax.set_xlabel("step"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(0, s[-1])
    S.panel_tag(ax, letter)


def panel_errors(ax, d, letter):
    """Probe-mean error norm and RMS input-dependent error of plain DFA."""
    drew = False
    # Labels sit at fixed data positions: beside the early drop of the shared error, above the input-dependent plateau.
    for col, label, colour, style, at in (("ebar_norm", r"shared $\|\bar e\|$", S.color(7), "-", (42, 0.72)),
                                          ("etil_norm", r"input-dependent $\|\tilde e\|$", S.color(6), "-",
                                           (40, 1.1))):
        t = d.traj(EXP, DFA_TAG, col)
        if t is None: continue
        s, m, sd = t
        ax.plot(s, m, color=colour, ls=style, zorder=3); S.band(ax, s, m, sd, colour, alpha=0.15); drew = True
        ax.text(*at, label, ha="left", va="center", fontsize=6.3, color=colour)
    if not drew:
        S.skip("fig1g errors", f"{EXP}/{DFA_TAG} has no error norms"); S.empty(ax, "no runs"); return
    ax.set_xlabel("step"); ax.set_ylabel("error norm"); ax.set_ylim(0, 1.45)
    S.panel_tag(ax, letter)


def main():
    args = F.parse(__doc__)
    S.use_style()
    d = F.Data(args.results)
    if not d.has(EXP):
        S.skip("figure 1", f"{os.path.join(args.results, EXP)} does not exist")
    fig = S.plt.figure(figsize=(S.FULL_WIDTH, 3.05))
    top = [fig.add_axes([0.04 + i * 0.335, 0.68, 0.285, 0.27]) for i in range(3)]
    schematic(fig, top)
    grid = fig.add_gridspec(1, 4, left=0.065, right=0.985, bottom=0.16, top=0.55, wspace=0.48)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    panel_loss(axes[0], d)
    panel_layers(axes[1], d, "cos", "hidden cosine", "e", ylim=(0, 1.05))
    panel_layers(axes[2], d, "p", "gate participation", "f", ylim=(0, 1.05))
    panel_errors(axes[3], d, "g")
    # Expand the motivating transient while retaining the late recovery.
    from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
    on=d.stat(EXP,DFA_TAG,'plateau_onset');ex=d.stat(EXP,DFA_TAG,'plateau_exit')
    for i,ax in enumerate(axes):
        ax.set_xscale('symlog',linthresh=150,linscale=.7);ax.set_xlim(0,3000)
        ax.xaxis.set_major_locator(FixedLocator([0,100,1000,3000]));ax.xaxis.set_major_formatter(FixedFormatter(['0','100','1000','3000']))
        ax.xaxis.set_minor_locator(NullLocator());ax.tick_params(axis='x',labelsize=6)
        if i:ax.axvspan(on,ex,color=S.color(0),alpha=.07,lw=0,zorder=0)
    S.rule_legend(fig, [(f"layer {l}", dict(color=S.color(l - 1), lw=S.LW)) for l in range(1, 4)] +
                       [("DFA", dict(color=S.INK, lw=S.LW)),
                        ("matched BP", dict(color=S.INK, lw=0.9, alpha=0.5, ls=(0, (3, 2))))], y=0.035)
    S.save(fig, NAME, root=args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
