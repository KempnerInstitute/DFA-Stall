#!/usr/bin/env python3
"""Figure 3, scaling and mean-channel controls. (a) Product law on the pure-drive sweeps of E2 (feedback gain g and output-bias
initialization q at a fixed task and head); the class-count, imbalance and learning-rate sweeps are shown in grey. (b) Windowed
participation minimum of the deepest layer against the transient collapse dose kappa_L, every uncentered DFA run, with the gate
quadrature asymptote 1.5/kappa. (c) Learning cost of collapse: steps to 50% probe accuracy and loss-based learning time
under the same controls as (d), plus centered output error and prior-bias initialization. (d) Peak cosine by layer under
input centering, frozen hidden biases and their combination. The shared panel_width helper is used by Figure 15f.

Reads results/kappa_transient_per_run.csv (scripts/derived_laws.py), results/submission_20260923/learning_times.csv
(scripts/submission_reporting.py) and the E1/E2 run CSVs.
Usage: PYTHONPATH=src python3 scripts/figures/fig3_master_curve.py [--results results]
"""
from __future__ import annotations
import glob, json, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F
from _figlib import S, A

NAME = "fig3_master_curve"
EXP = "E2_master"
INTERVENED = re.compile(r"center|whiten|gated|bn|muon|aligned")   # runs whose hidden layers do not receive the raw batch error


def loglog_fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3: return np.nan, np.nan, np.nan, int(m.sum())
    lx, ly = np.log10(x[m]), np.log10(y[m]); b, a = np.polyfit(lx, ly, 1); r = np.corrcoef(lx, ly)[0, 1]
    return b, a, r, int(m.sum())


def panel_product(ax, KT):
    e2 = KT[(KT.exp == EXP) & (KT.rule == "dfa")]
    e2 = e2[e2.tag != "q0.5"]  # same trajectories as g1.0
    if not len(e2): S.skip("fig3a", "no E2 rows in kappa_transient_per_run.csv"); S.empty(ax, "no E2 runs"); return
    is_g = e2.tag.str.match(r"^g[0-9.]+$"); is_q = e2.tag.str.match(r"^q[0-9.]+$")
    other = e2[~(is_g | is_q) & ~e2.tag.str.contains(INTERVENED)]
    ax.scatter(other.drive, 1 - other.maxcos_L, s=8, color="#c4c3be", lw=0, label="other sweeps", zorder=1)
    ax.scatter(e2[is_g].drive, 1 - e2[is_g].maxcos_L, s=13, color=S.color(0), lw=0, label="gain $g$", zorder=3)
    ax.scatter(e2[is_q].drive, 1 - e2[is_q].maxcos_L, s=13, color=S.color(1), lw=0, marker="s", label="bias setting $q$", zorder=3)
    pure = e2[is_g | is_q]; b, c, r, n = loglog_fit(pure.drive, 1 - pure.maxcos_L)
    if np.isfinite(b):
        xx = np.logspace(np.log10(pure.drive.min()), np.log10(pure.drive.max()), 50)
        ax.plot(xx, 10 ** c * xx ** b, color=S.INK, lw=1.1, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$g\,\|\bar e_0\|$"); ax.set_ylabel(r"$1-\max_t\,\cos_L$")
    ax.legend(fontsize=5.8, loc="upper right", handletextpad=0.2, borderpad=0.2, labelspacing=0.25)


def panel_kappa(ax, KT):
    sub = KT[KT.exp.isin(["E1_headline", "E2_master", "E5_recovery"]) & (KT.rule == "dfa") & ~KT.tag.str.contains(INTERVENED)]
    sub = sub.drop_duplicates("condition_key")
    sub = sub[np.isfinite(sub.kt_L) & (sub.kt_L > 0) & np.isfinite(sub.minp300_L)]
    if not len(sub): S.skip("fig3b", "no rows"); S.empty(ax, "no runs"); return
    ax.scatter(sub.kt_L, sub.minp300_L, s=7, color=S.color(2), lw=0, alpha=0.75, label="DFA runs", zorder=2)
    k = np.logspace(np.log10(max(sub.kt_L.min(), 1e-2)), np.log10(sub.kt_L.max() * 1.3), 100)
    ax.plot(k, np.minimum(1.0, 1.5 / k), color=S.INK, lw=1.0, ls=(0, (4, 2)), label=r"$\min(1,\,1.5/\kappa_L)$", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa_L$ (transient dose)"); ax.set_ylabel(r"$\min_{t\leq300}\,p_L$")
    ax.legend(fontsize=5.8, loc="lower left", handletextpad=0.3, borderpad=0.2, labelspacing=0.25)


COST_TAGS = (("base", "baseline"), ("pixcenter", "centered inputs"), ("frzbias", "frozen biases"),
             ("pixcenter_frzbias", "both"), ("center_e", "centered error"), ("priorbias", "prior bias"))


def panel_cost(ax, results):
    """Steps to 50% probe accuracy (filled) and loss-based learning time (open) for the controls of panel d and two
    output-side remedies; the dotted line marks BP's steps to 50% accuracy. Colours of the first four match panel d."""
    f = os.path.join(results, "submission_20260923", "learning_times.csv")
    if not os.path.exists(f): S.skip("fig3c", "run scripts/submission_reporting.py"); S.empty(ax, "no timing"); return
    t = pd.read_csv(f)
    for i, (tag, _) in enumerate(COST_TAGS):
        g = t[t.tag == tag]
        if not len(g): continue
        ax.scatter(i, g.acc50.median(), s=22, color=S.color(i), zorder=3, lw=0)
        ax.scatter(i, g.time.mean(), s=20, marker="s", facecolors="none", edgecolors=S.color(i), lw=1.0, zorder=3)
    bp = t[t.tag == "bp"]
    if len(bp):
        ax.axhline(bp.acc50.median(), color=S.MUTED, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
        ax.text(len(COST_TAGS) - 0.55, bp.acc50.median() * 0.86, "BP", color=S.MUTED, fontsize=6, ha="right", va="top")
    ax.set_yscale("log"); ax.set_ylim(100, 9000)   # headroom above the data for the marker key
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator([100, 300, 1000, 3000])); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xticks(range(len(COST_TAGS))); ax.set_xticklabels([lab for _, lab in COST_TAGS], rotation=50, ha="right",
                                                             rotation_mode="anchor", fontsize=6)
    ax.set_xlim(-0.6, len(COST_TAGS) - 0.4); ax.set_ylabel("steps")
    import matplotlib.lines as mlines
    keys = [mlines.Line2D([], [], color=S.INK, marker="o", ls="", ms=3.6, label="50% accuracy"),
            mlines.Line2D([], [], color=S.INK, marker="s", ls="", ms=3.6, mfc="none", label=r"loss $<0.9\,\mathcal{L}_{\rm prior}$")]
    ax.legend(handles=keys, fontsize=6, loc="upper left", handletextpad=0.1, borderpad=0.1, labelspacing=0.2,
              borderaxespad=0.1)


def panel_width(ax, KT, results):
    """Three-layer width sweep reused in the width–depth figure; exit times use the right-hand axis."""
    e2 = KT[(KT.exp == EXP) & (KT.rule == "dfa")]; rows = []; per_run = []
    for tag in ["w100", "w300", "w900", "w2000"]:
        sub = e2[e2.tag == tag]
        if not len(sub): continue
        ex = []
        for f in sorted(glob.glob(os.path.join(results, EXP, tag, "seed*_fb*.csv"))):
            if "reduced_" in os.path.basename(f): continue
            meta = json.load(open(f.replace(".csv", ".meta.json"))); df = pd.read_csv(f)
            pl = A.plateau(df, meta["prior_loss"], meta["prior"]); ex.append(pl["exit"] if pl and pl.get("exit") else np.nan)
            if np.isfinite(ex[-1]): per_run.append((int(tag[1:]), ex[-1]))
        rows.append(dict(width=int(tag[1:]), p1=sub.minp300_1.mean(), pL=sub.minp300_L.mean(), exit=np.nanmean(ex) if np.isfinite(ex).any() else np.nan))
    if not rows: S.skip("width panel", "no width sweep"); S.empty(ax, "no width sweep"); return
    Wd = pd.DataFrame(rows).sort_values("width")
    ax.plot(Wd.width, Wd.p1, "o-", color=S.color(0), lw=1.4, ms=3.5, label=r"$\min p_1$")
    ax.plot(Wd.width, Wd.pL, "s-", color=S.color(2), lw=1.4, ms=3.5, label=r"$\min p_L$")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("width $d$"); ax.set_ylabel(r"$\min_{t\leq300}\,p_\ell$")
    # Key inside the axes, right of the exit points, so the panel header stays level with its row.
    ax.legend(fontsize=5.8, loc="center right", ncol=1,
              handletextpad=0.3, borderpad=0.2, columnspacing=.8)
    ok = Wd.dropna(subset=["exit"])
    if len(ok) >= 2:
        ax2 = ax.twinx(); ax2.plot(ok.width, ok.exit, "^", color=S.color(1), ms=3.5, lw=0)
        pr = np.asarray(per_run, float); bb, cc, rr, _ = loglog_fit(pr[:, 0], pr[:, 1]); xx = np.array([ok.width.min(), ok.width.max()], float)   # per-run fit, as in make_numbers
        if np.isfinite(bb): ax2.plot(xx, 10 ** cc * xx ** bb, color=S.color(1), lw=1.0, ls=(0, (4, 2)))
        ax2.set_yscale("log"); ax2.set_ylabel("plateau exit (steps)", color=S.color(1)); ax2.tick_params(axis="y", colors=S.color(1))
        from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
        ax2.yaxis.set_major_locator(FixedLocator([200, 300, 500, 1000])); ax2.yaxis.set_minor_locator(NullLocator()); ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(S.color(1)); ax2.spines["left"].set_visible(False)


def main():
    args = F.parse(__doc__); S.use_style()
    f = os.path.join(args.results, "kappa_transient_per_run.csv")
    if not os.path.exists(f): raise SystemExit(f"missing {f}; run scripts/derived_laws.py first")
    KT = pd.read_csv(f)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(S.FULL_WIDTH, 2.25))
    panel_product(ax[0], KT); panel_kappa(ax[1], KT); panel_cost(ax[2], args.results); panel_centering(ax[3], args.results)
    for letter, a in zip("abcd", ax): S.panel_tag(a, letter)
    # Put the keys of a, b and d below the axes so they cannot hide observations; c keeps its marker key inside.
    for a in (ax[0], ax[1], ax[3]):
        a.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0, -.46),
                 handlelength=1.2, handletextpad=.3, borderaxespad=0, labelspacing=.2)
    fig.subplots_adjust(left=.07, right=.99, bottom=.42, top=.91, wspace=.65)
    S.save(fig, NAME, args.out_root)


def panel_centering(ax,results):
    """The factorial mean-channel control belongs beside the main predictions."""
    for i,(tag,label) in enumerate((("base","baseline"),("pixcenter","centered inputs"),
                                    ("frzbias","frozen biases"),("pixcenter_frzbias","both"))):
        runs=[pd.read_csv(f) for f in glob.glob(os.path.join(results,"E1_headline",tag,"seed*_fb*.csv"))]
        peaks=np.array([[d[d.step<=800][f"cos_l{l}"].max() for l in (1,2,3)] for d in runs])
        ax.errorbar(np.arange(1,4)+(i-1.5)*.035,peaks.mean(0),yerr=peaks.std(0,ddof=1),
                    fmt="o-",ms=2.6,lw=1,color=S.color(i),label=label,capsize=1.5)
    ax.set(xlabel="hidden layer",ylabel="peak hidden cosine",xticks=[1,2,3],ylim=(0,1.12),xlim=(.8,3.2))
    ax.legend(fontsize=5.1,loc="center left",bbox_to_anchor=(.01,.47),handlelength=1.1,handletextpad=.25,labelspacing=.15)


if __name__ == "__main__":
    main()
