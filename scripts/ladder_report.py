"""Build results/E6_ladder/SUMMARY.md and paper/figures/fig6b_ladder.png from the E6 ladder runs."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cmc.ladder_analysis import load_rung, run_table  # imported after sys.path is extended

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "E6_ladder")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures", "fig6b_ladder.png")

ORDER = ["rung0_scalar_K2M2", "rung0_scalar_K2M4", "rung0_scalar_K4M4", "rung0_scalar_K4M8",
         "rung0_vector_K4M4", "rung0_vector_K4M8", "rung1_nonzero_mean_targets", "rung2_hidden_biases",
         "rung3_depth3_tanh", "rung4_shifted_inputs", "rung5_gain0.3", "rung5_gain3"]
FIGCOLS = ["rung0_scalar_K4M4", "rung0_vector_K4M4", "rung1_nonzero_mean_targets", "rung2_hidden_biases",
           "rung3_depth3_tanh", "rung4_shifted_inputs", "rung5_gain0.3", "rung5_gain3"]
SHORT = {"rung0_scalar_K2M2": "R0 scalar K=M=2", "rung0_scalar_K2M4": "R0 scalar K=2 M=4",
         "rung0_scalar_K4M4": "R0 scalar K=M=4", "rung0_scalar_K4M8": "R0 scalar K=4 M=8",
         "rung0_vector_K4M4": "R0 vector C=K=M=4", "rung0_vector_K4M8": "R0 vector C=K=4 M=8",
         "rung1_nonzero_mean_targets": "R1 +nonzero-mean targets", "rung2_hidden_biases": "R2 +hidden biases",
         "rung3_depth3_tanh": "R3 +depth 3 x 300 tanh", "rung4_shifted_inputs": "R4 +shifted inputs",
         "rung5_gain0.3": "R5 +gain g=0.3", "rung5_gain3": "R5 +gain g=3"}
INGREDIENT = {
    "rung0_scalar_K2M2": "Refinetti replica: erf teacher/student, zero-mean scalar MSE targets, no biases, isotropic zero-mean inputs",
    "rung0_scalar_K2M4": "as above, over-parameterised student M = 2K",
    "rung0_scalar_K4M4": "as above, K = M = 4",
    "rung0_scalar_K4M8": "as above, K = 4, M = 8",
    "rung0_vector_K4M4": "C = K = 4 zero-mean vector targets (identity teacher readout)",
    "rung0_vector_K4M8": "C = K = 4 vector targets, M = 2K",
    "rung1_nonzero_mean_targets": "+ sigmoid one-versus-rest labels at prior 1/C, output bias 0 (nonzero-mean targets)",
    "rung2_hidden_biases": "+ trainable hidden biases",
    "rung3_depth3_tanh": "+ depth 3 x width 300 tanh",
    "rung4_shifted_inputs": "+ anisotropic shifted inputs x = 0.13 + 0.3 z",
    "rung5_gain0.3": "+ feedback gain g = 0.3",
    "rung5_gain3": "+ feedback gain g = 3",
}


def available():
    return [r for r in ORDER if any(os.path.exists(os.path.join(ROOT, f"{r}_{k}")) for k in ("dfa", "bp"))]


def fmt(v, prec=3):
    if v is None:
        return "--"
    m, s = v
    return f"{m:.{prec}f} ± {s:.{prec}f}"


def pick_window(per, rung, rule, kind=None):
    """The longest plateau window of the requested kind, one row per seed."""
    sub = per[(per.rung == f"{rung}_{rule}") & (per.window >= 0)]
    if kind is not None:
        sub = sub[sub.kind == kind]
    if len(sub) == 0:
        return None
    exited = sub[sub.drop_after >= 0.15]
    src = exited if len(exited) else sub
    return src.loc[src.groupby("seed").decades.idxmax()]


def _row(per, rung, rule, kind, n_seeds):
    win = pick_window(per, rung, rule, kind)
    label = "collapse window (loss at the constant predictor)" if kind == "prior-band" else \
        "sub-prior window (loss below the constant predictor)"
    if win is None or len(win) == 0:
        return f"| {SHORT[rung]} | {rule.upper()} | {label} | none | -- | -- | -- | -- | -- | -- |"
    a = {c: (float(win[c].mean()), float(win[c].std(ddof=0))) for c in
         ("t0", "t1", "loss_over_const", "wa_vec_plateau", "wa_ratio", "hcos_plateau", "chi_plateau", "part_plateau")
         if c in win}
    vc = win.verdict.value_counts()
    return (f"| {SHORT[rung]} | {rule.upper()} | {label} | {len(win)}/{n_seeds} seeds | "
            f"{a['t0'][0]:.3g}-{a['t1'][0]:.3g} | {fmt(a.get('loss_over_const'))} | {fmt(a.get('wa_vec_plateau'))} | "
            f"{fmt(a.get('wa_ratio'))} | {fmt(a.get('hcos_plateau'))} | {vc.index[0] if len(vc) else '--'} |")


def headline(per):
    """Two data-driven statements: where the collapse switches on, and where both plateau types coexist."""
    switch, coexist = None, []
    for rung in available():
        pb = pick_window(per, rung, "dfa", "prior-band")
        sp = pick_window(per, rung, "dfa", "sub-prior")
        collapsed = pb is not None and len(pb) and (pb.verdict == "common-mode collapse").any()
        if collapsed and switch is None:
            switch = rung
        if collapsed and sp is not None and len(sp) and (sp.verdict == "Refinetti post-alignment").any():
            seeds = set(pb[pb.verdict == "common-mode collapse"].seed) & \
                set(sp[sp.verdict == "Refinetti post-alignment"].seed)
            if seeds:
                coexist.append((rung, sorted(seeds)))
    out = ["## Headline", ""]
    out.append(f"- The collapse switches on at **{SHORT[switch]}**: no earlier rung produces it, and it is present "
               f"from that rung on." if switch else "- No rung produced a common-mode collapse.")
    if coexist:
        out.append("- Both plateau types coexist in single runs at: " +
                   ", ".join(f"{SHORT[r]} (seeds {', '.join(str(x) for x in ss)})" for r, ss in coexist) +
                   " -- a constant-predictor plateau while alignment is still low, then a second plateau below the "
                   "constant predictor once alignment has peaked.")
    else:
        out.append("- No single run contained both plateau types.")
    return out + [""]


def summary(per):
    lines = ["# E6 teacher-student ladder: Refinetti's post-alignment plateau versus our common-mode collapse",
             "",
             "Produced by `scripts/ladder_report.py` from `cmc.ladder` runs in this directory: 3 seeds per rung, DFA and a",
             "BP twin, batch-1 online SGD on a fresh Gaussian sample per step, D = 500, eta = 0.5, eta_out = eta/10,",
             "feedback entries N(0, g^2). Rung 0 replicates Refinetti, d'Ascoli, Ohana and Goldt (ICML 2021); every later",
             "rung adds exactly one ingredient to the rung above it. Time is t = step / D.",
             "",
             "Two kinds of flat window are reported for every run, because a single run can contain both: a window whose",
             "loss sits in a +-5% band around the best constant predictor, and a window strictly below it. `L/Lconst` is the",
             "median loss in the window over the constant-predictor loss; `wa` is the weight alignment cos(M_L, B_L) of the",
             "last hidden layer, `wa/wa_max` its ratio to the maximum alignment reached anywhere in that run, and `hcos` the",
             "mean pairwise cosine of hidden vectors on the probe. Mean +- sd over seeds. The alignment columns are blank",
             "for BP, whose updates never use B. The BP rows of rung 4 and both rung-5 gains are identical runs: BP ignores",
             "the feedback gain, so with the same seed it sees the same initialisation and the same data stream.",
             "",
             "| rung | rule | window | seeds | t range | L/Lconst | wa at plateau | wa/wa_max | hcos at plateau | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines = lines[:-2] + headline(per) + lines[-2:]
    for rung in available():
        for rule in ("dfa", "bp"):
            sub = per[per.rung == f"{rung}_{rule}"]
            if len(sub) == 0:
                continue
            n = sub.seed.nunique()
            for kind in ("prior-band", "sub-prior"):
                lines.append(_row(per, rung, rule, kind, n))
    lines += ["", "## Where the collapse switches on (DFA, run-wide extrema, mean ± sd over seeds)", "",
              "| rung | peak hidden cosine | min gate E[phi'^2] / init | min gate participation | collapse plateau found |",
              "|---|---|---|---|---|"]
    for rung in available():
        sub = per[(per.rung == f"{rung}_dfa")].drop_duplicates("seed")
        if len(sub) == 0:
            continue
        pb = pick_window(per, rung, "dfa", "prior-band")
        got = "yes" if (pb is not None and len(pb) and
                        (pb.verdict == "common-mode collapse").any()) else "no"
        lines.append(f"| {SHORT[rung]} | {sub.hcos_run_max.mean():.3f} ± {sub.hcos_run_max.std(ddof=0):.3f} | "
                     f"{sub.chi_run_min.mean():.3f} ± {sub.chi_run_min.std(ddof=0):.3f} | "
                     f"{sub.part_run_min.mean():.3f} ± {sub.part_run_min.std(ddof=0):.3f} | {got} |")
    lines += ["", "## Per-rung reading (DFA)", ""]
    for rung in available():
        pb = pick_window(per, rung, "dfa", "prior-band")
        sp = pick_window(per, rung, "dfa", "sub-prior")
        bits = []
        if pb is not None and len(pb):
            bits.append(f"constant-predictor plateau over t = {pb.t0.mean():.2f}-{pb.t1.mean():.3g} at "
                        f"L/Lconst = {pb.loss_over_const.mean():.3f}, hidden cosine {pb.hcos_plateau.mean():.3f}, "
                        f"alignment {pb.wa_vec_plateau.mean():.3f} = {100 * pb.wa_ratio.mean():.0f}% of its run maximum "
                        f"({pb.verdict.value_counts().index[0]})")
        else:
            bits.append("no plateau at the constant-predictor loss")
        if sp is not None and len(sp):
            bits.append(f"sub-prior plateau over t = {sp.t0.mean():.3g}-{sp.t1.mean():.3g} at "
                        f"L/Lconst = {sp.loss_over_const.mean():.3f}, hidden cosine {sp.hcos_plateau.mean():.3f}, "
                        f"alignment {100 * sp.wa_ratio.mean():.0f}% of its run maximum "
                        f"({sp.verdict.value_counts().index[0]})")
        else:
            bits.append("no sub-prior plateau")
        lines.append(f"- **{SHORT[rung]}** -- {INGREDIENT[rung]}. " + "; ".join(bits) + ".")
    return "\n".join(lines) + "\n"


def figure(per):
    """Four-row ladder figure: loss, weight alignment, hidden cosine and the gate statistic against t = step/D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in FIGCOLS if os.path.exists(os.path.join(ROOT, f"{c}_dfa"))]
    fig, axes = plt.subplots(4, len(cols), figsize=(2.15 * len(cols), 8.2), sharex="col")
    if len(cols) == 1:
        axes = axes.reshape(4, 1)
    cdfa, cbp = "#1f4fd8", "#9a9a9a"
    band_c, band_s = "#ffc861", "#9fd7c0"
    for j, rung in enumerate(cols):
        runs = {k: load_rung(ROOT, f"{rung}_{k}") for k in ("dfa", "bp")}
        ax_l, ax_a, ax_c, ax_g = axes[0, j], axes[1, j], axes[2, j], axes[3, j]
        Lc = None
        for rule, col in (("bp", cbp), ("dfa", cdfa)):
            for si, (df, meta) in enumerate(runs[rule]):
                nl = len(meta["widths_list"])
                Lc = meta["const_loss"]
                d = df[df.step > 0]
                kw = dict(color=col, lw=1.6 if si == 0 else 0.7, alpha=1.0 if si == 0 else 0.4,
                          label=rule.upper() if si == 0 else None, zorder=3 if rule == "dfa" else 2)
                ax_l.loglog(d.t, d.test_loss, **kw)
                ax_a.semilogx(d.t, d[f"wa_vec_l{nl}"], **kw)
                ax_c.semilogx(d.t, d[f"hcos_l{nl}"], **kw)
                ax_g.semilogx(d.t, d[f"chi_l{nl}"] / d[f"chi_l{nl}"].iloc[0], **kw)
        if Lc:
            ax_l.axhline(Lc, color="#d62728", ls=":", lw=1.2, zorder=1)
            lo = min(float(df[df.step > 0].test_loss.min()) for df, _ in runs["dfa"]) if runs["dfa"] else Lc
            ax_l.set_ylim(max(lo * 0.4, Lc * 3e-4), Lc * 1.8)
        for kind, colr in (("prior-band", band_c), ("sub-prior", band_s)):
            win = pick_window(per, rung, "dfa", kind)
            if win is None or len(win) == 0:
                continue
            if kind == "prior-band" and not bool(win.interior.mode().iloc[0]):
                continue
            t0, t1 = float(win.t0.mean()), float(win.t1.mean())
            for ax in (ax_l, ax_a, ax_c, ax_g):
                ax.axvspan(t0, t1, color=colr, alpha=0.5, lw=0, zorder=0)
        ax_l.set_title(SHORT[rung].replace(" +", "\n+"), fontsize=8.5)
        for ax in (ax_a, ax_c):
            ax.set_ylim(-0.12, 1.06)
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_g.set_ylim(-0.03, 1.15)
        ax_g.set_xlabel("t = step / D", fontsize=8)
        for ax in (ax_l, ax_a, ax_c, ax_g):
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        if j:
            for ax in (ax_a, ax_c, ax_g):
                ax.set_yticklabels([])
    axes[0, 0].set_ylabel("test loss\n(red: constant predictor)", fontsize=8)
    axes[1, 0].set_ylabel("weight alignment\ncos(M, B)", fontsize=8)
    axes[2, 0].set_ylabel("hidden cosine", fontsize=8)
    axes[3, 0].set_ylabel("gate E[phi'^2]\n(relative to init)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=band_c, alpha=0.5),
               plt.Rectangle((0, 0), 1, 1, fc=band_s, alpha=0.5)]
    axes[0, -1].legend(handles, ["plateau at the constant predictor", "plateau below it"],
                       fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("Teacher-student ladder: Refinetti's post-alignment plateau (rung 0) versus our pre-alignment "
                 "common-mode collapse", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    fig.savefig(FIG, dpi=200)
    fig.savefig(FIG.replace(".png", ".pdf"))
    print("wrote", FIG, "and", FIG.replace(".png", ".pdf"))


def main():
    rung_names = [f"{r}_{k}" for r in available() for k in ("dfa", "bp")
                  if os.path.exists(os.path.join(ROOT, f"{r}_{k}"))]
    per = run_table(ROOT, rung_names)
    per.to_csv(os.path.join(ROOT, "plateaus_per_run.csv"), index=False)
    open(os.path.join(ROOT, "SUMMARY.md"), "w").write(summary(per))
    print("wrote", os.path.join(ROOT, "SUMMARY.md"))
    figure(per)


if __name__ == "__main__":
    main()
