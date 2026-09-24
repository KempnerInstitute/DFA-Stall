#!/usr/bin/env python3
"""Write the appendix tables (paper/tables/*.tex) from results/<exp>/summary.csv. One row per condition: number of runs,
peak hidden cosine of the deepest layer, its windowed participation minimum, plateau onset/exit, and probe accuracy at
1500 and 3000 steps (mean +- sd over runs). Usage: PYTHONPATH=src python3 scripts/make_tables.py"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "tables")
TABLES = [  # (experiment, file stem, caption, label)
    ("E1_headline", "E1", "MNIST baseline and mean-drive controls (MNIST, three hidden layers of width 300, tanh, sigmoid readout unless the tag says otherwise). Fifteen trajectories for DFA conditions, five for BP.", "tab:E1"),
    ("E2_master", "E2", "Parameter sweeps: feedback gain (g), output-bias initialization (q), class count (C), softmax imbalance (smimb), learning rate (lr), width (w). Three (seed, feedback) pairs per tag.", "tab:E2"),
    ("E3_generality", "E3", "Generality: rules, depths, activations, architectures and readouts, each with a matched BP control (\\texttt{\\_bp}), with the broadcast error centered before projection (\\texttt{\\_centere}) and with the projected teaching signal centered (\\texttt{\\_center}).", "tab:E3"),
    ("E4_cures", "E4", "Intervention comparison. Tags are \\texttt{<setting>\\_\\_<intervention>}; \\texttt{whiten\\_lam<x>} is the local-error conditioner at relative ridge $\\lambda=x$ (plain \\texttt{whiten}: $\\lambda=0.1$); \\texttt{gated} never released its gate and duplicates \\texttt{center\\_e}.", "tab:E4"),
    ("E5_recovery", "E5", "Recovery law: hidden learning-rate multiplier (h), output learning-rate multiplier (o) and feedback gain (g).", "tab:E5"),
    ("E7_anatomy", "E7", "Representation and gradient measurements: activity-scale-normalized decoding and mean-update statistics.", "tab:E7"),
    ("E8_shift", "E8", "Label shift imposed mid-run on a balanced-softmax network (class-0 mixture weight $p_0=0.7$ after the shift step).", "tab:E8"),
    ("E9_headinit", "E9", "Readout-initialization controls: readout and target combinations that give a nonzero initial mean error.", "tab:E9"),
    ("E10_nonsat", "E10", "Centered broadcast error with non-saturating units at smaller learning rates and with a faster readout.", "tab:E10"),
    ("E11_cifar", "E11", "CIFAR-10: the small convolutional network (tanh or ReLU) and the flattened MLP under the sigmoid readout, a balanced softmax with standardized inputs, and an imbalanced softmax, with interventions and matched BP controls.", "tab:E11"),
    ("E12_cifar2", "E12", "CIFAR-10 controls: mean-logit calibration at initialization (\\texttt{calib}), centered error plus centered inputs with a faster readout, and the ReLU CNN at smaller learning rates.", "tab:E12"),
]


def esc(s): return str(s).replace("_", "\\_")


def pm(df, col, fmt="{:.2f}"):
    m, s = f"{col}_mean", f"{col}_sd"
    if m not in df.columns: return ["--"] * len(df)
    out = []
    for a, b in zip(df[m], df[s] if s in df.columns else [np.nan] * len(df)):
        if not np.isfinite(a): out.append("--"); continue
        out.append(fmt.format(a) + (r"$\pm$" + fmt.format(b) if np.isfinite(b) and len(df) else ""))
    return out


def deepest(df, stem):
    """Per-row value of <stem>_l<L>_mean/sd at the row's own depth, as a pm string."""
    out = []
    for _, r in df.iterrows():
        L = int(r.get("n_layers_mean", 3) or 3); m = r.get(f"{stem}_l{L}_mean", np.nan); s = r.get(f"{stem}_l{L}_sd", np.nan)
        if L == 0 or not np.isfinite(m):   # CNN rows log the fc block as layer 1
            m = r.get(f"{stem}_l1_mean", np.nan); s = r.get(f"{stem}_l1_sd", np.nan)
        out.append("--" if not np.isfinite(m) else f"{m:.2f}" + (r"$\pm$" + f"{s:.2f}" if np.isfinite(s) else ""))
    return out


def table(exp, stem, caption, label):
    if stem in ('E7','E8'):
        return os.path.join(OUT,f'{stem}.tex')  # experiment-specific tables: review_reporting.py
    f = os.path.join(ROOT, "results", exp, "summary.csv")
    if not os.path.exists(f): print(f"[skip] {exp}: no summary"); return None
    d = pd.read_csv(f).sort_values("tag")
    W = 300
    cols = {"tag": [esc(t) for t in d.tag], "runs": [f"{int(n)}" for n in d.get("n_runs", pd.Series([0] * len(d)))],
            r"peak $\cos_L$": deepest(d, "maxcos"), rf"$\min_{{t\le {W}}} p_L$": deepest(d, f"minp_w{W}"),
            "onset": pm(d, "plateau_onset", "{:.0f}"), "exit": pm(d, "plateau_exit", "{:.0f}"),
            "acc 1500": pm(d, "acc_1500"), "acc 3000": pm(d, "acc_3000")}
    T = pd.DataFrame(cols)
    if stem == 'E2':
        T.rename(columns={'acc 1500':r'initial $\|\bar e\|$'},inplace=True)
        T[r'initial $\|\bar e\|$']=pm(d,'ebar0','{:.3f}')
    if stem == 'E10':
        T.rename(columns={'acc 1500':'final loss'},inplace=True)
        T['final loss']=pm(d,'loss_final','{:.1e}')
    per=pd.read_csv(os.path.join(ROOT,'results',exp,'summary_per_run.csv'))
    for title,column in [('onset','plateau_onset'),('exit','plateau_exit')]:
        counts=per.groupby('tag')[column].count() if column in per else {}
        T[title]=[v+(f' [{int(counts.get(tag,0))}]' if v!='--' else '') for tag,v in zip(d.tag,T[title])]
    caption += ' Timing brackets give the number of contributing runs; an undetected plateau is not assigned zero duration.'
    if stem in ('E1','E2'):
        caption += ' Softmax conditions use globally standardized inputs; the sigmoid baseline uses pixels in $[0,1]$. Table~\\ref{tab:review_heads} gives a comparison with matched preprocessing.'
    lines = [f"% generated by scripts/make_tables.py from results/{exp}/summary.csv", r"{\scriptsize\setlength{\tabcolsep}{3.5pt}", r"\begin{longtable}{l r r r r r r r}",
             rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", " & ".join(T.columns) + r"\\", r"\midrule", r"\endfirsthead",
             r"\toprule", " & ".join(T.columns) + r"\\", r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in T.iterrows(): lines.append(" & ".join(str(v) for v in r.values) + r"\\")
    lines.append(r"\end{longtable}}")
    os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, f"{stem}.tex"); open(p, "w").write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(p, ROOT)} ({len(T)} rows)"); return p


def main():
    written = [table(*t) for t in TABLES]
    idx = os.path.join(OUT, "all.tex")
    with open(idx, "w") as fh:
        fh.write("% generated by scripts/make_tables.py\n")
        for (exp, stem, _, _), p in zip(TABLES, written):
            if p and stem == "E9": fh.write("\\clearpage\n")
            if p: fh.write(f"\\input{{tables/{stem}}}\n")
    print("wrote", os.path.relpath(idx, ROOT))


if __name__ == "__main__":
    sys.exit(main())
