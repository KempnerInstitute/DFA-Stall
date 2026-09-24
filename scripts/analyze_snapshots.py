#!/usr/bin/env python3
"""Per-unit analyses from snapshots.npz: drift-vs-common-mode-coefficient correlation and selected-set turnover.
Usage: python scripts/analyze_snapshots.py results/E1_headline/base  [--early 100] [--plateau 120] [--late 1000 3000]
Writes <dir>/snapshot_analysis.csv (one row per run) and prints means."""
import argparse, glob, os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr

def analyze(path, early=100, plateau=120, late=(1000, 3000), topfrac=0.2):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "seed*_fb*.snapshots.npz"))):
        z = np.load(f); r = dict(run=os.path.basename(f).replace(".snapshots.npz", ""))
        layers = sorted({int(k.split("_l")[1]) for k in z.files if k.startswith("step0/mu_l")})
        for l in layers:
            b1 = z.get(f"step0/B1_l{l}")
            if b1 is not None and f"step{early}/mu_l{l}" in z:
                r[f"drift_spearman_l{l}"] = spearmanr(z[f"step{early}/mu_l{l}"] - z[f"step0/mu_l{l}"], b1).correlation
                r[f"drift_std_l{l}"] = float(np.std(z[f"step{early}/mu_l{l}"] - z[f"step0/mu_l{l}"]))
            if f"step{plateau}/u_l{l}" in z:
                u0 = z[f"step{plateau}/u_l{l}"]; d = len(u0); k = int(topfrac * d); a = set(np.argsort(-u0)[:k]); r[f"jaccard_null"] = k / (2 * d - k)
                for t in late:
                    if f"step{t}/u_l{l}" in z:
                        b = set(np.argsort(-z[f"step{t}/u_l{l}"])[:k]); r[f"jaccard_{plateau}_{t}_l{l}"] = len(a & b) / len(a | b)
                if b1 is not None: r[f"spearman_u_plateau_vs_absB1_l{l}"] = spearmanr(u0, -np.abs(b1)).correlation
            if f"step{plateau}/sigma_l{l}" in z:
                r[f"sigma_plateau_l{l}"] = float(np.mean(z[f"step{plateau}/sigma_l{l}"])); r[f"sigma_init_l{l}"] = float(np.mean(z[f"step0/sigma_l{l}"]))
                r[f"muspread_plateau_l{l}"] = float(np.std(z[f"step{plateau}/mu_l{l}"])); r[f"muspread_init_l{l}"] = float(np.std(z[f"step0/mu_l{l}"]))
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(path, "snapshot_analysis.csv"), index=False); return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--early", type=int, default=100); ap.add_argument("--plateau", type=int, default=120)
    ap.add_argument("--late", type=int, nargs="+", default=[1000, 3000]); a = ap.parse_args()
    df = analyze(a.path, a.early, a.plateau, tuple(a.late))
    num = df.select_dtypes("number"); print(pd.DataFrame({"mean": num.mean(), "sd": num.std(), "n": num.count()}).round(3).to_string())
