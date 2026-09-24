#!/usr/bin/env python3
"""Reconstruct gate participation from the saved per-unit preactivation moments (mu_i, sigma_i) of the baseline runs.

For every snapshot of every baseline trajectory and every layer: u_i^Gauss = E[sech^4 Z], Z ~ N(mu_i, sigma_i^2), by 96-node
Gauss-Hermite quadrature, p^Gauss = (sum u)^2 / (d sum u^2); the lognormal closed form p^ln = exp(-16 Var[log cosh mu_i]); and the
measured p from the saved gate energies. Two additional reconstructions retain measured means with one shared RMS spread per
layer, or set every mean to zero while retaining measured spreads. These are moment substitutions, not training interventions.
Also the Jaccard overlap of the top-20% gate-energy set at each snapshot with the set at
step 120 (turnover). Writes results/gauss_reconstruction.csv (per run, step, layer) and results/gauss_reconstruction.json
(medians over runs and layers per step). Usage: PYTHONPATH=src python3 scripts/gauss_reconstruction.py"""
from __future__ import annotations
import glob, json, os, re
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = sorted(glob.glob(os.path.join(ROOT, "results", "E1_headline", "base", "seed*_fb*.snapshots.npz")))
X, W = np.polynomial.hermite.hermgauss(96)
REF_STEP, TOPFRAC = 120, 0.2


def gauss_u(mu, sigma):
    z = mu[:, None] + np.sqrt(2.0) * sigma[:, None] * X[None, :]
    return (W[None, :] * np.cosh(np.clip(z, -30, 30)) ** -4).sum(1) / np.sqrt(np.pi)


def participation(u): return float(u.sum() ** 2 / (len(u) * (u ** 2).sum()))


rows = []
for f in RUNS:
    z = np.load(f); run = os.path.basename(f).replace(".snapshots.npz", "")
    steps = sorted({int(m.group(1)) for k in z.files for m in [re.match(r"step(\d+)/", k)] if m})
    layers = sorted({int(m.group(1)) for k in z.files for m in [re.match(r"step\d+/mu_l(\d+)", k)] if m})
    for l in layers:
        u_ref = z[f"step{REF_STEP}/u_l{l}"] if f"step{REF_STEP}/u_l{l}" in z.files else None
        k = int(TOPFRAC * len(u_ref)) if u_ref is not None else 0; ref = set(np.argsort(-u_ref)[:k]) if u_ref is not None else set()
        for t in steps:
            mu, sg, u = z[f"step{t}/mu_l{l}"], z[f"step{t}/sigma_l{l}"], z[f"step{t}/u_l{l}"]
            p_meas = participation(u); p_g = participation(gauss_u(mu, sg)); V = 16.0 * np.var(np.log(np.cosh(np.clip(mu, -30, 30))))
            shared = np.full(len(sg), np.sqrt(np.mean(sg.astype(float) ** 2)))
            p_shared = participation(gauss_u(mu.astype(float), shared))
            p_zero = participation(gauss_u(np.zeros(len(mu)), sg.astype(float)))
            nlp = -np.log(p_meas)
            jac = (len(ref & set(np.argsort(-u)[:k])) / len(ref | set(np.argsort(-u)[:k]))) if ref else np.nan
            rows.append(dict(run=run, step=t, layer=l, p_meas=p_meas, p_gauss=p_g, p_lognormal=float(np.exp(-V)),
                             p_shared_spread=p_shared, p_zero_mean=p_zero,
                             ratio_shared_spread=(-np.log(p_shared)) / nlp if nlp > 0 else np.nan,
                             ratio_zero_mean=(-np.log(p_zero)) / nlp if nlp > 0 else np.nan,
                             ratio_gauss=(-np.log(p_g)) / nlp if nlp > 0 else np.nan, ratio_lognormal=V / nlp if nlp > 0 else np.nan,
                             mu_spread=float(mu.std()), sigma_mean=float(sg.mean()), jaccard_vs_ref=jac, jaccard_null=k / (2 * len(u) - k) if k else np.nan))
D = pd.DataFrame(rows); D.to_csv(os.path.join(ROOT, "results", "gauss_reconstruction.csv"), index=False)
S = {"n_runs": int(D.run.nunique()), "steps": sorted(D.step.unique().tolist()),
     "median_ratio_gauss": {int(t): float(g.ratio_gauss.median()) for t, g in D.groupby("step")},
     "median_ratio_lognormal": {int(t): float(g.ratio_lognormal.median()) for t, g in D.groupby("step")},
     "median_ratio_shared_spread": {int(t): float(g.ratio_shared_spread.median()) for t, g in D.groupby("step")},
     "median_ratio_zero_mean": {int(t): float(g.ratio_zero_mean.median()) for t, g in D.groupby("step")},
     "ablation_definitions": {
         "shared_spread": "Measured mu_i; sigma_i replaced by sqrt(mean_j sigma_j^2) at the same layer and step",
         "zero_mean": "All mu_i set to zero, retaining the measured sigma_i",
         "scope": "Gaussian reconstruction from saved moments, not an intervention on training"},
     "median_ratio_gauss_early": float(D[D.step.between(50, 300)].ratio_gauss.median()),
     "median_ratio_lognormal_early": float(D[D.step.between(50, 300)].ratio_lognormal.median()),
     "median_ratio_gauss_3000": float(D[D.step == 3000].ratio_gauss.median()), "median_ratio_lognormal_3000": float(D[D.step == 3000].ratio_lognormal.median()),
     "jaccard_by_step_layer": {f"l{l}": {int(t): float(g.jaccard_vs_ref.mean()) for t, g in dl.groupby("step")} for l, dl in D.groupby("layer")},
     "jaccard_null": float(D.jaccard_null.dropna().iloc[0]) if D.jaccard_null.notna().any() else None}
json.dump(S, open(os.path.join(ROOT, "results", "gauss_reconstruction.json"), "w"), indent=1)
print(f"{len(D)} rows from {S['n_runs']} runs; median Gaussian ratio 50-300: {S['median_ratio_gauss_early']:.3f}, at 3000: {S['median_ratio_gauss_3000']:.3f}; "
      f"lognormal 50-300: {S['median_ratio_lognormal_early']:.2f}, at 3000: {S['median_ratio_lognormal_3000']:.2f}")
print(pd.DataFrame({"gauss": S["median_ratio_gauss"], "lognormal": S["median_ratio_lognormal"]}).round(3).T.to_string())
