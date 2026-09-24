#!/usr/bin/env python3
"""Derived quantities that need per-run time series rather than summary rows:
 - kappa_transient: dose delivered by the end of the prior transient (first step with ||ebar|| < 0.15 ||ebar_0||, else run end)
 - product law restricted to the sweeps that vary only g or ||ebar_0|| at fixed task and head (gain and output-bias sweeps)
 - recovery-law exponents from E5 (log T on log eta_hid, log eta_out, log g) and the constant T sqrt(eta_hid eta_out g)
 - escape-cascade exponents n_l from the growth of Lambda_l after its minimum (E1 base)
 - generality correlation restricted to from-scratch uncured DFA/FA runs (E3 + E1 base/fa)
 - decodability and rank-one anatomy from E7
Writes results/derived_laws.json and prints a report."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cmc.analysis import load_runs, plateau
from cmc.conditions import condition_key
from cmc.reporting import condition_correlation

R = "results"; out = {}

def kappa_transient(df, L, frac=0.15):
    e0 = df.ebar_norm.iloc[0]; idx = np.where(df.ebar_norm.values < frac * e0)[0]
    i = int(idx[0]) if len(idx) else len(df) - 1
    return {f"kt_l{l}": float(df[f"kappa_l{l}"].iloc[i]) for l in range(1, L + 1)} | {"t_transient": int(df.step.iloc[i])}

def n_layers(meta): return len(meta["widths"])

# ---- kappa_transient for every run of E1, E2, E5, E3 (DFA family only)
kt_rows = []
for exp in ["E1_headline", "E2_master", "E5_recovery", "E3_generality", "E4_cures", "E7_anatomy", "E10_nonsat", "E11_cifar", "E12_cifar2"]:
    for f in sorted(glob.glob(f"{R}/{exp}/*/seed*_fb*.csv")):
        if "reduced_" in f: continue
        df = pd.read_csv(f); meta = json.load(open(f.replace(".csv", ".meta.json")))
        if meta["rule"] == "bp": continue
        L = n_layers(meta); r = dict(exp=exp, tag=meta["tag"], seed=meta["seed"], fb_seed=meta["fb_seed"], rule=meta["rule"],
                                   drive=meta["fb_scale"] * df.ebar_norm.iloc[0], ebar0=float(df.ebar_norm.iloc[0]),
                                   maxcos_L=float(df[f"cos_l{L}"].max()), minp300_L=float(df[df.step <= 300][f"p_l{L}"].min()),
                                   maxcos_1=float(df.cos_l1.max()), minp300_1=float(df[df.step <= 300].p_l1.min()), kappa_end_L=float(df[f"kappa_l{L}"].iloc[-1]))
        r.update(kappa_transient(df, L)); r["kt_L"] = r[f"kt_l{L}"]
        r["condition_key"]=condition_key(meta,include_seeds=True);kt_rows.append(r)
KT = pd.DataFrame(kt_rows); KT.to_csv(f"{R}/kappa_transient_per_run.csv", index=False)

def loglog(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0); x, y = np.log10(x[m]), np.log10(y[m])
    if m.sum() < 3: return dict(n=int(m.sum()))
    A = np.vstack([x, np.ones_like(x)]).T; b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    return dict(slope=float(b), intercept=float(a), prefactor=float(10 ** a), r=float(np.corrcoef(x, y)[0, 1]), n=int(m.sum()), fold=float(10 ** (x.max() - x.min())))

# ---- product law on the pure-drive sweeps (E2 tags g* and q*)
e2 = KT[KT.exp == "E2_master"]
pure = e2[e2.tag.str.match(r"^(g[0-9.]+|q[0-9.]+)$")]
# q=0.5 and g=1 label the same physical runs; count each seed only once.
pure = pure[pure.tag != "q0.5"]
out["product_law_pure"] = loglog(pure.drive.values, 1 - pure.maxcos_L.values)
out["product_law_pure_layer1"] = loglog(pure.drive.values, 1 - pure.maxcos_1.values)
# interleaving check: residuals of g-sweep vs q-sweep from the common fit
b, a = out["product_law_pure"]["slope"], out["product_law_pure"]["intercept"]
res = np.log10(1 - pure.maxcos_L.values) - (a + b * np.log10(pure.drive.values))
out["product_law_pure_resid_gsweep"] = float(np.mean(res[pure.tag.str.startswith("g").values])); out["product_law_pure_resid_qsweep"] = float(np.mean(res[pure.tag.str.startswith("q").values]))
# including class-count and softmax-imbalance sweeps for the appendix
mixed = e2[e2.tag.str.match(r"^(g[0-9.]+|q[0-9.]+|C[0-9]+|smimb[0-9.]+)$")]
keys=[]
for row in mixed.itertuples():
    meta=json.load(open(f"{R}/{row.exp}/{row.tag}/seed{row.seed}_fb{row.fb_seed}.meta.json"))
    keys.append(condition_key(meta,include_seeds=True))
mixed=mixed.loc[~pd.Series(keys,index=mixed.index).duplicated()]
out["product_law_mixed"] = loglog(mixed.drive.values, 1 - mixed.maxcos_L.values)
# participation vs transient kappa (all DFA runs of E1+E2+E5, deepest layer, kappa_t > 3)
allk = KT[KT.exp.isin(["E1_headline", "E2_master", "E5_recovery"]) & KT.rule.isin(["dfa"]) & ~KT.tag.str.contains(r"center|whiten|gated|bn|muon|aligned")].drop_duplicates("condition_key")
sat = allk[allk.kt_L > 3]
out["p_kappa_asymptote_median_kp"] = float(np.median(sat.minp300_L * sat.kt_L)); out["p_kappa_n"] = int(len(sat))
out["p_kappa_loglog_all"] = loglog(allk.kt_L.values, allk.minp300_L.values)

# ---- recovery law on E5 (plateau duration; exponents by multiple regression)
rows = []
for f in sorted(glob.glob(f"{R}/E5_recovery/*/seed*_fb*.csv")):
    df = pd.read_csv(f); meta = json.load(open(f.replace(".csv", ".meta.json"))); pl = plateau(df, meta["prior_loss"], meta["prior"])
    if pl and pl["exit"]:
        rows.append(dict(tag=meta["tag"], eh=meta["lr"] * meta["hid_lr_mult"], eo=meta["lr"] * meta["out_lr_mult"], g=meta["fb_scale"], T=pl["duration"], Texit=pl["exit"], onset=pl["onset"]))
E5 = pd.DataFrame(rows); E5.to_csv(f"{R}/E5_recovery/plateaus.csv", index=False)
for Tcol in ("T", "Texit"):
    X = np.vstack([np.log10(E5.eh), np.log10(E5.eo), np.log10(E5.g), np.ones(len(E5))]).T; y = np.log10(E5[Tcol])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]; pred = X @ coef
    v = E5[Tcol] * np.sqrt(E5.eh * E5.eo * E5.g)
    out[f"recovery_{Tcol}"] = dict(exp_hid=float(coef[0]), exp_out=float(coef[1]), exp_g=float(coef[2]), r2=float(1 - np.var(y - pred) / np.var(y)), n=int(len(E5)),
                                   const_geomean=float(np.exp(np.mean(np.log(v)))), const_geosd=float(np.exp(np.std(np.log(v)))))

# ---- escape-cascade exponents from E1 base: Lambda_l(t) after its minimum
def cascade_exponents(df, L):
    exps = {}
    for l in range(1, L + 1):
        lam = df[f"lam_l{l}"].values; s = df.step.values; m0 = int(np.argmin(lam[(s >= 30)])) + int(np.sum(s < 30)); s0 = s[m0]
        # fit window: from where Lambda has risen 20% above its minimum to where it reaches 5x the minimum (or run end)
        y = lam[m0:] - lam[m0]; x = s[m0:] - s0; ok = (y > 0.2 * lam[m0]) & (y < 20 * lam[m0]) & (x > 0)
        if ok.sum() >= 5:
            b = np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0]; exps[f"n_l{l}"] = float(b)
        exps[f"lam_min_l{l}"] = float(lam[m0]); exps[f"lam_drop_l{l}"] = float(lam[0] / lam[m0]) if lam[m0] > 0 else float("nan"); exps[f"t_min_l{l}"] = int(s0)
    return exps
casc = [cascade_exponents(df, n_layers(m)) for df, m in load_runs(f"{R}/E1_headline/base/seed*_fb*.csv")]
C = pd.DataFrame(casc); out["cascade"] = {c: [float(C[c].mean()), float(C[c].std())] for c in C.columns}

# ---- generality correlation: from-scratch, uncured DFA/FA runs of E3 plus E1 base/fa
rows = []
for exp, pat in [("E3_generality", "*"), ("E1_headline", "base"), ("E1_headline", "fa"), ("E1_headline", "softmax"), ("E1_headline", "softmax_imb07")]:
    for f in sorted(glob.glob(f"{R}/{exp}/{pat}/seed*_fb*.csv")):
        if "reduced_" in f: continue
        meta = json.load(open(f.replace(".csv", ".meta.json")))
        if meta["rule"] == "bp" or meta["center"] != "none": continue
        df = pd.read_csv(f); L = n_layers(meta); r10 = df[df.step == 10]
        rows.append(dict(exp=exp, tag=meta["tag"], condition_key=condition_key(meta,include_seeds=True), physical_condition=condition_key(meta), ratio10=float(r10.ebar_norm.iloc[0] / r10.etil_norm.iloc[0]), frac0=float(df.ebar_norm.iloc[0] ** 2 / (df.ebar_norm.iloc[0] ** 2 + df.etil_norm.iloc[0] ** 2)),
                         collapsed_steps=int(((df[f"cos_l{L}"] > 0.9) & (df.step <= 2000)).sum() * 10), maxcos=float(df[f"cos_l{L}"].max()), kt=float(kappa_transient(df, L)[f"kt_l{L}"])))
G = pd.DataFrame(rows).drop_duplicates("condition_key"); G.to_csv(f"{R}/generality_corr_rows.csv", index=False)
out["generality_ratio_vs_collapsed_steps"] = condition_correlation(G.ratio10, G.collapsed_steps, G.physical_condition)
out["generality_ratio_vs_maxcos"] = condition_correlation(G.ratio10, G.maxcos, G.physical_condition)
out["generality_kt_vs_maxcos_spearman"] = float(pd.Series(G.kt).corr(pd.Series(G.maxcos), method="spearman"))
out["generality_conditions"] = sorted(G.tag.unique().tolist())

# ---- anatomy (E7): decodability drop, rank-one variance fraction, PC1 fraction at the collapse peak, per tag
an = {}
for tag in sorted(os.listdir(f"{R}/E7_anatomy")):
    runs = load_runs(f"{R}/E7_anatomy/{tag}/seed*_fb*.csv")
    if not runs: continue
    vals = []
    for df, meta in runs:
        L = n_layers(meta); h = df.dropna(subset=[f"R_l{L}"]); k = int(h[f"cos_l{L}"].values.argmax()); hp = h.iloc[k]
        vals.append(dict(R_drop=float(h[f"R_l{L}"].iloc[0] / max(h[f"R_l{L}"].min(), 1e-9)), R_ratio_at_peak=float(hp[f"R_l{L}"] / h[f"R_l{L}"].iloc[0]),
                         lam_drop=float(h[f"lam_l{L}"].iloc[0] / max(h[f"lam_l{L}"].min(), 1e-12)), r1frac_peak=float(h[f"r1frac_l{L}"].max()) if f"r1frac_l{L}" in h else float("nan"),
                         r1frac_l1_peak=float(h["r1frac_l1"].max()) if "r1frac_l1" in h else float("nan"), pc1_peak=float(h[f"pc1_l{L}"].max()), pc1_init=float(h[f"pc1_l{L}"].iloc[0])))
    V = pd.DataFrame(vals); an[tag] = {c: [float(V[c].mean()), float(V[c].std())] for c in V.columns}
out["anatomy"] = an

json.dump(out, open(f"{R}/derived_laws.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])

def unsigned_dose(df, meta, t0, t1=None):
    """g * eta_hid * sum_t ||ebar_t|| (H_{L-1}+1) over t0 < t <= t1, from the batch mean error logged every probe interval
    (the unsigned counterpart of the online kappa_L, for runs whose mean-error direction changes mid-run)."""
    L = n_layers(meta); sub = df[(df.step > t0) & ((df.step <= t1) if t1 is not None else True)]
    if not len(sub) or "batch_ebar_norm" not in sub: return float("nan")
    dt = np.diff(np.concatenate([[t0], sub.step.values])); Hprev = sub[f"H_l{L-1}"].values if L > 1 else np.full(len(sub), np.nan)
    return float(meta["fb_scale"] * meta["lr"] * meta.get("hid_lr_mult", 1.0) * np.nansum(dt * sub.batch_ebar_norm.values * (Hprev + 1)))

# ---- E8: label shift after training (softmax), and E9: head-initialization tests
def shift_stats(tag):
    vals = []
    for f in sorted(glob.glob(f"{R}/E8_shift/{tag}/seed*_fb*.csv")):
        df = pd.read_csv(f); meta = json.load(open(f.replace(".csv", ".meta.json"))); L = n_layers(meta); s0 = meta["shift_step"]
        pre = df[df.step <= s0]; post = df[df.step > s0]
        at = lambda st, col: float(df[df.step == st][col].iloc[0]) if (df.step == st).any() else float("nan")
        vals.append(dict(cos_pre=float(pre[f"cos_l{L}"].iloc[-1]), cos_post_max=float(post[f"cos_l{L}"].max()), cos_post_50=at(s0 + 50, f"cos_l{L}"),
                         bal_pre=at(s0, "probe_bal_acc"), bal_post_250=at(s0 + 250, "probe_bal_acc"), bal_post_end=float(df.probe_bal_acc.iloc[-1]),
                         acc_pre=at(s0, "probe_acc"), acc_end=float(df.probe_acc.iloc[-1]),
                         ebar_shift_10=at(s0 + 10, "batch_ebar_norm"), ebar_shift_50=at(s0 + 50, "batch_ebar_norm"), ebar_shift_250=at(s0 + 250, "batch_ebar_norm"),
                         minp_post=float(post[f"p_l{L}"].min()), minp_pre=float(pre[f"p_l{L}"].min()),
                         kappa_post=float(df[f"kappa_l{L}"].iloc[-1] - at(s0, f"kappa_l{L}")), kappa_pre=at(s0, f"kappa_l{L}"),
                         kappa_post_300=at(s0 + 300, f"kappa_l{L}") - at(s0, f"kappa_l{L}"),
                         ebar_shift_0=at(s0, "batch_ebar_norm"), dose_post=unsigned_dose(df, meta, s0), dose_post_300=unsigned_dose(df, meta, s0, s0 + 300),
                         H_L_at_shift=at(s0, f"H_l{L}")))
    V = pd.DataFrame(vals); return {c: [float(V[c].mean()), float(V[c].std())] for c in V.columns} if len(V) else {}
out["shift"] = {tag: shift_stats(tag) for tag in ["shift1500", "shift1500_bp", "shift300", "shift300_bp", "headreset1500"] if os.path.isdir(f"{R}/E8_shift/{tag}")}
hi = {}
for tag in sorted(os.listdir(f"{R}/E9_headinit")) if os.path.isdir(f"{R}/E9_headinit") else []:
    runs = load_runs(f"{R}/E9_headinit/{tag}/seed*_fb*.csv")
    if not runs: continue
    V = pd.DataFrame([dict(ebar0=float(df.ebar_norm.iloc[0]), maxcos=float(df[f"cos_l{n_layers(m)}"].max()), minp300=float(df[df.step <= 300][f"p_l{n_layers(m)}"].min()),
                           acc_1000=float(df[df.step == 1000].probe_acc.iloc[0]) if (df.step == 1000).any() else np.nan, kt=kappa_transient(df, n_layers(m))[f"kt_l{n_layers(m)}"]) for df, m in runs])
    hi[tag] = {c: [float(V[c].mean()), float(V[c].std())] for c in V.columns}
out["headinit"] = hi
# selected-set turnover from the snapshot analysis
sa = f"{R}/E1_headline/base/snapshot_analysis.csv"
if os.path.exists(sa):
    S = pd.read_csv(sa); out["turnover"] = {c: [float(S[c].mean()), float(S[c].std())] for c in S.columns if c.startswith("jaccard") or c.startswith("drift_spearman") or c.startswith("spearman_u")}
json.dump(out, open(f"{R}/derived_laws.json", "w"), indent=1, default=float)
print("shift:", json.dumps(out["shift"], indent=1, default=float)); print("headinit:", json.dumps(out["headinit"], indent=1, default=float))
print("turnover:", json.dumps(out.get("turnover"), indent=1, default=float))
