"""Plateau detection, rung classification, SUMMARY.md and the ladder figure for the E6 teacher-student ladder.

Two plateau signatures are distinguished, both measured on the same runs produced by `cmc.ladder`:

  Refinetti-type (post-alignment): a flat window of the loss strictly below the constant-predictor loss, at
    which the weight alignment cos(M, B) has realised more than 80% of the alignment gain it reaches anywhere
    in the run, with no collapse of the representation (mean pairwise hidden cosine below 0.5).
  Common-mode collapse (ours, pre-alignment): a flat window pinned to the constant-predictor loss, at which the
    hidden cosine is above 0.7 and the alignment has realised less than half of that gain.

A "flat window" is a maximal run of log-spaced samples whose local slope d log10 L / d log10 step stays below
`slope_tol`, spanning at least `min_decade` decades once its arrival ramp has been trimmed off. "Interior"
means the smoothed loss fell by more than 5% before the window opened, which separates a genuine plateau from
the trivially flat start of a network initialised at the constant predictor. The alignment statistics are
computed for DFA only: under BP the feedback matrix is unused and cos(M, B) is a null control.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

def load_rung(root, rung):
    """All seeds of one rung: list of (df, meta), sorted by seed."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, rung, "seed*.csv"))):
        mf = f.replace(".csv", ".meta.json")
        if not os.path.exists(mf):
            continue
        out.append((pd.read_csv(f), json.load(open(mf))))
    return out


def _smooth_log(L, k=5):
    return pd.Series(np.log10(np.maximum(L, 1e-30))).rolling(k, center=True, min_periods=1).mean().values


def local_slope(step, logL, span=2.5):
    """d log10 L / d log10 step from the widest pair of samples within a factor `span` on each side."""
    ls = np.log10(np.maximum(step, 1.0))
    s = np.full(len(ls), np.nan)
    for i in range(len(ls)):
        lo = np.searchsorted(ls, ls[i] - np.log10(span), "left")
        hi = np.searchsorted(ls, ls[i] + np.log10(span), "right") - 1
        if hi - lo >= 2 and ls[hi] > ls[lo]:
            s[i] = (logL[hi] - logL[lo]) / (ls[hi] - ls[lo])
    return s


def _merge(flat, ls, gap=0.12):
    """Close gaps shorter than `gap` decades in the flat mask, so that probe noise does not split a plateau."""
    out = flat.copy()
    idx = np.where(flat)[0]
    for k in range(len(idx) - 1):
        i, j = idx[k], idx[k + 1]
        if j > i + 1 and ls[j] - ls[i] <= gap:
            out[i:j] = True
    return out


def _trim(L, i, j, tol=0.05):
    """Trim the arrival ramp off the left edge of a flat window, so that the onset is where the loss reaches the
    plateau rather than where the log-log slope first goes flat. The right edge is already set by the slope."""
    med = float(np.median(L[i:j + 1]))
    while i < j and L[i] > (1 + tol) * med:
        i += 1
    return i, j


def find_plateaus(df, slope_tol=0.20, min_decade=0.45, min_points=5):
    """Maximal windows in which the log-log slope of the test loss stays flat. Returns a list of dicts.

    The slope is measured over a factor-2.5 window on each side, and gaps shorter than 0.12 decades are closed,
    so that a plateau is not split by probe noise. `interior` records whether the smoothed loss had already
    fallen by more than 5% when the window opened, which excludes the trivial flat start of a network that is
    initialised at the constant predictor.
    """
    d = df[df.step > 0].reset_index(drop=True)
    if len(d) < min_points + 2:
        return []
    logL = _smooth_log(d.test_loss.values)
    ls = np.log10(np.maximum(d.step.values, 1.0))
    sl = local_slope(d.step.values, logL)
    flat = _merge(np.isfinite(sl) & (np.abs(sl) < slope_tol), ls)
    out, i = [], 0
    while i < len(flat):
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flat) and flat[j + 1]:
            j += 1
        a, b = _trim(d.test_loss.values, i, j)
        if b - a + 1 >= min_points and ls[b] - ls[a] >= min_decade:
            i, j = a, b
            pre = logL[:i + 1].max()
            drop_before = 1.0 - 10 ** (logL[i] - pre)
            drop_after = 1.0 - 10 ** (logL[j:].min() - logL[j])
            out.append(dict(i0=i, i1=j, step0=int(d.step.values[i]), step1=int(d.step.values[j]),
                            t0=float(d.t.values[i]), t1=float(d.t.values[j]),
                            loss=float(np.median(d.test_loss.values[i:j + 1])),
                            drop_before=float(drop_before), drop_after=float(drop_after),
                            interior=bool(drop_before > 0.05), decades=float(ls[j] - ls[i]), _d=d))
        i = j + 1
    return out


def describe(df, meta, win, rule="dfa"):
    """Diagnostics averaged over a plateau window, plus their run-wide maxima."""
    d = win["_d"]
    nl = len(meta["widths_list"])
    m = slice(win["i0"], win["i1"] + 1)
    Lc = float(meta["const_loss"])
    r = dict(t0=win["t0"], t1=win["t1"], decades=win["decades"], interior=bool(win["interior"]),
             drop_before=win["drop_before"],
             loss=win["loss"], loss_over_const=win["loss"] / Lc, drop_after=win["drop_after"])
    for key in ("wa_vec", "wa_row", "ga", "hcos", "chi", "part", "mu_spread"):
        col = f"{key}_l{nl}"
        if col not in d:
            continue
        v = d[col].values
        r[f"{key}_plateau"] = float(np.mean(v[m]))
        r[f"{key}_max"] = float(np.max(v))
    # fraction of the run's total alignment gain that has been realised at the plateau; undefined when the rule
    # does not build alignment at all (BP, whose cos(M, B) against an unused feedback matrix is a null control)
    wa0 = float(d[f"wa_vec_l{nl}"].values[0])
    gain = r["wa_vec_max"] - wa0
    r["wa_init"] = wa0
    r["wa_ratio"] = (r["wa_vec_plateau"] - wa0) / gain if (rule == "dfa" and gain > 0.05) else float("nan")
    r["acc_plateau"] = float(np.mean(d["acc"].values[m])) if "acc" in d else float("nan")
    r["in_prior_band"] = bool(abs(r["loss_over_const"] - 1.0) < 0.05)
    r["kind"] = "prior-band" if r["in_prior_band"] else ("sub-prior" if r["loss_over_const"] < 0.95 else "above-prior")
    return r


def classify(r):
    """Which signature a plateau window carries. `wa_ratio` is NaN under BP, where cos(M, B) means nothing,
    so a BP window is never given the DFA-specific common-mode label however collapsed its representation is."""
    hc = r.get("hcos_plateau", 0.0)
    wr = r["wa_ratio"]
    if r["in_prior_band"]:
        if not r["interior"]:
            return "flat start at the constant predictor"
        if np.isnan(wr):
            return "prior saddle with partial collapse" if hc > 0.5 else "prior saddle, no collapse"
        if hc > 0.7 and wr < 0.5:
            return "common-mode collapse"
        if hc < 0.5:
            return "prior saddle, no collapse"
        return "prior-band plateau, unclassified"
    if r["loss_over_const"] < 0.85 and (not np.isnan(wr)) and wr > 0.8 and hc < 0.5:
        return "Refinetti post-alignment"
    if np.isnan(wr):
        return "sub-prior plateau, alignment not defined"
    return "sub-prior plateau, unclassified"


def run_table(root, rungs):
    """One row per (rung, seed, plateau window)."""
    rows = []
    for rung in rungs:
        for df, meta in load_rung(root, rung):
            wins = find_plateaus(df)
            nl = len(meta["widths_list"])
            base = dict(rung=rung, seed=meta["seed"], rule=meta["rule"], const_loss=meta["const_loss"],
                        final_loss=float(df.test_loss.iloc[-1]),
                        hcos_run_max=float(df[f"hcos_l{nl}"].max()),
                        part_run_min=float(df[f"part_l{nl}"].min()),
                        chi_run_min=float(df[f"chi_l{nl}"].min() / df[f"chi_l{nl}"].iloc[0]),
                        wa_run_max=float(df[f"wa_vec_l{nl}"].max()),
                        wa_init=float(df[f"wa_vec_l{nl}"].iloc[0]), n_plateau=len(wins))
            if not wins:
                rows.append(dict(base, window=-1, verdict="no interior plateau"))
                continue
            for k, w in enumerate(wins):
                r = describe(df, meta, w, rule=meta["rule"])
                rows.append(dict(base, window=k, **r, verdict=classify(r)))
    return pd.DataFrame(rows)
