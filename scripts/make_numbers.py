#!/usr/bin/env python3
"""Generate paper/numbers.tex: one LaTeX macro per number the manuscript quotes, each with a comment naming the CSV
and column it came from. Every macro in the registry is always defined; a number whose experiment has not produced
data yet is emitted as '??', so the paper compiles at any stage of the sweeps.

Usage: PYTHONPATH=src python3 scripts/make_numbers.py [--results results] [--out paper/numbers.tex]
Run scripts/summarize_all.py first: values are read from results/<exp>/summary_per_run.csv, from the run CSVs and
snapshots for trajectory quantities, and from results/reduced/ for the reduced model.
The script also reports macros that paper/sections/*.tex uses but the registry does not define.
"""
from __future__ import annotations
import argparse, datetime, glob, json, os, re, sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cmc import analysis as A

MISSING = "??"
E1, E2, E3, E4, E5 = "E1_headline", "E2_master", "E3_generality", "E4_cures", "E5_recovery"
E10 = "E10_nonsat"
E11 = "E11_cifar"
E12 = "E12_cifar2"
W = A.P_WINDOW


class Ctx:
    """Lazy, failure-tolerant access to the summarized results."""

    def __init__(self, root):
        self.root = root; self._per = {}; self._traj = {}

    def per(self, exp):
        if exp not in self._per: self._per[exp] = A.load_per_run(self.root, exp)
        return self._per[exp]

    def exp_dir(self, exp):
        return os.path.join(self.root, exp)

    def layers(self, exp, tag):
        return A.n_layers_of(self.per(exp), tag)

    def traj(self, exp, tag, col):
        """Mean trajectory of one column over the runs of a tag, cached; (steps, mean) or None."""
        key = (exp, tag, col)
        if key not in self._traj:
            r = None
            if os.path.isdir(os.path.join(self.exp_dir(exp), tag)):
                r = A.mean_traj(self.exp_dir(exp), tag, col)
            self._traj[key] = None if r is None else (r[0], r[1])
        return self._traj[key]


@dataclass
class Macro:
    """One LaTeX macro: its name, a one-line description, the function that produces (value, provenance), and the
    print format applied when the value is numeric."""
    name: str
    desc: str
    fn: Callable[[Ctx], Tuple[Optional[object], str]]
    fmt: str = field(default="{:.3f}")


def _fmt(v, fmt):
    """Format a value; strings pass through, non-finite numbers become '??'."""
    if v is None: return MISSING
    if isinstance(v, str): return v or MISSING
    try: f = float(v)
    except (TypeError, ValueError): return str(v)
    return MISSING if not np.isfinite(f) else fmt.format(f)


# ------------------------------------------------------------------ small source helpers (each returns fn(ctx))
def stat(exp, tag, col, how="mean", scale=1.0):
    """One column of one tag from the per-run summary; '{L}' in col becomes the tag's deepest layer index."""
    def fn(ctx):
        per = ctx.per(exp)
        c = col.replace("{L}", str(ctx.layers(exp, tag))) if "{L}" in col else col
        src = f"results/{exp}/summary_per_run.csv column {c} (tag {tag}, {how} over runs)"
        if per is None: return None, src + " [experiment missing]"
        v = A.col_stat(per, tag, c, how)
        return (v * scale if np.isfinite(v) else v), src
    return fn


def diff(exp, tag_a, tag_b, col, scale=1.0):
    """Difference of one column between two tags (cure minus control)."""
    def fn(ctx):
        per = ctx.per(exp)
        ca = col.replace("{L}", str(ctx.layers(exp, tag_a))); cb = col.replace("{L}", str(ctx.layers(exp, tag_b)))
        src = f"results/{exp}/summary_per_run.csv column {col} (mean of tag {tag_a} minus tag {tag_b})"
        if per is None: return None, src + " [experiment missing]"
        return (A.col_stat(per, tag_a, ca) - A.col_stat(per, tag_b, cb)) * scale, src
    return fn


def ratio(exp, tag_a, tag_b, col):
    """Ratio of one column between two tags."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv column {col} (mean of tag {tag_a} over tag {tag_b})"
        if per is None: return None, src + " [experiment missing]"
        b = A.col_stat(per, tag_b, col)
        return (A.col_stat(per, tag_a, col) / b) if b else float("nan"), src
    return fn


def const(value, note):
    """A number fixed by the protocol or available in closed form."""
    return lambda ctx: (value, note)


def best_of(*fns):
    """First alternative that yields a usable value (a quantity may live under more than one tag)."""
    def fn(ctx):
        last = (None, "no source")
        for f in fns:
            v, src = f(ctx)
            last = (v, src)
            if isinstance(v, str) and v: return v, src
            if v is not None and not isinstance(v, str) and np.isfinite(float(v)): return v, src
        return last
    return fn


def n_runs(exp, tag=None):
    """Number of runs behind a tag, or behind a whole experiment."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv (row count{'' if tag is None else f', tag {tag}'})"
        if per is None: return None, src + " [experiment missing]"
        if tag is None: return float(len(per)), src
        sub = A.tag_rows(per, tag)
        return (float(len(sub)) if sub is not None else None), src
    return fn


def total_runs(exps=(E1, E2, E3, E4, E5, "E7_anatomy", "E8_shift", "E9_headinit", E10, E11, E12)):
    """Total number of runs over every experiment of the paper."""
    def fn(ctx):
        src = "results/*/summary_per_run.csv (total row count over " + ", ".join(exps) + ")"
        n = sum(len(ctx.per(e)) for e in exps if ctx.per(e) is not None)
        return (float(n) if n else None), src
    return fn


# ------------------------------------------------------------------ per-layer lists, ranges, trajectories
def layers_str(exp, tag, stem, fmt="{:.2f}", sep="/"):
    """Per-layer values of one summary column joined for the text, e.g. '0.71/0.35/0.23'."""
    def fn(ctx):
        per = ctx.per(exp); n = ctx.layers(exp, tag)
        src = f"results/{exp}/summary_per_run.csv columns {stem}_l1..{stem}_l{n or '?'} (tag {tag}, mean over runs)"
        if per is None or not n: return None, src + " [experiment or tag missing]"
        vals = [A.col_stat(per, tag, f"{stem}_l{l}") for l in range(1, n + 1)]
        if not any(np.isfinite(v) for v in vals): return None, src + " [no finite values]"
        return sep.join(MISSING if not np.isfinite(v) else fmt.format(v) for v in vals), src
    return fn


def range_str(exp, tags, col, fmt="{:.3f}", sep="--"):
    """'min--max' of one column across a set of tags."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv column {col} (range over tags {', '.join(tags)})"
        if per is None: return None, src + " [experiment missing]"
        vals = []
        for t in tags:
            c = col.replace("{L}", str(ctx.layers(exp, t))) if "{L}" in col else col
            v = A.col_stat(per, t, c)
            if np.isfinite(v): vals.append(v)
        if not vals: return None, src + " [no finite values]"
        return fmt.format(min(vals)) + sep + fmt.format(max(vals)), src
    return fn


def fold_range(exp, tags, col, fmt="{:.0f}"):
    """'50-fold': the ratio of the largest to the smallest value of a column across tags."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv column {col} (max/min over tags {', '.join(tags)})"
        if per is None: return None, src + " [experiment missing]"
        vals = [A.col_stat(per, t, col) for t in tags]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        if len(vals) < 2: return None, src + " [fewer than two tags]"
        return fmt.format(max(vals) / min(vals)) + "-fold", src
    return fn


def at_step(exp, tag, col, step):
    """A trajectory column at a fixed step (mean over the runs of a tag)."""
    def fn(ctx):
        src = f"results/{exp}/{tag}/seed*_fb*.csv column {col} at step {step} (mean over runs)"
        t = ctx.traj(exp, tag, col.replace("{L}", str(ctx.layers(exp, tag))))
        if t is None: return None, src + " [tag or column missing]"
        s, m = t
        ok = np.isfinite(m) & (s <= step)
        if not ok.any(): return None, src + " [run shorter than that step]"
        return float(m[ok][-1]), src
    return fn


def extremum(exp, tag, col, how="max", lo=0, hi=None):
    """Extremum of a trajectory column inside a step window (mean trajectory over runs)."""
    def fn(ctx):
        c = col.replace("{L}", str(ctx.layers(exp, tag)))
        src = (f"results/{exp}/{tag}/seed*_fb*.csv column {c} ({how} over steps "
               f"{lo}..{'end' if hi is None else hi}, mean over runs)")
        t = ctx.traj(exp, tag, c)
        if t is None: return None, src + " [tag or column missing]"
        s, m = t
        keep = (s >= lo) & (np.isfinite(m)) & (True if hi is None else (s <= hi))
        if not keep.any(): return None, src + " [no samples in the window]"
        return float(m[keep].max() if how == "max" else m[keep].min()), src
    return fn


# ------------------------------------------------------------------ fits, correlations and derived laws
def _deep(sub, stem):
    out = []
    for _, r in sub.iterrows():
        n = int(r.get("n_layers", 0) or 0); c = f"{stem}_l{n}"
        try: out.append(float(r[c]) if n and c in sub.columns else np.nan)
        except (TypeError, ValueError): out.append(np.nan)
    return np.asarray(out, float)


def product_law(field_name, exp=E2):
    """log-log fit of 1 - peak cosine against g||ebar_0|| over every DFA run of the master sweep."""
    def fn(ctx):
        per = ctx.per(exp)
        src = (f"results/{exp}/summary_per_run.csv columns drive and maxcos_l<L> (fit of log10(1-maxcos) on "
               "log10(drive) over the gain, output-bias, class-count, imbalance and learning-rate sweeps)")
        if per is None: return None, src + " [experiment missing]"
        sub = A.drive_sweep_rows(per)
        f = A.fit_loglog(sub.get("drive", np.full(len(sub), np.nan)), 1.0 - _deep(sub, "maxcos"))
        if field_name == "prefactor":
            return (10 ** f["intercept"] if np.isfinite(f["intercept"]) else float("nan")), src
        if field_name == "fold":
            if not np.isfinite(f["xmin"]) or f["xmin"] <= 0: return None, src + " [degenerate range]"
            return f"{f['xmax'] / f['xmin']:.0f}-fold", src
        return f.get(field_name), src
    return fn


def kappa_law(field_name, exp=E2, stem=f"minp_w{W}"):
    """log-log fit of the windowed participation minimum against the measured collapse number."""
    def fn(ctx):
        per = ctx.per(exp)
        src = (f"results/{exp}/summary_per_run.csv columns kappa_l<L> and {stem}_l<L> "
               "(fit of log10 p on log10 kappa over all DFA runs)")
        if per is None: return None, src + " [experiment missing]"
        sub = A.dfa_rows(per)
        return A.fit_loglog(_deep(sub, "kappa"), _deep(sub, stem)).get(field_name), src
    return fn


def scaling_law(field_name, exp, xcol, ycol, tags=None):
    """log-log fit of one summary column against another across the tags of a sweep."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv: fit of log10({ycol}) on log10({xcol})"
        if per is None: return None, src + " [experiment missing]"
        sub = per[per.tag.isin(tags)] if tags else per
        if xcol not in sub.columns or ycol not in sub.columns: return None, src + " [column missing]"
        return A.fit_loglog(sub[xcol], sub[ycol]).get(field_name), src
    return fn


def eta_exit_product(exp=E2, tags=("lr0.0003", "lr0.001", "lr0.003")):
    """eta * t_exit, the learning-rate-invariant plateau clock, as a range over the learning-rate sweep."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv columns lr and plateau_exit (eta*t_exit over tags {', '.join(tags)})"
        if per is None: return None, src + " [experiment missing]"
        vals = []
        for t in tags:
            lr = A.col_stat(per, t, "lr"); ex = A.col_stat(per, t, "plateau_exit")
            if np.isfinite(lr) and np.isfinite(ex): vals.append(lr * ex)
        if not vals: return None, src + " [no plateaus detected]"
        return (f"{min(vals):.2f}" if len(vals) == 1 else f"{min(vals):.2f}--{max(vals):.2f}"), src
    return fn


def recovery_law(field_name, exp=E5):
    """T_plateau sqrt(eta_hid eta_out g): its geometric mean, geometric sd, count, or the three separate exponents."""
    def fn(ctx):
        per = ctx.per(exp)
        need = ["plateau_duration", "lr", "hid_lr_mult", "out_lr_mult", "fb_scale"]
        src = (f"results/{exp}/summary_per_run.csv columns {', '.join(need)} "
               f"(T*sqrt(eta_hid*eta_out*g), {field_name})")
        if per is None: return None, src + " [experiment missing]"
        if any(c not in per.columns for c in need): return None, src + " [column missing]"
        T = pd.to_numeric(per.plateau_duration, errors="coerce").values
        eh = per.lr.values * per.hid_lr_mult.values; eo = per.lr.values * per.out_lr_mult.values
        g = per.fb_scale.values
        v = T * np.sqrt(eh * eo * g)
        m = np.isfinite(v) & (v > 0)
        if m.sum() < 3: return None, src + " [fewer than three plateaus]"
        lg = np.log(v[m])
        if field_name == "const": return float(np.exp(lg.mean())), src
        if field_name == "gsd": return float(np.exp(lg.std(ddof=1))), src
        if field_name == "n": return float(m.sum()), src
        if field_name == "exponents":
            out = []
            for name, x in (("eta_hid", eh), ("eta_out", eo), ("g", g)):
                f = A.fit_loglog(x[m], T[m])
                out.append(MISSING if not np.isfinite(f["slope"]) else f"{f['slope']:.2f}")
            return "/".join(out), src + " (separate log-log slopes in eta_hid, eta_out, g)"
        raise ValueError(field_name)
    return fn


def corr_with_ci(exp, xcol, ycol, want="r"):
    """Pearson r (and its 95% Fisher interval) between two summary columns across an experiment."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv columns {xcol} and {ycol} (Pearson r over all runs)"
        if per is None: return None, src + " [experiment missing]"
        if xcol not in per.columns or ycol not in per.columns: return None, src + " [column missing]"
        x = pd.to_numeric(per[xcol], errors="coerce").values; y = pd.to_numeric(per[ycol], errors="coerce").values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5 or x[m].std() == 0 or y[m].std() == 0: return None, src + " [too few or degenerate points]"
        r = float(np.corrcoef(x[m], y[m])[0, 1]); n = int(m.sum())
        if want == "r": return r, src
        z = np.arctanh(np.clip(r, -0.999999, 0.999999)); se = 1.0 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        return f"{lo:.2f} to {hi:.2f}", src + " (Fisher z interval)"
    return fn


def corr_p_cos(exp, tag):
    """Pearson r between participation and hidden cosine along the trajectory, averaged over runs."""
    def fn(ctx):
        src = f"results/{exp}/{tag}/seed*_fb*.csv columns p_l<L> and cos_l<L> (per-run Pearson r, mean over runs)"
        if not os.path.isdir(os.path.join(ctx.exp_dir(exp), tag)): return None, src + " [tag missing]"
        return A.corr_p_cos(ctx.exp_dir(exp), tag), src
    return fn


def drift_spearman(exp, tag, layer=None, step=120):
    """Spearman r between per-unit drift and the fixed common-mode coefficient (B_l 1)_i; all layers when layer=None."""
    def fn(ctx):
        src = (f"results/{exp}/{tag}/seed*_fb*.snapshots.npz: spearman(mu_l<l> at step<={step} minus at step 0, "
               "B1_l<l>), mean over runs")
        d = ctx.exp_dir(exp)
        if not os.path.isdir(os.path.join(d, tag)): return None, src + " [tag missing]"
        layers = [layer] if layer else list(range(1, (ctx.layers(exp, tag) or 3) + 1))
        vals = [A.drift_spearman(d, tag, l, step=step) for l in layers]
        if not any(np.isfinite(v) for v in vals): return None, src + " [no snapshots]"
        if layer: return vals[0], src
        return "/".join(MISSING if not np.isfinite(v) else f"{v:+.2f}" for v in vals), src
    return fn


def relu_constant_frac(exp, tag, bp=False):
    """Fraction of top-layer rectified units that are dead or always on at the deepest collapse."""
    def fn(ctx):
        t = tag + ("_bp" if bp else "")
        n = ctx.layers(exp, t) or 3
        src = f"results/{exp}/{t}/seed*_fb*.csv columns dead_l{n} and alwayson_l{n} (max of their sum up to step {W})"
        a = ctx.traj(exp, t, f"dead_l{n}"); b = ctx.traj(exp, t, f"alwayson_l{n}")
        if a is None or b is None: return None, src + " [tag missing or not a ReLU run]"
        s, m = a[0], a[1] + b[1]
        keep = (s <= W) & np.isfinite(m)
        if not keep.any(): return None, src + " [no samples]"
        return float(m[keep].max()), src
    return fn


def plateau_excess(exp, tag):
    """How far a plateau sits above the class-prior loss, in percent of that loss."""
    def fn(ctx):
        per = ctx.per(exp)
        src = (f"results/{exp}/{tag}/seed*_fb*.csv column probe_loss (minimum over steps 100..{W}) against "
               f"prior_loss in results/{exp}/summary_per_run.csv")
        t = ctx.traj(exp, tag, "probe_loss")
        if t is None or per is None: return None, src + " [tag missing]"
        prior = A.col_stat(per, tag, "prior_loss")
        s, m = t
        keep = (s >= 100) & (s <= W) & np.isfinite(m)
        if not keep.any() or not np.isfinite(prior) or prior <= 0: return None, src + " [no usable samples]"
        return 100.0 * float(m[keep].min() / prior - 1.0), src
    return fn


def ebar_pred_from_prior(exp, tag):
    """||1/C - pi|| for a softmax head under the sampling prior recorded in the run's metadata."""
    def fn(ctx):
        src = f"results/{exp}/{tag}/seed*_fb*.meta.json fields prior and C (||1/C - pi||)"
        fs = sorted(glob.glob(os.path.join(ctx.exp_dir(exp), tag, "seed*_fb*.csv")))
        if not fs: return None, src + " [tag missing]"
        meta = A.read_meta(fs[0])
        if meta is None or "prior" not in meta: return None, src + " [metadata missing]"
        return float(np.linalg.norm(1.0 / meta["C"] - np.asarray(meta["prior"], float))), src
    return fn


def ebar_rel_err(exp, tags):
    """Largest relative gap of seed-averaged ||ebar_0|| from a nonzero closed form."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv columns ebar0 and ebar0_pred (mean each column over seeds, then max relative gap over {', '.join(tags)}; zero predictions excluded)"
        if per is None: return None, src + " [experiment missing]"
        worst = 0.0; seen = False
        for t in tags:
            a = A.col_stat(per, t, "ebar0"); b = A.col_stat(per, t, "ebar0_pred")
            if np.isfinite(a) and np.isfinite(b) and b > 0: worst = max(worst, abs(a - b) / b); seen = True
        return (100 * worst if seen else None), src
    return fn


def reduced_table(field_name):
    """Agreement of the reduced model with the network, from results/reduced/reduced_fit.csv."""
    def fn(ctx):
        src = "results/reduced/reduced_fit.csv columns p_measured, p_model (and tag, layer)"
        fs = sorted(glob.glob(os.path.join(ctx.root, "reduced", "reduced_fit*.csv")))
        if not fs: return None, src + " [reduced model not run]"
        d = pd.read_csv(fs[0])
        if "p_measured" not in d or "p_model" not in d: return None, src + " [columns missing]"
        a = np.asarray(d.p_measured, float); b = np.asarray(d.p_model, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3: return None, src + " [too few points]"
        if field_name == "r": return float(np.corrcoef(a[m], b[m])[0, 1]), src
        if field_name == "mae": return float(np.mean(np.abs(a[m] - b[m]))), src
        if field_name == "n": return float(m.sum()), src
        if field_name == "conditions": return float(d.tag.nunique()) if "tag" in d else float(m.sum()), src
        if field_name in ("model_minp", "measured_minp"):
            col = "p_model" if field_name == "model_minp" else "p_measured"
            if "layer" not in d or "tag" not in d: return None, src + " [no layer column]"
            base = d[d.tag == d.tag.iloc[0]].sort_values("layer")
            return "/".join(f"{v:.2f}" for v in base[col]), src + f" (tag {d.tag.iloc[0]}, by layer)"
        raise ValueError(field_name)
    return fn


def reduced_scalar(key):
    """One scalar the reduced model reports, from results/reduced/reduced_summary.csv (columns key, value)."""
    def fn(ctx):
        src = f"results/reduced/reduced_summary.csv row {key} (column value)"
        fs = sorted(glob.glob(os.path.join(ctx.root, "reduced", "reduced_summary*.csv")))
        if not fs: return None, src + " [reduced model not run]"
        d = pd.read_csv(fs[0])
        if "key" not in d or "value" not in d: return None, src + " [columns missing]"
        row = d[d.key == key]
        return (float(row.value.iloc[0]) if len(row) else None), src + ("" if len(row) else " [key absent]")
    return fn


def derived(path, idx=0):
    """One value of results/derived_laws.json (written by scripts/derived_laws.py). `path` is dotted; a [mean, sd] pair yields entry idx."""
    def fn(ctx):
        f = os.path.join(ctx.root, "derived_laws.json"); src = f"results/derived_laws.json key {path} (scripts/derived_laws.py)"
        if not os.path.exists(f): return None, src + " [derived_laws.py not run]"
        v = json.load(open(f))
        for k in path.split("."):
            if not isinstance(v, dict) or k not in v: return None, src + " [key absent]"
            v = v[k]
        if isinstance(v, list): v = v[idx] if len(v) > idx else None
        return v, src
    return fn


def derived_join(paths, fmt="{:.2f}", sep="/"):
    """Several derived values joined into one string, for per-layer quantities."""
    def fn(ctx):
        vals = []
        for p in paths:
            v, src = derived(p)(ctx)
            if v is None or not np.isfinite(float(v)): return None, src
            vals.append(fmt.format(float(v)))
        return sep.join(vals), "results/derived_laws.json keys " + ", ".join(paths)
    return fn


def derived_range(lo_path, hi_path, fmt="{:.2f}", sep=" to "):
    """An interval from two derived values."""
    def fn(ctx):
        lo, src = derived(lo_path)(ctx); hi, _ = derived(hi_path)(ctx)
        if lo is None or hi is None: return None, src
        return fmt.format(float(lo)) + sep + fmt.format(float(hi)), f"results/derived_laws.json keys {lo_path}, {hi_path}"
    return fn


def crosscheck(path, filename="reduced_crossseed.json"):
    """One value of results/reduced_crosscheck.json (scripts/reduced_crosscheck.py)."""
    def fn(ctx):
        f = os.path.join(ctx.root, filename); src = f"results/{filename} key {path} (scripts/reduced_crosscheck.py)"
        if not os.path.exists(f): return None, src + " [reduced_crosscheck.py not run]"
        v = json.load(open(f))
        for k in path.split("."):
            if not isinstance(v, dict) or k not in v: return None, src + " [key absent]"
            v = v[k]
        return v, src
    return fn


def revision(key):
    def fn(ctx):
        f=os.path.join(ctx.root,"revision_summary.json")
        src=f"results/revision_summary.json numbers.{key} (scripts/summarize_revision.py)"
        if not os.path.exists(f):return None,src
        return json.load(open(f))["numbers"].get(key),src
    return fn


def validation(col, target="E1_headline/base", layers=None, fmt="{:.2f}"):
    """A column of results/reduced_validation.csv (scripts/validate_reduced.py) averaged over the runs of one target; with `layers`, a per-layer string."""
    def fn(ctx):
        f = os.path.join(ctx.root, "reduced_validation.csv"); src = f"results/reduced_validation.csv column {col} rows target {target} (scripts/validate_reduced.py)"
        if not os.path.exists(f): return None, src + " [validate_reduced.py not run]"
        d = pd.read_csv(f); d = d[d.target.astype(str).str.endswith(target)]
        if not len(d): return None, src + " [target absent]"
        if layers is None:
            if col not in d.columns: return None, src + " [column missing]"
            return float(pd.to_numeric(d[col], errors="coerce").mean()), src
        out = []
        for l in layers:
            c = col.format(l=l)
            if c not in d.columns: return None, src + f" [column {c} missing]"
            out.append(fmt.format(float(pd.to_numeric(d[c], errors="coerce").mean())))
        return "/".join(out), src + " (by layer)"
    return fn


def tag_ratio(exp, num_tag, den_tag, col):
    """Mean of `col` over one tag divided by its mean over another."""
    def fn(ctx):
        per = ctx.per(exp)
        src = f"results/{exp}/summary_per_run.csv column {col} (mean over tag {num_tag} divided by mean over tag {den_tag})"
        if per is None: return None, src + " [experiment missing]"
        a = A.col_stat(per, num_tag, col); b = A.col_stat(per, den_tag, col)
        if not np.isfinite(a) or not np.isfinite(b) or b == 0: return None, src + " [no plateau in one of the tags]"
        return a / b, src
    return fn


def kt(exp, tag, layer="L", how="mean"):
    """Transient collapse dose kappa_l accumulated until the batch mean error has fallen to 15 percent of its initial norm,
    from results/kappa_transient_per_run.csv (scripts/derived_laws.py). layer is 1..L or "L" for the deepest."""
    def fn(ctx):
        f = os.path.join(ctx.root, "kappa_transient_per_run.csv"); col = "kt_L" if layer == "L" else f"kt_l{layer}"
        src = f"results/kappa_transient_per_run.csv column {col}, rows exp {exp} tag {tag} ({how}; scripts/derived_laws.py)"
        if not os.path.exists(f): return None, src + " [derived_laws.py not run]"
        d = pd.read_csv(f); d = d[(d.exp == exp) & (d.tag == tag)]
        if not len(d) or col not in d.columns: return None, src + " [no rows]"
        v = pd.to_numeric(d[col], errors="coerce"); return float(getattr(v, how)()), src
    return fn


def kt_range(exp, tags, layer="L", fmt="{:.1f}", sep=" to "):
    """Transient doses of several tags joined as a range string."""
    def fn(ctx):
        vals = []
        for t in tags:
            v, src = kt(exp, t, layer)(ctx)
            if v is None or not np.isfinite(v): return None, src
            vals.append(v)
        return sep.join(fmt.format(v) for v in vals), f"results/kappa_transient_per_run.csv, tags {', '.join(tags)}"
    return fn


def kt_median_kp(field="kp", exps=("E1_headline", "E2_master", "E5_recovery"), kmin=3.0):
    """Median of kappa_L * min p_L over uncentered DFA runs with kappa_L > kmin (the population of Figure 3b); asymptote 1.50."""
    def fn(ctx):
        f = os.path.join(ctx.root, "kappa_transient_per_run.csv")
        src = f"results/kappa_transient_per_run.csv columns kt_L and minp300_L, exps {', '.join(exps)}, DFA runs without centering, whitening, gating, normalization, Muon or aligned feedback, kt_L > {kmin:g}"
        if not os.path.exists(f): return None, src + " [derived_laws.py not run]"
        d = pd.read_csv(f); d = d[d.exp.isin(exps) & (d.rule == "dfa") & ~d.tag.str.contains("center|whiten|gated|bn|muon|aligned")]
        d = d[(d.kt_L > kmin) & np.isfinite(d.minp300_L)]
        d = d.drop_duplicates("condition_key")
        if len(d) < 5: return None, src + " [too few runs]"
        return (float(np.median(d.kt_L * d.minp300_L)) if field == "kp" else float(len(d))), src
    return fn


def traj_mean(exp, tag, col):
    """Mean of a trajectory column over all logged steps and runs of a tag."""
    def fn(ctx):
        src = f"results/{exp}/{tag}/seed*_fb*.csv column {col} (mean over steps and runs)"
        t = ctx.traj(exp, tag, col)
        if t is None: return None, src + " [tag or column missing]"
        return float(np.nanmean(t[1])), src
    return fn


def gauss(key, fmt_step=None):
    """One value of results/gauss_reconstruction.json (scripts/gauss_reconstruction.py); nested step keys as 'median_ratio_gauss.3000'."""
    def fn(ctx):
        f = os.path.join(ctx.root, "gauss_reconstruction.json"); src = f"results/gauss_reconstruction.json key {key} (scripts/gauss_reconstruction.py)"
        if not os.path.exists(f): return None, src + " [gauss_reconstruction.py not run]"
        v = json.load(open(f))
        for k in key.split("."):
            if not isinstance(v, dict) or k not in v: return None, src + " [key absent]"
            v = v[k]
        return v, src
    return fn


def traj_window_mean(exp, tag, col, lo, hi):
    """Mean of a trajectory column over logged steps in [lo, hi] and over runs."""
    def fn(ctx):
        src = f"results/{exp}/{tag}/seed*_fb*.csv column {col}, mean over steps {lo}..{hi} and runs"
        t = ctx.traj(exp, tag, col)
        if t is None: return None, src + " [tag or column missing]"
        s, m = t[0], t[1]; keep = (s >= lo) & (s <= hi) & np.isfinite(m)
        return (float(np.mean(m[keep])) if keep.any() else None), src
    return fn


def resid(key, idx=None, fmt="{:.2f}"):
    """Values of results/residual_ratio.json (scripts/measure_residual.py): a scalar, one entry of a per-layer list, or the list joined."""
    def fn(ctx):
        f = os.path.join(ctx.root, "residual_ratio.json"); src = f"results/residual_ratio.json key {key} (scripts/measure_residual.py)"
        if not os.path.exists(f): return None, src + " [measure_residual.py not run]"
        v = json.load(open(f)).get(key)
        if v is None: return None, src + " [key absent]"
        if isinstance(v, list): return ("/".join(fmt.format(x) for x in v) if idx is None else v[idx]), src
        return v, src
    return fn


def text(sentence, source):
    """A prose macro whose wording is fixed here; `source` names the artefact it summarizes."""
    return lambda ctx: (sentence, source)


def todo(what):
    """A number the pipeline cannot produce yet; the macro is emitted as '??' with the reason."""
    return lambda ctx: (None, f"not produced by any current experiment: {what}")


# ======================================================================================================================
# REGISTRY. One entry per number the manuscript may quote: macro name, one-line description, source, print format.
# Names that paper/sections/*.tex already uses are marked [paper]; main() warns about any used name missing here.
# ======================================================================================================================
REGISTRY = [
    # ---- protocol and scale ------------------------------------------------------------------------------------
    Macro("numPriorLoss", "[paper] loss of the constant predictor at the sampling prior (the plateau value)",
          stat(E1, "base", "prior_loss"), "{:.4f}"),
    Macro("numTotalRuns", "[paper] total number of controlled runs behind the paper", total_runs(), "{:.0f}"),
    Macro("numHeadlineTraj", "[paper] trajectories per headline condition (seeds x feedback draws)",
          n_runs(E1, "base"), "{:.0f}"),
    Macro("numSigmaPrimeEff", "effective output slope (1/2-1/C)/|logit(1/C)| in the collapse number at C=10",
          const(A.sigma_prime_eff(10), "closed form, cmc.analysis.sigma_prime_eff(C=10)")),
    Macro("numEbarZeroPred", "closed-form ||ebar_0|| = sqrt(C)(1/2-1/C) for the sigmoid head at C=10",
          const(A.ebar0_closed_form("sigmoid_bce", 10), "closed form, cmc.analysis.ebar0_closed_form"), "{:.4f}"),
    Macro("numEbarInit", "[paper] measured ||ebar_0|| of an untrained sigmoid head", stat(E1, "base", "ebar0"), "{:.3f}"),
    Macro("numEbarErr", "[paper] largest relative gap of seed-averaged initial error norms from the closed form, C=3,5,10, percent",
          ebar_rel_err(E2, ["C3", "C5", "C10"]), "{:.0f}\\%"),
    Macro("numBinaryEbar", "[paper] seed-averaged initial error norm for C=2 (the ideal constant-prediction value is zero)",
          stat(E2, "C2", "ebar0")),
    Macro("numCommonModeFrac", "[paper] share of the error energy carried by the batch mean at step 10, percent, from the ratio r = ||ebar||/||etil|| as r^2/(1+r^2)",
          lambda ctx: ((lambda v, src: (None if v is None or not np.isfinite(v) else 100 * v * v / (1 + v * v), src))(*stat(E1, "base", "ebar_ratio_10")(ctx))), "{:.0f}\\%"),
    Macro("numCommonModeRatio", "[paper] ratio ||ebar||/||etil|| of the error at step 10", stat(E1, "base", "ebar_ratio_10"), "{:.2f}"),
    Macro("numHeadresetKappa", "[paper] top-layer collapse dose accumulated after a label shift with gated error centering at step 1500 (legacy tag headreset1500)", derived("shift.headreset1500.dose_post"), "{:.2f}"),
    Macro("numXbarBase", "[paper] mean input energy ||xbar||^2 under the baseline preprocessing",
          stat(E1, "base", "xbar2"), "{:.0f}"),
    Macro("numXbarStd", "[paper] mean input energy ||xbar||^2 after global standardization",
          best_of(stat(E2, "pre_standardize", "xbar2"), stat(E2, "pre_pixstd", "xbar2")), "{:.0f}"),

    # ---- figure 1: the phenomenon ------------------------------------------------------------------------------
    Macro("numInitCos", "[paper] mean pairwise hidden cosine at initialization, top layer",
          stat(E1, "base", "cos0_l{L}")),
    Macro("numBaseCosPeak", "[paper] peak hidden cosine, top layer, DFA baseline", stat(E1, "base", "maxcos_l{L}")),
    Macro("numBaseCosPeakSD", "seed sd of the peak hidden cosine, top layer", stat(E1, "base", "maxcos_l{L}", "sd")),
    Macro("numBPCosPeak", "[paper] peak hidden cosine, top layer, BP twin", stat(E1, "bp", "maxcos_l{L}")),
    Macro("numBaseMinPone", f"[paper] minimum gate participation, layer 1, step<={W}", stat(E1, "base", f"minp_w{W}_l1")),
    Macro("numBaseMinPtwo", f"[paper] minimum gate participation, layer 2, step<={W}", stat(E1, "base", f"minp_w{W}_l2")),
    Macro("numBaseMinPthree", f"[paper] minimum gate participation, layer 3, step<={W}", stat(E1, "base", f"minp_w{W}_l3")),
    Macro("numBPMinPthree", "minimum gate participation of the BP twin, layer 3", stat(E1, "bp", f"minp_w{W}_l3")),
    Macro("numPCosCorr", "[paper] Pearson r between participation and hidden cosine along the baseline trajectory",
          corr_p_cos(E1, "base"), "{:.2f}"),
    Macro("numOnsetSteps", "[paper] step at which the loss enters the class-prior band", stat(E1, "base", "plateau_onset"), "{:.0f}"),
    Macro("numPlateauOnsetSD", "seed sd of the plateau onset", stat(E1, "base", "plateau_onset", "sd"), "{:.0f}"),
    Macro("numPlateauExit", "plateau exit step (first step below 0.9 x the prior loss)",
          stat(E1, "base", "plateau_exit"), "{:.0f}"),
    Macro("numPlateauExitSD", "seed sd of the plateau exit", stat(E1, "base", "plateau_exit", "sd"), "{:.0f}"),
    Macro("numPlateauDuration", "plateau duration in steps", stat(E1, "base", "plateau_duration"), "{:.0f}"),
    Macro("numPlateauDurationSD", "seed sd of the plateau duration", stat(E1, "base", "plateau_duration", "sd"), "{:.0f}"),
    Macro("numPlateauBandAcc", "probe accuracy inside the plateau band", stat(E1, "base", "plateau_band_acc")),
    Macro("numBaseAccOneK", "baseline probe accuracy at step 1000", stat(E1, "base", "acc_1000")),
    Macro("numBaseAccFifteenH", "baseline probe accuracy at step 1500", stat(E1, "base", "acc_1500")),
    Macro("numBaseAccThreeK", "baseline probe accuracy at step 3000", stat(E1, "base", "acc_3000")),
    Macro("numBPAccFifteenH", "BP-twin probe accuracy at step 1500", stat(E1, "bp", "acc_1500")),
    Macro("numHinit", "[paper] top-layer mean-activity energy at initialization", stat(E1, "base", "H0_l{L}"), "{:.0f}"),
    Macro("numHplateau", "[paper] top-layer mean-activity energy at the deepest collapse",
          extremum(E1, "base", "H_l{L}", "max", hi=W), "{:.0f}"),
    Macro("numSigInit", "[paper] mean within-unit spread of the top layer at initialization",
          at_step(E1, "base", "sig_mean_l{L}", 0), "{:.2f}"),
    Macro("numSigPlateau", "[paper] mean within-unit spread of the top layer at the plateau",
          extremum(E1, "base", "sig_mean_l{L}", "min", hi=W), "{:.2f}"),
    Macro("numLambdaDrop", "[paper] mean per-run fold fall of layer-3 label-covariance energy from initialization to its minimum after step 30; same source as numCascadeDrops",
          derived("cascade.lam_drop_l3"), "{:.1f}"),
    Macro("numDecodDrop", "[paper] fold fall of the whitened linear decodability at the plateau",
          derived("anatomy.base.R_drop"), "{:.2f}"),
    Macro("numLambdaDropBP", "fold fall of the top-layer label-covariance energy at the same step under BP", derived("anatomy.bp.lam_drop"), "{:.2f}"),
    Macro("numPCOnePeak", "[paper] variance fraction of the leading principal component of the top hidden layer at the collapse peak", derived("anatomy.base.pc1_peak"), "{:.2f}"),
    Macro("numPCOneInit", "[paper] the same at initialization", derived("anatomy.base.pc1_init"), "{:.2f}"),
    Macro("numPCOneBP", "[paper] the same under BP at the matched step", derived("anatomy.bp.pc1_peak"), "{:.2f}"),

    # ---- the collapse number -----------------------------------------------------------------------------------
    Macro("numBaseKappa", "[paper] transient collapse dose of the top layer, baseline", kt(E1, "base"), "{:.1f}"),
    Macro("numBaseKappaEnd", "online collapse number of the top layer at the end of the baseline run", stat(E1, "base", "kappa_l{L}"), "{:.1f}"),
    Macro("numKappaOne", "[paper] transient collapse dose, layer 1, baseline", kt(E1, "base", 1), "{:.1f}"),
    Macro("numKappaTwo", "[paper] transient collapse dose, layer 2, baseline", kt(E1, "base", 2), "{:.1f}"),
    Macro("numKappaThree", "[paper] transient collapse dose, layer 3, baseline", kt(E1, "base", 3), "{:.1f}"),
    Macro("numImbScratchKappa", "[paper] transient dose of the from-scratch imbalanced softmax control", kt(E1, "softmax_imb07"), "{:.1f}"),
    Macro("numSoftmaxKappa", "[paper] transient dose of the balanced softmax head", kt(E1, "softmax"), "{:.2f}"),
    Macro("numCenterEHL", "[paper] top-layer mean-activity energy at step 300 under centered feedback", at_step(E1, "center_e", "H_l{L}", 300), "{:.1f}"),
    Macro("numBaseHLThree", "[paper] top-layer mean-activity energy at step 300, baseline", at_step(E1, "base", "H_l{L}", 300), "{:.0f}"),
    Macro("numCenterEEbarFifteen", "[paper] ||ebar|| on the probe at step 1500 under centered feedback", at_step(E1, "center_e", "ebar_norm", 1500), "{:.2f}"),
    Macro("numBaseEbarFifteen", "[paper] ||ebar|| on the probe at step 1500, baseline", at_step(E1, "base", "ebar_norm", 1500), "{:.2f}"),
    Macro("numCenterELossFifteen", "[paper] probe loss at step 1500 under centered feedback", at_step(E1, "center_e", "probe_loss", 1500), "{:.2f}"),
    Macro("numBaseLossFifteen", "[paper] probe loss at step 1500, baseline", at_step(E1, "base", "probe_loss", 1500), "{:.2f}"),
    Macro("numGatedOn", "[paper] fraction of the run during which the energy-fraction gate kept centering on", traj_mean(E1, "gated", "gate_on"), "{:.2f}"),
    Macro("numKappaCFOne", "closed-form collapse number, layer 1, baseline", stat(E1, "base", "kappa_cf_l1"), "{:.1f}"),
    Macro("numKappaCFTwo", "closed-form collapse number, layer 2, baseline", stat(E1, "base", "kappa_cf_l2"), "{:.1f}"),
    Macro("numKappaCFThree", "closed-form collapse number, layer 3, baseline", stat(E1, "base", "kappa_cf_l3"), "{:.1f}"),

    # ---- figure 2: the reduced model ---------------------------------------------------------------------------
    Macro("numModelOnset", "[paper] plateau onset predicted by the reduced model (loss within 3 percent of the prior loss)", validation("onset_model"), "{:.0f}"),
    Macro("numModelOnsetMeas", "[paper] measured onset by the same loss criterion", validation("onset_meas"), "{:.0f}"),
    Macro("numModelExit", "[paper] plateau exit predicted by the reduced model with the escape cascade", validation("exit_model"), "{:.0f}"),
    Macro("numModelExitMeas", "[paper] measured exit by the same criterion", validation("exit_meas"), "{:.0f}"),
    Macro("numModelMinP", "[paper] participation minima per layer predicted by the reduced model",
          validation("minp_l{l}_model", layers=(1, 2, 3))),
    Macro("numMeasMinP", "[paper] measured participation minima per layer, same runs", validation("minp_l{l}_meas", layers=(1, 2, 3))),
    Macro("numModelR", "[paper] correlation of reduced-model against measured participation minima, all layers and conditions",
          crosscheck("r_p"), "{:.2f}"),
    Macro("numModelMAE", "[paper] mean absolute error of the reduced model on participation minima", crosscheck("mae_p"), "{:.2f}"),
    Macro("numModelConds", "[paper] number of conditions the reduced model was tested on", crosscheck("n_conditions"), "{:.0f}"),
    Macro("numModelPoints", "[paper] number of (condition, seed, layer) points in the reduced-model comparison", crosscheck("n_points"), "{:.0f}"),
    Macro("numModelBiasLone", "[paper] signed bias of the reduced model on layer-1 participation minima", crosscheck("bias_p_by_layer.1"), "{:+.2f}"),
    Macro("numModelBiasLthree", "[paper] signed bias of the reduced model on layer-3 participation minima", crosscheck("bias_p_by_layer.3"), "{:+.3f}"),
    Macro("numModelCosR", "[paper] correlation of modelled against measured peak cosine", crosscheck("r_cos"), "{:.2f}"),
    Macro("numModelExitR", "[paper] correlation of modelled against measured plateau exit", crosscheck("exit_r"), "{:.2f}"),
    Macro("numModelPairs", "[paper] number of condition-seed trajectories with an observed network exit and model onset", crosscheck("onset_exit_n"), "{:.0f}"),
    Macro("numModelExitPairs", "[paper] number of uncensored exit pairs used for exit correlation", crosscheck("exit_valid_n"), "{:.0f}"),
    Macro("numModelCensored", "[paper] model exits censored at the integration horizon", crosscheck("exit_censored_n"), "{:.0f}"),
    Macro("numCrossSeedR", "[paper] participation correlation over distinct condition-seed pairs", crosscheck("r_p","reduced_crossseed.json"), "{:.2f}"),
    Macro("numCrossSeedRuns", "[paper] unique condition-seed pairs in model replication", crosscheck("n_trajectories","reduced_crossseed.json"), "{:.0f}"),
    Macro("numCrossSeedMAE", "[paper] participation MAE over distinct condition-seed pairs", crosscheck("mae_p","reduced_crossseed.json"), "{:.2f}"),
    Macro("numAuditSingleR", "[paper] audited original single-seed participation correlation", crosscheck("r_p","reduced_crosscheck.json"), "{:.2f}"),
    Macro("numGridR", "[paper] width-depth grid participation correlation", crosscheck("r_p","reduced_grid.json"), "{:.2f}"),
    Macro("numGridMAE", "[paper] width-depth grid participation MAE", crosscheck("mae_p","reduced_grid.json"), "{:.2f}"),
    Macro("numGridExitPairs", "[paper] uncensored width-depth grid exit pairs", crosscheck("exit_valid_n","reduced_grid.json"), "{:.0f}"),
    Macro("numGridCensored", "[paper] model exits censored on the width-depth grid", crosscheck("exit_censored_n","reduced_grid.json"), "{:.0f}"),
    Macro("numGridExitError", "[paper] median relative exit error on the grid", crosscheck("exit_rel_err_median","reduced_grid.json"), "{:.2f}"),
    Macro("numGridMissedPlateaus", "[paper] network plateaus without a detected model plateau on the grid", crosscheck("n_network_only_plateau","reduced_grid.json"), "{:.0f}"),
    Macro("numGridExtraPlateaus", "[paper] model plateaus without a detected network plateau on the grid", crosscheck("n_model_only_plateau","reduced_grid.json"), "{:.0f}"),
    Macro("numModelOnsetErr", "[paper] median relative error of the modelled onset, percent", lambda ctx: (None if crosscheck("onset_rel_err_median")(ctx)[0] is None else 100 * crosscheck("onset_rel_err_median")(ctx)[0], crosscheck("onset_rel_err_median")(ctx)[1]), "{:.0f}"),
    Macro("numModelExitErr", "[paper] median relative error of the modelled exit, percent", lambda ctx: (None if crosscheck("exit_rel_err_median")(ctx)[0] is None else 100 * crosscheck("exit_rel_err_median")(ctx)[0], crosscheck("exit_rel_err_median")(ctx)[1]), "{:.0f}"),

    # ---- figure 3: the master curve ----------------------------------------------------------------------------
    Macro("numProductLawSlope", "[paper] exponent of 1-peak cosine against g||ebar_0||", derived("product_law_pure.slope"), "{:.2f}"),
    Macro("numProductLawPrefactor", "[paper] prefactor of the product law", derived("product_law_pure.prefactor"), "{:.4f}"),
    Macro("numProductLawR", "[paper] correlation of the product-law fit in log-log", derived("product_law_pure.r"), "{:.3f}"),
    Macro("numMasterN", "[paper] number of runs on the product-law line", derived("product_law_pure.n"), "{:.0f}"),
    Macro("numMasterRange", "[paper] range of g||ebar_0|| spanned by the master curve", derived("product_law_pure.fold"), "{:.0f}-fold"),
    Macro("numProductLawResid", "[paper] interleaving residual: mean log-offset of the q sweep from the joint line, in log10 units", derived("product_law_pure_resid_qsweep"), "{:.3f}"),
    Macro("numProductLawMixedR", "[paper] correlation of the product law when the class-count and imbalance sweeps are added", derived("product_law_mixed.r"), "{:.3f}"),
    Macro("numProductLawMixedN", "[paper] number of runs in that mixed fit", derived("product_law_mixed.n"), "{:.0f}"),
    Macro("numProductLawLoneSlope", "[paper] exponent of the product law in layer 1", derived("product_law_pure_layer1.slope"), "{:.2f}"),
    Macro("numKappaP", "[paper] median of kappa_L times the participation minimum over runs with kappa_L > 3 (asymptote 1.50)", kt_median_kp("kp"), "{:.2f}"),
    Macro("numKappaPN", "[paper] number of runs in that median", kt_median_kp("n"), "{:.0f}"),
    Macro("numKappaPSlope", "exponent of the participation minimum against the collapse number", derived("p_kappa_loglog_all.slope"), "{:.2f}"),
    Macro("numKappaPR", "correlation of the participation-against-kappa fit in log-log", derived("p_kappa_loglog_all.r"), "{:.3f}"),
    Macro("numGainRange", "[paper] range of feedback gains in the master sweep",
          range_str(E2, ["g0.03", "g0.1", "g0.3", "g1.0", "g3.0", "g10.0"], "fb_scale", "{:g}")),
    Macro("numEtaCosRange", "[paper] peak hidden cosine across the learning-rate sweep",
          range_str(E2, ["lr0.0003", "lr0.001", "lr0.003"], "maxcos_l{L}", "{:.3f}")),
    Macro("numEtaPRange", "[paper] participation minima across the learning-rate sweep",
          range_str(E2, ["lr0.0003", "lr0.001", "lr0.003"], f"minp_w{W}_l3", "{:.2f}")),
    Macro("numEtaExitProduct", "[paper] eta times the plateau exit step across the learning-rate sweep",
          eta_exit_product()),
    Macro("numWidthExp", "[paper] exponent of the plateau duration against width",
          scaling_law("slope", E2, "width", "plateau_exit", ["w100", "w300", "w900", "w2000"]), "{:.2f}"),
    Macro("numWidthPone", "[paper] layer-1 participation minima across the width sweep",
          range_str(E2, ["w100", "w300", "w900", "w2000"], f"minp_w{W}_l1", "{:.2f}")),
    Macro("numSoftmaxCos", "[paper] mean peak hidden cosine under balanced softmax in the fifteen-run headline cohort",
          stat(E1, "softmax", "maxcos_l{L}")),
    Macro("numImbCos", "[paper] mean peak hidden cosine under softmax with class-0 mixture weight 0.7 in the fifteen-run headline cohort",
          stat(E1, "softmax_imb07", "maxcos_l{L}")),
    Macro("numImbEbarPred", "[paper] predicted ||ebar_0|| = ||1/C - pi|| for the imbalanced softmax head",
          best_of(ebar_pred_from_prior(E2, "smimb0.7"), ebar_pred_from_prior(E1, "softmax_imb07"))),
    Macro("numImbEbarMeas", "[paper] measured ||ebar_0|| for the imbalanced softmax head",
          best_of(stat(E2, "smimb0.7", "ebar0"), stat(E1, "softmax_imb07", "ebar0"))),

    # ---- figure 4: the causal dissection -------------------------------------------------------------------------
    Macro("numPixLoneBase", "[paper] peak hidden cosine of layer 1, baseline", stat(E1, "base", "maxcos_l1")),
    Macro("numPixLoneCentered", "[paper] peak hidden cosine of layer 1 with per-pixel centered inputs "
                     "(the manuscript writes numPixL1Base/numPixL1Centered; both resolve to this macro)",
          stat(E1, "pixcenter", "maxcos_l1")),
    Macro("numPixcenterCosOne", "peak hidden cosine, layer 1, per-pixel centered inputs", stat(E1, "pixcenter", "maxcos_l1")),
    Macro("numPixcenterCosThree", "peak hidden cosine, layer 3, per-pixel centered inputs", stat(E1, "pixcenter", "maxcos_l3")),
    Macro("numPixcenterMinPthree", "layer-3 participation minimum with per-pixel centered inputs",
          stat(E1, "pixcenter", f"minp_w{W}_l3")),
    Macro("numFrzbiasMinPthree", "layer-3 participation minimum with frozen hidden biases (the null)",
          stat(E1, "frzbias", f"minp_w{W}_l3")),
    Macro("numBothCosPeak", "peak hidden cosine with centered inputs and frozen biases",
          stat(E1, "pixcenter_frzbias", "maxcos_l{L}")),
    Macro("numBothMinPthree", "layer-3 participation minimum with both channels removed",
          stat(E1, "pixcenter_frzbias", f"minp_w{W}_l3")),
    Macro("numDriftSpearman", "[paper] Spearman r between per-unit drift and (B_l 1)_i, layers 1 to L",
          drift_spearman(E1, "base")),
    Macro("numDriftSpearmanOne", "[paper] Spearman r between per-unit drift and (B_l 1)_i, layer 1",
          drift_spearman(E1, "base", layer=1), "{:.2f}"),
    Macro("numDriftSpearmanTwo", "[paper] Spearman r between per-unit drift and (B_l 1)_i, layer 2",
          drift_spearman(E1, "base", layer=2), "{:.2f}"),
    Macro("numDriftSpearmanThree", "[paper] Spearman r between per-unit drift and (B_l 1)_i, layer 3",
          drift_spearman(E1, "base", layer=3), "{:.2f}"),
    Macro("numRankOneFrac", "[paper] variance fraction of the preactivation field along the fixed direction B_l ebar_0",
          derived("anatomy.base.r1frac_peak"), "{:.2f}"),
    Macro("numRankOneFracLone", "[paper] the same in layer 1", derived("anatomy.base.r1frac_l1_peak"), "{:.2f}"),
    Macro("numRankOneFracAligned", "[paper] the same with pre-aligned feedback, top layer", derived("anatomy.aligned.r1frac_peak"), "{:.2f}"),
    Macro("numJaccard", "[paper] Jaccard overlap of the plateau-selected units with the units useful at step 3000, layers 1 to 3",
          derived_join(["turnover.jaccard_120_3000_l1", "turnover.jaccard_120_3000_l2", "turnover.jaccard_120_3000_l3"])),
    Macro("numJaccardNull", "[paper] chance Jaccard for the selected-set comparison", derived("turnover.jaccard_null"), "{:.2f}"),
    Macro("numSelectedVsB", "[paper] Spearman r between a unit's plateau gate energy and |(B_l 1)_i|, layers 1 to 3",
          derived_join(["turnover.spearman_u_plateau_vs_absB1_l1", "turnover.spearman_u_plateau_vs_absB1_l2", "turnover.spearman_u_plateau_vs_absB1_l3"])),
    Macro("numBPxTwelveCos", "[paper] peak hidden cosine for BP with output weights scaled 12x "
                    "(the manuscript writes numBPx12Cos, which resolves to this macro)",
          stat(E1, "bp_outx12", "maxcos_l{L}")),
    Macro("numAlignedCos", "[paper] peak hidden cosine with feedback pre-aligned to the downstream map",
          stat(E1, "aligned", "maxcos_l{L}")),
    Macro("numAlignedAccTwentyFive", "[paper] probe accuracy of pre-aligned feedback early in training "
                           "(the manuscript writes numAlignedAcc25, which resolves to this macro)",
          at_step(E1, "aligned", "probe_acc", 30)),
    Macro("numFrzoutKappa", "[paper] layer-3 collapse dose with the head frozen (whole run, the mean error never decays)", stat(E1, "frzout", "kappa_l3"), "{:.0f}"),
    Macro("numFrzoutP", "[paper] layer-3 participation minimum with the head frozen",
          stat(E1, "frzout", f"minp_w{W}_l3")),
    Macro("numFrzoutLossInit", "[paper] probe loss at initialization with the head frozen",
          at_step(E1, "frzout", "probe_loss", 0), "{:.2f}"),
    Macro("numFrzoutLossEnd", "[paper] final probe loss with the head frozen", stat(E1, "frzout", "loss_final"), "{:.2f}"),
    Macro("numOutlrFactor", "[paper] how many fold a threefold head learning rate shortens the plateau",
          tag_ratio(E5, "h1.0_o1.0_g1.0", "h1.0_o3.0_g1.0", "plateau_duration"), "{:.1f}"),
    Macro("numOutlrTenKappa", "[paper] top-layer transient dose with a tenfold head learning rate", kt(E1, "outlr10", 3), "{:.1f}"),
    Macro("numOutlrKappa", "[paper] top-layer collapse number, baseline against a tenfold head learning rate",
          kt_range(E1, ["outlr10", "base"], 3)),
    Macro("numPriorbiasCosPeak", "peak hidden cosine with the output bias calibrated to the prior",
          stat(E1, "priorbias", "maxcos_l{L}")),

    # ---- recovery ------------------------------------------------------------------------------------------------
    Macro("numRecoveryConst", "[paper] geometric mean of T_plateau sqrt(eta_hid eta_out g)", derived("recovery_T.const_geomean"), "{:.2f}"),
    Macro("numRecoverySD", "[paper] geometric sd of T_plateau sqrt(eta_hid eta_out g)", derived("recovery_T.const_geosd"), "{:.2f}"),
    Macro("numRecoveryRuns", "[paper] number of runs behind the recovery law", derived("recovery_T.n"), "{:.0f}"),
    Macro("numRecoveryExps", "[paper] separate log-log exponents in eta_hid, eta_out and g", derived_join(["recovery_T.exp_hid", "recovery_T.exp_out", "recovery_T.exp_g"], fmt="{:.2f}")),
    Macro("numRecoveryRsq", "[paper] r squared of the three-exponent fit of the plateau duration", derived("recovery_T.r2"), "{:.2f}"),
    Macro("numCascadeExps", "[paper] growth exponents of the label covariance per layer during escape",
          derived_join(["cascade.n_l1", "cascade.n_l2", "cascade.n_l3"])),
    Macro("numCascadeDrops", "[paper] mean per-run fold fall of label-covariance energy from initialization to its minimum after step 30, layers 1 to 3", derived_join(["cascade.lam_drop_l1", "cascade.lam_drop_l2", "cascade.lam_drop_l3"], fmt="{:.1f}")),

    # ---- alignment demarcation -----------------------------------------------------------------------------------
    Macro("numWAPlateau", "[paper] rowwise weight alignment of layer 1 during the plateau",
          at_step(E1, "base", "wa_l1", W), "{:.2f}"),
    Macro("numWAAtThousand", "[paper] rowwise weight alignment of layer 1 at step 1000",
          at_step(E1, "base", "wa_l1", 1000), "{:.2f}"),
    Macro("numWAAtEnd", "[paper] rowwise weight alignment of layer 1 at step 3000",
          at_step(E1, "base", "wa_l1", 3000), "{:.2f}"),
    Macro("numLadderVerdict", "[paper] one-sentence verdict of the teacher-student ladder",
          text("The collapse switches on at the rung that adds trainable hidden biases and is present at every rung after it; adding nonzero-mean targets alone, with zero-mean inputs and no biases, does not produce it",
               "results/E6_ladder/SUMMARY.md, Headline (cmc.ladder, 3 seeds per rung)")),

    # ---- figure 5: cures ------------------------------------------------------------------------------------------
    Macro("numCureBaseAcc", "[paper] accuracy at 1500 steps of plain DFA in the cure bake-off",
          best_of(stat(E4, "base__none", "acc_1500"), stat(E1, "base", "acc_1500"))),
    Macro("numCurePriorAcc", "[paper] accuracy at 1500 steps with a prior-calibrated output bias",
          best_of(stat(E4, "base__priorbias", "acc_1500"), stat(E1, "priorbias", "acc_1500"))),
    Macro("numCureCenterMLP", "[paper] accuracy cost of teaching-signal centering on the baseline MLP, points at 1500",
          best_of(diff(E4, "base__center_delta", "base__none", "acc_1500", scale=100),
                  diff(E1, "center_delta", "base", "acc_1500", scale=100)), "{:+.1f}"),
    Macro("numCureCenterCNN", "[paper] accuracy change of teaching-signal centering on the CNN, points at 1500",
          best_of(diff(E4, "cnn__center_delta", "cnn__none", "acc_1500", scale=100),
                  diff(E3, "cnn_center", "cnn", "acc_1500", scale=100)), "{:+.1f}"),
    Macro("numCureCenterDepthSix", "[paper] accuracy change of teaching-signal centering at depth six, points at 1500",
          best_of(diff(E4, "depth6__center_delta", "depth6__none", "acc_1500", scale=100),
                  diff(E3, "depth6_center", "depth6", "acc_1500", scale=100)), "{:+.1f}"),
    Macro("numCureCenterReLU", "[paper] accuracy change of teaching-signal centering with ReLU units, points at 1500",
          diff(E3, "relu_center", "relu", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureGated", "accuracy change of energy-gated centering against plain DFA, points at 1500 (the gate never released)",
          diff(E4, "base__gated", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterEMLP", "[paper] accuracy change of centering the output error before projection, baseline MLP, points at 1500",
          diff(E4, "base__center_e", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterECNN", "[paper] the same on the CNN", diff(E4, "cnn__center_e", "cnn__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterEDepthSix", "[paper] the same at depth six", diff(E4, "depth6__center_e", "depth6__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterEMLPEnd", "[paper] the same on the baseline MLP at 3000 steps", diff(E4, "base__center_e", "base__none", "acc_3000", scale=100), "{:+.1f}"),
    Macro("numCureCenterECos", "[paper] peak top-layer cosine with the output error centered before projection", stat(E4, "base__center_e", "maxcos_l{L}")),
    Macro("numCureCenterFast", "[paper] accuracy change of centering plus a tenfold head learning rate, points at 1500",
          diff(E4, "base__center_outlr10", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterFastEnd", "[paper] the same at 3000 steps", diff(E4, "base__center_outlr10", "base__none", "acc_3000", scale=100), "{:+.1f}"),
    Macro("numCureCenterFastCos", "[paper] peak top-layer cosine of centering plus a tenfold head learning rate", stat(E4, "base__center_outlr10", "maxcos_l{L}")),
    Macro("numCureCenterFastDepthSix", "[paper] centering plus a tenfold head learning rate at depth six, points at 1500",
          diff(E4, "depth6__center_outlr10", "depth6__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterPrior", "[paper] centering plus a prior-calibrated output bias, points at 1500",
          diff(E4, "base__center_priorbias", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterEReLU", "[paper] centering the output error with ReLU units, points at 1500", diff(E3, "relu_centere", "relu", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterEReLUSlow", "[paper] the same at a learning rate of 3e-4, points at 3000", diff(E3, "relu_lr3e4_centere", "relu_lr3e4", "acc_3000", scale=100), "{:+.1f}"),
    Macro("numNonsatReLUPix", "[paper] accuracy at 1500 steps, ReLU units, output error centered and inputs per-pixel centered, lr 1e-3", stat(E10, "relu_lr1e3_centere_pixcenter", "acc_1500"), "{:.3f}"),
    Macro("numNonsatGELUPix", "[paper] the same for GELU units", stat(E10, "gelu_lr1e3_centere_pixcenter", "acc_1500"), "{:.3f}"),
    Macro("numNonsatLinearPix", "[paper] the same for linear units", stat(E10, "linear_lr1e3_centere_pixcenter", "acc_1500"), "{:.3f}"),
    Macro("numNonsatReLUPlain", "[paper] accuracy at 1500 steps, plain DFA with ReLU units, lr 1e-3", stat(E3, "relu", "acc_1500"), "{:.3f}"),
    Macro("numNonsatGELUPlain", "[paper] the same for GELU units", stat(E3, "gelu", "acc_1500"), "{:.3f}"),
    Macro("numNonsatLinearPlain", "[paper] the same for linear units", stat(E3, "linear", "acc_1500"), "{:.3f}"),
    Macro("numNonsatLinearPixCos", "[paper] peak top-layer cosine, linear units, error and inputs centered", stat(E10, "linear_lr1e3_centere_pixcenter", "maxcos_l{L}")),
    Macro("numNonsatLinearPlainCos", "[paper] peak top-layer cosine, linear units, plain DFA", stat(E3, "linear", "maxcos_l{L}")),
    Macro("numNonsatReLUPixCos", "[paper] peak top-layer cosine, ReLU units, error and inputs centered", stat(E10, "relu_lr1e3_centere_pixcenter", "maxcos_l{L}")),
    Macro("numNonsatReLUPlainCos", "[paper] peak top-layer cosine, ReLU units, plain DFA", stat(E3, "relu", "maxcos_l{L}")),
    Macro("numNonsatReLUBPCos", "[paper] peak top-layer cosine, ReLU units, BP", stat(E3, "relu_bp", "maxcos_l{L}")),
    Macro("numNonsatReLUPrior", "[paper] accuracy at 1500 steps, ReLU units with a prior-calibrated output bias", stat(E10, "relu_lr1e3_priorbias", "acc_1500"), "{:.3f}"),
    Macro("numNonsatReLUFastLoss", "[paper] final probe loss, ReLU units, error centered with a tenfold head learning rate (diverges)", stat(E10, "relu_lr1e3_centere_outlr10", "loss_final"), "{:.0e}"),
    Macro("numCifarCNNCos", "[paper] CIFAR-10 small CNN, sigmoid head, DFA: peak cosine of the deepest block", stat(E11, "cnn", "maxcos_l{L}")),
    Macro("numCifarCNNBP", "[paper] the same for the BP twin", stat(E11, "cnn_bp", "maxcos_l{L}")),
    Macro("numCifarBaseAcc", "[paper] CIFAR-10 CNN DFA accuracy at 3000 steps", stat(E11, "cnn", "acc_3000"), "{:.3f}"),
    Macro("numCifarBPAcc", "[paper] CIFAR-10 CNN BP accuracy at 3000 steps", stat(E11, "cnn_bp", "acc_3000"), "{:.3f}"),
    Macro("numCifarSoftmaxCos", "[paper] CIFAR-10 CNN, balanced softmax, standardized inputs: peak cosine", stat(E11, "cnn_softmax", "maxcos_l{L}")),
    Macro("numCifarImbCos", "[paper] CIFAR-10 CNN, softmax with 70 percent class-0 sampling: peak cosine", stat(E11, "cnn_softmax_imb07", "maxcos_l{L}")),
    Macro("numCifarImbBP", "[paper] the same for the BP twin", stat(E11, "cnn_softmax_imb07_bp", "maxcos_l{L}")),
    Macro("numCifarCentereCos", "[paper] CIFAR-10 CNN, sigmoid head, output error centered: peak cosine", stat(E11, "cnn_centere", "maxcos_l{L}")),
    Macro("numCifarCentereAcc", "[paper] its accuracy at 3000 steps", stat(E11, "cnn_centere", "acc_3000"), "{:.3f}"),
    Macro("numCifarPriorAcc", "[paper] CIFAR-10 CNN with a prior-calibrated bias, accuracy at 3000 steps", stat(E11, "cnn_priorbias", "acc_3000"), "{:.3f}"),
    Macro("numCifarCenterFastAcc", "[paper] CIFAR-10 CNN, centered error plus tenfold head learning rate, accuracy at 3000 steps", stat(E11, "cnn_centere_outlr10", "acc_3000"), "{:.3f}"),
    Macro("numCifarXbar", "[paper] input mean energy ||xbar||^2 of CIFAR-10 in [0,1]", stat(E11, "mlp", "xbar2"), "{:.0f}"),
    Macro("numCifarMLPLone", "[paper] CIFAR-10 MLP, DFA: peak cosine of layer 1", stat(E11, "mlp", "maxcos_l1")),
    Macro("numCifarMLPCos", "[paper] CIFAR-10 MLP, DFA: peak cosine of layer 3", stat(E11, "mlp", "maxcos_l{L}")),
    Macro("numCifarMLPBPLone", "[paper] CIFAR-10 MLP, BP: peak cosine of layer 1", stat(E11, "mlp_bp", "maxcos_l1")),
    Macro("numCifarMLPPixLone", "[paper] CIFAR-10 MLP with per-pixel centered inputs: peak cosine of layer 1", stat(E11, "mlp_pixcenter", "maxcos_l1")),
    Macro("numCifarMLPKappaOne", "[paper] CIFAR-10 MLP transient dose, layer 1", kt(E11, "mlp", 1), "{:.1f}"),
    Macro("numCifarMLPAcc", "[paper] CIFAR-10 MLP DFA accuracy at 3000 steps", stat(E11, "mlp", "acc_3000"), "{:.3f}"),
    Macro("numCifarMLPCentereAcc", "[paper] CIFAR-10 MLP, centered error, accuracy at 3000 steps", stat(E11, "mlp_centere", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluCos", "[paper] CIFAR-10 ReLU CNN, DFA: peak cosine", stat(E11, "cnnrelu", "maxcos_l{L}")),
    Macro("numCifarReluBPCos", "[paper] CIFAR-10 ReLU CNN, BP: peak cosine", stat(E11, "cnnrelu_bp", "maxcos_l{L}")),
    Macro("numCifarReluAcc", "[paper] CIFAR-10 ReLU CNN DFA accuracy at 3000 steps", stat(E11, "cnnrelu", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluPixAcc", "[paper] CIFAR-10 ReLU CNN, centered error and inputs, accuracy at 3000 steps", stat(E11, "cnnrelu_centere_pixcenter", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluPriorAcc", "[paper] CIFAR-10 ReLU CNN, prior-calibrated bias, accuracy at 3000 steps", stat(E11, "cnnrelu_priorbias", "acc_3000"), "{:.3f}"),
    Macro("numCifarCNNMinP", "[paper] CIFAR-10 CNN, sigmoid head, DFA: windowed participation minimum of the deepest block", stat(E11, "cnn", "minp_w300_l{L}")),
    Macro("numCifarCNNBPMinP", "[paper] the same for the BP twin", stat(E11, "cnn_bp", "minp_w300_l{L}")),
    Macro("numCifarCalibCos", "[paper] CIFAR-10 CNN, sigmoid head with the output bias calibrated to cancel the head's mean prediction: peak cosine", stat(E12, "cnn_calib", "maxcos_l{L}")),
    Macro("numCifarCalibAcc", "[paper] its accuracy at 3000 steps", stat(E12, "cnn_calib", "acc_3000"), "{:.3f}"),
    Macro("numCifarCalibEbar", "[paper] its ||ebar_0||", stat(E12, "cnn_calib", "ebar0"), "{:.3f}"),
    Macro("numCifarPriorEbar", "[paper] ||ebar_0|| with the prior-calibrated bias alone on CIFAR-10", stat(E11, "cnn_priorbias", "ebar0"), "{:.3f}"),
    Macro("numCifarCalibCentereCos", "[paper] CIFAR-10 CNN, calibrated head plus centered error: peak cosine", stat(E12, "cnn_calib_centere", "maxcos_l{L}")),
    Macro("numCifarCalibCentereAcc", "[paper] its accuracy at 3000 steps", stat(E12, "cnn_calib_centere", "acc_3000"), "{:.3f}"),
    Macro("numMnistCalibCos", "[paper] MNIST base network with the calibrated head: peak cosine", stat(E12, "mnist_calib", "maxcos_l{L}")),
    Macro("numCifarCenterPixFastAcc", "[paper] CIFAR-10 CNN, centered error, centered inputs and tenfold head learning rate: accuracy at 3000 steps", stat(E12, "cnn_centere_pixcenter_outlr10", "acc_3000"), "{:.3f}"),
    Macro("numCifarCenterPixFastCos", "[paper] its peak cosine", stat(E12, "cnn_centere_pixcenter_outlr10", "maxcos_l{L}")),
    Macro("numCifarCenterPixCos", "[paper] CIFAR-10 CNN, centered error and centered inputs: peak cosine", stat(E11, "cnn_centere_pixcenter", "maxcos_l{L}")),
    Macro("numCifarCenterPixAcc", "[paper] its accuracy at 3000 steps", stat(E11, "cnn_centere_pixcenter", "acc_3000"), "{:.3f}"),
    Macro("numCifarMLPCalibLone", "[paper] CIFAR-10 MLP with the calibrated head: layer-1 peak cosine", stat(E12, "mlp_calib", "maxcos_l1")),
    Macro("numCifarMLPCenterPixLone", "[paper] CIFAR-10 MLP, centered error and inputs: layer-1 peak cosine", stat(E12, "mlp_centere_pixcenter", "maxcos_l1")),
    Macro("numCifarMLPCenterPixCos", "[paper] the same, top layer", stat(E12, "mlp_centere_pixcenter", "maxcos_l{L}")),
    Macro("numCifarMLPCenterPixAcc", "[paper] its accuracy at 3000 steps", stat(E12, "mlp_centere_pixcenter", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluSlowCos", "[paper] CIFAR-10 ReLU CNN at lr 3e-4, DFA: peak cosine", stat(E12, "cnnrelu_lr3e4", "maxcos_l{L}")),
    Macro("numCifarReluSlowBPCos", "[paper] the same for BP", stat(E12, "cnnrelu_lr3e4_bp", "maxcos_l{L}")),
    Macro("numCifarReluSlowAcc", "[paper] CIFAR-10 ReLU CNN at lr 3e-4, DFA accuracy at 3000 steps", stat(E12, "cnnrelu_lr3e4", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluSlowBPAcc", "[paper] the same for BP", stat(E12, "cnnrelu_lr3e4_bp", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluSlowPixAcc", "[paper] CIFAR-10 ReLU CNN at lr 3e-4, centered error and inputs: accuracy at 3000 steps", stat(E12, "cnnrelu_lr3e4_centere_pixcenter", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluSlowPixCos", "[paper] its peak cosine", stat(E12, "cnnrelu_lr3e4_centere_pixcenter", "maxcos_l{L}")),
    Macro("numCifarReluSlowCalibAcc", "[paper] CIFAR-10 ReLU CNN at lr 3e-4 with the calibrated head: accuracy at 3000 steps", stat(E12, "cnnrelu_lr3e4_calib", "acc_3000"), "{:.3f}"),
    Macro("numCifarReluSlowCalibCos", "[paper] its peak cosine", stat(E12, "cnnrelu_lr3e4_calib", "maxcos_l{L}")),
    Macro("numGaussRatioEarly", "[paper] median predicted/measured participation loss of the per-unit Gaussian reconstruction over steps 50 to 300, all layers and baseline runs", gauss("median_ratio_gauss_early"), "{:.2f}"),
    Macro("numGaussRatioLate", "[paper] the same at step 3000", gauss("median_ratio_gauss_3000"), "{:.2f}"),
    Macro("numGaussSharedHundred", "[paper] shared-spread reconstruction: median predicted/measured participation loss at step 100", gauss("median_ratio_shared_spread.100"), "{:.3f}"),
    Macro("numGaussZeroMeanHundred", "[paper] zero-mean reconstruction: median predicted/measured participation loss at step 100", gauss("median_ratio_zero_mean.100"), "{:.4f}"),
    Macro("numGaussSharedLate", "[paper] shared-spread reconstruction: median predicted/measured participation loss at step 3000", gauss("median_ratio_shared_spread.3000"), "{:.3f}"),
    Macro("numGaussZeroMeanLate", "[paper] zero-mean reconstruction: median predicted/measured participation loss at step 3000", gauss("median_ratio_zero_mean.3000"), "{:.3f}"),
    Macro("numLognormalRatioEarly", "[paper] the same ratio for the lognormal closed form 16 Var[log cosh mu] over steps 50 to 300", gauss("median_ratio_lognormal_early"), "{:.1f}"),
    Macro("numLognormalRatioLate", "[paper] the lognormal ratio at step 3000", gauss("median_ratio_lognormal_3000"), "{:.0f}"),
    Macro("numGaussRuns", "[paper] number of baseline trajectories in the reconstruction", gauss("n_runs"), "{:.0f}"),
    Macro("numPThreeHundred", "[paper] layer-3 participation at step 100, baseline", at_step(E1, "base", "p_l3", 100), "{:.2f}"),
    Macro("numPThreeFiveHundred", "[paper] layer-3 participation at step 500, baseline", at_step(E1, "base", "p_l3", 500), "{:.2f}"),
    Macro("numPThreeThreeK", "[paper] layer-3 participation at step 3000, baseline", at_step(E1, "base", "p_l3", 3000), "{:.2f}"),
    Macro("numPThreeInit", "[paper] layer-3 participation at initialization", at_step(E1, "base", "p_l3", 0), "{:.2f}"),
    Macro("numPiBasePlateau", "[paper] signed projected descent of the layer-3 DFA update on the BP gradient, mean over steps 100 to 300, baseline", traj_window_mean(E1, "base", "pi_l3", 100, 300), "{:+.2f}"),
    Macro("numPiBaseEnd", "[paper] the same at step 3000", at_step(E1, "base", "pi_l3", 3000), "{:+.1f}"),
    Macro("numPiAlignedPlateau", "[paper] the same over steps 100 to 300 with pre-aligned feedback", traj_window_mean(E1, "aligned", "pi_l3", 100, 300), "{:+.1f}"),
    Macro("numCosBasePlateau", "[paper] gradient cosine in layer 3, baseline steps 100 to 300", traj_window_mean(E1, "base", "cosA_l3", 100, 300), "{:+.3f}"),
    Macro("numCosBaseEnd", "[paper] gradient cosine in layer 3 at step 3000", at_step(E1, "base", "cosA_l3", 3000), "{:+.3f}"),
    Macro("numCosAlignedPlateau", "[paper] gradient cosine in layer 3, aligned steps 100 to 300", traj_window_mean(E1, "aligned", "cosA_l3", 100, 300), "{:+.3f}"),
    Macro("numAlignedKappa", "[paper] transient dose of layer 3 with pre-aligned feedback", kt(E1, "aligned", 3), "{:.2f}"),
    Macro("numAlignedKappaCF", "[paper] initialization estimate of the layer-3 collapse number with pre-aligned feedback (same as random feedback)", stat(E1, "aligned", "kappa_cf_l3"), "{:.1f}"),
    Macro("numAlignedPThree", "[paper] layer-3 participation minimum with pre-aligned feedback", stat(E1, "aligned", f"minp_w{W}_l3")),
    Macro("numSoftmaxEbar", "[paper] measured ||ebar_0|| of the balanced softmax head with a random output layer", stat(E1, "softmax", "ebar0"), "{:.2f}"),
    Macro("numResidBaseInit", "[paper] largest layer ratio ||r|| / ||gamma_bar * B ebar|| at initialization, baseline", resid("base_init_max"), "{:.2f}"),
    Macro("numResidBasePlateau", "[paper] the same over steps 100 to 300, baseline (largest over layers and steps)", resid("base_plateau_max"), "{:.2f}"),
    Macro("numResidReLUPlateau", "[paper] the same over steps 100 to 300 with rectified units", resid("relu_plateau_max"), "{:.2f}"),
    Macro("numResidBaseMid", "[paper] largest layer ratio at step 50 (midway through the collapse), baseline", resid("base_mid_max"), "{:.2f}"),
    Macro("numResidReLUInit", "[paper] largest layer ratio at initialization, rectified units", resid("relu_init_max"), "{:.2f}"),
    Macro("numResidReLUMid", "[paper] largest layer ratio at step 50, rectified units", resid("relu_mid_max"), "{:.2f}"),
    Macro("numResidCifarMid", "[paper] largest layer ratio at step 50, CIFAR-10 MLP", resid("cifar_mlp_mid_max"), "{:.2f}"),
    Macro("numResidBaseMidLayers", "[paper] ratio by layer at step 50, baseline", resid("base_mid_by_layer")),
    Macro("numResidReLUMidLayers", "[paper] ratio by layer at step 50, rectified units", resid("relu_mid_by_layer")),
    Macro("numResidCifarMidLayers", "[paper] ratio by layer at step 50, CIFAR-10 MLP", resid("cifar_mlp_mid_by_layer")),
    Macro("numResidBaseInitLayers", "[paper] ratio by layer at initialization, baseline", resid("base_init_by_layer")),
    Macro("numResidBasePlateauLayers", "[paper] ratio by layer over steps 100 to 300, baseline", resid("base_plateau_by_layer")),
    Macro("numResidReLUPlateauLayers", "[paper] ratio by layer over steps 100 to 300, rectified units", resid("relu_plateau_by_layer")),
    Macro("numResidCifarPlateauLayers", "[paper] ratio by layer over steps 100 to 300, CIFAR-10 MLP", resid("cifar_mlp_plateau_by_layer")),
    Macro("numResidBaseRInitFirst", "[paper] layer-1 residual norm at initialization, diagnostic baseline", resid("base_step0_norm_r_by_layer", idx=0), "{:.2f}"),
    Macro("numResidBaseRFinalFirst", "[paper] layer-1 residual norm at step 300, diagnostic baseline", resid("base_step300_norm_r_by_layer", idx=0), "{:.2f}"),
    Macro("numResidBaseLeadInitFirst", "[paper] layer-1 leading mean-teacher norm at initialization, diagnostic baseline", resid("base_step0_norm_lead_by_layer", idx=0), "{:.2f}"),
    Macro("numResidBaseLeadFinalFirst", "[paper] layer-1 leading mean-teacher norm at step 300, diagnostic baseline", resid("base_step300_norm_lead_by_layer", idx=0), "{:.2f}"),
    Macro("numWhitenBestLam", "[paper] ridge of the error-side conditioner with the best accuracy at 3000 steps", text("10", "results/E4_cures/summary.csv, tags base__whiten_lam*")),
    Macro("numWhitenBestCos", "[paper] peak top-layer cosine at that ridge", stat(E4, "base__whiten_lam10.0", "maxcos_l{L}")),
    Macro("numWhitenBestDuration", "[paper] plateau duration at that ridge", stat(E4, "base__whiten_lam10.0", "plateau_duration"), "{:.0f}"),
    Macro("numWhitenBestAcc", "[paper] accuracy change at that ridge, points at 1500", diff(E4, "base__whiten_lam10.0", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numWhitenSmallAcc", "[paper] accuracy change at the smallest ridge tested (0.1), points at 1500", diff(E4, "base__whiten", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numWhitenSmallDuration", "[paper] plateau duration at the smallest ridge", stat(E4, "base__whiten", "plateau_duration"), "{:.0f}"),
    Macro("numBaseCureDuration", "[paper] plateau duration of plain DFA in the cure bake-off", stat(E4, "base__none", "plateau_duration"), "{:.0f}"),
    Macro("numCureBNCos", "[paper] peak top-layer cosine with batch normalization", stat(E4, "base__bn", "maxcos_l{L}")),
    Macro("numCureMuonCos", "[paper] peak top-layer cosine with Muon-orthogonalized updates", stat(E4, "base__muon", "maxcos_l{L}")),
    Macro("numCureOutlrCos", "[paper] peak top-layer cosine with a tenfold head learning rate", stat(E4, "base__outlr10", "maxcos_l{L}")),
    Macro("numCureOutlr", "[paper] accuracy change of a tenfold head learning rate, points at 1500",
          diff(E4, "base__outlr10", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureMuon", "[paper] accuracy change of Muon-orthogonalized updates, points at 1500",
          diff(E4, "base__muon", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureBN", "[paper] accuracy change of batch normalization, points at 1500",
          diff(E4, "base__bn", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureWhiten", "[paper] accuracy change of the error-side conditioner, points at 1500",
          diff(E4, "base__whiten", "base__none", "acc_1500", scale=100), "{:+.1f}"),
    Macro("numCureCenterMinP", "layer-3 participation minimum with a centered teaching signal",
          best_of(stat(E4, "base__center_delta", f"minp_w{W}_l3"), stat(E1, "center_delta", f"minp_w{W}_l3"))),
    Macro("numCureCenterOverhead", "[paper] wall clock of teaching-signal centering relative to plain DFA",
          ratio(E4, "base__center_delta", "base__none", "wall"), "{:.2f}"),
    Macro("numCureCenterEOverhead", "[paper] wall clock of output-error centering relative to plain DFA", ratio(E4, "base__center_e", "base__none", "wall"), "{:.2f}"),
    Macro("numCurePriorOverhead", "[paper] wall clock of the prior-calibrated bias relative to plain DFA", ratio(E4, "base__priorbias", "base__none", "wall"), "{:.2f}"),
    Macro("numCureOutlrOverhead", "[paper] wall clock of the tenfold head learning rate relative to plain DFA", ratio(E4, "base__outlr10", "base__none", "wall"), "{:.2f}"),
    Macro("numCureBNOverhead", "[paper] wall clock of batch normalization relative to plain DFA", ratio(E4, "base__bn", "base__none", "wall"), "{:.2f}"),
    Macro("numCureWhitenOverhead", "[paper] wall clock of the error-side conditioner relative to plain DFA",
          ratio(E4, "base__whiten", "base__none", "wall"), "{:.2f}"),
    Macro("numCureMuonOverhead", "[paper] wall clock of Muon-orthogonalized updates relative to plain DFA",
          ratio(E4, "base__muon", "base__none", "wall"), "{:.2f}"),

    # ---- figure 6: generality --------------------------------------------------------------------------------------
    Macro("numFACos", "[paper] peak hidden cosine under layerwise feedback alignment",
          best_of(stat(E1, "fa", "maxcos_l{L}"), stat(E3, "fa", "maxcos_l{L}"))),
    Macro("numFADuration", "[paper] plateau duration under layerwise feedback alignment",
          best_of(stat(E1, "fa", "plateau_duration"), stat(E3, "fa", "plateau_duration")), "{:.0f}"),
    Macro("numFAMinPthree", "layer-3 participation minimum under layerwise feedback alignment",
          best_of(stat(E1, "fa", f"minp_w{W}_l3"), stat(E3, "fa", f"minp_w{W}_l3"))),
    Macro("numDepthSixCos", "[paper] peak hidden cosine in the deepest layer at depth six",
          stat(E3, "depth6", "maxcos_l{L}")),
    Macro("numDepthSixP", "[paper] participation minimum in the deepest layer at depth six",
          stat(E3, "depth6", f"minp_w{W}_l6")),
    Macro("numDepthSixDuration", "[paper] plateau duration at depth six", stat(E3, "depth6", "plateau_duration"), "{:.0f}"),
    Macro("numCNNConvOneCos", "[paper] peak hidden cosine of the first convolutional block "
                        "(the manuscript writes numCNNConv1Cos and numCNNConv1BP; both resolve to this macro)",
          stat(E3, "cnn", "maxcos_l1")),
    Macro("numCNNConvOneBP", "peak hidden cosine of the first convolutional block under the BP twin",
          stat(E3, "cnn_bp", "maxcos_l1")),
    Macro("numCNNfcP", "[paper] participation minimum of the fully connected block of the CNN",
          stat(E3, "cnn", f"minp_w{W}_l3")),
    Macro("numMultilabelPrior", "[paper] prior loss of the multi-label head (its own plateau value)",
          stat(E3, "multilabel", "prior_loss"), "{:.4f}"),
    Macro("numMultilabelCos", "peak hidden cosine with a multi-label head", stat(E3, "multilabel", "maxcos_l{L}")),
    Macro("numReLUDead", "[paper] fraction of top-layer ReLU units that are dead or always on under DFA",
          relu_constant_frac(E3, "relu"), "{:.2f}"),
    Macro("numReLUDeadBP", "[paper] the same fraction under the BP twin", relu_constant_frac(E3, "relu", bp=True), "{:.2f}"),
    Macro("numReLUPlateauExcess", "[paper] how far the ReLU plateau sits above the class-prior loss",
          plateau_excess(E3, "relu"), "{:.0f}\\%"),
    Macro("numMSERatio", "[paper] common-mode fraction ||ebar||/||etil|| for squared error with targets of magnitude 5",
          stat(E3, "mse_s5", "ebar_ratio_10"), "{:.2f}"),
    Macro("numBCERatio", "[paper] common-mode fraction ||ebar||/||etil|| for the sigmoid head",
          stat(E1, "base", "ebar_ratio_10"), "{:.2f}"),
    Macro("numMSECos", "peak hidden cosine with a squared-error head and unit targets", stat(E3, "mse_s1", "maxcos_l{L}")),
    Macro("numFashionCos", "peak hidden cosine on Fashion-MNIST", stat(E3, "fashion", "maxcos_l{L}")),
    Macro("numGenR", "[paper] correlation between the common-mode fraction at step 10 and the steps spent collapsed",
          derived("generality_ratio_vs_collapsed_steps.r"), "{:.2f}"),
    Macro("numGenRCI", "[paper] 95 percent interval of that correlation",
          derived_range("generality_ratio_vs_collapsed_steps.lo", "generality_ratio_vs_collapsed_steps.hi")),
    Macro("numGenRCos", "[paper] correlation between the mean-to-input-dependent error ratio at step 10 and peak hidden cosine", derived("generality_ratio_vs_maxcos.r"), "{:.2f}"),
    Macro("numGenConditions", "[paper] distinct physical settings in the generality correlation", derived("generality_ratio_vs_maxcos.n_conditions"), "{:.0f}"),
    Macro("numGenRCosCI", "[paper] 95 percent interval of that correlation", derived_range("generality_ratio_vs_maxcos.lo", "generality_ratio_vs_maxcos.hi")),
    Macro("numGenN", "[paper] number of DFA-family runs in the generality correlation", derived("generality_ratio_vs_collapsed_steps.n"), "{:.0f}"),
    Macro("numImbScratchCos", "[paper] peak hidden cosine of the from-scratch imbalance control",
          best_of(stat(E1, "softmax_imb07", "maxcos_l{L}"), stat(E2, "smimb0.7", "maxcos_l{L}"))),
    Macro("numImbScratchBP", "[paper] peak hidden cosine of its BP twin", stat(E1, "softmax_imb07_bp", "maxcos_l{L}")),
    Macro("numShiftCos", "[paper] peak top-layer hidden cosine after a label shift imposed at step 1500 on a trained softmax network",
          derived("shift.shift1500.cos_post_max"), "{:.2f}"),
    Macro("numShiftEbarPre", "[paper] batch ||ebar|| 10 steps after the shift", derived("shift.shift1500.ebar_shift_10"), "{:.2f}"),
    Macro("numShiftEbarPost", "[paper] batch ||ebar|| 50 steps after the shift", derived("shift.shift1500.ebar_shift_50"), "{:.2f}"),
    Macro("numShiftMinorityPre", "[paper] balanced probe accuracy at the shift", derived("shift.shift1500.bal_pre"), "{:.2f}"),
    Macro("numShiftMinorityPost", "[paper] balanced probe accuracy 250 steps after the shift", derived("shift.shift1500.bal_post_250"), "{:.2f}"),
    Macro("numShiftBPMinorityPre", "[paper] the same at the shift for the BP twin", derived("shift.shift1500_bp.bal_pre"), "{:.2f}"),
    Macro("numShiftBPMinorityPost", "[paper] the same 250 steps after the shift for the BP twin", derived("shift.shift1500_bp.bal_post_250"), "{:.2f}"),
    Macro("numShiftKappa", "[paper] top-layer collapse dose accumulated after the shift at step 1500 to the end of the run", derived("shift.shift1500.dose_post"), "{:.2f}"),
    Macro("numShiftEbarAtShift", "[paper] batch ||ebar|| on the first shifted minibatch, shift at step 1500", derived("shift.shift1500.ebar_shift_0"), "{:.2f}"),
    Macro("numShiftHL", "[paper] top-layer mean-activity energy at the shift", derived("shift.shift1500.H_L_at_shift"), "{:.0f}"),
    Macro("numShiftEarlyCos", "[paper] peak top-layer cosine after a shift imposed at step 300", derived("shift.shift300.cos_post_max"), "{:.2f}"),
    Macro("numShiftEarlyKappa", "[paper] top-layer collapse dose accumulated after a shift at step 300", derived("shift.shift300.dose_post"), "{:.2f}"),
    Macro("numShiftEarlyEbar", "[paper] batch ||ebar|| 10 steps after a shift at step 300", derived("shift.shift300.ebar_shift_10"), "{:.2f}"),
    Macro("numHeadresetCos", "[paper] peak top-layer cosine after a label shift with gated error centering at step 1500 (legacy tag headreset1500)", derived("shift.headreset1500.cos_post_max"), "{:.2f}"),
    Macro("numHeadMSECtrCos", "[paper] peak cosine, squared error with centered targets and a random head", derived("headinit.mse_ctr.maxcos"), "{:.3f}"),
    Macro("numHeadMSECtrEbar", "[paper] ||ebar_0|| in that condition", derived("headinit.mse_ctr.ebar0"), "{:.2f}"),
    Macro("numHeadMSECtrWzeroCos", "[paper] peak cosine, centered targets with the head weights initialized at zero", derived("headinit.mse_ctr_w0.maxcos"), "{:.3f}"),
    Macro("numHeadMSECtrWzeroEbar", "[paper] ||ebar_0|| with the zero head", derived("headinit.mse_ctr_w0.ebar0"), "{:.2f}"),
    Macro("numHeadMSECtrWzeroKappa", "[paper] transient dose with the zero head", derived("headinit.mse_ctr_w0.kt"), "{:.2f}"),
    Macro("numHeadMSECtrKappa", "[paper] transient dose with the random head", derived("headinit.mse_ctr.kt"), "{:.2f}"),
    Macro("numHeadSigmoidWzeroCos", "[paper] peak cosine, sigmoid head with zero output weights", derived("headinit.sigmoid_w0.maxcos"), "{:.3f}"),
    Macro("numHeadSoftmaxWzeroCos", "[paper] peak cosine, balanced softmax with zero output weights", derived("headinit.softmax_w0.maxcos"), "{:.2f}"),
    Macro("numHeadMSEWzeroCos", "[paper] peak cosine, one-hot squared error with zero output weights", derived("headinit.mse_w0.maxcos"), "{:.2f}"),
    Macro("numHeadMSEWzeroEbar", "[paper] ||ebar_0|| for one-hot squared error with zero output weights (the label mean)", derived("headinit.mse_w0.ebar0"), "{:.2f}"),
]

for name,key,fmt in [
    ("AdamFastExit","adam_fast_exit","{:.0f}"),("AdamSlowExit","adam_slow_exit","{:.0f}"),
    ("AdamSGDExit","sgd_exit","{:.0f}"),("AdamFastP","adam_fast_minp","{:.3f}"),
    ("AdamSGDP","sgd_minp","{:.3f}"),("AdamDurationRatio","adam_duration_ratio","{:.1f}"),
    ("DriftUpstreamInit","drift_upstream_0","{:.1f}"),("DriftUpstreamFifty","drift_upstream_50","{:.2f}"),
    ("DriftCovFifty","drift_covariance_50","{:.2f}"),("DriftResidFifty","drift_residual_50","{:.2f}"),
    ("MaskedFull","masked_full_cos","{:.2f}"),("MaskedFixed","masked_fixed120_cos","{:.2f}"),
    ("MaskedCurrent","masked_current_cos","{:.2f}"),("MaskedRandom","masked_random_cos","{:.2f}"),
    ("MaskedComplement","masked_complement120_cos","{:.2f}"),
    ("NoiseSlopeMin","noise_slope_min","{:.3f}"),("NoiseSlopeMax","noise_slope_max","{:.3f}"),
    ("CifarHundredSigError","cifar100_sigmoid_bce_mean_error_init","{:.2f}"),
    ("CifarHundredSoftError","cifar100_softmax_ce_mean_error_init","{:.2f}"),
    ("CifarHundredSigCos","cifar100_sigmoid_bce_maxcos","{:.3f}"),
    ("CifarHundredSoftCos","cifar100_softmax_ce_maxcos","{:.3f}"),
    ("RowNormCos","row_norm_maxcos","{:.3f}"),("RowNormP","row_norm_minp","{:.3f}"),
    ("SigmoidP","sigmoid_minp","{:.3f}"),("SigmoidBPP","sigmoid_bp_minp","{:.3f}")]:
    REGISTRY.append(Macro("num"+name,"[paper] revision measurement: "+key,revision(key),fmt))


def review_value(key):
    def read(ctx):
        path=os.path.join(ctx.root,'review_20260919','summary.json')
        with open(path) as handle:value=json.load(handle).get(key)
        return value,f'results/review_20260919/summary.json: {key}; scripts/review_reporting.py'
    return read


for name,fmt in [('ConditionR','{:.2f}'),('ConditionCI','{}'),('PooledCI','{}'),
                 ('SGDLearn','{:.0f}'),('AdamLearn','{:.0f}'),('AdamSlowLearn','{:.0f}'),
                 ('MatchedCovDrop','{:.1f}'),('MatchedRDrop','{:.2f}'),('RawSoftmaxCos','{:.2f}'),
                 ('InitialDecode','{:.1f}'),('CollapseDecode','{:.1f}'),('RecoveryDecode','{:.1f}'),
                 ('FrozenRaw','{:.1f}'),('FrozenCentered','{:.1f}'),('FrozenScaled','{:.1f}'),
                 ('HiddenUpstreamInitial','{:.2f}'),('HiddenMeanFifty','{:.2f}'),('ReadoutMeanFifty','{:.2f}'),
                 ('WindowFixedR','{:.2f}'),('WindowRescaledR','{:.2f}'),
                 ('WindowFixedMAE','{:.3f}'),('WindowRescaledMAE','{:.3f}'),
                 ('RateHid','{:.2f}'),('RateOut','{:.2f}'),('RateRtwo','{:.2f}'),('RateRuns','{:.0f}'),('RateEligible','{:.0f}')]:
    REGISTRY.append(Macro('numReview'+name,'[paper] review validation: '+name,review_value(name),fmt))


def submission_value(key):
    def read(ctx):
        path=os.path.join(ctx.root,'submission_20260923','summary.json')
        with open(path) as handle:value=json.load(handle).get(key)
        return value,f'results/submission_20260923/summary.json: {key}; scripts/submission_reporting.py'
    return read


for name,key,fmt in [
    ('CostTimeBase','time_base','{:.0f}'),('CostTimePrior','time_priorbias','{:.0f}'),
    ('CostTimeBoth','time_pixcenter_frzbias','{:.0f}'),('CostTimeCenterE','time_center_e','{:.0f}'),
    ('CostTimeBP','time_bp','{:.0f}'),
    ('CostFiftyBase','acc50_base','{:.0f}'),('CostFiftyBoth','acc50_pixcenter_frzbias','{:.0f}'),
    ('CostFiftyCenterE','acc50_center_e','{:.0f}'),('CostFiftyPrior','acc50_priorbias','{:.0f}'),
    ('CostFiftyBP','acc50_bp','{:.0f}'),
    ('CostEightyBase','acc80_base','{:.0f}'),('CostEightyPrior','acc80_priorbias','{:.0f}'),
    ('SelfTermBase','ebar_ratio10_base','{:.2f}'),('SelfTermBP','ebar_ratio10_bp','{:.2f}'),
    ('SelfTermBPx','ebar_ratio10_bp_outx12','{:.2f}'),('SelfTermAligned','ebar_ratio10_aligned','{:.2f}'),
    ('BPxEbar','ebar0_bp_outx12','{:.2f}'),
    ('NullR','null_conditions_r','{:.2f}'),('NullPooledR','null_pooled_r','{:.2f}'),
    ('NullMAE','null_conditions_mae','{:.2f}'),('ModelCondMAE','model_conditions_mae','{:.2f}'),
    ('ReadoutInitRaw','readout_initial_raw','{:.1f}'),
    ('RMSFold','rms3_fold','{:.1f}'),('RMSRateFold','rms3_rate_fold','{:.0f}'),
    ('StdLOneCos','l1_cos_std','{:.2f}'),('RawLOneCos','l1_cos_raw','{:.2f}'),
    ('StdLOneP','l1_minp_std','{:.2f}'),('RawLOneP','l1_minp_raw','{:.2f}'),
    ('EbarHundred','ebar_100','{:.2f}'),('EtilPlateauLo','etil_plateau_min','{:.2f}'),
    ('EtilPlateauHi','etil_plateau_max','{:.2f}'),('EtilEnd','etil_3000','{:.2f}')]:
    REGISTRY.append(Macro('num'+name,'[paper] submission reporting: '+key,submission_value(key),fmt))

for name,key,fmt in [
    ('SignCosEnd','sign_cos3_end','{:.2f}'),('SignPEnd','sign_p3_end','{:.2f}'),('SignAcc','sign_accuracy','{:.2f}'),
    ('SignCenterCosEnd','sign_center_e_cos3_end','{:.2f}'),('SignCenterAcc','sign_center_e_accuracy','{:.2f}'),
    ('SignCenterP','sign_center_e_p3','{:.2f}'),
    ('GainLowP','gain_0p03_p3','{:.2f}'),('GainLowTime','gain_0p03_time','{:.0f}'),
    ('GainBaseTime','gain_1p0_time','{:.0f}'),('GainHighTime','gain_10p0_time','{:.0f}'),('GainHighP','gain_10p0_p3','{:.3f}'),
    ('CondDefaultAcc','cond_default_acc','{:.2f}'),('CNNDeepP','cnn_minp3','{:.3f}'),('CNNDeepBPP','cnn_bp_minp3','{:.2f}'),
    ('AdamBPCos','adam_bp_cos','{:.2f}'),('AdamBPP','adam_bp_minp','{:.2f}'),
    ('ReadoutInitScaled','readout_initial_scaled','{:.1f}'),
    ('ValidModelR','valid_model_conditions_r','{:.2f}'),('ValidNullR','valid_null_conditions_r','{:.2f}'),
    ('ValidConds','valid_conditions','{:.0f}'),
    ('SweepRMin','sweep_r_min','{:.2f}'),('SweepRMax','sweep_r_max','{:.2f}'),('SweepRClass','sweep_r_class_count','{:.2f}'),
    ('CenteredVarBase','centered_var_min_base','{:.3f}'),('CenteredVarBP','centered_var_min_bp','{:.2f}'),
    ('CenteredVarPrior','centered_var_min_priorbias','{:.2f}'),('CenteredVarAligned','centered_var_min_aligned','{:.2f}'),
    ('CenteredRankInit','centered_rank0_base','{:.0f}'),('CenteredRankBase','centered_rank_min_base','{:.1f}'),
    ('CenteredRankBP','centered_rank_min_bp','{:.0f}'),('CenteredRankCenterE','centered_rank_min_center_e','{:.1f}'),
    ('CenteredRankPrior','centered_rank_min_priorbias','{:.1f}'),
    ('InstabLinearHzero','instab_linear_raw_H0','{:.0f}'),('InstabLinearHend','instab_linear_raw_Hend_tex','{}'),
    ('InstabLinearCenteredHmax','instab_linear_centered_Hmax','{:.1f}'),
    ('InstabReluHend','instab_relu_raw_Hend_tex','{}'),('InstabGeluHend','instab_gelu_raw_Hend_tex','{}'),
    ('LrDFAHighEighty','lr_dfa_0p1_acc80','{:.0f}'),('LrPriorHighEighty','lr_prior_0p1_acc80','{:.0f}'),
    ('LrBPHighEighty','lr_bp_0p1_acc80','{:.0f}'),('LrDFAHighP','lr_dfa_0p1_p3','{:.3f}'),
    ('LrDFAHighFifty','lr_dfa_0p1_acc50','{:.0f}'),('LrPriorHighFifty','lr_prior_0p1_acc50','{:.0f}'),
    ('LrSavingLow','lr_saving80_0p001','{:.0f}'),('LrSavingHigh','lr_saving80_0p1','{:.0f}'),
    ('ImbUncalCos','imbcal_imb_cos3','{:.2f}'),('ImbCalCos','imbcal_calib_cos3','{:.2f}'),
    ('ImbUncalBalEarly','imbcal_imb_bal450','{:.2f}'),('ImbCalBalEarly','imbcal_calib_bal450','{:.2f}'),
    ('ImbUncalBalEnd','imbcal_imb_bal_end','{:.2f}'),('ImbCalBalEnd','imbcal_calib_bal_end','{:.2f}'),
    ('ImbCalEbar','imbcal_calib_ebar0','{:.2f}')]:
    REGISTRY.append(Macro('num'+name,'[paper] submission reporting: '+key,submission_value(key),fmt))

# The original DFA protocol (E19): arm x RMSprop rate x statistic, e.g. numNokDFAZeroFastCos.
for arm_name,arm in (('DFAZero','dfa_zero'),('BPFanin','bp_fanin'),('DFAXavier','dfa_xavier')):
    for rate_name,rate in (('Fast','fast'),('Slow','slow')):
        for stat_name,col,fmt in (('Cos','cos3','{:.3f}'),('Sat','sat3','{:.2f}'),('Chi','chi3','{:.3f}'),
                                  ('P','p3','{:.2f}'),('Chance','chance','{:.0f}'),('Time','time','{:.0f}'),
                                  ('Eighty','acc80','{:.0f}'),('Acc','accuracy','{:.3f}')):
            key=f'nokland_{arm}_{rate}_{col}'
            REGISTRY.append(Macro(f'numNok{arm_name}{rate_name}{stat_name}',
                                  '[paper] original DFA protocol: '+key,submission_value(key),fmt))


def used_macros(paper_dir):
    """Macro names that the manuscript sections reference."""
    names = set()
    for f in sorted(glob.glob(os.path.join(paper_dir, "sections", "*.tex"))) + \
             [os.path.join(paper_dir, "main.tex")]:
        if not os.path.exists(f): continue
        with open(f) as fh: names |= set(re.findall(r"\\(num[A-Za-z]+)", fh.read()))
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "numbers.tex"))
    ap.add_argument("--paper", default=os.path.join(ROOT, "paper"))
    args = ap.parse_args()

    names = [m.name for m in REGISTRY]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup: raise SystemExit(f"duplicate macro names in the registry: {dup}")

    ctx = Ctx(os.path.abspath(args.results))
    lines = ["% paper/numbers.tex — generated by scripts/make_numbers.py. Do not edit by hand.",
             f"% results root: {os.path.relpath(ctx.root, ROOT)}   generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
             f"% {len(REGISTRY)} macros. '??' means the experiment behind that number has not produced data yet.",
             ""]
    missing = []
    for m in REGISTRY:
        try:
            value, src = m.fn(ctx)
        except Exception as exc:                 # a broken source must never stop numbers.tex from being written
            value, src = None, f"failed: {type(exc).__name__}: {exc}"
        text = _fmt(value, m.fmt)
        if text == MISSING: missing.append(m.name)
        lines += [f"% {m.desc}", f"%   source: {src}", f"\\newcommand{{\\{m.name}}}{{{text}}}", ""]
    with open(args.out, "w") as fh: fh.write("\n".join(lines))

    print(f"wrote {args.out}: {len(REGISTRY)} macros, {len(missing)} still '{MISSING}'")
    if missing: print("  missing: " + ", ".join(missing))
    undefined = sorted(used_macros(args.paper) - set(names))
    if undefined:
        print(f"  WARNING: {len(undefined)} macro(s) used by the manuscript but not in the registry: "
              + ", ".join(undefined))
    return 0


if __name__ == "__main__":
    sys.exit(main())
