"""Post-hoc analysis: plateau detection with a chance-accuracy guard, collapse summaries, kappa checks."""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd

def load_runs(pattern):
    """pattern like results/exp/tag/seed*_fb*.csv -> list of (df, meta)."""
    out = []
    for f in sorted(glob.glob(pattern)):
        df = pd.read_csv(f); meta = json.load(open(f.replace(".csv", ".meta.json"))); out.append((df, meta))
    return out

def plateau(df, Lprior, prior, tol=0.03, min_len=50, col="probe_loss"):
    """Window where the loss is within tol of the constant-predictor loss for >= min_len steps and probe accuracy is at chance
    (<= max prior + 0.05). Returns dict(onset, exit, duration, band_acc) or None. exit = first later step with loss < 0.9 Lprior."""
    s = df.step.values; L = df[col].values; acc = df.probe_acc.values
    sm = pd.Series(L).rolling(5, center=True, min_periods=1).mean().values
    inband = np.abs(sm - Lprior) / Lprior < tol
    chance = acc <= max(prior) + 0.05
    ok = inband & chance
    # longest run of ok
    best = None; i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]: j += 1
            if s[j] - s[i] >= min_len and (best is None or s[j] - s[i] > best[1] - best[0]): best = (s[i], s[j])
            i = j + 1
        else: i += 1
    if best is None: return None
    later = np.where((s > best[1]) & (sm < 0.9 * Lprior))[0]
    exit_step = int(s[later[0]]) if len(later) else None
    m = (s >= best[0]) & (s <= best[1])
    return dict(onset=int(best[0]), band_end=int(best[1]), exit=exit_step, duration=(exit_step - best[0]) if exit_step else None, band_acc=float(acc[m].mean()))

def collapse_summary(df, meta, window=None):
    """Peak hidden cosine and min participation per layer (optionally restricted to the collapse window step <= window)."""
    L = len(meta["widths"]); d = {}
    sub = df if window is None else df[df.step <= window]
    for l in range(1, L + 1):
        d[f"maxcos_l{l}"] = float(sub[f"cos_l{l}"].max()); d[f"minp_l{l}"] = float(sub[f"p_l{l}"].min())
        d[f"kappa_l{l}"] = float(df[f"kappa_l{l}"].iloc[-1]) if f"kappa_l{l}" in df else float("nan")
    d["steps_cos_gt_0.9"] = int(((df[f"cos_l{L}"] > 0.9).sum()) * (df.step.iloc[1] - df.step.iloc[0])) if len(df) > 1 else 0
    for s in (450, 1000, 1500, 3000):
        r = df[df.step == s]
        if len(r): d[f"acc_{s}"] = float(r.probe_acc.iloc[0])
    pl = plateau(df, meta["prior_loss"], meta["prior"]); d.update({f"plateau_{k}": v for k, v in (pl or {}).items()})
    d["ebar0"] = float(df.ebar_norm.iloc[0]); d["ebar_ratio_10"] = float(df[df.step == 10].ebar_norm.iloc[0] / df[df.step == 10].etil_norm.iloc[0]) if (df.step == 10).any() else float("nan")
    return d

def summarize_exp(exp_dir):
    """Aggregate all tags of an experiment: mean/sd over seeds of the collapse summary. Writes summary.csv."""
    rows = []
    for tag in sorted(os.listdir(exp_dir)):
        for df, meta in load_runs(os.path.join(exp_dir, tag, "seed*_fb*.csv")):
            r = collapse_summary(df, meta); r.update(tag=tag, seed=meta["seed"], fb_seed=meta["fb_seed"]); rows.append(r)
    if not rows: return None
    per = pd.DataFrame(rows); per.to_csv(os.path.join(exp_dir, "summary_per_run.csv"), index=False)
    agg = per.groupby("tag").agg(["mean", "std", "count"]); agg.to_csv(os.path.join(exp_dir, "summary.csv"))
    return per, agg

def ebar0_closed_form(head, C, q=0.5, pi=None):
    """||e_bar_0||: sigmoid one-vs-rest with output prob q -> sqrt(C)|q-1/C| (balanced); softmax under prior pi -> ||1/C - pi||."""
    if head == "sigmoid_bce": return float(np.sqrt(C) * abs(q - 1.0 / C))
    if head == "softmax_ce": return float(np.linalg.norm(1.0 / C - np.asarray(pi)))
    raise ValueError(head)

# ---------------------------------------------------------------------------------------------------------------------
# Additions for the reporting pipeline (scripts/summarize_all.py, scripts/make_numbers.py, scripts/figures/*).
# Everything below is new; the functions above keep their signatures and behaviour.
# ---------------------------------------------------------------------------------------------------------------------
import math
from scipy import stats as _stats

P_WINDOW = 300          # participation minima are reported windowed to step <= P_WINDOW (notes/PROTOCOL.md)
ACC_STEPS = (450, 1000, 1500, 3000)
CONFIG_COLS = ("rule", "dataset", "preprocess", "arch", "act", "head", "center")
SCALAR_META = ("fb_scale", "lr", "out_lr_mult", "hid_lr_mult", "batch", "width", "depth", "steps", "out_bias_q",
               "freeze_bias", "freeze_out", "muon", "whiten", "batchnorm", "p0", "target_scale", "classes",
               "C", "prior_loss", "xbar2", "wall", "final_step")

def run_paths(exp_dir):
    """[(tag, csv_path)] for every run under an experiment directory, sorted."""
    out = []
    for tag in sorted(os.listdir(exp_dir)):
        d = os.path.join(exp_dir, tag)
        if not os.path.isdir(d): continue
        for f in sorted(glob.glob(os.path.join(d, "seed*_fb*.csv"))): out.append((tag, f))
    return out

def read_meta(csv_path):
    """The meta.json beside a run csv, or None if it is missing (run still in flight)."""
    p = csv_path.replace(".csv", ".meta.json")
    if not os.path.exists(p): return None
    try:
        with open(p) as fh: return json.load(fh)
    except (json.JSONDecodeError, OSError): return None

def acc_at_step(df, s, col="probe_acc"):
    """Probe accuracy at step s (last logged step <= s); NaN if the run never reached s."""
    if len(df) == 0 or df.step.max() < s: return float("nan")
    sub = df[df.step <= s]
    return float(sub[col].iloc[-1]) if len(sub) else float("nan")

def sigma_prime_eff(C):
    """Effective output slope in the collapse number: (1/2 - 1/C) / |logit(1/C)| (0.182 at C = 10)."""
    q = 1.0 / C
    return float((0.5 - q) / abs(math.log(q / (1.0 - q)))) if C > 2 else float("nan")

def kappa_closed_form(df, meta):
    """kappa_l = g||ebar_0||(H_{l-1}+1) / [sigma'_eff (H_L+1) (eta_out/eta_hid)], all quantities read at initialization."""
    n = len(meta["widths"]); out = {f"kappa_cf_l{l}": float("nan") for l in range(1, n + 1)}
    if meta.get("head") not in ("sigmoid_bce", "multilabel") or len(df) == 0: return out
    row0 = df.iloc[0]
    sp = sigma_prime_eff(meta["C"])
    if not math.isfinite(sp) or sp <= 0: return out
    HL = float(row0.get(f"H_l{n}", float("nan")))
    ratio = meta.get("out_lr_mult", 1.0) / max(meta.get("hid_lr_mult", 1.0), 1e-12)
    drive = meta.get("fb_scale", 1.0) * float(row0.get("ebar_norm", float("nan")))
    for l in range(1, n + 1):
        Hprev = float(meta["xbar2"]) if l == 1 else float(row0.get(f"H_l{l-1}", float("nan")))
        out[f"kappa_cf_l{l}"] = drive * (Hprev + 1.0) / (sp * (HL + 1.0) * ratio)
    return out

def enrich_row(csv_path, meta, window=P_WINDOW):
    """Extra per-run columns that collapse_summary does not provide: windowed participation minima, initial-state
    readouts, robust accuracies at fixed steps, the closed-form collapse number and the flattened configuration."""
    df = pd.read_csv(csv_path)
    n = len(meta["widths"]); sub = df[df.step <= window]
    d = {"csv": os.path.relpath(csv_path, os.path.dirname(os.path.dirname(os.path.dirname(csv_path)))), "n_layers": n}
    for l in range(1, n + 1):
        d[f"minp_w{window}_l{l}"] = float(sub[f"p_l{l}"].min()) if f"p_l{l}" in sub else float("nan")
        d[f"maxcos_w{window}_l{l}"] = float(sub[f"cos_l{l}"].max()) if f"cos_l{l}" in sub else float("nan")
        d[f"cos0_l{l}"] = float(df[f"cos_l{l}"].iloc[0]) if f"cos_l{l}" in df else float("nan")
        d[f"p0_l{l}"] = float(df[f"p_l{l}"].iloc[0]) if f"p_l{l}" in df else float("nan")
        d[f"H0_l{l}"] = float(df[f"H_l{l}"].iloc[0]) if f"H_l{l}" in df else float("nan")
        d[f"wa_end_l{l}"] = float(df[f"wa_l{l}"].dropna().iloc[-1]) if f"wa_l{l}" in df and df[f"wa_l{l}"].notna().any() else float("nan")
    for s in ACC_STEPS: d[f"acc_{s}"] = acc_at_step(df, s)
    d["acc_final"] = float(df.probe_acc.iloc[-1]); d["step_final"] = int(df.step.iloc[-1])
    d["loss_final"] = float(df.probe_loss.iloc[-1]); d["loss_min"] = float(df.probe_loss.min())
    d.update(kappa_closed_form(df, meta))
    d["drive"] = meta.get("fb_scale", 1.0) * float(df.ebar_norm.iloc[0])
    d["ebar0_pred"] = ebar0_closed_form(meta["head"], meta["C"]) if meta["head"] in ("sigmoid_bce",) else float("nan")
    for k in SCALAR_META:
        if k in meta: d[k] = meta[k]
    for k in CONFIG_COLS:
        if k in meta: d[k] = meta[k]
    return d

def enrich_per_run(exp_dir, per, window=P_WINDOW):
    """Add the columns of enrich_row to the per-run table returned by summarize_exp, matched on (tag, seed, fb_seed)."""
    rows = []
    for tag, f in run_paths(exp_dir):
        meta = read_meta(f)
        if meta is None: continue
        try: r = enrich_row(f, meta, window=window)
        except (KeyError, ValueError, pd.errors.EmptyDataError) as exc:
            print(f"  warning: cannot summarize {f}: {exc}"); continue
        r.update(tag=tag, seed=meta["seed"], fb_seed=meta["fb_seed"]); rows.append(r)
    if not rows: return per
    ext = pd.DataFrame(rows)
    keys = ["tag", "seed", "fb_seed"]
    per = per.drop(columns=[c for c in ext.columns if c in per.columns and c not in keys])
    out = per.merge(ext, on=keys, how="outer")
    for c in [c for c in out.columns if c.startswith("plateau_")]: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def aggregate_tags(per):
    """Flat per-tag table: <col>_mean, <col>_sd for every numeric column, n_runs, and the configuration of the first run."""
    num = [c for c in per.columns if pd.api.types.is_numeric_dtype(per[c]) and c not in ("seed", "fb_seed")]
    g = per.groupby("tag", sort=True)
    agg = pd.concat([g[num].mean().add_suffix("_mean"), g[num].std(ddof=1).add_suffix("_sd"),
                     g[num].count().add_suffix("_n")], axis=1)
    agg.insert(0, "n_runs", g.size())
    for c in CONFIG_COLS:
        if c in per.columns: agg[c] = g[c].first()
    return agg.reset_index()

def summarize_readable(exp_dir):
    """The per-run table of summarize_exp, computed over the runs that can currently be read. Used as a fallback
    while a sweep is in flight and some outputs are half-written."""
    rows, bad = [], []
    for tag, f in run_paths(exp_dir):
        meta = read_meta(f)
        if meta is None: bad.append(f); continue
        try:
            df = pd.read_csv(f)
            if not len(df): raise pd.errors.EmptyDataError(f)
            r = collapse_summary(df, meta)
        except READ_ERRORS: bad.append(f); continue
        r.update(tag=tag, seed=meta["seed"], fb_seed=meta["fb_seed"]); rows.append(r)
    if bad: print(f"  skipped {len(bad)} unreadable run(s), e.g. {os.path.basename(bad[0])}")
    return pd.DataFrame(rows) if rows else None

def summarize_and_write(exp_dir, window=P_WINDOW):
    """summarize_exp + the extra columns; writes summary_per_run.csv and a flat summary.csv. Returns (per, agg) or None.
    Falls back to summarize_readable when a run in the directory is half-written."""
    try:
        res = summarize_exp(exp_dir)
        per = None if res is None else res[0]
    except READ_ERRORS:
        per = None
    if per is None or not len(per): per = summarize_readable(exp_dir)
    if per is None or not len(per): return None
    per = enrich_per_run(exp_dir, per, window=window)
    per = per.sort_values(["tag", "seed", "fb_seed"]).reset_index(drop=True)
    agg = aggregate_tags(per)
    per.to_csv(os.path.join(exp_dir, "summary_per_run.csv"), index=False)
    agg.to_csv(os.path.join(exp_dir, "summary.csv"), index=False)
    return per, agg

# --------------------------------------------------------------- readers used by make_numbers.py and the figure scripts
def load_per_run(results_root, exp):
    """results/<exp>/summary_per_run.csv as a DataFrame, or None when the experiment has not been summarized."""
    p = os.path.join(results_root, exp, "summary_per_run.csv")
    if not os.path.exists(p): return None
    try: df = pd.read_csv(p)
    except (OSError, pd.errors.EmptyDataError): return None
    return df if len(df) else None

def tag_rows(per, tag):
    """Rows of one tag, or None when the tag is absent."""
    if per is None or "tag" not in per: return None
    sub = per[per.tag == tag]
    return sub if len(sub) else None

def col_stat(per, tag, col, stat="mean"):
    """mean / sd / n of one column for one tag; NaN when missing."""
    sub = tag_rows(per, tag)
    if sub is None or col not in sub.columns: return float("nan")
    v = pd.to_numeric(sub[col], errors="coerce").dropna()
    if len(v) == 0: return float("nan")
    if stat == "mean": return float(v.mean())
    if stat == "sd": return float(v.std(ddof=1)) if len(v) > 1 else 0.0
    if stat == "n": return float(len(v))
    if stat == "min": return float(v.min())
    if stat == "max": return float(v.max())
    raise ValueError(stat)

def n_layers_of(per, tag):
    """Number of hidden layers of a tag (from the per-run table)."""
    sub = tag_rows(per, tag)
    if sub is None or "n_layers" not in sub.columns: return 0
    return int(pd.to_numeric(sub.n_layers, errors="coerce").max())

READ_ERRORS = (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError)

def load_runs_safe(pattern):
    """load_runs, but a run whose csv or meta.json is unreadable (still being written) is skipped rather than raised."""
    out = []
    for f in sorted(glob.glob(pattern)):
        meta = read_meta(f)
        if meta is None: continue
        try: df = pd.read_csv(f)
        except READ_ERRORS: continue
        if len(df): out.append((df, meta))
    return out

def runs_of(exp_dir, tag):
    """[(df, meta)] for one tag, skipping runs that are still being written."""
    return load_runs_safe(os.path.join(exp_dir, tag, "seed*_fb*.csv"))

def mean_traj(exp_dir, tag, col):
    """Mean +- sd across runs of one column on the common step grid. Returns (steps, mean, sd, n) or None."""
    runs = runs_of(exp_dir, tag)
    series = [df.set_index("step")[col] for df, _ in runs if col in df.columns]
    if not series: return None
    M = pd.concat(series, axis=1).sort_index()
    ok = M.notna().sum(axis=1) > 0
    M = M[ok]
    if not len(M): return None
    return M.index.values.astype(float), M.mean(axis=1).values, M.std(axis=1, ddof=0).values, M.notna().sum(axis=1).values

def corr_p_cos(exp_dir, tag, layer=None):
    """Mean over runs of the Pearson r between participation and hidden cosine of one layer (default: deepest)."""
    rs = []
    for df, meta in runs_of(exp_dir, tag):
        l = layer or len(meta["widths"])
        if f"p_l{l}" not in df or f"cos_l{l}" not in df: continue
        a, b = df[f"p_l{l}"].values, df[f"cos_l{l}"].values
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 2: rs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")

def drift_spearman(exp_dir, tag, layer, step=None):
    """Spearman correlation between each unit's mean-preactivation drift and its fixed common-mode coefficient (B_l 1)_i.
    Uses the first snapshot and the snapshot nearest `step` (default: the last snapshot at or before step 120)."""
    vals = []
    for f in sorted(glob.glob(os.path.join(exp_dir, tag, "seed*_fb*.snapshots.npz"))):
        with np.load(f) as z:
            steps = sorted({int(k.split("/")[0][4:]) for k in z.files})
            later = [s for s in steps if s > 0 and (step is None or s <= step)]
            if not later or f"step{steps[0]}/mu_l{layer}" not in z.files: continue
            s1 = later[-1] if step is None else min(later, key=lambda s: abs(s - step))
            if f"step0/B1_l{layer}" not in z.files: continue
            dmu = z[f"step{s1}/mu_l{layer}"] - z[f"step0/mu_l{layer}"]; b1 = z[f"step0/B1_l{layer}"]
        if len(dmu) > 3: vals.append(float(_stats.spearmanr(dmu, b1).statistic))
    return float(np.mean(vals)) if vals else float("nan")

def drift_scatter(exp_dir, tag, layer, step=None):
    """(B_l 1)_i and the drift of mu_i for the first run of a tag, for the scatter panel of figure 4."""
    fs = sorted(glob.glob(os.path.join(exp_dir, tag, "seed*_fb*.snapshots.npz")))
    if not fs: return None
    with np.load(fs[0]) as z:
        steps = sorted({int(k.split("/")[0][4:]) for k in z.files})
        later = [s for s in steps if s > 0 and (step is None or s <= step)]
        if not later or f"step0/B1_l{layer}" not in z.files: return None
        s1 = later[-1] if step is None else min(later, key=lambda s: abs(s - step))
        return z[f"step0/B1_l{layer}"], z[f"step{s1}/mu_l{layer}"] - z[f"step0/mu_l{layer}"], s1

def fit_loglog(x, y):
    """Least-squares fit of log10 y on log10 x. Returns dict(slope, intercept, r, n, xmin, xmax); NaN when degenerate."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3: return dict(slope=float("nan"), intercept=float("nan"), r=float("nan"), n=int(m.sum()),
                                xmin=float("nan"), xmax=float("nan"))
    lx, ly = np.log10(x[m]), np.log10(y[m])
    if lx.std() == 0 or ly.std() == 0:                      # a single abscissa carries no slope
        return dict(slope=float("nan"), intercept=float("nan"), r=float("nan"), n=int(m.sum()),
                    xmin=float(x[m].min()), xmax=float(x[m].max()))
    res = _stats.linregress(lx, ly)
    return dict(slope=float(res.slope), intercept=float(res.intercept), r=float(res.rvalue), n=int(m.sum()),
                xmin=float(x[m].min()), xmax=float(x[m].max()))

def deepest(per, tag, stem):
    """Column name <stem>_l<L> for the deepest layer of a tag, e.g. deepest(per, 'base', 'maxcos')."""
    n = n_layers_of(per, tag)
    return f"{stem}_l{n}" if n else None

def fmt_mean_sd(mean, sd, digits=3):
    """'0.988+-0.002', or '--' when the mean is missing."""
    if mean is None or not np.isfinite(mean): return "--"
    if sd is None or not np.isfinite(sd): return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}+-{sd:.{digits}f}"

def per_layer_str(agg_row, stem, n_layers, digits=3, suffix="_mean", sdsuffix="_sd"):
    """'0.35+-0.01 / 0.74+-0.02 / 0.99+-0.00' across layers, for the markdown tables."""
    parts = []
    for l in range(1, max(n_layers, 0) + 1):
        m = agg_row.get(f"{stem}_l{l}{suffix}", float("nan")); s = agg_row.get(f"{stem}_l{l}{sdsuffix}", float("nan"))
        parts.append(fmt_mean_sd(m, s, digits))
    return " / ".join(parts) if parts else "--"

def experiment_dirs(results_root):
    """Experiment directories under a results root, skipping private ones (leading underscore)."""
    if not os.path.isdir(results_root): return []
    out = []
    for name in sorted(os.listdir(results_root)):
        d = os.path.join(results_root, name)
        if os.path.isdir(d) and not name.startswith("_") and not name.startswith("."): out.append(d)
    return out

def markdown_table(exp, per, agg, window=P_WINDOW):
    """One markdown table for an experiment: one row per tag with the numbers the paper quotes."""
    head = ["tag", "n", "peak cosine L1..LL", f"min p (t<={window})", "plateau on/exit/dur", "band acc",
            "kappa L1..LL", "acc@1000", "acc@1500", "acc@3000"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _, row in agg.iterrows():
        tag = row["tag"]; n = int(row["n_runs"]); nl = n_layers_of(per, tag)
        r = row.to_dict()
        onset, ex, dur = r.get("plateau_onset_mean"), r.get("plateau_exit_mean"), r.get("plateau_duration_mean")
        def i(v): return "--" if v is None or not np.isfinite(v) else f"{v:.0f}"
        lines.append("| " + " | ".join([
            tag, str(n),
            per_layer_str(r, "maxcos", nl),
            per_layer_str(r, f"minp_w{window}", nl),
            f"{i(onset)}/{i(ex)}/{i(dur)}",
            fmt_mean_sd(r.get("plateau_band_acc_mean"), r.get("plateau_band_acc_sd")),
            per_layer_str(r, "kappa", nl, digits=2),
            fmt_mean_sd(r.get("acc_1000_mean"), r.get("acc_1000_sd")),
            fmt_mean_sd(r.get("acc_1500_mean"), r.get("acc_1500_sd")),
            fmt_mean_sd(r.get("acc_3000_mean"), r.get("acc_3000_sd")),
        ]) + " |")
    return "\n".join(lines)

# The product law is a statement about the drive g||ebar_0||: it holds across sweeps that change the gain or the
# initial mean error (and across the learning rate, which cancels). Sweeps that change kappa through the other
# factors instead -- head learning rate, hidden learning rate, width, batch size, input preprocessing -- move off
# that line by construction and belong on the kappa axis, not the drive axis.
DRIVE_TAG_PATTERNS = (r"^g[\d.]+$", r"^q[\d.]+$", r"^C\d+$", r"^smimb", r"^lr")

def dfa_rows(per):
    """Uncentered DFA/FA runs of a summary table."""
    sub = per
    if "rule" in sub.columns: sub = sub[sub.rule.isin(["dfa", "fa"])]
    if "center" in sub.columns: sub = sub[sub.center == "none"]
    return sub

def on_drive_curve(tags):
    """Boolean mask: which tags belong to a sweep that varies only the drive."""
    import re as _re
    return np.asarray([any(_re.match(p, str(t)) for p in DRIVE_TAG_PATTERNS) for t in tags], bool)

def drive_sweep_rows(per):
    """The subset of a summary table on which the product law is claimed."""
    sub = dfa_rows(per)
    return sub[on_drive_curve(sub.tag)] if len(sub) else sub
