#!/usr/bin/env python3
"""Validate the reduced model (cmc.reduced) against measured runs, per layer and per run.

Reads regenerated trajectories in results/E1_headline/base and optional smoke runs in results/smoke/base.
Run the supplied training configurations first; the archive includes the compact reference results,
not these full trajectories. The optional polling arguments can wait for a local sweep to finish.

Reported per layer: minimum participation (model vs measured, windowed to step <= 300 as the protocol requires),
peak hidden cosine, and per run the plateau onset and exit with the model's error. The CSV also carries the layer-1
dose three ways: the model's closed form (kappa_l1_model), the model's own integrated dose (kappa_l1_dose) and the
value the run itself logged up to the model's last step (kappa_l1_meas, a minibatch estimate and so biased upward).

Usage: PYTHONPATH=src python scripts/validate_reduced.py [--poll_minutes 30] [--figures]
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from cmc.analysis import plateau
from cmc.reduced import compare_figure, from_run

WINDOW = 300

def measured_summary(df, Lprior, prior, nlayers, pcol="p_l{}", ccol="cos_l{}", losscol="probe_loss", acccol="probe_acc"):
    """Windowed participation minima, peak cosines and the plateau onset/exit of one measured trajectory."""
    sub = df[df.step <= WINDOW]
    out = {f"minp_l{l}": float(sub[pcol.format(l)].min()) for l in range(1, nlayers + 1) if pcol.format(l) in df}
    out.update({f"maxcos_l{l}": float(df[ccol.format(l)].max()) for l in range(1, nlayers + 1) if ccol.format(l) in df})
    pl = plateau(df.rename(columns={losscol: "probe_loss", acccol: "probe_acc"}), Lprior, prior)
    out["onset"] = pl["onset"] if pl else None; out["exit"] = pl["exit"] if pl else None
    return out

def model_summary(df, nlayers):
    """The same quantities from a reduced-model trajectory."""
    sub = df[df.step <= WINDOW]
    out = {f"minp_l{l}": float(sub[f"p_l{l}"].min()) for l in range(1, nlayers + 1)}
    out.update({f"maxcos_l{l}": float(df[f"cos_l{l}"].max()) for l in range(1, nlayers + 1)})
    out["onset"] = df.attrs["onset"]; out["exit"] = df.attrs["exit"]
    return out

def run_meta_targets(pattern, steps=0, figures=False):
    """Validate every run matching a results/<exp>/<tag>/seed*_fb*.csv pattern that has a meta.json."""
    rows = []
    for csv in sorted(glob.glob(pattern)):
        meta_path = csv.replace(".csv", ".meta.json")
        if not os.path.exists(meta_path) or os.path.basename(csv).startswith("reduced_"): continue
        meta = json.load(open(meta_path))
        meas = pd.read_csv(csv); nl = len(meta["widths"])
        mdf, init, rc = from_run(meta_path, steps=steps)
        m = model_summary(mdf, nl); q = measured_summary(meas, meta["prior_loss"], meta["prior"], nl)
        row = dict(target=os.path.relpath(os.path.dirname(csv), ROOT), run=os.path.basename(csv)[:-4])
        for l in range(1, nl + 1):
            row[f"minp_l{l}_model"] = m[f"minp_l{l}"]; row[f"minp_l{l}_meas"] = q.get(f"minp_l{l}", np.nan)
            row[f"maxcos_l{l}_model"] = m[f"maxcos_l{l}"]; row[f"maxcos_l{l}_meas"] = q.get(f"maxcos_l{l}", np.nan)
        last = int(mdf.step.max()); msub = meas[meas.step <= last]
        row.update(onset_model=m["onset"], onset_meas=q["onset"], exit_model=m["exit"], exit_meas=q["exit"],
                   kappa_l1_model=mdf.attrs["kappa_cf"][0], kappa_l1_dose=float(mdf["kappa_l1"].iloc[-1]),
                   kappa_l1_meas=float(msub["kappa_l1"].iloc[-1]) if ("kappa_l1" in meas and len(msub)) else np.nan)
        rows.append(row)
        if figures:
            stem = os.path.basename(csv)[:-4]
            compare_figure(mdf, meas, mdf.attrs["prior_loss"], os.path.join(os.path.dirname(csv), f"reduced_{stem}.png"),
                           title=f"Reduced model vs measured run: {os.path.relpath(csv, ROOT)}")
    return rows

def report(rows, nlayers=3):
    """Print the per-target table of model vs measured values and return the aggregated frame."""
    df = pd.DataFrame(rows)
    if df.empty: return df
    for tgt, sub in df.groupby("target"):
        print(f"\n=== {tgt}   ({len(sub)} runs)")
        print(f"{'quantity':>16} {'model':>16} {'measured':>16} {'error':>12}")
        for l in range(1, nlayers + 1):
            for q, fmt in (("minp", "{:.3f}"), ("maxcos", "{:.3f}")):
                a, b = sub[f"{q}_l{l}_model"], sub[f"{q}_l{l}_meas"]
                if a.isna().all() or b.isna().all(): continue
                print(f"{q+'_l'+str(l):>16} {fmt.format(a.mean())+' +- '+fmt.format(a.std(ddof=0)):>16} "
                      f"{fmt.format(b.mean())+' +- '+fmt.format(b.std(ddof=0)):>16} {a.mean()-b.mean():>+12.3f}")
        for q in ("onset", "exit"):
            a = pd.to_numeric(sub[f"{q}_model"], errors="coerce"); b = pd.to_numeric(sub[f"{q}_meas"], errors="coerce")
            if a.isna().all() or b.isna().all():
                print(f"{q:>16} {'none' if a.isna().all() else f'{a.mean():.0f}':>16} "
                      f"{'none' if b.isna().all() else f'{b.mean():.0f}':>16} {'-':>12}")
                continue
            rel = (a.mean() - b.mean()) / b.mean() * 100
            mae = float((a - b).abs().mean())
            print(f"{q:>16} {f'{a.mean():.0f} +- {a.std(ddof=0):.0f}':>16} {f'{b.mean():.0f} +- {b.std(ddof=0):.0f}':>16}"
                  f" {rel:>+11.1f}%   per-run mean |error| {mae:.0f} steps ({mae / b.mean() * 100:.0f}%)")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll_minutes", type=float, default=0.0, help="how long to wait for results/E1_headline/base")
    ap.add_argument("--poll_every", type=float, default=60.0)
    ap.add_argument("--figures", action="store_true", help="also write reduced_<stem>.png next to each run")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "reduced_validation.csv"))
    args = ap.parse_args()
    rows = []
    smoke = os.path.join(ROOT, "results", "smoke", "base", "seed*_fb*.csv")
    if glob.glob(smoke):
        print("validating on results/smoke/base"); rows += run_meta_targets(smoke, figures=args.figures)
    e1 = os.path.join(ROOT, "results", "E1_headline", "base", "seed*_fb*.csv")
    deadline = time.time() + args.poll_minutes * 60
    while not glob.glob(e1) and time.time() < deadline:
        print(f"waiting for {os.path.relpath(e1, ROOT)} ...", flush=True); time.sleep(args.poll_every)
    if glob.glob(e1):
        print(f"validating on results/E1_headline/base ({len(glob.glob(e1))} runs)")
        rows += run_meta_targets(e1, figures=args.figures)
    if not rows:
        ap.error("No training trajectories found. Regenerate the E1 baseline or a smoke run first; see REPRODUCING.md.")
    df = report(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
