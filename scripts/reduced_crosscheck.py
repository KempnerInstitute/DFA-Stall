#!/usr/bin/env python3
"""Audit model predictions with one vote per physical condition and seed.

The primary comparison retains the original seed-0 integrations and their
original horizons. Repeated sweep labels are listed explicitly. Censored
model exits remain in the coverage table and are excluded only from the
numerical exit-error statistics, whose denominator is reported separately.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from cmc.analysis import plateau
from cmc.conditions import condition_key


def collect(paths):
    seen={};rows=[];timing=[];aliases=[]
    for rf in sorted(paths):
        stem=rf.name.removeprefix("reduced_").removesuffix(".csv")
        mf=rf.parent/(stem+".csv");mp=rf.parent/(stem+".meta.json")
        if not mf.exists() or not mp.exists():continue
        meta=json.loads(mp.read_text());key=condition_key(meta,include_seeds=True)
        label=str(rf.parent.relative_to(ROOT/"results"))+"/"+stem
        if key in seen:
            aliases.append(dict(alias=label,canonical=seen[key]));continue
        seen[key]=label;R=pd.read_csv(rf);M=pd.read_csv(mf)
        exp,tag=rf.parent.parent.name,rf.parent.name
        om=float(R.onset_pred.iloc[0]);em=float(R.exit_pred.iloc[0]);horizon=int(R.step.max())
        base=dict(exp=exp,tag=tag,seed=meta["seed"],fb_seed=meta["fb_seed"],model_horizon=horizon,
                  onset_model=om,exit_model=em)
        for l in range(1,len(meta["widths"])+1):
            if f"p_l{l}" not in R or f"p_l{l}" not in M:continue
            rows.append(dict(**base,layer=l,p_model=float(R[R.step<=300][f"p_l{l}"].min()),
                             p_meas=float(M[M.step<=300][f"p_l{l}"].min()),
                             cos_model=float(R[f"cos_l{l}"].max()),cos_meas=float(M[f"cos_l{l}"].max())))
        pl=plateau(M,meta["prior_loss"],meta["prior"])
        model_onset=bool(np.isfinite(om) and om>=0)
        timing.append(dict(**base,network_horizon=int(M.step.max()),
                           onset_meas=pl["onset"] if pl else np.nan,
                           exit_meas=pl["exit"] if pl and pl["exit"] else np.nan,
                           model_no_plateau=not model_onset,
                           model_exit_censored=model_onset and not (np.isfinite(em) and em>0)))
    return pd.DataFrame(rows),pd.DataFrame(timing),aliases


def summarize(X,T,aliases):
    ok=X.dropna(subset=["p_model","p_meas"])
    conditions=ok[["exp","tag"]].drop_duplicates()
    PE=T[T.onset_meas.notna() & T.exit_meas.notna() & (T.onset_model>=0)]
    valid=PE[PE.exit_model>0]
    corr=lambda x,y:float(np.corrcoef(x,y)[0,1]) if len(x)>2 else None
    relative=lambda x,y:float(np.median(np.abs(x-y)/y)) if len(x) else None
    out=dict(n_points=len(ok),n_conditions=len(conditions),n_trajectories=len(T),n_aliases=len(aliases),
             r_p=corr(ok.p_model,ok.p_meas),mae_p=float(np.abs(ok.p_model-ok.p_meas).mean()),
             bias_p_by_layer={int(l):float((g.p_model-g.p_meas).mean()) for l,g in ok.groupby("layer")},
             r_cos=corr(ok.cos_model,ok.cos_meas),mae_cos=float(np.abs(ok.cos_model-ok.cos_meas).mean()),
             onset_exit_n=len(PE),exit_valid_n=len(valid),exit_censored_n=int(PE.model_exit_censored.sum()),
             n_network_only_plateau=int((T.onset_meas.notna() & (T.onset_model<0)).sum()),
             n_model_only_plateau=int((T.onset_meas.isna() & (T.onset_model>=0)).sum()),
             n_neither_plateau=int((T.onset_meas.isna() & (T.onset_model<0)).sum()),
             n_both_plateau=int((T.onset_meas.notna() & (T.onset_model>=0)).sum()),
             exit_r=corr(valid.exit_model,valid.exit_meas),onset_r=corr(PE.onset_model,PE.onset_meas),
             exit_rel_err_median=relative(valid.exit_model,valid.exit_meas),
             onset_rel_err_median=relative(PE.loc[PE.onset_meas>0,"onset_model"],PE.loc[PE.onset_meas>0,"onset_meas"]),
             onset_relative_valid_n=int((PE.onset_meas>0).sum()),aliases=aliases)
    return out,PE


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-seeds",action="store_true");ap.add_argument("--grid",action="store_true")
    args=ap.parse_args()
    pattern="E17_grid/*/reduced_seed*_fb*.csv" if args.grid else "*/*/reduced_seed*_fb*.csv" if args.all_seeds else "*/*/reduced_seed0_fb0.csv"
    paths=list((ROOT/"results").glob(pattern))
    if not args.grid:paths=[p for p in paths if not p.parent.parent.name.startswith(("E13_","E14_","E15_","E16_","E17_"))]
    X,T,aliases=collect(paths);out,PE=summarize(X,T,aliases)
    name="reduced_grid" if args.grid else "reduced_crossseed" if args.all_seeds else "reduced_crosscheck"
    dest=ROOT/"results"/name
    X.to_csv(str(dest)+".csv",index=False);T.to_csv(str(dest)+"_coverage.csv",index=False)
    PE.to_csv(str(dest)+"_plateaus.csv",index=False)
    Path(str(dest)+".json").write_text(json.dumps(out,indent=1)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="aliases"},indent=1))


if __name__=="__main__":main()
