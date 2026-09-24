#!/usr/bin/env python3
"""Submission controls: all runs, explicit censoring, no outcome-based selection."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from cmc.analysis import plateau


def main():
    rows=[]
    for family in ("E13_adam","E14_controls","E15_cifar100","E16_batch","E17_grid"):
        for f in sorted((ROOT/"results"/family).glob("*/seed*_fb*.csv")):
            d=pd.read_csv(f);m=json.loads(f.with_suffix(".meta.json").read_text());L=len(m["widths"])
            p=plateau(d,m["prior_loss"],m["prior"])
            rows.append(dict(family=family,tag=m["tag"],seed=m["seed"],fb_seed=m["fb_seed"],
                             width=m["width"],depth=m["depth"],batch=m["batch"],steps=int(d.step.max()),
                             maxcos=float(d[f"cos_l{L}"].max()),minp=float(d[d.step<=300][f"p_l{L}"].min()),
                             mean_error_init=float(d.ebar_norm.iloc[0]),loss_final=float(d.probe_loss.iloc[-1]),
                             acc_final=float(d.probe_acc.iloc[-1]),onset=p["onset"] if p else np.nan,
                             exit=p["exit"] if p and p["exit"] else np.nan,
                             duration=p["duration"] if p and p["duration"] else np.nan,
                             has_plateau=bool(p),exit_censored=bool(p and not p["exit"]),
                             actual_step_cos=float(d.loc[(d.step>=100)&(d.step<=300),f"step_cos_l{L}"].mean()),
                             wall=m["wall"]))
    R=pd.DataFrame(rows);R.to_csv(ROOT/"results/revision_per_run.csv",index=False)
    summary={"n_runs":len(R)}
    groups={}
    for (family,tag),g in R.groupby(["family","tag"]):
        groups[f"{family}/{tag}"]={k:{"mean":float(g[k].mean()),"sd":float(g[k].std()),"n":int(g[k].notna().sum())}
                                  for k in ("maxcos","minp","mean_error_init","loss_final","acc_final","onset","exit","duration","actual_step_cos")}
        groups[f"{family}/{tag}"].update(n=len(g),n_plateaus=int(g.has_plateau.sum()),n_censored=int(g.exit_censored.sum()))
    summary["groups"]=groups
    def value(family,tag,key):return groups.get(f"{family}/{tag}",{}).get(key,{}).get("mean")
    flat={}
    for alias,tag in (("adam_fast","adam0.001_base"),("adam_slow","adam0.0001_base"),("sgd","sgd_base")):
        for metric in ("maxcos","minp","exit","duration","acc_final"):
            flat[f"{alias}_{metric}"]=value("E13_adam",tag,metric)
    if flat["adam_fast_duration"]:flat["adam_duration_ratio"]=flat["sgd_duration"]/flat["adam_fast_duration"]
    for tag in ("sigmoid_bce","softmax_ce"):
        for metric in ("maxcos","minp","mean_error_init","acc_final"):
            flat[f"cifar100_{tag}_{metric}"]=value("E15_cifar100",tag,metric)
    for tag in ("row_norm","sigmoid","sigmoid_center_e","sigmoid_bp"):
        for metric in ("maxcos","minp","acc_final"):
            flat[f"{tag}_{metric}"]=value("E14_controls",tag,metric)
    dest=ROOT/"results/revision_dynamics"
    if (dest/"drift.csv").exists():
        drift=pd.read_csv(dest/"drift.csv")
        for term in ("residual","covariance","upstream","finite_cross"):
            drift[term+"_ratio"]=drift[term+"_norm"]/drift.leading_norm
        drift.groupby(["config","step","layer"]).agg(["mean","std"]).to_csv(dest/"drift_summary.csv")
        base=drift[(drift.config=="base")&(drift.layer==3)]
        for time in (0,50,300):
            for term in ("residual","covariance","upstream"):
                flat[f"drift_{term}_{time}"]=float(base.loc[base.step==time,term+"_ratio"].mean())
        flat["drift_reconstruction_max"]=float(drift.reconstruction_error.max())
        residual=pd.read_csv(dest/"residual.csv")
        residual.groupby(["config","step","layer"]).ratio.agg(["mean","std"]).to_csv(dest/"residual_summary.csv")
        masks=pd.read_csv(dest/"masks.csv")
        for name in ("full","current","fixed120","random","complement120"):
            flat["masked_"+name+"_cos"]=float(masks.loc[(masks.layer==3)&(masks.step==3000)&(masks.selection==name),"cosine"].mean())
    if (dest/"batch_noise.csv").exists():
        noise=pd.read_csv(dest/"batch_noise.csv")
        slopes=[]
        for (dataset,seed),g in noise.groupby(["dataset","seed"]):
            slopes.append(dict(dataset=dataset,seed=int(seed),slope=float(np.polyfit(np.log(g.batch),np.log(g.centered_rms),1)[0])))
        summary["noise_slopes"]=slopes
        flat["noise_slope_min"]=min(d["slope"] for d in slopes);flat["noise_slope_max"]=max(d["slope"] for d in slopes)
        flat["noise_prediction_relative_max"]=float((noise.centered_rms/noise.predicted_centered_rms-1).abs().max())
    summary["numbers"]=flat
    # JSON null explicitly denotes a measurement not yet available.
    text=json.dumps(summary,indent=1,allow_nan=True).replace("NaN","null")
    (ROOT/"results/revision_summary.json").write_text(text+"\n")
    print(R.groupby(["family","tag"])[["maxcos","minp","acc_final","onset","exit"]].mean().round(3).to_string())
    print(json.dumps(flat,indent=1))


if __name__=="__main__":main()
