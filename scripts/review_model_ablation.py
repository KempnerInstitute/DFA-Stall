#!/usr/bin/env python3
"""Compare four recovery closures on identical, per-update collapse trajectories.

All coefficients use initialization only. Every variant uses the same horizon
and loss-based plateau detector. The width-depth grid is reported separately.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cmc.analysis import plateau
from cmc.reduced import init_from_meta,simulate


def initialize():
    import torch
    from cmc import data as D
    torch.set_num_threads(1);D.load=lru_cache(maxsize=2)(D.load)


def cascades(collapse,init,cfg):
    L=len(init.widths);width=np.asarray(init.widths)
    u=collapse[[f'chi_l{l}' for l in range(1,L+1)]].to_numpy()
    p=collapse[[f'p_l{l}' for l in range(1,L+1)]].to_numpy()
    source=np.sqrt(max(init.Lam0,0));r=np.asarray(init.r)
    initial_energy=np.r_[init.Lam0,init.lam0]
    # This variant changes only transmission coefficients so the initial hidden
    # covariance amplitudes agree with measured initialization, without fitting.
    measured_r=initial_energy[1:]/np.maximum(u[0]*initial_energy[:-1],1e-30)
    output=[]
    for name in ['full','no_local','no_participation','measured_init']:
        gains=measured_r if name=='measured_init' else r
        J=np.zeros(L);excess=0.;rows=[]
        for idx,row in enumerate(collapse.itertuples()):
            A=np.zeros(L+1);A[0]=source
            for l in range(L):A[l+1]=np.sqrt(gains[l]*u[idx,l])*A[l]+J[l]
            loss=row.loss_cm-excess
            entry=dict(step=row.step,variant=name,loss_model=loss,
                       **{f'Lambda_l{l+1}':A[l+1]**2 for l in range(L)})
            rows.append(entry)
            if name!='no_local':
                rate=u[idx] if name=='no_participation' else u[idx]/np.sqrt(np.maximum(p[idx],1e-12))
                J+=cfg.eta_hid*cfg.g*np.sqrt(width/init.C)*rate*A[:-1]**2
            excess+=cfg.eta_out*A[-1]**2
        output.extend(rows)
    return pd.DataFrame(output)


def model_detection(frame,prior,cfg):
    """The original model's first persistent loss band and subsequent exit."""
    onset=None;start=None
    for row in frame.itertuples():
        if onset is None:
            if abs(row.loss_model-prior)/prior<cfg.onset_tol:
                if start is None:start=row.step
                if row.step-start>=cfg.onset_min_len:onset=start
            else:start=None
        if onset is not None and row.loss_model<cfg.exit_frac*prior:
            return dict(onset=onset,exit=row.step)
    return dict(onset=onset,exit=None)


def run(task):
    family,exp,tag,seed,fbseed=task
    dest=ROOT/'results/review_20260919/model';dest.mkdir(parents=True,exist_ok=True)
    name=f'{family}_{exp}_{tag}_seed{seed}_fb{fbseed}'
    file=dest/(name+'.json')
    if file.exists() and json.loads(file.read_text()).get('version')==2:return 'exists '+name
    mp=ROOT/'results'/exp/tag/f'seed{seed}_fb{fbseed}.meta.json'
    meta=json.loads(mp.read_text());network=pd.read_csv(mp.with_name(mp.name.replace('.meta.json','.csv')))
    init,cfg,B=init_from_meta(meta);cfg.escape=False;cfg.log_every=1
    cfg.steps=int(meta.get('final_step',meta['steps']))
    collapse=simulate(init,cfg,B);variants=cascades(collapse,init,cfg)
    true=plateau(network,meta['prior_loss'],meta['prior'])
    records=[];trajectories=[]
    for variant,v in variants.groupby('variant',sort=False):
        sampled=v[v.step%10==0].copy();sampled['probe_loss']=sampled.loss_model
        pred=model_detection(v,meta['prior_loss'],cfg)
        rec=dict(family=family,exp=exp,tag=tag,seed=seed,fb_seed=fbseed,variant=variant,
                 horizon=cfg.steps,onset_network=true['onset'] if true else None,
                 exit_network=true['exit'] if true else None,
                 onset_model=pred['onset'] if pred else None,exit_model=pred['exit'] if pred else None)
        # Compare covariance trajectories only until measured plateau exit (or
        # update 1000 if no exit), avoiding the extension's invalid late losses.
        stop=min(true['exit'] if true and true['exit'] else 1000,1000)
        for l in range(1,len(init.widths)+1):
            if f'lam_l{l}' not in network:continue
            pair=network.loc[network.step<=stop,['step',f'lam_l{l}']].merge(sampled[['step',f'Lambda_l{l}']],on='step').dropna()
            a=pair[f'Lambda_l{l}'].to_numpy();b=pair[f'lam_l{l}'].to_numpy()
            rec[f'cov_log_mae_l{l}']=float(np.mean(abs(np.log10(np.maximum(a,1e-30))-np.log10(np.maximum(b,1e-30)))))
        records.append(rec)
        if exp=='E1_headline' and tag=='base':
            trajectories.extend(sampled[sampled.step<=1000].assign(seed=seed).drop(columns='probe_loss').to_dict('records'))
    # Window sensitivity: compare both minima over the same per-run horizon.
    windows=[]
    for l in range(1,len(init.widths)+1):
        for window in [300,min(cfg.steps,max(300,int(300*1e-3/(cfg.eta_hid*cfg.g))))]:
            windows.append(dict(layer=l,window=window,p_model=float(collapse.loc[collapse.step<=window,f'p_l{l}'].min()),
                                p_network=float(network.loc[network.step<=window,f'p_l{l}'].min())))
    file.write_text(json.dumps(dict(version=2,records=records,baseline_trajectories=trajectories,windows=windows),indent=1)+'\n')
    return 'completed '+name


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int);args=ap.parse_args()
    tasks=[]
    for family,source in [('sweeps','reduced_crossseed_coverage.csv'),('grid','reduced_grid_coverage.csv')]:
        for r in pd.read_csv(ROOT/'results'/source).itertuples():tasks.append((family,r.exp,r.tag,int(r.seed),int(r.fb_seed)))
    for p in sorted((ROOT/'results/E18_fresh').glob('*/seed*_fb*.meta.json')):
        m=json.loads(p.read_text());tasks.append(('fresh',m['exp'],m['tag'],m['seed'],m['fb_seed']))
    if args.limit:tasks=tasks[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        for n,message in enumerate(pool.map(run,tasks),1):print(f'{n}/{len(tasks)} {message}',flush=True)
    dest=ROOT/'results/review_20260919';allrecords=[];trajectories=[]
    for p in sorted((dest/'model').glob('*.json')):
        d=json.loads(p.read_text());allrecords.extend(d['records']);trajectories.extend(d['baseline_trajectories'])
    pd.DataFrame(allrecords).to_csv(dest/'recovery_ablations.csv',index=False)
    pd.DataFrame(trajectories).to_csv(dest/'recovery_trajectories.csv',index=False)


if __name__=='__main__':main()
