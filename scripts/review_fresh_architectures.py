#!/usr/bin/env python3
"""Predict, then train new width/depth settings with fresh paired seeds."""
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from cmc import data as D
from cmc.run import Config,main as train
from cmc.rules import prior_loss
from cmc.reduced import init_from_meta,simulate
from review_model_ablation import cascades,model_detection


def main():
    torch.set_num_threads(2)
    ds=D.load('mnist');prior=D.Sampler(ds['ytr'],128,0).prior(10)
    dest=ROOT/'results/E18_fresh';dest.mkdir(exist_ok=True)
    configs=[Config(exp='E18_fresh',tag=f'w450_d{depth}',width=450,depth=depth,
                    seed=seed,fb_seed=seed,device='cuda',threads=2,anatomy=1,save_snapshots=0)
             for depth in (2,4,6) for seed in (10,11,12)]
    manifest=dest/'protocol.json'
    if not manifest.exists():manifest.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='New architecture and seed combinations; all four model predictions saved before each training run.',
        configs=[asdict(c) for c in configs]),indent=2)+'\n')
    for cfg in configs:
        path=dest/cfg.tag;path.mkdir(exist_ok=True);stem=f'seed{cfg.seed}_fb{cfg.fb_seed}'
        mp=path/(stem+'.meta.json')
        if mp.exists() and json.loads(mp.read_text()).get('final_step',0)>=cfg.steps:continue
        meta=dict(asdict(cfg),C=10,Cout=10,widths=[cfg.width]*cfg.depth,prior=prior.tolist(),prior_loss=prior_loss(prior,cfg,10))
        init,rc,B=init_from_meta(meta);rc.escape=False;rc.log_every=1
        prediction=cascades(simulate(init,rc,B),init,rc)
        prediction[prediction.step%10==0].to_csv(path/(stem+'.predictions.csv'),index=False)
        (path/(stem+'.prediction_record.json')).write_text(json.dumps(dict(saved_utc=datetime.now(timezone.utc).isoformat(),
            detectors={v:model_detection(g,meta['prior_loss'],rc) for v,g in prediction.groupby('variant')}),indent=2)+'\n')
        print('Predictions saved:',cfg.tag,stem,flush=True)
        train(cfg)
        print('Training complete:',cfg.tag,stem,flush=True)


if __name__=='__main__':main()
