#!/usr/bin/env python3
"""Separate population mean error from batch sampling noise at initialization.

On a frozen network, independent with-replacement batches obey
E||mean(e_batch)||^2 = ||mean(e_population)||^2 + tr Cov(e_population)/n.
The inverse-square-root law applies to the centered fluctuation, not to the
raw norm when a nonzero mean remains, nor directly to collapse depth.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from cmc import data as D
from cmc.models import MLP
from cmc.rules import head_forward
from cmc.run import Config


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--device",default="cuda")
    args=ap.parse_args();torch.set_num_threads(1);rows=[];populations=[]
    for dataset,head,cal in (("mnist","softmax_ce",0),("cifar10","sigmoid_bce",1)):
        ds=D.load(dataset,flatten=True);C=ds["C"]
        for seed in range(3):
            cfg=Config(dataset=dataset,head=head,calibrate_head=cal,seed=seed,fb_seed=seed)
            torch.manual_seed(seed);model=MLP(ds["Xtr"].shape[1],[300]*3,C).to(args.device)
            with torch.no_grad():
                if cal:
                    sampler=D.Sampler(ds["ytr"],1024,seed=seed+4242);idx=sampler.draw()
                    zbar=model.forward_blocks(ds["Xtr"][idx].to(args.device),detach=False)[0].mean(0)
                    pi=torch.tensor(sampler.prior(C),dtype=zbar.dtype,device=args.device).clamp(1e-6,1-1e-6)
                    model.out.bias.add_(torch.log(pi/(1-pi))-zbar)
                errors=[]
                for start in range(0,len(ds["Xtr"]),2048):
                    x=ds["Xtr"][start:start+2048].to(args.device);y=ds["ytr"][start:start+2048].to(args.device)
                    z=model.forward_blocks(x,detach=False)[0];errors.append(head_forward(z,y,cfg,C)[2].cpu().double())
            error=torch.cat(errors);mean=error.mean(0);variance=float((error-mean).square().sum(1).mean())
            populations.append(dict(dataset=dataset,seed=seed,mean_norm=float(mean.norm()),trace_cov=variance,n_population=len(error)))
            gen=torch.Generator().manual_seed(17000+seed)
            for batch in (16,32,128,512,2048):
                means=error[torch.randint(len(error),(512,batch),generator=gen)].mean(1)
                centered=float((means-mean).square().sum(1).mean())
                raw=float(means.square().sum(1).mean())
                rows.append(dict(dataset=dataset,seed=seed,batch=batch,replicates=512,
                                 centered_rms=math.sqrt(centered),predicted_centered_rms=math.sqrt(variance/batch),
                                 raw_rms=math.sqrt(raw),predicted_raw_rms=math.sqrt(float(mean.square().sum())+variance/batch),
                                 population_mean_norm=float(mean.norm())))
            print(f"Frozen-state noise: {dataset}, seed {seed}, mean norm {mean.norm():.5f}",flush=True)
    dest=ROOT/"results/revision_dynamics";dest.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/"batch_noise.csv",index=False)
    (dest/"batch_noise_population.json").write_text(json.dumps(populations,indent=1)+"\n")


if __name__=="__main__":main()
