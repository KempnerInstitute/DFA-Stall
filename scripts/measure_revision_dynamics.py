#!/usr/bin/env python3
"""Three-seed drift budgets, residuals and norm-matched gate-set diagnostics."""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cmc import data as D
from cmc.models import MLP
from cmc.rules import make_feedback, step
from cmc.run import Config,make_optimizer
from cmc.drift import drift_budget,gate_sets,masked_updates
from measure_residual import measure,tensor_hash


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--device",default="cuda")
    args=ap.parse_args();torch.set_num_threads(2)
    dest=ROOT/"results/revision_dynamics";dest.mkdir(parents=True,exist_ok=True)
    D.load=lru_cache(maxsize=1)(D.load)
    configs=dict(base={},relu={"act":"relu"},cifar_mlp={"dataset":"cifar10"})
    times=(0,10,25,50,100,120,200,300,500,1000,1500,3000)
    for name,changes in configs.items():
        for seed in range(3):
            path=dest/f"{name}_seed{seed}.json"
            if path.exists():continue
            cfg=Config(**changes,seed=seed,fb_seed=seed,device=args.device,threads=2)
            ds=D.load(cfg.dataset,preprocess=cfg.preprocess,flatten=True);C=ds["C"]
            torch.manual_seed(seed);np.random.seed(seed)
            model=MLP(ds["Xtr"].shape[1],[cfg.width]*cfg.depth,C,act=cfg.act).to(args.device)
            fb=make_feedback(model,C,cfg,args.device);opt=make_optimizer(model,cfg)
            sampler=D.Sampler(ds["ytr"],cfg.batch,seed=seed)
            ps=D.Sampler(ds["yval"],cfg.probe_n,seed=seed+999);idx=ps.draw()
            xp,yp=ds["Xval"][idx].to(args.device),ds["yval"][idx].to(args.device)
            drift,residual,masks=[],[],[];state={};fixed=None
            rng=torch.Generator().manual_seed(8400+seed)
            random=[torch.randperm(cfg.width,generator=rng)[:int(.2*cfg.width)].to(args.device) for _ in range(cfg.depth)]
            final=3000 if name=="base" else 300
            for it in range(final+1):
                if it:
                    j=sampler.draw();step(model,ds["Xtr"][j].to(args.device),ds["ytr"][j].to(args.device),C,cfg,fb,opt,state=state)
                if it not in times:continue
                for row in drift_budget(model,xp,yp,C,cfg,fb):drift.append(dict(step=it,**row))
                for row in measure(model,xp,yp,cfg,fb,C):residual.append(dict(step=it,**row))
                if name=="base" and it==120:fixed=gate_sets(model,xp)
                if fixed is not None:
                    for row in masked_updates(model,xp,yp,C,cfg,fb,fixed,random):masks.append(dict(step=it,**row))
            record=dict(config=asdict(cfg),sample_steps=[s for s in times if s<=final],
                        probe_indices_sha256=tensor_hash(idx),probe_inputs_sha256=tensor_hash(xp),
                        feedback_sha256=[tensor_hash(b) for b in fb["B"]],
                        drift=drift,residual=residual,masks=masks,
                        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [Path(__file__).resolve(),ROOT/"src/cmc/drift.py",ROOT/"src/cmc/rules.py"]})
            path.write_text(json.dumps(record,indent=1)+"\n")
            print(f"Completed {name} seed {seed}: drift {len(drift)}, masked directions {len(masks)}",flush=True)
    for key in ("drift","residual","masks"):
        frames=[]
        for f in sorted(dest.glob("*_seed*.json")):
            d=json.loads(f.read_text());frame=pd.DataFrame(d[key]);frame["config"]=f.stem.split("_seed")[0];frame["seed"]=d["config"]["seed"];frames.append(frame)
        pd.concat(frames,ignore_index=True).to_csv(dest/f"{key}.csv",index=False)


if __name__=="__main__":main()
