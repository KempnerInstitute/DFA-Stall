#!/usr/bin/env python3
"""Replicate existing model tests across seeds or integrate the width-depth grid."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import sys
import time
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))


def initialize():
    import torch
    from cmc import data as D
    torch.set_num_threads(1);D.load=lru_cache(maxsize=1)(D.load)


def integrate(task):
    from cmc.reduced import from_run
    meta,horizon,*replace=task;meta=Path(meta)
    out=meta.with_name("reduced_"+meta.name.replace(".meta.json",".csv"))
    if out.exists() and not (replace and replace[0]):return "exists "+str(out.relative_to(ROOT))
    df,init,cfg=from_run(str(meta),steps=horizon)
    df.to_csv(out,index=False)
    return "wrote "+str(out.relative_to(ROOT))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--grid",action="store_true")
    ap.add_argument("--follow-grid",action="store_true",help="process each of the 48 grid runs when its training finishes")
    ap.add_argument("--refresh-replicas",action="store_true",help="recompute seed-1/2 model outputs at the configured network horizon")
    ap.add_argument("--workers",type=int,default=2);args=ap.parse_args();tasks=[]
    if args.follow_grid:
        done=set();start=time.time()
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
            while len(done)<48:
                ready=[p for p in sorted((ROOT/"results/E17_grid").glob("*/seed*_fb*.meta.json")) if p not in done]
                for p,message in zip(ready,pool.map(integrate,[(str(p),3000) for p in ready])):
                    done.add(p);print(f"[{len(done)}/48] {message}",flush=True)
                if time.time()-start>7200:raise RuntimeError("grid training did not finish within two hours")
                if len(done)<48:time.sleep(10)
        return
    if args.grid:
        tasks=[(str(p),3000) for p in sorted((ROOT/"results/E17_grid").glob("*/seed*_fb*.meta.json"))]
    else:
        # The shipped coverage table identifies the tested conditions even in a
        # fresh checkout with no reduced-model CSVs yet.
        reference=ROOT/"results/reduced_crossseed_coverage.csv"
        if not reference.exists():
            raise FileNotFoundError("The primary comparison requires its saved coverage manifest")
        for row in pd.read_csv(reference).itertuples():
            p=ROOT/"results"/row.exp/row.tag/f"seed{row.seed}_fb{row.fb_seed}.meta.json"
            if not p.exists():
                raise FileNotFoundError(f"Regenerate the original training run first: {p}")
            m=json.loads(p.read_text());horizon=int(m.get("final_step",m["steps"]))
            tasks.append((str(p),horizon,args.refresh_replicas and row.seed in (1,2)))
    print(f"Integrations: {len(tasks)}",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        for message in pool.map(integrate,tasks):print(message,flush=True)


if __name__=="__main__":main()
