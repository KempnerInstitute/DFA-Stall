#!/usr/bin/env python3
"""The original DFA protocol of Nokland (2016) on MNIST, resumable, one GPU process.

His MNIST experiments used tanh hidden units, logistic outputs with binary cross-entropy, inputs scaled to [0, 1],
RMSprop without momentum or weight decay, and minibatches of 64. DFA and FA started from zero weights and biases with
feedback drawn from U(-1/sqrt(fanout), 1/sqrt(fanout)); BP started from U(-1/sqrt(fanin), 1/sqrt(fanin)). His learning
rate is not reported, so two common RMSprop rates are tested. A third arm keeps this paper's Xavier initialization and
unit-variance Gaussian feedback under the same optimizer, to separate initialization from optimizer. Architecture is his
3x800 tanh network. All arms use paired initialization/feedback seeds 0, 1, 2 and 6,000 updates (the example count of
3,000 updates at batch 128). No early stopping or selection by outcome.
"""
import argparse
from dataclasses import asdict, fields
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmc import data as D
from cmc.run import Config, main as train

EXP = "E19_nokland"
ARMS = {
    "dfa_zero": dict(rule="dfa", init="zero", fb_dist="uniform_fanout"),        # Nokland's DFA protocol
    "bp_fanin": dict(rule="bp", init="uniform_fanin"),                          # his BP protocol
    "dfa_xavier": dict(rule="dfa"),                                             # this paper's DFA initialization
}


def jobs(device="cuda"):
    common = dict(exp=EXP, width=800, depth=3, act="tanh", head="sigmoid_bce", preprocess="div255",
                  optimizer="rmsprop", batch=64, steps=6000, device=device, threads=1,
                  save_snapshots=0, measure_updates=1)
    for lr in (1e-3, 1e-4):
        for arm, changes in ARMS.items():
            for seed in range(3):
                yield Config(tag=f"{arm}_lr{lr:g}", lr=lr, seed=seed, fb_seed=seed, **changes, **common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    configs = list(jobs(args.device))
    manifest = ROOT / "configs/nokland_20260923.json"
    manifest.write_text(json.dumps([asdict(c) for c in configs], indent=1) + "\n")
    print(f"Planned {len(configs)} runs. Manifest: {manifest}", flush=True)
    if args.plan_only:
        return
    D.load = lru_cache(maxsize=1)(D.load)   # deterministic and independent of the run seed
    defaults = {f.name: f.default for f in fields(Config)}
    start = time.time()
    for i, cfg in enumerate(configs):
        meta = ROOT / f"results/{cfg.exp}/{cfg.tag}/seed{cfg.seed}_fb{cfg.fb_seed}.meta.json"
        if meta.exists():
            old = json.loads(meta.read_text())
            if old.get("final_step", 0) >= cfg.steps and all(old.get(k, defaults[k]) == v for k, v in asdict(cfg).items()):
                print(f"skip complete {cfg.tag}/seed{cfg.seed}", flush=True)
                continue
            raise RuntimeError(f"Refusing to overwrite a different or incomplete run: {meta}")
        print(f"[{i + 1}/{len(configs)}] {cfg.tag} seed={cfg.seed}, elapsed={time.time() - start:.0f}s", flush=True)
        train(cfg)


if __name__ == "__main__":
    main()
