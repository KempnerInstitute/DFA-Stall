#!/usr/bin/env python3
"""Sign-error DFA (E20): a test of the prediction that a broadcast teaching signal whose common mode does not decay with
readout calibration produces persistent collapse. Hidden layers receive sign(e) through the baseline Gaussian feedback;
the readout keeps its true gradient. Arms: plain, error centering (sign(e) minus its batch mean) and prior-bias readout
initialization. Baseline MNIST protocol otherwise (3x300 tanh, sigmoid/BCE, SGD 1e-3, batch 128), paired seeds 0, 1, 2 and
3,000 updates; resumable, no selection by outcome.
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

EXP = "E20_sign"
ARMS = {"sign": {}, "sign_center_e": {"center": "e"}, "sign_prior": {"out_bias_q": .1}}


def jobs(device="cuda"):
    for arm, changes in ARMS.items():
        for seed in range(3):
            yield Config(exp=EXP, tag=arm, fb_signal="sign", seed=seed, fb_seed=seed, device=device, threads=1,
                         save_snapshots=0, **changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    configs = list(jobs(args.device))
    (ROOT / "configs/sign_20260923.json").write_text(json.dumps([asdict(c) for c in configs], indent=1) + "\n")
    print(f"Planned {len(configs)} runs.", flush=True)
    if args.plan_only:
        return
    D.load = lru_cache(maxsize=1)(D.load)
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
