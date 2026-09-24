#!/usr/bin/env python3
"""Reproducible, resumable submission controls; one GPU process, cached data.

All comparisons use paired initialization/feedback seeds 0, 1, 2 and 3,000
updates, except the near-zero-error batch sweep (1,500 updates). No early
stopping or selection by outcome. The complete width-depth grid is rerun
with the same instrumentation. Adam uses its standard betas and epsilon.
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


def jobs(device="cuda"):
    common = dict(device=device, threads=2, save_snapshots=0, measure_updates=1)
    controls = dict(base={}, center_e={"center": "e"}, prior={"out_bias_q": .1}, bp={"rule": "bp"})
    for lr in (1e-3, 1e-4):
        for label, changes in controls.items():
            for seed in range(3):
                yield Config(exp="E13_adam", tag=f"adam{lr:g}_{label}", optimizer="adam", lr=lr,
                             seed=seed, fb_seed=seed, **changes, **common)
    for seed in range(3):
        yield Config(exp="E13_adam", tag="sgd_base", seed=seed, fb_seed=seed, **common)
    for label, changes in dict(row_norm={"fb_row_norm": 1}, sigmoid={"act": "sigmoid"},
                               sigmoid_center_e={"act": "sigmoid", "center": "e"},
                               sigmoid_bp={"act": "sigmoid", "rule": "bp"}).items():
        for seed in range(3):
            yield Config(exp="E14_controls", tag=label, seed=seed, fb_seed=seed, **changes, **common)
    for head in ("sigmoid_bce", "softmax_ce"):
        for seed in range(3):
            yield Config(exp="E15_cifar100", tag=head, dataset="cifar100", head=head,
                         seed=seed, fb_seed=seed, **common)
    # Matched BP references for the same readouts, initializations and minibatches (added 2026-09-23).
    for head in ("sigmoid_bce", "softmax_ce"):
        for seed in range(3):
            yield Config(exp="E15_cifar100", tag=f"{head}_bp", dataset="cifar100", head=head, rule="bp",
                         seed=seed, fb_seed=seed, **common)
    for dataset, head, calibrated in (("mnist", "softmax_ce", 0), ("cifar10", "sigmoid_bce", 1)):
        for batch in (16, 32, 128, 512, 2048):
            for seed in range(3):
                yield Config(exp="E16_batch", tag=f"{dataset}_b{batch}", dataset=dataset, head=head,
                             calibrate_head=calibrated, batch=batch, steps=1500,
                             seed=seed, fb_seed=seed, **common)
    for width in (100, 300, 600, 900):
        for depth in (1, 3, 5, 7):
            for seed in range(3):
                yield Config(exp="E17_grid", tag=f"w{width}_d{depth}", width=width, depth=depth,
                             seed=seed, fb_seed=seed, **common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    configs = [c for c in jobs(args.device) if not args.families or c.exp in args.families]
    manifest = ROOT / "configs/revision_20260918.json"
    manifest.write_text(json.dumps([asdict(c) for c in jobs(args.device)], indent=1) + "\n")
    print(f"Planned {len(configs)} runs. Manifest: {manifest}", flush=True)
    if args.plan_only:
        return
    # Loading a dataset is deterministic and does not depend on the run seed.
    D.load = lru_cache(maxsize=1)(D.load)
    start = time.time()
    for i, cfg in enumerate(configs):
        meta = ROOT / f"results/{cfg.exp}/{cfg.tag}/seed{cfg.seed}_fb{cfg.fb_seed}.meta.json"
        if meta.exists():
            old = json.loads(meta.read_text())
            # Fields added after a run was recorded take their defaults, which reproduce the earlier behaviour.
            defaults = {f.name: f.default for f in fields(Config)}
            if old.get("final_step", 0) >= cfg.steps and all(old.get(k, defaults[k]) == v for k,v in asdict(cfg).items()):
                print(f"skip complete {meta.parent.name}/seed{cfg.seed}", flush=True)
                continue
            raise RuntimeError(f"Refusing to overwrite a different or incomplete run: {meta}")
        print(f"[{i+1}/{len(configs)}] {cfg.exp}/{cfg.tag} seed={cfg.seed}, elapsed={time.time()-start:.0f}s", flush=True)
        train(cfg)


if __name__ == "__main__":
    main()
