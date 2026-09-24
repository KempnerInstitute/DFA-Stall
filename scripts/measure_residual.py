#!/usr/bin/env python3
"""Measure the gate--error residual on a fixed validation probe during DFA training.

Default: one seed-0 / feedback-seed-0 trajectory each for MNIST tanh, MNIST ReLU,
and CIFAR-10 tanh MLPs, sampled after 0, 50, 100, 200 and 300 updates. Uses the
same model, minibatch sampler and training step as cmc.run. No saved result is
used as input. Outputs residual_ratio.{csv,json,meta.json}; the JSON preserves
the summary keys consumed by make_numbers.py. Set CMC_DATA_ROOT for the data.

Usage: python scripts/measure_residual.py --output-dir results
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmc import data as D
from cmc.models import MLP, act_deriv
from cmc.rules import head_forward, make_feedback, n_outputs, step
from cmc.run import Config

CONFIGS = {
    "base": {"dataset": "mnist", "act": "tanh"},
    "relu": {"dataset": "mnist", "act": "relu"},
    "cifar_mlp": {"dataset": "cifar10", "act": "tanh"},
}
SAMPLE_STEPS = (0, 50, 100, 200, 300)


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def teaching_parts(gamma, error, feedback):
    """Return the exact mean teacher and its leading and residual contributions."""
    mean_error = error.mean(0)
    mean_gate = gamma.mean(0)
    lead = mean_gate * (feedback @ mean_error)
    residual = ((gamma - mean_gate) * ((error - mean_error) @ feedback.T)).mean(0)
    mean_teacher = (gamma * (error @ feedback.T)).mean(0)
    return mean_teacher, lead, residual


@torch.no_grad()
def measure(model, xp, yp, cfg, feedback, classes):
    z, pres, acts = model.forward_blocks(xp, detach=False)
    _, _, error, _ = head_forward(z, yp, cfg, classes)
    rows = []
    for layer, (a, h, b) in enumerate(zip(pres, acts, feedback["B"]), start=1):
        gamma = act_deriv(cfg.act, a, h)
        teacher, lead, residual = teaching_parts(gamma, error, b)
        norm_r, norm_lead = float(residual.norm()), float(lead.norm())
        rows.append(dict(layer=layer, norm_r=norm_r, norm_lead=norm_lead,
                         ratio=norm_r / norm_lead if norm_lead else float("nan"),
                         ebar_norm=float(error.mean(0).norm()),
                         norm_mean_teacher=float(teacher.norm()),
                         decomposition_error=float((teacher - lead - residual).norm())))
    return rows


def run_condition(name, cfg, sample_steps=SAMPLE_STEPS, dataset=None):
    """Train a fresh MLP; optional dataset injection supports data-free checks."""
    sample_steps = sorted(set(sample_steps))
    if not sample_steps or sample_steps[0] < 0:
        raise ValueError("sample steps must be nonnegative and nonempty")
    if cfg.arch != "mlp" or cfg.rule != "dfa" or cfg.center != "none":
        raise ValueError("residual measurement covers uncentered DFA MLPs")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(cfg.device)
    ds = dataset if dataset is not None else D.load(cfg.dataset, preprocess=cfg.preprocess, flatten=True)
    classes = ds["C"]
    torch.manual_seed(cfg.seed)
    model = MLP(ds["Xtr"].shape[1], [cfg.width] * cfg.depth, n_outputs(cfg, classes), act=cfg.act).to(dev)
    feedback = make_feedback(model, n_outputs(cfg, classes), cfg, dev)
    hidden = [p for l in range(cfg.depth) for p in model.block_params(l)]
    opt = torch.optim.SGD([{"params": hidden, "lr": cfg.lr * cfg.hid_lr_mult},
                           {"params": model.out.parameters(), "lr": cfg.lr * cfg.out_lr_mult}])
    sampler = D.Sampler(ds["ytr"], cfg.batch, seed=cfg.seed, p0=cfg.p0)
    probe_sampler = D.Sampler(ds["yval"], cfg.probe_n, seed=cfg.seed + 999, p0=cfg.p0)
    probe_idx = probe_sampler.draw()
    xp, yp = ds["Xval"][probe_idx].to(dev), ds["yval"][probe_idx].to(dev)
    metadata = dict(config=asdict(cfg), sample_steps=sample_steps,
                    training_examples=len(ds["ytr"]), validation_examples=len(ds["yval"]),
                    classes=classes, input_dim=ds["Xtr"].shape[1],
                    probe_seed=cfg.seed + 999, probe_indices_sha256=tensor_hash(probe_idx),
                    probe_inputs_sha256=tensor_hash(xp), probe_labels_sha256=tensor_hash(yp),
                    training_labels_sha256=tensor_hash(ds["ytr"]),
                    feedback_sha256=[tensor_hash(b) for b in feedback["B"]])
    rows, state = [], {"gate_on": 0.0}
    for it in range(sample_steps[-1] + 1):
        if it:
            idx = sampler.draw()
            result = step(model, ds["Xtr"][idx].to(dev), ds["ytr"][idx].to(dev),
                          classes, cfg, feedback, opt, state=state)
            if not np.isfinite(result["loss"]):
                raise RuntimeError(f"{name} diverged at step {it}")
        if it in sample_steps:
            for row in measure(model, xp, yp, cfg, feedback, classes):
                rows.append(dict(config=name, seed=cfg.seed, fb_seed=cfg.fb_seed,
                                 dataset=cfg.dataset, act=cfg.act, step=it, **row))
    return pd.DataFrame(rows), metadata


def summarize(frame):
    """Per-layer phase means and maxima over all sampled rows in each phase."""
    summary = {}
    for config, data in frame.groupby("config", sort=False):
        for phase, steps in (("init", [0]), ("mid", [50]), ("plateau", [100, 200, 300])):
            selected = data[data.step.isin(steps)]
            if selected.empty:
                continue
            summary[f"{config}_{phase}_by_layer"] = selected.groupby("layer").ratio.mean().tolist()
            summary[f"{config}_{phase}_max"] = float(selected.ratio.max())
        for it in (0, 300):
            selected = data[data.step.eq(it)]
            if not selected.empty:
                for metric in ("norm_r", "norm_lead"):
                    summary[f"{config}_step{it}_{metric}_by_layer"] = selected.groupby("layer")[metric].mean().tolist()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fb-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    frames, runs = [], {}
    for name in args.configs:
        cfg = Config(**CONFIGS[name], seed=args.seed, fb_seed=args.fb_seed,
                     device=args.device, threads=args.threads, steps=max(SAMPLE_STEPS),
                     exp="residual", tag=name, snapshots=",".join(map(str, SAMPLE_STEPS)))
        frame, metadata = run_condition(name, cfg, sample_steps=SAMPLE_STEPS)
        frames.append(frame)
        runs[name] = metadata
        print(f"measured {name}: {len(frame)} rows, seed={args.seed}, fb_seed={args.fb_seed}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    sources = [Path(__file__).resolve()] + [ROOT / "src" / "cmc" / f"{name}.py"
                                         for name in ("data", "models", "rules", "run")]
    provenance = dict(schema_version=1, python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, device=args.device, threads=args.threads,
                      runs=runs, aggregation="per-layer mean over sampled phase steps; phase maximum over all layers and steps",
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "residual_ratio.csv", index=False)
    (args.output_dir / "residual_ratio.json").write_text(json.dumps(summarize(frame), indent=1, allow_nan=False) + "\n")
    (args.output_dir / "residual_ratio.meta.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(frame.pivot_table(index=["config", "step"], columns="layer", values="ratio").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
