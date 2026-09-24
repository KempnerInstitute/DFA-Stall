"""One training run with full instrumentation. Output: results/<exp>/<tag>/seed<S>_fb<F>.csv, meta.json, snapshots.npz"""
from __future__ import annotations
import argparse, json, math, os, time, dataclasses
from dataclasses import dataclass, asdict
import numpy as np, torch
from . import data as D
from .models import MLP, SmallCNN
from .rules import make_feedback, step, ErrorWhitener, n_outputs, prior_loss
from .diagnostics import probe_pass, snapshot

@dataclass
class Config:
    exp: str = "smoke"; tag: str = "base"; out_root: str = ""
    dataset: str = "mnist"; preprocess: str = "div255"; classes: int = 0; p0: float = 0.0
    arch: str = "mlp"; width: int = 300; depth: int = 3; act: str = "tanh"; batchnorm: int = 0
    head: str = "sigmoid_bce"; target_scale: float = 1.0; target_center: int = 0; extra_bits: int = 2
    rule: str = "dfa"; fb_scale: float = 1.0; fb_seed: int = 0; seed: int = 0
    lr: float = 1e-3; out_lr_mult: float = 1.0; hid_lr_mult: float = 1.0; batch: int = 128; steps: int = 3000
    out_bias_q: float = -1.0; out_w_scale: float = 1.0; freeze_bias: int = 0; freeze_out: int = 0; calibrate_head: int = 0
    center: str = "none"; gate_thr: float = 0.1; muon: int = 0; whiten: int = 0; whiten_lam: float = 0.1
    shift_step: int = 0; shift_p0: float = 0.0
    log_every: int = 10; heavy_every: int = 50; probe_n: int = 2048; snapshots: str = "0,50,100,120,200,300,500,1000,1500,3000"
    device: str = "cpu"; threads: int = 4; save_snapshots: int = 1; anatomy: int = 0
    optimizer: str = "sgd"; fb_row_norm: int = 0; measure_updates: int = 0
    init: str = "xavier"; fb_dist: str = "normal"   # "zero"/"uniform_fanin" and "uniform_fanout" reproduce Nokland (2016)
    fb_signal: str = "error"   # "sign": hidden layers receive sign(e), as in sign-error DFA; the readout keeps its gradient

def make_optimizer(model, cfg):
    hidden = [p for l in range(len(model.hidden)) for p in model.block_params(l)]
    groups = [{"params": hidden, "lr": cfg.lr * cfg.hid_lr_mult},
              {"params": list(model.out.parameters()), "lr": cfg.lr * cfg.out_lr_mult}]
    if cfg.optimizer == "sgd": return torch.optim.SGD(groups)
    if cfg.optimizer == "adam": return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)
    if cfg.optimizer == "rmsprop": return torch.optim.RMSprop(groups, alpha=0.99, eps=1e-8)   # Torch7 optim.rmsprop defaults
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")

def apply_init(model, how):
    """Re-initialize an MLP: "zero" sets every weight and bias to zero (DFA/FA in Nokland, 2016); "uniform_fanin" draws
    weights and biases from U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (his BP protocol). "xavier" keeps the default."""
    if how == "xavier": return
    if not isinstance(model, MLP) or model.bns is not None:
        raise ValueError(f"init={how} is implemented for MLPs without batch normalization")
    with torch.no_grad():
        for layer in list(model.hidden) + [model.out]:
            if how == "zero":
                layer.weight.zero_(); layer.bias.zero_()
            elif how == "uniform_fanin":
                bound = 1.0 / math.sqrt(layer.in_features)
                layer.weight.uniform_(-bound, bound); layer.bias.uniform_(-bound, bound)
            else:
                raise ValueError(f"unknown init: {how}")

def parse():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return Config(**vars(ap.parse_args()))

def out_dir(cfg):
    root = cfg.out_root or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
    d = os.path.join(root, cfg.exp, cfg.tag); os.makedirs(d, exist_ok=True); return d

def main(cfg: Config):
    torch.set_num_threads(cfg.threads); torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ds = D.load(cfg.dataset, preprocess=cfg.preprocess, classes=(cfg.classes or None), flatten=(cfg.arch == "mlp"))
    C = ds["C"]; Cout = n_outputs(cfg, C)
    in_dim = ds["Xtr"].shape[1] if cfg.arch == "mlp" else ds["Xtr"].shape[1]
    torch.manual_seed(cfg.seed)   # init depends on seed only
    if cfg.arch == "mlp": model = MLP(in_dim, [cfg.width] * cfg.depth, Cout, act=cfg.act, batchnorm=bool(cfg.batchnorm))
    else: model = SmallCNN(in_dim, Cout, act=cfg.act, img=ds["Xtr"].shape[-1])
    apply_init(model, cfg.init)
    model = model.to(dev)
    if cfg.out_bias_q > 0:
        with torch.no_grad():
            if cfg.head == "softmax_ce": model.out.bias.fill_(0.0)
            else: model.out.bias.fill_(math.log(cfg.out_bias_q / (1 - cfg.out_bias_q)))
    if cfg.out_w_scale != 1.0:
        with torch.no_grad(): model.out.weight.mul_(cfg.out_w_scale)
    if cfg.calibrate_head:
        # Shift the mean initial logit to the prior logit, including W_out hbar_L.
        # For nonlinear heads this only approximately calibrates the mean prediction.
        with torch.no_grad():
            cal = D.Sampler(ds["ytr"], 1024, seed=cfg.seed + 4242, p0=cfg.p0); xi = cal.draw(1024)
            zbar = model.forward_blocks(ds["Xtr"][xi].to(dev), detach=False)[0].mean(0)
            pic = torch.tensor(cal.prior(C), dtype=zbar.dtype, device=dev).clamp(1e-6, 1 - 1e-6)
            if cfg.head == "softmax_ce": target = torch.log(pic)
            elif cfg.head == "sigmoid_bce": target = torch.log(pic / (1 - pic))
            elif cfg.head == "mse": target = cfg.target_scale * (pic - (1.0 / C if cfg.target_center else 0.0))
            else: raise NotImplementedError(f"calibrate_head is not defined for head {cfg.head}")
            model.out.bias.add_(target - zbar)
    fb = make_feedback(model, Cout, cfg, dev)
    opt = make_optimizer(model, cfg)
    whitener = ErrorWhitener(model.widths, lam_rel=cfg.whiten_lam, device=dev) if cfg.whiten else None
    sampler = D.Sampler(ds["ytr"], cfg.batch, seed=cfg.seed, p0=cfg.p0)     # batch order depends on seed only
    # probes: from the validation split, drawn with the training prior; plus a balanced-by-construction accuracy probe
    psamp = D.Sampler(ds["yval"], cfg.probe_n, seed=cfg.seed + 999, p0=cfg.p0); pidx = psamp.draw(cfg.probe_n)
    Xp, yp = ds["Xval"][pidx].to(dev), ds["yval"][pidx].to(dev)
    pi = sampler.prior(C); Lprior = prior_loss(pi, cfg, C)
    ehat0 = None; rows = []; snaps = {}; snap_steps = set(int(s) for s in cfg.snapshots.split(",") if s)
    state = {"gate_on": 0.0}; kappa = np.zeros(len(model.hidden)); t0 = time.time(); run_loss = []
    Xtr, ytr = ds["Xtr"], ds["ytr"]
    batch_stats = {"ebar": float("nan"), "etil": float("nan")}
    def log(stepno, heavy):
        d = probe_pass(model, Xp, yp, C, cfg, fb, heavy=heavy, ehat0=ehat0)
        d.update(step=stepno, train_loss=float(np.mean(run_loss)) if run_loss else float("nan"), gate_on=state["gate_on"], wall=time.time() - t0,
                 batch_ebar_norm=batch_stats["ebar"], batch_etil_norm=batch_stats["etil"])
        for l in range(len(kappa)): d[f"kappa_l{l+1}"] = float(kappa[l])
        rows.append(d)
    if 0 in snap_steps and cfg.save_snapshots: snaps["step0"] = snapshot(model, Xp, fb, cfg)
    log(0, heavy=True)
    for it in range(1, cfg.steps + 1):
        if cfg.shift_step and it == cfg.shift_step:
            sampler.p0 = cfg.shift_p0            # label-prior shift mid-training; the probe keeps the pre-shift prior
        idx = sampler.draw(); xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev)
        # online collapse number: kappa_l(t) = g * sum_t eta <ebar_t, ehat0> (||hbar_{l-1}||^2 + 1)  (batch quantities)
        state["measure_update"] = bool(cfg.measure_updates and (it % cfg.heavy_every == 0 or it == 1))
        res = step(model, xb, yb, C, cfg, fb, opt, whitener=whitener, state=state)
        run_loss.append(res["loss"]); run_loss = run_loss[-cfg.log_every:]
        with torch.no_grad():
            ebar = res["e"].mean(0)
            batch_stats["ebar"] = float(ebar.norm()); batch_stats["etil"] = float((res["e"] - ebar).norm(dim=1).pow(2).mean().sqrt())
            if ehat0 is None: ehat0 = ebar / (ebar.norm() + 1e-12)
            proj = float((ebar * ehat0).sum())
            hin = [xb.flatten(1)] + [h.flatten(1) for h in res["acts"][:-1]]
            for l in range(len(kappa)):
                Hprev = float((hin[l].mean(0) ** 2).sum()); kappa[l] += cfg.fb_scale * cfg.lr * cfg.hid_lr_mult * proj * (Hprev + 1)
        if it in snap_steps and cfg.save_snapshots: snaps[f"step{it}"] = snapshot(model, Xp, fb, cfg)
        if it % cfg.log_every == 0 or it == cfg.steps:
            log(it, heavy=(it % cfg.heavy_every == 0 or it == cfg.steps))
            rows[-1].update(res.get("update_metrics", {}))
        if not math.isfinite(res["loss"]):
            print("diverged at step", it); break
    d = out_dir(cfg); stem = f"seed{cfg.seed}_fb{cfg.fb_seed}"
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(d, stem + ".csv"), index=False)
    meta = dict(asdict(cfg), C=C, Cout=Cout, prior=pi.tolist(), prior_loss=Lprior, in_dim=int(in_dim), widths=model.widths,
                xbar2=float((ds["Xtr"].flatten(1).mean(0) ** 2).sum()), wall=time.time() - t0, final_step=rows[-1]["step"])
    json.dump(meta, open(os.path.join(d, stem + ".meta.json"), "w"), indent=1)
    if snaps and cfg.save_snapshots:
        flat = {f"{k}/{kk}": vv for k, v in snaps.items() for kk, vv in v.items()}
        np.savez_compressed(os.path.join(d, stem + ".snapshots.npz"), **flat)
    print(f"done {cfg.exp}/{cfg.tag}/{stem}: steps={rows[-1]['step']} loss={rows[-1]['probe_loss']:.3f} acc={rows[-1]['probe_acc']:.3f} "
          f"cos_last={rows[-1].get('cos_l%d' % len(model.hidden)):.3f} Lprior={Lprior:.4f} wall={time.time()-t0:.0f}s")
    return rows, meta

if __name__ == "__main__":
    main(parse())
