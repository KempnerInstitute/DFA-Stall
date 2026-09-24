"""Teacher-student ladder that demarcates the pre-alignment common-mode collapse from the post-alignment
plateau of Refinetti, d'Ascoli, Ohana and Goldt (ICML 2021, "Align, then memorise").

Rung 0 is a replica of their nonlinear teacher-student setting (isotropic Gaussian inputs, erf teacher with
unit second-layer weights, erf student, online SGD on the squared loss, no biases, DFA with fixed random
feedback plus a BP twin). Each further rung adds exactly one ingredient of our MNIST protocol, so the rung at
which our signature (hidden cosine -> 1, plateau pinned to the constant-predictor loss, weight alignment flat
and low) switches on identifies the ingredient that causes it.

Implementation note. This module runs its own minimal NumPy loop rather than `cmc.rules.step` / `cmc.models.MLP`.
Reasons: (i) rung 0 needs batch-1 online SGD for up to 1e6 steps, where the torch per-step overhead dominates
by ~20x; (ii) the teacher-student convention puts a 1/sqrt(fan_in) factor in the forward pass and uses no
biases at rungs 0-1, which `models.MLP` does not express. The DFA/BP update equations are identical to the
ones in `cmc.rules.step`; `selfcheck()` asserts that numerically (one DFA step, matched weights, < 1e-5).

Layer convention: a_l = W_l h_{l-1} / sqrt(d_{l-1}) [+ b_l], h_l = phi(a_l), z = V h_L / sqrt(d_L) + b_out,
with all weight entries drawn ~ N(0,1) (output layer ~ N(0, v_scale^2)). Hidden learning rate eta, output
learning rate eta_out = eta / D, which is the Saad-Solla / Goldt scaling that puts the second layer on the
same O(1) timescale as the order parameters when time is measured as t = step / D.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.special import erf

SQRT2 = math.sqrt(2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


# ---------------------------------------------------------------- activations
def act(name, a):
    if name == "erf":
        return erf(a / SQRT2)
    if name == "tanh":
        return np.tanh(a)
    raise ValueError(name)


def act_deriv(name, a, h):
    if name == "erf":
        return SQRT_2_OVER_PI * np.exp(-0.5 * a * a)
    if name == "tanh":
        return 1.0 - h * h
    raise ValueError(name)


# ---------------------------------------------------------------- config
@dataclass
class LadderConfig:
    """One run. `rung` is the output directory name; ingredients are the fields below it."""
    rung: str = "rung0_scalar_dfa"
    seed: int = 0
    rule: str = "dfa"              # dfa | bp
    D: int = 500                   # input dimension
    K: int = 2                     # teacher hidden units
    C: int = 1                     # outputs (1 = scalar sum of teacher units; K = identity readout)
    widths: str = "2"              # student hidden widths, comma separated
    student_act: str = "erf"       # erf | tanh
    head: str = "mse"              # mse | sigmoid_bce (one-versus-rest labels = argmax of the teacher)
    bias: int = 0                  # trainable hidden biases (initialised at zero)
    x_mean: float = 0.0            # inputs x = x_mean + x_std * z,  z ~ N(0, I_D)
    x_std: float = 1.0
    fb_scale: float = 1.0          # feedback gain g; B_l entries ~ N(0, g^2)
    lr: float = 0.5                # hidden learning rate eta
    out_lr_div: float = 0.0        # eta_out = eta / out_lr_div  (0 -> use D, the Goldt scaling)
    v_scale: float = 0.1           # output weight init scale
    steps: int = 200_000
    probe_n: int = 2048
    n_log: int = 300


def widths_of(cfg):
    return [int(w) for w in str(cfg.widths).split(",") if w]


# ---------------------------------------------------------------- network
class Net:
    """Fully connected student with the 1/sqrt(fan_in) forward convention."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        dims = [cfg.D] + widths_of(cfg)
        self.W = [rng.standard_normal((dims[i + 1], dims[i])) for i in range(len(dims) - 1)]
        self.b = [np.zeros(d) for d in dims[1:]] if cfg.bias else None
        self.V = cfg.v_scale * rng.standard_normal((cfg.C, dims[-1]))
        self.b_out = np.zeros(cfg.C)
        self.dims = dims
        self.scale = [1.0 / math.sqrt(d) for d in dims[:-1]]
        self.out_scale = 1.0 / math.sqrt(dims[-1])

    def forward(self, X):
        pres, acts, h = [], [], X
        for l, W in enumerate(self.W):
            a = (h @ W.T) * self.scale[l]
            if self.b is not None:
                a = a + self.b[l]
            h = act(self.cfg.student_act, a)
            pres.append(a)
            acts.append(h)
        z = (h @ self.V.T) * self.out_scale + self.b_out
        return pres, acts, z

    def downstream_maps(self):
        """M_l = (V W_L ... W_{l+1})^T in R^{d_l x C}, the target of rowwise weight alignment cos(M_i, B_i)."""
        eff = self.V * self.out_scale                       # [C, d_L]
        maps = [None] * len(self.W)
        maps[-1] = eff.T
        for l in range(len(self.W) - 2, -1, -1):
            eff = eff @ (self.W[l + 1] * self.scale[l + 1])
            maps[l] = eff.T
        return maps


class Teacher:
    """K erf units with unit second-layer weights: C=1 sums them, C=K reads them out with the identity."""

    def __init__(self, cfg, rng):
        self.Ws = rng.standard_normal((cfg.K, cfg.D))
        self.scale = 1.0 / math.sqrt(cfg.D)
        self.C, self.K = cfg.C, cfg.K
        if cfg.C not in (1, cfg.K):
            raise ValueError("C must be 1 or K")

    def __call__(self, X):
        u = act("erf", (X @ self.Ws.T) * self.scale)
        return u.sum(1, keepdims=True) if self.C == 1 else u


# ---------------------------------------------------------------- head
def head(cfg, z, target):
    """Returns (per-sample loss, error e = dL/dz). For sigmoid_bce `target` is the one-hot label matrix."""
    if cfg.head == "mse":
        r = z - target
        return 0.5 * (r * r).sum(1), r
    if cfg.head == "sigmoid_bce":
        p = 1.0 / (1.0 + np.exp(-z))
        loss = np.logaddexp(0.0, z).sum(1) - (z * target).sum(1)
        return loss, p - target
    raise ValueError(cfg.head)


def targets_of(cfg, teacher, X):
    """Regression targets, or one-versus-rest labels at prior 1/C from the argmax of the teacher outputs."""
    y = teacher(X)
    if cfg.head == "mse":
        return y, None
    lab = y.argmax(1)
    return np.eye(cfg.C)[lab], lab


def const_loss(cfg, T):
    """Loss of the best constant predictor on the probe targets T."""
    m = T.mean(0)
    if cfg.head == "mse":
        return float(0.5 * ((T - m) ** 2).sum(1).mean())
    q = np.clip(m, 1e-12, 1 - 1e-12)
    return float(-(q * np.log(q) + (1 - q) * np.log(1 - q)).sum())


# ---------------------------------------------------------------- gradients
def grads(net, cfg, X, target, fb, rule):
    """One batch of gradients. Returns (loss, dW, db, dV, db_out, e, deltas)."""
    pres, acts, z = net.forward(X)
    loss, e = head(cfg, z, target)
    n = X.shape[0]
    L = len(net.W)
    dV = (e.T @ acts[-1]) * net.out_scale / n
    db_out = e.mean(0)
    hin = [X] + acts[:-1]
    dW, db, deltas = [None] * L, [None] * L, [None] * L
    if rule == "dfa":
        gh = [e @ fb[l].T for l in range(L)]
    else:
        gh = [None] * L
        gh[L - 1] = (e @ net.V) * net.out_scale
    for l in range(L - 1, -1, -1):
        d = gh[l] * act_deriv(cfg.student_act, pres[l], acts[l])
        deltas[l] = d
        dW[l] = (d.T @ hin[l]) * net.scale[l] / n
        db[l] = d.mean(0)
        if rule == "bp" and l > 0:
            gh[l - 1] = (d @ net.W[l]) * net.scale[l]
    return float(loss.mean()), dW, db, dV, db_out, e, deltas


def apply_update(net, cfg, dW, db, dV, db_out, eta, eta_out):
    for l in range(len(net.W)):
        net.W[l] -= eta * dW[l]
        if net.b is not None:
            net.b[l] -= eta * db[l]
    net.V -= eta_out * dV
    net.b_out -= eta_out * db_out


# ---------------------------------------------------------------- diagnostics
def mean_pairwise_cos(H, n=512):
    X = H[:n]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    G = X @ X.T
    m = X.shape[0]
    return float((G.sum() - m) / (m * (m - 1)))


def participation(u):
    return float(u.sum() ** 2 / (u.size * (u * u).sum() + 1e-30))


def rowcos(A, B):
    na = np.linalg.norm(A, axis=1) + 1e-30
    nb = np.linalg.norm(B, axis=1) + 1e-30
    return float(((A * B).sum(1) / (na * nb)).mean())


def veccos(A, B):
    a, b = A.ravel(), B.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def probe_pass(net, cfg, Xp, Tp, labp, fb):
    pres, acts, z = net.forward(Xp)
    loss, e = head(cfg, z, Tp)
    ebar = e.mean(0)
    d = dict(test_loss=float(loss.mean()), ebar_norm=float(np.linalg.norm(ebar)),
             etil_norm=float(np.sqrt(((e - ebar) ** 2).sum(1).mean())))
    d["acc"] = float((z.argmax(1) == labp).mean()) if labp is not None else float("nan")
    maps = net.downstream_maps()
    _, dW_bp, _, _, _, _, _ = grads(net, cfg, Xp, Tp, fb, "bp")
    _, dW_dfa, _, _, _, _, _ = grads(net, cfg, Xp, Tp, fb, "dfa")
    for l, (a, h) in enumerate(zip(pres, acts)):
        u = (act_deriv(cfg.student_act, a, h) ** 2).mean(0)
        d[f"hcos_l{l+1}"] = mean_pairwise_cos(h)
        d[f"chi_l{l+1}"] = float(u.mean())
        d[f"part_l{l+1}"] = participation(u)
        d[f"mu_spread_l{l+1}"] = float(a.mean(0).std())
        d[f"sat_l{l+1}"] = float((np.abs(h.mean(0)) > 0.9).mean())
        d[f"wa_row_l{l+1}"] = rowcos(maps[l], fb[l])
        d[f"wa_vec_l{l+1}"] = veccos(maps[l], fb[l])
        d[f"ga_l{l+1}"] = veccos(dW_dfa[l], dW_bp[l])
    return d


# ---------------------------------------------------------------- training
def log_schedule(total, n_log):
    s = set(range(0, min(400, total) + 1, 10))
    s |= set(np.unique(np.geomspace(1, max(total, 2), n_log).astype(int)).tolist())
    s.add(0)
    s.add(total)
    return sorted(int(v) for v in s if v <= total)


def train(cfg: LadderConfig):
    rng = np.random.default_rng(1000 + cfg.seed)
    teacher = Teacher(cfg, np.random.default_rng(5000 + cfg.seed))
    net = Net(cfg, rng)
    fbrng = np.random.default_rng(9000 + cfg.seed)
    fb = [cfg.fb_scale * fbrng.standard_normal((d, cfg.C)) for d in widths_of(cfg)]
    eta = cfg.lr
    eta_out = cfg.lr / (cfg.out_lr_div if cfg.out_lr_div > 0 else cfg.D)
    Xp = cfg.x_mean + cfg.x_std * rng.standard_normal((cfg.probe_n, cfg.D))
    Tp, labp = targets_of(cfg, teacher, Xp)
    Lc = const_loss(cfg, Tp)
    prior = Tp.mean(0)
    sched = set(log_schedule(cfg.steps, cfg.n_log))
    rows, run_loss, t0 = [], [], time.time()

    def record(it):
        d = probe_pass(net, cfg, Xp, Tp, labp, fb)
        d.update(step=it, t=it / cfg.D, const_loss=Lc,
                 train_loss=float(np.mean(run_loss)) if run_loss else float("nan"),
                 wall=time.time() - t0)
        rows.append(d)

    record(0)
    for it in range(1, cfg.steps + 1):
        x = cfg.x_mean + cfg.x_std * rng.standard_normal((1, cfg.D))
        tgt, _ = targets_of(cfg, teacher, x)
        loss, dW, db, dV, db_out, _, _ = grads(net, cfg, x, tgt, fb, cfg.rule)
        apply_update(net, cfg, dW, db, dV, db_out, eta, eta_out)
        run_loss.append(loss)
        if len(run_loss) > 200:
            run_loss = run_loss[-200:]
        if it in sched:
            record(it)
            if not math.isfinite(rows[-1]["test_loss"]):
                print(f"diverged at step {it}")
                break
    meta = dict(asdict(cfg), const_loss=Lc, prior=prior.tolist(), eta_out=eta_out,
                widths_list=widths_of(cfg), wall=time.time() - t0, final_step=rows[-1]["step"])
    return rows, meta


# ---------------------------------------------------------------- the ladder
def rungs():
    """Named rungs. Rung 0 replicates Refinetti et al.; each later rung adds one ingredient to the previous.

    Protocol shared by every rung: D = 500, eta = 0.5, eta_out = eta/10, output weights initialised at scale
    0.3, feedback gain 1, batch-1 online SGD on a fresh sample per step. eta_out/eta = 1/10 rather than the
    1/D of Goldt et al. is the one deviation from the replica: with 1/D the second layer moves so slowly that
    the alignment phase does not separate from the initial transient inside our step budget.
    """
    base = dict(D=500, lr=0.5, out_lr_div=10.0, v_scale=0.3, fb_scale=1.0)
    r0s = dict(base, K=4, C=1, widths="4", student_act="erf", head="mse", bias=0, steps=1_000_000)
    r0v = dict(r0s, C=4, widths="4")
    r1 = dict(r0v, head="sigmoid_bce", steps=400_000)
    r2 = dict(r1, bias=1)
    r3 = dict(r2, widths="300,300,300", student_act="tanh", steps=400_000, probe_n=1024, n_log=200)
    r4 = dict(r3, x_mean=0.13, x_std=0.3)
    out = {
        "rung0_scalar_K2M2": dict(r0s, K=2, widths="2"),
        "rung0_scalar_K2M4": dict(r0s, K=2, widths="4"),
        "rung0_scalar_K4M4": r0s,
        "rung0_scalar_K4M8": dict(r0s, widths="8"),
        "rung0_vector_K4M4": r0v,
        "rung0_vector_K4M8": dict(r0v, widths="8"),
        "rung1_nonzero_mean_targets": r1,
        "rung2_hidden_biases": r2,
        "rung3_depth3_tanh": r3,
        "rung4_shifted_inputs": r4,
        "rung5_gain0.3": dict(r4, fb_scale=0.3),
        "rung5_gain3": dict(r4, fb_scale=3.0),
    }
    full = {}
    for name, kw in out.items():
        for rule in ("dfa", "bp"):
            full[f"{name}_{rule}"] = dict(kw, rule=rule, rung=f"{name}_{rule}")
    return full


# ---------------------------------------------------------------- selfcheck
def selfcheck(tol=1e-5):
    """Assert that one DFA update here equals one `cmc.rules.step` update for matched weights.

    The two codes use different parametrisations of the same network: here a_l = W h/sqrt(fan_in) + b with
    O(1) weights, there a_l = W_t h + b with W_t = W/sqrt(fan_in). The parameter maps are W_t = W/sqrt(fan_in),
    b_t = b, so a single hidden learning rate eta here is a torch learning rate eta/fan_in on the weight and
    eta on the bias; that per-parameter map is applied below and is the only difference between the two runs.
    """
    import torch
    from types import SimpleNamespace
    from cmc.models import MLP
    from cmc.rules import step

    D, M, C, B, eta, eta_out = 40, 12, 5, 16, 0.05, 0.02
    cfg = LadderConfig(D=D, C=C, widths=str(M), student_act="tanh", head="sigmoid_bce", bias=1, rule="dfa")
    rng = np.random.default_rng(0)
    net = Net(cfg, rng)
    net.b[0] = rng.standard_normal(M) * 0.1
    net.b_out = rng.standard_normal(C) * 0.1
    tcfg = SimpleNamespace(head="sigmoid_bce", rule="dfa", center="none", extra_bits=0, seed=0, freeze_bias=0,
                           freeze_out=0, muon=0, target_scale=1.0, target_center=0, gate_thr=0.5, fb_scale=1.0)
    model = MLP(D, [M], C, act="tanh").double()
    with torch.no_grad():
        model.hidden[0].weight.copy_(torch.tensor(net.W[0] * net.scale[0]))
        model.hidden[0].bias.copy_(torch.tensor(net.b[0]))
        model.out.weight.copy_(torch.tensor(net.V * net.out_scale))
        model.out.bias.copy_(torch.tensor(net.b_out))
    X = rng.standard_normal((B, D))
    lab = rng.integers(0, C, B)
    fbm = rng.standard_normal((M, C))
    opt = torch.optim.SGD([{"params": [model.hidden[0].weight], "lr": eta / D},
                           {"params": [model.hidden[0].bias], "lr": eta},
                           {"params": [model.out.weight], "lr": eta_out / M},
                           {"params": [model.out.bias], "lr": eta_out}])
    step(model, torch.tensor(X), torch.tensor(lab), C, tcfg, {"B": [torch.tensor(fbm)]}, opt)
    _, dW, db, dV, db_out, _, _ = grads(net, cfg, X, np.eye(C)[lab], [fbm], "dfa")
    apply_update(net, cfg, dW, db, dV, db_out, eta, eta_out)
    dev = [float(np.abs(model.hidden[0].weight.detach().numpy() - net.W[0] * net.scale[0]).max()),
           float(np.abs(model.hidden[0].bias.detach().numpy() - net.b[0]).max()),
           float(np.abs(model.out.weight.detach().numpy() - net.V * net.out_scale).max()),
           float(np.abs(model.out.bias.detach().numpy() - net.b_out).max())]
    ok = max(dev) < tol
    print(f"selfcheck vs cmc.rules.step: max abs deviation {max(dev):.2e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(dev)
    return dev


# ---------------------------------------------------------------- cli
def out_dir(root, rung):
    d = os.path.join(root, rung)
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="override the rung's step budget")
    ap.add_argument("--out_root", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "results", "E6_ladder"))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    table = rungs()
    if args.list:
        for k, v in table.items():
            print(k, v["steps"])
        return
    kw = dict(table[args.rung])
    if args.steps:
        kw["steps"] = args.steps
    cfg = LadderConfig(**{k: v for k, v in kw.items() if k in {f.name for f in dataclasses.fields(LadderConfig)}},
                       seed=args.seed)
    import pandas as pd
    rows, meta = train(cfg)
    d = out_dir(args.out_root, cfg.rung)
    pd.DataFrame(rows).to_csv(os.path.join(d, f"seed{cfg.seed}.csv"), index=False)
    json.dump(meta, open(os.path.join(d, f"seed{cfg.seed}.meta.json"), "w"), indent=1)
    last = rows[-1]
    print(f"done {cfg.rung}/seed{cfg.seed}: steps={last['step']} loss={last['test_loss']:.4f} "
          f"Lconst={meta['const_loss']:.4f} acc={last['acc']:.3f} wall={meta['wall']:.0f}s")


if __name__ == "__main__":
    main()
