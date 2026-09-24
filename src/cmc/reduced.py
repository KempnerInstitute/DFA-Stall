"""Initialization-based approximation of common-mode collapse and plateau exit.

The collapse closure evolves per-unit mean preactivations, layerwise within-unit
variation and mean output logits. Gaussian quadrature supplies mean activities,
derivative moments and participation. Inputs are measured at initialization;
there are no fitted trajectory parameters. The closure omits upstream motion,
local covariance and gate--error correlations, whose sizes are checked separately.

The optional recovery extension transmits activity--label covariance between
layers and adds phenomenological local growth. Its accumulated top-layer energy
approximates the readout's loss decrease. This is an exit-time approximation,
not a derivation of full covariance dynamics or participation rebound.

Usage:
    python -m cmc.reduced --from_run results/E1_headline/base/seed0_fb0.meta.json --compare
"""
from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass
import numpy as np

GH_X, GH_W = np.polynomial.hermite.hermgauss(64)

def gate_moments(mu: np.ndarray, sigma: float):
    """Per-unit (E tanh a, E sech^2 a, E sech^4 a) for a ~ N(mu_i, sigma^2), by 64-point Gauss-Hermite quadrature."""
    z = mu[:, None] + math.sqrt(2.0) * sigma * GH_X[None, :]
    w = GH_W[None, :] / math.sqrt(math.pi)
    th = np.tanh(z); s2 = 1.0 - th * th
    return (th * w).sum(1), (s2 * w).sum(1), (s2 * s2 * w).sum(1)

def participation(u: np.ndarray) -> float:
    """p = (sum u)^2 / (d sum u^2) = ubar^2 / mean(u^2)."""
    return float(u.mean() ** 2 / ((u * u).mean() + 1e-300))

def sigma_prime_eff(C: int) -> float:
    """Effective output slope in the collapse number: (1/2 - 1/C)/|logit(1/C)| (Theory A; judge adjudication D1).

    This is the balanced one-versus-rest sigmoid case of the general identity in `sigma_prime_general`, and it is
    not the asymptotic decay rate pi(1-pi), which is twice smaller at C = 10 and eleven times smaller at C = 100."""
    return (0.5 - 1.0 / C) / abs(math.log((1.0 / C) / (1.0 - 1.0 / C)))

def sigma_prime_general(zbar0: np.ndarray, zstar: np.ndarray, ebar0: np.ndarray) -> float:
    """sigma'_eff = ||ebar_0|| / |<zbar(0) - zstar, ehat_0>|, from integrating dzbar/dt = -eta_out (H_L+1) ebar.

    Integrating the head equation gives int (H_L+1) ebar dt = (zbar(0) - zstar)/eta_out for any head, so the dose
    int <ebar, ehat_0> dt -- the projection on the fixed common-mode direction, which is the quantity cmc.run
    accumulates -- equals ||ebar_0|| / [sigma'_eff eta_out <H_L+1>] with this ratio. Taking the component along
    ehat_0 rather than the norm matters whenever the mean error rotates (an unbalanced softmax head, say); for the
    balanced sigmoid head ebar stays parallel to ehat_0, zbar(0) = 0, and it reduces to Theory A's
    (1/2 - 1/C)/|logit(1/C)|."""
    ehat0 = ebar0 / (np.linalg.norm(ebar0) + 1e-30)
    return float(np.linalg.norm(ebar0) / (abs(float((zbar0 - zstar) @ ehat0)) + 1e-30))

def kappa_closed_form(g, ebar0_norm, H_prev=None, H_L=None, C=None, r_out=1.0, sig_eff=None, drive_ratio=None):
    """kappa_l = g ||ebar_0|| (H_{l-1}+1) / [sigma'_eff (H_L+1) r_out]  (Theory A eq. 4, exact dose integral).

    `drive_ratio` replaces (H_{l-1}+1)/(H_L+1) when that ratio is known directly: as a dose-weighted average along
    a trajectory, or with the hidden bias channel switched off, where the numerator is H_{l-1} rather than H_{l-1}+1."""
    s = sigma_prime_eff(C) if sig_eff is None else sig_eff
    ratio = drive_ratio if drive_ratio is not None else (H_prev + 1.0) / (H_L + 1.0)
    return float(g * ebar0_norm * ratio / (s * r_out))

# ---------------- heads: mean error, mean-route loss, and the prior fixed point ----------------
SIGMOID_HEADS = ("sigmoid_bce", "multilabel")

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def _softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def mean_error(zbar: np.ndarray, tbar: np.ndarray, head: str) -> np.ndarray:
    """ebar = dL/dz at the mean logit: s(zbar) - tbar for the sigmoid heads, softmax(zbar) - tbar, or zbar - tbar."""
    if head in SIGMOID_HEADS: return _sigmoid(zbar) - tbar
    if head == "softmax_ce": return _softmax(zbar) - tbar
    if head == "mse": return zbar - tbar
    raise ValueError(head)

def _loss_raw(zbar: np.ndarray, tbar: np.ndarray, head: str) -> float:
    if head in SIGMOID_HEADS:
        p = np.clip(_sigmoid(zbar), 1e-12, 1 - 1e-12)
        return float(-(tbar * np.log(p) + (1 - tbar) * np.log(1 - p)).sum())
    if head == "softmax_ce":
        return float(-(tbar * np.log(np.clip(_softmax(zbar), 1e-12, None))).sum())
    if head == "mse":
        return float(0.5 * ((zbar - tbar) ** 2).sum())
    raise ValueError(head)

def prior_fixed_point(tbar: np.ndarray, head: str, zbar0: np.ndarray = None) -> np.ndarray:
    """The zbar at which ebar = 0 (the constant predictor at the prior).

    The softmax head is shift invariant and its dynamics conserves sum_c zbar_c, so its fixed point is pinned to
    the same total as zbar(0) when one is given."""
    if head in SIGMOID_HEADS:
        t = np.clip(tbar, 1e-9, 1 - 1e-9); return np.log(t / (1 - t))
    if head == "softmax_ce":
        z = np.log(np.clip(tbar, 1e-12, None))
        return z if zbar0 is None else z + (zbar0.mean() - z.mean())
    if head == "mse": return tbar.copy()
    raise ValueError(head)

# ---------------- the initial state, read off a real network ----------------
@dataclass
class Init:
    """Everything the reduced model needs, all measurable on the network and the data before any training."""
    H0: float                       # ||xbar||^2, the layer-1 presynaptic mean energy
    mu0: list                       # per-layer arrays of unit mean preactivations at t=0
    sigma0: list                    # per-layer within-unit spread at t=0 (measured)
    zbar0: np.ndarray               # mean output logit at t=0
    q: list                         # per hidden layer: fan_in_l Var(W_l)   (forward variance gain, Theory A eq. 6)
    r: list                         # per hidden layer: d_l Var(W_l)        (Frobenius gain of W_l, for Lambda)
    Lam0: float                     # ||Cov(x, t)||_F^2, the fixed data seed of the escape cascade
    lam0: list                      # measured ||Cov(h_l, t)||_F^2 at t=0 (validation only, not an input)
    widths: list
    C: int                          # number of classes
    tbar: np.ndarray                # mean target vector per output unit (length C_out)
    sigma_x2: float                 # tr Cov(x) / d_0, the a-priori seed of the variance cascade
    prior_loss: float = float("nan")

def extract_init(model, X, y, C: int, head: str = "sigmoid_bce", target_scale: float = 1.0,
                 target_center: bool = False, extra_bits: int = 2, seed: int = 0, prior=None) -> Init:
    """Read H_0, sigma_l(0), mu_l(0), zbar(0), w_l^2 = fan_in Var(W_l) and ||Cov(x,t)||_F^2 off an MLP and a data batch.

    X: [N, d_0] inputs (the probe), y: [N] integer labels. `prior` overrides the label prior used for tbar
    (e.g. the sampling prior of an imbalanced run); otherwise it is the empirical prior of y."""
    import torch
    from .rules import multilabel_targets
    with torch.no_grad():
        Xf = X.flatten(1)
        t = torch.nn.functional.one_hot(y, C).float()
        if head == "multilabel": t = multilabel_targets(y, C, extra_bits, seed)
        elif head == "mse":
            t = t * target_scale
            if target_center: t = t - target_scale / C
        tbar = t.mean(0).numpy().astype(float)
        if prior is not None and head in ("sigmoid_bce", "softmax_ce"): tbar = np.asarray(prior, float)
        tc = t - t.mean(0, keepdim=True)
        Lam0 = float((((Xf - Xf.mean(0, keepdim=True)).t() @ tc) / Xf.shape[0]).norm() ** 2)
        z, pres, acts = model.forward_blocks(Xf, detach=False)
        mu0 = [a.mean(0).numpy().astype(float) for a in pres]
        sigma0 = [float(a.std(0).mean()) for a in pres]
        lam0 = [float((((h - h.mean(0, keepdim=True)).t() @ tc) / h.shape[0]).norm() ** 2) for h in acts]
        q = [float(l.weight.shape[1]) * float(l.weight.var()) for l in model.hidden]
        r = [float(l.weight.shape[0]) * float(l.weight.var()) for l in model.hidden]
        return Init(H0=float((Xf.mean(0) ** 2).sum()), mu0=mu0, sigma0=sigma0, zbar0=z.mean(0).numpy().astype(float),
                    q=q, r=r, Lam0=Lam0, lam0=lam0, widths=list(model.widths), C=C, tbar=tbar,
                    sigma_x2=float(Xf.var(0).sum() / Xf.shape[1]))

# ---------------- the reduced model ----------------
@dataclass
class ReducedConfig:
    """Integration settings. eta_hid/eta_out are per-step learning rates, g is the feedback scale."""
    eta_hid: float = 1e-3; eta_out: float = 1e-3; g: float = 1.0
    head: str = "sigmoid_bce"; steps: int = 3000; log_every: int = 10; substeps: int = 1
    bias_channel: float = 1.0       # 0 when the hidden biases are frozen
    drive_scale: float = 1.0        # 0 when the teaching signal is batch-centered (the common mode is removed)
    freeze_out: bool = False        # frozen head: the common mode never decays
    escape: bool = True             # integrate the Lambda cascade and the excess-loss variable
    sigma_seed: str = "measured"    # "measured": sigma_1(0) from the network; "data": q_1 tr Cov(x)/d_0
    exit_frac: float = 0.9          # exit when the modelled loss reaches exit_frac x prior loss
    onset_tol: float = 0.03         # onset when the modelled loss first enters the +-3% prior-loss band
    onset_min_len: int = 50         # ... and stays there this many steps (notes/PROTOCOL.md, cmc.analysis.plateau)

def simulate(init: Init, cfg: ReducedConfig, B: list):
    """Integrate the reduced model. B is the list of per-layer feedback matrices [d_l, C_out] (entries ~ N(0, g^2)).

    Returns a pandas DataFrame with one row per logged step: step, m, ebar_norm, loss_cm, excess, loss_model and,
    per layer, p_l*, chi_l*, H_l*, sig_l*, mu_spread_l*, cos_l*, kappa_l*, Lambda_l*, plus the constant columns
    onset_pred and exit_pred (-1 when the modelled trajectory never reaches them)."""
    import pandas as pd
    L = len(init.widths); C = init.C
    mu = [m.copy() for m in init.mu0]
    zbar = init.zbar0.copy().astype(float)
    G = np.zeros(L)                                   # accumulated local-DFA part of ||Lambda_l||_F
    dose = np.zeros(L)                                # online dimensionless dose s_l(t); kappa_l = g * dose
    excess = 0.0                                      # loss already removed by the input-dependent term
    zstar = prior_fixed_point(init.tbar, cfg.head, init.zbar0)
    offset = init.prior_loss - _loss_raw(zstar, init.tbar, cfg.head) if np.isfinite(init.prior_loss) else 0.0
    Lprior = _loss_raw(zstar, init.tbar, cfg.head) + offset
    ebar0 = mean_error(zbar, init.tbar, cfg.head); ehat0 = ebar0 / (np.linalg.norm(ebar0) + 1e-30)
    sig1_2 = init.sigma0[0] ** 2 if cfg.sigma_seed == "measured" else init.q[0] * init.sigma_x2
    dt = 1.0 / cfg.substeps
    rows = []; onset = None; exit_step = None; band_start = None

    def closure():
        """Bottom-up sweep: gates, mean-activity energies, the sigma cascade and the Lambda cascade."""
        gam, u, sig2, H = [], [], [], [init.H0]
        s2 = sig1_2
        for l in range(L):
            hb, ga, uu = gate_moments(mu[l], math.sqrt(s2))
            gam.append(ga); u.append(uu); sig2.append(s2); H.append(float((hb * hb).sum()))
            if l + 1 < L: s2 = init.q[l + 1] * uu.mean() * s2
        A = [math.sqrt(max(init.Lam0, 0.0))]
        for l in range(L):
            A.append(math.sqrt(max(init.r[l] * u[l].mean(), 0.0)) * A[l] + G[l])
        return gam, u, sig2, H, A

    wnum = np.zeros(L); wden = 0.0                    # proj-weighted drive and prior-clock integrals, for kappa_cf
    for it in range(cfg.steps + 1):
        state = closure()
        gam, u, sig2, H, A = state
        ebar = mean_error(zbar, init.tbar, cfg.head)
        loss_cm = _loss_raw(zbar, init.tbar, cfg.head) + offset
        loss_model = loss_cm - excess
        if it % cfg.log_every == 0 or it == cfg.steps:
            row = dict(step=it, m=float(_sigmoid(zbar).mean() if cfg.head in SIGMOID_HEADS else
                                        (_softmax(zbar).mean() if cfg.head == "softmax_ce" else zbar.mean())),
                       ebar_norm=float(np.linalg.norm(ebar)), loss_cm=loss_cm, excess=excess, loss_model=loss_model)
            for l in range(L):
                row[f"p_l{l+1}"] = participation(u[l]); row[f"chi_l{l+1}"] = float(u[l].mean())
                row[f"H_l{l+1}"] = H[l + 1]; row[f"sig_l{l+1}"] = math.sqrt(sig2[l])
                row[f"mu_spread_l{l+1}"] = float(mu[l].std())
                row[f"cos_l{l+1}"] = float(H[l + 1] / (H[l + 1] + init.widths[l] * u[l].mean() * sig2[l]))
                row[f"kappa_l{l+1}"] = float(cfg.g * dose[l]); row[f"Lambda_l{l+1}"] = float(A[l + 1] ** 2)
            rows.append(row)
        if onset is None:                                # the protocol asks for the band to be HELD, not just touched
            if abs(loss_model - Lprior) / Lprior < cfg.onset_tol:
                if band_start is None: band_start = it
                if it - band_start >= cfg.onset_min_len: onset = band_start
            else: band_start = None
        if exit_step is None and onset is not None and loss_model < cfg.exit_frac * Lprior: exit_step = it
        if exit_step is not None and loss_model < 0.5 * Lprior: break
        if it == cfg.steps or not np.isfinite(loss_model): break
        for k in range(cfg.substeps):
            if k:
                state = closure(); ebar = mean_error(zbar, init.tbar, cfg.head)
            gam, u, sig2, H, A = state
            proj = float(ebar @ ehat0); w = dt * proj
            wnum += w * (np.array(H[:L]) + cfg.bias_channel); wden += w * (H[L] + 1.0)
            for l in range(L):
                drive = H[l] + cfg.bias_channel
                mu[l] -= dt * cfg.eta_hid * cfg.drive_scale * (B[l] @ ebar) * gam[l] * drive
                dose[l] += dt * cfg.eta_hid * cfg.drive_scale * proj * drive
                if cfg.escape:                                   # Theory B eq. (13), the local DFA route
                    rate = float(u[l].mean()) / math.sqrt(max(participation(u[l]), 1e-12))
                    G[l] += dt * cfg.eta_hid * cfg.g * math.sqrt(init.widths[l]) * rate * A[l] ** 2 / math.sqrt(C)
            if cfg.escape: excess += dt * cfg.eta_out * A[L] ** 2
            if not cfg.freeze_out: zbar -= dt * cfg.eta_out * (H[L] + 1.0) * ebar
    df = pd.DataFrame(rows)
    df["onset_pred"] = onset if onset is not None else -1
    df["exit_pred"] = exit_step if exit_step is not None else -1
    ratio = wnum / (wden if abs(wden) > 1e-30 else 1e-30)   # proj-weighted <(H_{l-1}+bias)/(H_L+1)>, per the integral
    e0 = float(np.linalg.norm(ebar0)); sig_eff = sigma_prime_general(init.zbar0, zstar, ebar0)
    df.attrs["prior_loss"] = Lprior; df.attrs["onset"] = onset; df.attrs["exit"] = exit_step
    df.attrs["sigma_prime_eff"] = sig_eff
    if cfg.freeze_out and cfg.drive_scale:                   # the head never decays, so the dose has no finite limit
        df.attrs["kappa_cf"] = [float("inf")] * L
    else:
        df.attrs["kappa_cf"] = [cfg.drive_scale * kappa_closed_form(cfg.g, e0, r_out=cfg.eta_out / cfg.eta_hid,
                                                                    sig_eff=sig_eff, drive_ratio=ratio[l])
                                for l in range(L)]
    return df

def summarize(df) -> dict:
    """Collapse-depth, timing and dose summary of a reduced-model trajectory, in the same shape as cmc.analysis."""
    L = len([c for c in df.columns if c.startswith("p_l")])
    out = dict(prior_loss=df.attrs.get("prior_loss", float("nan")), onset=df.attrs.get("onset"),
               exit=df.attrs.get("exit"), loss_end=float(df.loss_model.iloc[-1]))
    for l in range(1, L + 1):
        out[f"minp_l{l}"] = float(df[f"p_l{l}"].min()); out[f"maxcos_l{l}"] = float(df[f"cos_l{l}"].max())
        out[f"kappa_l{l}"] = float(df[f"kappa_l{l}"].iloc[-1]); out[f"kappa_cf_l{l}"] = df.attrs["kappa_cf"][l - 1]
        out[f"H_l{l}_end"] = float(df[f"H_l{l}"].iloc[-1]); out[f"Lambda_l{l}_min"] = float(df[f"Lambda_l{l}"].min())
    return out

# ---------------- rebuilding a run's initial condition from its meta.json ----------------
def init_from_meta(meta: dict):
    """Rebuild a run's model, feedback and probe from its config (same seeds as cmc.run) and read the init off it.

    Returns (Init, ReducedConfig, B). Only the DFA-family rules ('dfa', 'aligned') have a common-mode direction
    that this model describes; other rules are rejected."""
    import torch
    from . import data as D
    from .models import MLP
    from .rules import make_feedback, n_outputs, prior_loss
    from .run import Config
    cfg = Config(**{k: v for k, v in meta.items() if k in Config.__dataclass_fields__})
    if cfg.optimizer != "sgd":
        raise ValueError("the reduced dynamics describe SGD, not adaptive optimizer steps")
    if cfg.rule not in ("dfa", "aligned"):
        raise ValueError(f"reduced model covers the DFA family only, got rule={cfg.rule!r}")
    if cfg.arch != "mlp":
        raise ValueError(f"reduced model covers the MLP only, got arch={cfg.arch!r}")
    if cfg.act != "tanh":
        raise ValueError(f"the gate quadrature is tanh-specific (sech^2/sech^4), got act={cfg.act!r}")
    if cfg.batchnorm:
        raise ValueError("batch normalization removes the presynaptic mean the reduced model is built on")
    ds = D.load(cfg.dataset, preprocess=cfg.preprocess, classes=(cfg.classes or None), flatten=True)
    C = ds["C"]; Cout = n_outputs(cfg, C)
    torch.manual_seed(cfg.seed)
    model = MLP(ds["Xtr"].shape[1], [cfg.width] * cfg.depth, Cout, act=cfg.act, batchnorm=bool(cfg.batchnorm))
    if cfg.out_bias_q > 0:
        with torch.no_grad():
            model.out.bias.fill_(0.0 if cfg.head == "softmax_ce" else math.log(cfg.out_bias_q / (1 - cfg.out_bias_q)))
    if getattr(cfg, "out_w_scale", 1.0) != 1.0:
        with torch.no_grad(): model.out.weight.mul_(cfg.out_w_scale)
    fb = make_feedback(model, Cout, cfg, "cpu")
    sampler = D.Sampler(ds["ytr"], cfg.batch, seed=cfg.seed, p0=cfg.p0)
    psamp = D.Sampler(ds["yval"], cfg.probe_n, seed=cfg.seed + 999, p0=cfg.p0); pidx = psamp.draw(cfg.probe_n)
    pi = np.asarray(meta.get("prior") or sampler.prior(C), float)
    consistent = pi.shape[0] == C
    if not consistent: pi = sampler.prior(C)
    init = extract_init(model, ds["Xval"][pidx], ds["yval"][pidx], C, head=cfg.head, target_scale=cfg.target_scale,
                        target_center=bool(cfg.target_center), extra_bits=cfg.extra_bits, seed=cfg.seed, prior=pi)
    init.prior_loss = float(meta["prior_loss"]) if (consistent and "prior_loss" in meta) else prior_loss(pi, cfg, C)
    rc = ReducedConfig(eta_hid=cfg.lr * cfg.hid_lr_mult, eta_out=cfg.lr * cfg.out_lr_mult, g=cfg.fb_scale,
                       head=cfg.head, steps=int(meta.get("final_step", cfg.steps)),
                       bias_channel=0.0 if cfg.freeze_bias else 1.0,
                       drive_scale=0.0 if cfg.center in ("e", "delta", "gated") else 1.0,
                       freeze_out=bool(cfg.freeze_out))
    return init, rc, [b.detach().numpy().astype(float) for b in fb["B"]]

def from_run(meta_path: str, steps: int = 0, **overrides):
    """Integrate the reduced model for the run described by meta_path. Returns (DataFrame, Init, ReducedConfig)."""
    meta = json.load(open(meta_path))
    init, rc, B = init_from_meta(meta)
    if steps: rc.steps = steps
    for k, v in overrides.items(): setattr(rc, k, v)
    return simulate(init, rc, B), init, rc

# ---------------- comparison figure ----------------
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
LC = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cd6", "#c02f6a", "#0f8a8a"]

def _style():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
    return plt

def _label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(3, 0), textcoords="offset points", color=color, fontsize=7.5, va="center")

def compare_figure(model_df, meas_df, prior_loss_value: float, out_png: str, title: str = ""):
    """Overlay measured (solid) and modelled (dashed) trajectories; direct labels, no legend boxes.

    The measured frame is clipped to the common x range first: the modelled trajectory stops once it has escaped,
    and leaving the rest of the measured run in the artist would autoscale the y axes (the loss and the log
    label-covariance panels in particular) to data that the x limits hide."""
    plt = _style()
    L = len([c for c in model_df.columns if c.startswith("p_l")])
    xmax = float(min(model_df.step.max(), meas_df.step.max()))
    meas_df = meas_df[meas_df.step <= xmax]
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.0))
    panels = [("p_l", "Gate participation", (0, 1.03), False), ("cos_l", "Mean pairwise cosine", (0, 1.03), False),
              ("Lambda_l", "Label-covariance energy", None, True)]
    sub = "solid: measured   dashed: reduced model (no fitted parameters)"
    a = ax[0]
    a.plot(meas_df.step, meas_df.probe_loss, color=LC[0], lw=2.0)
    a.plot(model_df.step, model_df.loss_model, color=LC[1], lw=2.0, ls=(0, (3, 2)))
    a.axhline(prior_loss_value, color=INK2, lw=0.8, ls=(0, (4, 3)))
    _label(a, xmax, float(meas_df.probe_loss.iloc[-1]), "measured", LC[0])
    _label(a, float(model_df.step.iloc[-1]), float(model_df.loss_model.iloc[-1]), "model", LC[1])
    a.text(xmax, prior_loss_value, "  prior loss", ha="right", va="bottom", fontsize=7.5, color=INK2)
    a.set_ylabel("Probe loss"); a.set_title("Loss: prior learning, plateau, escape", loc="left", fontsize=9)
    for j, (pref, ylab, ylim, logy) in enumerate(panels):
        a = ax[j + 1]
        mcol = {"p_l": "p_l", "cos_l": "cos_l", "Lambda_l": "lam_l"}[pref]
        for l in range(L):
            if f"{mcol}{l+1}" in meas_df: a.plot(meas_df.step, meas_df[f"{mcol}{l+1}"], color=LC[l], lw=2.0)
            a.plot(model_df.step, model_df[f"{pref}{l+1}"], color=LC[l], lw=2.0, ls=(0, (3, 2)))
            _label(a, float(model_df.step.iloc[-1]), float(model_df[f"{pref}{l+1}"].iloc[-1]), f"L{l+1}", LC[l])
        if logy: a.set_yscale("log")
        if ylim: a.set_ylim(*ylim)
        a.set_ylabel(ylab); a.set_title(ylab, loc="left", fontsize=9)
    for a in ax:
        a.set_xlim(0, xmax * 1.16); a.set_xlabel("Training step")
        a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle((title + "\n" if title else "") + sub, x=0.008, ha="left", fontsize=8.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out_png, dpi=170); plt.close(fig)
    return out_png

# ---------------- CLI ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Reduced common-mode collapse model with the covariance-cascade escape.")
    ap.add_argument("--from_run", required=True, help="path to results/<exp>/<tag>/seed<S>_fb<F>.meta.json")
    ap.add_argument("--steps", type=int, default=0, help="override the number of integrated steps")
    ap.add_argument("--substeps", type=int, default=1, help="Euler sub-steps per SGD step")
    ap.add_argument("--no_escape", action="store_true", help="collapse-only model (no Lambda cascade, no exit)")
    ap.add_argument("--compare", action="store_true", help="also write reduced_<stem>.png overlaying the measured run")
    ap.add_argument("--out_dir", default="", help="output directory (default: alongside the run)")
    args = ap.parse_args(argv)
    meta_path = os.path.abspath(args.from_run)
    df, init, rc = from_run(meta_path, steps=args.steps, substeps=args.substeps, escape=not args.no_escape)
    stem = os.path.basename(meta_path).replace(".meta.json", "")
    out_dir = args.out_dir or os.path.dirname(meta_path); os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"reduced_{stem}.csv"); df.to_csv(out_csv, index=False)
    L = len(init.widths); s = summarize(df)
    print(f"wrote {out_csv}  ({len(df)} rows, steps<={int(df.step.max())})")
    print(f"prior loss {s['prior_loss']:.4f}   onset {s['onset']}   exit {s['exit']}   "
          f"sigma'_eff {df.attrs['sigma_prime_eff']:.4f}")
    print("kappa_l closed form: " + " ".join(f"L{l+1}={s[f'kappa_cf_l{l+1}']:.2f}" for l in range(L)))
    print("kappa_l integrated:  " + " ".join(f"L{l+1}={s[f'kappa_l{l+1}']:.2f}" for l in range(L)))
    print("min participation:   " + " ".join(f"L{l+1}={s[f'minp_l{l+1}']:.3f}" for l in range(L)))
    print("peak cosine:         " + " ".join(f"L{l+1}={s[f'maxcos_l{l+1}']:.3f}" for l in range(L)))
    print("Lambda_l(0) model / measured: " + " ".join(
        f"L{l+1}={df[f'Lambda_l{l+1}'].iloc[0]:.3f}/{init.lam0[l]:.3f}" for l in range(L)))
    if args.compare:
        import pandas as pd
        meas = pd.read_csv(meta_path.replace(".meta.json", ".csv"))
        png = compare_figure(df, meas, df.attrs["prior_loss"], os.path.join(out_dir, f"reduced_{stem}.png"),
                             title=f"Reduced model vs measured run: {os.path.relpath(meta_path)}")
        print(f"wrote {png}")
    return df

if __name__ == "__main__":
    main()
