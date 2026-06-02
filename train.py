#!/usr/bin/env python3
"""
DFA-STALL — Self-contained DFA stall experiment on MNIST.

Model   : 3-layer × 300-unit tanh MLP, Direct Feedback Alignment.
Dataset : MNIST (from ./data/).
Output  : metrics.csv + 3 figures in ./figures/.

Order parameters tracked at every training step
------------------------------------------------
  gate_participation_lN   p_ℓ(t) = (Σ u_i)² / (d · Σ u_i²),  u_i = E[φ'(a_i)²]
  gate_strength_sq_lN     G²_ℓ(t) = (1/d) Σ u_i
  saturation_frac_lN      S_ℓ(t)  = fraction of (unit,sample) with |φ'(a)| < 0.01
  feedback_signal_norm_lN ||D_ℓ B_ℓ e||_F / sqrt(B·d)
  angular_update_lN       R_ℓ(t)  = Σ_i ||(I-v_iv_iᵀ) G_i||² / (||w_i||²+ε)
  feature_movement_lN     F_ℓ(t)  = Σ_s E[||h_ℓ(t)-h_ℓ(t-1)||²] / E[||h_ℓ(0)||²]
  weight_alignment_lN     cos(M_ℓ, B_ℓ)  (effective-map to feedback alignment)
  grad_alignment          cos(g_DFA, g_BP) concatenated across hidden layers
  dfa_loss / bp_loss      training loss for both modes

Figures
-------
  fig1_training_curves.png   : loss + weight alignment + gradient alignment
  fig2_order_params.png      : 6 order-parameter panels
  fig3_phase_portraits.png   : scatter phase portraits (stall = orange squares)

Usage
-----
  python train.py                        # defaults: 3000 steps, CPU
  python train.py --device cuda          # GPU
  python train.py --total-steps 5000 --seed 1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = Path(__file__).resolve().parent


# ── constants ─────────────────────────────────────────────────────────────────

STALL_COLOR = "#e8501a"          # vivid orange-red used for stall markers
LAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ── model ─────────────────────────────────────────────────────────────────────

class TanhMLP(nn.Module):
    """3-layer tanh MLP that caches pre-activations and activations."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        dims = [input_dim, hidden, hidden, hidden, output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.preacts: List[torch.Tensor] = []
        self.acts: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.preacts, self.acts = [], []
        h = x
        for layer in self.layers[:-1]:
            a = layer(h)
            h = torch.tanh(a)
            self.preacts.append(a)
            self.acts.append(h)
        out = self.layers[-1](h)
        return torch.sigmoid(out)


# ── loss ──────────────────────────────────────────────────────────────────────

def binary_log_loss(targets: torch.Tensor, preds: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    preds = preds.clamp(eps, 1 - eps)
    return -(targets * preds.log() + (1 - targets) * (1 - preds).log()).sum(1).mean()


def to_one_hot(labels: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(labels.long(), n).float()


# ── data ──────────────────────────────────────────────────────────────────────

def load_mnist(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision import datasets
    tr = datasets.MNIST(root=str(data_dir), train=True,  download=True)
    te = datasets.MNIST(root=str(data_dir), train=False, download=True)
    def flat(d):
        return d.data.float().div(255.0).view(-1, 784)
    return (flat(tr), torch.as_tensor(tr.targets, dtype=torch.long),
            flat(te),  torch.as_tensor(te.targets, dtype=torch.long))


# ── helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def effective_maps(model: TanhMLP) -> List[torch.Tensor]:
    """M_ℓ = (W_L … W_{ℓ+1})ᵀ  shape [h_ℓ, C]."""
    n = len(model.layers) - 1
    maps = [None] * n
    eff = model.layers[-1].weight.detach().t()
    maps[-1] = eff
    for i in range(n - 2, -1, -1):
        eff = model.layers[i + 1].weight.detach().t() @ eff
        maps[i] = eff
    return maps


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.reshape(-1), b.reshape(-1)
    d = (af.norm() * bf.norm()).item()
    return float(torch.dot(af, bf).item() / d) if d > 1e-12 else 0.0


@torch.no_grad()
def gate_stats(g_prime: torch.Tensor) -> Dict[str, float]:
    """g_prime: [B, d].  Returns gate participation, strength, saturation,
    teacher_reff and gate_cv."""
    u = g_prime.pow(2).mean(0).clamp_min(1e-12)          # per-unit squared activity
    u_sum = u.sum()
    participation = u_sum.pow(2) / (u.numel() * u.pow(2).sum().clamp_min(1e-12))
    strength_sq   = u.mean()
    sat           = (g_prime.abs() < 0.01).float().mean()
    # teacher_reff = exp(H(p_i))  — entropy-based effective participation
    p = u / u_sum
    teacher_reff = float(torch.exp(-(p * p.log()).sum()).item())
    # gate_cv = std/mean of per-unit averaged gate value
    gp_unit = g_prime.mean(0)
    gate_cv = float((gp_unit.std() / (gp_unit.mean().clamp_min(1e-12))).item())
    return {
        "gate_participation": float(participation.item()),
        "gate_strength_sq":   float(strength_sq.item()),
        "saturation_frac":    float(sat.item()),
        "teacher_reff":       teacher_reff,
        "gate_cv":            gate_cv,
    }


@torch.no_grad()
def _masked_response(
    model: TanhMLP,
    layer_idx: int,
    gw: torch.Tensor,
    gb: Optional[torch.Tensor],
    mask: torch.Tensor,
    baseline_loss: float,
    x_probe: torch.Tensor,
    t_probe: torch.Tensor,
    eps: float,
) -> float:
    """Virtual-step loss response for masked rows of one layer's DFA gradient."""
    layer = model.layers[layer_idx]
    gw_m = torch.zeros_like(gw); gw_m[mask] = gw[mask]
    gb_m = None
    if gb is not None and layer.bias is not None:
        gb_m = torch.zeros_like(gb); gb_m[mask] = gb[mask]
    norm = float((gw_m.pow(2).sum() +
                  (gb_m.pow(2).sum() if gb_m is not None else 0)).sqrt().item())
    if norm < 1e-12:
        return float("nan")
    layer.weight.add_(gw_m, alpha=-eps)
    if gb_m is not None:
        layer.bias.add_(gb_m, alpha=-eps)
    try:
        loss_after = float(binary_log_loss(t_probe, model(x_probe)).item())
        return (baseline_loss - loss_after) / (eps * norm)
    finally:
        layer.weight.add_(gw_m, alpha=eps)
        if gb_m is not None:
            layer.bias.add_(gb_m, alpha=eps)


@torch.no_grad()
def angular_update(W: torch.Tensor, G: torch.Tensor, eps: float = 1e-8) -> float:
    """R_ℓ = Σ_i ||(I - v_i v_iᵀ) G_i||² / (||w_i||² + ε)."""
    w2   = W.pow(2).sum(1).clamp_min(eps)
    v    = W / w2.sqrt().unsqueeze(1)
    perp = G - (G * v).sum(1, keepdim=True) * v
    return float((perp.pow(2).sum(1) / w2).sum().item())


def detect_stall(loss: np.ndarray, smooth_w: int = 100) -> Tuple[int, int]:
    n  = len(loss)
    w  = min(smooth_w, n)
    sm = np.convolve(loss, np.ones(w) / w, mode="same")
    vel = -np.gradient(sm)
    skip = min(50, n // 4)
    early = vel[skip:skip + min(200, n - skip)]
    if not len(early) or np.nanmax(early) <= 0:
        return n // 4, 3 * n // 4
    thresh = 0.05 * np.nanmax(early)
    idxs = np.where((vel < thresh) & (np.arange(n) >= skip))[0]
    if not len(idxs):
        return n // 4, 3 * n // 4
    s = int(idxs[0])
    rec = np.where((vel > thresh) & (np.arange(n) > s + min(50, n // 10)))[0]
    e = int(rec[0]) if len(rec) else n - 1
    return s, e


# ── training loop ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # data
    X_tr, y_tr, X_te, y_te = load_mnist(HERE / "data")
    print(f"MNIST loaded  train={tuple(X_tr.shape)}  test={tuple(X_te.shape)}")

    # model
    n_classes = 10
    model = TanhMLP(784, 300, n_classes, args.seed).to(device)
    feedback = [torch.randn(300, n_classes, device=device) for _ in range(3)]
    opt_dfa  = torch.optim.SGD(model.parameters(), lr=args.lr)

    # fixed probe set for feature movement, channel responses, val_loss
    PROBE_N   = 1024
    probe_x   = X_te[:PROBE_N].to(device).float()
    probe_t   = to_one_hot(y_te[:PROBE_N], n_classes).to(device)
    RESP_EPS  = 1e-3
    RESP_FRAC = 0.20
    VAL_EVERY = 10
    RESP_EVERY = 50

    with torch.no_grad():
        model(probe_x)
        h_prev   = [h.clone() for h in model.acts]
        h0_norm2 = [float(h.pow(2).sum(1).mean().item()) + 1e-8 for h in h_prev]
        feat_acc = np.zeros(3, dtype=np.float64)

    rng  = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    rows: List[Dict] = []
    n    = len(X_tr)

    print(f"Training {args.total_steps} steps  device={device}  lr={args.lr}")

    for step in range(1, args.total_steps + 1):
        idx     = torch.randint(0, n, (args.batch_size,), generator=rng)
        xb      = X_tr[idx].to(device).float()
        targets = to_one_hot(y_tr[idx], n_classes).to(device)

        # ── DFA forward ───────────────────────────────────────────────────────
        with torch.no_grad():
            preds = model(xb)
            error = preds - targets                       # [B, C]
            dfa_loss = binary_log_loss(targets, preds).item()

        dfa_grads, teachings, g_primes = [], [], []
        with torch.no_grad():
            for li, layer in enumerate(model.layers[:-1]):
                h_prev_l = xb if li == 0 else model.acts[li - 1]
                direct   = error @ feedback[li].t()       # [B, d] = B_l e
                gp       = 1.0 - model.preacts[li].tanh().pow(2)  # [B, d]
                da       = direct * gp                    # teaching signal
                gw       = da.t() @ h_prev_l / xb.size(0)
                gb       = da.mean(0)
                dfa_grads.append((gw, gb))
                teachings.append(da)
                g_primes.append(gp)
            # output layer
            hl = model.acts[-1]
            gw = error.t() @ hl / xb.size(0)
            gb = error.mean(0)
            dfa_grads.append((gw, gb))

        # ── BP grad for alignment ─────────────────────────────────────────────
        model.zero_grad(set_to_none=True)
        bp_preds = model(xb)
        bp_loss_val = binary_log_loss(targets, bp_preds)
        bp_loss_val.backward()
        bp_grads_hw = [model.layers[li].weight.grad.detach().clone() for li in range(3)]

        dfa_cat = torch.cat([dfa_grads[li][0].reshape(-1) for li in range(3)])
        bp_cat  = torch.cat([g.reshape(-1) for g in bp_grads_hw])
        grad_align = cosine_flat(dfa_cat, bp_cat)
        # per-layer gradient alignment
        layer_grad_align = [cosine_flat(dfa_grads[li][0], bp_grads_hw[li]) for li in range(3)]

        # ── apply DFA update ──────────────────────────────────────────────────
        opt_dfa.zero_grad(set_to_none=True)
        for layer, (gw, gb) in zip(model.layers, dfa_grads):
            layer.weight.grad = gw
            layer.bias.grad   = gb
        opt_dfa.step()

        # ── order parameters ──────────────────────────────────────────────────
        eff_maps = effective_maps(model)
        row: Dict = {
            "step":           step,
            "dfa_loss":       dfa_loss,
            "bp_loss":        bp_loss_val.item(),
            "grad_alignment": grad_align,
            "val_loss":       np.nan,
        }

        for li in range(3):
            lid   = li + 1
            t     = teachings[li]
            gp    = g_primes[li]
            W_l   = model.layers[li].weight.detach()
            G_l   = dfa_grads[li][0].detach()
            B_l   = feedback[li]
            eff   = eff_maps[li]

            gs = gate_stats(gp)
            row[f"gate_participation_l{lid}"]      = gs["gate_participation"]
            row[f"gate_participation_unit_l{lid}"] = gs["gate_participation"]   # alias
            row[f"gate_strength_sq_l{lid}"]        = gs["gate_strength_sq"]
            row[f"saturation_frac_l{lid}"]         = gs["saturation_frac"]
            row[f"gate_saturation_frac_low_g_l{lid}"] = gs["saturation_frac"]  # alias
            row[f"teacher_reff_l{lid}"]            = gs["teacher_reff"]
            row[f"gate_cv_l{lid}"]                 = gs["gate_cv"]

            fb_norm = float(t.norm().item() / math.sqrt(float(t.numel())))
            row[f"feedback_signal_norm_l{lid}"] = fb_norm

            row[f"angular_update_l{lid}"] = angular_update(W_l, G_l)

            # weight alignment
            dots = (eff * B_l).sum(1)
            wa   = float((dots / (eff.norm(1).clamp_min(1e-12) * B_l.norm(1).clamp_min(1e-12))).mean().item())
            row[f"weight_alignment_l{lid}"]       = wa
            row[f"param_grad_alignment_l{lid}"]   = layer_grad_align[li]

            row[f"feature_movement_l{lid}"]       = float(feat_acc[li])
            row[f"feature_path_l{lid}"]           = float(feat_acc[li])  # alias

            row[f"selected_loss_response_l{lid}"] = np.nan
            row[f"residual_loss_response_l{lid}"] = np.nan

        # val_loss every VAL_EVERY steps
        if step % VAL_EVERY == 0 or step == 1:
            with torch.no_grad():
                row["val_loss"] = float(binary_log_loss(probe_t, model(probe_x)).item())

        # channel responses every RESP_EVERY steps
        if step % RESP_EVERY == 0 or step == 1:
            with torch.no_grad():
                baseline = float(binary_log_loss(probe_t, model(probe_x)).item())
            for li in range(3):
                lid = li + 1
                unit_e = teachings[li].pow(2).mean(0)
                k = max(1, int(round(RESP_FRAC * unit_e.numel())))
                top_idx = torch.topk(unit_e, k=k, largest=True).indices
                sel_mask = torch.zeros(unit_e.numel(), dtype=torch.bool, device=device)
                sel_mask[top_idx] = True
                row[f"selected_loss_response_l{lid}"] = _masked_response(
                    model, li, dfa_grads[li][0], dfa_grads[li][1],
                    sel_mask, baseline, probe_x, probe_t, RESP_EPS)
                row[f"residual_loss_response_l{lid}"] = _masked_response(
                    model, li, dfa_grads[li][0], dfa_grads[li][1],
                    ~sel_mask, baseline, probe_x, probe_t, RESP_EPS)

        rows.append(row)

        # ── feature movement: update every step from probe pass ───────────────
        with torch.no_grad():
            model(probe_x)
            h_curr = model.acts
            for li in range(3):
                d2 = (h_curr[li] - h_prev[li]).pow(2).sum(1).mean().item()
                feat_acc[li] += d2 / h0_norm2[li]
            h_prev = [h.clone() for h in h_curr]

        if step % 500 == 0:
            print(f"  step {step:5d}/{args.total_steps}"
                  f"  loss={dfa_loss:.3f}  ga={grad_align:.3f}"
                  f"  wa_l2={row['weight_alignment_l2']:.3f}"
                  f"  p_l2={row['gate_participation_l2']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "metrics.csv", index=False)
    print(f"metrics.csv saved  ({len(df)} rows)")
    return df


# ── style ─────────────────────────────────────────────────────────────────────

def _style() -> None:
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         8,
        "axes.labelsize":    8,
        "axes.titlesize":    8.5,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "axes.linewidth":    0.75,
        "lines.linewidth":   1.35,
    })


def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _shade(ax: plt.Axes, steps: np.ndarray, s: int, e: int) -> None:
    ax.axvspan(steps[s], steps[min(e, len(steps)-1)],
               color=STALL_COLOR, alpha=0.12, lw=0, zorder=0)


def _sm(x: np.ndarray, a: float = 0.06) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


# ── figures ───────────────────────────────────────────────────────────────────

def fig1_training_curves(df: pd.DataFrame, stall_s: int, stall_e: int) -> None:
    """Loss + weight alignment + gradient alignment with stall shaded."""
    _style()
    steps  = df["step"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 6.0), sharex=True)
    fig.subplots_adjust(hspace=0.10, left=0.13, right=0.97, top=0.94, bottom=0.09)

    # (a) loss
    ax = axes[0]
    _shade(ax, steps, stall_s, stall_e)
    ax.semilogy(steps, _sm(df["dfa_loss"].to_numpy()),
                color="#1f4e79", lw=1.5, label="DFA")
    ax.semilogy(steps, _sm(df["bp_loss"].to_numpy()),
                color="#d62728", lw=1.1, ls="--", label="BP")
    ax.set_ylabel("Training loss")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.10, "stall", transform=ax.transAxes,
            color=STALL_COLOR, fontsize=7, va="bottom")
    ax.text(0.02, 0.97, "a", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
    _clean(ax)

    # (b) weight alignment per layer
    ax = axes[1]
    _shade(ax, steps, stall_s, stall_e)
    for li in range(3):
        ax.plot(steps, _sm(df[f"weight_alignment_l{li+1}"].to_numpy()),
                color=LAYER_COLORS[li], lw=1.3, label=f"L{li+1}")
    ax.axhline(0, color="0.7", lw=0.6, ls=":")
    ax.set_ylabel("Weight alignment\n" r"$\cos(M_\ell, B_\ell)$")
    ax.legend(frameon=False, loc="lower right", ncol=3)
    ax.text(0.02, 0.97, "b", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
    _clean(ax)

    # (c) gradient alignment
    ax = axes[2]
    _shade(ax, steps, stall_s, stall_e)
    ax.plot(steps, _sm(df["grad_alignment"].to_numpy()),
            color="#5c3317", lw=1.3)
    ax.axhline(0, color="0.7", lw=0.6, ls=":")
    ax.set_ylabel("Gradient alignment\n" r"$\cos(g_{\rm DFA}, g_{\rm BP})$")
    ax.set_xlabel("Training step")
    ax.text(0.02, 0.97, "c", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
    _clean(ax)

    fig.savefig(HERE / "figures" / "fig1_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig1_training_curves.png")


def fig2_order_params(df: pd.DataFrame, stall_s: int, stall_e: int) -> None:
    """6-panel order parameter figure."""
    _style()
    steps = df["step"].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.12, wspace=0.35,
                        left=0.10, right=0.97, top=0.95, bottom=0.08)

    specs = [
        ("gate_participation_l",  "Gate participation\n" r"$p_\ell(t)$",           None),
        ("gate_strength_sq_l",    "Gate strength\n"      r"$G^2_\ell(t)$",         None),
        ("saturation_frac_l",     "Saturation fraction\n" r"$S_\ell(t)$",           None),
        ("feedback_signal_norm_l","Feedback signal norm\n" r"$\|\delta_\ell\|(t)$", "log"),
        ("angular_update_l",      "Angular update\n"      r"$R_\ell(t)$",           "log"),
        ("feature_movement_l",    "Feature movement\n"    r"$F_\ell(t)$",           None),
    ]
    panels = "abcdef"

    for (prefix, ylabel, scale), ax, lbl in zip(specs, axes.flat, panels):
        _shade(ax, steps, stall_s, stall_e)
        for li in range(3):
            col = f"{prefix}{li+1}"
            ax.plot(steps, _sm(df[col].to_numpy()),
                    color=LAYER_COLORS[li], lw=1.2, label=f"L{li+1}")
        if scale == "log":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.legend(frameon=False, fontsize=6, ncol=3, loc="best")
        ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        _clean(ax)

    for ax in axes[-1]:
        ax.set_xlabel("Training step")

    fig.savefig(HERE / "figures" / "fig2_order_params.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig2_order_params.png")


def fig3_phase_portraits(df: pd.DataFrame, stall_s: int, stall_e: int) -> None:
    """Phase-portrait scatter: viridis circles outside stall, orange squares inside."""
    _style()
    steps   = df["step"].to_numpy()
    n_steps = len(steps)

    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=steps.min(), vmax=steps.max())
    colors = cmap(norm(steps))

    stall_mask = (steps >= steps[stall_s]) & (steps <= steps[min(stall_e, n_steps-1)])

    def scatter_stall(ax, x, y):
        m = stall_mask
        ax.scatter(x[~m], y[~m], c=colors[~m], s=6, marker="o", linewidths=0, alpha=0.85)
        ax.scatter(x[m],  y[m],  color=STALL_COLOR, s=22, marker="s",
                   linewidths=0, alpha=0.90, zorder=4)
        # start / end markers
        ax.scatter(x[0],  y[0],  marker="x", color="#222", s=22, linewidths=1.0, zorder=5)
        ax.scatter(x[-1], y[-1], marker="o", facecolor="none",
                   edgecolor="#222", s=22, linewidths=1.0, zorder=5)

    # use middle layer (L2) for clarity
    pa  = _sm(df["gate_participation_l2"].to_numpy())
    ga  = _sm(df["grad_alignment"].to_numpy())
    los = _sm(df["dfa_loss"].to_numpy())
    fm  = _sm(df["feature_movement_l2"].to_numpy())
    wa  = _sm(df["weight_alignment_l2"].to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.22, top=0.91, wspace=0.52)

    # (a) gate participation vs gradient alignment
    scatter_stall(axes[0], pa, ga)
    axes[0].set_xlabel(r"Gate participation $p_\ell(t)$")
    axes[0].set_ylabel("Gradient alignment")
    axes[0].text(0.04, 0.96, "a", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    _clean(axes[0])

    # (b) feature movement vs gradient alignment
    scatter_stall(axes[1], fm, ga)
    axes[1].set_xlabel(r"Feature movement $F_\ell(t)$")
    axes[1].set_ylabel("Gradient alignment")
    axes[1].text(0.04, 0.96, "b", transform=axes[1].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    _clean(axes[1])

    # (c) weight alignment vs gradient alignment
    scatter_stall(axes[2], wa, ga)
    axes[2].set_xlabel(r"Weight alignment $\cos(M_\ell, B_\ell)$")
    axes[2].set_ylabel("Gradient alignment")
    axes[2].text(0.04, 0.96, "c", transform=axes[2].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    _clean(axes[2])

    # shared colorbar + legend
    sm2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cax = fig.add_axes([0.905, 0.22, 0.013, 0.69])
    fig.colorbar(sm2, cax=cax).set_label("Training step", fontsize=7)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4",
               markersize=5, label="non-stall"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=STALL_COLOR,
               markersize=7, label="stall"),
    ]
    axes[0].legend(handles=legend_handles, frameon=False, fontsize=6,
                   loc="lower right", handletextpad=0.3)

    fig.savefig(HERE / "figures" / "fig3_phase_portraits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig3_phase_portraits.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DFA-STALL self-contained experiment")
    parser.add_argument("--total-steps", type=int,   default=3000)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable — using CPU.")
        args.device = "cpu"

    df = run(args)

    print("Generating figures …")
    stall_s, stall_e = detect_stall(df["dfa_loss"].to_numpy())
    print(f"  stall detected: steps {df['step'].iloc[stall_s]}–{df['step'].iloc[stall_e]}")

    fig1_training_curves(df, stall_s, stall_e)
    fig2_order_params(df, stall_s, stall_e)
    fig3_phase_portraits(df, stall_s, stall_e)

    print(f"\nAll outputs in: {HERE}")
    print(f"  metrics.csv    ({len(df)} rows × {len(df.columns)} cols)")
    print(f"  figures/fig1_training_curves.png")
    print(f"  figures/fig2_order_params.png")
    print(f"  figures/fig3_phase_portraits.png")


if __name__ == "__main__":
    main()
