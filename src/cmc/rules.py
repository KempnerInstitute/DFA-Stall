"""Heads, credit-assignment rules (BP, DFA, FA) and interventions. All teaching signals use the convention
e = dL/dz per sample with the loss averaged over the batch, so hidden 'gradients' are (1/B) sum_x delta(x) h(x)^T."""
from __future__ import annotations
import math, torch, torch.nn.functional as F
from .models import act_deriv

# ---------------- heads ----------------
def head_forward(z, y, cfg, C):
    """Returns per-sample loss vector, predicted probabilities/outputs p, error e = dL/dz, target t."""
    if cfg.head == "sigmoid_bce":
        t = F.one_hot(y, C).to(z.dtype); p = torch.sigmoid(z)
        loss = F.binary_cross_entropy_with_logits(z, t, reduction="none").sum(1); e = p - t
    elif cfg.head == "softmax_ce":
        t = F.one_hot(y, C).to(z.dtype); p = torch.softmax(z, 1)
        loss = -(t * torch.log_softmax(z, 1)).sum(1); e = p - t
    elif cfg.head == "mse":
        t = F.one_hot(y, C).to(z.dtype) * cfg.target_scale
        if cfg.target_center: t = t - cfg.target_scale / C
        p = z; loss = 0.5 * ((z - t) ** 2).sum(1); e = z - t
    elif cfg.head == "multilabel":
        t = multilabel_targets(y, C, cfg.extra_bits, cfg.seed); p = torch.sigmoid(z)
        loss = F.binary_cross_entropy_with_logits(z, t, reduction="none").sum(1); e = p - t
    else:
        raise ValueError(cfg.head)
    return loss, p, e, t

_ML_CACHE = {}
def multilabel_targets(y, C, extra_bits, seed):
    key = (C, extra_bits, seed)
    if key not in _ML_CACHE:
        g = torch.Generator().manual_seed(seed + 777)
        table = torch.zeros(C, C + extra_bits); table[:, :C] = torch.eye(C)
        for b in range(extra_bits):                       # each extra bit is on for 3 random classes
            on = torch.randperm(C, generator=g)[:3]; table[on, C + b] = 1.0
        _ML_CACHE[key] = table
    return _ML_CACHE[key].to(y.device)[y]

def n_outputs(cfg, C): return C + cfg.extra_bits if cfg.head == "multilabel" else C

def prior_loss(pi, cfg, C):
    """Loss of the constant predictor at the label prior pi (numpy array over classes)."""
    import numpy as np
    pi = np.clip(np.asarray(pi, float), 1e-12, 1 - 1e-12)
    if cfg.head == "sigmoid_bce": return float(-(pi * np.log(pi) + (1 - pi) * np.log(1 - pi)).sum())
    if cfg.head == "softmax_ce": return float(-(pi * np.log(pi)).sum())
    if cfg.head == "mse": return float(0.5 * cfg.target_scale ** 2 * (pi * (1 - pi)).sum())
    if cfg.head == "multilabel":
        table = multilabel_targets(torch.arange(C), C, cfg.extra_bits, cfg.seed).numpy(); q = np.clip(pi @ table, 1e-12, 1 - 1e-12)
        return float(-(q * np.log(q) + (1 - q) * np.log(1 - q)).sum())
    raise ValueError(cfg.head)

# ---------------- feedback matrices ----------------
def make_feedback(model, C_out, cfg, device):
    """DFA: B_l in R^{d_l x C_out}; FA: R_out in R^{C_out x d_L} and R_l in R^{d_l x d_{l-1}} (l>=1)."""
    g = torch.Generator(device="cpu").manual_seed(10_000 + cfg.fb_seed)
    fb = {}
    if cfg.rule in ("dfa", "aligned"):
        if getattr(cfg, "fb_dist", "normal") == "uniform_fanout":
            # U(-1/sqrt(d_l), 1/sqrt(d_l)): each output error fans out to the d_l units of layer l (Nokland, 2016)
            fb["B"] = [(cfg.fb_scale * (2 * torch.rand(w, C_out, generator=g) - 1) / math.sqrt(w)).to(device)
                       for w in model.widths]
        else:
            fb["B"] = [(cfg.fb_scale * torch.randn(w, C_out, generator=g)).to(device) for w in model.widths]
        if getattr(cfg, "fb_row_norm", 0):
            fb["B"] = [F.normalize(b, dim=1) * (b.norm() / math.sqrt(b.shape[0])) for b in fb["B"]]
        if cfg.rule == "aligned":
            maps = model.downstream_maps()
            fb["B"] = [(m * (b.norm() / m.norm())).to(device) for m, b in zip(maps, fb["B"])]
    elif cfg.rule == "fa":
        widths = model.widths
        fb["R_out"] = (cfg.fb_scale * torch.randn(C_out, widths[-1], generator=g)).to(device)
        fb["R"] = [None] + [(torch.randn(widths[l], widths[l - 1], generator=g) / math.sqrt(widths[l])).to(device) for l in range(1, len(widths))]
    return fb

# ---------------- interventions on the teaching signal ----------------
class ErrorWhitener:
    """Error-side conditioner (Sigma_delta + lambda I)^-1 with an EMA covariance, norm-matched (the nDFA E-operator)."""
    def __init__(self, dims, lam_rel=0.1, beta=0.99, device="cpu"):
        self.S = [torch.eye(d, device=device) * 0.0 for d in dims]; self.lam_rel, self.beta, self.t = lam_rel, beta, 0
    def apply(self, l, delta):
        Bn = delta.shape[0]; cov = delta.t() @ delta / Bn
        self.S[l] = self.beta * self.S[l] + (1 - self.beta) * cov
        S = self.S[l] / (1 - self.beta ** (self.t + 1)); lam = self.lam_rel * S.trace() / S.shape[0]
        out = torch.linalg.solve(S + lam * torch.eye(S.shape[0], device=S.device), delta.t()).t()
        return out * (delta.norm() / (out.norm() + 1e-12))

def newton_schulz(G, steps=5, eps=1e-7):
    """Muon-style orthogonalization of a 2-D gradient; returns U V^T approx, rescaled to ||G||_F."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps); transposed = X.shape[0] > X.shape[1]
    if transposed: X = X.t()
    for _ in range(steps):
        A = X @ X.t(); B = b * A + c * A @ A; X = a * X + B @ X
    if transposed: X = X.t()
    return X * G.norm() / (X.norm() + eps)

# ---------------- one training step ----------------
def step(model, x, y, C, cfg, fb, opt, whitener=None, state=None):
    """Performs one update. Returns dict with loss, e (batch), and the per-layer teaching signals (for diagnostics)."""
    B = x.shape[0]
    if cfg.rule == "bp":
        opt.zero_grad(set_to_none=True)
        z, pres, acts = model.forward_blocks(x, detach=False)
        loss_vec, p, e, t = head_forward(z, y, cfg, C); loss = loss_vec.mean(); loss.backward()
        deltas = None
    else:
        opt.zero_grad(set_to_none=True)
        z, pres, acts = model.forward_blocks(x, detach=True)
        with torch.no_grad():
            loss_vec, p, e, t = head_forward(z.detach(), y, cfg, C); loss = loss_vec.mean()
            # head: true gradient
            hL = acts[-1].detach().flatten(1)
            model.out.weight.grad = e.t() @ hL / B; model.out.bias.grad = e.mean(0)
            e_hid = e.clone()
            if getattr(cfg, "fb_signal", "error") == "sign": e_hid = torch.sign(e_hid)   # quantized broadcast error
            if cfg.center == "e": e_hid = e_hid - e_hid.mean(0, keepdim=True)
            if cfg.center == "gated":
                frac = float((e.mean(0) ** 2).sum() / ((e ** 2).sum(1).mean() + 1e-12))   # common-mode energy fraction
                state["gate_on"] = float(frac > cfg.gate_thr)
                if frac > cfg.gate_thr: e_hid = e_hid - e_hid.mean(0, keepdim=True)
        L = len(acts); deltas = [None] * L
        if cfg.rule in ("dfa", "aligned"):
            grads_h = [e_hid @ fb["B"][l].t() for l in range(L)]           # [B, d_l] (flattened for conv)
            order = list(range(L))
        else:  # fa: top-down
            grads_h = [None] * L; order = list(range(L - 1, -1, -1))
        g_a_next = None
        for l in order:
            a, h = pres[l], acts[l]
            if cfg.rule == "fa":
                if l == L - 1: grads_h[l] = e_hid @ fb["R_out"]                # R_out: [C, d_L]
                else: grads_h[l] = g_a_next @ fb["R"][l + 1]                  # R_{l+1}: [d_{l+1}, d_l]
            gh = grads_h[l].view_as(h)
            with torch.no_grad():
                delta = gh * act_deriv(model.act_name, a, h)                  # post-gate teaching signal
                if cfg.center == "delta": delta = delta - delta.mean(0, keepdim=True)
                if whitener is not None and not model.is_conv: delta = whitener.apply(l, delta.flatten(1)).view_as(delta)
            if model.bns is not None:  # BN: backprop from h with the pre-gate signal (autograd handles BN and phi')
                h.backward(gh / B if cfg.center != "delta" else (gh - gh.mean(0, keepdim=True)) / B)
                g_a = a.grad
            else:
                a.backward(delta / B); g_a = delta / B
            deltas[l] = delta.detach(); g_a_next = g_a.flatten(1).detach() * B if cfg.rule == "fa" else None
    # interventions on the parameter update
    if cfg.freeze_bias:
        for l in range(len(model.hidden)): model.hidden[l].bias.grad = None
    if cfg.freeze_out:
        model.out.weight.grad = None; model.out.bias.grad = None
    if cfg.muon:
        for l in range(len(model.hidden)):
            W = model.hidden[l].weight
            if W.grad is not None and W.grad.dim() >= 2: W.grad = newton_schulz(W.grad.flatten(1)).view_as(W.grad)
    measured = bool(state and state.get("measure_update"))
    if measured:
        if model.bns is not None:
            raise ValueError("actual-step measurement currently requires a model without batch normalization")
        # Actual parameter displacement, including optimizer state and bias updates.
        # autograd.grad leaves the local-rule gradients installed in .grad unchanged.
        params = list(model.parameters())
        old = [p.detach().clone() for p in params]
        zz, _, _ = model.forward_blocks(x, detach=False)
        losses, *_ = head_forward(zz, y, cfg, C)
        true = torch.autograd.grad(losses.mean(), params)
    opt.step()
    metrics = {}
    if measured:
        lookup = {id(p): (g.detach(), p.detach()-v) for p,g,v in zip(params,true,old)}
        for name, block in [(f"l{l+1}", model.block_params(l)) for l in range(len(model.hidden))] + [("out", list(model.out.parameters()))]:
            gs, ds = zip(*(lookup[id(p)] for p in block))
            grad = torch.cat([g.flatten() for g in gs]); displacement = torch.cat([d.flatten() for d in ds])
            metrics[f"step_descent_{name}"] = float(-(grad * displacement).sum())
            metrics[f"step_cos_{name}"] = float(F.cosine_similarity(-displacement, grad, dim=0))
            metrics[f"step_norm_{name}"] = float(displacement.norm())
        with torch.no_grad():
            new_z, _, _ = model.forward_blocks(x, detach=False)
            new_loss, *_ = head_forward(new_z, y, cfg, C)
            metrics["actual_batch_loss_decrease"] = float(losses.mean()-new_loss.mean())
    return dict(loss=float(loss.item()), e=e.detach(), deltas=deltas, acts=[h.detach() for h in acts], pres=[a.detach() for a in pres], update_metrics=metrics)
