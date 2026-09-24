"""Probe-set diagnostics. Everything is computed on a fixed probe drawn from the validation split."""
from __future__ import annotations
import math, torch, torch.nn.functional as F
from .models import act_deriv
from .rules import head_forward

def participation(u):
    return float((u.sum() ** 2 / (u.numel() * (u * u).sum() + 1e-30)).item())

def eff_rank(Hm, center=True):
    """Entropy effective rank of the second-moment (or covariance) spectrum of rows of Hm [N, d]."""
    X = Hm - Hm.mean(0, keepdim=True) if center else Hm
    s = torch.linalg.svdvals(X.double()) ** 2; p = s / (s.sum() + 1e-30); p = p[p > 1e-15]
    return float(torch.exp(-(p * p.log()).sum()).item())

def mean_pairwise_cos(Hm, n=512):
    X = F.normalize(Hm[:n], dim=1); G = X @ X.t(); m = X.shape[0]
    return float(((G.sum() - m) / (m * (m - 1))).item())

@torch.no_grad()
def layer_stats(model, a, h, t_centered, heavy=False):
    """Per-layer scalars from probe preactivations a [N,d] and activations h [N,d]."""
    a2, h2 = a.flatten(1), h.flatten(1)
    dphi = act_deriv(model.act_name, a2, h2); u = (dphi * dphi).mean(0)
    mu, sig = a2.mean(0), a2.std(0); hbar = h2.mean(0)
    out = dict(p=participation(u), chi=float(u.mean()), mu_spread=float(mu.std()), sig_mean=float(sig.mean()),
               H=float((hbar * hbar).sum()), Eh2=float((h2 * h2).sum(1).mean()), cos=mean_pairwise_cos(h2),
               lam=float(((h2 - hbar).t() @ t_centered / h2.shape[0]).norm() ** 2),
               sat90=float((hbar.abs() > 0.9).float().mean()))
    if model.act_name == "relu":
        pa = (a2 > 0).float().mean(0); out["dead"] = float((pa < 0.05).float().mean()); out["alwayson"] = float((pa > 0.95).float().mean())
    if heavy and h2.shape[1] <= 4096:
        out["rank_c"] = eff_rank(h2[:1024], center=True); out["rank_u"] = eff_rank(h2[:1024], center=False)
    return out

@torch.no_grad()
def probe_pass(model, Xp, yp, C, cfg, fb, heavy=False, ehat0=None):
    """All cheap diagnostics on the probe. Returns flat dict."""
    z, pres, acts = model.forward_blocks(Xp, detach=False)
    loss_vec, p, e, t = head_forward(z, yp, cfg, C)
    pred = z[:, :C].argmax(1); acc = float((pred == yp).float().mean())
    # balanced accuracy over the classes present
    accs = [float((pred[yp == c] == c).float().mean()) for c in range(C) if (yp == c).any()]
    ebar = e.mean(0); etil = e - ebar
    d = dict(probe_loss=float(loss_vec.mean()), probe_acc=acc, probe_bal_acc=float(sum(accs) / len(accs)),
             ebar_norm=float(ebar.norm()), etil_norm=float(etil.norm(dim=1).pow(2).mean().sqrt()),
             out_std=float(p.std(0).mean()))
    if ehat0 is not None: d["ebar_proj"] = float((ebar * ehat0).sum())
    tc = F.one_hot(yp, C).float(); tc = tc - tc.mean(0, keepdim=True)
    for l, (a, h) in enumerate(zip(pres, acts)):
        for k, v in layer_stats(model, a, h, tc, heavy=heavy).items(): d[f"{k}_l{l+1}"] = v
    if heavy and getattr(cfg, "anatomy", 0):
        # whitened label decodability R_l = tr[Lambda^T Sigma_h^-1 Lambda] / mean(pi(1-pi)); variance fraction of the
        # centered preactivation field along the fixed common-mode direction B_l ehat0; PC1 fraction of preactivations
        pri = tc.var(0).mean().clamp_min(1e-12)
        for l, (a, h) in enumerate(zip(pres, acts)):
            if h.flatten(1).shape[1] > 4096: continue
            H2 = h.flatten(1)[:1024].double(); Hc = H2 - H2.mean(0, keepdim=True); Tc = tc[:1024].double()
            Sig = Hc.t() @ Hc / Hc.shape[0]; Lam = Hc.t() @ Tc / Hc.shape[0]
            Sig = Sig + 1e-4 * torch.eye(Sig.shape[0], dtype=Sig.dtype, device=Sig.device) * Sig.diagonal().mean()
            d[f"R_l{l+1}"] = float((Lam * torch.linalg.solve(Sig, Lam)).sum() / pri)
            A = a.flatten(1)[:1024].double(); Ac = A - A.mean(0, keepdim=True); tot = (Ac * Ac).sum().clamp_min(1e-30)
            sv = torch.linalg.svdvals(Ac); d[f"pc1_l{l+1}"] = float(sv[0] ** 2 / tot)
            if ehat0 is not None and cfg.rule in ("dfa", "aligned"):
                v = (fb["B"][l].double() @ ehat0.double()); v = v / (v.norm() + 1e-12)
                d[f"r1frac_l{l+1}"] = float(((Ac @ v) ** 2).sum() / tot)
    if heavy:
        # rowwise weight alignment (MLP only) and projected BP step / cosine per layer on a probe sub-batch
        maps = model.downstream_maps()
        if maps is not None and cfg.rule in ("dfa", "aligned"):
            for l, (M, Bl) in enumerate(zip(maps, fb["B"])):
                d[f"wa_l{l+1}"] = float(F.cosine_similarity(M, Bl, dim=1).mean())
        n = min(1024, Xp.shape[0]); xb, yb = Xp[:n], yp[:n]
        with torch.enable_grad():
            for prm in model.parameters(): prm.grad = None
            z2, pres2, acts2 = model.forward_blocks(xb, detach=False)
            lv, _, e2, _ = head_forward(z2, yb, cfg, C); lv.mean().backward()
            gbp = [model.hidden[l].weight.grad.detach().clone() for l in range(len(model.hidden))]
            for prm in model.parameters(): prm.grad = None
        if cfg.rule in ("dfa", "aligned", "fa"):
            z3, pres3, acts3 = model.forward_blocks(xb, detach=True)
            e3 = e2.detach()
            if cfg.rule in ("dfa", "aligned"):
                gh = [e3 @ fb["B"][l].t() for l in range(len(acts3))]
            else:
                gh = [None] * len(acts3); g_a = e3
                for l in range(len(acts3) - 1, -1, -1):
                    mat = fb["R_out"] if l == len(acts3) - 1 else fb["R"][l + 1]; gh[l] = g_a @ mat
                    g_a = (gh[l].view_as(acts3[l]) * act_deriv(model.act_name, pres3[l], acts3[l])).flatten(1)
            hin = [xb] + [h.detach() for h in acts3[:-1]]
            for l in range(len(acts3)):
                delta = gh[l].view_as(acts3[l]) * act_deriv(model.act_name, pres3[l], acts3[l])
                if model.is_conv: continue
                gd = delta.t() @ hin[l].flatten(1) / n
                d[f"pi_l{l+1}"] = float((gd * gbp[l]).sum() / (gbp[l].norm() ** 2 + 1e-30))
                d[f"cosA_l{l+1}"] = float(F.cosine_similarity(gd.flatten(), gbp[l].flatten(), dim=0))
                d[f"gnorm_ratio_l{l+1}"] = float(gd.norm() / (gbp[l].norm() + 1e-30))
    return d

@torch.no_grad()
def snapshot(model, Xp, fb, cfg):
    """Per-unit mu_i, sigma_i, u_i per layer and the common-mode coefficients (B_l 1)_i."""
    z, pres, acts = model.forward_blocks(Xp, detach=False); out = {}
    for l, (a, h) in enumerate(zip(pres, acts)):
        a2, h2 = a.flatten(1), h.flatten(1); dphi = act_deriv(model.act_name, a2, h2)
        out[f"mu_l{l+1}"] = a2.mean(0).cpu().numpy(); out[f"sigma_l{l+1}"] = a2.std(0).cpu().numpy(); out[f"u_l{l+1}"] = (dphi * dphi).mean(0).cpu().numpy()
        out[f"hbar_l{l+1}"] = h2.mean(0).cpu().numpy()
        if cfg.rule in ("dfa", "aligned"): out[f"B1_l{l+1}"] = fb["B"][l].sum(1).cpu().numpy()
    return out
