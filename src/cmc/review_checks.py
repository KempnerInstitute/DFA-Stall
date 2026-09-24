"""Exact snapshot checks used in the manuscript's model validation."""
import copy
import torch
from .models import act_deriv
from .rules import head_forward, step
from .run import make_optimizer


def equivalent_readout(weight, bias, mean, scale):
    """Parameters on (h-mean)/scale that preserve the original logits."""
    return weight * scale, bias + weight @ mean


@torch.no_grad()
def mean_error_update_parts(gamma, h, v, eta):
    """Exact shared-error weight update and its mean/covariance components."""
    common = -eta * (gamma * v).T @ h / len(h)
    rank_one = -eta * (gamma.mean(0) * v)[:, None] * h.mean(0)[None, :]
    covariance = -eta * v[:, None] * ((gamma-gamma.mean(0)).T @ (h-h.mean(0))) / len(h)
    return common, rank_one, covariance


def readout_budget(model, x, y, cfg, fb, C):
    """Exact finite-step mean-logit budget, with the output-map closure separated.

    This is a virtual full-probe SGD step; the trained network is unchanged.
    All vectors have one coordinate per output. Projections are signed shares
    of the measured displacement, and sum to one up to rounding.
    """
    twin = copy.deepcopy(model)
    with torch.no_grad():
        z, pres, acts = model.forward_blocks(x, detach=False)
        _, _, e, target = head_forward(z, y, cfg, C)
    step(twin, x, y, C, cfg, fb, make_optimizer(twin, cfg), state={})
    rows = []
    with torch.no_grad():
        zz, _, aa = twin.forward_blocks(x, detach=False)
        h, hnew = acts[-1], aa[-1]
        hb, eb = h.mean(0), e.mean(0)
        # This study uses the sigmoid baseline; the exact update is head-agnostic.
        closure_error = z.mean(0).sigmoid()-target.mean(0)
        eta = cfg.lr*cfg.out_lr_mult
        dh = hnew.mean(0)-hb
        dw = twin.out.weight-model.out.weight
        terms = dict(leading=-eta*(hb.square().sum()+1)*closure_error,
                     output_map=-eta*(hb.square().sum()+1)*(eb-closure_error),
                     covariance=-eta*((e-eb).T @ (h-hb)/len(h)) @ hb,
                     upstream=model.out.weight @ dh, finite_cross=dw @ dh)
        actual = (zz-z).mean(0)
        row = dict(actual_norm=float(actual.norm()),
                   reconstruction_error=float((actual-sum(terms.values())).norm()),
                   mean_error_norm=float(eb.norm()),
                   output_map_error=float((eb-closure_error).norm()))
        for name, term in terms.items():
            row[name+'_norm'] = float(term.norm())
            row[name+'_projection'] = float((term*actual).sum()/actual.square().sum().clamp_min(1e-30))
        rows.append(row)
        shared = []
        for l,(a,h,b) in enumerate(zip(pres,acts,fb['B']),1):
            hin = x if l==1 else acts[l-2]
            full,rank,cov = mean_error_update_parts(act_deriv(cfg.act_name if hasattr(cfg,'act_name') else cfg.act,a,h),hin,b@eb,cfg.lr*cfg.hid_lr_mult)
            shared.append(dict(layer=l,common_norm=float(full.norm()),rank_one_norm=float(rank.norm()),
                               covariance_norm=float(cov.norm()),reconstruction_error=float((full-rank-cov).norm()),
                               covariance_to_rank=float(cov.norm()/rank.norm().clamp_min(1e-30))))
    return rows[0],shared
