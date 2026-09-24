"""Snapshot measurements of mean drift and gate-restricted local updates.

These are virtual, full-probe SGD updates, not the noisy training minibatch
updates. They do not modify the trained model or its optimizer state.
"""
import copy
import math
import torch
from .models import act_deriv
from .rules import head_forward, step
from .run import make_optimizer


def drift_budget(model, x, y, C, cfg, feedback):
    if cfg.optimizer != "sgd" or cfg.rule != "dfa" or cfg.center != "none" or model.bns is not None or model.is_conv:
        raise ValueError("drift budget requires plain SGD DFA without batch normalization")
    twin = copy.deepcopy(model)
    with torch.no_grad():
        z, pres, acts = model.forward_blocks(x, detach=False)
        _, _, e, _ = head_forward(z, y, cfg, C)
    result = step(twin, x, y, C, cfg, feedback, make_optimizer(twin, cfg), state={})
    with torch.no_grad():
        _, after, new_acts = twin.forward_blocks(x, detach=False)
        inputs = [x] + acts[:-1]
        new_inputs = [x] + new_acts[:-1]
        rows = []
        for l, (block, a, h, hin, newhin, delta, b) in enumerate(zip(
                model.hidden, pres, acts, inputs, new_inputs, result["deltas"], feedback["B"]), 1):
            gamma = act_deriv(cfg.act, a, h)
            hbar = hin.mean(0); dm = delta.mean(0)
            leading = gamma.mean(0) * (b @ e.mean(0))
            residual = ((gamma-gamma.mean(0)) * ((e-e.mean(0)) @ b.T)).mean(0)
            eta = cfg.lr * cfg.hid_lr_mult
            factor = hbar.square().sum() + (0 if cfg.freeze_bias else 1)
            change_h = newhin.mean(0)-hbar
            change_w = twin.hidden[l-1].weight-block.weight
            terms = dict(leading=-eta*factor*leading, residual=-eta*factor*residual,
                         covariance=-eta*((delta-dm)*((hin-hbar) @ hbar).unsqueeze(1)).mean(0),
                         upstream=block.weight @ change_h, finite_cross=change_w @ change_h)
            actual = after[l-1].mean(0)-a.mean(0)
            total = sum(terms.values())
            denom = actual.norm().clamp_min(1e-30)
            row = dict(layer=l, actual_norm=float(denom), reconstruction_error=float((actual-total).norm()),
                       mean_error_norm=float(e.mean(0).norm()), residual_teacher_norm=float(residual.norm()),
                       leading_teacher_norm=float(leading.norm()))
            for name, vector in terms.items():
                row[name+"_norm"] = float(vector.norm())
                row[name+"_projection"] = float((vector*actual).sum()/denom)
            rows.append(row)
    return rows


def gate_sets(model, x, fraction=.2):
    with torch.no_grad():
        _, pres, acts = model.forward_blocks(x, detach=False)
        return [act_deriv(model.act_name,a,h).square().mean(0).topk(max(1, math.ceil(fraction*h.shape[1]))).indices
                for a,h in zip(pres,acts)]


def masked_updates(model, x, y, C, cfg, feedback, fixed_sets, random_sets, step_norm=1e-3):
    """Single-layer virtual steps, all with the same weight-plus-bias norm.

    Current and fixed selections use mean squared activation derivatives;
    the fixed selection is taken at step 120. The true gradient and loss are
    evaluated on the same fixed validation probe. This tests the utility of
    snapshot update directions, not long-term causal recovery after ablation.
    """
    twin = copy.deepcopy(model)
    z, pres, acts = twin.forward_blocks(x, detach=False)
    losses, _, e, _ = head_forward(z, y, cfg, C)
    blocks = [list(twin.block_params(l)) for l in range(len(twin.hidden))]
    params = [p for block in blocks for p in block]
    true_grads = torch.autograd.grad(losses.mean(), params)
    base_loss = float(losses.detach().mean()); current = gate_sets(model,x)
    rows = []
    with torch.no_grad():
        inputs = [x] + acts[:-1]
        for l, (block, a, h, hin, b) in enumerate(zip(blocks,pres,acts,inputs,feedback["B"]),1):
            delta = act_deriv(cfg.act,a,h)*(e @ b.T)
            raw = [delta.T @ hin / len(x), delta.mean(0)]
            grad = torch.cat([g.flatten() for g in true_grads[2*(l-1):2*l]])
            selected = dict(full=torch.arange(h.shape[1],device=x.device),
                            current=current[l-1], fixed120=fixed_sets[l-1], random=random_sets[l-1])
            complement = torch.ones(h.shape[1],device=x.device,dtype=torch.bool)
            complement[fixed_sets[l-1]]=False
            selected["complement120"] = complement.nonzero().flatten()
            old = [p.clone() for p in block]
            for name, indices in selected.items():
                mask = torch.zeros(h.shape[1],device=x.device,dtype=x.dtype);mask[indices]=1
                restricted = [raw[0]*mask[:,None],raw[1]*mask]
                norm = torch.sqrt(sum(g.square().sum() for g in restricted)).clamp_min(1e-30)
                direction = torch.cat([g.flatten() for g in restricted])/norm
                for p,v,g in zip(block,old,restricted): p.copy_(v-step_norm*g/norm)
                zz,_,_ = twin.forward_blocks(x,detach=False)
                new_loss = float(head_forward(zz,y,cfg,C)[0].mean())
                for p,v in zip(block,old): p.copy_(v)
                rows.append(dict(layer=l, selection=name, n_selected=len(indices), step_norm=step_norm,
                                 predicted_decrease=float(step_norm*(grad*direction).sum()),
                                 actual_decrease=base_loss-new_loss,
                                 cosine=float((grad*direction).sum()/grad.norm().clamp_min(1e-30)),
                                 raw_direction_norm=float(norm), probe_loss=base_loss))
    return rows
