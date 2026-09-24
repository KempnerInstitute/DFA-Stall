"""Models expose their hidden blocks so that any credit-assignment rule can inject a teaching signal at each block."""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F

def act_fn(name):
    return {"tanh": torch.tanh, "relu": F.relu, "gelu": F.gelu, "sigmoid": torch.sigmoid, "linear": (lambda a: a)}[name]

def act_deriv(name, a, h):
    """phi'(a) given preactivation a and activation h."""
    if name == "tanh": return 1.0 - h * h
    if name == "relu": return (a > 0).to(a.dtype)
    if name == "sigmoid": return h * (1.0 - h)
    if name == "linear": return torch.ones_like(a)
    if name == "gelu":
        cdf = 0.5 * (1.0 + torch.erf(a / math.sqrt(2.0))); pdf = torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
        return cdf + a * pdf
    raise ValueError(name)

class MLP(nn.Module):
    def __init__(self, in_dim, widths, C, act="tanh", batchnorm=False):
        super().__init__()
        dims = [in_dim] + list(widths)
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(widths))])
        self.bns = nn.ModuleList([nn.BatchNorm1d(w) for w in widths]) if batchnorm else None
        self.out = nn.Linear(dims[-1], C)
        self.act_name, self.act = act, act_fn(act)
        for l in list(self.hidden) + [self.out]:
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
        self.widths = list(widths); self.is_conv = False
    def block_params(self, l):
        ps = list(self.hidden[l].parameters());
        if self.bns is not None: ps += list(self.bns[l].parameters())
        return ps
    def forward_blocks(self, x, detach=True):
        """Returns logits z, list of preactivations a_l, list of activations h_l. With detach=True each block's input is
        detached, so autograd from a_l or h_l only reaches block l's parameters (local credit assignment)."""
        h_in = x; pres, acts = [], []
        for i, lin in enumerate(self.hidden):
            a = lin(h_in)
            if self.bns is not None: a = self.bns[i](a)
            if detach and a.requires_grad: a.retain_grad()
            h = self.act(a); pres.append(a); acts.append(h)
            h_in = h.detach() if detach else h
        z = self.out(h_in)
        return z, pres, acts
    def downstream_maps(self):
        """M_l = (W_out W_L ... W_{l+1})^T in R^{d_l x C} for rowwise weight alignment."""
        with torch.no_grad():
            maps = [None] * len(self.hidden); eff = self.out.weight.t(); maps[-1] = eff
            for i in range(len(self.hidden) - 2, -1, -1):
                eff = self.hidden[i + 1].weight.t() @ eff; maps[i] = eff
        return maps

class SmallCNN(nn.Module):
    """conv(1->16,3)+act, conv(16->32,3,stride 2)+act, fc(32*14*14->128)+act, fc(128->C). Three hidden blocks."""
    def __init__(self, in_ch, C, act="tanh", img=28):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.f1 = nn.Linear(32 * (img // 2) * (img // 2), 128); self.out = nn.Linear(128, C)
        self.act_name, self.act = act, act_fn(act)
        for l in [self.c1, self.c2, self.f1, self.out]:
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
        self.hidden = nn.ModuleList([self.c1, self.c2, self.f1]); self.bns = None; self.is_conv = True
        self.widths = [16 * img * img, 32 * (img // 2) * (img // 2), 128]
    def block_params(self, l): return list(self.hidden[l].parameters())
    def forward_blocks(self, x, detach=True):
        pres, acts = [], []
        a = self.c1(x); (a.retain_grad() if (detach and a.requires_grad) else None); h = self.act(a); pres.append(a); acts.append(h)
        h_in = h.detach() if detach else h
        a = self.c2(h_in); (a.retain_grad() if (detach and a.requires_grad) else None); h = self.act(a); pres.append(a); acts.append(h)
        h_in = (h.detach() if detach else h).flatten(1)
        a = self.f1(h_in); (a.retain_grad() if (detach and a.requires_grad) else None); h = self.act(a); pres.append(a); acts.append(h)
        z = self.out(h.detach() if detach else h)
        return z, pres, acts
    def downstream_maps(self): return None
