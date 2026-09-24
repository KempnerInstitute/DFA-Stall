"""Datasets with explicit preprocessing. All statistics are computed on the training split only."""
from __future__ import annotations
import os
import numpy as np, torch
from torchvision import datasets

ROOT_DEFAULT = os.environ.get("CMC_DATA_ROOT", "./data")   # torchvision-style tree

def _raw(name: str, root: str):
    if name == "mnist":
        tr, te = datasets.MNIST(root, train=True, download=False), datasets.MNIST(root, train=False, download=False)
        X, y = tr.data.float().div(255.0), tr.targets.long(); Xt, yt = te.data.float().div(255.0), te.targets.long()
        X, Xt = X.unsqueeze(1), Xt.unsqueeze(1)                      # [N,1,28,28]
    elif name == "fashion":
        tr, te = datasets.FashionMNIST(root, train=True, download=False), datasets.FashionMNIST(root, train=False, download=False)
        X, y = tr.data.float().div(255.0).unsqueeze(1), tr.targets.long(); Xt, yt = te.data.float().div(255.0).unsqueeze(1), te.targets.long()
    elif name in ("cifar10", "cifar100"):
        dataset_type = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        tr, te = dataset_type(root + "/torchvision", train=True, download=False), dataset_type(root + "/torchvision", train=False, download=False)
        X = torch.tensor(tr.data).float().div(255.0).permute(0, 3, 1, 2); y = torch.tensor(tr.targets).long()
        Xt = torch.tensor(te.data).float().div(255.0).permute(0, 3, 1, 2); yt = torch.tensor(te.targets).long()
    else:
        raise ValueError(name)
    return X, y, Xt, yt

def load(name="mnist", root=ROOT_DEFAULT, preprocess="div255", classes=None, val_size=10000, flatten=True):
    """Returns dict with Xtr,ytr,Xval,yval,Xte,yte,C. preprocess in {div255, standardize, pixcenter, pixstd}.
    classes: keep the first `classes` classes (labels unchanged). Validation = last val_size of the official train set."""
    X, y, Xt, yt = _raw(name, root)
    if classes is not None:
        m = y < classes; X, y = X[m], y[m]; mt = yt < classes; Xt, yt = Xt[mt], yt[mt]; C = classes
    else:
        C = int(y.max().item()) + 1
    n_tr = len(X) - val_size
    Xtr, ytr, Xval, yval = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    if preprocess == "div255":
        pass
    elif preprocess == "standardize":
        mu, sd = Xtr.mean(), Xtr.std(); Xtr, Xval, Xt = (Xtr - mu) / sd, (Xval - mu) / sd, (Xt - mu) / sd
    elif preprocess == "pixcenter":
        mu = Xtr.mean(0, keepdim=True); Xtr, Xval, Xt = Xtr - mu, Xval - mu, Xt - mu
    elif preprocess == "pixstd":
        mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-3
        Xtr, Xval, Xt = (Xtr - mu) / sd, (Xval - mu) / sd, (Xt - mu) / sd
    else:
        raise ValueError(preprocess)
    if flatten:
        Xtr, Xval, Xt = Xtr.flatten(1), Xval.flatten(1), Xt.flatten(1)
    return dict(Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval, Xte=Xt, yte=yt, C=C)

class Sampler:
    """Deterministic minibatch sampler; optional class-0 oversampling with probability p0 (imbalanced prior)."""
    def __init__(self, y: torch.Tensor, batch: int, seed: int, p0: float = 0.0):
        self.y, self.batch, self.p0 = y, batch, p0
        self.g = torch.Generator().manual_seed(seed)
        self.idx0 = torch.nonzero(y == 0).squeeze(1)
        self.n = len(y)
    def draw(self, n=None):
        n = n or self.batch
        i_all = torch.randint(0, self.n, (n,), generator=self.g)
        if self.p0 <= 0: return i_all
        take0 = torch.rand(n, generator=self.g) < self.p0
        i_0 = self.idx0[torch.randint(0, len(self.idx0), (n,), generator=self.g)]
        return torch.where(take0, i_0, i_all)
    def prior(self, C: int) -> np.ndarray:
        """Label prior of the sampling distribution."""
        counts = np.bincount(self.y.numpy(), minlength=C).astype(float); base = counts / counts.sum()
        if self.p0 <= 0: return base
        pi = (1 - self.p0) * base; pi[0] += self.p0; return pi
