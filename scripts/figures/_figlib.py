"""Shared plumbing for the figure scripts: argument parsing, results access, and partial-data guards."""
from __future__ import annotations
import argparse, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(ROOT, "src") not in sys.path: sys.path.insert(0, os.path.join(ROOT, "src"))

from cmc import analysis as A
from cmc import plotstyle as S


def parse(doc):
    """Standard arguments of every figure script."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"), help="results root")
    ap.add_argument("--out-root", default=ROOT, help="project root; figures go to <out-root>/paper/figures")
    return ap.parse_args()


class Data:
    """Read-only view of a results root that never raises on missing experiments, tags or columns."""

    def __init__(self, root):
        self.root = os.path.abspath(root); self._per = {}

    def per(self, exp):
        if exp not in self._per: self._per[exp] = A.load_per_run(self.root, exp)
        return self._per[exp]

    def exp_dir(self, exp):
        return os.path.join(self.root, exp)

    def has(self, exp, tag=None):
        d = self.exp_dir(exp) if tag is None else os.path.join(self.exp_dir(exp), tag)
        if not os.path.isdir(d): return False
        if tag is None: return True
        import glob
        return bool(glob.glob(os.path.join(d, "seed*_fb*.csv")))

    def tags(self, exp):
        per = self.per(exp)
        return sorted(per.tag.unique()) if per is not None else []

    def layers(self, exp, tag):
        return A.n_layers_of(self.per(exp), tag)

    def traj(self, exp, tag, col):
        """(steps, mean, sd) across the runs of a tag, or None."""
        if not self.has(exp, tag): return None
        r = A.mean_traj(self.exp_dir(exp), tag, col)
        return None if r is None else (r[0], r[1], r[2])

    def stat(self, exp, tag, col, how="mean"):
        return A.col_stat(self.per(exp), tag, col, how)

    def rows(self, exp, tags=None):
        """Per-run rows of an experiment, optionally restricted to a tag list; empty DataFrame when absent."""
        import pandas as pd
        per = self.per(exp)
        if per is None: return pd.DataFrame()
        return per[per.tag.isin(tags)] if tags is not None else per


def deep_col(row, stem):
    """<stem>_l<L> for the row's own depth."""
    n = int(row.get("n_layers", 0) or 0)
    c = f"{stem}_l{n}"
    return float(row[c]) if n and c in row and np.isfinite(_f(row[c])) else float("nan")


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return float("nan")


def deep_series(df, stem):
    """Array of <stem> at each row's deepest layer."""
    return np.asarray([deep_col(r, stem) for _, r in df.iterrows()], float) if len(df) else np.zeros(0)


def layer_colors(n):
    """One palette colour per layer, deepest last."""
    return [S.color(i) for i in range(n)]
