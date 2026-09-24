"""Uncertainty summaries that preserve experimental conditions as sampling units."""
import numpy as np


def condition_correlation(x, y, conditions, draws=2000, seed=190919):
    """Pooled Pearson r with a percentile bootstrap of whole conditions.

    A sampled condition contributes all of its finite trajectory pairs. The
    point estimate retains the original run weighting; the uncertainty unit
    is the physical training setting, not an individual initialization.
    """
    x, y, conditions = np.asarray(x), np.asarray(y), np.asarray(conditions)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, conditions = x[valid], y[valid], conditions[valid]
    groups = [np.column_stack((x[conditions == k], y[conditions == k]))
              for k in np.unique(conditions)]
    if len(groups) < 2 or np.std(x) == 0 or np.std(y) == 0:
        raise ValueError('Correlation uncertainty requires distinct, varying conditions.')
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        sample = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if np.std(sample[:, 0]) > 0 and np.std(sample[:, 1]) > 0:
            boot.append(float(np.corrcoef(sample.T)[0, 1]))
    if not boot:
        raise ValueError('All condition-bootstrap samples have undefined correlation.')
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(r=float(np.corrcoef(x, y)[0, 1]), lo=float(lo), hi=float(hi),
                n=len(x), n_conditions=len(groups), draws=draws,
                valid_draws=len(boot), seed=seed, resampling='whole physical conditions')
