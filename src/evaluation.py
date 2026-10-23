"""Evaluation utilities: fast ROC-AUC, bootstrap confidence intervals.

Nested cross-validation and model metrics are added in stages 4 and 5.
"""
import numpy as np
from scipy.stats import rankdata

from src import config


def auc_score(y_true, score):
    """ROC-AUC computed from ranks (Mann-Whitney formulation), with average ranks for ties.

    AUC = P(score of a random positive > score of a random negative) + 0.5 P(tie).
    Equivalent to sklearn.metrics.roc_auc_score, but fast enough to be bootstrapped.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(score)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(statistic, *arrays, n_boot=config.N_BOOTSTRAP, level=config.CI_LEVEL,
                 seed=config.SEED, stratify=None):
    """Percentile bootstrap confidence interval of statistic(*arrays).

    Patients are resampled with replacement (n_boot times) and the statistic is recomputed on
    each resample. If `stratify` is given (e.g. the class labels), resampling is done within
    each class so that every resample keeps the original class sizes.

    Returns (point_estimate, lower, upper).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = statistic(*arrays)

    if stratify is not None:
        stratify = np.asarray(stratify)
        groups = [np.flatnonzero(stratify == g) for g in np.unique(stratify)]

    values = np.empty(n_boot)
    for b in range(n_boot):
        if stratify is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values[b] = statistic(*[a[idx] for a in arrays])

    alpha = (1 - level) / 2
    lower, upper = np.nanquantile(values, [alpha, 1 - alpha])
    return point, lower, upper
