"""
analysis.py -- the decision-level analysis, as functions with no simulator in them.

The same functions are used on (a) simulated stand-in data in the dry runs
(scripts/10_pipeline_dryruns.py) and (b) the real study data, so a check passed on
stand-ins is a check of the code that will touch the real results.

Vocabulary: a "cell" = one (severity, duration) combination. In a cell, outcomes form a
table `piv` with one row per seed and one column per policy (unmet demand, days; lower is
better). All policies share the same seeds, so differences are paired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scmval.metrics import bh_reject


def pair_stats(piv: pd.DataFrame | np.ndarray):
    """Paired t-statistics for every policy pair, vectorised.
    Returns (i, j, mean_diff, sd_diff, p) with mean_diff = mean(policy i - policy j)."""
    X = np.asarray(piv, float)
    n, k = X.shape
    i, j = np.triu_indices(k, 1)
    D = X[:, i] - X[:, j]
    m = D.mean(0)
    sd = D.std(0, ddof=1) if n > 1 else np.zeros_like(m)
    p = np.ones_like(m)
    ok = sd > 0
    t = m[ok] / (sd[ok] / np.sqrt(n))
    p[ok] = 2 * stats.t.sf(np.abs(t), n - 1)
    p[~ok] = np.where(m[~ok] == 0, 1.0, 0.0)
    return i, j, m, sd, p


def decidable_pairs(piv, delta: float, q: float = 0.10):
    """A pair is decidable when it is significant after Benjamini-Hochberg (level q, over
    all pairs of this cell) AND the paired mean difference is at least delta (practical margin).
    Returns (i, j, mean_diff, decidable_bool)."""
    i, j, m, sd, p = pair_stats(piv)
    sig = bh_reject(p, q)
    return i, j, m, sig & (np.abs(m) >= delta)


def flip_summary(true_piv, pred_means, delta: float, q: float = 0.10) -> dict:
    """Compare the simulator's policy ranking (from `true_piv`) with a surrogate's predicted
    mean per policy. Only decidable pairs can count as flips."""
    pred = np.asarray(pred_means, float)
    i, j, m, dec = decidable_pairs(true_piv, delta, q)
    pred_diff = pred[i] - pred[j]
    flip = dec & (np.sign(pred_diff) != np.sign(m))
    true_means = np.asarray(true_piv, float).mean(0)
    tau = stats.kendalltau(true_means, pred).statistic if len(pred) > 2 else np.nan
    order = np.argsort(true_means)
    a, b = int(order[0]), int(order[1])
    lo, hi = min(a, b), max(a, b)
    idx = np.nonzero((i == lo) & (j == hi))[0][0]
    top1_testable = bool(dec[idx])
    top1_flip = bool(top1_testable and int(np.argmin(pred)) != a)
    return {"n_pairs": int(len(i)), "n_decidable": int(dec.sum()), "n_flips": int(flip.sum()),
            "flip_rate": float(flip.sum() / dec.sum()) if dec.sum() else np.nan,
            "kendall_tau": float(tau), "top1_testable": top1_testable, "top1_flip": top1_flip,
            "flip_pairs": set(zip(i[flip].tolist(), j[flip].tolist()))}


def regret(true_means, pred_means) -> float:
    """Extra unmet demand (days) from acting on the surrogate's favourite policy."""
    t, p = np.asarray(true_means, float), np.asarray(pred_means, float)
    return float(t[int(np.argmin(p))] - t.min())


def conformal_halfwidth(cal_abs_residuals, alpha: float = 0.10) -> float:
    """Split-conformal half-width from absolute calibration residuals."""
    r = np.sort(np.asarray(cal_abs_residuals, float))
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[min(k, n) - 1]) if k <= n else float("inf")


def coverage(y, pred, halfwidth: float) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float)) <= halfwidth))
