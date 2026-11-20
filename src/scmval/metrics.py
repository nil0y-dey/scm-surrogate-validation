"""
metrics.py -- outcome metrics and small helper functions used by the gates.

Everything here is a plain function (no simulation, no file access) so it
can be unit-tested quickly (see tests/test_metrics.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scmval.recovery import RECOVERY_HOLD, RECOVERY_TOL, first_sustained_recovery

# Last-mile capacity self-calibrates to ~1.29 x demand (target_ratio 1.20 /
# packing_eff 0.93; see the stability-boundary script and proposal).
CAPACITY_RATIO = 1.20 / 0.93


def settle_day(series: pd.Series, steady: float, tol: float = 0.01,
               window: int = 20, noise_sd: float = None,
               noise_k: float = 3.0) -> int:
    """First day d such that EVERY 20-day average starting on day d or later
    stays within +/-tol of the steady-state level. 0 means it was settled
    from the very start.

    Policies that lose a little demand even in calm weather (random
    stockouts) never sit exactly on their steady level, so a flat tolerance
    would call them "never settled". If `noise_sd` (the day-to-day spread of
    the 20-day average once steady) is given, the tolerance is widened to
    max(tol, noise_k * noise_sd)."""
    if noise_sd is not None:
        tol = max(tol, noise_k * noise_sd)
    roll = series.rolling(window).mean().shift(-(window - 1))   # mean of days d..d+window-1
    bad = (roll - steady).abs() > tol                           # NaN (series tail) counts as fine
    bad_days = bad[bad].index
    return 0 if len(bad_days) == 0 else int(bad_days.max()) + 1


def window_metrics(daily: pd.DataFrame, t0: int, duration: int, length: int,
                   tol: float = RECOVERY_TOL, hold: int = RECOVERY_HOLD,
                   baseline_offset=(-40, -1)) -> dict:
    """Outcome metrics for ONE run over the fixed window [t0, length).

    `daily` is the table from sim_runner.load_daily_series (indexed by day,
    with demand, served, new_backlog, backlog_end, fill_rate).
    The disruption (if any) lasts days [t0, t0+duration).

      min_fill           lowest daily network fill rate in the window
      triangle_days      resilience triangle: sum of (1 - fill rate), in days
      unmet_demand_days  demand not served from stock, in days of mean demand
      peak_backlog_days  largest backlog, in days of mean demand
      ttr                days from restoration until fill rate stays within
                         tol of its pre-disruption level for `hold` days;
                         None if that never happens inside the window
      censored           True when ttr is None
      ttr_lower_bound    ttr, or (length - restore_day) when censored
    """
    d = daily.loc[:length - 1]
    win = d.loc[t0:length - 1]
    mean_demand = float(win["demand"].mean())
    base = float(d["fill_rate"].loc[t0 + baseline_offset[0]: t0 + baseline_offset[1]].mean())
    restore = t0 + duration
    rec = first_sustained_recovery(d["fill_rate"], base, restore, length - 1, tol, hold)
    return {
        "baseline_fill": base,
        "min_fill": float(win["fill_rate"].min()),
        "triangle_days": float((1.0 - win["fill_rate"]).sum()),
        "unmet_demand_days": float(win["new_backlog"].sum() / mean_demand),
        "peak_backlog_days": float(win["backlog_end"].max() / mean_demand),
        "ttr": None if rec is None else int(rec - restore),
        "censored": rec is None,
        "ttr_lower_bound": int(length - restore) if rec is None else int(rec - restore),
    }


def fluid_ttr_raw(c: float, duration: float, capacity_ratio: float = CAPACITY_RATIO) -> float:
    """Back-of-envelope ('fluid') recovery time before any calibration:
    backlog built during the event (duration x overload) divided by the
    spare capacity per day after restoration. Overload is the share of
    demand the cut capacity cannot carry: max(0, 1 - capacity_ratio*c)."""
    overload = max(0.0, 1.0 - capacity_ratio * c)
    spare = capacity_ratio - 1.0
    return duration * overload / spare


def fit_scale(pred, obs):
    """Least-squares k in obs ~ k * pred (through the origin), and R^2."""
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    k = float((pred * obs).sum() / (pred * pred).sum())
    ss_res = float(((obs - k * pred) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k, r2


def bh_reject(pvals, q: float = 0.10):
    """Benjamini-Hochberg: True where a hypothesis is rejected, FDR <= q."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ok = np.nonzero(p[order] <= q * (np.arange(1, m + 1) / m))[0]
    reject = np.zeros(m, bool)
    if len(ok):
        reject[order[: ok.max() + 1]] = True
    return reject


def paired_test(a, b):
    """Paired comparison on the same seeds. Returns (mean a-b, SD of differences,
    two-sided p). Identical vectors -> p=1; constant non-zero shift -> p=0."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return mean, 0.0, (1.0 if mean == 0.0 else 0.0)
    t = mean / (sd / np.sqrt(n))
    return mean, sd, float(2 * stats.t.sf(abs(t), n - 1))
