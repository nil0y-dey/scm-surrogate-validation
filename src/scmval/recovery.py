"""recovery.py -- how 'recovered from a disruption' is decided."""
from __future__ import annotations

RECOVERY_TOL = 0.02    # +/- 2 percentage points of fill rate
RECOVERY_HOLD = 7      # days the fill rate must STAY within tolerance


def first_sustained_recovery(series, baseline, first_day, last_day,
                             tol=RECOVERY_TOL, hold=RECOVERY_HOLD):
    """First day d in [first_day, last_day] such that the fill rate stays
    within +/-tol of `baseline` for `hold` consecutive days starting at d
    (and the whole hold window lies inside the run). Returns None if it
    never does -- i.e. the run is right-censored. A single lucky day
    touching the tolerance band does not count as recovery."""
    for day in range(first_day, last_day - hold + 2):
        window = series.loc[day:day + hold - 1]
        if len(window) == hold and (window - baseline).abs().max() <= tol:
            return day
    return None
