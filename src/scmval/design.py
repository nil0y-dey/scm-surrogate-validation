"""
design.py -- the PROPOSED study design, in one place.

Everything here is a proposal until the design is locked (commit + tag
`design-lock`). Changing a value here changes every script that imports it.

Severity is measured by OVERLOAD  nu = max(0, 1 - 1.29 c): the share of
demand the network cannot carry during the event (c = capacity scale).
nu = 0 for any cut the spare capacity absorbs (c >= 0.775). Gate 4 showed
nothing happens for nu = 0, and recovery time follows D * nu.
"""
from __future__ import annotations

from scmval.metrics import CAPACITY_RATIO

T0 = 100              # disruption start = warm-up margin (Gate 3: slowest viable policy settles by day 53)
RUN_LENGTH = 400      # L. Gate 4: at L=400 only the most extreme 90-day corner is censored
DURATIONS = {"short": 14, "medium": 30, "long": 90}

# overload bands. Largest scanned cut is c=0.25 -> nu=0.677.
NU_BANDS = {"mild": (0.00, 0.15), "moderate": (0.15, 0.35),
            "severe": (0.35, 0.55), "extreme": (0.55, 0.68)}
TRAIN_BAND = "mild"
TRAIN_DURATIONS = (14, 30)          # long events are test-only

# Pre-specified analysis rules (fixed BEFORE any training/test data exist; pilot-informed values are noted).
MIN_DECIDABLE_FRAC_EVAL = 0.30      # a test cell with fewer decidable policy pairs is reported but kept out of the headline flip tables
SENSITIVITY_TRAIN_NU_MAX = 0.25     # wider-training-band sensitivity run (replaces the old sigma<=0.3 variant)
ABLATION_CALM_ONLY = True           # also train a calm-only arm (nu = 0)

DELTA_FILL = 0.005    # practical margin: 0.5 percentage points of fill rate

# Non-overlapping seed ranges (pilots/gates used 2025-2029). Never reuse a
# range for another purpose: that keeps test data untouched.
SEED_RANGES = {"pilot": (2025, 2999), "training": (10000, 19999),
               "calibration": (20000, 29999), "test": (30000, 39999),
               "placebo": (40000, 49999)}


def capacity_from_nu(nu: float) -> float:
    """Capacity scale c that produces overload nu."""
    return round((1.0 - nu) / CAPACITY_RATIO, 4)


def nu_from_capacity(c: float) -> float:
    return max(0.0, 1.0 - CAPACITY_RATIO * c)


def delta_unmet_days(length: int = RUN_LENGTH, t0: int = T0) -> float:
    """DELTA_FILL in the units of the unmet-demand metric (days of demand over [t0, length))."""
    return DELTA_FILL * (length - t0)
