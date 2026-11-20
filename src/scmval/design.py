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


# ---------------------------------------------------------------------------
# Data plan for Simulator B (frozen at the design lock)
# ---------------------------------------------------------------------------
TEST_NUS = (0.05, 0.10, 0.15, 0.25, 0.45, 0.60)   # same grid as the Gate 5/6 pilot, so the pilot power numbers apply
TEST_DURATIONS = (14, 30, 90)
N_POLICIES = 20                                  # viable policies (Gate 3)
TRAIN_NU_RANGE = (0.0, 0.15)                     # Latin-hypercube over (0, 0.15]
N_TRAIN = 2500                                   # main training runs: nu in the Mild band, D in {14, 30}, policy and seed random
N_CALIBRATION = 600                              # held-out runs from the SAME distribution, for conformal intervals
N_CALM = 500                                     # ablation arm: calm-only training runs (nu = 0)
N_WIDE = 800                                     # sensitivity arm: wider band, 0.15 < nu <= 0.25
TEST_SEEDS_FIRST = 15                            # seeds per (cell, policy) in the first pass of the test grid
TEST_SEEDS_MAX = 30                              # after adaptive deepening of near-margin cells
PLACEBO_CELLS = ((0.15, 30), (0.15, 90), (0.45, 30), (0.45, 90))
PLACEBO_SEEDS = 15
SEC_PER_RUN_ESTIMATE = 400 / 7300 * 409.9        # seconds: released-baseline runtime scaled to L=400; replace with the measured value


def run_plan(sec_per_run: float = SEC_PER_RUN_ESTIMATE, workers: int = 2) -> dict:
    """Run counts and CPU/wall-clock hours for each block of the data plan."""
    cells = len(TEST_NUS) * len(TEST_DURATIONS)
    blocks = {
        "tier1_train": N_TRAIN,
        "tier1_calibration": N_CALIBRATION,
        "tier1_test_first_pass": cells * N_POLICIES * TEST_SEEDS_FIRST,
        "tier2_calm_ablation": N_CALM,
        "tier2_wide_band": N_WIDE,
        "tier2_placebo": len(PLACEBO_CELLS) * N_POLICIES * PLACEBO_SEEDS,
        "tier3_deepening_max": cells * N_POLICIES * (TEST_SEEDS_MAX - TEST_SEEDS_FIRST),
    }
    out = {}
    for k, n in blocks.items():
        cpu = n * sec_per_run / 3600
        out[k] = {"runs": n, "cpu_hours": round(cpu, 1), "wall_hours": round(cpu / workers, 1)}
    for tier in ("tier1", "tier2", "tier3"):
        n = sum(v["runs"] for k, v in out.items() if k.startswith(tier))
        cpu = n * sec_per_run / 3600
        out[tier + "_total"] = {"runs": n, "cpu_hours": round(cpu, 1), "wall_hours": round(cpu / workers, 1)}
    return out
