"""
datagen.py -- turns the frozen design (design.py) into concrete lists of simulation runs.

Blocks (see docs/DESIGN_LOCK.md):
  train        Mild-band training runs (latin-hypercube over overload nu, D in {14, 30}, random policy)
  calibration  held-out runs from the SAME distribution as `train` (for conformal intervals)
  calm         ablation arm: no disruption at all
  wide         sensitivity arm: 0.15 < nu <= 0.25
  test         factorial grid: every (nu, D, policy, seed). Seeds 0-14 first; 15-29 are the deepening seeds
  placebo      an independent repeat of a few test cells (different seeds), for the noise floor

Every run has its own seed from a block-specific range, so seeds never overlap between roles.
Demand is paired across policies within a test cell because all 20 policies use the same seeds.
"""
from __future__ import annotations

import itertools
import os

import numpy as np

from scmval import design, paths, policies
from scmval.sim_runner import RunConfig

POLICIES = policies.viable_grid()
SEED0 = {"train": 10000, "calm": 13000, "wide": 14000, "calibration": 20000, "test": 30000, "placebo": 40000}
BLOCKS = tuple(SEED0)
_RNG = {"train": 101, "calm": 102, "wide": 103, "calibration": 104}


def run_dir(block: str, name: str) -> str:
    return str(paths.RUNS_DIR / "B" / block / name)


def _cfg(block, name, nu, D, pol, seed, calm=False) -> tuple[dict, RunConfig]:
    meta = {"block": block, "run": name, "nu": nu, "D": D, "ss": pol[0], "pm": pol[1], "seed": seed}
    cfg = RunConfig(seed=seed, days=design.RUN_LENGTH, out_dir=run_dir(block, name), scenario_name=name,
                    ss_scale=pol[0], pipeline_mult=pol[1],
                    disrupt_capacity_scale=1.0 if calm else design.capacity_from_nu(nu),
                    disrupt_start_day=design.T0, disrupt_duration=0 if calm else D,
                    shock_count_scale=0.0, burst_rate_scale=0.0, slim=True)
    return meta, cfg


def _balanced(values, n, rng):
    v = np.resize(np.array(values, dtype=object), n)
    rng.shuffle(v)
    return list(v)


def _sampled(block: str, n: int, nu_lo: float, nu_hi: float, calm=False):
    rng = np.random.default_rng(_RNG[block])
    strata = (rng.permutation(n) + rng.uniform(1e-6, 1.0, n)) / n          # latin hypercube: one point per stratum
    nus = np.round(nu_lo + (nu_hi - nu_lo) * strata, 4)
    Ds = _balanced(list(design.TRAIN_DURATIONS), n, rng)
    pols = _balanced(list(range(len(POLICIES))), n, rng)
    out = []
    for i in range(n):
        nu = 0.0 if calm else float(nus[i])
        out.append(_cfg(block, f"{block}_{i:05d}", nu, int(Ds[i]), POLICIES[int(pols[i])], SEED0[block] + i, calm))
    return out


def _grid(block, cells, seed_lo, seed_hi):
    out = []
    for (nu, D), pol, s in itertools.product(cells, POLICIES, range(seed_lo, seed_hi)):
        name = f"{block}_nu{nu:g}_D{D}_{policies.label(*pol)}_s{s}"
        out.append(_cfg(block, name, nu, D, pol, SEED0[block] + s))
    return out


def plan(block: str, seed_lo: int | None = None, seed_hi: int | None = None, cells=None):
    """List of (meta, RunConfig) for a block. For test/placebo, `seed_lo:seed_hi` selects which seeds
    (default: the first-pass seeds); `cells` restricts to chosen (nu, D) cells."""
    if block == "train":
        return _sampled("train", design.N_TRAIN, 0.0, design.TRAIN_NU_RANGE[1])
    if block == "calibration":
        return _sampled("calibration", design.N_CALIBRATION, 0.0, design.TRAIN_NU_RANGE[1])
    if block == "calm":
        return _sampled("calm", design.N_CALM, 0.0, 0.0, calm=True)
    if block == "wide":
        return _sampled("wide", design.N_WIDE, design.TRAIN_NU_RANGE[1], design.SENSITIVITY_TRAIN_NU_MAX)
    if block == "test":
        cs = cells or list(itertools.product(design.TEST_NUS, design.TEST_DURATIONS))
        return _grid("test", cs, 0 if seed_lo is None else seed_lo,
                     design.TEST_SEEDS_FIRST if seed_hi is None else seed_hi)
    if block == "placebo":
        cs = cells or list(design.PLACEBO_CELLS)
        return _grid("placebo", cs, 0 if seed_lo is None else seed_lo,
                     design.PLACEBO_SEEDS if seed_hi is None else seed_hi)
    raise ValueError(f"unknown block {block!r}; choose from {BLOCKS}")


def table_plan(block: str):
    """Every run that could ever belong to this block's table (test: all 30 seeds)."""
    if block == "test":
        return plan("test", 0, design.TEST_SEEDS_MAX)
    return plan(block)
