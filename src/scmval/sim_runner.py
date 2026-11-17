"""
sim_runner.py
--------------
Shared utilities for running the patched ISOMORPH simulator
(nil0y-dey/ISOMORPH_Hmod, a disclosed three-patch fork of
Zhang et al. 2026, ISOMORPH) and aggregating its output.

Every calibration script in this package imports from here instead
of re-implementing simulator invocation or CSV parsing, so there is
exactly one place that knows how to run a simulation and one place
that knows how to read its output.

Patch summary (full derivation and no-op proofs: patch_verification.py
and the paper's Compute Budget section):
  1. Shock-event duration is scaled to the requested horizon
     (n_steps / 7300); no-op at the released 7,300-day horizon.
  2. containers_scale rescales the continuous container_volume field
     (not the integer container count) after last-mile
     self-calibration; no-op at containers_scale = 1.0.
  3. A scheduled per-day capacity factor gives timed, bounded
     disruptions: disrupt_capacity_scale applies only during
     [disrupt_start_day, disrupt_start_day + disrupt_duration);
     no-op at disrupt_duration = 0.
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from scmval.paths import ISOMORPH_DIR

SIM_REPO_URL = "https://github.com/nil0y-dey/ISOMORPH_Hmod.git"
SIM_REPO_DIR = str(ISOMORPH_DIR)
SIM_SCRIPT = os.path.join(SIM_REPO_DIR, "simulator", "Supplychaingeo_item50.py")

# Reference values from the released baseline configuration
# (C=50, seed=2025, days=7300, pipeline_mult=7), recorded directly
# from the simulator's own console output. Used by
# patch_verification.py to confirm all three patches are no-ops.
RELEASED_BASELINE_REFERENCE = {
    "total_demand": 75_262_671,
    "total_served": 68_841_943,
    "total_backlog": 6_420_728,
    "fill_rate_mean": 0.918,
    "fill_rate_min": 0.873,
    "fill_rate_max": 0.952,
}


# Files removed by RunConfig(slim=True). daily_records.csv (the
# idempotency marker and the source of every metric) is always kept.
SLIM_DELETE = ["shipments.csv", "inventory_history.csv", "backlog_history.csv",
               "intransit_history.csv", "demand_signals.npy",
               "demand_signals_cols.txt"]


def ensure_repo(force_fresh: bool = False) -> str:
    """Clone the patched fork if not already present locally.
    Returns the path to the simulator script."""
    if force_fresh and os.path.exists(SIM_REPO_DIR):
        subprocess.run(["rm", "-rf", SIM_REPO_DIR], check=True)
    # Normally the simulator arrives as a git submodule. If the folder is
    # missing or EMPTY (submodule not initialised), fall back to cloning it.
    if not os.path.exists(SIM_REPO_DIR) or not os.listdir(SIM_REPO_DIR):
        subprocess.run(
            ["git", "clone", "--depth", "1", SIM_REPO_URL, SIM_REPO_DIR],
            check=True,
        )
    if not os.path.exists(SIM_SCRIPT):
        raise FileNotFoundError(
            f"Simulator script not found at {SIM_SCRIPT}. If third_party/ "
            f"ISOMORPH_Hmod looks wrong, run: git submodule update --init "
            f"(or set ISOMORPH_REPO_DIR to another clone of {SIM_REPO_URL})."
        )
    return SIM_SCRIPT


def verify_patches_present(sim_script: Optional[str] = None) -> None:
    """Assert all three disclosed patches are present in the cloned
    fork's source before any calibration run trusts its output.
    This is a static source check; patch_verification.py additionally
    runs the simulator and checks its output against known reference
    values, which is the test that actually matters for a reader."""
    sim_script = sim_script or SIM_SCRIPT
    with open(sim_script, "r") as f:
        src = f.read()
    checks = {
        "Patch 1 (horizon-independent shock duration)": "n_steps / 7300" in src,
        "Patch 2 (permanent capacity rescaling via container_volume)":
            "e.container_volume = e.container_volume * containers_scale" in src,
        "Patch 3 (scheduled per-day capacity factor)":
            "disrupt_capacity_scale" in src and "disrupt_duration" in src,
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        raise AssertionError(
            f"Missing patch(es) in {sim_script}: {missing}. "
            f"Check {SIM_REPO_URL} and the branch/commit in use."
        )
    print("All three patches confirmed present (source check).")


@dataclass
class RunConfig:
    """One simulator invocation. Every patch-controlled field
    defaults to the value that makes that patch a no-op, so a
    RunConfig with no fields set beyond seed/days/out_dir/
    scenario_name reproduces the released baseline exactly (verified
    in patch_verification.py). pipeline_mult defaults to 7.0, which
    is not a patch default but ISOMORPH's own released-baseline value
    (its built-in default is 0.0, a reactive shipping rule; 7.0
    selects the proactive rule the released datasets use)."""

    seed: int
    days: int
    out_dir: str
    scenario_name: str
    pipeline_mult: float = 7.0
    containers_scale: float = 1.0
    disrupt_capacity_scale: float = 1.0
    disrupt_start_day: int = 0
    disrupt_duration: int = 0
    shock_count_scale: float = 1.0
    burst_rate_scale: float = 1.0
    leadtime_scale: float = 1.0
    ss_scale: float = 1.0
    # slim=True deletes the big per-run history files after a successful
    # run, keeping only daily_records.csv, service_summary.csv and
    # scenario.json. A 300-day run writes ~9.7 MB, ~97% of it in the
    # history/shipment files; a big grid would otherwise need hundreds
    # of GB. Leave False for patch_verification and the gate checks.
    slim: bool = False

    def to_cmd(self, sim_script: str) -> List[str]:
        return [
            sys.executable, sim_script,
            "--days", str(self.days),
            "--seed", str(self.seed),
            "--pipeline_mult", str(self.pipeline_mult),
            "--containers_scale", str(self.containers_scale),
            "--disrupt_capacity_scale", str(self.disrupt_capacity_scale),
            "--disrupt_start_day", str(self.disrupt_start_day),
            "--disrupt_duration", str(self.disrupt_duration),
            "--shock_count_scale", str(self.shock_count_scale),
            "--burst_rate_scale", str(self.burst_rate_scale),
            "--leadtime_scale", str(self.leadtime_scale),
            "--ss_scale", str(self.ss_scale),
            "--out_dir", self.out_dir,
            "--scenario_name", self.scenario_name,
        ]


def run_one(cfg: RunConfig, sim_script: Optional[str] = None) -> str:
    """Run one simulation. Idempotent: skips if its output already
    exists, so a batch can be safely stopped and resumed."""
    sim_script = sim_script or SIM_SCRIPT
    marker = os.path.join(cfg.out_dir, "daily_records.csv")
    if os.path.exists(marker):
        return f"SKIP (already done): {cfg.scenario_name} seed={cfg.seed}"
    result = subprocess.run(cfg.to_cmd(sim_script), capture_output=True, text=True)
    if result.returncode != 0:
        return f"FAIL: {cfg.scenario_name} seed={cfg.seed}: {result.stderr[-500:]}"
    if cfg.slim:
        for name in SLIM_DELETE:
            path = os.path.join(cfg.out_dir, name)
            if os.path.exists(path):
                os.remove(path)
    return f"OK: {cfg.scenario_name} seed={cfg.seed}"


def run_batch(configs: List[RunConfig], n_workers: int = 4,
              sim_script: Optional[str] = None) -> None:
    """Run a batch of configurations in parallel, printing progress
    as each completes (not in submission order)."""
    sim_script = sim_script or SIM_SCRIPT
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(run_one, c, sim_script) for c in configs]
        for i, fut in enumerate(as_completed(futures)):
            print(f"[{i + 1}/{len(configs)}] {fut.result()}")


def load_daily_series(out_dir: str) -> pd.DataFrame:
    """Load one run's daily_records.csv and return network-wide daily
    demand, served, backlog, and fill_rate, indexed by day."""
    df = pd.read_csv(os.path.join(out_dir, "daily_records.csv"))
    daily = df.groupby("day").agg(
        demand=("demand", "sum"),
        served=("served_from_stock", "sum"),
        new_backlog=("new_backlog_today", "sum"),
        backlog_end=("dest_backlog_end_before_ship", "sum"),
    ).reset_index()
    daily["fill_rate"] = daily["served"] / daily["demand"]
    return daily.set_index("day")


def ensemble_mean(out_dirs: List[str], column: str = "fill_rate") -> pd.Series:
    """Ensemble-average one metric across several seeds' output
    directories (same regime, different seed each)."""
    frames = [load_daily_series(d)[column] for d in out_dirs]
    return pd.concat(frames, axis=1).mean(axis=1)
