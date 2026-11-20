"""
severity_calibration.py
-------------------------
Calibrates the Moderate-severity band for timed, bounded disruptions
(patch 3) against the permanent-regime instability threshold
(~0.775, stability_boundary.py): finds a disrupt_capacity_scale value
that stays clear of the threshold even at the longest disruption
duration, so "Moderate" means genuine, fast-recovering queueing
pressure rather than an occasional re-entry into near-permanent
collapse.

Recovery is measured against each run's own pre-disruption plateau
(not a hardcoded target), and right-censored at the run length L
under test -- the same convention used for time-to-recovery in the
main study (paper, Evaluation Metrics section).

The pre-disruption baseline window is defined relative to the
disruption start day, not as absolute days, so it stays valid if the
warm-up margin or horizon changes.

Usage:
    python severity_calibration.py [--candidates 0.85,0.80,0.78]
                                    [--duration 90] [--days 400]
                                    [--seeds 5] [--workers 4]
Note: --days must exceed WARMUP + --duration, or the disruption
window falls outside the run.
"""
from __future__ import annotations

import argparse
import os
import pickle

from scmval import paths
from scmval.recovery import RECOVERY_HOLD, RECOVERY_TOL, first_sustained_recovery
from scmval.sim_runner import (RunConfig, ensemble_mean, ensure_repo, load_daily_series,
                               run_batch, verify_patches_present)

OUT_ROOT = str(paths.RUNS_DIR / "severity_calibration")
RESULTS_PATH = str(paths.DERIVED_DIR / "severity_calibration_results.pkl")

WARMUP = 100                        # from warmup_calibration.py
BASELINE_WINDOW_OFFSET = (-40, -1)  # relative to disrupt_start_day, not absolute days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default="0.85,0.80,0.78",
                     help="Comma-separated disrupt_capacity_scale values to test.")
    ap.add_argument("--duration", type=int, default=90,
                     help="Disruption duration in days (test the longest/worst case).")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.days <= WARMUP + args.duration:
        raise SystemExit(
            f"--days ({args.days}) must exceed WARMUP + --duration "
            f"({WARMUP + args.duration}), or the disruption window and "
            f"recovery observation fall outside the run."
        )

    candidates = [float(x) for x in args.candidates.split(",")]
    seeds = list(range(2025, 2025 + args.seeds))

    paths.make_dirs()
    sim_script = ensure_repo()
    verify_patches_present(sim_script)

    configs = []
    for cs in candidates:
        for seed in seeds:
            out_dir = os.path.join(OUT_ROOT, f"cs_{cs}_seed{seed}")
            configs.append(RunConfig(
                seed=seed, days=args.days, out_dir=out_dir,
                scenario_name=f"cs_{cs}_{seed}",
                disrupt_capacity_scale=cs,
                disrupt_start_day=WARMUP,
                disrupt_duration=args.duration,
                shock_count_scale=0.0, burst_rate_scale=0.0, slim=True,
            ))
    print(f"\nRunning {len(configs)} simulations "
          f"({len(candidates)} candidates x {len(seeds)} seeds, "
          f"{args.duration}-day disruption, {args.days}-day horizon)...")
    run_batch(configs, n_workers=args.workers, sim_script=sim_script)

    print("\n=== Recovery calibration (baseline-relative, censored at "
          f"day {args.days}) ===\n")
    results = {}
    for cs in candidates:
        dirs = [os.path.join(OUT_ROOT, f"cs_{cs}_seed{s}") for s in seeds]
        series = ensemble_mean(dirs, "fill_rate")
        baseline_lo = WARMUP + BASELINE_WINDOW_OFFSET[0]
        baseline_hi = WARMUP + BASELINE_WINDOW_OFFSET[1]
        baseline = series.loc[baseline_lo:baseline_hi].mean()
        restore_day = WARMUP + args.duration

        recovered_day = first_sustained_recovery(
            series, baseline, restore_day, args.days - 1)

        dip = series.loc[WARMUP:restore_day - 1].min()
        ttr = (recovered_day - restore_day) if recovered_day is not None else None
        censored = recovered_day is None

        # Per-seed view. The study reports per-run recovery. (The ensemble
        # mean is not a typical run: it waits for the slowest seed, so it
        # recovers LATER than most single runs.)
        per_seed = []
        for d_ in dirs:
            sr = load_daily_series(d_)["fill_rate"]
            b_ = sr.loc[baseline_lo:baseline_hi].mean()
            rd = first_sustained_recovery(sr, b_, restore_day, args.days - 1)
            per_seed.append(None if rd is None else rd - restore_day)
        n_cens = sum(1 for t in per_seed if t is None)

        results[cs] = {
            "baseline": baseline, "min_fill_rate": dip,
            "recovered_day": recovered_day, "ttr": ttr, "censored": censored,
            "per_seed_ttr": per_seed, "n_censored_seeds": n_cens,
        }
        status = (f"CENSORED at day {args.days}" if censored
                  else f"recovered day {recovered_day} (TTR={ttr}d)")
        print(f"disrupt_capacity_scale={cs} (pre-disruption baseline={baseline:.3f}): "
              f"min fill rate={dip:.3f}  {status}")
        print(f"    per-seed time-to-recovery (days, None = censored): {per_seed}  "
              f"-> {n_cens}/{len(per_seed)} seeds censored")

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("\nPick the largest-cut (most severe) candidate that both avoids "
          "censoring and keeps min fill rate comfortably above the collapse "
          "floor seen in stability_boundary.py; that value anchors the "
          "Moderate-band floor, and its TTR anchors the run length L.")


if __name__ == "__main__":
    main()
