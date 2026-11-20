"""
warmup_calibration.py
----------------------
Warm-up calibration: how many days does the destination need to run
before its fill rate reflects steady-state behaviour rather than the
simulator's empty-start transient?

Produces the paper's warm-up figure: 5-seed ensemble-mean daily fill
rate, calm (shock-free, no capacity disruption) vs. a permanent
containers_scale cut, over a pilot horizon long enough to see both
regimes settle.

Usage:
    python warmup_calibration.py [--days 1500] [--seeds 5] [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scmval import paths
from scmval.sim_runner import RunConfig, ensemble_mean, ensure_repo, run_batch, verify_patches_present

OUT_ROOT = str(paths.RUNS_DIR / "warmup_calibration")
FIGURE_PATH = str(paths.FIG_DIR / "warmup_figure.png")
RESULTS_PATH = str(paths.DERIVED_DIR / "warmup_calibration_results.pkl")

REGIMES = {
    # Calm: shock-free, no capacity disruption of any kind.
    "calm": dict(containers_scale=1.0, shock_count_scale=0.0, burst_rate_scale=0.0),
    # A permanent capacity cut, matching the paper's reported regime
    # (containers_scale=0.6), used here only as the design-rationale
    # comparison, not as part of the main timed-disruption design.
    "permanent_cut_0.6": dict(containers_scale=0.6, shock_count_scale=0.0, burst_rate_scale=0.0),
}


def build_configs(seeds, days):
    configs = []
    for regime, knobs in REGIMES.items():
        for seed in seeds:
            out_dir = os.path.join(OUT_ROOT, f"{regime}_seed{seed}")
            configs.append(RunConfig(
                seed=seed, days=days, out_dir=out_dir,
                scenario_name=f"{regime}_{seed}", slim=True, **knobs,
            ))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    paths.make_dirs()
    sim_script = ensure_repo()
    verify_patches_present(sim_script)

    seeds = list(range(2025, 2025 + args.seeds))
    configs = build_configs(seeds, args.days)

    print(f"\nRunning {len(configs)} simulations "
          f"({len(REGIMES)} regimes x {len(seeds)} seeds, {args.days} days each)...")
    run_batch(configs, n_workers=args.workers, sim_script=sim_script)

    print("\nAggregating ensemble means...")
    ensemble = {}
    for regime in REGIMES:
        dirs = [os.path.join(OUT_ROOT, f"{regime}_seed{s}") for s in seeds]
        ensemble[regime] = ensemble_mean(dirs, "fill_rate")

    print("\n=== Windowed ensemble-mean fill rate (50-day windows) ===")
    for regime, series in ensemble.items():
        print(f"\n--- {regime} ---")
        for start in range(0, args.days, 50):
            window = series.loc[start:start + 49]
            if len(window) == 0:
                continue
            print(f"  days {start:>4}-{start + 49:<4}: fill_rate={window.mean():.3f}")

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(ensemble, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ensemble["calm"].index, ensemble["calm"].values,
            label="Calm ($\\sigma=0$, no shocks)", color="#2b6cb0", linewidth=1.3)
    ax.plot(ensemble["permanent_cut_0.6"].index, ensemble["permanent_cut_0.6"].values,
            label="Permanent capacity cut, containers_scale=0.6",
            color="#c53030", linewidth=1.3)
    ax.axvline(100, color="gray", linestyle="--", linewidth=1, label="Warm-up margin (day 100)")
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Daily network fill rate (ensemble mean)")
    ax.set_title("Simulator B warm-up calibration")
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    print(f"\nFigure saved to {FIGURE_PATH}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
