"""
stability_boundary.py
-----------------------
Characterizes the permanent-regime capacity/demand stability boundary
on this network -- the finding that motivates the main study's use of
timed, bounded disruptions instead of permanent regime settings
(paper: "Design rationale", Compute Budget section; full derivation
in the companion technical note).

For each containers_scale value, runs an ensemble of seeds, then
classifies the resulting backlog trajectory as converging
(saturating exponential), diverging (linear), or a general power law
by AIC, with two guards a naive implementation is missing:

  1. AIC is uninformative on near-zero-variance trajectories (deep in
     the stable region, RSS ~ 0 makes AIC's log(RSS) term degenerate);
     these are classified directly from their variance instead of fit.
  2. A "saturating" classification is only trusted when the
     observation window exceeds 3x the fitted relaxation time tau
     (>=95% of the asymptote reached); otherwise it is reported as
     unresolved (consistent with a critical-slowing-down regime near
     the threshold), not misreported as convergence.

Usage:
    python stability_boundary.py [--seeds 5] [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from scmval import paths
from scmval.sim_runner import RunConfig, ensure_repo, load_daily_series, run_batch, verify_patches_present

OUT_ROOT = str(paths.RUNS_DIR / "stability_boundary")
FIGURE_PATH = str(paths.FIG_DIR / "stability_boundary_figure.png")
RESULTS_PATH = str(paths.DERIVED_DIR / "stability_boundary_results.pkl")

# containers_scale value -> horizon (days). Longer horizons near the
# theoretical threshold (~0.775, see below), where convergence is
# slowest.
REGIME_HORIZONS = {
    1.00: 800, 0.90: 800, 0.85: 1500,
    0.80: 3000, 0.775: 3000, 0.75: 3000, 0.70: 800,
}
WARMUP_EXCLUDE = 50          # days dropped before fitting
FLAT_VARIANCE_TOL = 1.0      # backlog std below this => classify as flat/stable directly
WINDOW_TO_TAU_RATIO = 3.0    # required for a saturating fit to be trusted

TARGET_RATIO = 1.20   # from simulator source: last-mile self-calibration target
PACKING_EFF = 0.93    # from simulator source: packing efficiency divisor
THEORETICAL_THRESHOLD = PACKING_EFF / TARGET_RATIO


def linear(t, a, b):
    return a * t + b


def power_law(t, a, p, t0, b):
    return a * np.power(np.maximum(t - t0, 1e-6), p) + b


def saturating(t, c, a, tau):
    return c - a * np.exp(-t / tau)


def aic(rss, n, k):
    rss = max(rss, 1e-9)  # guard: never take log(0)
    return n * np.log(rss / n) + 2 * k


def classify_regime(t: np.ndarray, y: np.ndarray) -> dict:
    """Fit saturating/linear/power-law models to a backlog trajectory
    and classify convergence behaviour, with the two guards described
    in the module docstring."""
    if np.std(y) < FLAT_VARIANCE_TOL:
        return {"classification": "stable (flat, no AIC needed)",
                "tau": None, "window_over_tau": None, "fits": {}}

    n = len(t)
    fits = {}
    try:
        popt, _ = curve_fit(linear, t, y, p0=[1.0, y[0]])
        fits["linear"] = (aic(np.sum((y - linear(t, *popt)) ** 2), n, 2), popt)
    except Exception:
        fits["linear"] = (np.inf, None)
    try:
        popt, _ = curve_fit(power_law, t, y, p0=[1.0, 0.5, 0.0, y[0]],
                             bounds=([0, 0.05, -50, -np.inf], [np.inf, 3.0, 50, np.inf]),
                             maxfev=20000)
        fits["power_law"] = (aic(np.sum((y - power_law(t, *popt)) ** 2), n, 4), popt)
    except Exception:
        fits["power_law"] = (np.inf, None)
    try:
        popt, _ = curve_fit(saturating, t, y, p0=[y[-1] * 1.2, y[-1], t[-1] / 3],
                             bounds=([0, 0, 1], [np.inf, np.inf, 1e6]), maxfev=20000)
        fits["saturating"] = (aic(np.sum((y - saturating(t, *popt)) ** 2), n, 3), popt)
    except Exception:
        fits["saturating"] = (np.inf, None)

    best = min(fits, key=lambda k: fits[k][0])
    tau = fits["saturating"][1][2] if fits["saturating"][1] is not None else None
    window = t[-1] - t[0]
    window_over_tau = (window / tau) if tau else None

    if best == "saturating" and (tau is None or window_over_tau < WINDOW_TO_TAU_RATIO):
        classification = ("unresolved (best-fit saturating, but window < "
                           f"{WINDOW_TO_TAU_RATIO}x tau -- consistent with critical "
                           "slowing down, not confirmed convergence)")
    else:
        classification = best
    return {"classification": classification, "tau": tau,
            "window_over_tau": window_over_tau, "fits": fits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    paths.make_dirs()
    sim_script = ensure_repo()
    verify_patches_present(sim_script)

    seeds = list(range(2025, 2025 + args.seeds))
    print(f"\nTheoretical stability threshold: containers_scale* = "
          f"packing_eff/target_ratio = {PACKING_EFF}/{TARGET_RATIO} "
          f"= {THEORETICAL_THRESHOLD:.4f}\n")

    configs = []
    for cs, days in REGIME_HORIZONS.items():
        for seed in seeds:
            out_dir = os.path.join(OUT_ROOT, f"cs_{cs}_seed{seed}")
            configs.append(RunConfig(
                seed=seed, days=days, out_dir=out_dir,
                scenario_name=f"cs_{cs}_{seed}", containers_scale=cs,
                shock_count_scale=0.0, burst_rate_scale=0.0,
            ))
    print(f"Running {len(configs)} simulations "
          f"({len(REGIME_HORIZONS)} containers_scale values x {len(seeds)} seeds)...")
    run_batch(configs, n_workers=args.workers, sim_script=sim_script)

    print("\nClassifying convergence per regime...")
    ensemble = {}
    classifications = {}
    for cs, days in REGIME_HORIZONS.items():
        dirs = [os.path.join(OUT_ROOT, f"cs_{cs}_seed{s}") for s in seeds]
        frames = [load_daily_series(d) for d in dirs]
        fr = pd.concat([f["fill_rate"] for f in frames], axis=1).mean(axis=1)
        backlog = pd.concat([f["backlog_end"] for f in frames], axis=1).mean(axis=1)
        ensemble[cs] = {"fill_rate": fr, "backlog_end": backlog}

        t = backlog.index.values.astype(float)
        y = backlog.values.astype(float)
        mask = t >= WARMUP_EXCLUDE
        result = classify_regime(t[mask], y[mask])
        classifications[cs] = result
        ratio = TARGET_RATIO / PACKING_EFF * cs
        print(f"  containers_scale={cs:<6} (theoretical ratio={ratio:.3f}, "
              f"horizon={days}d): {result['classification']}"
              + (f"  [tau={result['tau']:.0f}d, window/tau={result['window_over_tau']:.2f}]"
                 if result['tau'] else ""))

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"ensemble": ensemble, "classifications": classifications}, f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.RdYlGn
    values = list(REGIME_HORIZONS.keys())
    for i, cs in enumerate(values):
        color = cmap(i / (len(values) - 1))
        axes[0].plot(ensemble[cs]["fill_rate"].index, ensemble[cs]["fill_rate"].values,
                     label=f"containers_scale={cs}", color=color, linewidth=1.1)
        axes[1].plot(ensemble[cs]["backlog_end"].index, ensemble[cs]["backlog_end"].values,
                     label=f"containers_scale={cs}", color=color, linewidth=1.1)
    axes[0].set_xlabel("Simulated day"); axes[0].set_ylabel("Fill rate (ensemble mean)")
    axes[0].set_title("Fill rate across the stability boundary")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_xlabel("Simulated day"); axes[1].set_ylabel("Cumulative backlog (ensemble mean)")
    axes[1].set_title("Backlog trajectory (diverging vs. plateauing)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    print(f"\nFigure saved to {FIGURE_PATH}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
