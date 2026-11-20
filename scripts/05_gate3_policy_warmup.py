"""
05_gate3_policy_warmup.py
--------------------------
GATE 3. Two questions about every policy in the grid (src/scmval/policies.py),
in CALM weather (no disruption, no shocks):

  1. Is the policy VIABLE? A policy that already loses demand with no
     disruption at all cannot serve as a baseline to compare under stress.
     Viable = steady-state fill rate >= VIABLE_MIN_CALM_FILL.
  2. How many days until it SETTLES? Your warm-up margin (day 100) was
     measured on the default policy only. Lean and heavily-stocked
     policies start differently, so the margin must cover the slowest
     viable policy.

Writes: results/reports/gate3_report.json, gate3_policy_table.csv,
        results/figures/gate3_policy_grid.png

Usage:
    python scripts/05_gate3_policy_warmup.py [--days 500] [--seeds 3] [--workers N]
    python scripts/05_gate3_policy_warmup.py --quick        # 2x2 corners, 1 seed, 250 days
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmval import paths, policies
from scmval.metrics import settle_day
from scmval.sim_runner import (RunConfig, ensemble_mean, ensure_repo, run_batch,
                               verify_patches_present)

OUT_ROOT = str(paths.RUNS_DIR / "gate3")
REPORT_PATH = str(paths.REPORT_DIR / "gate3_report.json")
TABLE_PATH = str(paths.REPORT_DIR / "gate3_policy_table.csv")
FIGURE_PATH = str(paths.FIG_DIR / "gate3_policy_grid.png")

WARMUP = 100                  # current warm-up margin (warmup_calibration.py)
VIABLE_MIN_CALM_FILL = 0.99   # PROPOSED: calm steady-state fill needed to count as viable
SETTLE_TOL = 0.01             # "settled" = 20-day average fill rate stays within +/-1 pp of steady state ...
SETTLE_NOISE_K = 3.0          # ... or within 3 standard deviations of its own steady-state noise, if larger
MARGINAL_BELOW = 0.999        # viable but loses demand even in calm weather (random stockouts)
STEADY_DAYS = 200             # steady state = mean over the last 200 days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: corner policies only, 1 seed, 250 days")
    args = ap.parse_args()

    ss_list, pm_list = policies.SS_SCALES, policies.PIPELINE_MULTS
    if args.quick:
        ss_list = [ss_list[0], ss_list[-1]]
        pm_list = [pm_list[0], pm_list[-1]]
        args.days, args.seeds = 250, 1
    if args.days < STEADY_DAYS + 50:
        raise SystemExit("--days must be at least 250 so a steady state can be measured")

    paths.make_dirs()
    sim_script = ensure_repo()
    verify_patches_present(sim_script)

    seeds = list(range(2025, 2025 + args.seeds))
    grid = policies.grid(ss_list, pm_list)
    configs = []
    for ss, pm in grid:
        for seed in seeds:
            name = f"{policies.label(ss, pm)}_seed{seed}"
            configs.append(RunConfig(
                seed=seed, days=args.days, out_dir=os.path.join(OUT_ROOT, name),
                scenario_name=name, ss_scale=ss, pipeline_mult=pm,
                shock_count_scale=0.0, burst_rate_scale=0.0, slim=True))
    print(f"\nRunning {len(configs)} calm simulations "
          f"({len(grid)} policies x {len(seeds)} seeds, {args.days} days each)...")
    run_batch(configs, n_workers=args.workers, sim_script=sim_script)

    rows = []
    for ss, pm in grid:
        dirs = [os.path.join(OUT_ROOT, f"{policies.label(ss, pm)}_seed{s}") for s in seeds]
        series = ensemble_mean(dirs, "fill_rate")
        steady_part = series.loc[args.days - STEADY_DAYS:]
        steady = float(steady_part.mean())
        noise_sd = float(steady_part.rolling(20).mean().dropna().std())
        rows.append({
            "ss_scale": ss, "pipeline_mult": pm,
            "steady_fill": steady,
            "noise_sd_20d": noise_sd,
            "settle_day": settle_day(series, steady, tol=SETTLE_TOL,
                                     noise_sd=noise_sd, noise_k=SETTLE_NOISE_K),
            "early_fill_0_49": float(series.loc[0:49].mean()),
            "viable": steady >= VIABLE_MIN_CALM_FILL,
            "marginal": VIABLE_MIN_CALM_FILL <= steady < MARGINAL_BELOW,
        })
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_PATH, index=False)

    pd.set_option("display.width", 140)
    print("\n=== Calm-weather behaviour of every policy ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    viable = table[table["viable"]]
    bad = table[~table["viable"]]
    max_settle = int(viable["settle_day"].max()) if len(viable) else None
    slowest = (viable.loc[viable["settle_day"].idxmax()] if len(viable) else None)

    print(f"\nViable policies (calm fill >= {VIABLE_MIN_CALM_FILL}): {len(viable)} of {len(table)}")
    marg = table[table["marginal"]]
    if len(marg):
        print(f"MARGINAL ({len(marg)}; viable, but calm fill < {MARGINAL_BELOW}: random stockouts even with no disruption):")
        for _, r in marg.iterrows():
            print(f"   ss_scale={r.ss_scale:g}  pipeline_mult={r.pipeline_mult:g}  "
                  f"calm steady fill={r.steady_fill:.4f}")
    if len(bad):
        print("NOT viable (remove from the grid or accept that they fail with no disruption):")
        for _, r in bad.iterrows():
            print(f"   ss_scale={r.ss_scale:g}  pipeline_mult={r.pipeline_mult:g}  "
                  f"calm steady fill={r.steady_fill:.4f}")

    checks = []
    if max_settle is None:
        checks.append(("FAIL", "no policy is viable in calm weather"))
    else:
        ok = max_settle <= WARMUP
        checks.append(("PASS" if ok else "FAIL",
                       f"slowest viable policy settles by day {max_settle} "
                       f"(ss_scale={slowest.ss_scale:g}, pipeline_mult={slowest.pipeline_mult:g}); "
                       f"current warm-up margin = {WARMUP}"))
        if not ok:
            rec = int(math.ceil(1.25 * max_settle / 25.0) * 25)
            checks.append(("NOTE", f"suggested warm-up margin: {rec} days "
                                   f"(slowest settle day x 1.25, rounded up to a multiple of 25)"))
    print()
    for status, msg in checks:
        print(f"  [{status}] {msg}")

    # ---- figure: two heatmaps over the policy grid
    ss_vals = sorted(table["ss_scale"].unique())
    pm_vals = sorted(table["pipeline_mult"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, title, fmt, cmap in (
            (axes[0], "steady_fill", "Calm steady-state fill rate", "{:.3f}", "RdYlGn"),
            (axes[1], "settle_day", "Days until settled", "{:.0f}", "YlOrRd")):
        mat = np.full((len(ss_vals), len(pm_vals)), np.nan)
        for _, r in table.iterrows():
            mat[ss_vals.index(r.ss_scale), pm_vals.index(r.pipeline_mult)] = r[col]
        im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(pm_vals)), [f"{v:g}" for v in pm_vals])
        ax.set_yticks(range(len(ss_vals)), [f"{v:g}" for v in ss_vals])
        ax.set_xlabel("pipeline_mult"); ax.set_ylabel("ss_scale"); ax.set_title(title)
        for _, r in table.iterrows():
            i, j = ss_vals.index(r.ss_scale), pm_vals.index(r.pipeline_mult)
            mark = "" if r.viable else "\n(not viable)"
            ax.text(j, i, fmt.format(r[col]) + mark, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)

    report = {
        "settings": {"days": args.days, "seeds": seeds, "viable_min_calm_fill": VIABLE_MIN_CALM_FILL,
                     "settle_tol": SETTLE_TOL, "settle_noise_k": SETTLE_NOISE_K, "warmup_margin": WARMUP, "quick": args.quick},
        "n_policies": len(table), "n_viable": int(len(viable)),
        "marginal": marg[["ss_scale", "pipeline_mult", "steady_fill"]].to_dict("records"),
        "not_viable": bad[["ss_scale", "pipeline_mult", "steady_fill"]].to_dict("records"),
        "max_settle_day_viable": max_settle,
        "checks": [{"status": s, "message": m} for s, m in checks],
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTable saved to {TABLE_PATH}\nFigure saved to {FIGURE_PATH}\nReport saved to {REPORT_PATH}")
    if any(s == "FAIL" for s, _ in checks):
        print("\nGATE 3: FAILED (see above).")
        sys.exit(1)
    print("\nGATE 3: PASSED.")


if __name__ == "__main__":
    main()
