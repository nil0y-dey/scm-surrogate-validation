"""
07_gate56_policy_power.py
--------------------------
GATES 5 and 6 together (they share the same simulations).

For a grid of overload levels nu x durations D, run every VIABLE policy
(20 of 25, Gate 3) on the SAME seeds (demand is paired across policies,
Gate 1), and measure unmet demand (days of demand lost) per run.

GATE 5  Training signal. Do runs in the Mild training band (nu <= 0.15,
        D in 14/30) actually lose demand, and does the outcome vary across
        seeds and policies? If not, surrogates have nothing to learn.
GATE 6  Policy power. In each (nu, D) cell, over all policy pairs:
          * decidable pair  = paired mean difference >= delta (0.5 pp of
            fill rate = 1.5 days of unmet demand at L=400) AND significant
            under Benjamini-Hochberg (q=0.10);
          * random-advice regret = how much worse a randomly chosen policy
            is than the best one, on average. If this is below delta, the
            question "which policy is best?" has no practical answer there;
          * minimum detectable difference at the seeds used and at 30 seeds.

Writes results/reports/gate56_per_run.csv, gate56_cells.csv, gate56_report.json,
results/figures/gate56_power.png. Policy runs are 400 days; metrics over [100, 400).

Usage:
    python scripts/07_gate56_policy_power.py [--seeds 5] [--workers N]
    python scripts/07_gate56_policy_power.py --quick
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmval import design, paths, policies
from scmval.metrics import bh_reject, paired_test, window_metrics
from scmval.sim_runner import (RunConfig, ensure_repo, load_daily_series, run_batch,
                               verify_patches_present)

OUT_ROOT = str(paths.RUNS_DIR / "gate56")
PER_RUN = str(paths.REPORT_DIR / "gate56_per_run.csv")
CELLS = str(paths.REPORT_DIR / "gate56_cells.csv")
REPORT = str(paths.REPORT_DIR / "gate56_report.json")
FIG = str(paths.FIG_DIR / "gate56_power.png")

NUS = [0.05, 0.10, 0.15, 0.25, 0.45, 0.60]
DS = [14, 30, 90]
MIN_DECIDABLE_FRAC = 0.30       # PROPOSED pass threshold (edit before looking at results in the real study)
Q = 0.10
SEED0 = design.SEED_RANGES["pilot"][0]


def rdir(nu, D, pol, seed):
    return os.path.join(OUT_ROOT, f"nu{nu:g}_D{D}_{policies.label(*pol)}_s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 nu x 1 D x 4 policies x 3 seeds")
    a = ap.parse_args()
    nus, ds, pols = NUS, DS, policies.viable_grid()
    if a.quick:
        nus, ds, a.seeds = [0.10, 0.45], [30], 3
        pols = [(0.5, 5.0), (1.0, 7.0), (1.0, 3.0), (1.5, 14.0)]
    L, T0 = design.RUN_LENGTH, design.T0
    delta = design.delta_unmet_days(L, T0)
    paths.make_dirs()
    sim = ensure_repo(); verify_patches_present(sim)
    seeds = list(range(SEED0, SEED0 + a.seeds))

    cfgs = []
    for nu, D, pol, s in itertools.product(nus, ds, pols, seeds):
        name = os.path.basename(rdir(nu, D, pol, s))
        cfgs.append(RunConfig(seed=s, days=L, out_dir=rdir(nu, D, pol, s), scenario_name=name,
                              ss_scale=pol[0], pipeline_mult=pol[1],
                              disrupt_capacity_scale=design.capacity_from_nu(nu),
                              disrupt_start_day=T0, disrupt_duration=D,
                              shock_count_scale=0.0, burst_rate_scale=0.0, slim=True))
    print(f"\nRunning {len(cfgs)} simulations ({len(nus)} nu x {len(ds)} D x {len(pols)} policies x {len(seeds)} seeds)...")
    run_batch(cfgs, n_workers=a.workers, sim_script=sim)

    rows = []
    for nu, D, pol, s in itertools.product(nus, ds, pols, seeds):
        m = window_metrics(load_daily_series(rdir(nu, D, pol, s)), T0, D, L)
        rows.append({"nu": nu, "D": D, "ss": pol[0], "pm": pol[1], "seed": s,
                     "unmet": m["unmet_demand_days"], "min_fill": m["min_fill"],
                     "triangle": m["triangle_days"], "censored": m["censored"]})
    df = pd.DataFrame(rows); df.to_csv(PER_RUN, index=False)

    cells, checks = [], []
    for (nu, D), g in df.groupby(["nu", "D"]):
        piv = g.pivot_table(index="seed", columns=["ss", "pm"], values="unmet")
        cols = list(piv.columns)
        pairs = list(itertools.combinations(cols, 2))
        res = [paired_test(piv[x], piv[y]) for x, y in pairs]
        diffs = np.array([abs(r[0]) for r in res]); sds = np.array([r[1] for r in res])
        pv = np.array([r[2] for r in res])
        sig = bh_reject(pv, Q)
        decidable = sig & (diffs >= delta)
        means = piv.mean()
        random_regret = float((means - means.min()).mean())
        n = len(piv)
        mdd_now = float(2.8 * np.median(sds) / np.sqrt(n))
        mdd_30 = float(2.8 * np.median(sds) / np.sqrt(30))
        cells.append({"nu": nu, "D": D, "n_seeds": n, "n_pairs": len(pairs),
                      "decidable_frac": float(decidable.mean()),
                      "best_policy": str(means.idxmin()), "best_unmet": float(means.min()),
                      "worst_unmet": float(means.max()), "random_advice_regret": random_regret,
                      "median_pair_sd": float(np.median(sds)),
                      "mdd_at_seeds": mdd_now, "mdd_at_30": mdd_30,
                      "mean_unmet": float(means.mean()),
                      "frac_runs_with_loss": float((g["unmet"] > 0.01).mean())})
    cell = pd.DataFrame(cells); cell.to_csv(CELLS, index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== Per-cell policy discrimination (delta = {delta:.2f} days of unmet demand, BH q={Q}) ===")
    print(cell.drop(columns=["best_policy"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    def add(st, msg):
        checks.append({"status": st, "message": msg}); print(f"  [{st}] {msg}")
    print("\n=== GATE 5: training signal ===")
    tr = cell[(cell.nu <= 0.15) & (cell.nu > 0) & (cell.D.isin(design.TRAIN_DURATIONS))]
    if len(tr):
        add("PASS" if (tr.mean_unmet.max() >= 1.0) else "WARN",
            f"Mild training cells: mean unmet demand ranges {tr.mean_unmet.min():.2f}-{tr.mean_unmet.max():.2f} days "
            f"(want some cell >= 1.0); share of runs with any loss: {tr.frac_runs_with_loss.min():.2f}-{tr.frac_runs_with_loss.max():.2f}")
        add("PASS" if (tr.median_pair_sd.max() > 0) else "WARN",
            "outcomes vary across policies/seeds inside the training band" if tr.median_pair_sd.max() > 0
            else "no variation between policies in the training band: nothing to learn about policy effects")
    print("\n=== GATE 6: policy power ===")
    live = cell[cell.mean_unmet > delta]
    for _, r in live.iterrows():
        pass
    if len(live):
        add("PASS" if live.decidable_frac.median() >= MIN_DECIDABLE_FRAC else "WARN",
            f"median decidable-pair share over cells with real loss = {live.decidable_frac.median():.2f} (want >= {MIN_DECIDABLE_FRAC})")
        add("PASS" if (live.random_advice_regret > delta).mean() >= 0.5 else "WARN",
            f"random-advice regret exceeds delta in {(live.random_advice_regret > delta).sum()} of {len(live)} cells with real loss")
        add("NOTE", f"minimum detectable difference: median {live.mdd_at_seeds.median():.2f} days at {a.seeds} seeds, "
                    f"{live.mdd_at_30.median():.2f} days at 30 seeds (delta = {delta:.2f})")
    else:
        add("WARN", "no cell loses more than delta: no policy comparison is possible at these severities")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for D, g in cell.groupby("D"):
        g = g.sort_values("nu")
        ax[0].plot(g.nu, g.decidable_frac, "o-", label=f"D={D}")
        ax[1].plot(g.nu, g.random_advice_regret, "o-", label=f"D={D}")
    ax[1].axhline(delta, color="k", ls="--", lw=0.8); ax[1].set_yscale("symlog", linthresh=0.1)
    ax[0].set_ylabel("share of policy pairs that are decidable"); ax[1].set_ylabel("random-advice regret (days of demand)")
    for x in ax:
        x.set_xlabel("overload nu"); x.grid(alpha=.3); x.legend()
    fig.tight_layout(); fig.savefig(FIG, dpi=200)
    json.dump({"delta": delta, "seeds": seeds, "checks": checks}, open(REPORT, "w"), indent=2)
    print(f"\nSaved: {PER_RUN}\n       {CELLS}\n       {REPORT}\n       {FIG}")
    print("\nGATES 5/6: done (WARN lines are design flags to read, not errors).")


if __name__ == "__main__":
    main()
