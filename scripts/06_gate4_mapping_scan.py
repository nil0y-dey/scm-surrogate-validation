"""
06_gate4_mapping_scan.py
--------------------------
GATE 4. Turns the proposal's severity scale into real simulator behaviour.

For every severity level sigma (0, 0.1, ..., 1.0) and disruption duration
D (14 / 30 / 90 days), run a timed capacity cut and measure what happens.
The proposal's current mapping is   capacity scale c = 1 - 0.75 * sigma
(so Mild, sigma <= 0.2, means c >= 0.85). The scan answers:

  * How hard does each sigma really hit? (lowest fill rate, service lost)
  * How long does recovery take, and in which cells does a run of length L
    never recover (right-censoring)? Shown for L = 300, 400, 500.
  * Is the Mild TRAINING band (sigma <= 0.2) strong enough to teach a
    surrogate anything about disruption?
  * Is outcome monotone in severity and duration (as the proposal assumes)?
  * Does a simple formula predict recovery time? (a hypothesis check)

Disruptions start on day 100 (the warm-up margin). Policy = released
baseline (ss_scale 1.0, pipeline_mult 7). Calm, shock-free, like every
calibration run.

Note: runs are 500 days long and shorter L values are read off the first L
days. That is statistically equivalent to running length L, but not
bit-identical (the simulator's random draws depend on the run length).

Writes (small, committed): results/reports/gate4_per_seed.csv,
  gate4_scan_table.csv, gate4_report.json, results/figures/gate4_mapping.png

Usage:
    python scripts/06_gate4_mapping_scan.py [--days 500] [--seeds 3] [--workers N]
    python scripts/06_gate4_mapping_scan.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmval import paths
from scmval.metrics import CAPACITY_RATIO, fit_scale, fluid_ttr_raw, window_metrics
from scmval.recovery import RECOVERY_HOLD
from scmval.sim_runner import (RunConfig, ensure_repo, load_daily_series, run_batch,
                               verify_patches_present)

OUT_ROOT = str(paths.RUNS_DIR / "gate4")
PER_SEED_PATH = str(paths.REPORT_DIR / "gate4_per_seed.csv")
TABLE_PATH = str(paths.REPORT_DIR / "gate4_scan_table.csv")
REPORT_PATH = str(paths.REPORT_DIR / "gate4_report.json")
FIGURE_PATH = str(paths.FIG_DIR / "gate4_mapping.png")

T0 = 100                       # disruption start = warm-up margin
SLOPE = 0.75                   # proposal's mapping: c = 1 - SLOPE * sigma
SIGMAS = [round(0.1 * i, 1) for i in range(11)]
DURATIONS = [14, 30, 90]
HEADLINE_L = 300               # proposal's run length; headline metrics use days [T0, 300)
L_CANDIDATES = [300, 400, 500]
SIGMA_CRIT = (1 - 0.775) / SLOPE   # sigma at which c equals the permanent-cut stability threshold

# PROPOSED sanity thresholds (edit here; they are not pre-registered results)
CONTROL_MAX_TRIANGLE = 0.5     # no-disruption control must lose < 0.5 'full-service days'
MILD_SIGNAL_MIN_DEMAND_DAYS = 1.0   # Mild training cells should lose >= 1 day of demand
MONOTONE_TOL_DAYS = 0.25       # tolerated dip when checking monotonicity (noise allowance)


def capacity_scale(sigma: float) -> float:
    return round(1.0 - SLOPE * sigma, 6)


def run_dir(sigma, duration, seed):
    return os.path.join(OUT_ROOT, f"s{sigma:g}_D{duration}_seed{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 4 severities x 2 durations, 1 seed, 300 days")
    args = ap.parse_args()

    sigmas, durations = SIGMAS, DURATIONS
    if args.quick:
        sigmas, durations, args.seeds, args.days = [0.0, 0.2, 0.5, 0.8], [14, 30], 1, 300
    l_cands = [L for L in L_CANDIDATES if L <= args.days] or [args.days]
    headline_l = HEADLINE_L if HEADLINE_L <= args.days else args.days
    if args.days <= T0 + max(durations) + RECOVERY_HOLD:
        raise SystemExit("--days is too short for the longest disruption plus recovery")

    paths.make_dirs()
    sim_script = ensure_repo()
    verify_patches_present(sim_script)

    seeds = list(range(2025, 2025 + args.seeds))
    configs = []
    for seed in seeds:
        # sigma = 0 control: no disruption at all (run once per seed, reused for every duration)
        configs.append(RunConfig(
            seed=seed, days=args.days, out_dir=run_dir(0.0, 0, seed),
            scenario_name=f"s0_ctrl_{seed}", shock_count_scale=0.0,
            burst_rate_scale=0.0, slim=True))
        for sigma in sigmas:
            if sigma == 0.0:
                continue
            for D in durations:
                configs.append(RunConfig(
                    seed=seed, days=args.days, out_dir=run_dir(sigma, D, seed),
                    scenario_name=f"s{sigma:g}_D{D}_{seed}",
                    disrupt_capacity_scale=capacity_scale(sigma),
                    disrupt_start_day=T0, disrupt_duration=D,
                    shock_count_scale=0.0, burst_rate_scale=0.0, slim=True))
    print(f"\nRunning {len(configs)} simulations ({len(sigmas)} severities x "
          f"{len(durations)} durations x {len(seeds)} seeds + controls, {args.days} days each)...")
    run_batch(configs, n_workers=args.workers, sim_script=sim_script)

    # ---------------- per-seed metrics ----------------
    rows = []
    for seed in seeds:
        ctrl = load_daily_series(run_dir(0.0, 0, seed))
        for sigma in sigmas:
            for D in durations:
                daily = ctrl if sigma == 0.0 else load_daily_series(run_dir(sigma, D, seed))
                dur = D                      # restoration day is the same for the control, for comparability
                row = {"sigma": sigma, "c": capacity_scale(sigma), "D": D, "seed": seed}
                head = window_metrics(daily, T0, dur, headline_l)
                for k in ("baseline_fill", "min_fill", "triangle_days",
                          "unmet_demand_days", "peak_backlog_days"):
                    row[k] = head[k]
                for L in l_cands:
                    m = window_metrics(daily, T0, dur, L)
                    row[f"ttr_{L}"] = m["ttr"]
                    row[f"ttr_lb_{L}"] = m["ttr_lower_bound"]
                    row[f"cens_{L}"] = m["censored"]
                rows.append(row)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(PER_SEED_PATH, index=False)

    # ---------------- per-cell table ----------------
    agg = {"min_fill": "median", "triangle_days": "median", "unmet_demand_days": "median",
           "peak_backlog_days": "median"}
    for L in l_cands:
        agg[f"ttr_lb_{L}"] = "median"
        agg[f"cens_{L}"] = "mean"
    cell = per_seed.groupby(["sigma", "c", "D"]).agg(agg).reset_index()
    # nu = "overload": the share of demand the cut network cannot carry during the
    # event, max(0, 1 - 1.29c). It is 0 for any cut that capacity can absorb (c >= 0.775).
    cell.insert(2, "nu", [max(0.0, 1.0 - CAPACITY_RATIO * c) for c in cell["c"]])
    cell.to_csv(TABLE_PATH, index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(f"\n=== Scan table (medians over {len(seeds)} seed(s); metrics over days "
          f"{T0}..{headline_l - 1}) ===")
    print("   triangle = lost 'full-service days'; unmet/peak in days of demand;")
    print("   ttr_lb = median recovery days (a LOWER BOUND when runs are censored);")
    print("   cens = share of runs that never recovered within the run length L\n")
    show = cell.copy()
    show.columns = [c.replace("ttr_lb_", "ttr_L").replace("cens_", "cens_L") for c in show.columns]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------- checks ----------------
    checks = []

    def add(status, msg):
        checks.append((status, msg)); print(f"  [{status}] {msg}")

    print("\n=== Checks ===")
    ctrl_tri = float(per_seed[per_seed.sigma == 0.0]["triangle_days"].max())
    add("PASS" if ctrl_tri < CONTROL_MAX_TRIANGLE else "FAIL",
        f"no-disruption control loses {ctrl_tri:.3f} full-service days (limit {CONTROL_MAX_TRIANGLE})")

    viol = []
    for D in durations:
        sub = cell[cell.D == D].sort_values("sigma")
        tri = sub["triangle_days"].to_numpy()
        for i in range(1, len(tri)):
            if tri[i] < tri[i - 1] - MONOTONE_TOL_DAYS:
                viol.append(f"D={D}: triangle falls from sigma={sub.sigma.iloc[i-1]:g} "
                            f"({tri[i-1]:.1f}) to sigma={sub.sigma.iloc[i]:g} ({tri[i]:.1f})")
    for sigma in sigmas:
        sub = cell[cell.sigma == sigma].sort_values("D")
        tri = sub["triangle_days"].to_numpy()
        for i in range(1, len(tri)):
            if tri[i] < tri[i - 1] - MONOTONE_TOL_DAYS:
                viol.append(f"sigma={sigma:g}: triangle falls from D={sub.D.iloc[i-1]} "
                            f"({tri[i-1]:.1f}) to D={sub.D.iloc[i]} ({tri[i]:.1f})")
    if viol:
        add("WARN", f"resilience triangle is not monotone in severity/duration in {len(viol)} place(s):")
        for v in viol:
            print(f"        - {v}")
    else:
        add("PASS", "resilience triangle is monotone in severity and in duration "
                    f"(within {MONOTONE_TOL_DAYS} days of noise)")

    mild = cell[(cell.sigma > 0) & (cell.sigma <= 0.2) & (cell.D.isin([d for d in durations if d <= 30]))]
    if len(mild):
        best = float(mild["unmet_demand_days"].max())
        add("PASS" if best >= MILD_SIGNAL_MIN_DEMAND_DAYS else "WARN",
            f"Mild training band (sigma<=0.2, D<=30): most service lost in any cell = "
            f"{best:.2f} days of demand (want >= {MILD_SIGNAL_MIN_DEMAND_DAYS}); "
            + ("enough disruption signal to learn from" if best >= MILD_SIGNAL_MIN_DEMAND_DAYS
               else "training data will look almost like calm weather"))
        worst_min = float(mild["min_fill"].min())
        print(f"        lowest fill rate reached anywhere in the Mild band: {worst_min:.3f}")

    # ---------------- censoring forecast ----------------
    print("\n=== Share of runs that NEVER recover within run length L (censored) ===")
    for L in l_cands:
        piv = cell.pivot(index="sigma", columns="D", values=f"cens_{L}")
        piv.columns = [f"D={d}" for d in piv.columns]
        print(f"\nL = {L}:")
        print(piv.to_string(float_format=lambda v: f"{v:.2f}"))

    # ---------------- fluid-model hypothesis check ----------------
    big = max(l_cands)
    use = per_seed[(per_seed.sigma > 0) & (~per_seed[f"cens_{big}"].astype(bool))].copy()
    use["pred_raw"] = [fluid_ttr_raw(c, D) for c, D in zip(use["c"], use["D"])]
    use = use[use.pred_raw > 0]
    fluid = {}
    print(f"\n=== Hypothesis check: does recovery time follow  TTR ~ k * D * overload / spare ? ===")
    if len(use) >= 4:
        k, r2 = fit_scale(use["pred_raw"], use[f"ttr_{big}"].astype(float))
        fluid = {"k": k, "r2": r2, "n": int(len(use))}
        print(f"   fitted k = {k:.2f}   R^2 = {r2:.2f}   (n = {len(use)} uncensored runs with overload)")
        print("   (overload = max(0, 1 - 1.29c); spare = 0.29. k is a calibration factor; an R^2 "
              "close to 1 means the simple formula tracks recovery time up to that factor.)")
        agree = tot = 0
        for (sg, D), g in per_seed[per_seed.sigma > 0].groupby(["sigma", "D"]):
            c = capacity_scale(sg)
            pred_cens = (T0 + D + k * fluid_ttr_raw(c, D) + RECOVERY_HOLD) > headline_l
            obs_cens = g[f"cens_{headline_l}"].astype(float).mean() >= 0.5
            tot += 1; agree += int(pred_cens == obs_cens)
        fluid["censoring_agreement"] = f"{agree}/{tot}"
        print(f"   predicts 'censored at L={headline_l}' correctly in {agree} of {tot} cells")
    else:
        print("   not enough uncensored runs with overload to fit (run the full scan).")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {14: "#2b6cb0", 30: "#dd6b20", 90: "#c53030"}
    for D in durations:
        sub = cell[cell.D == D].sort_values("sigma")
        axes[0].plot(sub.sigma, sub.min_fill, "o-", color=colors.get(D), label=f"D={D}")
        axes[1].plot(sub.sigma, sub.unmet_demand_days, "o-", color=colors.get(D), label=f"D={D}")
        cens = sub[f"cens_{headline_l}"] >= 0.5
        axes[2].plot(sub.sigma, sub[f"ttr_lb_{headline_l}"], "-", color=colors.get(D), label=f"D={D}")
        axes[2].scatter(sub.sigma[~cens], sub[f"ttr_lb_{headline_l}"][~cens], color=colors.get(D), zorder=3)
        axes[2].scatter(sub.sigma[cens], sub[f"ttr_lb_{headline_l}"][cens], facecolors="none",
                        edgecolors=colors.get(D), zorder=3)
    axes[0].set_ylabel("lowest daily fill rate"); axes[0].set_title(f"How low does service go? (L={headline_l})")
    axes[1].set_ylabel("service lost (days of demand)"); axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].set_title("Total unmet demand")
    axes[2].set_ylabel("days to recover"); axes[2].set_title("Recovery time (hollow = mostly censored: lower bound)")
    for ax in axes:
        ax.set_xlabel("severity sigma  (capacity scale c = 1 - 0.75 sigma)")
        for x in (0.2, 0.5, 0.8):
            ax.axvline(x, color="gray", ls=":", lw=0.8)
        ax.axvline(SIGMA_CRIT, color="black", ls="--", lw=0.9)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, title="dotted = proposal bins;\ndashed = permanent-cut\nstability threshold", title_fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE_PATH, dpi=200)

    report = {
        "settings": {"days": args.days, "seeds": seeds, "t0": T0, "slope": SLOPE,
                     "L_candidates": l_cands, "headline_L": headline_l, "quick": args.quick},
        "fluid_model": fluid,
        "checks": [{"status": s, "message": m} for s, m in checks],
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {PER_SEED_PATH}\n       {TABLE_PATH}\n       {REPORT_PATH}\n       {FIGURE_PATH}")
    if any(s == "FAIL" for s, _ in checks):
        print("\nGATE 4: FAILED (see above)."); sys.exit(1)
    print("\nGATE 4: PASSED (WARN lines, if any, are design flags to read, not errors).")


if __name__ == "__main__":
    main()
