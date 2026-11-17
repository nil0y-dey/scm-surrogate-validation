"""
patch_verification.py
----------------------
Confirms all three disclosed patches are mathematical no-ops at the
simulator's released default configuration, by running the full
7,300-day baseline (C=50, seed=2025, pipeline_mult=7) with every
patch-controlled parameter left at its default, and checking the
result against ISOMORPH's own published console output.

This is the test the paper refers to wherever a patch is described
as "a no-op at released defaults" -- it is a computational check
against known reference values, not an assertion about the source
code. Run once per commit to the fork; ~7 minutes on a single CPU.
"""
from __future__ import annotations

import sys

import pandas as pd

from scmval import paths
from scmval.sim_runner import (
    RELEASED_BASELINE_REFERENCE,
    RunConfig,
    ensure_repo,
    run_one,
    verify_patches_present,
)

FILL_RATE_TOL = 0.001  # allows for the published reference's own rounding


def main() -> None:
    paths.make_dirs()
    sim_script = ensure_repo()
    verify_patches_present(sim_script)

    print("\nRunning full 7,300-day baseline reproduction "
          "(every patch-controlled parameter at its default)...")
    cfg = RunConfig(
        seed=2025,
        days=7300,
        out_dir=str(paths.RUNS_DIR / "verify_baseline_full"),
        scenario_name="baseline_verify",
        # containers_scale=1.0, disrupt_capacity_scale=1.0,
        # disrupt_duration=0 all left at their dataclass defaults --
        # that is the condition under test.
    )
    print(run_one(cfg, sim_script))

    svc = pd.read_csv(f"{cfg.out_dir}/service_summary.csv")
    total_demand = int(svc["total_demand"].sum())
    total_served = int(svc["served_from_stock"].sum())
    total_backlog = int(svc["new_backlog_added"].sum())
    fr = svc["fill_rate_stock_only"]

    observed = {
        "total_demand": total_demand,
        "total_served": total_served,
        "total_backlog": total_backlog,
        "fill_rate_mean": round(float(fr.mean()), 3),
        "fill_rate_min": round(float(fr.min()), 3),
        "fill_rate_max": round(float(fr.max()), 3),
    }

    print("\n=== No-op verification: patched simulator vs. published baseline ===")
    all_ok = True
    for key, ref in RELEASED_BASELINE_REFERENCE.items():
        obs = observed[key]
        if isinstance(ref, int):
            ok = obs == ref
        else:
            ok = abs(obs - ref) < FILL_RATE_TOL
        all_ok = all_ok and ok
        print(f"  {key:<16} observed={obs:<15} reference={ref:<15} "
              f"{'PASS' if ok else 'FAIL'}")

    if all_ok:
        print("\nPASS: the patched simulator reproduces the released "
              "baseline exactly. All three patches confirmed as no-ops.")
    else:
        print("\nFAIL: patched output diverges from the published "
              "baseline. Do not trust downstream calibration results "
              "until this passes -- check the fork for unintended "
              "changes beyond the three disclosed patches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
