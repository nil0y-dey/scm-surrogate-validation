"""
run_all.py
-----------
Runs the full validation suite in the order the paper's Compute
Budget section presents it:

  1. patch_verification  -- confirms all three patches are no-ops
                             at released defaults (must pass before
                             anything else is trusted)
  2. gate1_integrity      -- determinism, same-demand-across-policies,
                             and exact patch-3 schedule (must pass)
  3. warmup_calibration   -- fixes the warm-up margin (Figure: warmup)
  4. stability_boundary   -- the permanent-regime design-rationale
                             finding (Figure: stability, companion note)
  5. severity_calibration -- pins the Moderate-band floor and
                             confirms/adjusts the run length L
  6. gate3_policy_warmup  -- is every policy viable in calm weather, and
                             does warm-up cover the slowest one?
  7. gate4_mapping_scan   -- what each severity level really does; which
                             (severity, duration) cells never recover

Each script also runs standalone (see each file's docstring for its
own CLI flags); this just sequences them with the settings used in
the paper, and stops immediately if patch verification fails.

Usage:
    python run_all.py            # full paper-scale run (~2 hours)
    python run_all.py --quick    # small smoke-scale run, to check the
                                  # pipeline before committing to the
                                  # full run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list) -> None:
    print(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFAILED: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                     help="Run a small smoke-scale pass instead of the "
                          "full paper-scale study, to check the pipeline.")
    ap.add_argument("--skip-patch-verification", action="store_true",
                     help="Skip step 1 (not recommended -- only for "
                          "iterating on steps 2-4 once step 1 has "
                          "already passed once in this session).")
    args = ap.parse_args()

    py = sys.executable

    if not args.skip_patch_verification:
        run([py, str(HERE / "00_patch_verification.py")])
    else:
        print("Skipping patch verification (--skip-patch-verification set).")

    run([py, str(HERE / "01_gate1_integrity.py")])

    if args.quick:
        run([py, str(HERE / "02_warmup_calibration.py"), "--days", "60", "--seeds", "2"])
        run([py, str(HERE / "04_severity_calibration.py"), "--candidates", "0.9,0.8",
             "--duration", "20", "--days", "160", "--seeds", "2"])
        run([py, str(HERE / "05_gate3_policy_warmup.py"), "--quick"])
        run([py, str(HERE / "06_gate4_mapping_scan.py"), "--quick"])
        run([py, str(HERE / "07_gate56_policy_power.py"), "--quick"])
        print("\n--quick does not include stability_boundary.py "
              "(it needs long horizons to be meaningful at all -- run "
              "it directly, at full scale, when you want it).")
    else:
        run([py, str(HERE / "02_warmup_calibration.py")])
        run([py, str(HERE / "03_stability_boundary.py")])
        run([py, str(HERE / "04_severity_calibration.py")])
        run([py, str(HERE / "05_gate3_policy_warmup.py")])
        run([py, str(HERE / "06_gate4_mapping_scan.py")])
        run([py, str(HERE / "07_gate56_policy_power.py")])

    print("\nAll requested steps completed.")


if __name__ == "__main__":
    main()
