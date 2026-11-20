"""
12_generate_B.py
-----------------
GENERATES THE REAL STUDY DATA for Simulator B, one block at a time. Run it AFTER the design
lock (git tag `design-lock`). Blocks, in the order to run them:

    train        2,500 runs   Tier 1   surrogate training data (Mild band)
    calibration    600 runs   Tier 1   held-out, same distribution as train (conformal intervals)
    test         5,400 runs   Tier 1   the full test grid, first 15 seeds per (cell, policy)
    calm           500 runs   Tier 2   ablation: no disruption at all
    wide           800 runs   Tier 2   sensitivity: 0.15 < nu <= 0.25
    placebo      1,200 runs   Tier 2   independent repeat of 4 test cells (noise floor)

Safe to stop and restart at any moment: a finished run is never repeated (it is detected by its
saved file). `--max-hours` makes the script stop cleanly by itself, finish the table, and tell you
to run it again. Each run is thinned to its small daily file, so 10,000 runs take a few GB on Drive.

After the runs finish (or whenever you stop) it writes results/tables/B_<block>.csv: one row per
finished run with unmet demand, minimum fill rate, resilience triangle, peak backlog and recovery time.

Usage:
    python scripts/12_generate_B.py --block train --max-hours 3
    python scripts/12_generate_B.py --block test --seeds 15:30 --cells 0.25:30,0.45:14   # deepening
    python scripts/12_generate_B.py --block train --limit 6                              # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from scmval import datagen, design, paths
from scmval.metrics import window_metrics
from scmval.sim_runner import ensure_repo, load_daily_series, run_one, verify_patches_present


def parse_cells(s):
    if not s:
        return None
    return [(float(a), int(b)) for a, b in (x.split(":") for x in s.split(","))]


def build_table(block, out_csv):
    rows = []
    for meta, cfg in datagen.table_plan(block):
        if not os.path.exists(os.path.join(cfg.out_dir, "daily_records.csv")):
            continue
        calm = cfg.disrupt_duration == 0
        m = window_metrics(load_daily_series(cfg.out_dir), design.T0, 0 if calm else meta["D"], design.RUN_LENGTH)
        rows.append({**{k: meta[k] for k in ("block", "nu", "D", "ss", "pm", "seed")},
                     "unmet": m["unmet_demand_days"], "min_fill": m["min_fill"], "triangle": m["triangle_days"],
                     "peak_backlog": m["peak_backlog_days"], "ttr": m["ttr"], "censored": m["censored"],
                     "baseline_fill": m["baseline_fill"]})
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=datagen.BLOCKS)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this many hours")
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs (smoke test)")
    ap.add_argument("--seeds", default=None, help="test/placebo only, e.g. 15:30")
    ap.add_argument("--cells", default=None, help="test/placebo only, e.g. 0.25:30,0.45:14")
    a = ap.parse_args()
    lo = hi = None
    if a.seeds:
        lo, hi = (int(x) for x in a.seeds.split(":"))
    paths.make_dirs()
    sim = ensure_repo(); verify_patches_present(sim)
    runs = datagen.plan(a.block, lo, hi, parse_cells(a.cells))
    if a.limit:
        runs = runs[: a.limit]
    todo = [(m, c) for m, c in runs if not os.path.exists(os.path.join(c.out_dir, "daily_records.csv"))]
    print(f"Block '{a.block}': {len(runs)} runs planned, {len(runs) - len(todo)} already finished, {len(todo)} to do. "
          f"Using {a.workers} parallel workers.")
    t_start = time.time()
    deadline = None if a.max_hours is None else t_start + a.max_hours * 3600
    done = fail = 0
    chunk = max(4, 6 * a.workers)
    stopped_early = False
    for k in range(0, len(todo), chunk):
        if deadline and time.time() + 90 > deadline:          # no time for another chunk
            stopped_early = True
            break
        batch = todo[k:k + chunk]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_one, c, sim) for _, c in batch]
            for f in as_completed(futs):
                msg = f.result()
                if msg.startswith("FAIL"):
                    fail += 1
                    if fail <= 5:
                        print("  " + msg[:300])
                else:
                    done += 1
        el = (time.time() - t_start) / 3600
        rate = done / max(el, 1e-9)
        left = len(todo) - done - fail
        print(f"  finished {done}/{len(todo)}  (failed {fail})  elapsed {el:.2f} h  "
              f"~{left / rate:.1f} h to go at this speed" if done else f"  failed {fail}")
        sys.stdout.flush()
    print("\nBuilding the table of finished runs...")
    df = build_table(a.block, paths.RESULTS_DIR / "tables" / f"B_{a.block}.csv")
    n_all = len(datagen.table_plan(a.block))
    print(f"results/tables/B_{a.block}.csv: {len(df)} finished runs (block total {n_all})")
    if len(df):
        print(f"  unmet demand (days): mean {df.unmet.mean():.2f}, max {df.unmet.max():.1f};  "
              f"censored recoveries: {int(df.censored.sum())};  any missing values: {bool(df[['unmet','min_fill','triangle']].isna().any().any())}")
        print(df.groupby(["nu", "D"]).unmet.agg(["count", "mean"]).round(2).head(12).to_string()
              if a.block in ("test", "placebo") else
              df.groupby("D").unmet.agg(["count", "mean", "std"]).round(2).to_string())
    if fail:
        print(f"\n{fail} runs FAILED. Run the same command again; if they fail again, send me the lines above.")
    elif stopped_early or len(todo) > done:
        print(f"\nTIME LIMIT REACHED. {len(todo) - done} runs remain. Run this same cell again to continue.")
    else:
        print(f"\nBLOCK '{a.block}' COMPLETE." + (" (smoke test only: --limit was used)" if a.limit else ""))


if __name__ == "__main__":
    main()
