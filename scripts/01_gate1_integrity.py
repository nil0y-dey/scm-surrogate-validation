"""
gate1_integrity.py
-------------------
GATE 1 (run this first, takes ~1-2 minutes). Checks the simulator is
trustworthy enough to collect data from. Four questions:

  A. Determinism      -- same settings + same seed twice => identical
                         output? (If not, nothing is reproducible.)
  B. Prefix identity  -- a disrupted run and an undisrupted run with the
                         same seed must be identical BEFORE the
                         disruption starts (cause cannot precede effect),
                         and must differ afterwards (patch 3 does
                         something).
  C. Demand pairing   -- the customer demand must be identical across
                         different policies and with/without disruption
                         for the same seed. The study compares policies
                         "on the same demand" (common random numbers);
                         this proves it.
  D. Patch-3 schedule -- directly watches the simulator and confirms the
                         capacity factor is exactly c on days
                         [start, start+duration) and exactly 1.0 on every
                         other day, on every edge; and that duration=0
                         is a no-op.

Prints PASS / FAIL / NOTE per check; exits non-zero on any FAIL.
Writes gate1_report.json.

Usage:
    python gate1_integrity.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

from scmval import paths
from scmval.sim_runner import RunConfig, ensure_repo, run_one, verify_patches_present

OUT_ROOT = str(paths.RUNS_DIR / "gate1")
REPORT_PATH = str(paths.REPORT_DIR / "gate1_report.json")

SEED = 2025
DAYS = 300
START, DUR, CSCALE = 100, 30, 0.5     # the disruption used in checks B, C
DISRUPT = dict(disrupt_capacity_scale=CSCALE, disrupt_start_day=START,
               disrupt_duration=DUR)

results = []   # (status, name, detail)


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_daily(d: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(d, "daily_records.csv"))
    return df.sort_values(["day", "item"]).reset_index(drop=True)


def make_cfg(name: str, **kw) -> RunConfig:
    # Shock-free like every calibration run, so differences we see are
    # caused only by the thing we changed.
    base = dict(seed=SEED, days=DAYS, out_dir=os.path.join(OUT_ROOT, name),
                scenario_name=name, shock_count_scale=0.0, burst_rate_scale=0.0)
    base.update(kw)
    return RunConfig(**base)


# --------------------------------------------------------------------------
# Part 1: black-box checks A, B, C (real command-line runs)
# --------------------------------------------------------------------------
def run_blackbox(sim_script: str) -> None:
    cfgs = {
        "base_a":   make_cfg("base_a"),
        "base_b":   make_cfg("base_b"),                       # exact repeat
        "disrupt":  make_cfg("disrupt", **DISRUPT),
        "lean":     make_cfg("lean", ss_scale=0.5),           # other policy
        "buffered": make_cfg("buffered", ss_scale=1.5),       # other policy
        "pipe3":    make_cfg("pipe3", pipeline_mult=3.0),     # other policy
        # the same three policies, but UNDER the disruption: this is
        # where policies can actually differ
        "d_lean":     make_cfg("d_lean", ss_scale=0.5, **DISRUPT),
        "d_buffered": make_cfg("d_buffered", ss_scale=1.5, **DISRUPT),
        "d_pipe3":    make_cfg("d_pipe3", pipeline_mult=3.0, **DISRUPT),
    }
    print(f"\nRunning {len(cfgs)} short simulations ({DAYS} days each)...")
    for name, cfg in cfgs.items():
        msg = run_one(cfg, sim_script)
        print(f"  {name}: {msg}")
        if msg.startswith("FAIL"):
            record("FAIL", "simulation ran", msg)
            return

    d = {k: v.out_dir for k, v in cfgs.items()}

    print("\nA. Determinism")
    same = sha(os.path.join(d["base_a"], "daily_records.csv")) == \
        sha(os.path.join(d["base_b"], "daily_records.csv"))
    record("PASS" if same else "FAIL", "same settings + seed => identical daily_records.csv")

    print("\nB. Prefix identity (disrupted vs undisrupted, same seed)")
    base, dis = load_daily(d["base_a"]), load_daily(d["disrupt"])
    pre_b, pre_d = base[base.day < START], dis[dis.day < START]
    pre_same = pre_b.equals(pre_d)
    record("PASS" if pre_same else "FAIL",
           f"days 0..{START - 1} identical before the disruption starts")
    post_b = base[base.day >= START].drop(columns=["day", "item"]).to_numpy()
    post_d = dis[dis.day >= START].drop(columns=["day", "item"]).to_numpy()
    differs = not np.array_equal(post_b, post_d)
    record("PASS" if differs else "FAIL",
           f"days >= {START} differ (the disruption has an effect)")
    if differs:
        win = lambda df: df[(df.day >= START) & (df.day < START + DUR + 60)]
        fb = win(base).served_from_stock.sum() / win(base).demand.sum()
        fd = win(dis).served_from_stock.sum() / win(dis).demand.sum()
        record("NOTE", "fill rate over disruption + 60 days",
               f"undisrupted {fb:.3f} vs disrupted {fd:.3f}")

    print("\nC. Demand pairing across policies / disruption (same seed)")
    ref = base[["day", "item", "demand"]]
    others = ("disrupt", "lean", "buffered", "pipe3",
              "d_lean", "d_buffered", "d_pipe3")
    for name in others:
        other = load_daily(d[name])[["day", "item", "demand"]]
        ok = ref.equals(other)
        record("PASS" if ok else "FAIL", f"demand identical: base vs {name}")
    ds_ref = np.load(os.path.join(d["base_a"], "demand_signals.npy"))
    ds_ok = all(np.array_equal(ds_ref, np.load(os.path.join(d[n], "demand_signals.npy")))
                for n in others)
    record("PASS" if ds_ok else "FAIL", "demand intensity (demand_signals.npy) identical across runs")
    record("NOTE", "lead-time draws are NOT paired across policies",
           "by design of the simulator: lead times use a separate random stream "
           "that is only consumed when a node reorders, so different policies "
           "see different lead-time draws. Demand (the larger source of noise) "
           "is paired; say this in the paper.")

    print("\nPolicy knobs actually change outcomes (sanity, under the disruption)")
    note_calm = load_daily(d["buffered"])
    note_calm = note_calm[note_calm.day >= 100]
    record("NOTE", "in calm weather every policy serves ~100% of demand",
           f"buffered (ss_scale=1.5) calm fill rate = "
           f"{note_calm.served_from_stock.sum() / note_calm.demand.sum():.4f}; "
           "policies can only be told apart under disruption")

    def window_fill(name):
        o = load_daily(d[name])
        o = o[(o.day >= START) & (o.day < START + DUR + 60)]
        return o.served_from_stock.sum() / o.demand.sum()

    ref_fr = window_fill("disrupt")
    for name in ("d_lean", "d_buffered", "d_pipe3"):
        fr = window_fill(name)
        record("PASS" if abs(fr - ref_fr) > 1e-9 else "FAIL",
               f"{name} gives a different outcome from the default policy under disruption",
               f"fill rate {fr:.4f} vs default {ref_fr:.4f}")


# --------------------------------------------------------------------------
# Part 2: white-box check D (patch-3 schedule), watching the real simulator
# --------------------------------------------------------------------------
def watch_capacity_schedule(sim_script: str, days: int, start: int, dur: int,
                            c: float):
    """Run the real simulator for `days` days and return, per day, the
    capacity factor it applied AND the realised per-edge container volume
    as a fraction of its full volume."""
    # Importing the simulator must not leave __pycache__ files inside the
    # simulator submodule (git would then report it as modified).
    sys.dont_write_bytecode = True
    sim_dir = os.path.dirname(sim_script)
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    import importlib
    mod = importlib.import_module(os.path.splitext(os.path.basename(sim_script))[0])

    with contextlib.redirect_stdout(io.StringIO()):
        sim, net, items, dsig = mod.build_example_simulation_from_adjacency(
            seed=SEED, horizon_days=days, pipeline_multiplier=7.0,
            streaming_out_dir=None, packing="greedy",
            scenario={"shock_count_scale": 0.0, "burst_rate_scale": 0.0,
                      "disrupt_capacity_scale": c, "disrupt_start_day": start,
                      "disrupt_duration": dur})
    factors, realised = [], []
    orig = net.reset_daily_edges

    def spy(capacity_factor: float = 1.0):
        orig(capacity_factor)
        factors.append(capacity_factor)
        fr = [e.daily_containers[0] / e.container_volume
              for e in net.edges.values()
              if e.num_containers_per_day > 0 and e.container_volume > 0]
        realised.append(fr)

    net.reset_daily_edges = spy
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        sim.run()
    return factors, realised


def run_whitebox(sim_script: str) -> None:
    print("\nD. Patch-3 schedule (watching the simulator's own capacity factor)")
    days, start, dur, c = 60, 20, 10, 0.5
    try:
        factors, realised = watch_capacity_schedule(sim_script, days, start, dur, c)
    except Exception as exc:                                  # pragma: no cover
        record("FAIL", "could not watch the simulator", repr(exc))
        return

    record("PASS" if len(factors) == days else "FAIL",
           f"capacity reset called once per day ({len(factors)} calls, expected {days})")
    expect = [c if start <= t < start + dur else 1.0 for t in range(days)]
    ok = len(factors) == days and all(abs(a - b) < 1e-12 for a, b in zip(factors, expect))
    record("PASS" if ok else "FAIL",
           f"factor == {c} exactly on days [{start}, {start + dur}) and 1.0 elsewhere")
    for t in (start - 1, start, start + dur - 1, start + dur):
        if 0 <= t < len(factors):
            record("NOTE", f"day {t}", f"factor applied = {factors[t]}")
    if realised and realised[0]:
        edge_ok = all(all(abs(x - expect[t]) < 1e-9 for x in realised[t])
                      for t in range(len(realised)))
        record("PASS" if edge_ok else "FAIL",
               "every edge's real container volume matches the factor, every day")
    else:
        record("NOTE", "per-edge volume check skipped",
               "could not read edges from the network object")

    # duration = 0 must be a no-op even if the scale is not 1.0
    f0, _ = watch_capacity_schedule(sim_script, 30, 5, 0, 0.3)
    record("PASS" if all(x == 1.0 for x in f0) else "FAIL",
           "duration=0 is a no-op even with capacity_scale=0.3")


def main() -> None:
    paths.make_dirs()
    sim_script = ensure_repo()
    verify_patches_present(sim_script)
    os.makedirs(OUT_ROOT, exist_ok=True)

    run_blackbox(sim_script)
    run_whitebox(sim_script)

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_pass = sum(1 for s, _, _ in results if s == "PASS")
    print(f"\n=== GATE 1: {n_pass} passed, {n_fail} failed ===")
    with open(REPORT_PATH, "w") as f:
        json.dump([{"status": s, "check": n, "detail": dtl} for s, n, dtl in results],
                  f, indent=2)
    print(f"Report saved to {REPORT_PATH}")
    if n_fail:
        print("GATE 1 FAILED -- do not collect data until this is fixed.")
        sys.exit(1)
    print("GATE 1 PASSED.")


if __name__ == "__main__":
    main()
