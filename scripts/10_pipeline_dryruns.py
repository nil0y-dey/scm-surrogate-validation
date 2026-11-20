"""
10_pipeline_dryruns.py
-----------------------
PIPELINE DRY RUNS (about a minute; NO simulator runs).

Before spending compute on the real data we check that the ANALYSIS machinery gives the
right answer when we KNOW the right answer. Stand-in data are built from the Gate 5/6 pilot
table: for every (severity, duration) cell we keep the pilot's policy means and its measured
noise (a part shared by all policies on the same seed, and a part per run), and draw as many
fresh seeds as we like. Four checks:

 D1 No-effect check   All policies made identical (true differences = 0). The pipeline must call
                      "decidable difference" in at most about 10% of cells (that is what the
                      Benjamini-Hochberg level q=0.10 promises).
 D2 Placebo floor     Compare the stand-in simulator with an independent repeat of itself. Any
                      "rank flip" it reports is pure noise; this is the noise floor of the flip count.
 D3 Planted flips     A fake surrogate whose ranking is wrong on known policy pairs. How many of the
                      planted flips does the pipeline find, and does it invent any? Repeated at 5,
                      10 and 30 seeds.
 D4 Conformal         A fake surrogate with a known error. Interval coverage should be near 90% on
                      data like the calibration data, and should visibly drop when its error grows
                      (as it would outside the training band).

These test the arithmetic of the pipeline, not real surrogates.

Usage:  python scripts/10_pipeline_dryruns.py [--reps 200]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scmval import analysis, design, paths
from scmval.metrics import bh_reject

Q = 0.10


def cell_models(per_run: pd.DataFrame, pcols=("ss", "pm")):
    """Per cell: policy means, shared-seed SD, per-run SD, estimated from the pilot table."""
    out = {}
    for (nu, D), g in per_run.groupby(["nu", "D"]):
        piv = g.pivot_table(index="seed", columns=list(pcols), values="unmet")
        X = piv.to_numpy(); n, k = X.shape
        mu = X.mean(0)
        R = X - mu
        a = R.mean(1)
        e = R - a[:, None]
        var_e = (e ** 2).sum() / ((n - 1) * (k - 1))
        var_a = max(0.0, a.var(ddof=1) - var_e / k)
        out[(nu, D)] = {"mu": mu, "sd_seed": float(np.sqrt(var_a)), "sd_run": float(np.sqrt(var_e))}
    return out


def draw(model, n_seeds, rng, mu=None):
    mu = model["mu"] if mu is None else mu
    k = len(mu)
    a = rng.normal(0, model["sd_seed"], n_seeds)
    e = rng.normal(0, model["sd_run"], (n_seeds, k))
    return np.clip(mu[None, :] + a[:, None] + e, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--csv", default=str(paths.REPORT_DIR / "gate56_per_run.csv"))
    a = ap.parse_args()
    paths.make_dirs()
    rng = np.random.default_rng(20261120)
    delta = design.delta_unmet_days()
    models = cell_models(pd.read_csv(a.csv))
    cells = list(models)
    checks, report = [], {}

    def add(s, msg):
        checks.append({"status": s, "message": msg}); print(f"  [{s}] {msg}")

    # D1 -------------------------------------------------------------------
    print("D1. No true differences between policies")
    n_seeds = 30
    any_rej = any_dec = fam = 0
    for key in cells:
        m = models[key]
        flat = np.full_like(m["mu"], m["mu"].mean())
        for _ in range(a.reps):
            X = draw(m, n_seeds, rng, flat)
            i, j, d, dec = analysis.decidable_pairs(X, delta, Q)
            _, _, _, _, p = analysis.pair_stats(X)
            any_rej += bool(bh_reject(p, Q).any()); any_dec += bool(dec.any()); fam += 1
    r1, r2 = any_rej / fam, any_dec / fam
    report["D1"] = {"families": fam, "share_with_any_significant_pair": r1, "share_with_any_decidable_pair": r2}
    print(f"  {fam} simulated cells: {100*r1:.1f}% show a significant pair, {100*r2:.1f}% show a decidable pair")
    add("PASS" if r1 <= Q + 0.02 else "FAIL",
        f"with no true effect, {100*r1:.1f}% of cells report a significant policy difference (promise: <= {100*Q:.0f}%, tolerance +2 points)")
    add("PASS" if r2 <= r1 else "FAIL", f"requiring the practical margin delta cuts that to {100*r2:.1f}%")

    # D2 / D3 -------------------------------------------------------------
    print("\nD2. Placebo floor and D3. Planted flips")
    rows = []
    for ns in (5, 10, 30):
        pl_flips = pl_dec = 0
        tp = fn = fp = near = 0
        top1_testable = top1_flip = 0
        for key in cells:
            m = models[key]; mu = m["mu"]
            for r in range(max(1, a.reps // 10)):
                truth = draw(m, ns, rng)
                # D2: independent repeat as the "surrogate"
                rep = draw(m, ns, rng)
                s = analysis.flip_summary(truth, rep.mean(0), delta, Q)
                pl_flips += s["n_flips"]; pl_dec += s["n_decidable"]
                top1_testable += s["top1_testable"]; top1_flip += s["top1_flip"]
                # D3: plant swaps on 4 disjoint pairs whose true gap is >= delta
                order = np.argsort(mu); k = len(mu)
                pred = mu.copy(); used = set(); planted = 0
                for x in rng.permutation(k):
                    for y in rng.permutation(k):
                        if x < y and x not in used and y not in used and abs(mu[x] - mu[y]) >= delta:
                            pred[x], pred[y] = mu[y], mu[x]; used |= {x, y}; planted += 1; break
                    if planted >= 4: break
                iu, ju = np.triu_indices(k, 1)
                oracle = {(int(p_), int(q_)) for p_, q_ in zip(iu, ju)
                          if abs(mu[p_] - mu[q_]) >= delta and np.sign(pred[p_] - pred[q_]) != np.sign(mu[p_] - mu[q_])}
                got = analysis.flip_summary(truth, pred, delta, Q)["flip_pairs"]
                tp += len(oracle & got); fn += len(oracle - got)
                for (p_, q_) in got - oracle:      # a flip the pipeline reports but we did not plant
                    gap = abs(mu[p_] - mu[q_])
                    if gap < delta / 2: fp += 1    # clearly not decidable: a real false alarm
                    else: near += 1                # near the margin: the noisy estimate crossed delta
        rows.append({"seeds": ns, "placebo_flips": pl_flips, "placebo_decidable": pl_dec,
                     "placebo_flip_rate": pl_flips / max(pl_dec, 1),
                     "placebo_top1_flips": f"{top1_flip}/{top1_testable}",
                     "planted_recall": tp / max(tp + fn, 1), "false_alarms": fp, "near_margin_extra": near, "planted_found": tp,
                     "planted_total": tp + fn})
    t = pd.DataFrame(rows); report["D2_D3"] = rows
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    at30 = t[t.seeds == 30].iloc[0]
    add("PASS" if at30.placebo_flip_rate <= 0.02 else "WARN",
        f"placebo: {100*at30.placebo_flip_rate:.2f}% of decidable pairs flip against an independent repeat at 30 seeds (noise floor)")
    add("PASS" if at30.planted_recall >= 0.90 else "WARN",
        f"planted flips found at 30 seeds: {100*at30.planted_recall:.0f}% (want >= 90%)")
    fa30 = at30.false_alarms / max(at30.planted_found + at30.false_alarms + at30.near_margin_extra, 1)
    add("PASS" if fa30 <= 0.01 else "WARN",
        f"false alarms at 30 seeds (reported flips on pairs whose true gap is under delta/2): {int(at30.false_alarms)} = {100*fa30:.2f}% of reported flips (want <= 1%)")
    add("NOTE", f"{int(t.near_margin_extra.sum())} extra flips sit on pairs whose true gap is between delta/2 and delta: "
                "the noisy estimate crossed the margin. They are not errors, but pairs near delta are inherently borderline")
    add("NOTE", "recall at 5 / 10 / 30 seeds = " + " / ".join(f"{100*r:.0f}%" for r in t.planted_recall)
        + " -- this is why the design asks for 30 seeds in test cells")

    # D4 -------------------------------------------------------------------
    print("\nD4. Conformal coverage")
    n_cal, n_test = 3000, 20000
    sd = np.mean([np.hypot(m["sd_seed"], m["sd_run"]) for m in models.values()])
    cal = np.abs(rng.normal(0, sd, n_cal))
    hw = analysis.conformal_halfwidth(cal, 0.10)
    cov_in = analysis.coverage(rng.normal(0, sd, n_test), np.zeros(n_test), hw)
    cov_shift = analysis.coverage(rng.normal(0, sd, n_test) + 2 * sd, np.zeros(n_test), hw)
    report["D4"] = {"halfwidth": hw, "coverage_in_distribution": cov_in, "coverage_shifted": cov_shift}
    print(f"  half-width {hw:.2f} days; coverage in-distribution {100*cov_in:.1f}%, with a bias of 2 SD {100*cov_shift:.1f}%")
    add("PASS" if abs(cov_in - 0.90) <= 0.02 else "FAIL", f"in-distribution coverage = {100*cov_in:.1f}% (target 90% +- 2)")
    add("PASS" if (0.90 - cov_shift) >= 0.10 else "FAIL",
        f"a surrogate with a known 2-SD bias is caught: coverage deficit {100*(0.90-cov_shift):.0f} points")

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    json.dump({"settings": {"reps": a.reps, "delta": delta, "q": Q}, "report": report, "checks": checks},
              open(paths.REPORT_DIR / "pipeline_dryruns.json", "w"), indent=2, default=float)
    print(f"\nPIPELINE DRY RUNS: {'FAILED' if n_fail else 'PASSED'} (WARN/NOTE lines are design flags, not errors).")
    print("Saved: results/reports/pipeline_dryruns.json")


if __name__ == "__main__":
    main()
