"""
08_learnability.py
-------------------
SURROGATE LEARNABILITY (runs in seconds; NO new simulations).

Question: before we spend the big data budget, can a cheap ML copy
("surrogate") of the simulator learn anything useful from Mild-band
training data -- and is there room for it to fail under harder
disruptions? Both answers matter for the paper:
  * if surrogates cannot even learn the training band, the study measures
    nothing;
  * if surrogates are perfect everywhere (also far outside the training
    band), there is no failure to audit.

Data: one table, one row per simulation run, with columns
    nu, D, <policy columns>, seed, unmet
(for simulator B this is results/reports/gate56_per_run.csv written by
script 07; simulator A must write the same shape, see docs). "unmet" =
days of demand lost over the evaluation window.

Surrogates (4 families, scikit-learn): ridge on quadratic features,
random forest, gradient boosting, small neural net. Target = log(1+unmet).
Train band = design.TRAIN_BAND cells (nu <= 0.15, D in 14/30).

Tests:
  T1 in-band, held-out seeds : accuracy (R^2, RMSE in days) and how well the
                               surrogate ranks policies and picks the best one
  T2 shifted cells           : train on the band, test on cells outside it,
                               by severity band and duration
  Regret = true mean unmet of the policy the surrogate picks minus that of
  the truly best policy (true = mean over the available seeds). Compared with
  delta (practical margin), with "random advice", and with a "constant
  policy" (the policy that was best on average in the training data).

Usage:
    python scripts/08_learnability.py
    python scripts/08_learnability.py --csv path/to/table.csv --policy-cols ss,pm --tag B
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from scmval import design, paths

warnings.filterwarnings("ignore")
MIN_FRAC_OF_CEILING = 0.90


def band_of(nu: float) -> str:
    for name, (lo, hi) in design.NU_BANDS.items():
        if lo < nu <= hi + 1e-9:
            return name
    return "none"


def features(df: pd.DataFrame, pcols: list[str]) -> np.ndarray:
    cols = [df["nu"].to_numpy(), np.log(df["D"].to_numpy())]
    for c in pcols:
        cols.append(np.log(df[c].to_numpy()))
    return np.column_stack(cols)


def make_models(seed: int = 0) -> dict:
    return {
        "ridge-quad": make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-2)),
        "random-forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "grad-boost": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                random_state=seed),
        "neural-net": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                                                                   random_state=seed)),
    }


def cell_regret(test: pd.DataFrame, pred: np.ndarray, pcols: list[str], const_policy) -> list[dict]:
    """Per (nu, D) cell: regret of the surrogate's pick, of a constant policy, of random advice."""
    t = test.assign(pred=pred)
    out = []
    for (nu, D), g in t.groupby(["nu", "D"]):
        true = g.groupby(pcols)["unmet"].mean()
        sp = g.groupby(pcols)["pred"].mean()
        pick = sp.idxmin()
        best = true.min()
        rho = spearmanr(true.loc[sp.index], sp).correlation if len(true) > 2 else np.nan
        out.append({"nu": nu, "D": int(D), "band": band_of(nu),
                    "regret": float(true.loc[pick] - best),
                    "const_regret": float(true.loc[const_policy] - best) if const_policy in true.index else np.nan,
                    "random_regret": float((true - best).mean()),
                    "spearman": float(rho), "mean_unmet": float(true.mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(paths.REPORT_DIR / "gate56_per_run.csv"))
    ap.add_argument("--policy-cols", default="ss,pm")
    ap.add_argument("--tag", default="B")
    ap.add_argument("--outdir", default=str(paths.RESULTS_DIR), help="where reports/ and figures/ are written (tests use a temp folder)")
    a = ap.parse_args()
    pcols = a.policy_cols.split(",")
    paths.make_dirs()
    out = Path(a.outdir); (out / "reports").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    delta = design.delta_unmet_days()
    lo, hi = design.NU_BANDS[design.TRAIN_BAND]
    in_band = (df.nu > lo) & (df.nu <= hi + 1e-9) & df.D.isin(design.TRAIN_DURATIONS)
    train_all, outside = df[in_band], df[~in_band]
    seeds = sorted(train_all.seed.unique())
    hold = [int(s) for s in seeds[-2:]]
    print(f"Table: {len(df)} runs; training band: {in_band.sum()} runs in "
          f"{train_all.groupby(['nu', 'D']).ngroups} cells; outside the band: {len(outside)} runs.")
    print(f"delta = {delta:.2f} days of unmet demand. Held-out seeds for T1: {hold}.\n")

    const = train_all.groupby(pcols)["unmet"].mean().idxmin()
    results, rows_t1, rows_t2 = {}, [], []
    for name in make_models():
        # T1: in-band, held-out seeds
        tr, te = train_all[~train_all.seed.isin(hold)], train_all[train_all.seed.isin(hold)]
        m = make_models()[name].fit(features(tr, pcols), np.log1p(tr.unmet))
        p = np.expm1(m.predict(features(te, pcols))).clip(min=0)
        y = te.unmet.to_numpy()
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rmse = float(np.sqrt(((y - p) ** 2).mean()))
        # noise ceiling: an oracle that knows each (cell, policy) mean from the training seeds
        orc = te[["nu", "D"] + pcols].merge(tr.groupby(["nu", "D"] + pcols).unmet.mean().rename("o").reset_index(),
                                           on=["nu", "D"] + pcols, how="left").o.to_numpy()
        ceil = 1 - ((y - orc) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        c1 = cell_regret(te, p, pcols, const)
        # T2: fit on all in-band seeds, predict outside
        m2 = make_models()[name].fit(features(train_all, pcols), np.log1p(train_all.unmet))
        p2 = np.expm1(m2.predict(features(outside, pcols))).clip(min=0)
        c2 = cell_regret(outside, p2, pcols, const)
        for r in c1: r.update(model=name); rows_t1.append(r)
        for r in c2: r.update(model=name); rows_t2.append(r)
        e2 = outside.unmet.to_numpy()
        results[name] = {"inband_R2": float(r2), "inband_R2_ceiling": float(ceil), "inband_RMSE_days": rmse,
                         "inband_mean_regret": float(np.mean([r["regret"] for r in c1])),
                         "inband_mean_spearman": float(np.nanmean([r["spearman"] for r in c1])),
                         "shift_RMSE_days": float(np.sqrt(((e2 - p2) ** 2).mean())),
                         "shift_mean_regret": float(np.mean([r["regret"] for r in c2]))}

    t1, t2 = pd.DataFrame(rows_t1), pd.DataFrame(rows_t2)
    pd.set_option("display.width", 200)
    print("=== T1: in the training band, on held-out seeds ===")
    print(pd.DataFrame(results).T[["inband_R2", "inband_R2_ceiling", "inband_RMSE_days", "inband_mean_regret",
                                   "inband_mean_spearman"]].round(3).to_string())
    base = t1.groupby("model")[["const_regret", "random_regret"]].mean().iloc[0]
    print(f"  constant-policy regret = {base.const_regret:.2f} days, random-advice regret = "
          f"{base.random_regret:.2f} days (constant policy: safety-stock scale {const[0]:g}, pipeline multiplier {const[1]:g})\n")

    print("=== T2: outside the training band (the shift) -- mean regret in days, by severity band ===")
    piv = t2.pivot_table(index="model", columns="band", values="regret", aggfunc="mean")
    piv = piv[[b for b in design.NU_BANDS if b in piv.columns]]
    print(piv.round(2).to_string())
    print("  same, by duration D:")
    print(t2.pivot_table(index="model", columns="D", values="regret", aggfunc="mean").round(2).to_string())
    print("  reference (mean over the same cells):  constant policy "
          f"{t2.groupby('model').const_regret.mean().iloc[0]:.2f},  random advice "
          f"{t2.groupby('model').random_regret.mean().iloc[0]:.2f} days")
    print("  in-band cells with 14-day D outside the training durations are counted as shift too.\n")

    best = min(results, key=lambda k: results[k]["inband_mean_regret"])
    b = results[best]
    shift_best = t2[t2.model == best]
    checks = []

    def add(s, msg):
        checks.append({"status": s, "message": msg}); print(f"  [{s}] {msg}")
    print("=== LEARNABILITY CHECKS (best in-band model: %s) ===" % best)
    best_r2 = max(v["inband_R2"] for v in results.values())
    ceil_r2 = results[best]["inband_R2_ceiling"]
    add("PASS" if best_r2 >= MIN_FRAC_OF_CEILING * ceil_r2 else "FAIL",
        f"in the training band a surrogate predicts unmet demand on unseen seeds: best R^2 = {best_r2:.2f}; "
        f"the ceiling set by seed-to-seed noise is {ceil_r2:.2f} (want >= {MIN_FRAC_OF_CEILING:.0%} of it)")
    add("PASS" if b["inband_mean_regret"] < delta else "WARN",
        f"in-band, the surrogate's chosen policy costs {b['inband_mean_regret']:.2f} days more than the best (delta = {delta:.2f})")
    add("NOTE", f"in-band the best policy is almost the same in every cell (constant-policy regret {base.const_regret:.2f} days), "
                f"so the in-band task cannot separate surrogates from a fixed rule; the shift test below can")
    gap = b["shift_mean_regret"] - b["inband_mean_regret"]
    add("NOTE", f"headroom for failure: mean regret rises from {b['inband_mean_regret']:.2f} (in band) to "
                f"{b['shift_mean_regret']:.2f} days (outside the band); "
                + ("there is a shift effect worth auditing" if gap > delta
                   else "surrogates do NOT clearly degrade under shift in this data: the paper's story becomes 'surrogates held up' -- still a valid result, but plan the contribution around it"))
    n_bad = int((shift_best.regret > delta).sum())
    add("NOTE", f"outside the band, {n_bad} of {len(shift_best)} cells have surrogate regret above delta")
    status = "FAILED" if any(c["status"] == "FAIL" for c in checks) else "PASSED"
    print(f"\nLEARNABILITY: {status} (WARN/NOTE lines are design flags, not errors).")

    t1.assign(test="in_band").pipe(lambda d: pd.concat([d, t2.assign(test="shift")])).to_csv(
        out / "reports" / f"learnability_{a.tag}_cells.csv", index=False)
    json.dump({"tag": a.tag, "delta": delta, "constant_policy": [float(x) for x in const],
               "models": results, "checks": checks, "best_model": best},
              open(out / "reports" / f"learnability_{a.tag}.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, g in t2.groupby("model"):
        s = g.groupby("nu").regret.mean()
        ax[0].plot(s.index, s.values, "o-", label=name)
    s = t2.groupby("nu").const_regret.mean(); ax[0].plot(s.index, s.values, "k--", label="constant policy")
    ax[0].axhline(delta, color="gray", lw=0.8); ax[0].set_xlabel("overload nu (outside training band)")
    ax[0].set_ylabel("regret (days of demand)"); ax[0].set_yscale("symlog", linthresh=0.5); ax[0].legend(fontsize=7)
    for name, g in t2.groupby("model"):
        s = g.groupby("D").regret.mean()
        ax[1].plot(s.index, s.values, "o-", label=name)
    ax[1].set_xlabel("duration D (days)"); ax[1].set_ylabel("regret (days of demand)")
    for x in ax: x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "figures" / f"learnability_{a.tag}.png", dpi=200)
    print(f"Saved in {out}: reports/learnability_{a.tag}.json, reports/learnability_{a.tag}_cells.csv, "
          f"figures/learnability_{a.tag}.png")


if __name__ == "__main__":
    main()
