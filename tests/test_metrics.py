import numpy as np
import pandas as pd

from scmval.metrics import (CAPACITY_RATIO, bh_reject, fit_scale, fluid_ttr_raw,
                            paired_test, settle_day, window_metrics)


def test_settle_day_finds_the_end_of_a_transient():
    v = np.concatenate([np.linspace(0.5, 1.0, 60), np.ones(240)])
    s = pd.Series(v, index=range(300))
    d = settle_day(s, steady=1.0, tol=0.01, window=20)
    # the ramp ends at day 59; a 20-day AVERAGE is within 1 pp a few days earlier
    assert 45 <= d <= 62


def test_settle_day_zero_when_always_settled():
    assert settle_day(pd.Series(np.ones(100)), steady=1.0) == 0


def _daily(fill, demand=100.0):
    n = len(fill)
    fill = np.asarray(fill, float)
    return pd.DataFrame({
        "demand": np.full(n, demand), "served": fill * demand,
        "new_backlog": (1 - fill) * demand,
        "backlog_end": np.cumsum((1 - fill) * demand),
        "fill_rate": fill}, index=range(n))


def test_window_metrics_on_a_known_dip():
    fill = np.ones(300)
    fill[100:120] = 0.5          # 20 days at 50% fill
    m = window_metrics(_daily(fill), t0=100, duration=20, length=300)
    assert abs(m["min_fill"] - 0.5) < 1e-12
    assert abs(m["triangle_days"] - 10.0) < 1e-9          # 20 days x 0.5
    assert abs(m["unmet_demand_days"] - 10.0) < 1e-9      # 20 days x 50 units unserved / 100 per day
    assert m["ttr"] == 0 and not m["censored"]            # fill is back at 1.0 at restoration


def test_window_metrics_censored_when_never_recovering():
    fill = np.ones(300)
    fill[100:] = 0.5
    m = window_metrics(_daily(fill), t0=100, duration=30, length=300)
    assert m["censored"] and m["ttr"] is None
    assert m["ttr_lower_bound"] == 300 - 130


def test_fluid_formula_basics():
    assert fluid_ttr_raw(0.9, 90) == 0.0                  # 1.29*0.9 > 1: no overload
    assert fluid_ttr_raw(0.5, 30) > fluid_ttr_raw(0.5, 14) > 0
    assert abs(fluid_ttr_raw(0.5, 14) - 14 * (1 - CAPACITY_RATIO * 0.5) / (CAPACITY_RATIO - 1)) < 1e-12


def test_fit_scale_recovers_known_factor():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    k, r2 = fit_scale(pred, 1.6 * pred)
    assert abs(k - 1.6) < 1e-12 and abs(r2 - 1.0) < 1e-12


def test_settle_day_is_noise_aware():
    rng = np.random.default_rng(0)
    s = pd.Series(0.99 + rng.normal(0, 0.03, 400), index=range(400))   # always noisy, never a transient
    steady = float(s.loc[200:].mean())
    sd = float(s.loc[200:].rolling(20).mean().dropna().std())
    flat = settle_day(s, steady, tol=0.005)
    aware = settle_day(s, steady, tol=0.005, noise_sd=sd, noise_k=3.0)
    assert aware <= flat
    assert aware < 50


def test_bh_rejects_all_when_all_small():
    assert bh_reject([0.01, 0.04, 0.03, 0.005], q=0.05).all()


def test_bh_rejects_only_the_clear_one():
    assert list(bh_reject([0.005, 0.2, 0.3], q=0.05)) == [True, False, False]


def test_paired_test_edge_cases():
    a = np.array([1.0, 2.0, 3.0])
    assert paired_test(a, a) == (0.0, 0.0, 1.0)
    m, sd, p = paired_test(a + 5, a)
    assert m == 5.0 and sd == 0.0 and p == 0.0


def test_paired_test_detects_a_real_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(10, 1, 30)
    m, sd, p = paired_test(base + 1.0 + rng.normal(0, 0.2, 30), base)
    assert m > 0.8 and p < 1e-6
