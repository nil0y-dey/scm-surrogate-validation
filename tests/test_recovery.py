import numpy as np
import pandas as pd

from scmval.recovery import first_sustained_recovery


def series(values):
    return pd.Series(values, index=range(len(values)))


def test_one_lucky_day_is_not_recovery():
    v = np.full(100, 0.5); v[50] = 1.0; v[70:] = 1.0
    assert first_sustained_recovery(series(v), 1.0, 40, 99) == 70


def test_never_recovers_is_censored():
    assert first_sustained_recovery(series(np.full(100, 0.5)), 1.0, 40, 99) is None


def test_recovery_too_late_to_hold_is_censored():
    v = np.full(100, 0.5); v[97:] = 1.0
    assert first_sustained_recovery(series(v), 1.0, 40, 99) is None


def test_recovery_exactly_in_time():
    v = np.full(100, 0.5); v[93:] = 1.0
    assert first_sustained_recovery(series(v), 1.0, 40, 99) == 93
