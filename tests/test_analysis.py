import numpy as np
from scmval import analysis
from scmval.metrics import paired_test


def test_pair_stats_matches_paired_test():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 4)) + np.arange(4)
    i, j, m, sd, p = analysis.pair_stats(X)
    for a, b, mm, ss, pp in zip(i, j, m, sd, p):
        ref = paired_test(X[:, a], X[:, b])
        assert np.allclose([mm, ss, pp], ref)


def test_decidable_needs_margin_and_significance():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.05, size=(30, 1))
    X = np.hstack([base, base + 0.2, base + 5.0]) + rng.normal(0, 0.01, (30, 3))
    i, j, m, dec = analysis.decidable_pairs(X, delta=1.0, q=0.10)
    got = {(a, b) for a, b, d in zip(i, j, dec) if d}
    assert got == {(0, 2), (1, 2)}          # the 0.2 gap is significant but below the margin


def test_flip_summary_counts_a_swap():
    rng = np.random.default_rng(3)
    X = np.array([0.0, 3.0, 6.0]) + rng.normal(0, 0.05, (30, 3))
    s = analysis.flip_summary(X, np.array([6.0, 3.0, 0.0]), delta=1.0)
    assert s["n_decidable"] == 3 and s["n_flips"] == 3 and s["top1_flip"]
    ok = analysis.flip_summary(X, np.array([0.1, 3.0, 6.2]), delta=1.0)
    assert ok["n_flips"] == 0 and ok["kendall_tau"] == 1.0


def test_regret_and_conformal():
    assert analysis.regret([1.0, 2.0, 5.0], [3.0, 1.0, 2.0]) == 1.0
    r = np.arange(1, 101) / 100
    hw = analysis.conformal_halfwidth(r, 0.10)
    assert 0.9 <= hw <= 0.92
    assert analysis.coverage([0.0, 1.0], [0.0, 0.0], 0.5) == 0.5
