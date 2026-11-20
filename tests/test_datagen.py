import numpy as np
from scmval import datagen, design


def test_counts_match_design_plan():
    p = design.run_plan()
    assert len(datagen.plan("train")) == p["tier1_train"]["runs"]
    assert len(datagen.plan("calibration")) == p["tier1_calibration"]["runs"]
    assert len(datagen.plan("test")) == p["tier1_test_first_pass"]["runs"]
    assert len(datagen.plan("calm")) == p["tier2_calm_ablation"]["runs"]
    assert len(datagen.plan("wide")) == p["tier2_wide_band"]["runs"]
    assert len(datagen.plan("placebo")) == p["tier2_placebo"]["runs"]
    assert len(datagen.plan("test", 15, 30)) == p["tier3_deepening_max"]["runs"]


def test_seeds_stay_in_their_ranges_and_never_overlap():
    seen = {}
    for b in datagen.BLOCKS:
        seeds = [m["seed"] for m, _ in datagen.table_plan(b)]
        lo, hi = design.SEED_RANGES["training" if b in ("train", "calm", "wide") else b]
        assert min(seeds) >= lo and max(seeds) <= hi, b
        if b in ("train", "calm", "wide", "calibration"):
            assert len(set(seeds)) == len(seeds), b           # a fresh seed per run
        seen[b] = set(seeds)
    assert not (seen["train"] & seen["calm"]) and not (seen["train"] & seen["wide"]) and not (seen["calm"] & seen["wide"])
    assert not (seen["test"] & seen["placebo"]) and not (seen["train"] & seen["test"])


def test_training_design_is_balanced_and_in_band():
    tr = datagen.plan("train")
    nus = np.array([m["nu"] for m, _ in tr])
    assert nus.min() > 0 and nus.max() <= 0.15
    assert np.histogram(nus, bins=10, range=(0, 0.15))[0].min() > 200      # latin hypercube: evenly spread
    assert {m["D"] for m, _ in tr} == set(design.TRAIN_DURATIONS)
    counts = np.bincount([datagen.POLICIES.index((m["ss"], m["pm"])) for m, _ in tr])
    assert counts.min() >= 120 and counts.max() <= 130


def test_calm_runs_have_no_disruption_and_configs_are_consistent():
    for m, c in datagen.plan("calm")[:20]:
        assert c.disrupt_duration == 0 and c.disrupt_capacity_scale == 1.0
    m, c = datagen.plan("test")[0]
    assert c.days == design.RUN_LENGTH and c.disrupt_start_day == design.T0 and c.slim
    assert abs((1 - 1.29 * c.disrupt_capacity_scale) - m["nu"]) < 0.01
