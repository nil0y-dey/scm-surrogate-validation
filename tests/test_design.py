from scmval import design
from scmval.policies import grid, viable_grid


def test_nu_and_capacity_are_inverses():
    for nu in (0.0, 0.1, 0.35, 0.6):
        assert abs(design.nu_from_capacity(design.capacity_from_nu(nu)) - nu) < 1e-3


def test_no_overload_when_capacity_absorbs_the_cut():
    assert design.nu_from_capacity(0.8) == 0.0
    assert design.nu_from_capacity(0.775) < 1e-3


def test_delta_in_unmet_days():
    assert abs(design.delta_unmet_days(400, 100) - 1.5) < 1e-12


def test_seed_ranges_do_not_overlap():
    r = sorted(design.SEED_RANGES.values())
    for (a0, a1), (b0, b1) in zip(r, r[1:]):
        assert a1 < b0


def test_viable_grid_drops_exactly_the_gate3_failures():
    assert len(grid()) == 25 and len(viable_grid()) == 20
