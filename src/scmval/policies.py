"""
policies.py -- the policy grid compared in the study (Simulator B).

PROPOSED values: they are finalised only after Gate 3 (calm-weather
viability and warm-up) and Gate 6 (can the policies be told apart under
disruption?). Edit here; every script reads the grid from this one file.

Two knobs, both exposed by the patched simulator:
  ss_scale       multiplies every node's reorder threshold s, order-up-to
                 level S and starting stock (small = lean, large = buffered)
  pipeline_mult  how many days of expected demand the network tries to keep
                 in transit to the destination (small = reactive, large =
                 proactive); 7 is ISOMORPH's released baseline.

Probe in calm weather (1 seed, 400 days): pipeline_mult = 2 already loses
~25% (ss 0.25) or ~2.5% (ss 1.5) of demand with NO disruption, so it is
not a viable policy; pipeline_mult >= 4 looked fine.
"""
from __future__ import annotations

SS_SCALES = [0.25, 0.5, 0.75, 1.0, 1.5]
PIPELINE_MULTS = [3.0, 5.0, 7.0, 10.0, 14.0]
BASELINE_POLICY = (1.0, 7.0)          # (ss_scale, pipeline_mult) = released baseline


def grid(ss_scales=None, pipeline_mults=None):
    """All (ss_scale, pipeline_mult) pairs, row-major in ss_scale."""
    ss = SS_SCALES if ss_scales is None else ss_scales
    pm = PIPELINE_MULTS if pipeline_mults is None else pipeline_mults
    return [(float(s), float(m)) for s in ss for m in pm]


def label(ss_scale: float, pipeline_mult: float) -> str:
    return f"ss{ss_scale:g}_pm{pipeline_mult:g}"


# Gate 3 (calm, 3 seeds, 500 days): these lose >1% of demand with NO disruption.
NOT_VIABLE = {(0.25, 3.0), (0.25, 5.0), (0.25, 7.0), (0.25, 10.0), (0.5, 3.0)}


def viable_grid():
    """The grid without the Gate-3 non-viable policies (20 of 25)."""
    return [p for p in grid() if p not in NOT_VIABLE]
