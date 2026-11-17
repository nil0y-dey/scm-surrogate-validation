from scmval.sim_runner import RunConfig


def cfg(**kw):
    return RunConfig(seed=1, days=10, out_dir="x", scenario_name="y", **kw)


def test_defaults_are_noops_for_all_three_patches():
    c = cfg()
    assert c.containers_scale == 1.0          # patch 2
    assert c.disrupt_capacity_scale == 1.0    # patch 3
    assert c.disrupt_duration == 0            # patch 3
    assert c.pipeline_mult == 7.0             # released-baseline value
    assert c.slim is False


def test_command_line_contains_every_knob():
    cmd = cfg(ss_scale=0.5).to_cmd("sim.py")
    for flag in ("--days", "--seed", "--pipeline_mult", "--containers_scale",
                 "--disrupt_capacity_scale", "--disrupt_start_day",
                 "--disrupt_duration", "--ss_scale", "--out_dir"):
        assert flag in cmd
    assert cmd[cmd.index("--ss_scale") + 1] == "0.5"
