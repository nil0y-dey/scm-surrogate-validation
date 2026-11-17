from scmval import paths


def test_repo_root_is_the_repo():
    assert (paths.REPO_ROOT / "pyproject.toml").exists()


def test_big_data_is_separate_from_committed_results():
    assert paths.RUNS_DIR.parent == paths.DATA_DIR
    assert paths.FIG_DIR.parent == paths.RESULTS_DIR
