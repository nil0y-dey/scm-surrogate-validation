"""
paths.py -- the ONE place that decides where files live.

  data/        big, regenerable simulator output (NOT in git)
  results/     small, committed outputs: figures and PASS/FAIL reports
  third_party/ the patched ISOMORPH simulator (a git submodule)

On Colab, local files vanish when the session ends. To keep your
simulation runs between sessions, mount Google Drive and point the data
folder there BEFORE running any script:

    import os
    os.environ["SCMVAL_DATA_DIR"] = "/content/drive/MyDrive/scmval_data"
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("SCMVAL_DATA_DIR", REPO_ROOT / "data"))
RUNS_DIR = DATA_DIR / "runs"          # raw simulator runs, one folder per run
DERIVED_DIR = DATA_DIR / "derived"    # pickles / aggregates made from runs

RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_DIR = RESULTS_DIR / "reports"

ISOMORPH_DIR = Path(os.environ.get(
    "ISOMORPH_REPO_DIR", REPO_ROOT / "third_party" / "ISOMORPH_Hmod"))


def make_dirs() -> None:
    for d in (RUNS_DIR, DERIVED_DIR, FIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
