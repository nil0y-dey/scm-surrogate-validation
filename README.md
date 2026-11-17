# scm-surrogate-validation

Can we trust a fast machine-learning copy of a supply-chain simulator when
a disruption hits? This repo holds the code for a decision-regret
validation of ML surrogates (see `docs/proposal/proposal.tex`).

## Folder map

```
scm-surrogate-validation/
├── README.md                  <- you are here
├── pyproject.toml             <- makes the shared code installable (pip install -e .)
├── requirements.txt
├── src/scmval/                <- shared code (imported by every script)
│   ├── sim_runner.py          <- how to run the simulator and read its output
│   ├── recovery.py            <- the rule that decides "recovered"
│   └── paths.py               <- where files live (change data location here)
├── scripts/                   <- one runnable script per study, numbered in order
│   ├── 00_patch_verification.py
│   ├── 01_gate1_integrity.py
│   ├── 02_warmup_calibration.py
│   ├── 03_stability_boundary.py
│   ├── 04_severity_calibration.py
│   └── run_all.py             <- runs them in order
├── tests/                     <- fast self-checks (python -m pytest)
├── configs/                   <- frozen study settings (design lock goes here)
├── docs/                      <- proposal.tex, RUNBOOK.md
├── results/                   <- small outputs that ARE committed (figures, PASS/FAIL reports)
├── data/                      <- big simulator output (NOT committed; created on demand)
└── third_party/
    └── ISOMORPH_Hmod/         <- the patched simulator, as a git submodule
```

Simulator A (our own SimPy model) will go in `src/scmval/sim_a/` when built.

## First-time setup (once, on your own computer)

```
cd scm-surrogate-validation
git init -b main
git add .
git commit -m "Initial structure: validation suite, tests, docs"
git submodule add https://github.com/nil0y-dey/ISOMORPH_Hmod.git third_party/ISOMORPH_Hmod
git commit -m "Pin patched ISOMORPH simulator as submodule"
```
Then create an EMPTY repository on GitHub (no README) and:
```
git remote add origin https://github.com/.git
git push -u origin main
```

## Using it on Colab

```
!git clone --recurse-submodules https://github.com/nil0y-dey/scm-surrogate-validation.git
%cd scm-surrogate-validation
!pip install -q -e .
```
Colab erases its files when a session ends. To keep simulation runs, mount
Google Drive and set the data folder BEFORE running anything:
```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["SCMVAL_DATA_DIR"] = "/content/drive/MyDrive/scmval_data"
```
Then follow `docs/RUNBOOK.md`.

## The simulator submodule

`third_party/ISOMORPH_Hmod` is its own repository (ISOMORPH plus three
disclosed patches). This repo records the EXACT commit used, so any result
can be traced to a precise simulator version. If you ever change the
simulator (e.g. a future patch), commit inside the submodule first, then
commit the updated pointer here:
```
cd third_party/ISOMORPH_Hmod && git add -A && git commit -m "..." && git push
cd ../.. && git add third_party/ISOMORPH_Hmod && git commit -m "Bump simulator"
```

## What is committed and what is not

Committed: code, tests, configs, docs, and small results (figures, reports).
Not committed: `data/` (hundreds of MB to GB of regenerable runs). At
publication time, archive the data on Zenodo and cite the DOI.

## Conventions

- Every simulator change is single-purpose, disclosed, and a verified
  no-op at released defaults (`scripts/00_patch_verification.py`).
- Seeds: pilots 2025-2029 (already used). Training, calibration, test and
  placebo runs will each get their own non-overlapping seed range, recorded
  in `configs/`.
- Before ANY test-grid run, freeze the design (thresholds, run length,
  mapping, seeds) in `configs/` and tag the commit: `git tag design-lock`.
