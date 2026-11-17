# RUNBOOK -- what to run, in order (plain version)

Everything below is for Google Colab. If Colab disconnects, run the same
command again: finished simulations are skipped, so nothing is lost
(as long as the data folder is on Google Drive -- see Step 0).

## Step 0 -- Setup (once per Colab session)
```
!git clone --recurse-submodules https://github.com/<your-username>/scm-surrogate-validation.git
%cd scm-surrogate-validation
!pip install -q -e .
```
Keep your runs between sessions (recommended):
```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["SCMVAL_DATA_DIR"] = "/content/drive/MyDrive/scmval_data"
```

## Step 1 -- Self-check of the code (2 seconds)
`!python -m pytest -q`
What it does: tests the small pieces (the "recovered" rule, default settings).
Good result: "8 passed".

## Step 2 -- Rehearsal (about 5 min)
`!python scripts/run_all.py --quick --skip-patch-verification`
What it does: runs a tiny version of every check so you know nothing is
broken before the long runs. Good result: "All requested steps completed."
and Gate 1 says "18 passed, 0 failed".

## Step 3 -- Prove the three patches change nothing (about 7 min)
`!python scripts/00_patch_verification.py`
What it does: re-runs ISOMORPH's own published baseline and compares six
numbers. Good result: "PASS". You already passed this once; only repeat it
if the simulator changed.

## Step 4 -- Gate 1: can we trust the simulator? (1-2 min)
`!python scripts/01_gate1_integrity.py`
What it does: checks (a) same settings + same seed = identical results,
(b) a disruption cannot change anything before it starts, (c) customer
demand is identical across different policies, so policy comparisons are
fair, (d) the capacity cut switches on and off on exactly the right days.
Good result: "GATE 1 PASSED". If FAILED, stop and send me the output.

## Step 5 -- Warm-up (about 15 min)
`!python scripts/02_warmup_calibration.py`
What it does: shows how many days the simulator needs before its numbers
settle. Saves results/figures/warmup_figure.png. Expect about day 100.

## Step 6 -- Stability boundary (about 45-60 min)
`!python scripts/03_stability_boundary.py`
What it does: finds the capacity level below which the system can never
catch up (about 0.775). Saves results/figures/stability_boundary_figure.png.

## Step 7 -- Severity calibration (about 10 min)
`!python scripts/04_severity_calibration.py`
What it does: tries three capacity cuts (0.85, 0.80, 0.78) for a 90-day
disruption and measures how long recovery takes. Look at the "per-seed
time-to-recovery" line: "None" means that run never recovered within the
run length. Pick the harshest cut that does not leave seeds censored.

## Shortcut
`!python scripts/run_all.py` runs the patch check, Gate 1, and Steps 5-7 in
order (about 1.5-2 hours).

## Saving your work to GitHub (after a good run)
Small results (figures, reports) are in `results/`. Commit them:
```
!git add results && git commit -m "Add calibration results" && git push
```
(Colab will ask for your GitHub login/token the first time.)

## What to send back to me
Paste the printed output of each step (not the images). I will read it and
tell you the next step.
