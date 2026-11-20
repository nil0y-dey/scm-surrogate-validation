import subprocess, sys, pathlib
import numpy as np, pandas as pd

def test_learnability_runs_on_synthetic_table(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for nu in (0.05, 0.10, 0.15, 0.45):
        for D in (14, 30, 90):
            for ss in (0.5, 1.0):
                for pm in (5.0, 7.0):
                    for seed in range(2025, 2030):
                        rows.append(dict(nu=nu, D=D, ss=ss, pm=pm, seed=seed,
                                         unmet=max(0, nu * D * (2 - ss) + pm * 0.1 + rng.normal(0, .3))))
    csv = tmp_path / "t.csv"; pd.DataFrame(rows).to_csv(csv, index=False)
    root = pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "08_learnability.py"),
                        "--csv", str(csv), "--tag", "test", "--outdir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "LEARNABILITY" in r.stdout
