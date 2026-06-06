#!/usr/bin/env python3
"""Run all computations needed to regenerate the paper's computational outputs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "simulation/stage1_oracle_scalar.py",
    "simulation/stage1_qc_robustness.py",
    "simulation/stage2_estimating_equation.py",
    "simulation/stage3_semiparametric_ate.py",
    "simulation/stage4_learned_generator_diagnostics.py",
    "simulation/stage4_qc_calibration.py",
    "simulation/stage4_calibration_holdout.py",
    "simulation/selection_repair_feasible.py",
    "simulation/stage5_modern_tabular_generator.py",
    "analysis/hillstrom_pilot.py",
    "analysis/hillstrom_generator_sensitivity.py",
    "analysis/hillstrom_task_aware_pilot.py",
    "analysis/hillstrom_task_aware_qc.py",
    "make_paper_outputs.py",
]


def main() -> None:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)

    print("\nAll computational outputs have been regenerated.", flush=True)


if __name__ == "__main__":
    main()
