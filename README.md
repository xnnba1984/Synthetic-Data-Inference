# Code for Synthetic-Data Inference Paper

This repository contains the code used to generate the computational results
for a methodology paper on combining audited real data with synthetic data for
valid statistical inference.

The paper studies a setting with a real target distribution `P`, a synthetic
generator distribution `Q`, a small audited sample, and a larger synthetic
sample. The main question is when synthetic data can improve inference without
silently importing bias from the generator. The code reproduces the simulation
studies and the Hillstrom semi-real analysis reported in the paper.

## Repository Contents

Only code and this README are included.

```text
README.md
run_all.py
make_paper_outputs.py
simulation/
analysis/
```

The code creates the output folders when it runs:

```text
analysis/data/hillstrom/
analysis/results/
simulation/results/
figures/
tables/
```

No cached results, figures, tables, manuscript files, review files, or local
planning notes are included in this package.

## Dependencies

The code was run with Python 3.12 and the following main packages:

```text
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
matplotlib==3.9.2
scikit-learn==1.5.1
sdv==1.33.0
ctgan==0.12.0
torch==2.8.0
```

The core simulations use NumPy, pandas, SciPy, matplotlib, and scikit-learn.
The modern tabular-generator experiment and the generic-generator Hillstrom
sensitivity analysis require SDV, CTGAN, and Torch.

One possible setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 matplotlib==3.9.2 scikit-learn==1.5.1 sdv==1.33.0 ctgan==0.12.0 torch==2.8.0
```

If Torch or SDV wheels are difficult to install on the local machine, use a
conda or mamba environment with the same package versions.

## How to Reproduce the Results

From the repository root, run:

```bash
python run_all.py
```

This executes the scripts in the order needed to regenerate all computational
outputs. The full run can take a long time because several scripts fit
synthetic generators and run Monte Carlo experiments.

The Hillstrom scripts download the public Hillstrom MineThatData e-mail
analytics CSV on first use and save it under `analysis/data/hillstrom/`.

## Script Order

`run_all.py` runs:

```text
simulation/stage1_oracle_scalar.py
simulation/stage1_qc_robustness.py
simulation/stage2_estimating_equation.py
simulation/stage3_semiparametric_ate.py
simulation/stage4_learned_generator_diagnostics.py
simulation/stage4_qc_calibration.py
simulation/stage4_calibration_holdout.py
simulation/selection_repair_feasible.py
simulation/stage5_modern_tabular_generator.py
analysis/hillstrom_pilot.py
analysis/hillstrom_generator_sensitivity.py
analysis/hillstrom_task_aware_pilot.py
analysis/hillstrom_task_aware_qc.py
make_paper_outputs.py
```

The final script, `make_paper_outputs.py`, reads the generated CSV summaries and
creates the paper-facing figures and summary tables.

## Outputs

After `python run_all.py`, the main outputs are:

```text
figures/manuscript_simulation/
figures/manuscript_empirical/
tables/manuscript_empirical/
tables/simulation_story_map/
simulation/results/
analysis/results/
```

The intermediate folders under `figures/` and `tables/` contain diagnostic and
supporting outputs used to build the main figures and tables.

## Data Source

The empirical analysis uses the public Hillstrom MineThatData e-mail analytics
experiment:

```text
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
```

The code treats the full public data set as a semi-real target population for
repeated audited-sample experiments. The original randomized e-mail assignment
is used to define the treatment, and website visit is the primary outcome.
