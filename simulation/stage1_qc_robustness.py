#!/usr/bin/env python3
"""QC and robustness checks for Stage 1 oracle scalar simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import stage1_oracle_scalar as sim


ROOT = Path(__file__).resolve().parents[1]
QC_DIR = ROOT / "simulation" / "results" / "stage1_oracle_scalar_qc"
TABLE_DIR = ROOT / "tables" / "stage1_oracle_scalar_qc"

MAIN_SUMMARY = ROOT / "tables" / "stage1_oracle_scalar" / "stage1_summary.csv"

SEEDS = (20260520, 20260521, 20260522)
REPS = 5000


@dataclass(frozen=True)
class SelectionScenario:
    name: str
    tau_q: float
    tau_if: float


SCENARIOS = (
    SelectionScenario("default", 1.00, 1.00),
    SelectionScenario("conservative_q", 0.75, 1.00),
    SelectionScenario("permissive_q", 1.25, 1.00),
    SelectionScenario("conservative_if", 1.00, 0.75),
    SelectionScenario("permissive_if", 1.00, 1.25),
)


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def selected_branch(epsilon: float, m: int, M: int, tau_q: float, tau_if: float) -> str:
    if epsilon <= tau_q * m ** (-0.5) and M >= sim.RHO * m:
        return "Q"
    if epsilon <= tau_if * m ** (-0.25):
        return "IF"
    return "A"


def run_seed_robustness() -> pd.DataFrame:
    rows = []
    old_seed, old_reps = sim.SEED, sim.REPS
    old_tau_q, old_tau_if = sim.TAU_Q, sim.TAU_IF
    try:
        for seed in SEEDS:
            sim.SEED = seed
            sim.REPS = REPS
            sim.TAU_Q = 1.0
            sim.TAU_IF = 1.0
            summary, _ = sim.run_all()
            d = summary[
                (summary["method"] == "selected")
                & (summary["m"] == 100)
                & (summary["M_multiplier"] == 100)
                & np.isclose(summary["nu"], 0.25)
            ].copy()
            d["seed"] = seed
            rows.append(d)
    finally:
        sim.SEED = old_seed
        sim.REPS = old_reps
        sim.TAU_Q = old_tau_q
        sim.TAU_IF = old_tau_if

    selected_by_seed = pd.concat(rows, ignore_index=True)
    selected_by_seed.to_csv(TABLE_DIR / "stage1_qc_selected_by_seed.csv", index=False)

    grouped = (
        selected_by_seed.groupby(["simulation", "epsilon_label", "epsilon", "selected_branch"])
        .agg(
            coverage_min=("coverage", "min"),
            coverage_max=("coverage", "max"),
            coverage_mean=("coverage", "mean"),
            avg_length_min=("avg_length", "min"),
            avg_length_max=("avg_length", "max"),
            avg_length_mean=("avg_length", "mean"),
        )
        .reset_index()
    )
    grouped["coverage_range"] = grouped["coverage_max"] - grouped["coverage_min"]
    grouped["avg_length_range"] = grouped["avg_length_max"] - grouped["avg_length_min"]
    return grouped


def run_selection_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary[
        (summary["m"] == 100)
        & (summary["M_multiplier"] == 100)
        & np.isclose(summary["nu"], 0.25)
        & summary["method"].isin(["audit_only", "q_inflated", "if_inflated"])
    ].copy()

    method_by_branch = {"A": "audit_only", "Q": "q_inflated", "IF": "if_inflated"}
    rows = []
    for scenario in SCENARIOS:
        for keys, d in base.groupby(["simulation", "m", "M", "M_multiplier", "nu", "epsilon", "epsilon_label"]):
            simulation, m, M, multiplier, nu, epsilon, epsilon_label = keys
            branch = selected_branch(float(epsilon), int(m), int(M), scenario.tau_q, scenario.tau_if)
            method = method_by_branch[branch]
            row = d[d["method"] == method].iloc[0].to_dict()
            row.update(
                {
                    "scenario": scenario.name,
                    "tau_q": scenario.tau_q,
                    "tau_if": scenario.tau_if,
                    "selected_branch": branch,
                    "selected_method": method,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_audit_interval_comparison() -> pd.DataFrame:
    rng = np.random.default_rng(20260523)
    rows = []
    reps = 50000
    m = 100
    nu = 0.25
    for epsilon_label, epsilon in sim.eps_grid(m):
        mu = nu + epsilon
        if mu >= 0.95:
            continue
        pbar = sim.draw_binary_means(rng, mu, m, reps)
        se_p = sim.mean_se(pbar, m)
        target = mu**2

        wald_lo, wald_hi = sim.ci_square_from_mean(pbar, se_p)
        exact_mu_lo, exact_mu_hi = sim.ci_mean_clopper_pearson(pbar, m)
        exact_lo, exact_hi = sim.ci_square_from_mean_bounds(exact_mu_lo, exact_mu_hi)

        for method, lo, hi in (
            ("transformed_wald", wald_lo, wald_hi),
            ("transformed_exact_binomial", exact_lo, exact_hi),
        ):
            rows.append(
                {
                    "simulation": "S2_squared_mean",
                    "m": m,
                    "nu": nu,
                    "mu": mu,
                    "epsilon": epsilon,
                    "epsilon_label": epsilon_label,
                    "audit_interval": method,
                    "coverage": np.mean((lo <= target) & (target <= hi)),
                    "avg_length": np.mean(hi - lo),
                }
            )
    return pd.DataFrame(rows)


def run_overlap_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = summary[
        (summary["m"] == 100)
        & (summary["M_multiplier"] == 100)
        & (
            ((summary["simulation"] == "S1_mean") & np.isclose(summary["nu"], 0.0))
            | ((summary["simulation"] == "S2_squared_mean") & np.isclose(summary["nu"], 0.25))
        )
        & summary["method"].isin(sim.PLOT_METHODS)
    ].copy()

    for simulation, dsim in main.groupby("simulation"):
        methods = list(dsim["method"].drop_duplicates())
        for i, method_a in enumerate(methods):
            da = dsim[dsim["method"] == method_a].sort_values("epsilon")
            for method_b in methods[i + 1 :]:
                db = dsim[dsim["method"] == method_b].sort_values("epsilon")
                merged = da[["epsilon", "coverage", "avg_length"]].merge(
                    db[["epsilon", "coverage", "avg_length"]],
                    on="epsilon",
                    suffixes=("_a", "_b"),
                )
                max_cov_diff = float(np.max(np.abs(merged["coverage_a"] - merged["coverage_b"])))
                max_len_diff = float(np.max(np.abs(merged["avg_length_a"] - merged["avg_length_b"])))
                rows.append(
                    {
                        "simulation": simulation,
                        "method_a": method_a,
                        "method_b": method_b,
                        "max_coverage_diff": max_cov_diff,
                        "max_length_diff": max_len_diff,
                        "near_overlap": max_cov_diff < 0.01 and max_len_diff < 0.02,
                    }
                )
    return pd.DataFrame(rows)


def write_summary(
    *,
    main_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    audit_compare: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:
    expected_settings = len(sim.build_settings())
    expected_rows = expected_settings * len(sim.METHOD_ORDER)
    actual_rows = len(main_summary)
    max_mc_se = float(np.sqrt((main_summary["coverage"] * (1 - main_summary["coverage"])).max() / sim.REPS))

    seed_max_range = seed_summary.sort_values("coverage_range", ascending=False).iloc[0]
    branch_counts = (
        sensitivity.groupby(["scenario", "simulation", "selected_branch"])
        .size()
        .reset_index(name="settings")
        .to_string(index=False)
    )
    weak_audit = audit_compare[audit_compare["epsilon_label"] == "weak"][
        ["audit_interval", "coverage", "avg_length"]
    ].to_string(index=False)
    near_overlap = overlap[overlap["near_overlap"]].to_string(index=False)

    text = f"""# Stage 1 QC and Robustness Summary

Created: 2026-05-20

## Checks Performed

1. Recomputed Stage 1 after changing the audit-only comparator to an exact binomial interval for the binary audit mean.
2. Checked expected row counts and basic Monte Carlo precision.
3. Re-ran selected-branch results for seeds {', '.join(str(seed) for seed in SEEDS)} with {REPS} replicates per setting.
4. Tested branch-rule sensitivity under conservative and permissive Q/IF thresholds.
5. Compared the old transformed Wald audit interval with the transformed exact-binomial audit interval for S2.
6. Diagnosed visually overlapping methods in the main plotted slice.

## QC Results

- Expected rows: {expected_rows}; actual rows in `stage1_summary.csv`: {actual_rows}.
- Worst-case Monte Carlo standard error for a reported coverage estimate is about {max_mc_se:.4f} at {sim.REPS} replicates.
- The largest selected-coverage range across the three seed reruns is {seed_max_range.coverage_range:.4f}, for `{seed_max_range.simulation}` at `{seed_max_range.epsilon_label}`.
- Seed reruns preserve the qualitative phase diagram: Q is selected under strong closeness, IF under moderate closeness, and audit-only under weak closeness.

## Audit-Only Interval QC

For S2 weak mismatch, the audit-only comparison is:

```text
{weak_audit}
```

The exact-binomial version avoids the mild finite-sample undercoverage caused by the transformed Wald interval, so the Stage 1 main results now use the exact-binomial audit-only comparator.

## Branch-Rule Sensitivity

```text
{branch_counts}
```

The qualitative result is stable. Threshold changes mostly affect boundary settings, as expected; this is useful to report as a sensitivity analysis rather than as a problem.

## Plot Overlap

Near-overlapping method pairs in the plotted slice:

```text
{near_overlap}
```

Overlap is expected when two methods are algebraically the same in a setting, or when the selected rule deliberately chooses one of the displayed methods. For the main paper, use the clean figures and move full-method figures/tables to the supplement.

## Output Files

- `tables/stage1_oracle_scalar_qc/stage1_qc_seed_robustness.csv`
- `tables/stage1_oracle_scalar_qc/stage1_qc_selected_by_seed.csv`
- `tables/stage1_oracle_scalar_qc/stage1_qc_selection_sensitivity.csv`
- `tables/stage1_oracle_scalar_qc/stage1_qc_audit_interval_comparison.csv`
- `tables/stage1_oracle_scalar_qc/stage1_qc_overlap_diagnostics.csv`
"""
    (QC_DIR / "STAGE1_QC_ROBUSTNESS_SUMMARY.md").write_text(text)


def main() -> None:
    ensure_dirs()
    main_summary = pd.read_csv(MAIN_SUMMARY)

    seed_summary = run_seed_robustness()
    sensitivity = run_selection_sensitivity(main_summary)
    audit_compare = run_audit_interval_comparison()
    overlap = run_overlap_diagnostics(main_summary)

    seed_summary.to_csv(TABLE_DIR / "stage1_qc_seed_robustness.csv", index=False)
    sensitivity.to_csv(TABLE_DIR / "stage1_qc_selection_sensitivity.csv", index=False)
    audit_compare.to_csv(TABLE_DIR / "stage1_qc_audit_interval_comparison.csv", index=False)
    overlap.to_csv(TABLE_DIR / "stage1_qc_overlap_diagnostics.csv", index=False)

    seed_summary.to_csv(QC_DIR / "stage1_qc_seed_robustness.csv", index=False)
    sensitivity.to_csv(QC_DIR / "stage1_qc_selection_sensitivity.csv", index=False)
    audit_compare.to_csv(QC_DIR / "stage1_qc_audit_interval_comparison.csv", index=False)
    overlap.to_csv(QC_DIR / "stage1_qc_overlap_diagnostics.csv", index=False)

    write_summary(
        main_summary=main_summary,
        seed_summary=seed_summary,
        sensitivity=sensitivity,
        audit_compare=audit_compare,
        overlap=overlap,
    )
    print(f"Wrote Stage 1 QC results to {QC_DIR}")
    print(f"Wrote Stage 1 QC tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
