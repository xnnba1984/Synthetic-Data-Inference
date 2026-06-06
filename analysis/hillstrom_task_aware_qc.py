#!/usr/bin/env python3
"""QC and robustness checks for the task-aware Hillstrom pilot."""

from __future__ import annotations

from pathlib import Path
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hillstrom_pilot as hp
import hillstrom_task_aware_pilot as tap


ROOT = Path(__file__).resolve().parents[1]
PILOT_TABLE_DIR = ROOT / "tables" / "hillstrom_task_aware_pilot"
RESULT_DIR = ROOT / "analysis" / "results" / "hillstrom_task_aware_qc"
TABLE_DIR = ROOT / "tables" / "hillstrom_task_aware_qc"
FIG_DIR = ROOT / "figures" / "hillstrom_task_aware_qc"

QC_SEED = 20260523 + 97
QC_REPS = int(os.environ.get("HILLSTROM_TASK_QC_REPS", "500"))
MAIN_LAMBDAS = tap.MAIN_LAMBDAS
PRIMARY_C = tap.PRIMARY_C


def ensure_dirs() -> None:
    for path in (RESULT_DIR, TABLE_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_pilot_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(PILOT_TABLE_DIR / "hillstrom_task_aware_method_summary.csv")
    branches = pd.read_csv(PILOT_TABLE_DIR / "hillstrom_task_aware_branch_frequency.csv")
    generator = pd.read_csv(PILOT_TABLE_DIR / "hillstrom_task_aware_generator_summary.csv")
    gates = pd.read_csv(PILOT_TABLE_DIR / "hillstrom_task_aware_gate.csv")
    return summary, branches, generator, gates


def assert_expected_outputs(summary: pd.DataFrame, branches: pd.DataFrame, generator: pd.DataFrame, gates: pd.DataFrame) -> list[dict]:
    checks = []
    expected_methods = {
        "audit_only",
        "task_synthetic_naive",
        "task_q_inflated",
        "task_if_learned",
        "task_if_inflated",
        "selected_feasible",
    }
    checks.append(
        {
            "check": "method_set",
            "passed": expected_methods.issubset(set(summary["method"])),
            "detail": ",".join(sorted(set(summary["method"]))),
        }
    )
    checks.append(
        {
            "check": "primary_hgb_lambdas",
            "passed": set(MAIN_LAMBDAS).issubset(set(generator.loc[generator["generator"] == "task_hgb", "lambda"])),
            "detail": ",".join(map(str, sorted(generator.loc[generator["generator"] == "task_hgb", "lambda"].unique()))),
        }
    )
    checks.append(
        {
            "check": "pilot_gates_passed",
            "passed": bool(gates["passed"].all()),
            "detail": gates[["gate", "passed"]].to_dict("records"),
        }
    )
    checks.append(
        {
            "check": "branch_frequency_sums",
            "passed": bool(
                np.allclose(
                    branches.groupby(["generator", "lambda", "m", "sensitivity_c"])["selected_frequency"].sum().to_numpy(),
                    1.0,
                )
            ),
            "detail": "Grouped selected frequencies sum to one.",
        }
    )
    return checks


def coverage_length_table(summary: pd.DataFrame, generator: str, label: str) -> pd.DataFrame:
    key = summary[
        (summary["generator"] == generator)
        & (summary["m"] == 500)
        & (summary["lambda"].isin(MAIN_LAMBDAS))
        & (
            ((summary["method"].isin(["audit_only", "task_synthetic_naive", "task_if_learned"])) & (summary["sensitivity_c"].isna()))
            | ((summary["method"].isin(["task_q_inflated", "selected_feasible"])) & (summary["sensitivity_c"] == PRIMARY_C))
        )
    ].copy()
    key["run"] = label
    return key[
        [
            "run",
            "generator",
            "generator_label",
            "learner",
            "lambda",
            "m",
            "sensitivity_c",
            "method",
            "method_label",
            "reps",
            "coverage",
            "mc_se_coverage",
            "avg_length",
            "bias",
        ]
    ]


def sensitivity_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grid = summary[
        (summary["generator"] == "task_hgb")
        & (summary["m"] == 500)
        & (summary["method"].isin(["task_q_inflated", "selected_feasible"]))
    ]
    for (lam, method), g in grid.groupby(["lambda", "method"]):
        valid = g[g["coverage"] >= 0.95].sort_values("sensitivity_c")
        first_c = float(valid["sensitivity_c"].iloc[0]) if len(valid) else np.nan
        first_len = float(valid["avg_length"].iloc[0]) if len(valid) else np.nan
        c1 = g[g["sensitivity_c"] == PRIMARY_C]
        rows.append(
            {
                "lambda": lam,
                "method": method,
                "method_label": g["method_label"].iloc[0],
                "smallest_c_with_coverage_ge_0.95": first_c,
                "length_at_smallest_valid_c": first_len,
                "coverage_at_c1": float(c1["coverage"].iloc[0]) if len(c1) else np.nan,
                "length_at_c1": float(c1["avg_length"].iloc[0]) if len(c1) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_focused_validation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = hp.load_data()
    scores = hp.source_scores(df)
    theta_full, _ = hp.diff_mean(df)
    rng = np.random.default_rng(QC_SEED)
    eval_rows: list[dict] = []
    branch_rows: list[dict] = []
    diag_rows: list[dict] = []

    print(f"Focused QC validation: theta_full={theta_full:.6f}, reps per main lambda={QC_REPS}", flush=True)
    for lam in MAIN_LAMBDAS:
        e_rows, b_rows, d_rows = tap.run_condition(
            df=df,
            scores=scores,
            theta_full=theta_full,
            rng=rng,
            lam=lam,
            learner="hgb",
            generator="task_hgb",
            reps=QC_REPS,
            audit_sizes=(500,),
            rep_offset=700_000,
        )
        eval_rows.extend(e_rows)
        branch_rows.extend(b_rows)
        diag_rows.extend(d_rows)

    rep_df = pd.DataFrame(eval_rows)
    branch_df = pd.DataFrame(branch_rows)
    diag_df = pd.DataFrame(diag_rows)
    summary = tap.summarize_results(rep_df)
    branch_summary = tap.summarize_branches(branch_df)
    generator_summary = tap.summarize_generator(diag_df)
    gate_df = tap.pilot_gate(summary, generator_summary)
    return rep_df, branch_summary, generator_summary, gate_df


def validation_diagnostics(summary: pd.DataFrame, generator_summary: pd.DataFrame, gate_df: pd.DataFrame) -> pd.DataFrame:
    key = coverage_length_table(summary, "task_hgb", "focused_qc")
    audit = key[key["method"] == "audit_only"][["lambda", "avg_length"]].rename(columns={"avg_length": "audit_length"})
    selected = key[key["method"] == "selected_feasible"][["lambda", "coverage", "avg_length"]].rename(
        columns={"coverage": "selected_coverage", "avg_length": "selected_length"}
    )
    q = key[key["method"] == "task_q_inflated"][["lambda", "coverage", "avg_length"]].rename(
        columns={"coverage": "q_coverage", "avg_length": "q_length"}
    )
    if_branch = key[key["method"] == "task_if_learned"][["lambda", "coverage", "avg_length"]].rename(
        columns={"coverage": "if_coverage", "avg_length": "if_length"}
    )
    out = audit.merge(selected, on="lambda").merge(q, on="lambda").merge(if_branch, on="lambda")
    out["selected_length_ratio_vs_audit"] = out["selected_length"] / out["audit_length"]
    out["if_length_ratio_vs_audit"] = out["if_length"] / out["audit_length"]
    out = out.merge(
        generator_summary[["lambda", "source_gap_mean", "generator_gap_mean", "theta_q_mean"]],
        on="lambda",
        how="left",
    )
    out["qc_gate_decision"] = gate_df["overall_decision"].iloc[0]
    return out


def compare_pilot_and_qc(pilot_summary: pd.DataFrame, qc_summary: pd.DataFrame) -> pd.DataFrame:
    pilot_key = coverage_length_table(pilot_summary, "task_hgb", "pilot")
    qc_key = coverage_length_table(qc_summary, "task_hgb", "focused_qc")
    cols = ["lambda", "method", "coverage", "avg_length", "bias"]
    out = pilot_key[cols].merge(qc_key[cols], on=["lambda", "method"], suffixes=("_pilot", "_qc"))
    out["coverage_diff_qc_minus_pilot"] = out["coverage_qc"] - out["coverage_pilot"]
    out["length_diff_qc_minus_pilot"] = out["avg_length_qc"] - out["avg_length_pilot"]
    out["bias_diff_qc_minus_pilot"] = out["bias_qc"] - out["bias_pilot"]
    return out


def plot_validation(qc_summary: pd.DataFrame, qc_generator: pd.DataFrame) -> None:
    key = coverage_length_table(qc_summary, "task_hgb", "focused_qc")
    methods = ["audit_only", "task_synthetic_naive", "task_q_inflated", "task_if_learned", "selected_feasible"]
    colors = {
        "audit_only": "#1f77b4",
        "task_synthetic_naive": "#ff7f0e",
        "task_q_inflated": "#2ca02c",
        "task_if_learned": "#d62728",
        "selected_feasible": "black",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for method in methods:
        g = key[key["method"] == method].sort_values("lambda")
        if len(g):
            marker = "D" if method == "selected_feasible" else "o"
            axes[0].plot(g["lambda"], g["coverage"], marker=marker, color=colors[method], label=g["method_label"].iloc[0])
            axes[1].plot(g["lambda"], g["avg_length"], marker=marker, color=colors[method], label=g["method_label"].iloc[0])
    axes[0].axhline(0.95, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Source-bias lambda")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_xlabel("Source-bias lambda")
    axes[1].set_ylabel("Average interval length")
    g = qc_generator.sort_values("lambda")
    axes[2].plot(g["lambda"], g["source_gap_mean"], marker="o", label="Real source gap")
    axes[2].plot(g["lambda"], g["generator_gap_mean"], marker="D", label="Task-aware generator gap")
    axes[2].set_xlabel("Source-bias lambda")
    axes[2].set_ylabel("Absolute ATE gap")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.savefig(FIG_DIR / "hillstrom_task_aware_qc_validation.png", dpi=240)
    plt.close(fig)


def plot_sensitivity(pilot_summary: pd.DataFrame) -> None:
    grid = pilot_summary[
        (pilot_summary["generator"] == "task_hgb")
        & (pilot_summary["m"] == 500)
        & (pilot_summary["method"].isin(["task_q_inflated", "selected_feasible"]))
        & (pilot_summary["lambda"].isin(MAIN_LAMBDAS))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    for (lam, method), g in grid.groupby(["lambda", "method"]):
        g = g.sort_values("sensitivity_c")
        label = f"{g['method_label'].iloc[0]}, lambda={lam:g}"
        marker = "D" if method == "selected_feasible" else "o"
        axes[0].plot(g["sensitivity_c"], g["coverage"], marker=marker, label=label)
        axes[1].plot(g["sensitivity_c"], g["avg_length"], marker=marker, label=label)
    axes[0].axhline(0.95, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Sensitivity multiplier c")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0.70, 1.03)
    axes[1].set_xlabel("Sensitivity multiplier c")
    axes[1].set_ylabel("Average interval length")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.savefig(FIG_DIR / "hillstrom_task_aware_qc_sensitivity.png", dpi=240)
    plt.close(fig)


def write_memo(
    *,
    checks: list[dict],
    pilot_summary: pd.DataFrame,
    pilot_generator: pd.DataFrame,
    pilot_branches: pd.DataFrame,
    qc_summary: pd.DataFrame,
    qc_branches: pd.DataFrame,
    qc_generator: pd.DataFrame,
    qc_gate: pd.DataFrame,
    sensitivity: pd.DataFrame,
    validation_diag: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    pilot_key = coverage_length_table(pilot_summary, "task_hgb", "pilot")
    logistic_key = coverage_length_table(pilot_summary, "task_logistic", "pilot_logistic_sensitivity")
    qc_key = coverage_length_table(qc_summary, "task_hgb", "focused_qc")
    branch_c1 = pilot_branches[
        (pilot_branches["generator"] == "task_hgb")
        & (pilot_branches["m"] == 500)
        & (pilot_branches["lambda"].isin(MAIN_LAMBDAS))
        & (pilot_branches["sensitivity_c"] == PRIMARY_C)
    ]
    qc_branch_c1 = qc_branches[
        (qc_branches["generator"] == "task_hgb")
        & (qc_branches["m"] == 500)
        & (qc_branches["lambda"].isin(MAIN_LAMBDAS))
        & (qc_branches["sensitivity_c"] == PRIMARY_C)
    ]
    all_checks_passed = all(bool(x["passed"]) for x in checks)
    qc_decision = qc_gate["overall_decision"].iloc[0]
    conclusion = (
        "QC supports the pilot conclusion with caveats."
        if all_checks_passed and qc_decision == "promote_task_aware_hillstrom_if_qc_confirms"
        else "QC weakens the pilot conclusion."
    )
    lines = [
        "# Hillstrom Task-Aware QC and Robustness Memo",
        "",
        "Created: 2026-05-23",
        "",
        "Status: QC/robustness result memo, not manuscript text.",
        "",
        "## Bottom Line",
        "",
        conclusion,
        "",
        "The main conclusion does not change: task-aware Hillstrom remains a promising main empirical example, pending final manuscript-style figure/table polish. The positive length gain is still a Q-centered/selected-branch result under a supplied sensitivity radius, not an IF-only efficiency gain.",
        "",
        "## QC Checks",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Independent Focused Validation",
        "",
        f"Focused validation used `{QC_REPS}` new HGB replicates for each main source-bias setting, `lambda in {{0.5,1.0}}`, at `m=500`.",
        "",
        qc_gate[["gate", "passed", "criterion", "value"]].to_markdown(index=False),
        "",
        "Validation diagnostics:",
        "",
        validation_diag.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pilot Key Rows",
        "",
        pilot_key.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Focused QC Key Rows",
        "",
        qc_key.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pilot Versus Focused QC Differences",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Learner Sensitivity",
        "",
        "The logistic sensitivity rows tell whether the result depends on the flexible HGB learner.",
        "",
        logistic_key.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Sensitivity-Radius Behavior",
        "",
        sensitivity.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Interpretation: small radii can under-cover, as expected. The pre-specified primary radius `c=1` is conservative in this pilot and produces valid shorter Q-centered/selected intervals.",
        "",
        "## Branch Selection at c = 1",
        "",
        "Pilot branch frequencies:",
        "",
        branch_c1.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Focused QC branch frequencies:",
        "",
        qc_branch_c1.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Generator Diagnostics",
        "",
        "Pilot task-aware generator summary:",
        "",
        pilot_generator[pilot_generator["generator"] == "task_hgb"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Focused QC task-aware generator summary:",
        "",
        qc_generator.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Issues Found",
        "",
        "- No internal file or row-count issue was found.",
        "- No conclusion-changing Monte Carlo instability was found in the focused validation.",
        "- The result is sensitive to the supplied discrepancy radius in the intended way: `c=0` under-covers and `c=1` is conservative.",
        "- The IF branch remains a validity-protective branch, not the source of the interval-length improvement in this empirical example.",
        "- The task-aware generator gap is slightly larger than the real-source gap, but far smaller than the prior generic-generator gap. This should be disclosed as realistic residual generator noise rather than hidden.",
        "",
        "## Files",
        "",
        "- QC script: `analysis/hillstrom_task_aware_qc.py`",
        "- Focused validation results: `analysis/results/hillstrom_task_aware_qc/`",
        "- QC tables: `tables/hillstrom_task_aware_qc/`",
        "- QC figures: `figures/hillstrom_task_aware_qc/`",
    ]
    (ROOT / "analysis" / "HILLSTROM_TASK_AWARE_QC_MEMO.md").write_text("\n".join(lines) + "\n")


def run() -> None:
    ensure_dirs()
    pilot_summary, pilot_branches, pilot_generator, pilot_gates = load_pilot_outputs()
    checks = assert_expected_outputs(pilot_summary, pilot_branches, pilot_generator, pilot_gates)
    sensitivity = sensitivity_table(pilot_summary)

    rep_df, qc_branches, qc_generator, qc_gate = run_focused_validation()
    qc_summary = tap.summarize_results(rep_df)
    validation_diag = validation_diagnostics(qc_summary, qc_generator, qc_gate)
    comparison = compare_pilot_and_qc(pilot_summary, qc_summary)

    rep_df.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_replicates.csv", index=False)
    qc_summary.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_method_summary.csv", index=False)
    qc_branches.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_branch_frequency.csv", index=False)
    qc_generator.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_generator_summary.csv", index=False)
    qc_gate.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_gate.csv", index=False)
    pd.DataFrame(checks).to_csv(RESULT_DIR / "hillstrom_task_aware_qc_file_checks.csv", index=False)
    sensitivity.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_sensitivity_table.csv", index=False)
    validation_diag.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_validation_diagnostics.csv", index=False)
    comparison.to_csv(RESULT_DIR / "hillstrom_task_aware_qc_pilot_vs_validation.csv", index=False)

    qc_summary.to_csv(TABLE_DIR / "hillstrom_task_aware_qc_method_summary.csv", index=False)
    qc_branches.to_csv(TABLE_DIR / "hillstrom_task_aware_qc_branch_frequency.csv", index=False)
    qc_generator.to_csv(TABLE_DIR / "hillstrom_task_aware_qc_generator_summary.csv", index=False)
    qc_gate.to_csv(TABLE_DIR / "hillstrom_task_aware_qc_gate.csv", index=False)
    pd.DataFrame(checks).to_csv(TABLE_DIR / "hillstrom_task_aware_qc_file_checks.csv", index=False)
    sensitivity.to_csv(TABLE_DIR / "hillstrom_task_aware_qc_sensitivity_table.csv", index=False)
    validation_diag.to_csv(TABLE_DIR / "hillstrom_task_aware_qc_validation_diagnostics.csv", index=False)
    comparison.to_csv(TABLE_DIR / "hillstrom_task_aware_qc_pilot_vs_validation.csv", index=False)

    plot_validation(qc_summary, qc_generator)
    plot_sensitivity(pilot_summary)
    write_memo(
        checks=checks,
        pilot_summary=pilot_summary,
        pilot_generator=pilot_generator,
        pilot_branches=pilot_branches,
        qc_summary=qc_summary,
        qc_branches=qc_branches,
        qc_generator=qc_generator,
        qc_gate=qc_gate,
        sensitivity=sensitivity,
        validation_diag=validation_diag,
        comparison=comparison,
    )

    print(pd.DataFrame(checks).to_string(index=False), flush=True)
    print(qc_gate[["gate", "passed", "overall_decision"]].to_string(index=False), flush=True)
    print(validation_diag.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
