#!/usr/bin/env python3
"""Create manuscript-facing figures and summary tables from generated outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SIM_FIG_DIR = ROOT / "figures" / "manuscript_simulation"
EMP_FIG_DIR = ROOT / "figures" / "manuscript_empirical"
EMP_TABLE_DIR = ROOT / "tables" / "manuscript_empirical"
STORY_TABLE_DIR = ROOT / "tables" / "simulation_story_map"


def ensure_dirs() -> None:
    for path in (SIM_FIG_DIR, EMP_FIG_DIR, EMP_TABLE_DIR, STORY_TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def simulation_figure1(out: Path) -> None:
    df = pd.read_csv(ROOT / "tables" / "stage1_oracle_scalar" / "stage1_s2_residual_scaling.csv")
    eps_vals = np.sort(df["epsilon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharex=True, sharey=True)
    for ax, nu in zip(axes, (0.0, 0.25)):
        dnu = df[np.isclose(df["nu"], nu)]
        for quantity, marker in [("Direct target gap", "o"), ("IF residual", "s")]:
            d = dnu[dnu["quantity"] == quantity].sort_values("epsilon")
            ax.loglog(d["epsilon"], d["value"], marker=marker, linewidth=2.2, markersize=6.5, label=quantity)
        ax.loglog(eps_vals, eps_vals, color="gray", linestyle="--", linewidth=1.5, label="linear in discrepancy")
        ax.loglog(eps_vals, eps_vals**2, color="gray", linestyle=":", linewidth=1.8, label="quadratic in discrepancy")
        ax.set_title(f"Synthetic anchor nu = {nu:g}")
        ax.set_xlabel("True discrepancy epsilon")
        ax.grid(alpha=0.28, which="both")
    axes[0].set_ylabel("Absolute target gap")
    axes[0].legend(loc="upper left", frameon=True)
    save_figure(fig, out)


def simulation_figure2(out: Path) -> None:
    df = pd.read_csv(ROOT / "tables" / "stage3_semiparametric_ate" / "stage3_summary.csv")
    data = df[df["method"].isin(["audit_only", "if_inflated"])].copy()
    variants = [("moderate_nuisance", "Moderate nuisance variation"), ("hard_nuisance", "Hard nuisance variation")]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharex=True)
    for variant, label in variants:
        ifd = data[(data["variant"] == variant) & (data["method"] == "if_inflated")].sort_values("direct_gap")
        aud = data[(data["variant"] == variant) & (data["method"] == "audit_only")].sort_values("direct_gap")
        axes[0].plot(ifd["direct_gap"], ifd["avg_length"], marker="o", linewidth=2.2, markersize=6.5, label=f"IF, {label}")
        axes[0].plot(aud["direct_gap"], aud["avg_length"], marker="x", linestyle="--", linewidth=2.0, markersize=7.0, label=f"Audit, {label}")
        ratio = aud["avg_length"].to_numpy() / ifd["avg_length"].to_numpy()
        axes[1].plot(ifd["direct_gap"], ratio, marker="o", linewidth=2.2, markersize=6.5, label=label)
    for ax in axes:
        ax.set_xlabel("Direct ATE gap")
        ax.grid(alpha=0.28)
    axes[0].set_ylabel("Average interval length")
    axes[1].set_ylabel("Audit-only length / IF length")
    axes[1].axhline(1.0, color="black", linewidth=1.0)
    axes[0].legend(loc="best", frameon=True)
    axes[1].legend(loc="best", frameon=True)
    save_figure(fig, out)


def simulation_figure3(out: Path) -> None:
    df = pd.read_csv(ROOT / "tables" / "stage4_learned_generator_diagnostics_qc" / "stage4_qc_key_rows.csv")
    calibrated_multiplier = 1.03
    base = df[
        ((df["method"] != "if_learned_rich") & np.isclose(df["se_multiplier"], 1.0))
        | ((df["method"] == "if_learned_rich") & np.isclose(df["se_multiplier"], calibrated_multiplier))
    ].copy()
    label_map = {
        "audit_only": "Audit only",
        "q_inflated": "Q centered",
        "if_oracle": "IF oracle",
        "if_learned_rich": "IF learned rich, x1.03",
        "if_learned_linear": "IF learned linear",
    }
    order = ["audit_only", "q_inflated", "if_oracle", "if_learned_rich", "if_learned_linear"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharex=True)
    for method in order:
        d = base[base["method"] == method].sort_values("direct_gap")
        axes[0].plot(d["direct_gap"], d["coverage"], marker="o", linewidth=2.2, markersize=6.5, label=label_map[method])
        axes[1].plot(d["direct_gap"], d["avg_length"], marker="o", linewidth=2.2, markersize=6.5, label=label_map[method])
    axes[0].axhline(0.95, color="black", linewidth=1.0)
    for ax in axes:
        ax.grid(alpha=0.28)
        ax.set_xlabel("Direct ATE gap")
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Average interval length")
    axes[0].legend(loc="best", frameon=True)
    axes[1].legend(loc="best", frameon=True)
    save_figure(fig, out)


def simulation_figure4(out: Path) -> None:
    df = pd.read_csv(ROOT / "tables" / "selection_repair_feasible" / "selection_repair_summary.csv")
    selected = df[df["method"] == "selected_feasible"].copy()
    label_map = {
        "zero": "zero gap",
        "strong_boundary": "strong boundary",
        "moderate_mid": "moderate mid",
        "moderate_boundary": "moderate boundary",
        "weak": "weak",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharex=True)
    for gamma_label, d in selected.groupby("gamma_label", sort=False):
        d = d.sort_values("sensitivity_radius")
        label = label_map.get(gamma_label, gamma_label.replace("_", " "))
        axes[0].plot(d["sensitivity_radius"], d["coverage"], marker="o", linewidth=2.2, markersize=6.5, label=label)
        axes[1].plot(d["sensitivity_radius"], d["avg_length"], marker="o", linewidth=2.2, markersize=6.5, label=label)
    axes[0].axhline(0.95, color="black", linewidth=1.0)
    for ax in axes:
        ax.set_xlabel("Provided sensitivity bound for Q centered branch")
        ax.grid(alpha=0.28)
    axes[0].set_ylabel("Selected interval coverage")
    axes[1].set_ylabel("Selected interval average length")
    axes[0].legend(loc="best", frameon=True)
    axes[1].legend(loc="best", frameon=True)
    save_figure(fig, out)


def simulation_figure5(out: Path) -> None:
    df = pd.read_csv(ROOT / "tables" / "stage5_modern_tabular_generator" / "stage5_summary.csv")
    generator_order = ["gaussian_copula", "tvae", "ctgan"]
    generator_labels = {"gaussian_copula": "Gaussian copula", "tvae": "TVAE", "ctgan": "CTGAN"}
    scenario_order = ["strong_boundary", "moderate_boundary", "weak"]
    scenario_labels = {"strong_boundary": "strong", "moderate_boundary": "moderate", "weak": "weak"}
    method_order = ["audit_only", "synthetic_naive", "q_inflated_oracle", "if_generator_rich", "selected"]
    method_labels = {
        "audit_only": "Audit only",
        "synthetic_naive": "Synthetic naive",
        "q_inflated_oracle": "Q centered, oracle gap",
        "if_generator_rich": "Audit driven IF, x1.03",
        "selected": "Selected",
    }
    styles = {
        "audit_only": dict(marker="o", linewidth=2.1),
        "synthetic_naive": dict(marker="o", linewidth=2.1),
        "q_inflated_oracle": dict(marker="o", linewidth=2.1),
        "if_generator_rich": dict(marker="D", linewidth=2.1),
        "selected": dict(marker="D", linewidth=1.6, color="black", markerfacecolor="white"),
    }
    pos = {label: i for i, label in enumerate(scenario_order)}
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.6), sharex=True)
    for col, generator in enumerate(generator_order):
        gen = df[df["generator"] == generator].copy()
        for method in method_order:
            d = gen[gen["method"] == method].copy()
            d["x_pos"] = d["gamma_label"].map(pos)
            d = d.sort_values("x_pos")
            label = method_labels[method]
            axes[0, col].plot(d["x_pos"], d["coverage"], label=label, markersize=6.5, **styles[method])
            axes[1, col].plot(d["x_pos"], d["avg_length"], label=label, markersize=6.5, **styles[method])
        axes[0, col].axhline(0.95, color="black", linewidth=1.0)
        axes[0, col].set_title(generator_labels[generator])
        axes[0, col].set_ylim(0, 1.03)
        axes[1, col].set_xticks(list(pos.values()), [scenario_labels[s] for s in scenario_order], rotation=0)
        axes[1, col].set_xlabel("Source mismatch regime")
        for row in range(2):
            axes[row, col].grid(alpha=0.28)
    axes[0, 0].set_ylabel("Empirical coverage")
    axes[1, 0].set_ylabel("Average interval length")
    axes[0, 0].legend(loc="lower left", frameon=True)
    save_figure(fig, out)


def write_simulation_story_map() -> None:
    rows = [
        {
            "setting": "1",
            "role": "Scalar mechanism check",
            "main_script": "simulation/stage1_oracle_scalar.py",
            "claim_checked": "Direct Q-centered transfer is first order; audit-driven residual can be quadratic.",
        },
        {
            "setting": "2",
            "role": "Estimating equation check",
            "main_script": "simulation/stage2_estimating_equation.py",
            "claim_checked": "Moment and derivative discrepancies determine the post-correction residual.",
        },
        {
            "setting": "3",
            "role": "Oracle semiparametric ATE value benchmark",
            "main_script": "simulation/stage3_semiparametric_ate.py",
            "claim_checked": "Audit-driven IF intervals can be much shorter than audit-only intervals when synthetic nuisances are useful.",
        },
        {
            "setting": "4",
            "role": "Learned nuisance and calibration check",
            "main_script": "simulation/stage4_learned_generator_diagnostics.py",
            "claim_checked": "Learned synthetic nuisances preserve much of the oracle gain after modest calibration.",
        },
        {
            "setting": "5",
            "role": "Feasible selected reporting",
            "main_script": "simulation/selection_repair_feasible.py",
            "claim_checked": "Selected reporting is valid when the provided discrepancy bound is valid and can fall back when Q-centered borrowing is not attractive.",
        },
        {
            "setting": "6",
            "role": "Modern tabular-generator stress test",
            "main_script": "simulation/stage5_modern_tabular_generator.py",
            "claim_checked": "Generic high-capacity tabular generators can still distort the target-relevant ATE gap.",
        },
    ]
    pd.DataFrame(rows).to_csv(STORY_TABLE_DIR / "simulation_story_map.csv", index=False)


def read_hillstrom_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = pd.read_csv(ROOT / "tables" / "hillstrom_task_aware_qc" / "hillstrom_task_aware_qc_method_summary.csv")
    generator = pd.read_csv(ROOT / "tables" / "hillstrom_task_aware_qc" / "hillstrom_task_aware_qc_generator_summary.csv")
    generic = pd.read_csv(ROOT / "tables" / "hillstrom_generator_sensitivity" / "hillstrom_generator_gap_summary.csv")
    return method, generator, generic


def build_hillstrom_main_table(method: pd.DataFrame) -> pd.DataFrame:
    keep_methods = [
        ("audit_only", np.nan, "Audit only", "Trusted baseline"),
        ("task_synthetic_naive", np.nan, "Synthetic naive", "No discrepancy correction"),
        ("task_q_inflated", 1.0, "Q centered, c = 1", "Low-discrepancy branch under provided bound"),
        ("task_if_learned", np.nan, "Audit driven IF", "Bias-corrected branch"),
        ("selected_feasible", 1.0, "Selected, c = 1", "Selected among candidates under provided bound"),
    ]
    rows = []
    for method_name, c, label, role in keep_methods:
        row = {"Method": label, "Role": role}
        for lam in (0.5, 1.0):
            subset = method[(method["lambda"] == lam) & (method["method"] == method_name)]
            if np.isnan(c):
                subset = subset[subset["sensitivity_c"].isna()]
            else:
                subset = subset[subset["sensitivity_c"] == c]
            if subset.empty:
                raise RuntimeError(f"Missing row for {method_name}, lambda={lam}, c={c}")
            rec = subset.iloc[0]
            row[f"Coverage, lambda={lam}"] = f"{rec['coverage']:.3f}"
            row[f"Length, lambda={lam}"] = f"{rec['avg_length']:.3f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(EMP_TABLE_DIR / "hillstrom_empirical_main_table.csv", index=False)


def hillstrom_main_figure(method: pd.DataFrame, generator: pd.DataFrame, generic: pd.DataFrame) -> None:
    methods = [
        ("audit_only", np.nan, "Audit only", "#2f6f9f"),
        ("task_synthetic_naive", np.nan, "Synthetic naive", "#c44e52"),
        ("task_q_inflated", 1.0, "Q centered, c=1", "#dd8452"),
        ("task_if_learned", np.nan, "Audit driven IF", "#55a868"),
        ("selected_feasible", 1.0, "Selected, c=1", "#222222"),
    ]
    lambdas = [0.5, 1.0]

    def values(metric: str) -> list[list[float]]:
        out = []
        for method_name, c, _, _ in methods:
            vals = []
            for lam in lambdas:
                sub = method[(method["lambda"] == lam) & (method["method"] == method_name)]
                if np.isnan(c):
                    sub = sub[sub["sensitivity_c"].isna()]
                else:
                    sub = sub[sub["sensitivity_c"] == c]
                vals.append(float(sub.iloc[0][metric]))
            out.append(vals)
        return out

    coverage = values("coverage")
    length = values("avg_length")
    source_gap = [float(generator[generator["lambda"] == lam]["source_gap_mean"].iloc[0]) for lam in lambdas]
    task_gap = [float(generator[generator["lambda"] == lam]["generator_gap_mean"].iloc[0]) for lam in lambdas]
    generic_gap = [float(generic[generic["lambda"] == lam]["generator_gap_mean"].mean()) for lam in lambdas]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    x = np.arange(len(lambdas))
    width = 0.15
    offsets = np.linspace(-2, 2, len(methods)) * width

    for i, (_, _, label, color) in enumerate(methods):
        axes[0].bar(x + offsets[i], coverage[i], width=width, label=label, color=color, edgecolor="black", linewidth=0.4)
        axes[1].bar(x + offsets[i], length[i], width=width, label=label, color=color, edgecolor="black", linewidth=0.4)

    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(["lambda = 0.5", "lambda = 1.0"])
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Source bias setting")
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.70, 1.04)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Coverage")
    axes[1].set_ylim(0.0, 0.14)
    axes[1].set_ylabel("Average interval length")
    axes[1].set_title("Interval length")

    gap_labels = ["Source", "Task-aware generator", "Generic generator"]
    gap_values = [source_gap, task_gap, generic_gap]
    gap_colors = ["#8172b3", "#4c72b0", "#ccb974"]
    gap_width = 0.23
    for i, (label, vals, color) in enumerate(zip(gap_labels, gap_values, gap_colors)):
        axes[2].bar(x + (i - 1) * gap_width, vals, width=gap_width, label=label, color=color, edgecolor="black", linewidth=0.4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["lambda = 0.5", "lambda = 1.0"])
    axes[2].set_xlabel("Source bias setting")
    axes[2].set_ylabel("Absolute ATE gap")
    axes[2].set_title("ATE gap")
    axes[2].grid(axis="y", alpha=0.25)

    axes[0].legend(loc="lower left", frameon=True)
    axes[2].legend(loc="upper left", frameon=True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(EMP_FIG_DIR / "hillstrom_empirical_main.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_plot_style()
    simulation_figure1(SIM_FIG_DIR / "figure1_scalar_mechanism.png")
    simulation_figure2(SIM_FIG_DIR / "figure2_semiparametric_value.png")
    simulation_figure3(SIM_FIG_DIR / "figure3_learned_nuisance.png")
    simulation_figure4(SIM_FIG_DIR / "figure4_feasible_selection.png")
    simulation_figure5(SIM_FIG_DIR / "figure5_modern_generators.png")
    write_simulation_story_map()

    method, generator, generic = read_hillstrom_inputs()
    build_hillstrom_main_table(method)
    hillstrom_main_figure(method, generator, generic)

    print("Wrote manuscript-facing figures under figures/.")
    print("Wrote manuscript-facing tables under tables/.")


if __name__ == "__main__":
    main()
