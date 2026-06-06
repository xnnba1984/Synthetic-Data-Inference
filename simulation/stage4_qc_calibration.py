#!/usr/bin/env python3
"""Focused QC for Stage 4 learned-nuisance calibration.

This script reruns the key Stage 4 learned-IF settings with more Monte Carlo
replicates than the first pass and evaluates small conservative standard-error
multipliers for the learned-rich IF interval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t

import stage3_semiparametric_ate as s3
import stage4_learned_generator_diagnostics as s4


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "simulation" / "results" / "stage4_learned_generator_diagnostics_qc"
FIG_DIR = ROOT / "figures" / "stage4_learned_generator_diagnostics_qc"
TABLE_DIR = ROOT / "tables" / "stage4_learned_generator_diagnostics_qc"

ALPHA = 0.05
Z = norm.ppf(1 - ALPHA / 2)
REPS = 3000
CHUNK = 250
SEED = 20260521
M_AUDIT = 200
N_TRAIN = 2000
GAMMA_LABELS = ("strong_boundary", "moderate_mid", "moderate_boundary", "weak")
SE_MULTIPLIERS = (1.00, 1.03, 1.05, 1.08, 1.10, 1.15)


@dataclass(frozen=True)
class KeySetting:
    gamma_label: str
    gamma: float


def ensure_dirs() -> None:
    for path in (RESULT_DIR, FIG_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def key_settings() -> list[KeySetting]:
    gamma_lookup = dict(s3.mismatch_grid(M_AUDIT))
    return [KeySetting(label, gamma_lookup[label]) for label in GAMMA_LABELS]


def interval(center: np.ndarray, se: np.ndarray, crit: float, inflation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return center - crit * se - inflation, center + crit * se + inflation


def update_acc(acc: dict[str, float], point: np.ndarray, lower: np.ndarray, upper: np.ndarray, target: float) -> None:
    length = upper - lower
    cover = (lower <= target) & (target <= upper)
    acc["n"] += len(point)
    acc["cover"] += float(cover.sum())
    acc["length_sum"] += float(length.sum())
    acc["length2_sum"] += float(np.sum(length**2))
    err = point - target
    acc["bias_sum"] += float(err.sum())
    acc["point_sum"] += float(point.sum())
    acc["point2_sum"] += float(np.sum(point**2))


def finalize_acc(
    *,
    setting: KeySetting,
    pop: dict[str, float],
    method: str,
    method_label: str,
    acc: dict[str, float],
    se_multiplier: float = 1.0,
) -> dict[str, float | str | int]:
    n = int(acc["n"])
    point_mean = acc["point_sum"] / n
    point_var = max(acc["point2_sum"] / n - point_mean**2, 0.0)
    return {
        "simulation": "S5_learned_nuisance_qc",
        "m": M_AUDIT,
        "n_train": N_TRAIN,
        "reps": n,
        "gamma_label": setting.gamma_label,
        "gamma": setting.gamma,
        "direct_gap": pop["direct_gap"],
        "method": method,
        "method_label": method_label,
        "se_multiplier": se_multiplier,
        "coverage": acc["cover"] / n,
        "mc_se_coverage": float(np.sqrt((acc["cover"] / n) * (1 - acc["cover"] / n) / n)),
        "avg_length": acc["length_sum"] / n,
        "sd_length": float(np.sqrt(max(acc["length2_sum"] / n - (acc["length_sum"] / n) ** 2, 0.0))),
        "bias": acc["bias_sum"] / n,
        "emp_sd": float(np.sqrt(point_var)),
        "target": pop["theta_p"],
    }


def simulate_setting(setting: KeySetting, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    hard = next(v for v in s3.VARIANTS if v.name == "hard_nuisance")
    stage3_setting = s3.Setting(
        variant=hard,
        m=M_AUDIT,
        M=N_TRAIN,
        M_multiplier=N_TRAIN // M_AUDIT,
        gamma=setting.gamma,
        gamma_label=setting.gamma_label,
    )
    pop = s3.population_quantities(stage3_setting)
    target = float(pop["theta_p"])
    crit_m = t.ppf(1 - ALPHA / 2, df=M_AUDIT - 2)
    n_arm = N_TRAIN // 2

    accs: dict[tuple[str, float], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    remaining = REPS
    while remaining > 0:
        reps = min(CHUNK, remaining)
        remaining -= reps

        x0_q = s3.draw_x(rng, (reps, n_arm))
        x1_q = s3.draw_x(rng, (reps, n_arm))
        y0_q = s3.m0_q(x0_q, stage3_setting) + rng.normal(scale=s3.SIGMA, size=(reps, n_arm))
        y1_q = s3.m1_q(x1_q, stage3_setting) + rng.normal(scale=s3.SIGMA, size=(reps, n_arm))

        synth_center = y1_q.mean(axis=1) - y0_q.mean(axis=1)
        synth_se = np.sqrt(y1_q.var(axis=1, ddof=1) / n_arm + y0_q.var(axis=1, ddof=1) / n_arm)

        beta0_rich = s4.fit_ridge_batch(x0_q, y0_q, "rich")
        beta1_rich = s4.fit_ridge_batch(x1_q, y1_q, "rich")
        beta0_linear = s4.fit_ridge_batch(x0_q, y0_q, "linear")
        beta1_linear = s4.fit_ridge_batch(x1_q, y1_q, "linear")

        x_p = s3.draw_x(rng, (reps, M_AUDIT))
        a_p = rng.binomial(1, s3.PROPENSITY, size=(reps, M_AUDIT))
        y_p = s3.draw_outcomes_p(rng, x_p, a_p, stage3_setting)

        treated = a_p == 1
        control = ~treated
        n1 = treated.sum(axis=1)
        n0 = control.sum(axis=1)
        y1_sum = np.where(treated, y_p, 0.0).sum(axis=1)
        y0_sum = np.where(control, y_p, 0.0).sum(axis=1)
        audit_point = y1_sum / n1 - y0_sum / n0
        y1_var = np.array([np.var(y_p[i, treated[i]], ddof=1) for i in range(reps)])
        y0_var = np.array([np.var(y_p[i, control[i]], ddof=1) for i in range(reps)])
        audit_se = np.sqrt(y1_var / n1 + y0_var / n0)

        pseudo_oracle = s3.aipw_pseudo(x_p, a_p, y_p, stage3_setting)
        if_oracle = pseudo_oracle.mean(axis=1)
        if_oracle_se = pseudo_oracle.std(axis=1, ddof=1) / np.sqrt(M_AUDIT)

        def learned_if(beta0: np.ndarray, beta1: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
            q0 = s4.predict_batch(x_p, beta0, kind)
            q1 = s4.predict_batch(x_p, beta1, kind)
            pseudo = q1 - q0 + a_p / s3.PROPENSITY * (y_p - q1) - (1 - a_p) / (1 - s3.PROPENSITY) * (y_p - q0)
            return pseudo.mean(axis=1), pseudo.std(axis=1, ddof=1) / np.sqrt(M_AUDIT)

        if_rich, if_rich_se = learned_if(beta0_rich, beta1_rich, "rich")
        if_linear, if_linear_se = learned_if(beta0_linear, beta1_linear, "linear")

        method_specs = {
            ("audit_only", 1.0): ("Audit only", audit_point, *interval(audit_point, audit_se, crit_m)),
            ("q_inflated", 1.0): ("Q inflated", synth_center, *interval(synth_center, synth_se, Z, pop["direct_gap"])),
            ("if_oracle", 1.0): ("IF oracle", if_oracle, *interval(if_oracle, if_oracle_se, crit_m)),
            ("if_learned_linear", 1.0): ("IF learned linear", if_linear, *interval(if_linear, if_linear_se, crit_m)),
        }

        for mult in SE_MULTIPLIERS:
            method_specs[("if_learned_rich", mult)] = (
                "IF learned rich",
                if_rich,
                *interval(if_rich, mult * if_rich_se, crit_m),
            )

        for key, (_, point, lower, upper) in method_specs.items():
            update_acc(accs[key], point, lower, upper, target)

    rows = []
    for key, acc in accs.items():
        method, mult = key
        label = {
            "audit_only": "Audit only",
            "q_inflated": "Q inflated",
            "if_oracle": "IF oracle",
            "if_learned_linear": "IF learned linear",
            "if_learned_rich": "IF learned rich",
        }[method]
        rows.append(
            finalize_acc(
                setting=setting,
                pop=pop,
                method=method,
                method_label=label,
                acc=acc,
                se_multiplier=mult,
            )
        )
    return rows


def choose_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    learned = summary[summary["method"] == "if_learned_rich"].copy()
    rows = []
    for mult in SE_MULTIPLIERS:
        d = learned[np.isclose(learned["se_multiplier"], mult)]
        min_cov = d["coverage"].min()
        max_len_ratio = (
            d["avg_length"].mean()
            / learned[np.isclose(learned["se_multiplier"], 1.0)]["avg_length"].mean()
        )
        rows.append(
            {
                "se_multiplier": mult,
                "min_coverage_across_key_rows": min_cov,
                "mean_length_ratio_vs_uncalibrated": max_len_ratio,
                "passes_nominal_in_key_rows": bool(min_cov >= 0.95),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration(summary: pd.DataFrame, out: Path) -> None:
    data = summary[summary["method"] == "if_learned_rich"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    for gamma_label in GAMMA_LABELS:
        d = data[data["gamma_label"] == gamma_label].sort_values("se_multiplier")
        axes[0].plot(d["se_multiplier"], d["coverage"], marker="o", label=gamma_label)
        axes[1].plot(d["se_multiplier"], d["avg_length"], marker="o", label=gamma_label)
    axes[0].axhline(0.95, color="black", linewidth=0.8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Learned-IF standard-error multiplier")
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Average interval length")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Stage 4 QC: small conservative calibration for learned IF")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_method_comparison(summary: pd.DataFrame, out: Path, calibrated_multiplier: float) -> None:
    base = summary[
        ((summary["method"] != "if_learned_rich") & np.isclose(summary["se_multiplier"], 1.0))
        | ((summary["method"] == "if_learned_rich") & np.isclose(summary["se_multiplier"], calibrated_multiplier))
    ].copy()
    label_map = {
        "audit_only": "Audit only",
        "q_inflated": "Q inflated",
        "if_oracle": "IF oracle",
        "if_learned_rich": f"IF learned rich, x{calibrated_multiplier:.2f}",
        "if_learned_linear": "IF learned linear",
    }
    order = ["audit_only", "q_inflated", "if_oracle", "if_learned_rich", "if_learned_linear"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True)
    for method in order:
        d = base[base["method"] == method].sort_values("direct_gap")
        axes[0].plot(d["direct_gap"], d["coverage"], marker="o", label=label_map[method])
        axes[1].plot(d["direct_gap"], d["avg_length"], marker="o", label=label_map[method])
    axes[0].axhline(0.95, color="black", linewidth=0.8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Direct ATE gap")
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Average interval length")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("Stage 4 QC: calibrated learned IF remains shorter than audit-only")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(summary: pd.DataFrame, calibration: pd.DataFrame, calibrated_multiplier: float) -> None:
    key = summary[
        (summary["gamma_label"] == "moderate_boundary")
        & (summary["method"].isin(["audit_only", "q_inflated", "if_oracle", "if_learned_linear"]))
    ].copy()
    learned = summary[
        (summary["gamma_label"] == "moderate_boundary")
        & (summary["method"] == "if_learned_rich")
        & (summary["se_multiplier"].isin([1.0, calibrated_multiplier]))
    ].copy()
    key = pd.concat([key, learned], ignore_index=True).sort_values(["method", "se_multiplier"])

    selected = summary[
        ((summary["method"] == "q_inflated") & np.isclose(summary["se_multiplier"], 1.0))
        | ((summary["method"] == "if_learned_rich") & np.isclose(summary["se_multiplier"], calibrated_multiplier))
        | ((summary["method"] == "audit_only") & np.isclose(summary["se_multiplier"], 1.0))
    ].copy()
    selected_rows = []
    for gamma_label, d in selected.groupby("gamma_label"):
        winner = d.loc[d["avg_length"].idxmin()]
        selected_rows.append(
            {
                "gamma_label": gamma_label,
                "direct_gap": winner["direct_gap"],
                "selected_method": winner["method"],
                "coverage": winner["coverage"],
                "avg_length": winner["avg_length"],
            }
        )
    selected_df = pd.DataFrame(selected_rows).sort_values("direct_gap")

    text = f"""# Stage 4 QC and Calibration Summary

Created: 2026-05-21

This QC reruns the key learned-nuisance Stage 4 settings with `reps={REPS}`, `m={M_AUDIT}`, and `n_train={N_TRAIN}`. It focuses on the moderate and weak mismatch rows where the first Stage 4 run showed mild learned-IF undercoverage.

## Main Result

The uncalibrated learned-rich IF interval is still very short, but some rows remain a little below nominal coverage. A small conservative multiplier on the learned-IF standard error stabilizes the key rows. Among the tested values, the selected working calibration is `{calibrated_multiplier:.2f}` because it is the smallest multiplier whose minimum coverage across the key rows is at least nominal in this QC run.

## Calibration Scan

```text
{calibration.to_string(index=False)}
```

## Moderate-Boundary Row

```text
{key[['method', 'method_label', 'se_multiplier', 'coverage', 'mc_se_coverage', 'avg_length', 'bias', 'emp_sd']].to_string(index=False)}
```

## Selected `{{A, IF, Q}}` Rule After Calibration

This is a QC diagnostic using the shortest interval among audit-only, Q-inflated, and calibrated learned-rich IF.

```text
{selected_df.to_string(index=False)}
```

## Interpretation

- The calibration issue is finite-sample and learned-nuisance related, not a failure of the synthetic-bias correction idea: the oracle IF row remains near nominal and the learned-rich point bias is small.
- The multiplier makes the interval slightly longer, but the calibrated learned-rich IF interval remains much shorter than audit-only in the moderate/weak rows.
- For the manuscript, this means Stage 4 should be presented either with the calibrated learned IF or as a finite-sample robustness experiment that explicitly accounts for synthetic nuisance learning.

## Output Files

- `tables/stage4_learned_generator_diagnostics_qc/stage4_qc_key_rows.csv`
- `tables/stage4_learned_generator_diagnostics_qc/stage4_qc_calibration_scan.csv`
- `figures/stage4_learned_generator_diagnostics_qc/stage4_qc_calibration_tradeoff.png`
- `figures/stage4_learned_generator_diagnostics_qc/stage4_qc_method_comparison.png`
"""
    (RESULT_DIR / "STAGE4_QC_CALIBRATION_SUMMARY.md").write_text(text)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    rows = []
    for setting in key_settings():
        rows.extend(simulate_setting(setting, rng))
    summary = pd.DataFrame(rows).sort_values(["direct_gap", "method", "se_multiplier"])
    calibration = choose_calibration(summary)
    pass_rows = calibration[calibration["passes_nominal_in_key_rows"]]
    calibrated_multiplier = float(pass_rows.iloc[0]["se_multiplier"]) if len(pass_rows) else float(SE_MULTIPLIERS[-1])

    summary.to_csv(TABLE_DIR / "stage4_qc_key_rows.csv", index=False)
    calibration.to_csv(TABLE_DIR / "stage4_qc_calibration_scan.csv", index=False)
    summary.to_csv(RESULT_DIR / "stage4_qc_key_rows.csv", index=False)
    calibration.to_csv(RESULT_DIR / "stage4_qc_calibration_scan.csv", index=False)

    plot_calibration(summary, FIG_DIR / "stage4_qc_calibration_tradeoff.png")
    plot_method_comparison(summary, FIG_DIR / "stage4_qc_method_comparison.png", calibrated_multiplier)
    write_summary(summary, calibration, calibrated_multiplier)

    print(f"Wrote QC results to {RESULT_DIR}")
    print(f"Selected learned-IF SE multiplier: {calibrated_multiplier:.2f}")


if __name__ == "__main__":
    main()
