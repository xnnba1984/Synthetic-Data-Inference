#!/usr/bin/env python3
"""Feasible selected-branch repair simulation.

This script addresses the Review 2 concern that the existing selected-branch
rows are partly oracle/benchmark procedures. It evaluates a feasible selected
rule in the Stage 4 learned-nuisance ATE design.

The rule uses only:
- observed interval lengths;
- sample-estimated standard errors;
- a pre-specified sensitivity radius provided by the analyst;
- Bonferroni candidate intervals so post-selection reporting is covered by a
  simultaneous-coverage argument.

The true direct gap is used only for evaluation and plotting diagnostics, not
for selection.
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
RESULT_DIR = ROOT / "simulation" / "results" / "selection_repair_feasible"
FIG_DIR = ROOT / "figures" / "selection_repair_feasible"
TABLE_DIR = ROOT / "tables" / "selection_repair_feasible"

ALPHA = 0.05
REPS = 3000
CHUNK = 250
SEED = 20260522
M_AUDIT = 200
N_TRAIN = 2000
N_ARM = N_TRAIN // 2
IF_SE_MULTIPLIER = 1.03

GAMMA_LABELS = ("zero", "strong_boundary", "moderate_mid", "moderate_boundary", "weak")
SENSITIVITY_RADII = (0.00, 0.10, 0.20, 0.40, 0.60)
CANDIDATE_BRANCHES = ("A", "Q", "IF")


@dataclass(frozen=True)
class SelectionSetting:
    gamma_label: str
    gamma: float

    @property
    def stage3_setting(self) -> s3.Setting:
        hard = next(v for v in s3.VARIANTS if v.name == "hard_nuisance")
        return s3.Setting(
            variant=hard,
            m=M_AUDIT,
            M=N_TRAIN,
            M_multiplier=N_TRAIN // M_AUDIT,
            gamma=self.gamma,
            gamma_label=self.gamma_label,
        )


def ensure_dirs() -> None:
    for path in (RESULT_DIR, FIG_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def settings() -> list[SelectionSetting]:
    gamma_lookup = dict(s3.mismatch_grid(M_AUDIT))
    return [SelectionSetting(label, gamma_lookup[label]) for label in GAMMA_LABELS]


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
    setting: SelectionSetting,
    pop: dict[str, float],
    sensitivity_radius: float,
    method: str,
    method_label: str,
    acc: dict[str, float],
) -> dict[str, float | str | int | bool]:
    n = int(acc["n"])
    point_mean = acc["point_sum"] / n
    point_var = max(acc["point2_sum"] / n - point_mean**2, 0.0)
    return {
        "simulation": "selection_repair_feasible",
        "m": M_AUDIT,
        "n_train": N_TRAIN,
        "reps": n,
        "gamma_label": setting.gamma_label,
        "gamma": setting.gamma,
        "direct_gap": pop["direct_gap"],
        "sensitivity_radius": sensitivity_radius,
        "radius_covers_direct_gap": bool(sensitivity_radius >= pop["direct_gap"]),
        "method": method,
        "method_label": method_label,
        "coverage": acc["cover"] / n,
        "mc_se_coverage": float(np.sqrt((acc["cover"] / n) * (1 - acc["cover"] / n) / n)),
        "avg_length": acc["length_sum"] / n,
        "sd_length": float(np.sqrt(max(acc["length2_sum"] / n - (acc["length_sum"] / n) ** 2, 0.0))),
        "bias": acc["bias_sum"] / n,
        "emp_sd": float(np.sqrt(point_var)),
        "target": pop["theta_p"],
    }


def simulate_setting(setting: SelectionSetting, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    st = setting.stage3_setting
    pop = s3.population_quantities(st)
    target = float(pop["theta_p"])

    # Bonferroni candidate intervals: simultaneous coverage for A, Q, IF.
    candidate_alpha = ALPHA / len(CANDIDATE_BRANCHES)
    z_bonf = norm.ppf(1 - candidate_alpha / 2)
    t_bonf = t.ppf(1 - candidate_alpha / 2, df=M_AUDIT - 2)

    accs: dict[tuple[float, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    branch_counts: dict[tuple[float, str], int] = defaultdict(int)

    remaining = REPS
    while remaining > 0:
        reps = min(CHUNK, remaining)
        remaining -= reps

        # Independent synthetic training sample from Q.
        x0_q = s3.draw_x(rng, (reps, N_ARM))
        x1_q = s3.draw_x(rng, (reps, N_ARM))
        y0_q = s3.m0_q(x0_q, st) + rng.normal(scale=s3.SIGMA, size=(reps, N_ARM))
        y1_q = s3.m1_q(x1_q, st) + rng.normal(scale=s3.SIGMA, size=(reps, N_ARM))
        synth_center = y1_q.mean(axis=1) - y0_q.mean(axis=1)
        synth_se = np.sqrt(y1_q.var(axis=1, ddof=1) / N_ARM + y0_q.var(axis=1, ddof=1) / N_ARM)

        beta0_rich = s4.fit_ridge_batch(x0_q, y0_q, "rich")
        beta1_rich = s4.fit_ridge_batch(x1_q, y1_q, "rich")

        # Independent audit sample from P.
        x_p = s3.draw_x(rng, (reps, M_AUDIT))
        a_p = rng.binomial(1, s3.PROPENSITY, size=(reps, M_AUDIT))
        y_p = s3.draw_outcomes_p(rng, x_p, a_p, st)

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

        q0 = s4.predict_batch(x_p, beta0_rich, "rich")
        q1 = s4.predict_batch(x_p, beta1_rich, "rich")
        pseudo = q1 - q0 + a_p / s3.PROPENSITY * (y_p - q1) - (1 - a_p) / (1 - s3.PROPENSITY) * (y_p - q0)
        if_point = pseudo.mean(axis=1)
        if_se = pseudo.std(axis=1, ddof=1) / np.sqrt(M_AUDIT)

        audit_lo, audit_hi = interval(audit_point, audit_se, t_bonf)
        if_lo, if_hi = interval(if_point, IF_SE_MULTIPLIER * if_se, t_bonf)

        for radius in SENSITIVITY_RADII:
            q_lo, q_hi = interval(synth_center, synth_se, z_bonf, radius)
            candidate = {
                "A": (audit_point, audit_lo, audit_hi),
                "Q": (synth_center, q_lo, q_hi),
                "IF": (if_point, if_lo, if_hi),
            }
            lengths = np.vstack([candidate[b][2] - candidate[b][1] for b in CANDIDATE_BRANCHES]).T
            selected_index = np.argmin(lengths, axis=1)
            selected_branch = np.array(CANDIDATE_BRANCHES, dtype=object)[selected_index]

            selected_point = np.empty(reps)
            selected_lo = np.empty(reps)
            selected_hi = np.empty(reps)
            for branch in CANDIDATE_BRANCHES:
                mask = selected_branch == branch
                branch_counts[(radius, branch)] += int(mask.sum())
                selected_point[mask] = candidate[branch][0][mask]
                selected_lo[mask] = candidate[branch][1][mask]
                selected_hi[mask] = candidate[branch][2][mask]

            for branch in CANDIDATE_BRANCHES:
                update_acc(accs[(radius, branch)], *candidate[branch], target)
            update_acc(accs[(radius, "selected_feasible")], selected_point, selected_lo, selected_hi, target)

    rows = []
    method_labels = {
        "A": "Audit only, Bonferroni",
        "Q": "Q centered with provided bound, Bonferroni",
        "IF": f"Learned IF x{IF_SE_MULTIPLIER:.2f}, Bonferroni",
        "selected_feasible": "Feasible selected, Bonferroni",
    }
    for (radius, method), acc in sorted(accs.items()):
        rows.append(
            finalize_acc(
                setting=setting,
                pop=pop,
                sensitivity_radius=radius,
                method=method,
                method_label=method_labels[method],
                acc=acc,
            )
        )

    branch_rows = []
    for radius in SENSITIVITY_RADII:
        total = sum(branch_counts[(radius, branch)] for branch in CANDIDATE_BRANCHES)
        for branch in CANDIDATE_BRANCHES:
            branch_rows.append(
                {
                    "simulation": "selection_repair_feasible",
                    "m": M_AUDIT,
                    "n_train": N_TRAIN,
                    "gamma_label": setting.gamma_label,
                    "direct_gap": pop["direct_gap"],
                    "sensitivity_radius": radius,
                    "radius_covers_direct_gap": bool(radius >= pop["direct_gap"]),
                    "selected_branch": branch,
                    "selected_count": branch_counts[(radius, branch)],
                    "selected_frequency": branch_counts[(radius, branch)] / total,
                }
            )
    return rows, branch_rows


def run_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    branch_rows = []
    for setting in settings():
        setting_rows, setting_branch = simulate_setting(setting, rng)
        rows.extend(setting_rows)
        branch_rows.extend(setting_branch)
    return pd.DataFrame(rows), pd.DataFrame(branch_rows)


def plot_coverage_length(summary: pd.DataFrame, out: Path) -> None:
    selected = summary[summary["method"] == "selected_feasible"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for gamma_label, d in selected.groupby("gamma_label"):
        d = d.sort_values("sensitivity_radius")
        axes[0].plot(d["sensitivity_radius"], d["coverage"], marker="o", label=gamma_label)
        axes[1].plot(d["sensitivity_radius"], d["avg_length"], marker="o", label=gamma_label)
    axes[0].axhline(0.95, color="black", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Provided sensitivity bound for Q centered branch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Selected interval coverage")
    axes[1].set_ylabel("Selected interval average length")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("Feasible selected rule with provided sensitivity bound")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_branch_frequencies(branch: pd.DataFrame, out: Path) -> None:
    labels = list(GAMMA_LABELS)
    radii = list(SENSITIVITY_RADII)
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.5, 10), sharex=True)
    colors = {"A": "#1f77b4", "Q": "#d62728", "IF": "#2ca02c"}
    for ax, gamma_label in zip(axes, labels):
        d = branch[branch["gamma_label"] == gamma_label]
        bottom = np.zeros(len(radii))
        for b in CANDIDATE_BRANCHES:
            vals = []
            for radius in radii:
                row = d[(d["sensitivity_radius"] == radius) & (d["selected_branch"] == b)]
                vals.append(float(row.iloc[0]["selected_frequency"]))
            ax.bar(radii, vals, bottom=bottom, width=0.055, color=colors[b], label=b if ax is axes[0] else None)
            bottom += np.array(vals)
        gap = float(d["direct_gap"].iloc[0])
        ax.axvline(gap, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(gamma_label)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2, axis="y")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Provided sensitivity bound. Dashed line is true direct gap for evaluation")
    fig.suptitle("Feasible selected rule: branch frequencies")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(summary: pd.DataFrame, branch: pd.DataFrame) -> None:
    selected = summary[summary["method"] == "selected_feasible"].copy()
    valid = selected[selected["radius_covers_direct_gap"]]
    invalid = selected[~selected["radius_covers_direct_gap"]]
    selected_key = selected[
        selected["gamma_label"].isin(["strong_boundary", "moderate_boundary", "weak"])
        & selected["sensitivity_radius"].isin([0.10, 0.40, 0.60])
    ][
        [
            "gamma_label",
            "direct_gap",
            "sensitivity_radius",
            "radius_covers_direct_gap",
            "coverage",
            "mc_se_coverage",
            "avg_length",
        ]
    ]
    branch_wide = (
        branch.pivot_table(
            index=["gamma_label", "direct_gap", "sensitivity_radius", "radius_covers_direct_gap"],
            columns="selected_branch",
            values="selected_frequency",
            fill_value=0.0,
        )
        .reset_index()
        .sort_values(["direct_gap", "sensitivity_radius"])
    )

    text = f"""# Feasible Selected-Branch Repair Summary

Created: 2026-05-21

This repair addresses the Review 2 concern that earlier selected-branch rows were oracle or benchmark selections. The feasible rule here uses only observed interval lengths, sample standard errors, Bonferroni candidate intervals, and a pre-specified sensitivity radius for the Q-centered branch. The true direct gap is used only to evaluate whether the provided radius was large enough.

## Design

- Data-generating model: hard-nuisance Stage 4 ATE setting.
- Audit size: `m={M_AUDIT}`.
- Synthetic training size: `n_train={N_TRAIN}`.
- Monte Carlo replicates per setting: `{REPS}`.
- Candidate branches: audit-only, Q-centered with provided sensitivity radius, and learned-rich IF with fixed `x{IF_SE_MULTIPLIER:.2f}` standard-error multiplier.
- Candidate intervals use Bonferroni critical values for three branches before selecting the shortest realized interval.
- Sensitivity radii are the fixed grid `{SENSITIVITY_RADII}`; they are not estimated from the true target gap.

## Main Result

When the provided sensitivity radius covers the true direct gap, feasible selected coverage ranges from `{valid.coverage.min():.3f}` to `{valid.coverage.max():.3f}`. When the provided radius is too small, coverage can drop as low as `{invalid.coverage.min():.3f}`. This is expected and useful because the feasible selected rule is only valid under an honest sensitivity radius.

This repair removes the most important oracle-selection criticism. The selected interval is now chosen by an implementable rule. The remaining assumption is the paper's intended one: the user must supply, certify, or sensitivity-scan a radius large enough for the Q branch.

## Selected Key Rows

```text
{selected_key.to_string(index=False)}
```

## Branch Frequencies

```text
{branch_wide.to_string(index=False)}
```

## Interpretation

- This should become the manuscript-facing selected-rule simulation, replacing broad claims based on oracle/benchmark selected rows.
- Earlier selected rows remain useful as oracle benchmarks, but should be labeled that way.
- The figure also makes sensitivity-radius honesty visible: undercoverage appears when the provided radius is too small, while valid bounds recover coverage and often select IF or audit-only instead of overtrusting Q.

## Output Files

- `tables/selection_repair_feasible/selection_repair_summary.csv`
- `tables/selection_repair_feasible/selection_repair_branch_frequencies.csv`
- `figures/selection_repair_feasible/feasible_selection_coverage_length.png`
- `figures/selection_repair_feasible/feasible_selection_branch_frequencies.png`
"""
    (RESULT_DIR / "SELECTION_REPAIR_FEASIBLE_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    summary, branch = run_all()
    summary.to_csv(TABLE_DIR / "selection_repair_summary.csv", index=False)
    branch.to_csv(TABLE_DIR / "selection_repair_branch_frequencies.csv", index=False)
    summary.to_csv(RESULT_DIR / "selection_repair_summary.csv", index=False)
    branch.to_csv(RESULT_DIR / "selection_repair_branch_frequencies.csv", index=False)
    plot_coverage_length(summary, FIG_DIR / "feasible_selection_coverage_length.png")
    plot_branch_frequencies(branch, FIG_DIR / "feasible_selection_branch_frequencies.png")
    write_summary(summary, branch)
    print(f"Wrote feasible selected-rule repair results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
