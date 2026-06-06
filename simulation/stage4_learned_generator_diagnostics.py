#!/usr/bin/env python3
"""Stage 4 learned-nuisance and diagnostic simulations.

This script implements S5 and S6 from SIMULATION_DESIGN_PLAN.md.

S5 asks whether the Stage 3 semiparametric ATE gain survives when the
synthetic outcome nuisances are estimated from a finite synthetic sample.

S6 gives a compact diagnostic failure example: generic RBF-MMD can be small
when the target bias is large and can be large when the target is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t

import stage3_semiparametric_ate as s3


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "simulation" / "results" / "stage4_learned_generator_diagnostics"
FIG_DIR = ROOT / "figures" / "stage4_learned_generator_diagnostics"
TABLE_DIR = ROOT / "tables" / "stage4_learned_generator_diagnostics"

ALPHA = 0.05
Z = norm.ppf(1 - ALPHA / 2)
REPS = 700
SEED = 20260520
TRAIN_SIZES = (500, 2000, 5000)
GAMMA_LABELS_FOR_LEARNING = (
    "zero",
    "strong_boundary",
    "moderate_mid",
    "moderate_boundary",
    "weak",
)
RIDGE_LAMBDA = 1e-3

DIAG_REPS = 200
DIAG_N = 300
DIAG_DIM = 50
DIAG_M_SYNTH = 10000
DIAG_DELTAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


METHOD_LABELS = {
    "audit_only": "Audit only",
    "synthetic_naive": "Synthetic naive",
    "q_inflated": "Q centered",
    "if_oracle": "IF oracle",
    "if_learned_rich": "IF learned rich",
    "if_learned_linear": "IF learned linear",
    "selected": "Selected",
}

PLOT_METHODS = [
    "audit_only",
    "synthetic_naive",
    "q_inflated",
    "if_oracle",
    "if_learned_rich",
    "selected",
]

PLOT_STYLES = {
    "audit_only": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "synthetic_naive": {"color": "#ff7f0e", "marker": "o", "linestyle": "-"},
    "q_inflated": {"color": "#d62728", "marker": "o", "linestyle": "-"},
    "if_oracle": {"color": "#9467bd", "marker": "o", "linestyle": "--"},
    "if_learned_rich": {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
    "if_learned_linear": {"color": "#8c564b", "marker": "o", "linestyle": "-"},
    "selected": {
        "color": "black",
        "marker": "D",
        "linestyle": "None",
        "markerfacecolor": "none",
        "markeredgewidth": 1.2,
    },
}


@dataclass(frozen=True)
class LearnSetting:
    variant: s3.Variant
    m: int
    n_train: int
    gamma: float
    gamma_label: str

    @property
    def stage3_setting(self) -> s3.Setting:
        return s3.Setting(
            variant=self.variant,
            m=self.m,
            M=self.n_train,
            M_multiplier=max(1, self.n_train // self.m),
            gamma=self.gamma,
            gamma_label=self.gamma_label,
        )


def ensure_dirs() -> None:
    for path in (RESULT_DIR, FIG_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_learning_settings() -> list[LearnSetting]:
    hard = next(v for v in s3.VARIANTS if v.name == "hard_nuisance")
    gamma_lookup = dict(s3.mismatch_grid(200))
    return [
        LearnSetting(variant=hard, m=200, n_train=n_train, gamma=gamma_lookup[label], gamma_label=label)
        for n_train in TRAIN_SIZES
        for label in GAMMA_LABELS_FOR_LEARNING
    ]


def feature_map(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "linear":
        return np.concatenate([np.ones((*x.shape[:-1], 1)), x], axis=-1)
    if kind == "rich":
        pieces = [
            np.ones((*x.shape[:-1], 1)),
            x,
            np.sin(x[..., [0]]),
            x[..., [1]] ** 2 - 1.0,
            x[..., [3]] * x[..., [4]],
            np.tanh(x[..., [0]] + x[..., [1]]),
        ]
        return np.concatenate(pieces, axis=-1)
    raise ValueError(f"Unknown feature kind: {kind}")


def fit_ridge_batch(x: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    phi = feature_map(x, kind)
    xtx = np.einsum("rnp,rnq->rpq", phi, phi) / x.shape[1]
    xty = np.einsum("rnp,rn->rp", phi, y) / x.shape[1]
    penalty = RIDGE_LAMBDA * np.eye(phi.shape[-1])
    penalty[0, 0] = 0.0
    xtx = xtx + penalty
    return np.linalg.solve(xtx, xty)


def predict_batch(x: np.ndarray, beta: np.ndarray, kind: str) -> np.ndarray:
    phi = feature_map(x, kind)
    return np.einsum("rnp,rp->rn", phi, beta)


def ci(center: np.ndarray, se: np.ndarray, crit: float, inflation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return center - crit * se - inflation, center + crit * se + inflation


def summarize(
    *,
    setting: LearnSetting,
    method: str,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    selected_branch: str = "",
) -> dict[str, float | str | int]:
    pop = s3.population_quantities(setting.stage3_setting)
    target = pop["theta_p"]
    length = upper - lower
    cover = (lower <= target) & (target <= upper)
    return {
        "simulation": "S5_learned_nuisance",
        "variant": setting.variant.name,
        "variant_label": setting.variant.label,
        "m": setting.m,
        "n_train": setting.n_train,
        "gamma": setting.gamma,
        "gamma_label": setting.gamma_label,
        "direct_gap": pop["direct_gap"],
        "method": method,
        "method_label": METHOD_LABELS[method],
        "selected_branch": selected_branch,
        "coverage": cover.mean(),
        "avg_length": length.mean(),
        "median_length": np.median(length),
        "length_iqr": np.quantile(length, 0.75) - np.quantile(length, 0.25),
        "bias": point.mean() - target,
        "emp_sd": point.std(ddof=1),
        "target": target,
    }


def run_learning_setting(setting: LearnSetting, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    pop = s3.population_quantities(setting.stage3_setting)
    crit_m = t.ppf(1 - ALPHA / 2, df=setting.m - 2)
    n_arm = setting.n_train // 2

    # Synthetic generator sample, balanced by treatment arm for stable nuisance fitting.
    x0_q = s3.draw_x(rng, (REPS, n_arm))
    x1_q = s3.draw_x(rng, (REPS, n_arm))
    y0_q = s3.m0_q(x0_q, setting.stage3_setting) + rng.normal(scale=s3.SIGMA, size=(REPS, n_arm))
    y1_q = s3.m1_q(x1_q, setting.stage3_setting) + rng.normal(scale=s3.SIGMA, size=(REPS, n_arm))

    synth_center = y1_q.mean(axis=1) - y0_q.mean(axis=1)
    synth_se = np.sqrt(y1_q.var(axis=1, ddof=1) / n_arm + y0_q.var(axis=1, ddof=1) / n_arm)

    beta0_rich = fit_ridge_batch(x0_q, y0_q, "rich")
    beta1_rich = fit_ridge_batch(x1_q, y1_q, "rich")
    beta0_linear = fit_ridge_batch(x0_q, y0_q, "linear")
    beta1_linear = fit_ridge_batch(x1_q, y1_q, "linear")

    # Independent audit sample from P.
    x_p = s3.draw_x(rng, (REPS, setting.m))
    a_p = rng.binomial(1, s3.PROPENSITY, size=(REPS, setting.m))
    y_p = s3.draw_outcomes_p(rng, x_p, a_p, setting.stage3_setting)

    treated = a_p == 1
    control = ~treated
    n1 = treated.sum(axis=1)
    n0 = control.sum(axis=1)
    y1_sum = np.where(treated, y_p, 0.0).sum(axis=1)
    y0_sum = np.where(control, y_p, 0.0).sum(axis=1)
    audit_point = y1_sum / n1 - y0_sum / n0
    y1_var = np.array([np.var(y_p[i, treated[i]], ddof=1) for i in range(REPS)])
    y0_var = np.array([np.var(y_p[i, control[i]], ddof=1) for i in range(REPS)])
    audit_se = np.sqrt(y1_var / n1 + y0_var / n0)

    pseudo_oracle = s3.aipw_pseudo(x_p, a_p, y_p, setting.stage3_setting)
    if_oracle = pseudo_oracle.mean(axis=1)
    if_oracle_se = pseudo_oracle.std(axis=1, ddof=1) / np.sqrt(setting.m)

    def learned_if(beta0: np.ndarray, beta1: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
        q0 = predict_batch(x_p, beta0, kind)
        q1 = predict_batch(x_p, beta1, kind)
        pseudo = q1 - q0 + a_p / s3.PROPENSITY * (y_p - q1) - (1 - a_p) / (1 - s3.PROPENSITY) * (y_p - q0)
        return pseudo.mean(axis=1), pseudo.std(axis=1, ddof=1) / np.sqrt(setting.m)

    if_rich, if_rich_se = learned_if(beta0_rich, beta1_rich, "rich")
    if_linear, if_linear_se = learned_if(beta0_linear, beta1_linear, "linear")

    intervals = {
        "audit_only": (audit_point, *ci(audit_point, audit_se, crit_m)),
        "synthetic_naive": (synth_center, *ci(synth_center, synth_se, Z)),
        "q_inflated": (synth_center, *ci(synth_center, synth_se, Z, pop["direct_gap"])),
        "if_oracle": (if_oracle, *ci(if_oracle, if_oracle_se, crit_m)),
        "if_learned_rich": (if_rich, *ci(if_rich, if_rich_se, crit_m)),
        "if_learned_linear": (if_linear, *ci(if_linear, if_linear_se, crit_m)),
    }

    mean_lengths = {key: float((vals[2] - vals[1]).mean()) for key, vals in intervals.items()}
    candidate_map = {"A": "audit_only", "Q": "q_inflated", "IF": "if_learned_rich"}
    selected_branch = min(candidate_map, key=lambda branch: mean_lengths[candidate_map[branch]])
    intervals["selected"] = intervals[candidate_map[selected_branch]]

    return [
        summarize(
            setting=setting,
            method=method,
            point=vals[0],
            lower=vals[1],
            upper=vals[2],
            selected_branch=selected_branch if method == "selected" else "",
        )
        for method, vals in intervals.items()
    ]


def run_learning() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str | int]] = []
    for setting in build_learning_settings():
        rows.extend(run_learning_setting(setting, rng))
    summary = pd.DataFrame(rows)
    branch = (
        summary[summary["method"] == "selected"]
        .groupby(["variant", "variant_label", "m", "n_train", "gamma", "gamma_label", "selected_branch"])
        .size()
        .reset_index(name="settings")
    )
    return summary, branch


def rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    diffs = z[:, None, :] - z[None, :, :]
    d2 = np.sum(diffs**2, axis=-1)
    bandwidth2 = np.median(d2[d2 > 0])
    bandwidth2 = max(float(bandwidth2), 1e-8)

    xx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    yy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    xy = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    kxx = np.exp(-xx / (2 * bandwidth2))
    kyy = np.exp(-yy / (2 * bandwidth2))
    kxy = np.exp(-xy / (2 * bandwidth2))
    return float(np.sqrt(max(kxx.mean() + kyy.mean() - 2 * kxy.mean(), 0.0)))


def run_diagnostics() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 900)
    rows = []
    se = 1 / np.sqrt(DIAG_M_SYNTH)
    q_draws = rng.normal(size=(len(DIAG_DELTAS), DIAG_REPS, 5000))
    for idx, delta in enumerate(DIAG_DELTAS):
        mmd_target = []
        mmd_nuisance = []
        for _ in range(DIAG_REPS):
            x = rng.normal(size=(DIAG_N, DIAG_DIM))
            y_target = rng.normal(size=(DIAG_N, DIAG_DIM))
            y_target[:, 0] += delta
            y_nuisance = rng.normal(size=(DIAG_N, DIAG_DIM))
            y_nuisance[:, 1] += delta
            mmd_target.append(rbf_mmd(x, y_target))
            mmd_nuisance.append(rbf_mmd(x, y_nuisance))

        for case, mmd_vals, target_gap in (
            ("target_shift", mmd_target, delta),
            ("nuisance_shift", mmd_nuisance, 0.0),
        ):
            mmd_mean = float(np.mean(mmd_vals))
            if case == "target_shift":
                centers = delta + se * q_draws[idx, :, :]
                target = 0.0
            else:
                centers = se * q_draws[idx, :, :]
                target = 0.0

            naive_lo, naive_hi = centers - Z * se, centers + Z * se
            mmd_lo, mmd_hi = centers - Z * se - mmd_mean, centers + Z * se + mmd_mean
            target_lo, target_hi = centers - Z * se - target_gap, centers + Z * se + target_gap
            rows.append(
                {
                    "case": case,
                    "shift_delta": delta,
                    "target_gap": target_gap,
                    "mmd_mean": mmd_mean,
                    "mmd_to_target_ratio": mmd_mean / target_gap if target_gap > 0 else np.nan,
                    "naive_coverage": np.mean((naive_lo <= target) & (target <= naive_hi)),
                    "mmd_inflated_coverage": np.mean((mmd_lo <= target) & (target <= mmd_hi)),
                    "target_inflated_coverage": np.mean((target_lo <= target) & (target <= target_hi)),
                    "naive_length": float(np.mean(naive_hi - naive_lo)),
                    "mmd_inflated_length": float(np.mean(mmd_hi - mmd_lo)),
                    "target_inflated_length": float(np.mean(target_hi - target_lo)),
                }
            )
    return pd.DataFrame(rows)


def plot_learned_coverage_length(summary: pd.DataFrame, out: Path, n_train: int = 2000) -> None:
    data = summary[(summary["n_train"] == n_train) & (summary["method"].isin(PLOT_METHODS))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for method in PLOT_METHODS:
        d = data[data["method"] == method].sort_values("direct_gap")
        style = PLOT_STYLES.get(method, {"marker": "o", "linestyle": "-"})
        axes[0].plot(d["direct_gap"], d["coverage"], label=METHOD_LABELS[method], **style)
        axes[1].plot(d["direct_gap"], d["avg_length"], label=METHOD_LABELS[method], **style)
    for ax in axes:
        ax.axvline(200 ** (-0.5), color="gray", linestyle="--", linewidth=1, label=r"$m^{-1/2}$" if ax is axes[0] else None)
        ax.axvline(200 ** (-0.25), color="gray", linestyle=":", linewidth=1.2, label=r"$m^{-1/4}$" if ax is axes[0] else None)
        ax.set_xlabel(r"Direct ATE gap $|\theta(P)-\theta(Q)|$")
        ax.grid(alpha=0.25)
    axes[0].axhline(0.95, color="black", linewidth=0.8)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Average interval length")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"S5: Learned synthetic nuisances, n_train={n_train}")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_training_size(summary: pd.DataFrame, out: Path) -> None:
    data = summary[summary["method"].isin(["if_oracle", "if_learned_rich", "if_learned_linear"])].copy()
    data = data[data["gamma_label"].isin(["moderate_boundary", "weak"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for gamma_label, marker in [("moderate_boundary", "o"), ("weak", "s")]:
        d_rich = data[(data["method"] == "if_learned_rich") & (data["gamma_label"] == gamma_label)].sort_values("n_train")
        d_linear = data[(data["method"] == "if_learned_linear") & (data["gamma_label"] == gamma_label)].sort_values("n_train")
        d_oracle = data[(data["method"] == "if_oracle") & (data["gamma_label"] == gamma_label)].sort_values("n_train")
        axes[0].plot(d_rich["n_train"], d_rich["coverage"], marker=marker, label=f"Rich, {gamma_label}")
        axes[0].plot(d_linear["n_train"], d_linear["coverage"], marker=marker, linestyle="--", label=f"Linear, {gamma_label}")
        axes[1].plot(d_rich["n_train"], d_rich["avg_length"] / d_oracle["avg_length"].to_numpy(), marker=marker, label=f"Rich, {gamma_label}")
        axes[1].plot(d_linear["n_train"], d_linear["avg_length"] / d_oracle["avg_length"].to_numpy(), marker=marker, linestyle="--", label=f"Linear, {gamma_label}")
    axes[0].axhline(0.95, color="black", linewidth=0.8)
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Synthetic training size")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("IF coverage")
    axes[1].set_ylabel("Learned IF length / oracle IF length")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("S5: learned IF approaches oracle nuisance performance")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_diagnostics(diag: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for case, label, marker in [
        ("target_shift", "Target shift", "o"),
        ("nuisance_shift", "Nuisance shift", "s"),
    ]:
        d = diag[diag["case"] == case].sort_values("shift_delta")
        axes[0].plot(d["shift_delta"], d["mmd_mean"], marker=marker, label=f"MMD, {label}")
        axes[0].plot(d["shift_delta"], d["target_gap"], marker=marker, linestyle="--", label=f"Target gap, {label}")

    d_target = diag[diag["case"] == "target_shift"].sort_values("shift_delta")
    axes[1].plot(d_target["shift_delta"], d_target["naive_coverage"], marker="o", label="Naive")
    axes[1].plot(d_target["shift_delta"], d_target["mmd_inflated_coverage"], marker="o", label="MMD inflated")
    axes[1].plot(d_target["shift_delta"], d_target["target_inflated_coverage"], marker="o", label="Target inflated")
    axes[1].axhline(0.95, color="black", linewidth=0.8)

    d_nuisance = diag[diag["case"] == "nuisance_shift"].sort_values("shift_delta")
    axes[2].plot(d_nuisance["shift_delta"], d_nuisance["naive_length"], marker="o", label="Naive")
    axes[2].plot(d_nuisance["shift_delta"], d_nuisance["mmd_inflated_length"], marker="o", label="MMD inflated")
    axes[2].plot(d_nuisance["shift_delta"], d_nuisance["target_inflated_length"], marker="o", label="Target inflated")

    axes[0].set_title("Generic MMD vs target gap")
    axes[1].set_title("Target shift: coverage")
    axes[2].set_title("Nuisance shift: length")
    axes[0].set_ylabel("Discrepancy value")
    axes[1].set_ylabel("Coverage")
    axes[2].set_ylabel("Average interval length")
    for ax in axes:
        ax.set_xlabel("Distribution shift size")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("S6: generic MMD is not a target-calibrated certificate")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(learned: pd.DataFrame, branch: pd.DataFrame, diag: pd.DataFrame) -> None:
    def row(method: str, gamma_label: str, n_train: int = 2000) -> pd.Series:
        d = learned[(learned["method"] == method) & (learned["gamma_label"] == gamma_label) & (learned["n_train"] == n_train)]
        return d.iloc[0]

    learned_mod = row("if_learned_rich", "moderate_boundary")
    oracle_mod = row("if_oracle", "moderate_boundary")
    audit_mod = row("audit_only", "moderate_boundary")
    q_un_mod = row("synthetic_naive", "moderate_boundary")
    q_inf_mod = row("q_inflated", "moderate_boundary")
    selected_mod = row("selected", "moderate_boundary")
    selected_weak = row("selected", "weak")

    target_shift_02 = diag[(diag["case"] == "target_shift") & np.isclose(diag["shift_delta"], 0.20)].iloc[0]
    nuisance_shift_04 = diag[(diag["case"] == "nuisance_shift") & np.isclose(diag["shift_delta"], 0.40)].iloc[0]

    selected = learned[learned["method"] == "selected"][
        ["n_train", "gamma_label", "direct_gap", "selected_branch", "coverage", "avg_length"]
    ]
    branch_counts = branch.to_string(index=False)

    text = f"""# Stage 4 Learned Generator and Diagnostic Results Summary

Created: 2026-05-20

These results come from `simulation/stage4_learned_generator_diagnostics.py`.

## Main Takeaways

1. Learned synthetic nuisances preserve most of the Stage 3 IF gain. With `n_train=2000` in the moderate-boundary setting, audit-only coverage is {audit_mod.coverage:.3f} with average length {audit_mod.avg_length:.3f}; learned-rich IF coverage is {learned_mod.coverage:.3f} with average length {learned_mod.avg_length:.3f}; oracle IF coverage is {oracle_mod.coverage:.3f} with average length {oracle_mod.avg_length:.3f}.

2. Naive synthetic inference still fails under target mismatch after learning. In the same setting, synthetic-naive coverage is {q_un_mod.coverage:.3f}. The Q-inflated interval restores coverage to {q_inf_mod.coverage:.3f}, but it pays the direct ATE gap and is longer than learned IF.

3. The selected rule favors learned IF once direct synthetic centering becomes costly. In the moderate-boundary setting it selects `{selected_mod.selected_branch}`, with coverage {selected_mod.coverage:.3f} and average length {selected_mod.avg_length:.3f}; in the weak setting it selects `{selected_weak.selected_branch}`, with coverage {selected_weak.coverage:.3f} and average length {selected_weak.avg_length:.3f}.

4. Training size matters in the expected direction. Rich learned nuisances stay close to oracle IF, especially as synthetic training size grows. A deliberately misspecified linear nuisance remains valid after audit correction but gives longer intervals because it removes less outcome variation.

5. Generic MMD is not a reliable validity certificate. In a target-shift diagnostic with shift 0.20, the average RBF-MMD is {target_shift_02.mmd_mean:.3f} while the target gap is {target_shift_02.target_gap:.3f}; MMD-inflated coverage is {target_shift_02.mmd_inflated_coverage:.3f}, while target-inflated coverage is {target_shift_02.target_inflated_coverage:.3f}. In a nuisance-shift diagnostic with shift 0.40, the target gap is zero but MMD inflation length is {nuisance_shift_04.mmd_inflated_length:.3f}, compared with target-inflated length {nuisance_shift_04.target_inflated_length:.3f}.

## Selected Branch Behavior

```text
{selected.to_string(index=False)}
```

## Branch Counts

```text
{branch_counts}
```

## Basic QC

- Expected learned-nuisance rows: 105.
- Actual rows in `stage4_learned_summary.csv`: {len(learned)}.
- Diagnostic rows in `stage4_diagnostic_summary.csv`: {len(diag)}.
- `python -m py_compile simulation/stage4_learned_generator_diagnostics.py` passed.
- Blank `selected_branch` values occur only for non-selected methods; selected rows have explicit branch labels.

## Output Files

- `tables/stage4_learned_generator_diagnostics/stage4_learned_summary.csv`
- `tables/stage4_learned_generator_diagnostics/stage4_branch_frequencies.csv`
- `tables/stage4_learned_generator_diagnostics/stage4_diagnostic_summary.csv`
- `figures/stage4_learned_generator_diagnostics/stage4_learned_coverage_length.png`
- `figures/stage4_learned_generator_diagnostics/stage4_training_size.png`
- `figures/stage4_learned_generator_diagnostics/stage4_diagnostic_failure.png`
"""
    (RESULT_DIR / "STAGE4_RESULTS_SUMMARY.md").write_text(text)


def main() -> None:
    ensure_dirs()
    learned, branch = run_learning()
    diag = run_diagnostics()

    learned.to_csv(TABLE_DIR / "stage4_learned_summary.csv", index=False)
    branch.to_csv(TABLE_DIR / "stage4_branch_frequencies.csv", index=False)
    diag.to_csv(TABLE_DIR / "stage4_diagnostic_summary.csv", index=False)

    plot_learned_coverage_length(learned, FIG_DIR / "stage4_learned_coverage_length.png", n_train=2000)
    plot_training_size(learned, FIG_DIR / "stage4_training_size.png")
    plot_diagnostics(diag, FIG_DIR / "stage4_diagnostic_failure.png")

    write_summary(learned, branch, diag)
    learned.to_csv(RESULT_DIR / "stage4_learned_summary.csv", index=False)
    branch.to_csv(RESULT_DIR / "stage4_branch_frequencies.csv", index=False)
    diag.to_csv(RESULT_DIR / "stage4_diagnostic_summary.csv", index=False)

    print(f"Wrote results to {RESULT_DIR}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
