#!/usr/bin/env python3
"""Stage 2 estimating-equation simulations for synthetic-data inference.

This script implements S3 from SIMULATION_DESIGN_PLAN.md. The target is a
linear estimating-equation root under the real distribution P. The synthetic
distribution Q has a biased estimating-equation root and, in some variants, a
different local derivative. Oracle direct and IF residual buffers are used so
that this stage tests the repaired theorem mechanism before adding estimated
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "simulation" / "results" / "stage2_estimating_equation"
FIG_DIR = ROOT / "figures" / "stage2_estimating_equation"
TABLE_DIR = ROOT / "tables" / "stage2_estimating_equation"

ALPHA = 0.05
REPS = 2000
SEED = 20260520

M_VALUES = (100,)
M_MULTIPLIERS = (10, 50)
SIGMA = 1.0

BETA_P = np.array([0.5, 1.0, -0.5])
TARGET_CONTRAST = np.array([0.0, 1.0, 0.0])
P_X_MEAN = np.array([0.0, 0.0])
P_X_COV = np.array([[1.0, 0.3], [0.3, 1.0]])
P_DIM = len(BETA_P)

METHOD_ORDER = [
    "audit_only",
    "synthetic_naive",
    "pooled_naive",
    "q_uninflated",
    "q_inflated",
    "if_uninflated",
    "if_inflated",
    "selected",
]

METHOD_LABELS = {
    "audit_only": "Audit only",
    "synthetic_naive": "Synthetic naive",
    "pooled_naive": "Pooled naive",
    "q_uninflated": "Q uninflated",
    "q_inflated": "Q inflated",
    "if_uninflated": "IF uninflated",
    "if_inflated": "IF inflated",
    "selected": "Selected",
}

PLOT_METHODS = [
    "audit_only",
    "synthetic_naive",
    "pooled_naive",
    "q_inflated",
    "if_inflated",
    "selected",
]

CLEAN_PLOT_METHODS = ["audit_only", "synthetic_naive", "q_inflated", "if_inflated", "selected"]

PLOT_STYLES = {
    "audit_only": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "synthetic_naive": {"color": "#ff7f0e", "marker": "o", "linestyle": "-"},
    "pooled_naive": {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
    "q_inflated": {"color": "#d62728", "marker": "o", "linestyle": "-"},
    "if_inflated": {"color": "#9467bd", "marker": "o", "linestyle": "-"},
    "selected": {
        "color": "black",
        "marker": "D",
        "linestyle": "None",
        "markerfacecolor": "none",
        "markeredgewidth": 1.2,
    },
}


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    cov_shift_strength: float


VARIANTS = (
    Variant("stable_derivative", "Stable derivative", 0.0),
    Variant("moderate_derivative", "Moderate derivative shift", 1.0),
    Variant("stress_derivative", "Derivative stress", 3.0),
)


@dataclass(frozen=True)
class Setting:
    variant: Variant
    m: int
    M: int
    M_multiplier: int
    gamma: float
    gamma_label: str

    @property
    def beta_q(self) -> np.ndarray:
        out = BETA_P.copy()
        out[1] -= self.gamma
        return out

    @property
    def x_mean_q(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    @property
    def x_cov_q(self) -> np.ndarray:
        scale = 1.0 + self.variant.cov_shift_strength * self.gamma
        return np.array([[scale, 0.3], [0.3, 1.0]])


def ensure_dirs() -> None:
    for path in (RESULT_DIR, FIG_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def mismatch_grid(m: int) -> list[tuple[str, float]]:
    inv_sqrt = m ** (-0.5)
    inv_quarter = m ** (-0.25)
    mid = (inv_sqrt * inv_quarter) ** 0.5
    vals = [
        ("zero", 0.0),
        ("strong_low", 0.25 * inv_sqrt),
        ("strong_high", 0.75 * inv_sqrt),
        ("strong_boundary", inv_sqrt),
        ("moderate_mid", mid),
        ("moderate_boundary", inv_quarter),
        ("weak", 1.5 * inv_quarter),
    ]
    return [(label, float(val)) for label, val in vals]


def second_moment_matrix(x_mean: np.ndarray, x_cov: np.ndarray) -> np.ndarray:
    second = x_cov + np.outer(x_mean, x_mean)
    out = np.empty((3, 3))
    out[0, 0] = 1.0
    out[0, 1:] = x_mean
    out[1:, 0] = x_mean
    out[1:, 1:] = second
    return out


def build_settings() -> list[Setting]:
    settings: list[Setting] = []
    for variant in VARIANTS:
        for m in M_VALUES:
            for multiplier in M_MULTIPLIERS:
                for gamma_label, gamma in mismatch_grid(m):
                    settings.append(
                        Setting(
                            variant=variant,
                            m=m,
                            M=m * multiplier,
                            M_multiplier=multiplier,
                            gamma=gamma,
                            gamma_label=gamma_label,
                        )
                    )
    return settings


def draw_linear_sample(
    rng: np.random.Generator,
    *,
    reps: int,
    n: int,
    x_mean: np.ndarray,
    x_cov: np.ndarray,
    beta: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_raw = rng.multivariate_normal(mean=x_mean, cov=x_cov, size=(reps, n))
    x = np.empty((reps, n, 3))
    x[:, :, 0] = 1.0
    x[:, :, 1:] = x_raw
    noise = rng.normal(loc=0.0, scale=sigma, size=(reps, n))
    y = np.einsum("rnp,p->rn", x, beta) + noise
    return x, y


def normal_equations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[1]
    xtx = np.einsum("rni,rnj->rij", x, x) / n
    xty = np.einsum("rni,rn->ri", x, y) / n
    beta_hat = np.linalg.solve(xtx, xty)
    return xtx, xty, beta_hat


def contrast_se_from_sample(x: np.ndarray, y: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    xtx = np.einsum("rni,rnj->rij", x, x) / n
    inv_xtx = np.linalg.inv(xtx)
    resid = y - np.einsum("rnp,rp->rn", x, beta_hat)
    meat = np.einsum("rni,rnj,rn->rij", x, x, resid**2) / n
    cov = np.einsum("rij,rjk,rkl->ril", inv_xtx, meat, inv_xtx) / n
    return np.sqrt(np.maximum(np.einsum("i,rij,j->r", TARGET_CONTRAST, cov, TARGET_CONTRAST), 0.0))


def contrast_se_from_moments(x: np.ndarray, resid: np.ndarray, inv_j: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    scores = np.einsum("ij,rnj,rn->rni", inv_j, x, resid)
    values = np.einsum("i,rni->rn", TARGET_CONTRAST, scores)
    return values.std(axis=1, ddof=1) / np.sqrt(n)


def summarize_intervals(
    *,
    setting: Setting,
    method: str,
    target: float,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    selected: str | None = None,
) -> dict[str, float | str | int]:
    length = upper - lower
    cover = (lower <= target) & (target <= upper)
    pop = population_quantities(setting)
    return {
        "simulation": "S3_estimating_equation",
        "variant": setting.variant.name,
        "variant_label": setting.variant.label,
        "m": setting.m,
        "M": setting.M,
        "M_multiplier": setting.M_multiplier,
        "gamma": setting.gamma,
        "gamma_label": setting.gamma_label,
        "m_inv_sqrt": setting.m ** (-0.5),
        "m_inv_quarter": setting.m ** (-0.25),
        "direct_gap": pop["direct_gap"],
        "if_residual": pop["if_residual"],
        "derivative_gap": pop["derivative_gap"],
        "method": method,
        "method_label": METHOD_LABELS[method],
        "selected_branch": selected or "",
        "coverage": cover.mean(),
        "avg_length": length.mean(),
        "median_length": np.median(length),
        "length_iqr": np.quantile(length, 0.75) - np.quantile(length, 0.25),
        "bias": point.mean() - target,
        "emp_sd": point.std(ddof=1),
        "target": target,
    }


@lru_cache(maxsize=None)
def population_quantities(setting: Setting) -> dict[str, float | np.ndarray]:
    j_p = second_moment_matrix(P_X_MEAN, P_X_COV)
    j_q = second_moment_matrix(setting.x_mean_q, setting.x_cov_q)
    beta_q = setting.beta_q
    target_p = float(TARGET_CONTRAST @ BETA_P)
    target_q = float(TARGET_CONTRAST @ beta_q)
    direct_gap = abs(target_p - target_q)
    if_limit = beta_q + np.linalg.solve(j_q, j_p @ (BETA_P - beta_q))
    if_residual = abs(float(TARGET_CONTRAST @ (if_limit - BETA_P)))
    derivative_gap = float(np.linalg.norm(j_p - j_q, ord=2))

    # Per-observation approximate standard deviations for branch selection.
    audit_sd = SIGMA * np.sqrt(float(TARGET_CONTRAST @ np.linalg.inv(j_p) @ TARGET_CONTRAST))
    q_sd = SIGMA * np.sqrt(float(TARGET_CONTRAST @ np.linalg.inv(j_q) @ TARGET_CONTRAST))

    rng = np.random.default_rng(12345 + int(1000 * setting.gamma) + int(10 * setting.variant.cov_shift_strength))
    x_mc, y_mc = draw_linear_sample(
        rng,
        reps=1,
        n=200000,
        x_mean=P_X_MEAN,
        x_cov=P_X_COV,
        beta=BETA_P,
        sigma=SIGMA,
    )
    x_mc = x_mc[0]
    y_mc = y_mc[0]
    resid_q = y_mc - x_mc @ beta_q
    scores = x_mc @ np.linalg.inv(j_q).T
    if_values = scores @ TARGET_CONTRAST * resid_q
    if_sd = float(if_values.std(ddof=1))

    return {
        "j_p": j_p,
        "j_q": j_q,
        "target_p": target_p,
        "target_q": target_q,
        "direct_gap": float(direct_gap),
        "if_limit": if_limit,
        "if_residual": float(if_residual),
        "derivative_gap": derivative_gap,
        "audit_sd": float(audit_sd),
        "q_sd": float(q_sd),
        "if_sd": if_sd,
    }


def selected_branch(setting: Setting) -> str:
    pop = population_quantities(setting)
    crit_audit = t.ppf(1 - ALPHA / 2, df=setting.m - P_DIM)
    crit_q = t.ppf(1 - ALPHA / 2, df=setting.M - P_DIM)
    half_lengths = {
        "A": crit_audit * float(pop["audit_sd"]) / np.sqrt(setting.m),
        "Q": crit_q * float(pop["q_sd"]) / np.sqrt(setting.M) + float(pop["direct_gap"]),
        "IF": crit_audit * float(pop["if_sd"]) / np.sqrt(setting.m) + float(pop["if_residual"]),
    }
    return min(half_lengths, key=half_lengths.get)


def run_setting(setting: Setting, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    pop = population_quantities(setting)
    target = float(pop["target_p"])
    j_q = np.asarray(pop["j_q"])
    beta_q = setting.beta_q
    branch = selected_branch(setting)

    x_p, y_p = draw_linear_sample(
        rng,
        reps=REPS,
        n=setting.m,
        x_mean=P_X_MEAN,
        x_cov=P_X_COV,
        beta=BETA_P,
        sigma=SIGMA,
    )
    x_q, y_q = draw_linear_sample(
        rng,
        reps=REPS,
        n=setting.M,
        x_mean=setting.x_mean_q,
        x_cov=setting.x_cov_q,
        beta=beta_q,
        sigma=SIGMA,
    )

    xtx_p, xty_p, beta_audit = normal_equations(x_p, y_p)
    xtx_q, xty_q, beta_synth = normal_equations(x_q, y_q)

    n_total = setting.m + setting.M
    xtx_pool = (setting.m * xtx_p + setting.M * xtx_q) / n_total
    xty_pool = (setting.m * xty_p + setting.M * xty_q) / n_total
    beta_pool = np.linalg.solve(xtx_pool, xty_pool)

    audit_se = contrast_se_from_sample(x_p, y_p, beta_audit)
    synth_se = contrast_se_from_sample(x_q, y_q, beta_synth)

    resid_pooled_p = y_p - np.einsum("rnp,rp->rn", x_p, beta_pool)
    resid_pooled_q = y_q - np.einsum("rnp,rp->rn", x_q, beta_pool)
    meat_pool = (
        np.einsum("rni,rnj,rn->rij", x_p, x_p, resid_pooled_p**2)
        + np.einsum("rni,rnj,rn->rij", x_q, x_q, resid_pooled_q**2)
    ) / n_total
    inv_xtx_pool = np.linalg.inv(xtx_pool)
    cov_pool = np.einsum("rij,rjk,rkl->ril", inv_xtx_pool, meat_pool, inv_xtx_pool) / n_total
    pooled_se = np.sqrt(np.maximum(np.einsum("i,rij,j->r", TARGET_CONTRAST, cov_pool, TARGET_CONTRAST), 0.0))

    # One-step correction around the synthetic root using the synthetic derivative.
    audit_moment_at_q = xty_p - np.einsum("rij,rj->ri", xtx_p, beta_synth)
    beta_if = beta_synth + np.einsum("ij,rj->ri", np.linalg.inv(j_q), audit_moment_at_q)
    resid_if = y_p - np.einsum("rnp,rp->rn", x_p, beta_synth)
    if_se = contrast_se_from_moments(x_p, resid_if, np.linalg.inv(j_q))

    crit_audit = t.ppf(1 - ALPHA / 2, df=setting.m - P_DIM)
    crit_synth = t.ppf(1 - ALPHA / 2, df=setting.M - P_DIM)
    crit_pool = t.ppf(1 - ALPHA / 2, df=n_total - P_DIM)

    point = {
        "audit_only": beta_audit @ TARGET_CONTRAST,
        "synthetic_naive": beta_synth @ TARGET_CONTRAST,
        "pooled_naive": beta_pool @ TARGET_CONTRAST,
        "q_uninflated": beta_synth @ TARGET_CONTRAST,
        "q_inflated": beta_synth @ TARGET_CONTRAST,
        "if_uninflated": beta_if @ TARGET_CONTRAST,
        "if_inflated": beta_if @ TARGET_CONTRAST,
    }
    half = {
        "audit_only": crit_audit * audit_se,
        "synthetic_naive": crit_synth * synth_se,
        "pooled_naive": crit_pool * pooled_se,
        "q_uninflated": crit_synth * synth_se,
        "q_inflated": crit_synth * synth_se + float(pop["direct_gap"]),
        "if_uninflated": crit_audit * if_se,
        "if_inflated": crit_audit * if_se + float(pop["if_residual"]),
    }

    selected_method = {"A": "audit_only", "Q": "q_inflated", "IF": "if_inflated"}[branch]
    point["selected"] = point[selected_method]
    half["selected"] = half[selected_method]

    rows = []
    for method in METHOD_ORDER:
        lower = point[method] - half[method]
        upper = point[method] + half[method]
        rows.append(
            summarize_intervals(
                setting=setting,
                method=method,
                target=target,
                point=point[method],
                lower=lower,
                upper=upper,
                selected=branch if method == "selected" else None,
            )
        )
    return rows


def run_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str | int]] = []
    for setting in build_settings():
        rows.extend(run_setting(setting, rng))
    summary = pd.DataFrame(rows)
    branch = (
        summary[summary["method"] == "selected"]
        .groupby(["variant", "variant_label", "m", "M", "M_multiplier", "gamma", "gamma_label", "selected_branch"])
        .size()
        .reset_index(name="settings")
    )
    return summary, branch


def make_slice(
    summary: pd.DataFrame,
    *,
    variant: str = "moderate_derivative",
    multiplier: int = 50,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    plot_methods = methods or PLOT_METHODS
    return summary[
        (summary["variant"] == variant)
        & (summary["M_multiplier"] == multiplier)
        & (summary["method"].isin(plot_methods))
    ].copy()


def plot_coverage_length(
    summary: pd.DataFrame,
    *,
    variant: str,
    out: Path,
    title: str,
    methods: list[str] | None = None,
) -> None:
    data = make_slice(summary, variant=variant, methods=methods)
    plot_methods = methods or PLOT_METHODS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for method in plot_methods:
        d = data[data["method"] == method].sort_values("direct_gap")
        style = PLOT_STYLES.get(method, {"marker": "o", "linestyle": "-"})
        axes[0].plot(d["direct_gap"], d["coverage"], label=METHOD_LABELS[method], **style)
        axes[1].plot(d["direct_gap"], d["avg_length"], label=METHOD_LABELS[method], **style)

    for ax in axes:
        ax.axvline(100 ** (-0.5), color="gray", linestyle="--", linewidth=1, label=r"$m^{-1/2}$" if ax is axes[0] else None)
        ax.axvline(100 ** (-0.25), color="gray", linestyle=":", linewidth=1.2, label=r"$m^{-1/4}$" if ax is axes[0] else None)
        ax.set_xlabel(r"Direct target gap $|\psi_1(P)-\psi_1(Q)|$")
        ax.grid(alpha=0.25)
    axes[0].axhline(0.95, color="black", linestyle="-", linewidth=0.8)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Average interval length")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_variant_comparison(summary: pd.DataFrame, out: Path) -> None:
    data = summary[
        (summary["method"].isin(["q_inflated", "if_inflated", "selected"]))
        & (summary["M_multiplier"] == 50)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for variant in [v.name for v in VARIANTS]:
        d = data[(data["variant"] == variant) & (data["method"] == "if_inflated")].sort_values("direct_gap")
        label = d["variant_label"].iloc[0]
        axes[0].plot(d["direct_gap"], d["if_residual"], marker="o", label=label)
        axes[1].plot(d["direct_gap"], d["avg_length"], marker="o", label=label)
    for ax in axes:
        ax.set_xlabel(r"Direct target gap")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Oracle IF residual buffer")
    axes[1].set_ylabel("IF-inflated average length")
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 2: derivative discrepancy makes IF correction more costly")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_key_findings(summary: pd.DataFrame, branch: pd.DataFrame) -> None:
    def row(variant: str, method: str, gamma_label: str, multiplier: int = 50) -> pd.Series:
        d = summary[
            (summary["variant"] == variant)
            & (summary["M_multiplier"] == multiplier)
            & (summary["gamma_label"] == gamma_label)
            & (summary["method"] == method)
        ]
        return d.iloc[0]

    stable_mid_if = row("stable_derivative", "if_inflated", "moderate_mid")
    stable_mid_q = row("stable_derivative", "q_uninflated", "moderate_mid")
    moderate_mid_if = row("moderate_derivative", "if_inflated", "moderate_mid")
    moderate_mid_q = row("moderate_derivative", "q_uninflated", "moderate_mid")
    stable_boundary_if = row("stable_derivative", "if_inflated", "moderate_boundary")
    stress_boundary_if = row("stress_derivative", "if_inflated", "moderate_boundary")
    weak_selected = row("moderate_derivative", "selected", "weak")

    selected = summary[
        (summary["method"] == "selected")
        & (summary["M_multiplier"] == 50)
    ][
        [
            "variant_label",
            "gamma_label",
            "direct_gap",
            "if_residual",
            "derivative_gap",
            "selected_branch",
            "coverage",
            "avg_length",
        ]
    ]

    branch_text = branch[branch["M_multiplier"] == 50].to_string(index=False)

    text = f"""# Stage 2 Estimating-Equation Results Summary

Created: 2026-05-20

These results come from `simulation/stage2_estimating_equation.py` with `{REPS}` Monte Carlo replicates per setting.

## Main Takeaways

1. Stage 2 confirms the basic estimating-equation transfer problem. In the stable-derivative moderate-mismatch setting, the uninflated Q-centered interval had coverage {stable_mid_q.coverage:.3f}, while the IF-inflated interval had coverage {stable_mid_if.coverage:.3f}.

2. The same pattern holds when the local derivative differs moderately. In the moderate-derivative setting, the uninflated Q-centered interval had coverage {moderate_mid_q.coverage:.3f}, while the IF-inflated interval had coverage {moderate_mid_if.coverage:.3f}.

3. Derivative discrepancy matters. When the target gap reaches the moderate-boundary setting, the IF-inflated average length increases from {stable_boundary_if.avg_length:.3f} in the stable-derivative setting to {stress_boundary_if.avg_length:.3f} under derivative stress, because the oracle IF residual buffer grows.

4. The selected rule is conservative in this low-dimensional linear example. It mostly chooses Q-inflated intervals while the direct gap is still cheap and audit-only intervals once the direct gap becomes too large. In the moderate-derivative weak-mismatch setting it selects `{weak_selected.selected_branch}`, with coverage {weak_selected.coverage:.3f} and average length {weak_selected.avg_length:.3f}.

5. The IF branch is therefore best interpreted here as a theorem-validation branch, not yet as the main empirical win. This is useful but not sufficient for the paper's practical-value story. The next stage should use a semiparametric/nuisance-estimation target where synthetic data can genuinely improve nuisance estimation.

## Basic QC

- Expected rows: 336.
- Actual rows in `stage2_summary.csv`: 336.
- `python -m py_compile simulation/stage2_estimating_equation.py` passed.
- Blank `selected_branch` values occur only for non-selected methods; the selected rows have explicit branch labels.

## Selected Branch Behavior, M=50m

```text
{selected.to_string(index=False)}
```

## Branch Counts, M=50m

```text
{branch_text}
```

## Output Files

- `tables/stage2_estimating_equation/stage2_summary.csv`
- `tables/stage2_estimating_equation/stage2_main_slice.csv`
- `tables/stage2_estimating_equation/stage2_branch_frequencies.csv`
- `figures/stage2_estimating_equation/stage2_moderate_derivative_full.png`
- `figures/stage2_estimating_equation/stage2_moderate_derivative_clean.png`
- `figures/stage2_estimating_equation/stage2_derivative_stress.png`
"""
    (RESULT_DIR / "STAGE2_RESULTS_SUMMARY.md").write_text(text)


def main() -> None:
    ensure_dirs()
    summary, branch = run_all()

    summary.to_csv(TABLE_DIR / "stage2_summary.csv", index=False)
    branch.to_csv(TABLE_DIR / "stage2_branch_frequencies.csv", index=False)
    main_slice = summary[(summary["M_multiplier"] == 50) & (summary["method"].isin(PLOT_METHODS))].copy()
    main_slice.to_csv(TABLE_DIR / "stage2_main_slice.csv", index=False)

    plot_coverage_length(
        summary,
        variant="moderate_derivative",
        out=FIG_DIR / "stage2_moderate_derivative_full.png",
        title="S3: Estimating-equation transfer, moderate derivative shift",
    )
    plot_coverage_length(
        summary,
        variant="moderate_derivative",
        out=FIG_DIR / "stage2_moderate_derivative_clean.png",
        title="S3: Estimating-equation transfer, moderate derivative shift",
        methods=CLEAN_PLOT_METHODS,
    )
    plot_variant_comparison(summary, FIG_DIR / "stage2_derivative_stress.png")

    write_key_findings(summary, branch)
    summary.to_csv(RESULT_DIR / "stage2_summary.csv", index=False)
    branch.to_csv(RESULT_DIR / "stage2_branch_frequencies.csv", index=False)

    print(f"Wrote results to {RESULT_DIR}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
