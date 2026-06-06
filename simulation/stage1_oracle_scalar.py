#!/usr/bin/env python3
"""Stage 1 oracle scalar simulations for synthetic-data inference.

This script implements S1 and S2 from SIMULATION_DESIGN_PLAN.md.
It uses the bounded binary model Y in {-1,1}, where P_mu and Q_nu
are known exactly. The goal is to test the first-order direct transfer
cost and the quadratic IF residual in a setting where the true
real-synthetic discrepancy is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, norm


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "simulation" / "results" / "stage1_oracle_scalar"
FIG_DIR = ROOT / "figures" / "stage1_oracle_scalar"
TABLE_DIR = ROOT / "tables" / "stage1_oracle_scalar"

ALPHA = 0.05
Z = norm.ppf(1 - ALPHA / 2)
REPS = 5000
SEED = 20260520

M_VALUES = (50, 100, 250, 500)
M_MULTIPLIERS = (1, 10, 100)
NU_VALUES = (0.0, 0.25)

TAU_Q = 1.0
TAU_IF = 1.0
RHO = 10


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

CLEAN_PLOT_METHODS = {
    "S1_mean": ["audit_only", "synthetic_naive", "q_inflated", "selected"],
    "S2_squared_mean": ["audit_only", "synthetic_naive", "q_inflated", "if_inflated", "selected"],
}

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
class Setting:
    simulation: str
    m: int
    M: int
    M_multiplier: int
    nu: float
    epsilon: float
    epsilon_label: str

    @property
    def mu(self) -> float:
        return self.nu + self.epsilon


def ensure_dirs() -> None:
    for path in (RESULT_DIR, FIG_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def eps_grid(m: int) -> list[tuple[str, float]]:
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
    return [(label, min(float(eps), 0.80)) for label, eps in vals]


def draw_binary_means(rng: np.random.Generator, mean: float, n: int, reps: int) -> np.ndarray:
    if not (-1 < mean < 1):
        raise ValueError(f"Binary mean must be in (-1,1), got {mean}")
    p = (1 + mean) / 2
    counts = rng.binomial(n=n, p=p, size=reps)
    return 2 * counts / n - 1


def mean_se(ybar: np.ndarray, n: int) -> np.ndarray:
    var = np.maximum(1 - ybar**2, 0)
    return np.sqrt(var / n)


def ci_mean(center: np.ndarray, se: np.ndarray, inflation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return center - Z * se - inflation, center + Z * se + inflation


def ci_mean_clopper_pearson(mean_hat: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact binomial confidence interval for the {-1,1} mean."""
    counts = np.rint((mean_hat + 1) * n / 2).astype(int)
    counts = np.clip(counts, 0, n)

    lo_p = np.zeros_like(mean_hat, dtype=float)
    hi_p = np.ones_like(mean_hat, dtype=float)

    lower_mask = counts > 0
    upper_mask = counts < n
    lo_p[lower_mask] = beta.ppf(ALPHA / 2, counts[lower_mask], n - counts[lower_mask] + 1)
    hi_p[upper_mask] = beta.ppf(1 - ALPHA / 2, counts[upper_mask] + 1, n - counts[upper_mask])

    return 2 * lo_p - 1, 2 * hi_p - 1


def ci_square_from_mean_bounds(
    lower_mu: np.ndarray, upper_mu: np.ndarray, inflation: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(lower_mu**2, upper_mu**2)
    crosses_zero = (lower_mu <= 0) & (upper_mu >= 0)
    lo = np.where(crosses_zero, 0.0, lo)
    hi = np.maximum(lower_mu**2, upper_mu**2)
    return lo - inflation, hi + inflation


def ci_square_from_mean(mean_hat: np.ndarray, se: np.ndarray, inflation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    lower_mu = mean_hat - Z * se
    upper_mu = mean_hat + Z * se
    return ci_square_from_mean_bounds(lower_mu, upper_mu, inflation)


def selected_branch(epsilon: float, m: int, M: int) -> str:
    if epsilon <= TAU_Q * m ** (-0.5) and M >= RHO * m:
        return "Q"
    if epsilon <= TAU_IF * m ** (-0.25):
        return "IF"
    return "A"


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
    return {
        "simulation": setting.simulation,
        "m": setting.m,
        "M": setting.M,
        "M_multiplier": setting.M_multiplier,
        "nu": setting.nu,
        "mu": setting.mu,
        "epsilon": setting.epsilon,
        "epsilon_label": setting.epsilon_label,
        "m_inv_sqrt": setting.m ** (-0.5),
        "m_inv_quarter": setting.m ** (-0.25),
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


def run_s1(setting: Setting, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    pbar = draw_binary_means(rng, setting.mu, setting.m, REPS)
    qbar = draw_binary_means(rng, setting.nu, setting.M, REPS)
    pooled = (setting.m * pbar + setting.M * qbar) / (setting.m + setting.M)

    se_p = mean_se(pbar, setting.m)
    se_q = mean_se(qbar, setting.M)
    se_pool = mean_se(pooled, setting.m + setting.M)

    target = setting.mu
    branch = selected_branch(setting.epsilon, setting.m, setting.M)
    audit_lo, audit_hi = ci_mean_clopper_pearson(pbar, setting.m)

    intervals = {}
    intervals["audit_only"] = (pbar, audit_lo, audit_hi)
    intervals["synthetic_naive"] = (qbar, *ci_mean(qbar, se_q))
    intervals["pooled_naive"] = (pooled, *ci_mean(pooled, se_pool))
    intervals["q_uninflated"] = (qbar, *ci_mean(qbar, se_q))
    intervals["q_inflated"] = (qbar, *ci_mean(qbar, se_q, setting.epsilon))

    # For a mean target, the IF correction collapses algebraically to the audit mean.
    intervals["if_uninflated"] = intervals["audit_only"]
    intervals["if_inflated"] = intervals["audit_only"]

    selected_method = {"Q": "q_inflated", "IF": "if_inflated", "A": "audit_only"}[branch]
    intervals["selected"] = intervals[selected_method]

    return [
        summarize_intervals(
            setting=setting,
            method=method,
            target=target,
            point=vals[0],
            lower=vals[1],
            upper=vals[2],
            selected=branch if method == "selected" else None,
        )
        for method, vals in intervals.items()
    ]


def run_s2(setting: Setting, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    pbar = draw_binary_means(rng, setting.mu, setting.m, REPS)
    qbar = draw_binary_means(rng, setting.nu, setting.M, REPS)
    pooled = (setting.m * pbar + setting.M * qbar) / (setting.m + setting.M)

    se_p = mean_se(pbar, setting.m)
    se_q = mean_se(qbar, setting.M)
    se_pool = mean_se(pooled, setting.m + setting.M)

    target = setting.mu**2
    omega_q = 2 * abs(setting.nu) * setting.epsilon + setting.epsilon**2
    omega_if = setting.epsilon**2
    branch = selected_branch(setting.epsilon, setting.m, setting.M)

    q_point = qbar**2
    pooled_point = pooled**2
    audit_point = pbar**2
    audit_mu_lo, audit_mu_hi = ci_mean_clopper_pearson(pbar, setting.m)

    # First-order expansion at the synthetic anchor:
    # theta(Q_hat) + P_m[2 Q_hat (Y - Q_hat)].
    if_point = qbar**2 + 2 * qbar * (pbar - qbar)
    if_se = np.abs(2 * qbar) * se_p

    intervals = {}
    intervals["audit_only"] = (audit_point, *ci_square_from_mean_bounds(audit_mu_lo, audit_mu_hi))
    intervals["synthetic_naive"] = (q_point, *ci_square_from_mean(qbar, se_q))
    intervals["pooled_naive"] = (pooled_point, *ci_square_from_mean(pooled, se_pool))
    intervals["q_uninflated"] = (q_point, *ci_square_from_mean(qbar, se_q))
    intervals["q_inflated"] = (q_point, *ci_square_from_mean(qbar, se_q, omega_q))
    intervals["if_uninflated"] = (if_point, *ci_mean(if_point, if_se))
    intervals["if_inflated"] = (if_point, *ci_mean(if_point, if_se, omega_if))

    selected_method = {"Q": "q_inflated", "IF": "if_inflated", "A": "audit_only"}[branch]
    intervals["selected"] = intervals[selected_method]

    return [
        summarize_intervals(
            setting=setting,
            method=method,
            target=target,
            point=vals[0],
            lower=vals[1],
            upper=vals[2],
            selected=branch if method == "selected" else None,
        )
        for method, vals in intervals.items()
    ]


def build_settings() -> list[Setting]:
    settings: list[Setting] = []
    for simulation in ("S1_mean", "S2_squared_mean"):
        for m in M_VALUES:
            for multiplier in M_MULTIPLIERS:
                M = m * multiplier
                for nu in NU_VALUES:
                    for eps_label, eps in eps_grid(m):
                        if nu + eps >= 0.95:
                            continue
                        settings.append(
                            Setting(
                                simulation=simulation,
                                m=m,
                                M=M,
                                M_multiplier=multiplier,
                                nu=nu,
                                epsilon=eps,
                                epsilon_label=eps_label,
                            )
                        )
    return settings


def run_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str | int]] = []
    for setting in build_settings():
        if setting.simulation == "S1_mean":
            rows.extend(run_s1(setting, rng))
        else:
            rows.extend(run_s2(setting, rng))
    summary = pd.DataFrame(rows)
    branch = (
        summary[summary["method"] == "selected"]
        .groupby(["simulation", "m", "M", "M_multiplier", "nu", "epsilon", "epsilon_label", "selected_branch"])
        .size()
        .reset_index(name="settings")
    )
    return summary, branch


def make_slice(summary: pd.DataFrame, simulation: str, m: int = 100, multiplier: int = 100, nu: float = 0.25) -> pd.DataFrame:
    return summary[
        (summary["simulation"] == simulation)
        & (summary["m"] == m)
        & (summary["M_multiplier"] == multiplier)
        & np.isclose(summary["nu"], nu)
        & (summary["method"].isin(PLOT_METHODS))
    ].copy()


def plot_coverage_length(
    summary: pd.DataFrame,
    simulation: str,
    out: Path,
    title: str,
    nu: float,
    methods: list[str] | None = None,
) -> None:
    data = make_slice(summary, simulation, nu=nu)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    plot_methods = methods or PLOT_METHODS
    for method in plot_methods:
        d = data[data["method"] == method].sort_values("epsilon")
        style = PLOT_STYLES.get(method, {"marker": "o", "linestyle": "-"})
        axes[0].plot(d["epsilon"], d["coverage"], label=METHOD_LABELS[method], **style)
        axes[1].plot(d["epsilon"], d["avg_length"], label=METHOD_LABELS[method], **style)

    for ax in axes:
        ax.axvline(100 ** (-0.5), color="gray", linestyle="--", linewidth=1, label=r"$m^{-1/2}$" if ax is axes[0] else None)
        ax.axvline(100 ** (-0.25), color="gray", linestyle=":", linewidth=1.2, label=r"$m^{-1/4}$" if ax is axes[0] else None)
        ax.set_xlabel(r"True discrepancy $\epsilon=|\mu-\nu|$")
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


def plot_s2_residual_scaling(out: Path) -> pd.DataFrame:
    eps_vals = np.array([0.01, 0.015, 0.025, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.35])
    rows = []
    for nu in (0.0, 0.25):
        for eps in eps_vals:
            mu = nu + eps
            direct_gap = abs(mu**2 - nu**2)
            if_residual = eps**2
            rows.append({"nu": nu, "epsilon": eps, "quantity": "Direct target gap", "value": direct_gap})
            rows.append({"nu": nu, "epsilon": eps, "quantity": "IF residual", "value": if_residual})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, nu in zip(axes, (0.0, 0.25)):
        dnu = df[df["nu"] == nu]
        for quantity, marker in [("Direct target gap", "o"), ("IF residual", "s")]:
            d = dnu[dnu["quantity"] == quantity]
            ax.loglog(d["epsilon"], d["value"], marker=marker, label=quantity)
        ax.loglog(eps_vals, eps_vals, color="gray", linestyle="--", linewidth=1, label=r"slope 1: $\epsilon$")
        ax.loglog(eps_vals, eps_vals**2, color="gray", linestyle=":", linewidth=1.2, label=r"slope 2: $\epsilon^2$")
        ax.set_title(rf"Synthetic anchor $\nu={nu}$")
        ax.set_xlabel(r"True discrepancy $\epsilon$")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Absolute target gap")
    axes[0].legend(fontsize=8)
    fig.suptitle("S2: Direct target gap versus IF residual")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return df


def write_key_findings(summary: pd.DataFrame, residual_df: pd.DataFrame) -> None:
    def row(sim: str, method: str, eps_label: str, nu: float = 0.25) -> pd.Series:
        d = summary[
            (summary["simulation"] == sim)
            & (summary["m"] == 100)
            & (summary["M_multiplier"] == 100)
            & np.isclose(summary["nu"], nu)
            & (summary["epsilon_label"] == eps_label)
            & (summary["method"] == method)
        ]
        return d.iloc[0]

    s1_weak_naive = row("S1_mean", "synthetic_naive", "weak", nu=0.0)
    s1_weak_q = row("S1_mean", "q_inflated", "weak", nu=0.0)
    s1_strong_q = row("S1_mean", "q_inflated", "strong_low", nu=0.0)
    s2_mod_if = row("S2_squared_mean", "if_inflated", "moderate_mid", nu=0.25)
    s2_mod_q_un = row("S2_squared_mean", "q_uninflated", "moderate_mid", nu=0.25)
    selected = summary[
        (summary["method"] == "selected")
        & (summary["m"] == 100)
        & (summary["M_multiplier"] == 100)
        & np.isclose(summary["nu"], 0.25)
    ][["simulation", "epsilon_label", "epsilon", "selected_branch", "coverage", "avg_length"]]

    text = f"""# Stage 1 Oracle Scalar Results Summary

Created: 2026-05-20

These results come from `simulation/stage1_oracle_scalar.py` with `{REPS}` Monte Carlo replicates per setting.

## Main Takeaways

1. S1 confirms the direct-transfer problem. In the mean-target model, naive synthetic-only inference under weak mismatch had coverage {s1_weak_naive.coverage:.3f}, while the Q-centered interval inflated by the true discrepancy had coverage {s1_weak_q.coverage:.3f}. Under strong closeness, the inflated Q-centered interval remained short, with average length {s1_strong_q.avg_length:.3f}.

2. S2 confirms the correction mechanism. Under a nonzero synthetic anchor and moderate mismatch, the Q-centered uninflated interval had coverage {s2_mod_q_un.coverage:.3f}, while the IF interval with the `epsilon^2` residual buffer had coverage {s2_mod_if.coverage:.3f}.

3. The residual-scaling plot shows the intended geometry: with a nonzero anchor, the direct target gap tracks the first-order scale, while the IF residual tracks the second-order scale. With anchor `nu=0`, the direct leading term vanishes, which is a useful special case and should be described carefully if included in the manuscript.

4. These are mechanism simulations, not practical application evidence. Their role is to validate the phase diagram and Proposition 1 before moving to estimating equations, nuisance estimation, learned generators, and real data.

5. The audit-only comparator uses an exact binomial confidence interval for the {-1,1} mean, transformed when the target is squared mean. This avoids making the audit-only safety baseline look weak because of a finite-sample Wald artifact.

## Selected Branch Behavior, m=100, M=100m, nu=0.25

```text
{selected.to_string(index=False)}
```

## Output Files

- `tables/stage1_oracle_scalar/stage1_summary.csv`
- `tables/stage1_oracle_scalar/stage1_main_slice.csv`
- `tables/stage1_oracle_scalar/stage1_branch_frequencies.csv`
- `figures/stage1_oracle_scalar/s1_mean_coverage_length.png`
- `figures/stage1_oracle_scalar/s2_squared_coverage_length.png`
- `figures/stage1_oracle_scalar/s2_residual_scaling.png`
- `figures/stage1_oracle_scalar/s1_mean_main_clean.png`
- `figures/stage1_oracle_scalar/s2_squared_main_clean.png`
"""
    (RESULT_DIR / "STAGE1_RESULTS_SUMMARY.md").write_text(text)


def main() -> None:
    ensure_dirs()
    summary, branch = run_all()
    residual_df = plot_s2_residual_scaling(FIG_DIR / "s2_residual_scaling.png")

    summary.to_csv(TABLE_DIR / "stage1_summary.csv", index=False)
    branch.to_csv(TABLE_DIR / "stage1_branch_frequencies.csv", index=False)
    residual_df.to_csv(TABLE_DIR / "stage1_s2_residual_scaling.csv", index=False)

    main_slice = summary[
        (summary["m"] == 100)
        & (summary["M_multiplier"] == 100)
        & (summary["method"].isin(PLOT_METHODS))
    ].copy()
    main_slice.to_csv(TABLE_DIR / "stage1_main_slice.csv", index=False)

    plot_coverage_length(
        summary,
        "S1_mean",
        FIG_DIR / "s1_mean_coverage_length.png",
        "S1: Mean target, oracle discrepancy (m=100, M=100m, nu=0)",
        nu=0.0,
    )
    plot_coverage_length(
        summary,
        "S1_mean",
        FIG_DIR / "s1_mean_main_clean.png",
        "S1: Mean target, oracle discrepancy (m=100, M=100m, nu=0)",
        nu=0.0,
        methods=CLEAN_PLOT_METHODS["S1_mean"],
    )
    plot_coverage_length(
        summary,
        "S2_squared_mean",
        FIG_DIR / "s2_squared_coverage_length.png",
        "S2: Squared-mean target, oracle discrepancy (m=100, M=100m, nu=0.25)",
        nu=0.25,
    )
    plot_coverage_length(
        summary,
        "S2_squared_mean",
        FIG_DIR / "s2_squared_main_clean.png",
        "S2: Squared-mean target, oracle discrepancy (m=100, M=100m, nu=0.25)",
        nu=0.25,
        methods=CLEAN_PLOT_METHODS["S2_squared_mean"],
    )

    write_key_findings(summary, residual_df)
    summary.to_csv(RESULT_DIR / "stage1_summary.csv", index=False)
    branch.to_csv(RESULT_DIR / "stage1_branch_frequencies.csv", index=False)
    residual_df.to_csv(RESULT_DIR / "stage1_s2_residual_scaling.csv", index=False)

    print(f"Wrote results to {RESULT_DIR}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
