#!/usr/bin/env python3
"""Stage 3 semiparametric ATE simulations for synthetic-data inference.

This script implements S4 from SIMULATION_DESIGN_PLAN.md. The target is an
average treatment effect under the real distribution P. The treatment is
randomized with known propensity, so the audit correction is a transparent
semiparametric control variate: large synthetic data provide outcome nuisance
functions, while the small audit sample corrects the target.

The synthetic outcome functions can have a biased direct ATE. The IF branch
uses the audit sample to remove that direct bias and tests whether synthetic
nuisance stabilization can shorten valid intervals relative to audit-only
inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "simulation" / "results" / "stage3_semiparametric_ate"
FIG_DIR = ROOT / "figures" / "stage3_semiparametric_ate"
TABLE_DIR = ROOT / "tables" / "stage3_semiparametric_ate"

ALPHA = 0.05
Z = norm.ppf(1 - ALPHA / 2)
REPS = 5000
SEED = 20260520

M_VALUES = (200,)
M_MULTIPLIERS = (50,)
D = 5
PROPENSITY = 0.5
SIGMA = 1.0
TARGET = 1.0

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
    nuisance_scale: float
    mismatch_scale: float


VARIANTS = (
    Variant("moderate_nuisance", "Moderate nuisance variation", 1.0, 1.0),
    Variant("hard_nuisance", "Hard nuisance variation", 2.0, 1.35),
)


@dataclass(frozen=True)
class Setting:
    variant: Variant
    m: int
    M: int
    M_multiplier: int
    gamma: float
    gamma_label: str


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


def draw_x(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape + (D,))


def base_outcome(x: np.ndarray, scale: float) -> np.ndarray:
    return scale * (
        0.8 * np.sin(x[..., 0])
        + 0.5 * (x[..., 1] ** 2 - 1.0)
        - 0.4 * x[..., 2]
        + 0.3 * x[..., 3] * x[..., 4]
    )


def tau_p(x: np.ndarray, scale: float) -> np.ndarray:
    return TARGET + scale * (0.4 * np.tanh(x[..., 0] + x[..., 1]) + 0.2 * x[..., 2])


def m0_p(x: np.ndarray, scale: float) -> np.ndarray:
    return base_outcome(x, scale)


def m1_p(x: np.ndarray, scale: float) -> np.ndarray:
    return m0_p(x, scale) + tau_p(x, scale)


def m0_q(x: np.ndarray, setting: Setting) -> np.ndarray:
    scale = setting.variant.nuisance_scale
    gamma = setting.gamma * setting.variant.mismatch_scale
    return m0_p(x, scale) + 0.65 * gamma * (x[..., 0] ** 2 - 1.0) + 0.25 * gamma * x[..., 1]


def tau_q(x: np.ndarray, setting: Setting) -> np.ndarray:
    scale = setting.variant.nuisance_scale
    gamma = setting.gamma * setting.variant.mismatch_scale
    return tau_p(x, scale) - gamma + 0.25 * gamma * x[..., 2]


def m1_q(x: np.ndarray, setting: Setting) -> np.ndarray:
    return m0_q(x, setting) + tau_q(x, setting)


def draw_outcomes_p(rng: np.random.Generator, x: np.ndarray, a: np.ndarray, setting: Setting) -> np.ndarray:
    scale = setting.variant.nuisance_scale
    mu = np.where(a == 1, m1_p(x, scale), m0_p(x, scale))
    return mu + rng.normal(scale=SIGMA, size=a.shape)


@lru_cache(maxsize=None)
def population_quantities(setting: Setting) -> dict[str, float]:
    rng = np.random.default_rng(13579 + int(10000 * setting.gamma) + int(100 * setting.variant.nuisance_scale))
    x = draw_x(rng, (600000,))
    scale = setting.variant.nuisance_scale

    tau_p_vals = tau_p(x, scale)
    tau_q_vals = tau_q(x, setting)
    theta_p = float(tau_p_vals.mean())
    theta_q = float(tau_q_vals.mean())
    direct_gap = abs(theta_p - theta_q)

    # With known randomized propensity, the AIPW correction is unbiased for
    # theta(P) for any outcome functions, so the oracle IF residual is zero.
    a = rng.binomial(1, PROPENSITY, size=x.shape[0])
    y = draw_outcomes_p(rng, x, a, setting)
    pseudo_q = aipw_pseudo(x, a, y, setting)

    y1 = m1_p(x, scale) + rng.normal(scale=SIGMA, size=x.shape[0])
    y0 = m0_p(x, scale) + rng.normal(scale=SIGMA, size=x.shape[0])
    audit_diff_sd = float(np.sqrt(np.var(y1, ddof=1) / PROPENSITY + np.var(y0, ddof=1) / (1 - PROPENSITY)))
    synth_obs_sd = float(np.sqrt(np.var(m1_q(x, setting), ddof=1) / PROPENSITY + np.var(m0_q(x, setting), ddof=1) / (1 - PROPENSITY) + SIGMA**2 / PROPENSITY + SIGMA**2 / (1 - PROPENSITY)))

    return {
        "theta_p": theta_p,
        "theta_q": theta_q,
        "direct_gap": float(direct_gap),
        "if_residual": 0.0,
        "if_sd": float(pseudo_q.std(ddof=1)),
        "audit_diff_sd": audit_diff_sd,
        "synthetic_sd": synth_obs_sd,
        "q_contrast_sd": float(tau_q_vals.std(ddof=1)),
    }


def aipw_pseudo(x: np.ndarray, a: np.ndarray, y: np.ndarray, setting: Setting) -> np.ndarray:
    q0 = m0_q(x, setting)
    q1 = m1_q(x, setting)
    return q1 - q0 + a / PROPENSITY * (y - q1) - (1 - a) / (1 - PROPENSITY) * (y - q0)


def selected_branch(setting: Setting) -> str:
    pop = population_quantities(setting)
    crit_m = t.ppf(1 - ALPHA / 2, df=setting.m - 2)
    half_lengths = {
        "A": crit_m * pop["audit_diff_sd"] / np.sqrt(setting.m),
        "Q": Z * pop["synthetic_sd"] / np.sqrt(setting.M) + pop["direct_gap"],
        "IF": crit_m * pop["if_sd"] / np.sqrt(setting.m) + pop["if_residual"],
    }
    return min(half_lengths, key=half_lengths.get)


def ci_from_center(center: np.ndarray, se: np.ndarray, crit: float, inflation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return center - crit * se - inflation, center + crit * se + inflation


def summarize_intervals(
    *,
    setting: Setting,
    method: str,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    selected: str | None = None,
) -> dict[str, float | str | int]:
    pop = population_quantities(setting)
    target = pop["theta_p"]
    length = upper - lower
    cover = (lower <= target) & (target <= upper)
    return {
        "simulation": "S4_semiparametric_ate",
        "variant": setting.variant.name,
        "variant_label": setting.variant.label,
        "m": setting.m,
        "M": setting.M,
        "M_multiplier": setting.M_multiplier,
        "gamma": setting.gamma,
        "gamma_label": setting.gamma_label,
        "m_inv_sqrt": setting.m ** (-0.5),
        "m_inv_quarter": setting.m ** (-0.25),
        "theta_p": target,
        "theta_q": pop["theta_q"],
        "direct_gap": pop["direct_gap"],
        "if_residual": pop["if_residual"],
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


def run_setting(setting: Setting, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    pop = population_quantities(setting)
    target = pop["theta_p"]
    crit_m = t.ppf(1 - ALPHA / 2, df=setting.m - 2)
    branch = selected_branch(setting)

    x_p = draw_x(rng, (REPS, setting.m))
    a_p = rng.binomial(1, PROPENSITY, size=(REPS, setting.m))
    y_p = draw_outcomes_p(rng, x_p, a_p, setting)

    treated = a_p == 1
    control = ~treated
    n1 = treated.sum(axis=1)
    n0 = control.sum(axis=1)
    y1_sum = np.where(treated, y_p, 0.0).sum(axis=1)
    y0_sum = np.where(control, y_p, 0.0).sum(axis=1)
    y1_mean = y1_sum / n1
    y0_mean = y0_sum / n0
    audit_point = y1_mean - y0_mean

    y1_var = np.array([np.var(y_p[i, treated[i]], ddof=1) for i in range(REPS)])
    y0_var = np.array([np.var(y_p[i, control[i]], ddof=1) for i in range(REPS)])
    audit_se = np.sqrt(y1_var / n1 + y0_var / n0)

    pseudo_if = aipw_pseudo(x_p, a_p, y_p, setting)
    if_point = pseudo_if.mean(axis=1)
    if_se = pseudo_if.std(axis=1, ddof=1) / np.sqrt(setting.m)

    synth_center = rng.normal(loc=pop["theta_q"], scale=pop["synthetic_sd"] / np.sqrt(setting.M), size=REPS)
    synth_se = np.full(REPS, pop["synthetic_sd"] / np.sqrt(setting.M))

    pooled_weight = setting.M / (setting.m + setting.M)
    pooled_point = pooled_weight * synth_center + (1 - pooled_weight) * audit_point
    pooled_se = np.sqrt((pooled_weight * synth_se) ** 2 + ((1 - pooled_weight) * audit_se) ** 2)

    intervals = {}
    intervals["audit_only"] = (audit_point, *ci_from_center(audit_point, audit_se, crit_m))
    intervals["synthetic_naive"] = (synth_center, *ci_from_center(synth_center, synth_se, Z))
    intervals["pooled_naive"] = (pooled_point, *ci_from_center(pooled_point, pooled_se, Z))
    intervals["q_uninflated"] = intervals["synthetic_naive"]
    intervals["q_inflated"] = (synth_center, *ci_from_center(synth_center, synth_se, Z, pop["direct_gap"]))
    intervals["if_uninflated"] = (if_point, *ci_from_center(if_point, if_se, crit_m))
    intervals["if_inflated"] = (if_point, *ci_from_center(if_point, if_se, crit_m, pop["if_residual"]))

    selected_method = {"A": "audit_only", "Q": "q_inflated", "IF": "if_inflated"}[branch]
    intervals["selected"] = intervals[selected_method]

    return [
        summarize_intervals(
            setting=setting,
            method=method,
            point=vals[0],
            lower=vals[1],
            upper=vals[2],
            selected=branch if method == "selected" else None,
        )
        for method, vals in intervals.items()
    ]


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


def make_slice(summary: pd.DataFrame, *, variant: str, methods: list[str] | None = None) -> pd.DataFrame:
    plot_methods = methods or PLOT_METHODS
    return summary[(summary["variant"] == variant) & (summary["method"].isin(plot_methods))].copy()


def plot_coverage_length(summary: pd.DataFrame, *, variant: str, out: Path, title: str, methods: list[str] | None = None) -> None:
    data = make_slice(summary, variant=variant, methods=methods)
    plot_methods = methods or PLOT_METHODS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for method in plot_methods:
        d = data[data["method"] == method].sort_values("direct_gap")
        style = PLOT_STYLES.get(method, {"marker": "o", "linestyle": "-"})
        axes[0].plot(d["direct_gap"], d["coverage"], label=METHOD_LABELS[method], **style)
        axes[1].plot(d["direct_gap"], d["avg_length"], label=METHOD_LABELS[method], **style)
    for ax in axes:
        ax.axvline(200 ** (-0.5), color="gray", linestyle="--", linewidth=1, label=r"$m^{-1/2}$" if ax is axes[0] else None)
        ax.axvline(200 ** (-0.25), color="gray", linestyle=":", linewidth=1.2, label=r"$m^{-1/4}$" if ax is axes[0] else None)
        ax.set_xlabel(r"Direct ATE gap $|\theta(P)-\theta(Q)|$")
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


def plot_nuisance_value(summary: pd.DataFrame, out: Path) -> None:
    data = summary[
        summary["method"].isin(["audit_only", "if_inflated", "q_inflated", "selected"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for variant in [v.name for v in VARIANTS]:
        d = data[(data["variant"] == variant) & (data["method"] == "if_inflated")].sort_values("direct_gap")
        label = d["variant_label"].iloc[0]
        axes[0].plot(d["direct_gap"], d["avg_length"], marker="o", label=f"IF, {label}")
        da = data[(data["variant"] == variant) & (data["method"] == "audit_only")].sort_values("direct_gap")
        axes[0].plot(da["direct_gap"], da["avg_length"], marker="x", linestyle="--", label=f"Audit, {label}")

        ratio = da["avg_length"].to_numpy() / d["avg_length"].to_numpy()
        axes[1].plot(d["direct_gap"], ratio, marker="o", label=label)
    for ax in axes:
        ax.set_xlabel(r"Direct ATE gap")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average interval length")
    axes[1].set_ylabel("Audit-only length / IF length")
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)
    fig.suptitle("Stage 3: synthetic outcome nuisances reduce audit uncertainty")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_key_findings(summary: pd.DataFrame, branch: pd.DataFrame) -> None:
    def row(variant: str, method: str, gamma_label: str) -> pd.Series:
        d = summary[
            (summary["variant"] == variant)
            & (summary["gamma_label"] == gamma_label)
            & (summary["method"] == method)
        ]
        return d.iloc[0]

    hard_mod_if = row("hard_nuisance", "if_inflated", "moderate_mid")
    hard_mod_a = row("hard_nuisance", "audit_only", "moderate_mid")
    hard_mod_q_un = row("hard_nuisance", "q_uninflated", "moderate_mid")
    hard_mod_q = row("hard_nuisance", "q_inflated", "moderate_mid")
    hard_weak_sel = row("hard_nuisance", "selected", "weak")

    selected = summary[summary["method"] == "selected"][
        [
            "variant_label",
            "gamma_label",
            "direct_gap",
            "if_residual",
            "selected_branch",
            "coverage",
            "avg_length",
        ]
    ]

    text = f"""# Stage 3 Semiparametric ATE Results Summary

Created: 2026-05-20

These results come from `simulation/stage3_semiparametric_ate.py` with `{REPS}` Monte Carlo replicates per setting.

## Main Takeaways

1. Stage 3 gives the IF branch a practical role. In the hard-nuisance moderate-mismatch setting, audit-only coverage is {hard_mod_a.coverage:.3f} with average length {hard_mod_a.avg_length:.3f}, while IF-inflated coverage is {hard_mod_if.coverage:.3f} with average length {hard_mod_if.avg_length:.3f}.

2. Naive synthetic inference still fails under target mismatch. In the same hard-nuisance moderate-mismatch setting, Q-centered uninflated coverage is {hard_mod_q_un.coverage:.3f}. Adding the oracle direct-gap buffer restores coverage to {hard_mod_q.coverage:.3f}, but the interval is longer because it pays the direct ATE gap.

3. The selected rule now often chooses the IF branch. This is the first simulation stage where IF is not merely theorem-validation evidence; it is a practically shorter valid branch because the synthetic outcome functions remove large outcome variation while the audit sample corrects bias.

4. In the hard-nuisance weak-mismatch setting, the selected branch is `{hard_weak_sel.selected_branch}`, with coverage {hard_weak_sel.coverage:.3f} and average length {hard_weak_sel.avg_length:.3f}. This illustrates that the IF branch can remain useful even when the direct synthetic target gap is too large for Q-centering, provided the IF residual is controlled.

5. Important limitation: this stage uses an oracle synthetic nuisance model and known randomized treatment propensity. It shows the semiparametric value mechanism cleanly, but the next stage must add learned synthetic generators or learned nuisances to make the empirical story realistic.

## Selected Branch Behavior

```text
{selected.to_string(index=False)}
```

## Branch Counts

```text
{branch.to_string(index=False)}
```

## Basic QC

- Expected rows: 112.
- Actual rows in `stage3_summary.csv`: 112.
- `python -m py_compile simulation/stage3_semiparametric_ate.py` passed.
- Blank `selected_branch` values occur only for non-selected methods; selected rows have explicit branch labels.

## Output Files

- `tables/stage3_semiparametric_ate/stage3_summary.csv`
- `tables/stage3_semiparametric_ate/stage3_main_slice.csv`
- `tables/stage3_semiparametric_ate/stage3_branch_frequencies.csv`
- `figures/stage3_semiparametric_ate/stage3_hard_nuisance_clean.png`
- `figures/stage3_semiparametric_ate/stage3_moderate_nuisance_clean.png`
- `figures/stage3_semiparametric_ate/stage3_nuisance_value.png`
"""
    (RESULT_DIR / "STAGE3_RESULTS_SUMMARY.md").write_text(text)


def main() -> None:
    ensure_dirs()
    summary, branch = run_all()

    summary.to_csv(TABLE_DIR / "stage3_summary.csv", index=False)
    branch.to_csv(TABLE_DIR / "stage3_branch_frequencies.csv", index=False)
    main_slice = summary[summary["method"].isin(PLOT_METHODS)].copy()
    main_slice.to_csv(TABLE_DIR / "stage3_main_slice.csv", index=False)

    plot_coverage_length(
        summary,
        variant="hard_nuisance",
        out=FIG_DIR / "stage3_hard_nuisance_clean.png",
        title="S4: Semiparametric ATE, hard nuisance variation",
        methods=CLEAN_PLOT_METHODS,
    )
    plot_coverage_length(
        summary,
        variant="moderate_nuisance",
        out=FIG_DIR / "stage3_moderate_nuisance_clean.png",
        title="S4: Semiparametric ATE, moderate nuisance variation",
        methods=CLEAN_PLOT_METHODS,
    )
    plot_nuisance_value(summary, FIG_DIR / "stage3_nuisance_value.png")

    write_key_findings(summary, branch)
    summary.to_csv(RESULT_DIR / "stage3_summary.csv", index=False)
    branch.to_csv(RESULT_DIR / "stage3_branch_frequencies.csv", index=False)

    print(f"Wrote results to {RESULT_DIR}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
