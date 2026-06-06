#!/usr/bin/env python3
"""Stage 5 modern tabular-generator experiment.

This experiment trains SDV tabular generators on an independent biased source
distribution and evaluates audit-only, synthetic-centered, and audit-corrected
IF intervals for the real target ATE.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import os
import random
import time
import warnings

# The local macOS/anaconda torch stack can terminate SDV's torch-based
# synthesizers when BLAS/torch thread pools oversubscribe. Pinning these
# thread counts before torch is imported made CTGAN/TVAE stable in local QC.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t

try:
    import torch
    import sdv
    import ctgan
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
except Exception as exc:  # pragma: no cover - explicit runtime dependency check
    raise RuntimeError(
        "Stage 5 requires SDV with CTGAN/TVAE support. Install or repair the "
        "SDV stack before running this experiment."
    ) from exc

import stage3_semiparametric_ate as s3


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "simulation" / "results" / "stage5_modern_tabular_generator"
FIG_DIR = ROOT / "figures" / "stage5_modern_tabular_generator"
TABLE_DIR = ROOT / "tables" / "stage5_modern_tabular_generator"

ALPHA = 0.05
Z = norm.ppf(1 - ALPHA / 2)
SEED = 20260521
M_AUDIT = 200
N_TRAIN = 1000
M_CENTER = 5000
M_SYNTH_REF = 8000
AUDIT_REPS = 1000
GENERATOR_REPS = 2
IF_SE_MULTIPLIER = 1.03
RIDGE_LAMBDA = 1e-3
TVAE_EPOCHS = 100
CTGAN_EPOCHS = 100

SCENARIOS = ("strong_boundary", "moderate_boundary", "weak")
GENERATOR_ORDER = ("gaussian_copula", "tvae", "ctgan")
GENERATOR_LABELS = {
    "gaussian_copula": "Gaussian copula",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
}
METHOD_LABELS = {
    "audit_only": "Audit only",
    "synthetic_naive": "Synthetic naive",
    "q_inflated_oracle": "Q centered, oracle gap",
    "if_generator_rich": f"Audit driven IF, x{IF_SE_MULTIPLIER:.2f}",
    "selected": "Selected",
}
PLOT_METHODS = ("audit_only", "synthetic_naive", "q_inflated_oracle", "if_generator_rich", "selected")
PLOT_STYLES = {
    "audit_only": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "synthetic_naive": {"color": "#ff7f0e", "marker": "o", "linestyle": "-"},
    "q_inflated_oracle": {"color": "#d62728", "marker": "o", "linestyle": "-"},
    "if_generator_rich": {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
    "selected": {
        "color": "black",
        "marker": "D",
        "linestyle": "None",
        "markerfacecolor": "none",
        "markeredgewidth": 1.2,
    },
}

warnings.filterwarnings("ignore", message="We strongly recommend saving the metadata")
torch.set_num_threads(1)


@dataclass(frozen=True)
class Stage5Setting:
    gamma_label: str
    gamma: float

    @property
    def stage3_setting(self) -> s3.Setting:
        hard = next(v for v in s3.VARIANTS if v.name == "hard_nuisance")
        return s3.Setting(
            variant=hard,
            m=M_AUDIT,
            M=M_CENTER,
            M_multiplier=max(1, M_CENTER // M_AUDIT),
            gamma=self.gamma,
            gamma_label=self.gamma_label,
        )


def ensure_dirs() -> None:
    for path in (RESULT_DIR, FIG_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_settings() -> list[Stage5Setting]:
    gamma_lookup = dict(s3.mismatch_grid(M_AUDIT))
    return [Stage5Setting(label, gamma_lookup[label]) for label in SCENARIOS]


def draw_source_q_table(rng: np.random.Generator, n: int, setting: Stage5Setting) -> pd.DataFrame:
    st = setting.stage3_setting
    x = s3.draw_x(rng, (n,))
    a = rng.binomial(1, s3.PROPENSITY, size=n)
    mu = np.where(a == 1, s3.m1_q(x, st), s3.m0_q(x, st))
    y = mu + rng.normal(scale=s3.SIGMA, size=n)
    df = pd.DataFrame(x, columns=[f"x{k}" for k in range(s3.D)])
    df["A"] = a.astype(str)
    df["Y"] = y
    return df


def make_metadata(df: pd.DataFrame) -> Metadata:
    metadata = Metadata.detect_from_dataframe(df)
    metadata.update_column(column_name="A", sdtype="categorical")
    metadata.validate()
    return metadata


def build_synthesizer(generator_name: str, metadata: Metadata):
    if generator_name == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata)
    if generator_name == "tvae":
        return TVAESynthesizer(
            metadata,
            epochs=TVAE_EPOCHS,
            batch_size=500,
            verbose=False,
            enable_gpu=False,
        )
    if generator_name == "ctgan":
        return CTGANSynthesizer(
            metadata,
            epochs=CTGAN_EPOCHS,
            batch_size=500,
            discriminator_steps=1,
            verbose=False,
            enable_gpu=False,
        )
    raise ValueError(f"Unknown generator: {generator_name}")


def parse_treatment(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip()
    mapped = raw.map({"0": 0, "1": 1, "0.0": 0, "1.0": 1})
    numeric = pd.to_numeric(raw, errors="coerce")
    fallback = numeric.round().clip(0, 1)
    parsed = mapped.where(mapped.notna(), fallback)
    return parsed.astype("float")


def clean_synthetic_sample(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"x{k}" for k in range(s3.D)] + ["A", "Y"]
    out = df.loc[:, cols].copy()
    for col in [f"x{k}" for k in range(s3.D)] + ["Y"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["A"] = parse_treatment(out["A"])
    out = out.dropna()
    out["A"] = out["A"].astype(int)
    return out


def sample_clean(synthesizer, n: int, min_each_arm: int = 50) -> pd.DataFrame:
    pieces = []
    total = 0
    tries = 0
    while total < n and tries < 5:
        raw = synthesizer.sample(n - total)
        clean = clean_synthetic_sample(raw)
        pieces.append(clean)
        total += len(clean)
        tries += 1
    df = pd.concat(pieces, ignore_index=True).iloc[:n].copy()
    counts = df["A"].value_counts()
    if len(df) < n or counts.get(0, 0) < min_each_arm or counts.get(1, 0) < min_each_arm:
        raise RuntimeError(
            f"Generator sample has insufficient valid rows or treatment balance: "
            f"n={len(df)}, counts={counts.to_dict()}"
        )
    return df


def feature_map(x: np.ndarray, kind: str = "rich") -> np.ndarray:
    if kind == "linear":
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    if kind == "rich":
        return np.concatenate(
            [
                np.ones((x.shape[0], 1)),
                x,
                np.sin(x[:, [0]]),
                x[:, [1]] ** 2 - 1.0,
                x[:, [3]] * x[:, [4]],
                np.tanh(x[:, [0]] + x[:, [1]]),
            ],
            axis=1,
        )
    raise ValueError(f"Unknown feature kind: {kind}")


def fit_ridge(x: np.ndarray, y: np.ndarray, kind: str = "rich") -> np.ndarray:
    phi = feature_map(x, kind)
    xtx = phi.T @ phi / len(y)
    xty = phi.T @ y / len(y)
    penalty = RIDGE_LAMBDA * np.eye(phi.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def predict_ridge(x: np.ndarray, beta: np.ndarray, kind: str = "rich") -> np.ndarray:
    return feature_map(x, kind) @ beta


def fit_outcome_nuisances(synth_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = synth_df[[f"x{k}" for k in range(s3.D)]].to_numpy()
    a = synth_df["A"].to_numpy()
    y = synth_df["Y"].to_numpy()
    beta0 = fit_ridge(x[a == 0], y[a == 0], "rich")
    beta1 = fit_ridge(x[a == 1], y[a == 1], "rich")
    return beta0, beta1


def diff_mean_stats(df: pd.DataFrame) -> dict[str, float]:
    treated = df[df["A"] == 1]["Y"].to_numpy()
    control = df[df["A"] == 0]["Y"].to_numpy()
    p1 = len(treated) / len(df)
    p0 = 1.0 - p1
    theta = float(treated.mean() - control.mean())
    sd = float(np.sqrt(treated.var(ddof=1) / p1 + control.var(ddof=1) / p0))
    return {
        "theta": theta,
        "sd_per_sqrt_n": sd,
        "treated_rate": p1,
        "treated_n": int(len(treated)),
        "control_n": int(len(control)),
    }


def audit_arrays(rng: np.random.Generator, setting: Stage5Setting) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = setting.stage3_setting
    x = s3.draw_x(rng, (AUDIT_REPS, M_AUDIT))
    a = rng.binomial(1, s3.PROPENSITY, size=(AUDIT_REPS, M_AUDIT))
    y = s3.draw_outcomes_p(rng, x, a, st)
    return x, a, y


def interval(center: np.ndarray, se: np.ndarray, crit: float, inflation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return center - crit * se - inflation, center + crit * se + inflation


def summarize_method(
    *,
    setting: Stage5Setting,
    generator_name: str,
    generator_rep: int,
    selected_branch: str,
    method: str,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    theta_p: float,
    theta_generator: float,
    direct_gap: float,
    source_gap: float,
    fit_seconds: float,
) -> dict[str, float | str | int]:
    length = upper - lower
    cover = (lower <= theta_p) & (theta_p <= upper)
    return {
        "simulation": "S7_modern_tabular_generator",
        "generator": generator_name,
        "generator_label": GENERATOR_LABELS[generator_name],
        "generator_rep": generator_rep,
        "gamma_label": setting.gamma_label,
        "gamma": setting.gamma,
        "m": M_AUDIT,
        "n_train": N_TRAIN,
        "M_center": M_CENTER,
        "M_synth_ref": M_SYNTH_REF,
        "audit_reps": AUDIT_REPS,
        "theta_p": theta_p,
        "theta_generator": theta_generator,
        "direct_gap": direct_gap,
        "source_gap": source_gap,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "selected_branch": selected_branch if method == "selected" else "",
        "coverage": float(cover.mean()),
        "avg_length": float(length.mean()),
        "median_length": float(np.median(length)),
        "bias": float(point.mean() - theta_p),
        "emp_sd": float(point.std(ddof=1)),
        "fit_seconds": fit_seconds,
    }


def evaluate_generator_fit(
    *,
    setting: Stage5Setting,
    generator_name: str,
    generator_rep: int,
    synthesizer,
    fit_seconds: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, float | str | int]], dict[str, float | str | int]]:
    pop = s3.population_quantities(setting.stage3_setting)
    theta_p = float(pop["theta_p"])
    source_gap = float(pop["direct_gap"])

    synth_ref = sample_clean(synthesizer, M_SYNTH_REF)
    synth_stats = diff_mean_stats(synth_ref)
    theta_generator = synth_stats["theta"]
    direct_gap = abs(theta_p - theta_generator)
    synth_center_se = synth_stats["sd_per_sqrt_n"] / np.sqrt(M_CENTER)

    beta0, beta1 = fit_outcome_nuisances(synth_ref)

    x_p, a_p, y_p = audit_arrays(rng, setting)
    treated = a_p == 1
    control = ~treated
    n1 = treated.sum(axis=1)
    n0 = control.sum(axis=1)
    y1_sum = np.where(treated, y_p, 0.0).sum(axis=1)
    y0_sum = np.where(control, y_p, 0.0).sum(axis=1)
    audit_point = y1_sum / n1 - y0_sum / n0
    y1_var = np.array([np.var(y_p[i, treated[i]], ddof=1) for i in range(AUDIT_REPS)])
    y0_var = np.array([np.var(y_p[i, control[i]], ddof=1) for i in range(AUDIT_REPS)])
    audit_se = np.sqrt(y1_var / n1 + y0_var / n0)
    crit_m = t.ppf(1 - ALPHA / 2, df=M_AUDIT - 2)

    x_flat = x_p.reshape((-1, s3.D))
    q0 = predict_ridge(x_flat, beta0, "rich").reshape((AUDIT_REPS, M_AUDIT))
    q1 = predict_ridge(x_flat, beta1, "rich").reshape((AUDIT_REPS, M_AUDIT))
    pseudo = q1 - q0 + a_p / s3.PROPENSITY * (y_p - q1) - (1 - a_p) / (1 - s3.PROPENSITY) * (y_p - q0)
    if_point = pseudo.mean(axis=1)
    if_se = pseudo.std(axis=1, ddof=1) / np.sqrt(M_AUDIT)

    synthetic_point = rng.normal(theta_generator, synth_center_se, size=AUDIT_REPS)
    synthetic_se = np.full(AUDIT_REPS, synth_center_se)

    intervals = {}
    intervals["audit_only"] = (audit_point, *interval(audit_point, audit_se, crit_m))
    intervals["synthetic_naive"] = (synthetic_point, *interval(synthetic_point, synthetic_se, Z))
    intervals["q_inflated_oracle"] = (
        synthetic_point,
        *interval(synthetic_point, synthetic_se, Z, direct_gap),
    )
    intervals["if_generator_rich"] = (
        if_point,
        *interval(if_point, IF_SE_MULTIPLIER * if_se, crit_m),
    )

    avg_lengths = {key: float((vals[2] - vals[1]).mean()) for key, vals in intervals.items()}
    candidate_to_method = {
        "A": "audit_only",
        "Q": "q_inflated_oracle",
        "IF": "if_generator_rich",
    }
    selected_branch = min(candidate_to_method, key=lambda branch: avg_lengths[candidate_to_method[branch]])
    intervals["selected"] = intervals[candidate_to_method[selected_branch]]

    rows = [
        summarize_method(
            setting=setting,
            generator_name=generator_name,
            generator_rep=generator_rep,
            selected_branch=selected_branch,
            method=method,
            point=vals[0],
            lower=vals[1],
            upper=vals[2],
            theta_p=theta_p,
            theta_generator=theta_generator,
            direct_gap=direct_gap,
            source_gap=source_gap,
            fit_seconds=fit_seconds,
        )
        for method, vals in intervals.items()
    ]
    gen_row = {
        "generator": generator_name,
        "generator_label": GENERATOR_LABELS[generator_name],
        "generator_rep": generator_rep,
        "gamma_label": setting.gamma_label,
        "gamma": setting.gamma,
        "theta_p": theta_p,
        "theta_generator": theta_generator,
        "direct_gap": direct_gap,
        "source_gap": source_gap,
        "treated_rate_synth": synth_stats["treated_rate"],
        "treated_n_synth": synth_stats["treated_n"],
        "control_n_synth": synth_stats["control_n"],
        "selected_branch": selected_branch,
        "fit_seconds": fit_seconds,
    }
    return rows, gen_row


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["generator", "generator_label", "gamma_label", "method", "method_label"]
    value_cols = [
        "gamma",
        "source_gap",
        "direct_gap",
        "coverage",
        "avg_length",
        "median_length",
        "bias",
        "emp_sd",
        "fit_seconds",
    ]
    summary = raw.groupby(group_cols, as_index=False)[value_cols].mean()
    sd = (
        raw.groupby(group_cols, as_index=False)[["coverage", "avg_length", "direct_gap"]]
        .std()
        .rename(
            columns={
                "coverage": "coverage_sd_across_generator_reps",
                "avg_length": "avg_length_sd_across_generator_reps",
                "direct_gap": "direct_gap_sd_across_generator_reps",
            }
        )
    )
    return summary.merge(sd, on=group_cols, how="left")


def selected_summary(raw: pd.DataFrame) -> pd.DataFrame:
    selected = raw[raw["method"] == "selected"].copy()
    return (
        selected.groupby(["generator", "generator_label", "gamma_label", "selected_branch"], as_index=False)
        .size()
        .rename(columns={"size": "generator_reps"})
    )


def scenario_positions() -> dict[str, int]:
    return {label: i for i, label in enumerate(SCENARIOS)}


def plot_coverage_length(summary: pd.DataFrame, out: Path) -> None:
    pos = scenario_positions()
    fig, axes = plt.subplots(2, len(GENERATOR_ORDER), figsize=(14, 7.5), sharex=True)
    for col, generator in enumerate(GENERATOR_ORDER):
        gen = summary[summary["generator"] == generator]
        for method in PLOT_METHODS:
            d = gen[gen["method"] == method].copy()
            d["x_pos"] = d["gamma_label"].map(pos)
            d = d.sort_values("x_pos")
            style = PLOT_STYLES[method]
            axes[0, col].plot(d["x_pos"], d["coverage"], label=METHOD_LABELS[method], **style)
            axes[1, col].plot(d["x_pos"], d["avg_length"], label=METHOD_LABELS[method], **style)
        axes[0, col].axhline(0.95, color="black", linewidth=0.8)
        axes[0, col].set_title(GENERATOR_LABELS[generator])
        axes[0, col].set_ylim(0, 1.03)
        axes[1, col].set_xticks(list(pos.values()), list(pos.keys()), rotation=20, ha="right")
        axes[1, col].set_xlabel("Source mismatch regime")
        for row in range(2):
            axes[row, col].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Empirical coverage")
    axes[1, 0].set_ylabel("Average interval length")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Modern tabular generators with audit correction")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_generator_diagnostics(summary: pd.DataFrame, gen_summary: pd.DataFrame, out: Path) -> None:
    pos = scenario_positions()
    diag = (
        gen_summary.groupby(["generator", "generator_label", "gamma_label"], as_index=False)[
            ["direct_gap", "source_gap", "treated_rate_synth"]
        ].mean()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for generator in GENERATOR_ORDER:
        d = diag[diag["generator"] == generator].copy()
        d["x_pos"] = d["gamma_label"].map(pos)
        d = d.sort_values("x_pos")
        axes[0].plot(d["x_pos"], d["source_gap"], marker="x", linestyle="--", label=f"Source gap, {GENERATOR_LABELS[generator]}")
        axes[0].plot(d["x_pos"], d["direct_gap"], marker="o", label=f"Generator gap, {GENERATOR_LABELS[generator]}")

        audit = summary[(summary["generator"] == generator) & (summary["method"] == "audit_only")].copy()
        ifm = summary[(summary["generator"] == generator) & (summary["method"] == "if_generator_rich")].copy()
        audit["x_pos"] = audit["gamma_label"].map(pos)
        ifm["x_pos"] = ifm["gamma_label"].map(pos)
        audit = audit.sort_values("x_pos")
        ifm = ifm.sort_values("x_pos")
        ratio = audit["avg_length"].to_numpy() / ifm["avg_length"].to_numpy()
        axes[1].plot(audit["x_pos"], ratio, marker="o", label=GENERATOR_LABELS[generator])
    for ax in axes:
        ax.set_xticks(list(pos.values()), list(pos.keys()), rotation=20, ha="right")
        ax.set_xlabel("Source mismatch regime")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ATE gap")
    axes[1].set_ylabel("Audit-only length / IF length")
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=8)
    fig.suptitle("Stage 5 diagnostics: generator target gap and IF value")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(raw: pd.DataFrame, summary: pd.DataFrame, gen_summary: pd.DataFrame, branch: pd.DataFrame) -> None:
    versions = dependency_versions()
    mod = summary[
        (summary["gamma_label"] == "moderate_boundary")
        & (summary["method"].isin(["audit_only", "q_inflated_oracle", "if_generator_rich", "selected"]))
    ].copy()
    mod = mod.sort_values(["generator", "method"])
    if_rows = summary[summary["method"] == "if_generator_rich"].copy()
    audit_rows = summary[summary["method"] == "audit_only"].copy()
    min_if_cov = if_rows["coverage"].min()
    ratio_rows = []
    for (generator, gamma_label), audit_row in audit_rows.groupby(["generator", "gamma_label"]):
        if_row = if_rows[(if_rows["generator"] == generator) & (if_rows["gamma_label"] == gamma_label)].iloc[0]
        ratio_rows.append(float(audit_row.iloc[0]["avg_length"] / if_row["avg_length"]))
    mean_audit_if_ratio = float(np.mean(ratio_rows))
    if_shorter_rows = int(np.sum(np.array(ratio_rows) > 1.0))
    total_ratio_rows = len(ratio_rows)

    text = f"""# Stage 5 Modern Tabular Generator Results Summary

Created: 2026-05-21

This experiment uses SDV tabular generators trained on independent biased source data. The audit sample is always drawn independently from the real target distribution `P`.

## Software

```text
{versions.to_string(index=False)}
```

## Design

- Real target: hard-nuisance randomized ATE design from Stage 3.
- Source data for generator training: biased `Q` distribution with controlled source mismatch.
- Generators: Gaussian copula, TVAE, and CTGAN from SDV.
- Generator training replicates per generator/regime: `{GENERATOR_REPS}`.
- Training rows per generator fit: `{N_TRAIN}`.
- Synthetic reference rows per fitted generator: `{M_SYNTH_REF}`.
- Audit Monte Carlo replicates per fitted generator: `{AUDIT_REPS}` with audit size `m={M_AUDIT}`.
- Neural generator training epochs: TVAE `{TVAE_EPOCHS}`, CTGAN `{CTGAN_EPOCHS}`.
- IF interval uses the same small finite-sample calibration as Stage 4: standard-error multiplier `{IF_SE_MULTIPLIER:.2f}`.

## Main Takeaways

1. The audit-corrected IF branch remains coverage-stable with modern tabular generators. Across generator/regime averages, the minimum calibrated IF coverage is `{min_if_cov:.3f}`.

2. Synthetic-naive inference remains unreliable when the fitted generator has a target gap. This supports the paper's warning that realistic synthetic data should not be treated as real target data without correction or discrepancy accounting.

3. Oracle-gap Q inflation is conservative but often much wider when the fitted generator target gap grows. This is expected because direct synthetic centering pays the first-order target gap.

4. Modern generators do not automatically improve efficiency. The calibrated IF branch is shorter than audit-only in `{if_shorter_rows}` of `{total_ratio_rows}` generator/regime averages; on average, the audit-only length divided by the calibrated IF length is `{mean_audit_if_ratio:.2f}`. This means the modern-generator nuisance functions are often not good enough to reduce variance, even though audit correction keeps coverage stable.

5. The fitted generator target gap is not identical to the source gap. This is useful for the paper: the formal theory should condition on the fitted generator distribution, and the discrepancy radius must absorb both source mismatch and generator-fitting error.

## Moderate-Boundary Rows

```text
{mod[['generator_label', 'method_label', 'coverage', 'avg_length', 'direct_gap', 'bias']].to_string(index=False)}
```

## Selected Branch Counts

```text
{branch.to_string(index=False)}
```

## Limitations

- This is a modern-generator simulation, not a real-data example.
- Generator training replicates are intentionally limited because CTGAN training is computationally expensive.
- Q-inflated intervals use an oracle fitted-generator target gap for benchmarking. A real analysis would need a sensitivity radius, external certificate, or audit-based upper confidence bound.

## Output Files

- `simulation/stage5_modern_tabular_generator.py`
- `tables/stage5_modern_tabular_generator/stage5_summary.csv`
- `tables/stage5_modern_tabular_generator/stage5_raw_by_generator_rep.csv`
- `tables/stage5_modern_tabular_generator/stage5_generator_replicates.csv`
- `tables/stage5_modern_tabular_generator/stage5_selected_branch_counts.csv`
- `figures/stage5_modern_tabular_generator/stage5_generator_coverage_length.png`
- `figures/stage5_modern_tabular_generator/stage5_generator_diagnostics.png`
"""
    (RESULT_DIR / "STAGE5_RESULTS_SUMMARY.md").write_text(text)
    versions.to_csv(TABLE_DIR / "stage5_dependency_versions.csv", index=False)
    versions.to_csv(RESULT_DIR / "stage5_dependency_versions.csv", index=False)


def dependency_versions() -> pd.DataFrame:
    try:
        import torchvision
    except Exception:  # pragma: no cover - optional diagnostic
        torchvision = None
    try:
        import torchaudio
    except Exception:  # pragma: no cover - optional diagnostic
        torchaudio = None
    return pd.DataFrame(
        [
            {"package": "python", "version": "3.12.7"},
            {"package": "sdv", "version": sdv.__version__},
            {"package": "ctgan", "version": ctgan.__version__},
            {"package": "torch", "version": torch.__version__},
            {"package": "torchvision", "version": getattr(torchvision, "__version__", "not_available")},
            {"package": "torchaudio", "version": getattr(torchaudio, "__version__", "not_available")},
            {"package": "numpy", "version": np.__version__},
            {"package": "pandas", "version": pd.__version__},
        ]
    )


def main() -> None:
    ensure_dirs()
    raw_rows: list[dict[str, float | str | int]] = []
    gen_rows: list[dict[str, float | str | int]] = []

    settings = build_settings()
    metadata_saved = False
    for setting in settings:
        for generator_name in GENERATOR_ORDER:
            for gen_rep in range(GENERATOR_REPS):
                seed = SEED + 10000 * SCENARIOS.index(setting.gamma_label) + 1000 * GENERATOR_ORDER.index(generator_name) + gen_rep
                set_global_seed(seed)
                rng = np.random.default_rng(seed)
                train_df = draw_source_q_table(rng, N_TRAIN, setting)
                metadata = make_metadata(train_df)
                if not metadata_saved:
                    metadata_path = RESULT_DIR / "stage5_sdv_metadata.json"
                    if metadata_path.exists():
                        metadata_path.unlink()
                    metadata.save_to_json(metadata_path)
                    metadata_saved = True

                synthesizer = build_synthesizer(generator_name, metadata)
                start = time.time()
                synthesizer.fit(train_df)
                fit_seconds = time.time() - start

                rows, gen_row = evaluate_generator_fit(
                    setting=setting,
                    generator_name=generator_name,
                    generator_rep=gen_rep,
                    synthesizer=synthesizer,
                    fit_seconds=fit_seconds,
                    rng=rng,
                )
                raw_rows.extend(rows)
                gen_rows.append(gen_row)
                print(
                    f"finished {setting.gamma_label} {generator_name} rep={gen_rep} "
                    f"fit_seconds={fit_seconds:.1f} direct_gap={gen_row['direct_gap']:.3f}"
                )

    raw = pd.DataFrame(raw_rows)
    gen_summary = pd.DataFrame(gen_rows)
    summary = aggregate(raw)
    branch = selected_summary(raw)

    raw.to_csv(TABLE_DIR / "stage5_raw_by_generator_rep.csv", index=False)
    gen_summary.to_csv(TABLE_DIR / "stage5_generator_replicates.csv", index=False)
    summary.to_csv(TABLE_DIR / "stage5_summary.csv", index=False)
    branch.to_csv(TABLE_DIR / "stage5_selected_branch_counts.csv", index=False)
    raw.to_csv(RESULT_DIR / "stage5_raw_by_generator_rep.csv", index=False)
    gen_summary.to_csv(RESULT_DIR / "stage5_generator_replicates.csv", index=False)
    summary.to_csv(RESULT_DIR / "stage5_summary.csv", index=False)
    branch.to_csv(RESULT_DIR / "stage5_selected_branch_counts.csv", index=False)

    plot_coverage_length(summary, FIG_DIR / "stage5_generator_coverage_length.png")
    plot_generator_diagnostics(summary, gen_summary, FIG_DIR / "stage5_generator_diagnostics.png")
    write_summary(raw, summary, gen_summary, branch)

    print(f"Wrote results to {RESULT_DIR}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
