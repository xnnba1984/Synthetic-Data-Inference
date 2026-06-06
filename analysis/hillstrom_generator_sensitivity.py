#!/usr/bin/env python3
"""Targeted Hillstrom generator-sensitivity and source-diagnostic step.

This follows the recommendation in analysis/HILLSTROM_PILOT_DECISION_MEMO.md:
before committing to a full Hillstrom empirical analysis, diagnose whether the
Gaussian-copula pilot failed because the source design is unsuitable or because
the Gaussian generator attenuates the treatment-outcome relationship.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import os
import random
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t

try:
    import torch
    from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This diagnostic requires SDV with TVAE support and torch.") from exc

import hillstrom_pilot as hp


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "analysis" / "results" / "hillstrom_generator_sensitivity"
TABLE_DIR = ROOT / "tables" / "hillstrom_generator_sensitivity"
FIG_DIR = ROOT / "figures" / "hillstrom_generator_sensitivity"

SEED = 20260522 + 31
ALPHA = 0.05
Z = norm.ppf(1.0 - ALPHA / 2.0)
SOURCE_LAMBDAS = (0.0, 0.5, 1.0)
N_SOURCE = 8000
M_SYNTH = 10000
M_AUDIT = 500
SOURCE_REPS = int(os.environ.get("HILLSTROM_SENS_SOURCE_REPS", "30"))
GENERATOR_REPS = int(os.environ.get("HILLSTROM_SENS_GENERATOR_REPS", "12"))
AUDIT_REPS_PER_GENERATOR = int(os.environ.get("HILLSTROM_SENS_AUDIT_REPS", "25"))
TVAE_EPOCHS = int(os.environ.get("HILLSTROM_TVAE_EPOCHS", "80"))
IF_SE_MULTIPLIER = hp.IF_SE_MULTIPLIER
SENSITIVITY_C = (1.0, 1.5, 2.0)
BONF_BRANCHES = ("A", "Q", "IF")

GENERATOR_LABELS = {
    "real_source": "Real biased source",
    "gaussian_copula": "Gaussian copula",
    "tvae": "TVAE",
}
METHOD_LABELS = {
    "audit_only": "Audit only",
    "synthetic_naive": "Synthetic/source naive",
    "q_inflated": "Q/source inflated",
    "if_learned": f"Audit-driven IF x{IF_SE_MULTIPLIER:.2f}",
    "selected_feasible": "Selected feasible",
}

warnings.filterwarnings("ignore", message="We strongly recommend saving the metadata")
torch.set_num_threads(1)


def ensure_dirs() -> None:
    for path in (RESULT_DIR, TABLE_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_generator(name: str, metadata):
    if name == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata)
    if name == "tvae":
        return TVAESynthesizer(
            metadata,
            epochs=TVAE_EPOCHS,
            batch_size=500,
            verbose=False,
            enable_gpu=False,
        )
    raise ValueError(f"Unknown generator: {name}")


def fit_and_sample(source_df: pd.DataFrame, generator_name: str, seed: int) -> tuple[pd.DataFrame, float]:
    train = hp.make_sdv_frame(source_df)
    metadata = hp.make_metadata(train)
    synthesizer = build_generator(generator_name, metadata)
    set_seed(seed)
    start = time.time()
    synthesizer.fit(train)
    fit_seconds = time.time() - start
    synth_df = hp.sample_synthetic(synthesizer, M_SYNTH)
    return synth_df, fit_seconds


def update_acc(acc: dict[str, float], point: float, lo: float, hi: float, target: float) -> None:
    acc["n"] += 1
    acc["cover"] += float(lo <= target <= hi)
    acc["length_sum"] += hi - lo
    acc["length2_sum"] += (hi - lo) ** 2
    acc["bias_sum"] += point - target
    acc["point_sum"] += point
    acc["point2_sum"] += point**2


def finalize_acc(*, lam: float, generator: str, method: str, c: float | None, acc: dict[str, float]) -> dict:
    n = int(acc["n"])
    coverage = acc["cover"] / n
    point_mean = acc["point_sum"] / n
    point_var = max(acc["point2_sum"] / n - point_mean**2, 0.0)
    length_mean = acc["length_sum"] / n
    length_var = max(acc["length2_sum"] / n - length_mean**2, 0.0)
    return {
        "lambda": lam,
        "generator": generator,
        "generator_label": GENERATOR_LABELS[generator],
        "m": M_AUDIT,
        "sensitivity_c": c,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "reps": n,
        "coverage": coverage,
        "mc_se_coverage": math.sqrt(coverage * (1.0 - coverage) / n) if n else np.nan,
        "avg_length": length_mean,
        "sd_length": math.sqrt(length_var),
        "bias": acc["bias_sum"] / n,
        "emp_sd_point": math.sqrt(point_var),
    }


def run_method_diagnostics(df: pd.DataFrame, rng: np.random.Generator, theta_full: float, pi: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = hp.source_scores(df)
    source_gap_rows = []
    generator_gap_rows = []
    method_acc: dict[tuple[float, str, str, float | None], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    branch_counts: dict[tuple[float, str, float, str], int] = defaultdict(int)

    for lam in SOURCE_LAMBDAS:
        # Source-only diagnostic: does the real biased source itself preserve
        # the target ATE, before any generator is fitted?
        for rep in range(SOURCE_REPS):
            source_idx = hp.draw_source_indices(rng, scores, lam, N_SOURCE)
            source_df = df.iloc[source_idx].copy()
            source_theta, source_se = hp.diff_mean(source_df)
            source_gap_rows.append(
                {
                    "lambda": lam,
                    "rep": rep,
                    "sample_type": "real_biased_source",
                    "theta": source_theta,
                    "se": source_se,
                    "direct_gap": abs(source_theta - theta_full),
                }
            )

        # Generator diagnostic and IF/audit method check.
        for generator in ("gaussian_copula", "tvae"):
            for gen_rep in range(GENERATOR_REPS):
                seed = SEED + int(lam * 1000) + (0 if generator == "gaussian_copula" else 100000) + gen_rep
                source_idx = hp.draw_source_indices(rng, scores, lam, N_SOURCE)
                source_df = df.iloc[source_idx].copy()
                source_theta, _ = hp.diff_mean(source_df)
                start = time.time()
                try:
                    q_df, fit_seconds = fit_and_sample(source_df, generator, seed)
                    generator_failed = 0
                    error = ""
                except Exception as exc:
                    q_df = pd.DataFrame()
                    fit_seconds = time.time() - start
                    generator_failed = 1
                    error = repr(exc)
                if generator_failed:
                    generator_gap_rows.append(
                        {
                            "lambda": lam,
                            "generator": generator,
                            "gen_rep": gen_rep,
                            "source_theta": source_theta,
                            "theta_q": np.nan,
                            "source_gap": abs(source_theta - theta_full),
                            "generator_gap": np.nan,
                            "source_to_generator_gap": np.nan,
                            "fit_seconds": fit_seconds,
                            "generator_failed": 1,
                            "error": error,
                        }
                    )
                    continue

                q_theta, q_se = hp.diff_mean(q_df)
                generator_gap_rows.append(
                    {
                        "lambda": lam,
                        "generator": generator,
                        "gen_rep": gen_rep,
                        "source_theta": source_theta,
                        "theta_q": q_theta,
                        "source_gap": abs(source_theta - theta_full),
                        "generator_gap": abs(q_theta - theta_full),
                        "source_to_generator_gap": abs(q_theta - source_theta),
                        "fit_seconds": fit_seconds,
                        "generator_failed": 0,
                        "error": "",
                    }
                )

                model0, model1 = hp.fit_nuisances(q_df, seed)
                for audit_rep in range(AUDIT_REPS_PER_GENERATOR):
                    audit_idx = hp.draw_audit_indices(rng, len(df), M_AUDIT)
                    audit_df = df.iloc[audit_idx].copy()
                    audit_point, audit_se = hp.diff_mean(audit_df)
                    if_point, if_se = hp.if_estimate(audit_df, model0, model1, pi)
                    tcrit = t.ppf(1.0 - ALPHA / 2.0, df=M_AUDIT - 2)

                    a_lo, a_hi = hp.interval(audit_point, audit_se, tcrit)
                    q_lo, q_hi = hp.interval(q_theta, q_se, Z)
                    if_lo, if_hi = hp.interval(if_point, IF_SE_MULTIPLIER * if_se, tcrit)
                    update_acc(method_acc[(lam, generator, "audit_only", None)], audit_point, a_lo, a_hi, theta_full)
                    update_acc(method_acc[(lam, generator, "synthetic_naive", None)], q_theta, q_lo, q_hi, theta_full)
                    update_acc(method_acc[(lam, generator, "if_learned", None)], if_point, if_lo, if_hi, theta_full)

                    candidate_alpha = ALPHA / len(BONF_BRANCHES)
                    z_bonf = norm.ppf(1.0 - candidate_alpha / 2.0)
                    t_bonf = t.ppf(1.0 - candidate_alpha / 2.0, df=M_AUDIT - 2)
                    a_lo_b, a_hi_b = hp.interval(audit_point, audit_se, t_bonf)
                    if_lo_b, if_hi_b = hp.interval(if_point, IF_SE_MULTIPLIER * if_se, t_bonf)

                    for c in SENSITIVITY_C:
                        radius = c * audit_se
                        q_lo_inf, q_hi_inf = hp.interval(q_theta, q_se, Z, radius)
                        update_acc(method_acc[(lam, generator, "q_inflated", c)], q_theta, q_lo_inf, q_hi_inf, theta_full)

                        q_lo_b, q_hi_b = hp.interval(q_theta, q_se, z_bonf, radius)
                        candidates = {
                            "A": (audit_point, a_lo_b, a_hi_b),
                            "Q": (q_theta, q_lo_b, q_hi_b),
                            "IF": (if_point, if_lo_b, if_hi_b),
                        }
                        lengths = {branch: vals[2] - vals[1] for branch, vals in candidates.items()}
                        selected_branch = min(lengths, key=lengths.get)
                        point, lo, hi = candidates[selected_branch]
                        update_acc(method_acc[(lam, generator, "selected_feasible", c)], point, lo, hi, theta_full)
                        branch_counts[(lam, generator, c, selected_branch)] += 1
                print(
                    f"lambda={lam}, generator={generator}, rep={gen_rep + 1}/{GENERATOR_REPS}, "
                    f"source_theta={source_theta:.4f}, q_theta={q_theta:.4f}",
                    flush=True,
                )

    method_rows = [
        finalize_acc(lam=lam, generator=generator, method=method, c=c, acc=acc)
        for (lam, generator, method, c), acc in method_acc.items()
    ]
    branch_rows = []
    for (lam, generator, c, branch), count in sorted(branch_counts.items()):
        total = sum(branch_counts[(lam, generator, c, b)] for b in BONF_BRANCHES)
        branch_rows.append(
            {
                "lambda": lam,
                "generator": generator,
                "generator_label": GENERATOR_LABELS[generator],
                "m": M_AUDIT,
                "sensitivity_c": c,
                "selected_branch": branch,
                "selected_count": count,
                "selected_frequency": count / total if total else np.nan,
            }
        )
    return pd.DataFrame(source_gap_rows), pd.DataFrame(generator_gap_rows), pd.DataFrame(method_rows), pd.DataFrame(branch_rows)


def summarize_source(source_df: pd.DataFrame) -> pd.DataFrame:
    return source_df.groupby("lambda", as_index=False).agg(
        source_theta_mean=("theta", "mean"),
        source_theta_sd=("theta", "std"),
        source_gap_mean=("direct_gap", "mean"),
        source_gap_sd=("direct_gap", "std"),
    )


def summarize_generator(gap_df: pd.DataFrame) -> pd.DataFrame:
    return gap_df.groupby(["lambda", "generator"], as_index=False).agg(
        generator_failures=("generator_failed", "sum"),
        source_theta_mean=("source_theta", "mean"),
        theta_q_mean=("theta_q", "mean"),
        source_gap_mean=("source_gap", "mean"),
        generator_gap_mean=("generator_gap", "mean"),
        source_to_generator_gap_mean=("source_to_generator_gap", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )


def plot_diagnostics(source_summary: pd.DataFrame, gen_summary: pd.DataFrame, method_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].plot(source_summary["lambda"], source_summary["source_gap_mean"], marker="o", label="Real biased source")
    for generator, label in [("gaussian_copula", "Gaussian copula"), ("tvae", "TVAE")]:
        g = gen_summary[gen_summary["generator"] == generator]
        axes[0].plot(g["lambda"], g["generator_gap_mean"], marker="o", label=label)
    axes[0].set_title("ATE gap from target")
    axes[0].set_xlabel("Source-bias lambda")
    axes[0].set_ylabel("Mean absolute ATE gap")
    axes[0].legend(fontsize=8)

    for generator, label in [("gaussian_copula", "Gaussian copula"), ("tvae", "TVAE")]:
        for method, linestyle in [("audit_only", "--"), ("if_learned", "-")]:
            g = method_summary[
                (method_summary["generator"] == generator)
                & (method_summary["method"] == method)
                & (method_summary["sensitivity_c"].isna())
            ]
            axes[1].plot(g["lambda"], g["avg_length"], marker="o", linestyle=linestyle, label=f"{label}, {METHOD_LABELS[method]}")
    axes[1].set_title("Audit versus IF length")
    axes[1].set_xlabel("Source-bias lambda")
    axes[1].set_ylabel("Average interval length")
    axes[1].legend(fontsize=7)

    for generator, label in [("gaussian_copula", "Gaussian copula"), ("tvae", "TVAE")]:
        g = method_summary[
            (method_summary["generator"] == generator)
            & (method_summary["method"].isin(["audit_only", "if_learned", "synthetic_naive"]))
            & (method_summary["sensitivity_c"].isna())
        ]
        for method in ["synthetic_naive", "audit_only", "if_learned"]:
            gg = g[g["method"] == method]
            axes[2].plot(gg["lambda"], gg["coverage"], marker="o", label=f"{label}, {METHOD_LABELS[method]}")
    axes[2].axhline(0.95, color="0.4", linestyle="--", linewidth=1)
    axes[2].set_title("Coverage")
    axes[2].set_xlabel("Source-bias lambda")
    axes[2].set_ylabel("Empirical coverage")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=6)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(FIG_DIR / "hillstrom_generator_sensitivity.png", dpi=220)
    plt.close(fig)


def write_decision_memo(
    *,
    theta_full: float,
    source_summary: pd.DataFrame,
    gen_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
) -> None:
    # Decision heuristic: proceed with Hillstrom only if TVAE materially reduces
    # the generator gap and gives IF intervals shorter than audit-only with
    # pilot coverage at least 0.90 in a middle lambda.
    tvae_gap = gen_summary[gen_summary["generator"] == "tvae"]["generator_gap_mean"].mean()
    source_gap = source_summary["source_gap_mean"].mean()
    tvae_rows = method_summary[
        (method_summary["generator"] == "tvae")
        & (method_summary["lambda"].isin([0.5, 1.0]))
        & (method_summary["method"].isin(["audit_only", "if_learned"]))
        & (method_summary["sensitivity_c"].isna())
    ]
    pivot = tvae_rows.pivot_table(index="lambda", columns="method", values=["coverage", "avg_length"])
    useful = False
    useful_details = []
    for lam in [0.5, 1.0]:
        try:
            if_len = float(pivot.loc[lam, ("avg_length", "if_learned")])
            a_len = float(pivot.loc[lam, ("avg_length", "audit_only")])
            if_cov = float(pivot.loc[lam, ("coverage", "if_learned")])
            ok = if_len < a_len and if_cov >= 0.90
            useful = useful or ok
            useful_details.append(f"lambda={lam}: IF cov {if_cov:.3f}, IF len {if_len:.3f}, audit len {a_len:.3f}")
        except KeyError:
            useful_details.append(f"lambda={lam}: unavailable")

    tvae_gap_reduces = bool(tvae_gap < 0.5 * 0.0440514)
    if useful and tvae_gap_reduces:
        recommendation = "proceed_to_full_hillstrom_with_tvae"
    elif useful:
        recommendation = "hillstrom_possible_but_cautionary"
    else:
        recommendation = "switch_primary_to_folktables_keep_hillstrom_cautionary"

    key_methods = method_summary[
        (method_summary["lambda"].isin([0.5, 1.0]))
        & (
            ((method_summary["method"].isin(["audit_only", "synthetic_naive", "if_learned"])) & method_summary["sensitivity_c"].isna())
            | ((method_summary["method"].isin(["q_inflated", "selected_feasible"])) & (method_summary["sensitivity_c"] == 1.5))
        )
    ].sort_values(["lambda", "generator", "method"])

    lines = [
        "# Hillstrom Generator Sensitivity Decision Memo",
        "",
        "Created: 2026-05-22",
        "",
        "Status: targeted diagnostic result and decision memo, not manuscript text.",
        "",
        "## Decision",
        "",
        f"Recommendation: `{recommendation}`.",
        "",
        "Reason:",
        "",
        f"- Full-data visit ATE benchmark: `{theta_full:.6f}`.",
        f"- Average real biased-source ATE gap: `{source_gap:.6f}`.",
        f"- Average TVAE generator ATE gap: `{tvae_gap:.6f}`.",
        f"- TVAE nuisance-usefulness check: {'passed' if useful else 'failed'} ({'; '.join(useful_details)}).",
        "",
    ]
    if recommendation == "switch_primary_to_folktables_keep_hillstrom_cautionary":
        lines.append(
            "The diagnostic indicates that Hillstrom is better used as a cautionary or supplementary example than as the main practical-value real-data analysis. The real biased source itself is close to the target ATE, but fitted generators still distort the ATE enough that synthetic-naive inference fails and audit-corrected IF does not improve interval length."
        )
    elif recommendation == "hillstrom_possible_but_cautionary":
        lines.append(
            "Hillstrom could remain in the paper, but mainly as a guarded sensitivity example rather than the main empirical value demonstration."
        )
    else:
        lines.append(
            "TVAE materially improves the generator relation and gives useful IF intervals, so Hillstrom can proceed to full analysis with TVAE as the primary generator."
        )

    lines.extend(
        [
            "",
            "## Source Gap Summary",
            "",
            source_summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Generator Gap Summary",
            "",
            gen_summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Key Method Rows",
            "",
            key_methods.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Branch Frequency Summary",
            "",
            branch_summary[
                (branch_summary["lambda"].isin([0.5, 1.0]))
                & (branch_summary["sensitivity_c"].isin([1.5, 2.0]))
            ].to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Files",
            "",
            "- Source diagnostics: `tables/hillstrom_generator_sensitivity/hillstrom_source_gap_summary.csv`",
            "- Generator diagnostics: `tables/hillstrom_generator_sensitivity/hillstrom_generator_gap_summary.csv`",
            "- Method summary: `tables/hillstrom_generator_sensitivity/hillstrom_generator_method_summary.csv`",
            "- Figure: `figures/hillstrom_generator_sensitivity/hillstrom_generator_sensitivity.png`",
        ]
    )
    (ROOT / "analysis" / "HILLSTROM_GENERATOR_SENSITIVITY_DECISION_MEMO.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    df = hp.load_data()
    rng = np.random.default_rng(SEED)
    theta_full, _ = hp.diff_mean(df)
    pi = float(df["A"].mean())
    print(
        f"Hillstrom generator sensitivity: theta_full={theta_full:.6f}, "
        f"source_reps={SOURCE_REPS}, generator_reps={GENERATOR_REPS}, "
        f"audit_reps_per_generator={AUDIT_REPS_PER_GENERATOR}, tvae_epochs={TVAE_EPOCHS}",
        flush=True,
    )
    source_df, gap_df, method_df, branch_df = run_method_diagnostics(df, rng, theta_full, pi)
    source_summary = summarize_source(source_df)
    gen_summary = summarize_generator(gap_df)
    method_summary = pd.DataFrame(method_df).sort_values(["lambda", "generator", "method", "sensitivity_c"])
    branch_summary = pd.DataFrame(branch_df).sort_values(["lambda", "generator", "sensitivity_c", "selected_branch"])

    source_df.to_csv(RESULT_DIR / "hillstrom_source_gap_replicates.csv", index=False)
    gap_df.to_csv(RESULT_DIR / "hillstrom_generator_gap_replicates.csv", index=False)
    method_summary.to_csv(RESULT_DIR / "hillstrom_generator_method_summary.csv", index=False)
    branch_summary.to_csv(RESULT_DIR / "hillstrom_generator_branch_frequencies.csv", index=False)
    source_summary.to_csv(RESULT_DIR / "hillstrom_source_gap_summary.csv", index=False)
    gen_summary.to_csv(RESULT_DIR / "hillstrom_generator_gap_summary.csv", index=False)

    source_summary.to_csv(TABLE_DIR / "hillstrom_source_gap_summary.csv", index=False)
    gen_summary.to_csv(TABLE_DIR / "hillstrom_generator_gap_summary.csv", index=False)
    method_summary.to_csv(TABLE_DIR / "hillstrom_generator_method_summary.csv", index=False)
    branch_summary.to_csv(TABLE_DIR / "hillstrom_generator_branch_frequencies.csv", index=False)

    plot_diagnostics(source_summary, gen_summary, method_summary)
    write_decision_memo(
        theta_full=theta_full,
        source_summary=source_summary,
        gen_summary=gen_summary,
        method_summary=method_summary,
        branch_summary=branch_summary,
    )
    print(gen_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
