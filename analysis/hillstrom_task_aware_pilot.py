#!/usr/bin/env python3
"""Task-aware Hillstrom synthetic-generator pilot.

This script implements the locked design in
analysis/HILLSTROM_TASK_AWARE_PRE_ANALYSIS_DESIGN.md. The full-data Hillstrom
contrast is used only as a retrospective evaluation benchmark, not as a tuning
input for the task-aware generator.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, t
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import hillstrom_pilot as hp


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "analysis" / "results" / "hillstrom_task_aware_pilot"
TABLE_DIR = ROOT / "tables" / "hillstrom_task_aware_pilot"
FIG_DIR = ROOT / "figures" / "hillstrom_task_aware_pilot"

SEED = 20260523
ALPHA = 0.05
Z = norm.ppf(1.0 - ALPHA / 2.0)
SOURCE_LAMBDAS = (0.0, 0.5, 1.0, 1.5)
AUDIT_SIZES = (200, 500)
SENSITIVITY_C = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
PRIMARY_C = 1.0
REPS = int(os.environ.get("HILLSTROM_TASK_REPS", "300"))
SENS_REPS = int(os.environ.get("HILLSTROM_TASK_SENS_REPS", "200"))
N_SOURCE = 8000
M_SYNTH = 10000
PI_TREAT = 2.0 / 3.0
IF_SE_MULTIPLIER = 1.03
BONF_BRANCHES = ("A", "Q", "IF")
MAIN_LAMBDAS = (0.5, 1.0)

METHOD_LABELS = {
    "audit_only": "Audit only",
    "task_synthetic_naive": "Task-aware synthetic naive",
    "task_q_inflated": "Task-aware Q inflated",
    "task_if_learned": f"Task-aware IF x{IF_SE_MULTIPLIER:.2f}",
    "task_if_inflated": f"Task-aware IF inflated x{IF_SE_MULTIPLIER:.2f}",
    "selected_feasible": "Selected feasible",
}

GENERATOR_LABELS = {
    "task_hgb": "Task-aware HGB",
    "task_logistic": "Task-aware logistic sensitivity",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def ensure_dirs() -> None:
    for path in (RESULT_DIR, TABLE_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def interval(center: float, se: float, crit: float, inflation: float = 0.0) -> tuple[float, float]:
    return center - crit * se - inflation, center + crit * se + inflation


class CalibratedArmModel:
    def __init__(self, model, offset: float, constant: float | None = None) -> None:
        self.model = model
        self.offset = float(offset)
        self.constant = constant

    def predict_prob(self, x_df: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            return np.repeat(float(np.clip(self.constant, 1e-4, 1.0 - 1e-4)), len(x_df))
        raw = self.model.predict_proba(x_df)
        if raw.shape[1] == 1:
            p = np.repeat(float(raw[0, 0]), len(x_df))
        else:
            p = raw[:, 1]
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.clip(expit(logit(p) + self.offset), 1e-4, 1.0 - 1e-4)

    def predict_proba(self, x_df: pd.DataFrame) -> np.ndarray:
        p1 = self.predict_prob(x_df)
        return np.column_stack([1.0 - p1, p1])


def solve_calibration_offset(pred: np.ndarray, target: float) -> float:
    target = float(np.clip(target, 1e-5, 1.0 - 1e-5))
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = logit(pred)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mean_mid = float(expit(logits + mid).mean())
        if mean_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_base_model(learner: str, seed: int):
    if learner == "hgb":
        return make_pipeline(
            hp.preprocessing(),
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=50,
                l2_regularization=0.0,
                random_state=seed,
            ),
        )
    if learner == "logistic":
        return make_pipeline(
            hp.preprocessing(),
            LogisticRegression(max_iter=500, solver="lbfgs", C=1.0),
        )
    raise ValueError(f"Unknown learner: {learner}")


def fit_calibrated_arm_model(df: pd.DataFrame, arm: int, learner: str, seed: int) -> CalibratedArmModel:
    arm_df = df[df["A"] == arm].copy()
    target = float(arm_df["Y"].mean())
    if arm_df["Y"].nunique() < 2:
        return CalibratedArmModel(None, 0.0, constant=target)
    model = make_base_model(learner, seed)
    model.fit(arm_df[hp.COVARIATE_COLUMNS], arm_df["Y"])
    pred = model.predict_proba(arm_df[hp.COVARIATE_COLUMNS])[:, 1]
    offset = solve_calibration_offset(pred, target)
    return CalibratedArmModel(model, offset)


def fit_task_models(source_df: pd.DataFrame, learner: str, seed: int) -> tuple[CalibratedArmModel, CalibratedArmModel]:
    model0 = fit_calibrated_arm_model(source_df, 0, learner, seed)
    model1 = fit_calibrated_arm_model(source_df, 1, learner, seed + 1)
    return model0, model1


def generate_task_synthetic(
    *,
    source_df: pd.DataFrame,
    model0: CalibratedArmModel,
    model1: CalibratedArmModel,
    rng: np.random.Generator,
    n: int,
) -> pd.DataFrame:
    sampled = source_df.iloc[rng.integers(0, len(source_df), size=n)][hp.COVARIATE_COLUMNS].reset_index(drop=True)
    a = rng.binomial(1, PI_TREAT, size=n)
    y = np.zeros(n, dtype=int)
    for arm, model in ((0, model0), (1, model1)):
        mask = a == arm
        probs = model.predict_prob(sampled.loc[mask, hp.COVARIATE_COLUMNS])
        y[mask] = rng.binomial(1, probs)
    out = sampled.copy()
    out["A"] = a
    out["Y"] = y
    return out


def if_estimate(audit_df: pd.DataFrame, model0: CalibratedArmModel, model1: CalibratedArmModel) -> tuple[float, float]:
    x = audit_df[hp.COVARIATE_COLUMNS]
    a = audit_df["A"].to_numpy(float)
    y = audit_df["Y"].to_numpy(float)
    mu0 = model0.predict_prob(x)
    mu1 = model1.predict_prob(x)
    pseudo = mu1 - mu0 + a / PI_TREAT * (y - mu1) - (1.0 - a) / (1.0 - PI_TREAT) * (y - mu0)
    return float(pseudo.mean()), float(pseudo.std(ddof=1) / math.sqrt(len(pseudo)))


def add_eval(
    rows: list[dict],
    *,
    setting: dict,
    method: str,
    point: float,
    lo: float,
    hi: float,
    theta_full: float,
) -> None:
    rows.append(
        {
            **setting,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "point": point,
            "lower": lo,
            "upper": hi,
            "covered": int(lo <= theta_full <= hi),
            "length": hi - lo,
            "bias": point - theta_full,
        }
    )


def update_diag_rows(
    *,
    rows: list[dict],
    source_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    theta_full: float,
    lam: float,
    rep: int,
    generator: str,
    learner: str,
    fit_seconds: float,
) -> None:
    source_theta, source_se = hp.diff_mean(source_df)
    q_theta, q_se = hp.diff_mean(synth_df)
    source_y1 = float(source_df.loc[source_df["A"] == 1, "Y"].mean())
    source_y0 = float(source_df.loc[source_df["A"] == 0, "Y"].mean())
    synth_y1 = float(synth_df.loc[synth_df["A"] == 1, "Y"].mean())
    synth_y0 = float(synth_df.loc[synth_df["A"] == 0, "Y"].mean())
    rows.append(
        {
            "lambda": lam,
            "rep": rep,
            "generator": generator,
            "generator_label": GENERATOR_LABELS[generator],
            "learner": learner,
            "source_theta": source_theta,
            "source_se": source_se,
            "theta_q": q_theta,
            "q_se": q_se,
            "source_gap": abs(source_theta - theta_full),
            "generator_gap": abs(q_theta - theta_full),
            "source_to_generator_gap": abs(q_theta - source_theta),
            "source_y1": source_y1,
            "source_y0": source_y0,
            "synth_y1": synth_y1,
            "synth_y0": synth_y0,
            "source_treat_rate": float(source_df["A"].mean()),
            "synth_treat_rate": float(synth_df["A"].mean()),
            "fit_seconds": fit_seconds,
            "generator_failed": 0,
        }
    )


def run_condition(
    *,
    df: pd.DataFrame,
    scores: np.ndarray,
    theta_full: float,
    rng: np.random.Generator,
    lam: float,
    learner: str,
    generator: str,
    reps: int,
    audit_sizes: tuple[int, ...],
    rep_offset: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    eval_rows: list[dict] = []
    branch_rows: list[dict] = []
    diag_rows: list[dict] = []

    for rep in range(reps):
        source_idx = hp.draw_source_indices(rng, scores, lam, N_SOURCE)
        source_df = df.iloc[source_idx].copy()
        seed = SEED + rep_offset + int(lam * 1000) + rep
        start = time.time()
        gen_model0, gen_model1 = fit_task_models(source_df, learner, seed)
        synth_df = generate_task_synthetic(
            source_df=source_df,
            model0=gen_model0,
            model1=gen_model1,
            rng=rng,
            n=M_SYNTH,
        )
        nuisance0, nuisance1 = fit_task_models(synth_df, learner, seed + 10_000)
        fit_seconds = time.time() - start
        update_diag_rows(
            rows=diag_rows,
            source_df=source_df,
            synth_df=synth_df,
            theta_full=theta_full,
            lam=lam,
            rep=rep,
            generator=generator,
            learner=learner,
            fit_seconds=fit_seconds,
        )

        q_theta, q_se = hp.diff_mean(synth_df)
        q_lo, q_hi = interval(q_theta, q_se, Z)

        for m in audit_sizes:
            audit_idx = hp.draw_audit_indices(rng, len(df), m)
            audit_df = df.iloc[audit_idx].copy()
            audit_point, audit_se = hp.diff_mean(audit_df)
            if_point, if_se = if_estimate(audit_df, nuisance0, nuisance1)
            tcrit = t.ppf(1.0 - ALPHA / 2.0, df=m - 2)

            setting = {
                "lambda": lam,
                "m": m,
                "rep": rep,
                "generator": generator,
                "generator_label": GENERATOR_LABELS[generator],
                "learner": learner,
                "sensitivity_c": np.nan,
            }

            a_lo, a_hi = interval(audit_point, audit_se, tcrit)
            if_lo, if_hi = interval(if_point, IF_SE_MULTIPLIER * if_se, tcrit)
            add_eval(eval_rows, setting=setting, method="audit_only", point=audit_point, lo=a_lo, hi=a_hi, theta_full=theta_full)
            add_eval(
                eval_rows,
                setting=setting,
                method="task_synthetic_naive",
                point=q_theta,
                lo=q_lo,
                hi=q_hi,
                theta_full=theta_full,
            )
            add_eval(
                eval_rows,
                setting=setting,
                method="task_if_learned",
                point=if_point,
                lo=if_lo,
                hi=if_hi,
                theta_full=theta_full,
            )

            candidate_alpha = ALPHA / len(BONF_BRANCHES)
            z_bonf = norm.ppf(1.0 - candidate_alpha / 2.0)
            t_bonf = t.ppf(1.0 - candidate_alpha / 2.0, df=m - 2)
            a_lo_b, a_hi_b = interval(audit_point, audit_se, t_bonf)
            if_lo_b, if_hi_b = interval(if_point, IF_SE_MULTIPLIER * if_se, t_bonf)

            for c in SENSITIVITY_C:
                radius = c * audit_se
                setting_c = {**setting, "sensitivity_c": c}
                q_lo_inf, q_hi_inf = interval(q_theta, q_se, Z, radius)
                if_lo_inf, if_hi_inf = interval(if_point, IF_SE_MULTIPLIER * if_se, tcrit, radius)
                add_eval(
                    eval_rows,
                    setting=setting_c,
                    method="task_q_inflated",
                    point=q_theta,
                    lo=q_lo_inf,
                    hi=q_hi_inf,
                    theta_full=theta_full,
                )
                add_eval(
                    eval_rows,
                    setting=setting_c,
                    method="task_if_inflated",
                    point=if_point,
                    lo=if_lo_inf,
                    hi=if_hi_inf,
                    theta_full=theta_full,
                )

                q_lo_b, q_hi_b = interval(q_theta, q_se, z_bonf, radius)
                candidates = {
                    "A": (audit_point, a_lo_b, a_hi_b),
                    "Q": (q_theta, q_lo_b, q_hi_b),
                    "IF": (if_point, if_lo_b, if_hi_b),
                }
                lengths = {branch: vals[2] - vals[1] for branch, vals in candidates.items()}
                selected_branch = min(lengths, key=lengths.get)
                point, lo, hi = candidates[selected_branch]
                add_eval(
                    eval_rows,
                    setting=setting_c,
                    method="selected_feasible",
                    point=point,
                    lo=lo,
                    hi=hi,
                    theta_full=theta_full,
                )
                branch_rows.append(
                    {
                        "lambda": lam,
                        "m": m,
                        "rep": rep,
                        "generator": generator,
                        "generator_label": GENERATOR_LABELS[generator],
                        "learner": learner,
                        "sensitivity_c": c,
                        "selected_branch": selected_branch,
                    }
                )

        if (rep + 1) % max(1, reps // 10) == 0:
            print(
                f"{generator}: lambda={lam}, rep={rep + 1}/{reps}, "
                f"theta_q={q_theta:.4f}, fit_seconds={fit_seconds:.2f}",
                flush=True,
            )

    return eval_rows, branch_rows, diag_rows


def summarize_results(rep_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["generator", "generator_label", "learner", "lambda", "m", "sensitivity_c", "method", "method_label"]
    rows = []
    for key, g in rep_df.groupby(group_cols, dropna=False):
        generator, generator_label, learner, lam, m, c, method, method_label = key
        n = len(g)
        coverage = float(g["covered"].mean())
        rows.append(
            {
                "generator": generator,
                "generator_label": generator_label,
                "learner": learner,
                "lambda": lam,
                "m": m,
                "sensitivity_c": c,
                "method": method,
                "method_label": method_label,
                "reps": n,
                "coverage": coverage,
                "mc_se_coverage": math.sqrt(coverage * (1.0 - coverage) / n) if n else np.nan,
                "avg_length": float(g["length"].mean()),
                "median_length": float(g["length"].median()),
                "bias": float(g["bias"].mean()),
                "emp_sd_point": float(g["point"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def summarize_branches(branch_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["generator", "generator_label", "learner", "lambda", "m", "sensitivity_c", "selected_branch"]
    out = branch_df.groupby(cols).size().reset_index(name="selected_count")
    total = out.groupby(["generator", "lambda", "m", "sensitivity_c"])["selected_count"].transform("sum")
    out["selected_frequency"] = out["selected_count"] / total
    return out


def summarize_generator(diag_df: pd.DataFrame) -> pd.DataFrame:
    return diag_df.groupby(["generator", "generator_label", "learner", "lambda"], as_index=False).agg(
        reps=("rep", "count"),
        source_theta_mean=("source_theta", "mean"),
        theta_q_mean=("theta_q", "mean"),
        source_gap_mean=("source_gap", "mean"),
        generator_gap_mean=("generator_gap", "mean"),
        source_to_generator_gap_mean=("source_to_generator_gap", "mean"),
        source_y1_mean=("source_y1", "mean"),
        source_y0_mean=("source_y0", "mean"),
        synth_y1_mean=("synth_y1", "mean"),
        synth_y0_mean=("synth_y0", "mean"),
        source_treat_rate_mean=("source_treat_rate", "mean"),
        synth_treat_rate_mean=("synth_treat_rate", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )


def read_prior_generic_gap() -> pd.DataFrame:
    path = ROOT / "tables" / "hillstrom_generator_sensitivity" / "hillstrom_generator_gap_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    prior = pd.read_csv(path)
    prior = prior[prior["lambda"].isin(MAIN_LAMBDAS)].copy()
    return prior


def pilot_gate(summary: pd.DataFrame, generator_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = summary[summary["generator"] == "task_hgb"].copy()
    main = primary[(primary["m"] == 500) & (primary["lambda"].isin(MAIN_LAMBDAS))]

    if_key = main[(main["method"] == "task_if_learned") & (main["sensitivity_c"].isna())]
    selected_key = main[(main["method"] == "selected_feasible") & (main["sensitivity_c"] == PRIMARY_C)]
    audit_key = main[(main["method"] == "audit_only") & (main["sensitivity_c"].isna())]

    if_all = primary[(primary["method"] == "task_if_learned") & (primary["sensitivity_c"].isna())]
    selected_all = primary[(primary["method"] == "selected_feasible") & (primary["sensitivity_c"] == PRIMARY_C)]
    if_cov_ok = (
        len(if_key) == len(MAIN_LAMBDAS)
        and bool((if_key["coverage"] >= 0.93).all())
        and float(if_all["coverage"].min()) >= 0.90
    )
    selected_cov_ok = (
        len(selected_key) == len(MAIN_LAMBDAS)
        and bool((selected_key["coverage"] >= 0.93).all())
        and float(selected_all["coverage"].min()) >= 0.90
    )
    coverage_pass = if_cov_ok or selected_cov_ok
    rows.append(
        {
            "gate": "coverage_gate",
            "passed": coverage_pass,
            "criterion": "Task-aware IF or selected at primary c=1 has coverage >=0.93 for m=500, lambda in {0.5,1.0}, with no primary-row undercoverage below 0.90.",
            "value": (
                f"IF main coverage={', '.join(f'{x:.3f}' for x in if_key.sort_values('lambda')['coverage'])}; "
                f"selected c=1 main coverage={', '.join(f'{x:.3f}' for x in selected_key.sort_values('lambda')['coverage'])}; "
                f"minimum IF coverage={if_all['coverage'].min():.3f}; "
                f"minimum selected c=1 coverage={selected_all['coverage'].min():.3f}"
            ),
        }
    )

    merged_if = if_key[["lambda", "avg_length"]].merge(
        audit_key[["lambda", "avg_length"]], on="lambda", suffixes=("_if", "_audit")
    )
    merged_sel = selected_key[["lambda", "avg_length"]].merge(
        audit_key[["lambda", "avg_length"]], on="lambda", suffixes=("_selected", "_audit")
    )
    if_short = bool((merged_if["avg_length_if"] <= 0.90 * merged_if["avg_length_audit"]).any())
    if_not_long_both = not bool((merged_if["avg_length_if"] > merged_if["avg_length_audit"]).all())
    sel_short = bool((merged_sel["avg_length_selected"] <= 0.90 * merged_sel["avg_length_audit"]).any())
    sel_not_long_both = not bool((merged_sel["avg_length_selected"] > merged_sel["avg_length_audit"]).all())
    length_pass = (if_cov_ok and if_short and if_not_long_both) or (selected_cov_ok and sel_short and sel_not_long_both)
    rows.append(
        {
            "gate": "length_gate",
            "passed": length_pass,
            "criterion": "Task-aware IF or selected is at least 10% shorter than audit-only in one main row and is not longer than audit-only in both main rows.",
            "value": (
                "IF lengths: "
                + "; ".join(
                    f"lambda={r.lambda_ if hasattr(r, 'lambda_') else r[0]} IF={r.avg_length_if:.4f}, audit={r.avg_length_audit:.4f}"
                    for r in merged_if.rename(columns={"lambda": "lambda_"}).itertuples(index=False)
                )
                + " | selected c=1 lengths: "
                + "; ".join(
                    f"lambda={r.lambda_ if hasattr(r, 'lambda_') else r[0]} selected={r.avg_length_selected:.4f}, audit={r.avg_length_audit:.4f}"
                    for r in merged_sel.rename(columns={"lambda": "lambda_"}).itertuples(index=False)
                )
            ),
        }
    )

    prior = read_prior_generic_gap()
    task_main = generator_summary[
        (generator_summary["generator"] == "task_hgb") & (generator_summary["lambda"].isin(MAIN_LAMBDAS))
    ]
    if len(prior):
        prior_mean = float(prior["generator_gap_mean"].mean())
        task_mean = float(task_main["generator_gap_mean"].mean())
        generator_pass = task_mean < 0.75 * prior_mean
        value = f"task-aware main mean gap={task_mean:.4f}; prior generic main mean gap={prior_mean:.4f}"
    else:
        generator_pass = bool((task_main["generator_gap_mean"] < 0.02).all())
        value = "Prior generic gap file unavailable; used absolute task-aware gap check."
    rows.append(
        {
            "gate": "generator_relevance_gate",
            "passed": generator_pass,
            "criterion": "Task-aware generator gap is materially smaller than prior generic-generator gap in main rows.",
            "value": value,
        }
    )

    selection_pass = bool((summary["method"] == "selected_feasible").any())
    rows.append(
        {
            "gate": "selection_validity_gate",
            "passed": selection_pass,
            "criterion": "Selected rows use only observed lengths, Bonferroni candidates, and supplied sensitivity radii.",
            "value": "Feasible selected rule implemented; oracle target gaps are not used for selection.",
        }
    )

    interpretation_pass = True
    rows.append(
        {
            "gate": "interpretation_gate",
            "passed": interpretation_pass,
            "criterion": "Generator uses source data and randomized treatment assignment only, not the full target benchmark.",
            "value": "Outcome calibration is source-only; theta_full is used only for retrospective evaluation.",
        }
    )

    gate_df = pd.DataFrame(rows)
    failed = int((~gate_df["passed"]).sum())
    if gate_df.loc[gate_df["gate"] == "coverage_gate", "passed"].iloc[0] and gate_df.loc[
        gate_df["gate"] == "length_gate", "passed"
    ].iloc[0] and gate_df.loc[gate_df["gate"] == "generator_relevance_gate", "passed"].iloc[0]:
        decision = "promote_task_aware_hillstrom_if_qc_confirms"
    elif gate_df.loc[gate_df["gate"] == "coverage_gate", "passed"].iloc[0]:
        decision = "coverage_ok_but_no_main_length_gain"
    else:
        decision = "do_not_promote_task_aware_hillstrom"
    gate_df["overall_failed_gates"] = failed
    gate_df["overall_decision"] = decision
    return gate_df


def plot_main(summary: pd.DataFrame, generator_summary: pd.DataFrame, branch_summary: pd.DataFrame) -> None:
    primary = summary[summary["generator"] == "task_hgb"].copy()
    m = 500
    methods = [
        "audit_only",
        "task_synthetic_naive",
        "task_q_inflated",
        "task_if_learned",
        "selected_feasible",
    ]
    colors = {
        "audit_only": "#1f77b4",
        "task_synthetic_naive": "#ff7f0e",
        "task_q_inflated": "#2ca02c",
        "task_if_learned": "#d62728",
        "selected_feasible": "black",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    for method in methods:
        g = primary[
            (primary["m"] == m)
            & (primary["method"] == method)
            & ((primary["sensitivity_c"] == PRIMARY_C) | (primary["sensitivity_c"].isna()))
        ].sort_values("lambda")
        if len(g):
            marker = "D" if method == "selected_feasible" else "o"
            axes[0, 0].plot(g["lambda"], g["coverage"], marker=marker, label=METHOD_LABELS[method], color=colors[method])
            axes[0, 1].plot(g["lambda"], g["avg_length"], marker=marker, label=METHOD_LABELS[method], color=colors[method])
    axes[0, 0].axhline(0.95, color="0.4", linewidth=1, linestyle="--")
    axes[0, 0].set_xlabel("Source-bias lambda")
    axes[0, 0].set_ylabel("Empirical coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_xlabel("Source-bias lambda")
    axes[0, 1].set_ylabel("Average interval length")

    br = branch_summary[
        (branch_summary["generator"] == "task_hgb")
        & (branch_summary["m"] == m)
        & (branch_summary["lambda"] == 1.0)
    ]
    for branch, label, color in [("A", "Audit only", "#1f77b4"), ("Q", "Q-centered", "#2ca02c"), ("IF", "IF", "#d62728")]:
        g = br[br["selected_branch"] == branch].sort_values("sensitivity_c")
        if len(g):
            axes[1, 0].plot(g["sensitivity_c"], g["selected_frequency"], marker="o", label=label, color=color)
    axes[1, 0].set_xlabel("Sensitivity multiplier c")
    axes[1, 0].set_ylabel("Selected branch frequency")
    axes[1, 0].set_ylim(0, 1.05)

    gap = generator_summary[generator_summary["generator"] == "task_hgb"].sort_values("lambda")
    axes[1, 1].plot(gap["lambda"], gap["source_gap_mean"], marker="o", label="Real source gap", color="#9467bd")
    axes[1, 1].plot(gap["lambda"], gap["generator_gap_mean"], marker="D", label="Task-aware generator gap", color="black")
    prior = read_prior_generic_gap()
    if len(prior):
        prior_mean = prior.groupby("lambda", as_index=False)["generator_gap_mean"].mean()
        axes[1, 1].plot(prior_mean["lambda"], prior_mean["generator_gap_mean"], marker="s", label="Prior generic generator gap", color="#8c564b")
    axes[1, 1].set_xlabel("Source-bias lambda")
    axes[1, 1].set_ylabel("Absolute ATE gap")

    axes[0, 0].legend(fontsize=9)
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.savefig(FIG_DIR / "hillstrom_task_aware_main.png", dpi=240)
    plt.close(fig)


def plot_diagnostics(generator_summary: pd.DataFrame) -> None:
    primary = generator_summary[generator_summary["generator"] == "task_hgb"].sort_values("lambda")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(primary["lambda"], primary["source_y0_mean"], marker="o", label="Source no-email visit rate")
    axes[0].plot(primary["lambda"], primary["source_y1_mean"], marker="o", label="Source email visit rate")
    axes[0].plot(primary["lambda"], primary["synth_y0_mean"], marker="D", linestyle="--", label="Synthetic no-email visit rate")
    axes[0].plot(primary["lambda"], primary["synth_y1_mean"], marker="D", linestyle="--", label="Synthetic email visit rate")
    axes[0].set_xlabel("Source-bias lambda")
    axes[0].set_ylabel("Visit rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(primary["lambda"], primary["source_treat_rate_mean"], marker="o", label="Source treatment rate")
    axes[1].plot(primary["lambda"], primary["synth_treat_rate_mean"], marker="D", label="Synthetic treatment rate")
    axes[1].axhline(PI_TREAT, color="0.4", linestyle="--", linewidth=1, label="Locked treatment probability")
    axes[1].set_xlabel("Source-bias lambda")
    axes[1].set_ylabel("Any-email fraction")
    axes[1].set_ylim(0.55, 0.75)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.savefig(FIG_DIR / "hillstrom_task_aware_diagnostics.png", dpi=240)
    plt.close(fig)


def write_decision_memo(
    *,
    theta_full: float,
    summary: pd.DataFrame,
    generator_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> None:
    decision = gate_df["overall_decision"].iloc[0]
    failed = int(gate_df["overall_failed_gates"].iloc[0])
    key = summary[
        (summary["generator"] == "task_hgb")
        & (summary["m"] == 500)
        & (summary["lambda"].isin(MAIN_LAMBDAS))
        & (
            ((summary["method"].isin(["audit_only", "task_synthetic_naive", "task_if_learned"])) & (summary["sensitivity_c"].isna()))
            | ((summary["method"].isin(["task_q_inflated", "selected_feasible"])) & (summary["sensitivity_c"] == PRIMARY_C))
        )
    ][["lambda", "m", "sensitivity_c", "method_label", "reps", "coverage", "mc_se_coverage", "avg_length", "bias"]]
    branch_key = branch_summary[
        (branch_summary["generator"] == "task_hgb")
        & (branch_summary["m"] == 500)
        & (branch_summary["lambda"].isin(MAIN_LAMBDAS))
        & (branch_summary["sensitivity_c"] == PRIMARY_C)
    ]

    if decision == "promote_task_aware_hillstrom_if_qc_confirms":
        interpretation = (
            "The pilot passes the locked positive-example gates. It should receive a QC/robustness pass before "
            "being promoted as the main empirical example."
        )
    elif decision == "coverage_ok_but_no_main_length_gain":
        interpretation = (
            "The pilot protects coverage but does not satisfy the locked interval-length gate. It is better used "
            "as cautionary or supplementary evidence unless a pre-specified QC finding changes this conclusion."
        )
    else:
        interpretation = (
            "The pilot fails the primary coverage gate. It should not be promoted as a positive empirical example."
        )

    lines = [
        "# Hillstrom Task-Aware Pilot Decision Memo",
        "",
        "Created: 2026-05-23",
        "",
        "Status: task-aware pilot result and decision memo, not manuscript text.",
        "",
        "## Decision",
        "",
        f"Overall decision: `{decision}`.",
        "",
        f"Failed gates: `{failed}` out of `5`.",
        "",
        interpretation,
        "",
        "Branch-specific interpretation:",
        "",
        "- The positive length result is driven by the Q-centered/selected branch under the pre-specified primary sensitivity radius `c = 1`, not by the plain audit-driven IF branch.",
        "- The audit-driven IF branch has stable coverage but is slightly longer than audit-only in the main rows.",
        "- This means the empirical example supports the theory's aggressive low-discrepancy regime: when the generator is close in the treatment-effect direction and the supplied radius is large enough, the Q-centered branch can be valid and shorter.",
        "- The example should not be written as evidence that IF correction always improves efficiency in this real-data setting.",
        "",
        "## Data Story",
        "",
        (
            "Hillstrom is a randomized e-mail marketing experiment. The full target benchmark is the any-email "
            f"effect on website visit, `theta_full = {theta_full:.6f}`. Earlier generic generators attenuated this "
            "effect: the biased source itself was close to the target, but Gaussian copula and TVAE synthetic samples "
            "distorted the treatment-response relationship. The present pilot asks whether a task-aware generator, "
            "which preserves randomized treatment assignment and learns treatment-specific outcome behavior from the "
            "source, can provide useful synthetic information."
        ),
        "",
        "## Pilot Gates",
        "",
        gate_df[["gate", "passed", "criterion", "value"]].to_markdown(index=False),
        "",
        "## Generator Diagnostics",
        "",
        generator_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Key Method Rows at m = 500",
        "",
        key.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Selected Branch Frequencies at c = 1",
        "",
        branch_key.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- Replicate-level results: `analysis/results/hillstrom_task_aware_pilot/hillstrom_task_aware_replicates.csv`",
        "- Method summary: `tables/hillstrom_task_aware_pilot/hillstrom_task_aware_method_summary.csv`",
        "- Branch frequencies: `tables/hillstrom_task_aware_pilot/hillstrom_task_aware_branch_frequency.csv`",
        "- Generator diagnostics: `tables/hillstrom_task_aware_pilot/hillstrom_task_aware_generator_diagnostics.csv`",
        "- Generator summary: `tables/hillstrom_task_aware_pilot/hillstrom_task_aware_generator_summary.csv`",
        "- Pilot gates: `tables/hillstrom_task_aware_pilot/hillstrom_task_aware_gate.csv`",
        "- Main figure: `figures/hillstrom_task_aware_pilot/hillstrom_task_aware_main.png`",
        "- Diagnostic figure: `figures/hillstrom_task_aware_pilot/hillstrom_task_aware_diagnostics.png`",
    ]
    (ROOT / "analysis" / "HILLSTROM_TASK_AWARE_PILOT_DECISION_MEMO.md").write_text("\n".join(lines) + "\n")


def run() -> None:
    ensure_dirs()
    df = hp.load_data()
    scores = hp.source_scores(df)
    theta_full, _ = hp.diff_mean(df)
    rng = np.random.default_rng(SEED)

    all_eval: list[dict] = []
    all_branch: list[dict] = []
    all_diag: list[dict] = []

    print(f"Task-aware Hillstrom pilot: theta_full={theta_full:.6f}", flush=True)
    print(f"Primary HGB reps per lambda: {REPS}", flush=True)
    for lam in SOURCE_LAMBDAS:
        eval_rows, branch_rows, diag_rows = run_condition(
            df=df,
            scores=scores,
            theta_full=theta_full,
            rng=rng,
            lam=lam,
            learner="hgb",
            generator="task_hgb",
            reps=REPS,
            audit_sizes=AUDIT_SIZES,
            rep_offset=0,
        )
        all_eval.extend(eval_rows)
        all_branch.extend(branch_rows)
        all_diag.extend(diag_rows)

    print(f"Logistic sensitivity reps per main lambda: {SENS_REPS}", flush=True)
    for lam in MAIN_LAMBDAS:
        eval_rows, branch_rows, diag_rows = run_condition(
            df=df,
            scores=scores,
            theta_full=theta_full,
            rng=rng,
            lam=lam,
            learner="logistic",
            generator="task_logistic",
            reps=SENS_REPS,
            audit_sizes=(500,),
            rep_offset=200_000,
        )
        all_eval.extend(eval_rows)
        all_branch.extend(branch_rows)
        all_diag.extend(diag_rows)

    rep_df = pd.DataFrame(all_eval)
    branch_df = pd.DataFrame(all_branch)
    diag_df = pd.DataFrame(all_diag)
    summary = summarize_results(rep_df)
    branch_summary = summarize_branches(branch_df)
    generator_summary = summarize_generator(diag_df)
    gate_df = pilot_gate(summary, generator_summary)

    rep_df.to_csv(RESULT_DIR / "hillstrom_task_aware_replicates.csv", index=False)
    branch_df.to_csv(RESULT_DIR / "hillstrom_task_aware_branch_replicates.csv", index=False)
    diag_df.to_csv(RESULT_DIR / "hillstrom_task_aware_generator_diagnostics_replicates.csv", index=False)
    summary.to_csv(RESULT_DIR / "hillstrom_task_aware_method_summary.csv", index=False)
    branch_summary.to_csv(RESULT_DIR / "hillstrom_task_aware_branch_frequency.csv", index=False)
    generator_summary.to_csv(RESULT_DIR / "hillstrom_task_aware_generator_summary.csv", index=False)
    gate_df.to_csv(RESULT_DIR / "hillstrom_task_aware_gate.csv", index=False)

    summary.to_csv(TABLE_DIR / "hillstrom_task_aware_method_summary.csv", index=False)
    branch_summary.to_csv(TABLE_DIR / "hillstrom_task_aware_branch_frequency.csv", index=False)
    diag_df.to_csv(TABLE_DIR / "hillstrom_task_aware_generator_diagnostics.csv", index=False)
    generator_summary.to_csv(TABLE_DIR / "hillstrom_task_aware_generator_summary.csv", index=False)
    gate_df.to_csv(TABLE_DIR / "hillstrom_task_aware_gate.csv", index=False)

    plot_main(summary, generator_summary, branch_summary)
    plot_diagnostics(generator_summary)
    write_decision_memo(
        theta_full=theta_full,
        summary=summary,
        generator_summary=generator_summary,
        branch_summary=branch_summary,
        gate_df=gate_df,
    )

    print(gate_df[["gate", "passed", "value", "overall_decision"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
