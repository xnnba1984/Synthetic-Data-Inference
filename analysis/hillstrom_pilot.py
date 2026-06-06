#!/usr/bin/env python3
"""Hillstrom semi-real pilot analysis.

This script implements the locked pre-analysis design in
analysis/HILLSTROM_PRE_ANALYSIS_DESIGN.md. It intentionally treats oracle gaps
and full-data contrasts as evaluation quantities, not as inputs to method
selection.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import os
import time
import urllib.request
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Hillstrom pilot requires SDV with GaussianCopulaSynthesizer support.") from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data" / "hillstrom"
RESULT_DIR = ROOT / "analysis" / "results" / "hillstrom_pilot"
TABLE_DIR = ROOT / "tables" / "hillstrom_pilot"
FIG_DIR = ROOT / "figures" / "hillstrom_pilot"

DATA_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
DATA_PATH = DATA_DIR / "hillstrom.csv"

SEED = 20260522
ALPHA = 0.05
Z = norm.ppf(1.0 - ALPHA / 2.0)
SOURCE_LAMBDAS = (0.0, 0.5, 1.0, 1.5)
AUDIT_SIZES = (200, 500)
SENSITIVITY_C = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
REPS = int(os.environ.get("HILLSTROM_REPS", "200"))
N_SOURCE = 8000
M_SYNTH = 10000
IF_SE_MULTIPLIER = 1.03
BONF_BRANCHES = ("A", "Q", "IF")

COVARIATE_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]
CATEGORICAL_COLUMNS = ["history_segment", "mens", "womens", "zip_code", "newbie", "channel"]
NUMERIC_COLUMNS = ["recency", "history"]

METHOD_LABELS = {
    "audit_only": "Audit only",
    "synthetic_naive": "Synthetic naive",
    "pooled_naive": "Pooled naive",
    "q_inflated": "Q inflated",
    "if_learned": f"Audit-driven IF x{IF_SE_MULTIPLIER:.2f}",
    "selected_feasible": "Selected feasible",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="We strongly recommend saving the metadata")


def ensure_dirs() -> None:
    for path in (DATA_DIR, RESULT_DIR, TABLE_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_data() -> None:
    if DATA_PATH.exists():
        return
    print(f"Downloading Hillstrom data to {DATA_PATH}")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        DATA_PATH.write_bytes(response.read())


def load_data() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(DATA_PATH)
    df["A"] = (df["segment"] != "No E-Mail").astype(int)
    df["Y"] = df["visit"].astype(int)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)
    df["recency"] = pd.to_numeric(df["recency"], errors="raise")
    df["history"] = pd.to_numeric(df["history"], errors="raise")
    return df


def source_scores(df: pd.DataFrame) -> np.ndarray:
    log_history = np.log1p(df["history"].to_numpy(float))
    z_history = (log_history - log_history.mean()) / log_history.std(ddof=0)
    recency = df["recency"].to_numpy(float)
    z_recency = (recency - recency.mean()) / recency.std(ddof=0)
    i_multi = (df["channel"].astype(str).to_numpy() == "Multichannel").astype(float)
    high_levels = {
        "4) $350 - $500",
        "5) $500 - $750",
        "6) $750 - $1,000",
        "7) $1,000 +",
    }
    i_high = df["history_segment"].astype(str).isin(high_levels).to_numpy(float)
    return 0.70 * z_history - 0.40 * z_recency + 0.30 * i_multi + 0.30 * i_high


def draw_source_indices(rng: np.random.Generator, base_scores: np.ndarray, lam: float, n: int) -> np.ndarray:
    scaled = lam * base_scores
    scaled = scaled - scaled.max()
    weights = np.exp(scaled)
    weights = weights / weights.sum()
    return rng.choice(len(base_scores), size=n, replace=False, p=weights)


def draw_audit_indices(rng: np.random.Generator, n_total: int, m: int) -> np.ndarray:
    return rng.choice(n_total, size=m, replace=False)


def make_sdv_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COVARIATE_COLUMNS + ["A", "Y"]].copy()
    for col in CATEGORICAL_COLUMNS + ["A", "Y"]:
        out[col] = out[col].astype(str)
    return out


def make_metadata(df: pd.DataFrame) -> Metadata:
    metadata = Metadata.detect_from_dataframe(df)
    for col in CATEGORICAL_COLUMNS + ["A", "Y"]:
        metadata.update_column(column_name=col, sdtype="categorical")
    for col in NUMERIC_COLUMNS:
        metadata.update_column(column_name=col, sdtype="numerical")
    metadata.validate()
    return metadata


def parse_binary(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    mapped = raw.map({"0": 0, "1": 1, "0.0": 0, "1.0": 1, "False": 0, "True": 1})
    numeric = pd.to_numeric(raw, errors="coerce")
    fallback = numeric.round().clip(0, 1)
    parsed = mapped.where(mapped.notna(), fallback)
    return parsed.astype(float)


def clean_synthetic(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str)
    out["A"] = parse_binary(out["A"])
    out["Y"] = parse_binary(out["Y"])
    out = out.dropna(subset=NUMERIC_COLUMNS + ["A", "Y"]).copy()
    out["A"] = out["A"].astype(int)
    out["Y"] = out["Y"].astype(int)
    return out[COVARIATE_COLUMNS + ["A", "Y"]]


def sample_synthetic(synthesizer: GaussianCopulaSynthesizer, n: int, min_each_arm: int = 50) -> pd.DataFrame:
    pieces = []
    attempts = 0
    while sum(len(p) for p in pieces) < n and attempts < 6:
        raw = synthesizer.sample(n)
        pieces.append(clean_synthetic(raw))
        attempts += 1
    out = pd.concat(pieces, ignore_index=True).iloc[:n].copy()
    counts = out["A"].value_counts()
    if len(out) < n or counts.get(0, 0) < min_each_arm or counts.get(1, 0) < min_each_arm:
        raise RuntimeError(f"Degenerate synthetic sample: n={len(out)}, A counts={counts.to_dict()}")
    return out


def fit_generator(source_df: pd.DataFrame, seed: int) -> tuple[GaussianCopulaSynthesizer, float]:
    train = make_sdv_frame(source_df)
    metadata = make_metadata(train)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    np.random.seed(seed)
    start = time.time()
    synthesizer.fit(train)
    return synthesizer, time.time() - start


def diff_mean(df: pd.DataFrame) -> tuple[float, float]:
    y1 = df.loc[df["A"] == 1, "Y"].to_numpy(float)
    y0 = df.loc[df["A"] == 0, "Y"].to_numpy(float)
    theta = float(y1.mean() - y0.mean())
    se = math.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return theta, se


def pooled_frame(audit_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([audit_df[COVARIATE_COLUMNS + ["A", "Y"]], synth_df], ignore_index=True)


def preprocessing() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("cat", encoder, CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def fit_arm_model(synth_df: pd.DataFrame, arm: int, seed: int):
    arm_df = synth_df[synth_df["A"] == arm]
    if arm_df["Y"].nunique() < 2:
        constant = float(arm_df["Y"].mean())
        return ConstantPredictor(constant)
    clf = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=20,
        random_state=seed,
    )
    pipe = make_pipeline(preprocessing(), clf)
    pipe.fit(arm_df[COVARIATE_COLUMNS], arm_df["Y"])
    return pipe


class ConstantPredictor:
    def __init__(self, value: float) -> None:
        self.value = float(np.clip(value, 0.0, 1.0))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p1 = np.repeat(self.value, len(x))
        return np.column_stack([1.0 - p1, p1])


def predict_prob(model, x_df: pd.DataFrame) -> np.ndarray:
    pred = model.predict_proba(x_df)
    if pred.shape[1] == 1:
        return np.repeat(float(pred[0, 0]), len(x_df))
    return np.clip(pred[:, 1], 1e-4, 1.0 - 1e-4)


def fit_nuisances(synth_df: pd.DataFrame, seed: int):
    return fit_arm_model(synth_df, 0, seed), fit_arm_model(synth_df, 1, seed + 1)


def if_estimate(audit_df: pd.DataFrame, model0, model1, pi: float) -> tuple[float, float]:
    x = audit_df[COVARIATE_COLUMNS]
    a = audit_df["A"].to_numpy(float)
    y = audit_df["Y"].to_numpy(float)
    mu0 = predict_prob(model0, x)
    mu1 = predict_prob(model1, x)
    pseudo = mu1 - mu0 + a / pi * (y - mu1) - (1.0 - a) / (1.0 - pi) * (y - mu0)
    return float(pseudo.mean()), float(pseudo.std(ddof=1) / math.sqrt(len(pseudo)))


def interval(center: float, se: float, crit: float, inflation: float = 0.0) -> tuple[float, float]:
    return center - crit * se - inflation, center + crit * se + inflation


def add_eval(rows: list[dict], *, setting: dict, method: str, point: float, lo: float, hi: float, theta_full: float) -> None:
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


def fit_full_aipw_benchmark(df: pd.DataFrame, pi: float) -> float:
    # This is a robustness benchmark only; the primary benchmark remains the
    # randomized full-data contrast. Use a fast logistic learner here so the
    # pilot runtime is spent on the locked Monte Carlo grid.
    model0 = fit_logistic_arm_model(df[COVARIATE_COLUMNS + ["A", "Y"]], 0)
    model1 = fit_logistic_arm_model(df[COVARIATE_COLUMNS + ["A", "Y"]], 1)
    theta, _ = if_estimate(df[COVARIATE_COLUMNS + ["A", "Y"]], model0, model1, pi)
    return theta


def fit_logistic_arm_model(df: pd.DataFrame, arm: int):
    arm_df = df[df["A"] == arm]
    if arm_df["Y"].nunique() < 2:
        return ConstantPredictor(float(arm_df["Y"].mean()))
    pipe = make_pipeline(
        preprocessing(),
        LogisticRegression(max_iter=300, solver="lbfgs", C=1.0),
    )
    pipe.fit(arm_df[COVARIATE_COLUMNS], arm_df["Y"])
    return pipe


def summarize_results(rep_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["lambda", "m", "sensitivity_c", "method", "method_label"]
    rows = []
    for key, g in rep_df.groupby(group_cols, dropna=False):
        lam, m, c, method, method_label = key
        n = len(g)
        coverage = g["covered"].mean()
        rows.append(
            {
                "lambda": lam,
                "m": m,
                "sensitivity_c": c,
                "method": method,
                "method_label": method_label,
                "reps": n,
                "coverage": coverage,
                "mc_se_coverage": math.sqrt(coverage * (1 - coverage) / n) if n else np.nan,
                "avg_length": g["length"].mean(),
                "median_length": g["length"].median(),
                "bias": g["bias"].mean(),
                "emp_sd_point": g["point"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def summarize_branches(branch_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["lambda", "m", "sensitivity_c", "selected_branch"]
    out = branch_df.groupby(cols).size().reset_index(name="selected_count")
    total = out.groupby(["lambda", "m", "sensitivity_c"])["selected_count"].transform("sum")
    out["selected_frequency"] = out["selected_count"] / total
    return out


def plot_main(summary: pd.DataFrame, gap_df: pd.DataFrame, branch_summary: pd.DataFrame) -> None:
    plot_c = 1.0
    m = 500
    methods = ["audit_only", "synthetic_naive", "q_inflated", "if_learned", "selected_feasible"]
    labels = [METHOD_LABELS[x] for x in methods]
    colors = {
        "audit_only": "#1f77b4",
        "synthetic_naive": "#ff7f0e",
        "q_inflated": "#2ca02c",
        "if_learned": "#d62728",
        "selected_feasible": "black",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for method, label in zip(methods, labels):
        g = summary[
            (summary["m"] == m)
            & (summary["method"] == method)
            & ((summary["sensitivity_c"] == plot_c) | (summary["sensitivity_c"].isna()))
        ].sort_values("lambda")
        if len(g):
            marker = "D" if method == "selected_feasible" else "o"
            axes[0, 0].plot(g["lambda"], g["coverage"], marker=marker, label=label, color=colors[method])
            axes[0, 1].plot(g["lambda"], g["avg_length"], marker=marker, label=label, color=colors[method])
    axes[0, 0].axhline(0.95, color="0.4", linewidth=1, linestyle="--")
    axes[0, 0].set_title("Coverage at m = 500")
    axes[0, 0].set_xlabel("Source-bias lambda")
    axes[0, 0].set_ylabel("Empirical coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_title("Average interval length at m = 500")
    axes[0, 1].set_xlabel("Source-bias lambda")
    axes[0, 1].set_ylabel("Average length")

    br = branch_summary[(branch_summary["m"] == m) & (branch_summary["lambda"] == 1.0)]
    for branch, label in [("A", "Audit only"), ("Q", "Q-centered"), ("IF", "IF")]:
        g = br[br["selected_branch"] == branch].sort_values("sensitivity_c")
        axes[1, 0].plot(g["sensitivity_c"], g["selected_frequency"], marker="o", label=label)
    axes[1, 0].set_title("Selected branch frequency at lambda = 1.0")
    axes[1, 0].set_xlabel("Sensitivity multiplier c")
    axes[1, 0].set_ylabel("Selection frequency")
    axes[1, 0].set_ylim(0, 1.05)

    gap = gap_df.groupby("lambda", as_index=False).agg(
        direct_gap_mean=("direct_gap", "mean"),
        direct_gap_sd=("direct_gap", "std"),
        generator_failures=("generator_failed", "sum"),
    )
    axes[1, 1].errorbar(gap["lambda"], gap["direct_gap_mean"], yerr=gap["direct_gap_sd"], marker="o")
    axes[1, 1].set_title("Retrospective generator ATE gap")
    axes[1, 1].set_xlabel("Source-bias lambda")
    axes[1, 1].set_ylabel("Mean absolute gap")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.savefig(FIG_DIR / "hillstrom_pilot_main.png", dpi=220)
    plt.close(fig)


def pilot_gate(
    *,
    full_theta: float,
    gap_df: pd.DataFrame,
    summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    treatment_signal = abs(full_theta) >= 0.003
    rows.append(
        {
            "gate": "treatment_signal",
            "passed": treatment_signal,
            "criterion": "|theta_full| >= 0.003",
            "value": full_theta,
            "interpretation": "Primary visit effect is large enough for an interpretable pilot."
            if treatment_signal
            else "Primary visit effect is too small for a useful empirical example.",
        }
    )

    gap_by_lambda = gap_df.groupby("lambda")["direct_gap"].mean()
    middle = gap_by_lambda.loc[[0.5, 1.0, 1.5]]
    useful_gap = bool(((middle > 0.001) & (middle < 0.08)).any())
    rows.append(
        {
            "gate": "useful_discrepancy_range",
            "passed": useful_gap,
            "criterion": "At least one lambda in {0.5,1.0,1.5} has mean direct gap between 0.001 and 0.08.",
            "value": "; ".join(f"lambda={k}: {v:.4f}" for k, v in middle.items()),
            "interpretation": "The source-bias grid creates a middle real-synthetic discrepancy regime."
            if useful_gap
            else "The source-bias grid does not create a useful middle discrepancy regime.",
        }
    )

    failure_rate = gap_df["generator_failed"].mean()
    stable = failure_rate == 0
    rows.append(
        {
            "gate": "generator_stability",
            "passed": stable,
            "criterion": "No Gaussian-copula replicate failures.",
            "value": failure_rate,
            "interpretation": "Gaussian-copula generation was stable."
            if stable
            else "Gaussian-copula generation had failures that require repair.",
        }
    )

    key = summary[
        (summary["m"] == 500)
        & (summary["method"].isin(["audit_only", "if_learned"]))
        & (summary["lambda"].isin([0.5, 1.0, 1.5]))
    ]
    pivot = key.pivot_table(index="lambda", columns="method", values=["coverage", "avg_length"])
    nuisance_pass = False
    details = []
    for lam in [0.5, 1.0, 1.5]:
        try:
            if_cov = float(pivot.loc[lam, ("coverage", "if_learned")])
            if_len = float(pivot.loc[lam, ("avg_length", "if_learned")])
            a_len = float(pivot.loc[lam, ("avg_length", "audit_only")])
            ok = (if_cov >= 0.90) and (if_len < a_len)
            nuisance_pass = nuisance_pass or ok
            details.append(f"lambda={lam}: IF cov {if_cov:.3f}, IF len {if_len:.3f}, audit len {a_len:.3f}")
        except KeyError:
            details.append(f"lambda={lam}: unavailable")
    rows.append(
        {
            "gate": "nuisance_usefulness",
            "passed": nuisance_pass,
            "criterion": "At least one middle lambda has IF length < audit length and pilot IF coverage >= 0.90 at m=500.",
            "value": "; ".join(details),
            "interpretation": "Synthetic nuisances appear useful for audit-corrected inference."
            if nuisance_pass
            else "Synthetic nuisances did not clearly improve over audit-only in the pilot.",
        }
    )

    valid = summary[
        (summary["m"] == 500)
        & (summary["method"] == "selected_feasible")
        & (summary["sensitivity_c"].isin([0.5, 1.0, 1.5, 2.0]))
        & (summary["coverage"] >= 0.90)
    ]
    selected_not_all_audit = False
    selected_details = []
    for _, row in valid.iterrows():
        br = branch_summary[
            (branch_summary["m"] == 500)
            & (branch_summary["lambda"] == row["lambda"])
            & (branch_summary["sensitivity_c"] == row["sensitivity_c"])
            & (branch_summary["selected_branch"] == "A")
        ]
        audit_freq = float(br["selected_frequency"].iloc[0]) if len(br) else 0.0
        selected_details.append(f"lambda={row['lambda']}, c={row['sensitivity_c']}: audit freq {audit_freq:.2f}")
        selected_not_all_audit = selected_not_all_audit or (audit_freq < 0.95)
    rows.append(
        {
            "gate": "selected_rule_informativeness",
            "passed": selected_not_all_audit,
            "criterion": "Among pilot-valid selected rows, at least one has audit-only selection frequency < 0.95.",
            "value": "; ".join(selected_details[:8]) if selected_details else "No selected rows with pilot coverage >= 0.90.",
            "interpretation": "Selected reporting is not merely audit-only."
            if selected_not_all_audit
            else "Selected reporting mostly falls back to audit-only.",
        }
    )

    gate_df = pd.DataFrame(rows)
    failed = int((~gate_df["passed"]).sum())
    if failed >= 2:
        decision = "switch_to_folktables"
    elif failed == 1:
        decision = "inspect_single_failure"
    else:
        decision = "proceed_to_full_hillstrom"
    gate_df["overall_failed_gates"] = failed
    gate_df["overall_decision"] = decision
    return gate_df


def write_decision_memo(
    *,
    full_theta: float,
    full_aipw: float,
    gap_summary: pd.DataFrame,
    summary: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> None:
    decision = gate_df["overall_decision"].iloc[0]
    failed = int(gate_df["overall_failed_gates"].iloc[0])
    key_rows = summary[
        (summary["m"] == 500)
        & (summary["lambda"].isin([0.5, 1.0, 1.5]))
        & (
            ((summary["method"].isin(["audit_only", "if_learned"])) & (summary["sensitivity_c"].isna()))
            | ((summary["method"].isin(["q_inflated", "selected_feasible"])) & (summary["sensitivity_c"] == 1.0))
        )
    ][["lambda", "m", "sensitivity_c", "method_label", "coverage", "avg_length", "bias"]]

    lines = [
        "# Hillstrom Pilot Decision Memo",
        "",
        "Created: 2026-05-22",
        "",
        "Status: pilot result and decision memo, not manuscript text.",
        "",
        "## Decision",
        "",
        f"Overall decision: `{decision}`.",
        "",
        f"Failed gates: `{failed}` out of `5`.",
        "",
        "Interpretation:",
    ]
    if decision == "proceed_to_full_hillstrom":
        lines.append(
            "The pilot supports proceeding to the full Hillstrom analysis under the locked pre-analysis design."
        )
    elif decision == "inspect_single_failure":
        lines.append(
            "Exactly one gate failed. Inspect whether the failure is a useful cautionary empirical finding or a reason to redesign before proceeding."
        )
    else:
        lines.append(
            "Two or more gates failed. The locked decision rule recommends switching the primary empirical analysis to Folktables/ACS."
        )

    lines.extend(
        [
            "",
            "## Full-Data Benchmarks",
            "",
            f"- Primary full-data randomized contrast for `visit`: `{full_theta:.6f}`.",
            f"- Full-data AIPW robustness benchmark: `{full_aipw:.6f}`.",
            "",
            "The full-data randomized contrast is the primary benchmark. The AIPW value is a robustness check only.",
            "",
            "## Pilot Gates",
            "",
            gate_df[["gate", "passed", "criterion", "value", "interpretation"]].to_markdown(index=False),
            "",
            "## Generator Gap Summary",
            "",
            gap_summary.to_markdown(index=False),
            "",
            "## Key Method Rows at m = 500",
            "",
            key_rows.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Files",
            "",
            "- Replicate-level results: `analysis/results/hillstrom_pilot/hillstrom_pilot_replicates.csv`",
            "- Summary table: `tables/hillstrom_pilot/hillstrom_pilot_summary.csv`",
            "- Branch frequencies: `tables/hillstrom_pilot/hillstrom_pilot_branch_frequencies.csv`",
            "- Pilot gate table: `tables/hillstrom_pilot/hillstrom_pilot_gate.csv`",
            "- Main diagnostic figure: `figures/hillstrom_pilot/hillstrom_pilot_main.png`",
        ]
    )
    (ROOT / "analysis" / "HILLSTROM_PILOT_DECISION_MEMO.md").write_text("\n".join(lines) + "\n")


def run_pilot() -> None:
    ensure_dirs()
    df = load_data()
    rng = np.random.default_rng(SEED)
    scores = source_scores(df)
    pi = float(df["A"].mean())
    theta_full, se_full = diff_mean(df)
    full_aipw = fit_full_aipw_benchmark(df, pi)

    rows: list[dict] = []
    branch_rows: list[dict] = []
    gap_rows: list[dict] = []
    failures: list[dict] = []

    print(
        f"Hillstrom pilot: n={len(df)}, theta_full={theta_full:.6f}, "
        f"pi={pi:.6f}, reps={REPS}, M={M_SYNTH}"
        ,
        flush=True,
    )
    for lam in SOURCE_LAMBDAS:
        for rep in range(REPS):
            rep_seed = SEED + int(lam * 1000) + rep
            setting_base = {"lambda": lam, "rep": rep}
            try:
                source_idx = draw_source_indices(rng, scores, lam, N_SOURCE)
                source_df = df.iloc[source_idx].copy()
                synthesizer, fit_time = fit_generator(source_df, rep_seed)
                synth_df = sample_synthetic(synthesizer, M_SYNTH)
                q_point, q_se = diff_mean(synth_df)
                direct_gap = abs(q_point - theta_full)
                model0, model1 = fit_nuisances(synth_df, rep_seed)
                gap_rows.append(
                    {
                        "lambda": lam,
                        "rep": rep,
                        "direct_gap": direct_gap,
                        "theta_q": q_point,
                        "theta_full": theta_full,
                        "fit_seconds": fit_time,
                        "generator_failed": 0,
                    }
                )
                for m in AUDIT_SIZES:
                    audit_idx = draw_audit_indices(rng, len(df), m)
                    audit_df = df.iloc[audit_idx].copy()
                    audit_point, audit_se = diff_mean(audit_df)
                    if_point, if_se = if_estimate(audit_df, model0, model1, pi)
                    pooled_point, pooled_se = diff_mean(pooled_frame(audit_df, synth_df))
                    tcrit = t.ppf(1.0 - ALPHA / 2.0, df=max(m - 2, 1))
                    zcrit = Z

                    fixed_setting = {
                        **setting_base,
                        "m": m,
                        "sensitivity_c": np.nan,
                        "theta_full": theta_full,
                        "theta_full_se": se_full,
                        "theta_q": q_point,
                        "direct_gap": direct_gap,
                    }
                    lo, hi = interval(audit_point, audit_se, tcrit)
                    add_eval(rows, setting=fixed_setting, method="audit_only", point=audit_point, lo=lo, hi=hi, theta_full=theta_full)
                    lo, hi = interval(q_point, q_se, zcrit)
                    add_eval(rows, setting=fixed_setting, method="synthetic_naive", point=q_point, lo=lo, hi=hi, theta_full=theta_full)
                    lo, hi = interval(pooled_point, pooled_se, zcrit)
                    add_eval(rows, setting=fixed_setting, method="pooled_naive", point=pooled_point, lo=lo, hi=hi, theta_full=theta_full)
                    lo, hi = interval(if_point, IF_SE_MULTIPLIER * if_se, tcrit)
                    add_eval(rows, setting=fixed_setting, method="if_learned", point=if_point, lo=lo, hi=hi, theta_full=theta_full)

                    candidate_alpha = ALPHA / len(BONF_BRANCHES)
                    z_bonf = norm.ppf(1.0 - candidate_alpha / 2.0)
                    t_bonf = t.ppf(1.0 - candidate_alpha / 2.0, df=max(m - 2, 1))
                    a_lo_b, a_hi_b = interval(audit_point, audit_se, t_bonf)
                    if_lo_b, if_hi_b = interval(if_point, IF_SE_MULTIPLIER * if_se, t_bonf)

                    for c in SENSITIVITY_C:
                        radius = c * audit_se
                        c_setting = {**fixed_setting, "sensitivity_c": c, "supplied_radius": radius}
                        q_lo, q_hi = interval(q_point, q_se, zcrit, radius)
                        add_eval(rows, setting=c_setting, method="q_inflated", point=q_point, lo=q_lo, hi=q_hi, theta_full=theta_full)

                        q_lo_b, q_hi_b = interval(q_point, q_se, z_bonf, radius)
                        candidates = {
                            "A": (audit_point, a_lo_b, a_hi_b),
                            "Q": (q_point, q_lo_b, q_hi_b),
                            "IF": (if_point, if_lo_b, if_hi_b),
                        }
                        lengths = {name: val[2] - val[1] for name, val in candidates.items()}
                        branch = min(lengths, key=lengths.get)
                        selected_point, selected_lo, selected_hi = candidates[branch]
                        add_eval(
                            rows,
                            setting=c_setting,
                            method="selected_feasible",
                            point=selected_point,
                            lo=selected_lo,
                            hi=selected_hi,
                            theta_full=theta_full,
                        )
                        branch_rows.append(
                            {
                                "lambda": lam,
                                "rep": rep,
                                "m": m,
                                "sensitivity_c": c,
                                "supplied_radius": radius,
                                "direct_gap": direct_gap,
                                "selected_branch": branch,
                            }
                        )
            except Exception as exc:
                failures.append({"lambda": lam, "rep": rep, "error": repr(exc)})
                gap_rows.append(
                    {
                        "lambda": lam,
                        "rep": rep,
                        "direct_gap": np.nan,
                        "theta_q": np.nan,
                        "theta_full": theta_full,
                        "fit_seconds": np.nan,
                        "generator_failed": 1,
                    }
                )
            if (rep + 1) % 25 == 0:
                print(f"  lambda={lam}: finished {rep + 1}/{REPS}", flush=True)

    rep_df = pd.DataFrame(rows)
    branch_df = pd.DataFrame(branch_rows)
    gap_df = pd.DataFrame(gap_rows)
    failure_df = pd.DataFrame(failures, columns=["lambda", "rep", "error"])

    summary = summarize_results(rep_df)
    branch_summary = summarize_branches(branch_df)
    gap_summary = gap_df.groupby("lambda", as_index=False).agg(
        generator_failures=("generator_failed", "sum"),
        direct_gap_mean=("direct_gap", "mean"),
        direct_gap_sd=("direct_gap", "std"),
        theta_q_mean=("theta_q", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )
    gate_df = pilot_gate(full_theta=theta_full, gap_df=gap_df.dropna(subset=["direct_gap"]), summary=summary, branch_summary=branch_summary)

    rep_df.to_csv(RESULT_DIR / "hillstrom_pilot_replicates.csv", index=False)
    branch_df.to_csv(RESULT_DIR / "hillstrom_pilot_branch_replicates.csv", index=False)
    gap_df.to_csv(RESULT_DIR / "hillstrom_pilot_generator_gaps.csv", index=False)
    failure_df.to_csv(RESULT_DIR / "hillstrom_pilot_failures.csv", index=False)
    summary.to_csv(RESULT_DIR / "hillstrom_pilot_summary.csv", index=False)
    branch_summary.to_csv(RESULT_DIR / "hillstrom_pilot_branch_frequencies.csv", index=False)
    gap_summary.to_csv(RESULT_DIR / "hillstrom_pilot_gap_summary.csv", index=False)
    gate_df.to_csv(RESULT_DIR / "hillstrom_pilot_gate.csv", index=False)

    summary.to_csv(TABLE_DIR / "hillstrom_pilot_summary.csv", index=False)
    branch_summary.to_csv(TABLE_DIR / "hillstrom_pilot_branch_frequencies.csv", index=False)
    gap_summary.to_csv(TABLE_DIR / "hillstrom_pilot_gap_summary.csv", index=False)
    gate_df.to_csv(TABLE_DIR / "hillstrom_pilot_gate.csv", index=False)

    plot_main(summary, gap_df.dropna(subset=["direct_gap"]), branch_summary)
    write_decision_memo(
        full_theta=theta_full,
        full_aipw=full_aipw,
        gap_summary=gap_summary,
        summary=summary,
        gate_df=gate_df,
    )
    print(gate_df[["gate", "passed", "value", "overall_decision"]].to_string(index=False))


if __name__ == "__main__":
    run_pilot()
