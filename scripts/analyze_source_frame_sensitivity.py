"""Analyze selection-frame sensitivity in the 5,000-prompt benchmark.

The final benchmark combines prompts retained from the earlier Stage C frame
with prompts selected from a later Stage D candidate frame. This script joins
that prompt-level selection metadata to the final four-judge composite and the
21-model mean-pooled outcome, then reports:

1. strict mapping and panel-coverage validation;
2. the unadjusted and source-controlled sup-Wald searches;
3. linear, source-specific-slope, and piecewise fit comparisons;
4. separate unbinned breakpoint tests within each selection frame; and
5. a decomposition of the original 13.75 regime gap by selection frame.

The controlled search intentionally matches the implementation used by
``src/rebuttal/09_threshold_with_task_type.py``. The pooled null includes a
source-frame dummy alongside the composite. Each split regression includes
the same columns, so the alternative permits the source coefficient to differ
between regimes. The report also includes specifications with source-specific
linear slopes so this behavior is visible rather than hidden.

Raw prompt and generation bundles are not part of the anonymous Git snapshot.
Point ``--data-root`` to the retained local ``data`` directory when necessary.

Example:

    python scripts/analyze_source_frame_sensitivity.py \
      --data-root D:/path/to/ComplexityKinkResearch/data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_kink import (  # noqa: E402
    build_combined_df,
    discover_models,
    load_rubric_scores,
    load_scored_model,
)
from config import MIN_REGIME_SIZE, STAGE_C_EXCLUDED_MODELS  # noqa: E402
from run_stage2_iv import build_threshold_grid  # noqa: E402


EXPECTED_N = 5_000
EXPECTED_MODELS = 21
EXPECTED_SOURCES = {
    "stage_c_existing": 2_246,
    "stage_d_candidate": 2_754,
}
SOURCE_LABELS = {
    "stage_c_existing": "retained_earlier_frame",
    "stage_d_candidate": "later_candidate_frame",
}


@dataclass(frozen=True)
class CandidateDesign:
    gamma: float
    low_mask: np.ndarray
    q_low: np.ndarray
    q_high: np.ndarray


@dataclass(frozen=True)
class ThresholdDesign:
    q_pool: np.ndarray
    candidates: tuple[CandidateDesign, ...]
    n: int
    k_regime: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Retained raw data directory. Defaults to <repository>/data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "source_frame_sensitivity.json",
        help="Machine-readable JSON output path.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=300,
        help="Wild-bootstrap draws per threshold test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _basis(matrix: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the column space of ``matrix``."""
    q, r = np.linalg.qr(np.asarray(matrix, dtype=float), mode="reduced")
    scale = np.max(np.abs(np.diag(r))) if r.size else 0.0
    tolerance = np.finfo(float).eps * max(matrix.shape) * max(scale, 1.0)
    rank = int(np.sum(np.abs(np.diag(r)) > tolerance))
    return q[:, :rank]


def _rss(y: np.ndarray, q: np.ndarray) -> float:
    projected = q.T @ y
    value = float(y @ y - projected @ projected)
    return max(value, 0.0)


def _design_matrix(
    frame: pd.DataFrame,
    control_cols: Sequence[str],
) -> np.ndarray:
    columns = [
        np.ones(len(frame), dtype=float),
        frame["composite"].to_numpy(dtype=float),
    ]
    columns.extend(frame[column].to_numpy(dtype=float) for column in control_cols)
    return np.column_stack(columns)


def prepare_threshold_design(
    frame: pd.DataFrame,
    control_cols: Sequence[str],
) -> ThresholdDesign:
    """Precompute projection bases for a deterministic sup-Wald search."""
    x = frame["composite"].to_numpy(dtype=float)
    pooled_matrix = _design_matrix(frame, control_cols)
    q_pool = _basis(pooled_matrix)
    k_regime = pooled_matrix.shape[1]
    candidates = []
    for gamma in build_threshold_grid(frame, "composite"):
        low_mask = x <= float(gamma)
        high_mask = ~low_mask
        if low_mask.sum() < MIN_REGIME_SIZE or high_mask.sum() < MIN_REGIME_SIZE:
            continue
        q_low = _basis(pooled_matrix[low_mask])
        q_high = _basis(pooled_matrix[high_mask])
        if q_low.shape[1] != k_regime or q_high.shape[1] != k_regime:
            continue
        candidates.append(
            CandidateDesign(
                gamma=float(gamma),
                low_mask=low_mask,
                q_low=q_low,
                q_high=q_high,
            )
        )
    if not candidates:
        raise ValueError("No threshold candidate satisfies the regime-size rule.")
    return ThresholdDesign(
        q_pool=q_pool,
        candidates=tuple(candidates),
        n=len(frame),
        k_regime=k_regime,
    )


def threshold_curve(
    y: np.ndarray,
    design: ThresholdDesign,
) -> list[tuple[float, float]]:
    rss_pool = _rss(y, design.q_pool)
    curve = []
    for candidate in design.candidates:
        low_mask = candidate.low_mask
        rss_split = _rss(y[low_mask], candidate.q_low) + _rss(
            y[~low_mask], candidate.q_high
        )
        if rss_split <= 0:
            continue
        numerator = (rss_pool - rss_split) / design.k_regime
        denominator = rss_split / (
            design.n - 2 * design.k_regime
        )
        curve.append((candidate.gamma, float(numerator / denominator)))
    return curve


def wild_bootstrap_threshold(
    frame: pd.DataFrame,
    control_cols: Sequence[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Run the manuscript-style threshold search and wild bootstrap."""
    design = prepare_threshold_design(frame, control_cols)
    y = frame["pass_rate"].to_numpy(dtype=float)
    curve = threshold_curve(y, design)
    gamma, sup_wald = max(curve, key=lambda item: item[1])
    low = frame[frame["composite"] <= gamma]
    high = frame[frame["composite"] > gamma]

    fitted = design.q_pool @ (design.q_pool.T @ y)
    residual = y - fitted
    rng = np.random.RandomState(seed)
    boot_sup_wald = []
    for _ in range(n_boot):
        weights = rng.choice([-1.0, 1.0], size=len(frame))
        y_boot = fitted + residual * weights
        boot_curve = threshold_curve(y_boot, design)
        boot_sup_wald.append(max(value for _, value in boot_curve))

    bootstrap = np.asarray(boot_sup_wald, dtype=float)
    exceedances = int(np.sum(bootstrap >= sup_wald))
    return {
        "threshold": gamma,
        "sup_wald": sup_wald,
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "mean_pass_low": float(low["pass_rate"].mean()),
        "mean_pass_high": float(high["pass_rate"].mean()),
        "raw_regime_gap": float(
            high["pass_rate"].mean() - low["pass_rate"].mean()
        ),
        "threshold_grid_candidates": len(design.candidates),
        "wild_bootstrap": {
            "seed": seed,
            "draws": n_boot,
            "exceedances": exceedances,
            "p_raw": float(exceedances / n_boot),
            "p_finite_mc": float((exceedances + 1) / (n_boot + 1)),
            "q95": float(np.quantile(bootstrap, 0.95)),
            "maximum": float(np.max(bootstrap)),
        },
    }


def fit_summary(y: pd.Series, matrix: pd.DataFrame | np.ndarray) -> dict:
    result = sm.OLS(y, sm.add_constant(matrix, has_constant="add")).fit()
    return {
        "n_parameters": int(len(result.params)),
        "r2": float(result.rsquared),
        "adjusted_r2": float(result.rsquared_adj),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "rss": float(result.ssr),
    }


def piecewise_matrix(
    frame: pd.DataFrame,
    gamma: float,
    base_cols: Sequence[str],
    source_regime_interaction: bool = False,
) -> pd.DataFrame:
    x = frame["composite"].to_numpy(dtype=float)
    high = (x > gamma).astype(float)
    matrix = pd.DataFrame(
        {column: frame[column].to_numpy(dtype=float) for column in base_cols}
    )
    matrix["regime_jump"] = high
    matrix["slope_change"] = (x - gamma) * high
    if source_regime_interaction:
        matrix["source_regime_interaction"] = (
            frame["later_candidate"].to_numpy(dtype=float) * high
        )
    return matrix


def best_piecewise_fit(
    frame: pd.DataFrame,
    base_cols: Sequence[str],
    source_regime_interaction: bool = False,
) -> dict:
    y = frame["pass_rate"]
    candidates = []
    for gamma in build_threshold_grid(frame, "composite"):
        low_n = int((frame["composite"] <= gamma).sum())
        high_n = len(frame) - low_n
        if low_n < MIN_REGIME_SIZE or high_n < MIN_REGIME_SIZE:
            continue
        matrix = piecewise_matrix(
            frame,
            float(gamma),
            base_cols,
            source_regime_interaction=source_regime_interaction,
        )
        summary = fit_summary(y, matrix)
        candidates.append((float(gamma), summary))
    gamma, summary = min(candidates, key=lambda item: item[1]["rss"])
    return {"threshold": gamma, **summary}


def load_analysis_frame(data_root: Path) -> tuple[pd.DataFrame, dict]:
    stage_d = data_root / "stage_d"
    rubric_path = stage_d / "ensemble_scores_current_aggregated.jsonl"
    scored_dir = stage_d / "scored_combined"
    prompt_path = stage_d / "stage_d_prompts.jsonl"

    for path in (rubric_path, scored_dir, prompt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rubric = load_rubric_scores(rubric_path)
    model_frames = {}
    for model_name, model_path in discover_models(scored_dir):
        if model_name in STAGE_C_EXCLUDED_MODELS:
            continue
        model_frames[model_name] = load_scored_model(model_path, rubric)

    model_row_counts = {
        model_name: int(len(frame))
        for model_name, frame in sorted(model_frames.items())
    }
    combined = build_combined_df(model_frames)

    prompt_metadata = {}
    prompt_rows = 0
    duplicate_prompt_ids = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prompt_rows += 1
            record = json.loads(line)
            prompt_id = record["prompt_id"]
            if prompt_id in prompt_metadata:
                duplicate_prompt_ids.append(prompt_id)
            prompt_metadata[prompt_id] = record.get("selection_source")

    combined["selection_source"] = combined["prompt_id"].map(prompt_metadata)
    missing_mapping = int(combined["selection_source"].isna().sum())
    source_counts = {
        str(key): int(value)
        for key, value in combined["selection_source"].value_counts().items()
    }
    unexpected_sources = sorted(set(source_counts) - set(EXPECTED_SOURCES))

    validation = {
        "expected_prompts": EXPECTED_N,
        "analyzed_prompts": int(len(combined)),
        "prompt_metadata_rows": prompt_rows,
        "duplicate_prompt_ids": len(duplicate_prompt_ids),
        "missing_source_mappings": missing_mapping,
        "expected_models": EXPECTED_MODELS,
        "analyzed_models": int(len(model_frames)),
        "model_row_counts": model_row_counts,
        "source_counts": source_counts,
        "expected_source_counts": EXPECTED_SOURCES,
        "unexpected_sources": unexpected_sources,
    }

    failures = []
    if len(combined) != EXPECTED_N:
        failures.append(f"analyzed prompt count is {len(combined)}, expected {EXPECTED_N}")
    if prompt_rows != EXPECTED_N:
        failures.append(f"metadata row count is {prompt_rows}, expected {EXPECTED_N}")
    if duplicate_prompt_ids:
        failures.append("prompt metadata contains duplicate prompt ids")
    if missing_mapping:
        failures.append(f"{missing_mapping} analyzed prompts lack source mappings")
    if len(model_frames) != EXPECTED_MODELS:
        failures.append(
            f"analyzed model count is {len(model_frames)}, expected {EXPECTED_MODELS}"
        )
    bad_model_counts = {
        model: count
        for model, count in model_row_counts.items()
        if count != EXPECTED_N
    }
    if bad_model_counts:
        failures.append(f"model row counts differ from {EXPECTED_N}: {bad_model_counts}")
    if source_counts != EXPECTED_SOURCES:
        failures.append(
            f"source counts are {source_counts}, expected {EXPECTED_SOURCES}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))

    combined["later_candidate"] = (
        combined["selection_source"] == "stage_d_candidate"
    ).astype(float)
    combined["source_by_composite"] = (
        combined["later_candidate"] * combined["composite"]
    )
    return combined, validation


def source_descriptives(frame: pd.DataFrame) -> dict:
    output = {}
    for source, subset in frame.groupby("selection_source"):
        output[SOURCE_LABELS[source]] = {
            "selection_source_value": source,
            "n": int(len(subset)),
            "mean_composite": float(subset["composite"].mean()),
            "min_composite": float(subset["composite"].min()),
            "max_composite": float(subset["composite"].max()),
            "mean_pass": float(subset["pass_rate"].mean()),
        }
    return output


def original_threshold_decomposition(
    frame: pd.DataFrame,
    gamma: float,
) -> dict:
    working = frame.copy()
    working["regime"] = np.where(
        working["composite"] <= gamma,
        "at_or_below",
        "above",
    )
    source_weights = working["selection_source"].value_counts(normalize=True)
    cells = {}
    cell_means = {}
    for (source, regime), subset in working.groupby(
        ["selection_source", "regime"]
    ):
        label = SOURCE_LABELS[source]
        cells.setdefault(label, {})[regime] = {
            "n": int(len(subset)),
            "mean_pass": float(subset["pass_rate"].mean()),
        }
        cell_means[(source, regime)] = float(subset["pass_rate"].mean())

    standardized = {}
    for regime in ("at_or_below", "above"):
        standardized[regime] = float(
            sum(
                source_weights[source] * cell_means[(source, regime)]
                for source in source_weights.index
            )
        )

    low = working[working["regime"] == "at_or_below"]
    high = working[working["regime"] == "above"]
    return {
        "threshold": gamma,
        "raw": {
            "n_at_or_below": int(len(low)),
            "n_above": int(len(high)),
            "mean_pass_at_or_below": float(low["pass_rate"].mean()),
            "mean_pass_above": float(high["pass_rate"].mean()),
            "gap_above_minus_at_or_below": float(
                high["pass_rate"].mean() - low["pass_rate"].mean()
            ),
            "later_candidate_share_at_or_below": float(
                low["later_candidate"].mean()
            ),
            "later_candidate_share_above": float(
                high["later_candidate"].mean()
            ),
        },
        "within_source_cells": cells,
        "overall_source_weights": {
            SOURCE_LABELS[source]: float(weight)
            for source, weight in source_weights.items()
        },
        "source_standardized_to_overall_weights": {
            "mean_pass_at_or_below": standardized["at_or_below"],
            "mean_pass_above": standardized["above"],
            "gap_above_minus_at_or_below": float(
                standardized["above"] - standardized["at_or_below"]
            ),
            "caution": (
                "The retained-earlier frame has limited support above 13.75, "
                "so this standardization is an overlap diagnostic rather than "
                "a population estimate."
            ),
        },
    }


def analyze(args: argparse.Namespace) -> dict:
    frame, validation = load_analysis_frame(args.data_root.resolve())
    y = frame["pass_rate"]

    uncontrolled = wild_bootstrap_threshold(
        frame,
        control_cols=[],
        n_boot=args.n_boot,
        seed=args.seed,
    )
    source_controlled = wild_bootstrap_threshold(
        frame,
        control_cols=["later_candidate"],
        n_boot=args.n_boot,
        seed=args.seed,
    )

    fit_comparisons = {
        "source_only": fit_summary(y, frame[["later_candidate"]]),
        "composite_only_linear": fit_summary(y, frame[["composite"]]),
        "source_plus_composite_linear": fit_summary(
            y,
            frame[["later_candidate", "composite"]],
        ),
        "source_specific_linear_slopes": fit_summary(
            y,
            frame[
                [
                    "later_candidate",
                    "composite",
                    "source_by_composite",
                ]
            ],
        ),
    }
    fit_comparisons["unadjusted_piecewise"] = best_piecewise_fit(
        frame,
        base_cols=["composite"],
    )
    fit_comparisons["common_source_fe_piecewise"] = best_piecewise_fit(
        frame,
        base_cols=["later_candidate", "composite"],
    )
    fit_comparisons[
        "task_style_source_controlled_split"
    ] = {
        "threshold": source_controlled["threshold"],
        **fit_summary(
            y,
            piecewise_matrix(
                frame,
                source_controlled["threshold"],
                base_cols=["later_candidate", "composite"],
                source_regime_interaction=True,
            ),
        ),
    }
    fit_comparisons[
        "source_specific_linear_plus_piecewise"
    ] = best_piecewise_fit(
        frame,
        base_cols=[
            "later_candidate",
            "composite",
            "source_by_composite",
        ],
    )

    within_frame = {}
    for source, subset in frame.groupby("selection_source"):
        result = wild_bootstrap_threshold(
            subset.reset_index(drop=True),
            control_cols=[],
            n_boot=args.n_boot,
            seed=args.seed,
        )
        result["linear_fit"] = fit_summary(
            subset["pass_rate"],
            subset[["composite"]],
        )
        result["piecewise_fit"] = best_piecewise_fit(
            subset.reset_index(drop=True),
            base_cols=["composite"],
        )
        within_frame[SOURCE_LABELS[source]] = result

    return {
        "schema_version": 1,
        "inputs": {
            "prompt_metadata": "data/stage_d/stage_d_prompts.jsonl",
            "source_mapping_field": "selection_source",
            "rubric": (
                "data/stage_d/ensemble_scores_current_aggregated.jsonl"
            ),
            "scored_models": "data/stage_d/scored_combined/*.jsonl",
            "join_key": "prompt_id",
        },
        "validation": validation,
        "source_descriptives": source_descriptives(frame),
        "method": {
            "minimum_regime_size": MIN_REGIME_SIZE,
            "bootstrap_draws": args.n_boot,
            "bootstrap_seed": args.seed,
            "source_controlled_search": (
                "Matches src/rebuttal/09_threshold_with_task_type.py: "
                "the pooled null and both regime regressions include the "
                "composite and source-frame dummy. The split alternative "
                "therefore permits the source coefficient to differ by regime."
            ),
            "regime_means": (
                "Reported regime means are raw summaries at each selected "
                "threshold, matching the manuscript's task-type sensitivity."
            ),
        },
        "unadjusted_threshold": uncontrolled,
        "source_controlled_threshold": source_controlled,
        "sup_wald_attenuation_ratio": float(
            source_controlled["sup_wald"] / uncontrolled["sup_wald"]
        ),
        "fit_comparisons": fit_comparisons,
        "within_frame_thresholds": within_frame,
        "original_13_75_decomposition": original_threshold_decomposition(
            frame,
            gamma=13.75,
        ),
        "interpretive_checks": {
            "source_specific_linear_r2_minus_source_linear_r2": float(
                fit_comparisons["source_specific_linear_slopes"]["r2"]
                - fit_comparisons["source_plus_composite_linear"]["r2"]
            ),
            "task_style_split_r2_minus_source_specific_linear_r2": float(
                fit_comparisons["task_style_source_controlled_split"]["r2"]
                - fit_comparisons["source_specific_linear_slopes"]["r2"]
            ),
            "later_frame_raw_regime_gap": within_frame[
                "later_candidate_frame"
            ]["raw_regime_gap"],
            "retained_frame_breakpoint_bootstrap_p_finite_mc": within_frame[
                "retained_earlier_frame"
            ]["wild_bootstrap"]["p_finite_mc"],
        },
    }


def main() -> None:
    args = parse_args()
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    report = analyze(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source = report["source_controlled_threshold"]
    retained = report["within_frame_thresholds"]["retained_earlier_frame"]
    later = report["within_frame_thresholds"]["later_candidate_frame"]
    print(f"Wrote {output}")
    print(
        "Source-controlled: "
        f"gamma={source['threshold']:.2f}, "
        f"sup-Wald={source['sup_wald']:.2f}, "
        f"gap={source['raw_regime_gap']:+.4f}, "
        f"p_MC={source['wild_bootstrap']['p_finite_mc']:.6f}"
    )
    print(
        "Retained earlier frame: "
        f"gamma={retained['threshold']:.2f}, "
        f"sup-Wald={retained['sup_wald']:.2f}, "
        f"p_MC={retained['wild_bootstrap']['p_finite_mc']:.6f}"
    )
    print(
        "Later candidate frame: "
        f"gamma={later['threshold']:.2f}, "
        f"sup-Wald={later['sup_wald']:.2f}, "
        f"gap={later['raw_regime_gap']:+.4f}, "
        f"p_MC={later['wild_bootstrap']['p_finite_mc']:.6f}"
    )


if __name__ == "__main__":
    main()
