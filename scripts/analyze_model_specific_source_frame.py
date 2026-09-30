"""Audit construction-frame sensitivity for downward model-specific fits.

The unadjusted original-benchmark analysis contains five models whose raw mean
pass rate is lower above their selected breakpoint. This script applies the
same construction-frame audit to all five. It does not select a preferred
model after inspecting the source-specific results.

The output reports:

1. the locked unadjusted fit used to define the five-model audit set;
2. a construction-frame-controlled threshold search;
3. separate threshold searches in the retained-earlier and later-candidate
   frames;
4. wild-bootstrap exceedances for each search;
5. pairs-bootstrap intervals for each within-frame threshold; and
6. source-specific linear and piecewise BIC comparisons.

Raw prompt and generation bundles are not part of the anonymous Git snapshot.
Point ``--data-root`` to the retained local ``data`` directory when necessary.

Example:

    python scripts/analyze_model_specific_source_frame.py \
      --data-root D:/path/to/ComplexityKinkResearch/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_kink import (  # noqa: E402
    compute_wald,
    discover_models,
    load_rubric_scores,
    load_scored_model,
)
from analyze_source_frame_sensitivity import (  # noqa: E402
    EXPECTED_N,
    EXPECTED_SOURCES,
    SOURCE_LABELS,
    best_piecewise_fit,
    fit_summary,
    wild_bootstrap_threshold,
)
from config import STAGE_C_EXCLUDED_MODELS  # noqa: E402
from run_stage2_iv import build_threshold_grid  # noqa: E402


DISPLAY_NAMES = {
    "azure_grok-3": "Grok-3",
    "azure_kimi-k2.5": "Kimi K2.5",
    "google_gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "openai_gpt-5.4": "GPT-5.4",
    "qwen_qwen3.6-plus": "Qwen 3.6 Plus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Retained raw data directory. Defaults to <repository>/data.",
    )
    parser.add_argument(
        "--pooling-report",
        type=Path,
        default=(
            ROOT
            / "results"
            / "rebuttal"
            / "pooling_submitted"
            / "pooling_and_kappa_report.json"
        ),
        help="Locked report used to select every negative unadjusted fit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "model_specific_source_frame_sensitivity.json",
    )
    parser.add_argument(
        "--expect-reviewed-set",
        action="store_true",
        help="Require the five downward fits of the reviewed outcome (a reproduction check).",
    )
    parser.add_argument("--n-wild", type=int, default=300)
    parser.add_argument("--n-pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source_mapping(data_root: Path) -> dict[str, str]:
    prompt_path = data_root / "stage_d" / "stage_d_prompts.jsonl"
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    mapping: dict[str, str] = {}
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_id = str(record["prompt_id"])
            if prompt_id in mapping:
                raise RuntimeError(f"Duplicate prompt_id in metadata: {prompt_id}")
            mapping[prompt_id] = str(record.get("selection_source"))
    if len(mapping) != EXPECTED_N:
        raise RuntimeError(
            f"Source metadata contains {len(mapping)} prompts, expected {EXPECTED_N}"
        )
    return mapping


def selected_downward_models(pooling_report: Path, expect_reviewed_set: bool) -> list[dict]:
    report = json.loads(pooling_report.read_text(encoding="utf-8"))
    selected = [
        row for row in report["per_model"] if row.get("direction") == "down"
    ]
    selected_names = {str(row["model"]) for row in selected}
    expected_names = set(DISPLAY_NAMES)
    if expect_reviewed_set and selected_names != expected_names:
        raise RuntimeError(
            "Downward-fit selection changed: "
            f"observed={sorted(selected_names)}, expected={sorted(expected_names)}"
        )
    return selected


def load_model_frames(
    data_root: Path,
    selected_names: set[str],
    source_mapping: dict[str, str],
) -> dict[str, pd.DataFrame]:
    stage_d = data_root / "stage_d"
    rubric_path = stage_d / "ensemble_scores_current_aggregated.jsonl"
    # CK_SCORED_DIR selects the outcome definition, e.g. data/stage_d/scored_independent_audit.
    scored_dir = Path(os.environ.get("CK_SCORED_DIR", stage_d / "scored_combined"))
    for path in (rubric_path, scored_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    rubric = load_rubric_scores(rubric_path)
    discovered = {
        model_name: model_path
        for model_name, model_path in discover_models(scored_dir)
        if model_name not in STAGE_C_EXCLUDED_MODELS
    }
    missing = selected_names - set(discovered)
    if missing:
        raise RuntimeError(f"Missing scored model files: {sorted(missing)}")

    output = {}
    for model_name in sorted(selected_names):
        frame = load_scored_model(discovered[model_name], rubric).copy()
        if len(frame) != EXPECTED_N or frame["prompt_id"].nunique() != EXPECTED_N:
            raise RuntimeError(
                f"{model_name} has invalid panel coverage: "
                f"rows={len(frame)}, unique_prompts={frame['prompt_id'].nunique()}"
            )
        frame["selection_source"] = frame["prompt_id"].map(source_mapping)
        if frame["selection_source"].isna().any():
            raise RuntimeError(f"{model_name} has missing source mappings")
        source_counts = frame["selection_source"].value_counts().to_dict()
        if source_counts != EXPECTED_SOURCES:
            raise RuntimeError(
                f"{model_name} source counts are {source_counts}, "
                f"expected {EXPECTED_SOURCES}"
            )
        frame["later_candidate"] = (
            frame["selection_source"] == "stage_d_candidate"
        ).astype(float)
        frame["source_by_composite"] = (
            frame["later_candidate"] * frame["composite"]
        )
        output[model_name] = frame.reset_index(drop=True)
    return output


def pairs_bootstrap_interval(
    frame: pd.DataFrame,
    control_cols: Sequence[str],
    draws: int,
    seed: int,
) -> dict:
    if control_cols:
        raise ValueError(
            "The manuscript-style pairs bootstrap is used only within frames"
        )
    grid = list(build_threshold_grid(frame, "composite"))
    thresholds = []
    failed_draws = 0
    for draw in range(draws):
        sample = frame.sample(
            frac=1.0,
            replace=True,
            random_state=seed + draw,
        ).reset_index(drop=True)
        curve = [
            (float(gamma), compute_wald(sample, gamma, "composite"))
            for gamma in grid
        ]
        valid = [
            (gamma, value)
            for gamma, value in curve
            if not np.isnan(value)
        ]
        if not valid:
            failed_draws += 1
            continue
        thresholds.append(float(max(valid, key=lambda item: item[1])[0]))

    if not thresholds:
        raise RuntimeError("Every pairs-bootstrap threshold draw failed")
    values = np.asarray(thresholds, dtype=float)
    return {
        "seed": seed,
        "requested_draws": draws,
        "successful_draws": int(len(values)),
        "failed_draws": failed_draws,
        "ci_percentiles": [2.5, 97.5],
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def analyze_model(
    model_name: str,
    frame: pd.DataFrame,
    unadjusted: dict,
    n_wild: int,
    n_pairs: int,
    seed: int,
) -> dict:
    source_controlled = wild_bootstrap_threshold(
        frame,
        control_cols=["later_candidate"],
        n_boot=n_wild,
        seed=seed,
    )

    source_specific_linear = fit_summary(
        frame["pass_rate"],
        frame[["later_candidate", "composite", "source_by_composite"]],
    )
    source_specific_piecewise = best_piecewise_fit(
        frame,
        base_cols=["later_candidate", "composite", "source_by_composite"],
    )

    within_frame = {}
    for source_value, subset in frame.groupby("selection_source"):
        subset = subset.reset_index(drop=True)
        result = wild_bootstrap_threshold(
            subset,
            control_cols=[],
            n_boot=n_wild,
            seed=seed,
        )
        result["pairs_bootstrap_threshold"] = pairs_bootstrap_interval(
            subset,
            control_cols=[],
            draws=n_pairs,
            seed=seed,
        )
        result["linear_fit"] = fit_summary(
            subset["pass_rate"],
            subset[["composite"]],
        )
        result["piecewise_fit"] = best_piecewise_fit(
            subset,
            base_cols=["composite"],
        )
        within_frame[SOURCE_LABELS[source_value]] = result

    return {
        "model_id": model_name,
        "display_name": DISPLAY_NAMES.get(model_name, model_name),
        "selection_basis_unadjusted_fit": unadjusted,
        "source_controlled_threshold": source_controlled,
        "source_specific_fit_comparison": {
            "linear_slopes": source_specific_linear,
            "linear_slopes_plus_piecewise": source_specific_piecewise,
            "piecewise_minus_linear_bic": float(
                source_specific_piecewise["bic"]
                - source_specific_linear["bic"]
            ),
        },
        "within_frame_thresholds": within_frame,
    }


def analyze(args: argparse.Namespace) -> dict:
    data_root = args.data_root.resolve()
    pooling_report = args.pooling_report.resolve()
    if not pooling_report.exists():
        raise FileNotFoundError(pooling_report)

    selected = selected_downward_models(pooling_report, args.expect_reviewed_set)
    selected_by_name = {str(row["model"]): row for row in selected}
    source_mapping = load_source_mapping(data_root)
    frames = load_model_frames(
        data_root,
        set(selected_by_name),
        source_mapping,
    )

    models = []
    for model_name in sorted(frames):
        print(f"Analyzing {DISPLAY_NAMES.get(model_name, model_name)}...", flush=True)
        models.append(
            analyze_model(
                model_name,
                frames[model_name],
                selected_by_name[model_name],
                n_wild=args.n_wild,
                n_pairs=args.n_pairs,
                seed=args.seed,
            )
        )

    return {
        "schema_version": 1,
        "selection_rule": (
            "All and only models whose locked unadjusted original-benchmark "
            "fit has a negative raw above-minus-below regime contrast."
        ),
        "selection_source": str(pooling_report.relative_to(ROOT)),
        "inputs": {
            "prompt_metadata": "data/stage_d/stage_d_prompts.jsonl",
            "source_mapping_field": "selection_source",
            "rubric": "data/stage_d/ensemble_scores_current_aggregated.jsonl",
            "scored_models": "data/stage_d/scored_combined/*.jsonl",
            "join_key": "prompt_id",
        },
        "method": {
            "wild_bootstrap_draws": args.n_wild,
            "pairs_bootstrap_draws": args.n_pairs,
            "seed": args.seed,
            "regime_means": (
                "Raw descriptive means at each selected threshold; the "
                "sup-Wald test is not a separate test of their difference."
            ),
            "interpretation": (
                "Model-specific breakpoint estimates are conditional on the "
                "constructed benchmark frame, prompt index, generation "
                "settings, and unit-test protocol."
            ),
        },
        "validation": {
            "selected_model_count": len(selected),
            "expected_selected_model_count": len(DISPLAY_NAMES),
            "analyzed_model_count": len(models),
            "expected_prompts_per_model": EXPECTED_N,
            "expected_source_counts": EXPECTED_SOURCES,
        },
        "models": models,
    }


def main() -> None:
    args = parse_args()
    if args.n_wild <= 0 or args.n_pairs <= 0:
        raise ValueError("Bootstrap draw counts must be positive")
    report = analyze(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    for model in report["models"]:
        source = model["source_controlled_threshold"]
        later = model["within_frame_thresholds"]["later_candidate_frame"]
        interval = later["pairs_bootstrap_threshold"]
        print(
            f"{model['display_name']}: "
            f"source gamma={source['threshold']:.2f}, "
            f"later gamma={later['threshold']:.2f}, "
            f"later gap={100 * later['raw_regime_gap']:+.2f} pp, "
            f"later CI=[{interval['ci_lower']:.2f},"
            f"{interval['ci_upper']:.2f}]"
        )


if __name__ == "__main__":
    main()
