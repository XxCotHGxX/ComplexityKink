"""Regenerate the scoped external-library mechanism diagnostic.

This is a descriptive, post hoc diagnostic on the retained 24-bin data. It
uses the public half-open integer display-bin convention and fits prompt-level
OLS with HC1 standard errors on original and mined-pool prompts in bins 13 and
17. It is not a causal mechanism test.

The large raw prompt and generation bundles are retained separately from the
Git snapshot. Point ``--data-root`` at the root of that retained bundle when it
is not available below this repository.

Examples:
    python scripts/regenerate_mechanism_diagnostic.py
    python scripts/regenerate_mechanism_diagnostic.py --data-root D:/data-copy
    python scripts/regenerate_mechanism_diagnostic.py --data-root D:/data-copy --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_kink import (  # noqa: E402
    build_combined_df,
    discover_models,
    load_rubric_scores,
    load_scored_model,
)
from config import STAGE_C_EXCLUDED_MODELS  # noqa: E402
from display_bins import half_open_integer_bin  # noqa: E402


ORIGINAL_SOURCE = "current_stage_d_5000"
MINED_SOURCE = "opencodeinstruct_mined_scored_pool"
DIAGNOSTIC_BINS = (13, 17)
EXPECTED_MODELS = 21
EXPECTED_PROMPTS = 459
EXPECTED_COUNTS = {
    (13, ORIGINAL_SOURCE): 228,
    (13, MINED_SOURCE): 21,
    (17, ORIGINAL_SOURCE): 39,
    (17, MINED_SOURCE): 171,
}
EXPECTED_LIBRARY_COEFFICIENT = -0.05010345682074289
EXPECTED_LIBRARY_P_VALUE = 0.6083020430717365

LIBRARY_TERMS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "scikit",
    "torch",
    "tensorflow",
    "keras",
    "matplotlib",
    "seaborn",
    "django",
    "flask",
    "fastapi",
    "sqlalchemy",
    "requests",
    "beautifulsoup",
    "bs4",
    "pytest",
    "asyncio",
    "aiohttp",
    "networkx",
    "sympy",
    "pillow",
    "PIL",
    "boto3",
    "pydantic",
    "regex",
    "itertools",
    "collections",
    "heapq",
    "bisect",
    "functools",
    "dataclasses",
)
LIBRARY_RE = re.compile(
    r"\b(" + "|".join(LIBRARY_TERMS) + r")\b", re.IGNORECASE
)
FEATURES = (
    "prompt_chars",
    "prompt_words",
    "n_unit_tests",
    "test_chars",
    "reference_cc",
    "composite_std",
    "mentions_library",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prompt_features(record: dict) -> dict:
    text = record.get("input") or ""
    tests = record.get("unit_tests")
    if isinstance(tests, list):
        test_text = "\n".join(str(test) for test in tests)
        test_count = len(tests)
    else:
        test_text = str(tests or "")
        test_count = test_text.count("assert")
    return {
        "prompt_chars": len(text),
        "prompt_words": len(text.split()),
        "n_unit_tests": test_count,
        "test_chars": len(test_text),
        "reference_cc": record.get("reference_cc"),
        "composite_std": record.get("composite_std"),
        "mentions_library": int(bool(LIBRARY_RE.search(text))),
    }


def load_prompt_metadata(path: Path) -> pd.DataFrame:
    metadata = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            prompt_id = record["prompt_id"]
            require(
                prompt_id not in metadata,
                f"Duplicate prompt metadata for {prompt_id}",
            )
            metadata[prompt_id] = {
                "source_label": record.get("source_label", "?"),
                **prompt_features(record),
            }
    frame = pd.DataFrame.from_dict(metadata, orient="index")
    frame.index.name = "prompt_id"
    return frame.reset_index()


def build_diagnostic(data_root: Path) -> dict:
    prompts_path = data_root / "data/stage_d_24bin_equal/prompts.jsonl"
    scored_dir = data_root / "data/stage_d_24bin_equal/scored_combined_final"

    rubric = load_rubric_scores(str(prompts_path))
    model_frames = {}
    for model_name, model_path in discover_models(str(scored_dir)):
        if model_name in STAGE_C_EXCLUDED_MODELS:
            continue
        frame = load_scored_model(model_path, rubric)
        if len(frame) >= 200:
            model_frames[model_name] = frame
    require(
        len(model_frames) == EXPECTED_MODELS,
        f"Expected {EXPECTED_MODELS} eligible models, found {len(model_frames)}",
    )

    combined = build_combined_df(model_frames)
    metadata = load_prompt_metadata(prompts_path)
    frame = combined.merge(metadata, on="prompt_id", how="left")
    frame["bin"] = half_open_integer_bin(frame["composite"])
    frame = frame[
        ~frame["source_label"].str.contains("synthetic", case=False, na=False)
    ]
    frame = frame[
        frame["bin"].isin(DIAGNOSTIC_BINS)
        & frame["source_label"].isin((ORIGINAL_SOURCE, MINED_SOURCE))
    ].copy()
    frame["mined"] = (frame["source_label"] == MINED_SOURCE).astype(int)
    frame = frame.dropna(subset=["pass_rate", "mined", *FEATURES])
    require(
        len(frame) == EXPECTED_PROMPTS,
        f"Expected {EXPECTED_PROMPTS} complete diagnostic prompts, "
        f"found {len(frame)}",
    )

    observed_counts = {
        (int(bin_id), source): int(count)
        for (bin_id, source), count in frame.groupby(
            ["bin", "source_label"]
        ).size().items()
    }
    require(
        observed_counts == EXPECTED_COUNTS,
        f"Unexpected diagnostic cell counts: {observed_counts}",
    )

    regressors = ["mined", *FEATURES, "composite"]
    design = sm.add_constant(frame[regressors])
    fit = sm.OLS(frame["pass_rate"], design).fit(cov_type="HC1")
    coefficient = float(fit.params["mentions_library"])
    p_value = float(fit.pvalues["mentions_library"])
    require(
        np.isclose(coefficient, EXPECTED_LIBRARY_COEFFICIENT, atol=1e-12),
        f"Unexpected external-library coefficient: {coefficient}",
    )
    require(
        np.isclose(p_value, EXPECTED_LIBRARY_P_VALUE, atol=1e-12),
        f"Unexpected external-library p-value: {p_value}",
    )

    names = list(fit.params.index)
    return {
        "display_bin_rule": (
            "bin b is [b - 0.5, b + 0.5); exact half-points go upward"
        ),
        "diagnostic_bins": list(DIAGNOSTIC_BINS),
        "n_prompts": len(frame),
        "n_models_in_prompt_mean": len(model_frames),
        "source_counts": [
            {
                "bin": bin_id,
                "source": source,
                "n": EXPECTED_COUNTS[(bin_id, source)],
            }
            for bin_id in DIAGNOSTIC_BINS
            for source in (ORIGINAL_SOURCE, MINED_SOURCE)
        ],
        "library_tag": {
            "definition": "case-insensitive whole-word match against LIBRARY_TERMS",
            "terms": list(LIBRARY_TERMS),
        },
        "regression": {
            "outcome": "prompt-level mean pass rate",
            "covariance": "HC1",
            "regressors": regressors,
            "external_library_coefficient": coefficient,
            "external_library_robust_standard_error": float(
                fit.bse["mentions_library"]
            ),
            "external_library_p_value": p_value,
            "r_squared": float(fit.rsquared),
            "coefficients": {name: float(fit.params[name]) for name in names},
            "robust_standard_errors": {
                name: float(fit.bse[name]) for name in names
            },
            "p_values": {name: float(fit.pvalues[name]) for name in names},
        },
        "interpretation": (
            "This scoped diagnostic provides no support for an "
            "external-library explanation; it does not identify the cause of "
            "the pooled nonmonotonicity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Repository or retained-data root containing data/ (default: repo root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/mechanism_diagnostic.json",
        help="Destination JSON (default: results/mechanism_diagnostic.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the regenerated result with the existing JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_diagnostic(args.data_root)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.check:
        require(args.output.exists(), f"Missing expected output: {args.output}")
        require(
            args.output.read_text(encoding="utf-8") == rendered,
            f"Regenerated diagnostic differs from {args.output}",
        )
        print(
            "Verified half-open bins 13 and 17: "
            f"N={report['n_prompts']}, "
            "external-library coefficient="
            f"{report['regression']['external_library_coefficient']:.6f}, "
            f"p={report['regression']['external_library_p_value']:.6f}."
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
