"""Generated-output complexity diagnostics for the camera-ready, for any outcome.

Recomputes the output-side diagnostics of the manuscript from the per-model
scored files of one outcome definition (``--scored-dir``; default
``data/stage_d/scored_combined``, the reviewed outcome; the ``CK_SCORED_DIR``
environment variable overrides the default as in the other analysis scripts):

1. reverse-threshold cells among zero-pass generations and the shares at four
   pass-rate cutoffs (``reverse_threshold_zero_pass_cells.csv``);
2. mean pass rate by generated-output cyclomatic complexity
   (``pass_vs_output_cc.csv``);
3. the passing-generation mapping from prompt composite to output CC
   (``kappa_map``) and the output CC at ``--prompt-threshold``;
4. the library/framework-mention diagnostic at display bins 13 and 17.

``scripts/generate_stage_d_paper_figures.py`` reads the two CSVs, which keep
the columns and row order of ``results/``. Everything else goes to
``output_cc_diagnostics.json``.

Rows come from the paper's loaders in ``src/analyze_kink.py``:
``discover_models`` and ``load_scored_model``, joined to the four-judge
composite in ``ensemble_scores_current_aggregated.jsonl``, with
``config.STAGE_C_EXCLUDED_MODELS`` removed. ``load_scored_model`` turns a null
``pass_rate`` into 0.0. ``kappa_cyclomatic`` is the Lizard CC of the cleaned
generated code, and it is null when not computable.

Definitions. On the reviewed outcome, items 1 to 3 reproduce the reviewed
values. Item 4 cannot, for the reason given below.

* Complete case: output CC not null (103,948 generations; 1,052 missing).
  ``n_scored_generations`` keeps its reviewed meaning: generations with
  computable CC.
* Zero-pass: ``pass_rate <= 0``. There are 14,776 complete cases and 977 rows
  missing CC. Cells split the prompt composite at ``<= 8`` versus ``> 8`` and
  output CC at ``<= 10`` versus ``> 10``. The four counts are 4,454, 676,
  4,216, and 5,430. For each cutoff c in 0, 0.25, 0.50, and 0.65, the share
  among complete cases with ``pass_rate <= c`` that have composite > 8 and
  CC <= 10 is 0.28533, 0.26773, 0.24885, and 0.24076.
* Curve: complete cases grouped by ``min(CC, 40)``. Columns are n, mean
  pass_rate, pandas ``sem`` (ddof=1, generation rows, no clustering), and
  mean composite. The file matches ``results/pass_vs_output_cc.csv``.
* Passing-generation mapping (``kappa_map``): a generation-level OLS of raw,
  untop-coded output CC on the prompt composite, fitted on the generations
  with ``pass_rate == 1`` (all tests pass) and computable CC. On the reviewed
  outcome that is 80,723 rows, with intercept -5.73861, slope 2.08412, and
  R^2 0.62331; 13.75 maps to 22.92, reported as 22.9. The row count and
  ``by_bin`` match the reviewed kappa map exactly. The coefficients and R^2
  agree to within 1e-13, a floating-point difference in the solver. Other
  variants do not reproduce it. Using ``pass_rate > 0`` gives n 89,172 and
  R^2 0.605. Top-coding CC at 40 gives R^2 0.675. ``by_bin`` groups the same
  rows by numpy round-half-to-even of the composite, as in the reviewed kappa
  map. Its bins differ from the half-open display bins at exact .5
  composites. ``passing_generation_mapping.by_display_bin`` repeats the table
  with the half-open rule [b-0.5, b+0.5).
* Library diagnostic: prompt-level mean pass rate over the models
  (``build_combined_df``) in half-open display bins 13 and 17 of the
  5,000-prompt benchmark, which hold 323 prompts. The HC1 OLS keeps the
  reviewed regressor order: source pool (1 if construction frame is the later
  candidate frame), prompt characters and words, unit-test count and length,
  reference-solution CC, scorer disagreement (``ens_composite_sd``), the
  mention tag, and the composite. The tag and the text and test features come
  from ``prompt_features`` in ``scripts/regenerate_mechanism_diagnostic.py``,
  applied unchanged to ``stage_d_prompts.jsonl``. The tag is a
  case-insensitive whole-word match to the reviewed term list. Unit tests are
  stored there as a JSON string, so the count is the number of "assert"
  substrings and the length is the raw string length. The ``variants`` block
  swaps in the labeler flag ``names_external_library`` and the parsed test
  list (snippet count and joined length). Only 3 of the 323 prompts carry the
  regex tag, and 5 carry the labeler flag. The coefficient therefore rests on
  a handful of prompts; read it together with ``n_tagged``.
  The reviewed value (n=459, coefficient -0.0501, robust SE 0.0978, p 0.608)
  is NOT reproducible from these inputs. It was estimated by
  ``scripts/regenerate_mechanism_diagnostic.py`` on the 24-bin equal-support
  frame, ``data/stage_d_24bin_equal``. That frame holds 267 benchmark prompts
  and 192 OpenCodeInstruct mined-pool prompts. Its source pool means mined
  versus benchmark, and its 21-model outcomes come from
  ``scored_combined_final``. Its composites also differ from the current
  ensemble: it puts 228 and 39 benchmark prompts in bins 13 and 17, where
  this frame has 278 and 45. The public release excludes that frame, and the
  independent audit, which covers the 105,000 benchmark generations, does not
  include its mined-pool generations.

Usage (from the repository root):
    python scripts/camera_ready/output_cc_diagnostics.py \\
        --out-dir results/camera_ready/reviewed_outcome
    python scripts/camera_ready/output_cc_diagnostics.py \\
        --scored-dir data/stage_d/scored_independent_audit \\
        --out-dir results/camera_ready/independent_audit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analyze_kink import (  # noqa: E402
    build_combined_df,
    discover_models,
    load_rubric_scores,
    load_scored_model,
)
from config import STAGE_C_EXCLUDED_MODELS  # noqa: E402
from display_bins import half_open_integer_bin  # noqa: E402
from regenerate_mechanism_diagnostic import LIBRARY_TERMS, prompt_features  # noqa: E402

PROMPT_COMPOSITE_MAX = 8.0
OUTPUT_CC_MAX = 10.0
PASS_RATE_CUTOFFS = {
    "pass_rate_eq_0": 0.0,
    "pass_rate_lte_0_25": 0.25,
    "pass_rate_lte_0_50": 0.50,
    "pass_rate_lte_0_65": 0.65,
}
PROMPT_GROUPS = ("Prompt composite <= 8", "Prompt composite > 8")
OUTPUT_GROUPS = ("Output CC <= 10", "Output CC > 10")
CC_TOP_CODE = 40
LIBRARY_BINS = (13, 17)
LATER_FRAME = "stage_d_candidate"
TEXT_TEST_FEATURES = ("prompt_chars", "prompt_words", "n_unit_tests", "test_chars")
METADATA_COLUMNS = [
    "prompt_id", "construction_frame", "ens_composite", "ens_composite_sd",
    "names_external_library", "unit_tests", "reference_cc",
]


def parse_args() -> argparse.Namespace:
    data = ROOT / "data"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored-dir", type=Path,
                        default=Path(os.environ.get("CK_SCORED_DIR", data / "stage_d" / "scored_combined")),
                        help="Per-model scored JSONL directory (one outcome definition).")
    parser.add_argument("--rubric", type=Path,
                        default=data / "stage_d" / "ensemble_scores_current_aggregated.jsonl")
    parser.add_argument("--prompts", type=Path, default=data / "stage_d" / "stage_d_prompts.jsonl")
    parser.add_argument("--prompt-metadata", type=Path, default=data / "public_release" / "prompts",
                        help="Public-release prompts config (Parquet file or directory).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-threshold", type=float, default=13.75,
                        help="Prompt-composite threshold to translate to output CC.")
    return parser.parse_args()


def display_path(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def clean(value):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_generations(scored_dir: Path, rubric_path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rubric = load_rubric_scores(str(rubric_path))
    frames = {}
    for name, path in discover_models(str(scored_dir)):
        if name in STAGE_C_EXCLUDED_MODELS:
            continue
        frames[name] = load_scored_model(path, rubric)
    if not frames:
        raise SystemExit(f"No scored model files found in {scored_dir}")
    generations = pd.concat([frame.assign(model=name) for name, frame in frames.items()], ignore_index=True)
    duplicates = int(generations.duplicated(["model", "prompt_id"]).sum())
    if duplicates:
        raise SystemExit(f"{duplicates} duplicate (model, prompt) rows in {scored_dir}")
    generations["output_cc"] = pd.to_numeric(generations["kappa_cyclomatic"], errors="coerce")
    return generations, frames


def reverse_threshold(generations: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    has_cc = generations["output_cc"].notna()
    zero = generations["pass_rate"] <= 0.0
    complete = generations[has_cc]
    zero_complete = generations[has_cc & zero]
    low_prompt = zero_complete["composite"] <= PROMPT_COMPOSITE_MAX
    low_output = zero_complete["output_cc"] <= OUTPUT_CC_MAX
    rows = []
    for prompt_label, prompt_mask in zip(PROMPT_GROUPS, (low_prompt, ~low_prompt)):
        for output_label, output_mask in zip(OUTPUT_GROUPS, (low_output, ~low_output)):
            rows.append({"prompt_group": prompt_label, "output_group": output_label,
                         "n": int((prompt_mask & output_mask).sum())})
    shares, share_counts = {}, {}
    for key, cutoff in PASS_RATE_CUTOFFS.items():
        subset = complete[complete["pass_rate"] <= cutoff]
        flagged = (subset["composite"] > PROMPT_COMPOSITE_MAX) & (subset["output_cc"] <= OUTPUT_CC_MAX)
        shares[key] = float(flagged.mean()) if len(subset) else None
        share_counts[key] = {"n_complete_cases": int(len(subset)), "n_flagged": int(flagged.sum())}
    summary = {
        "n_scored_generations": int(has_cc.sum()),
        "n_missing_output_cyclomatic_complexity": int((~has_cc).sum()),
        "n_zero_pass_complete_cases": int(len(zero_complete)),
        "n_zero_pass_missing_output_cyclomatic_complexity": int((zero & ~has_cc).sum()),
        "prompt_composite_gt": PROMPT_COMPOSITE_MAX,
        "output_cyclomatic_complexity_lte": OUTPUT_CC_MAX,
        "shares": shares,
        "share_counts": share_counts,
        "zero_pass_cells": rows,
    }
    return summary, pd.DataFrame(rows, columns=["prompt_group", "output_group", "n"])


def pass_vs_output_cc(generations: pd.DataFrame) -> pd.DataFrame:
    complete = generations[generations["output_cc"].notna()]
    binned = complete["output_cc"].clip(upper=CC_TOP_CODE).astype(int)
    return (
        complete.assign(cc_binned=binned)
        .groupby("cc_binned")
        .agg(
            n=("pass_rate", "size"),
            mean_pass=("pass_rate", "mean"),
            sem=("pass_rate", "sem"),
            mean_rubric=("composite", "mean"),
        )
        .reset_index()
    )


def cc_by_bin(passing: pd.DataFrame, bins: np.ndarray) -> list[dict]:
    rows = []
    for bin_id, cc in passing["output_cc"].groupby(bins):
        rows.append({
            "bin": int(bin_id),
            "n": int(cc.size),
            "median_cc": float(cc.median()),
            "mean_cc": float(cc.mean()),
            "q25_cc": float(cc.quantile(0.25)),
            "q75_cc": float(cc.quantile(0.75)),
        })
    return rows


def passing_generation_mapping(generations: pd.DataFrame, threshold: float) -> tuple[dict, dict]:
    passing = generations[(generations["pass_rate"] >= 1.0) & generations["output_cc"].notna()]
    composite = passing["composite"].to_numpy(dtype=float)
    fit = sm.OLS(passing["output_cc"].to_numpy(dtype=float), sm.add_constant(composite)).fit()
    intercept, slope = (float(value) for value in fit.params)
    kappa_map = {
        "intercept": intercept,
        "slope": slope,
        "r2": float(fit.rsquared),
        "n_passing": int(len(passing)),
        "by_bin": cc_by_bin(passing, np.round(composite).astype(int)),
    }
    estimate = intercept + slope * threshold
    details = {
        "definition": (
            "Generation-level OLS of raw output CC on the prompt composite over "
            "generations with pass_rate == 1 and computable CC. kappa_map.by_bin uses "
            "numpy round-half-to-even bins of the composite (reviewed convention)."
        ),
        "prompt_threshold": threshold,
        "estimated_output_cyclomatic_complexity": estimate,
        "n_prompts": int(passing["prompt_id"].nunique()),
        "by_display_bin_rule": "half-open: bin b holds composites in [b-0.5, b+0.5)",
        "by_display_bin": cc_by_bin(passing, half_open_integer_bin(composite)),
    }
    return kappa_map, details


def load_prompt_frame(prompts_path: Path, metadata_path: Path) -> pd.DataFrame:
    records = {}
    with prompts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["prompt_id"] in records:
                raise SystemExit(f"Duplicate prompt_id {record['prompt_id']} in {prompts_path}")
            records[record["prompt_id"]] = record
    rows = []
    for prompt_id, record in records.items():
        features = prompt_features(record)
        rows.append({
            "prompt_id": prompt_id,
            "selection_source": record.get("selection_source"),
            **{name: features[name] for name in TEXT_TEST_FEATURES},
            "mentions_library": features["mentions_library"],
            "prompt_reference_cc": record.get("reference_cc"),
        })
    prompts = pd.DataFrame(rows)
    metadata = pd.read_parquet(metadata_path, columns=METADATA_COLUMNS)
    metadata["n_unit_tests_parsed"] = metadata["unit_tests"].map(len)
    metadata["test_chars_parsed"] = metadata["unit_tests"].map(lambda tests: len("\n".join(tests)))
    metadata["names_external_library"] = metadata["names_external_library"].map(
        lambda flag: None if flag is None else int(bool(flag)))
    frame = prompts.merge(metadata.drop(columns=["unit_tests"]), on="prompt_id", how="outer", indicator=True)
    unmatched = int((frame["_merge"] != "both").sum())
    if unmatched:
        raise SystemExit(f"{unmatched} prompt ids are not in both {prompts_path} and {metadata_path}")
    frame = frame.drop(columns="_merge")
    frame["later_candidate_frame"] = (frame["selection_source"] == LATER_FRAME).astype(int)
    consistency = {
        "frame_label_mismatches": int(
            (frame["later_candidate_frame"] != (frame["construction_frame"] == "later_candidate")).sum()),
        "reference_cc_mismatches": int((frame["prompt_reference_cc"] != frame["reference_cc"]).sum()),
    }
    if any(consistency.values()):
        raise SystemExit(f"Prompt file and public-release metadata disagree: {consistency}")
    return frame


def hc1_fit(frame: pd.DataFrame, regressors: list[str], tag: str, full: bool = False) -> dict:
    data = frame.dropna(subset=["pass_rate", *regressors])
    n_tagged = int(data[tag].sum())
    result = {"n": int(len(data)), "tag": tag, "n_tagged": n_tagged, "regressors": regressors}
    if n_tagged in (0, len(data)):
        result.update(coefficient=None, robust_standard_error=None, p_value=None,
                      note="tag has no variation in this sample")
        return result
    design = sm.add_constant(data[regressors].astype(float), has_constant="add")
    fit = sm.OLS(data["pass_rate"].astype(float), design).fit(cov_type="HC1")
    result.update(
        coefficient=float(fit.params[tag]),
        robust_standard_error=float(fit.bse[tag]),
        p_value=float(fit.pvalues[tag]),
        r_squared=float(fit.rsquared),
    )
    if full:
        result["coefficients"] = {name: float(value) for name, value in fit.params.items()}
        result["robust_standard_errors"] = {name: float(value) for name, value in fit.bse.items()}
        result["p_values"] = {name: float(value) for name, value in fit.pvalues.items()}
    return result


def library_diagnostic(combined: pd.DataFrame, prompts: pd.DataFrame) -> dict:
    frame = combined[["prompt_id", "pass_rate", "composite"]].merge(prompts, on="prompt_id", how="left")
    if frame["construction_frame"].isna().any():
        raise SystemExit("Analyzed prompts lack public-release metadata")
    composite_gap = float((frame["composite"] - frame["ens_composite"]).abs().max())
    if composite_gap > 1e-9:
        raise SystemExit(f"Rubric composite differs from public-release ens_composite by {composite_gap}")
    frame["bin"] = half_open_integer_bin(frame["composite"])
    frame = frame[frame["bin"].isin(LIBRARY_BINS)].rename(columns={"ens_composite_sd": "composite_std"})

    def regressors(tests: str = "raw", tag: str = "mentions_library") -> list[str]:
        suffix = "" if tests == "raw" else "_parsed"
        return ["later_candidate_frame", "prompt_chars", "prompt_words", f"n_unit_tests{suffix}",
                f"test_chars{suffix}", "reference_cc", "composite_std", tag, "composite"]

    primary = hc1_fit(frame, regressors(), "mentions_library", full=True)
    variants = {
        "labeler_flag_tag": hc1_fit(frame, regressors(tag="names_external_library"), "names_external_library"),
        "regex_tag_parsed_tests": hc1_fit(frame, regressors(tests="parsed"), "mentions_library"),
        "labeler_flag_tag_parsed_tests": hc1_fit(
            frame, regressors(tests="parsed", tag="names_external_library"), "names_external_library"),
    }
    cells = frame.groupby(["bin", "construction_frame"]).size()
    return {
        "sample": "5,000-prompt benchmark, half-open display bins 13 and 17",
        "outcome": "prompt-level mean pass_rate over the analyzed models",
        "covariance": "HC1",
        "tag_definition": "case-insensitive whole-word match of prompt text against LIBRARY_TERMS "
                          "(scripts/regenerate_mechanism_diagnostic.py)",
        "tag_terms": list(LIBRARY_TERMS),
        "test_feature_definition": "prompt_features() on stage_d_prompts.jsonl, whose unit_tests are a JSON "
                                   "string: count = occurrences of 'assert', length = raw string length",
        "source_pool_definition": "1 if construction frame is later_candidate (selection_source "
                                  "stage_d_candidate), else 0",
        "cell_counts": [{"bin": int(b), "construction_frame": f, "n": int(n)} for (b, f), n in cells.items()],
        "primary": primary,
        "variants": variants,
        "reviewed_value_note": (
            "The reviewed n=459 estimate (-0.0501, SE 0.0978, p 0.608) used the 24-bin equal-support frame "
            "(benchmark plus OpenCodeInstruct mined-pool prompts; source pool = mined vs benchmark), which is "
            "not part of these inputs; see scripts/regenerate_mechanism_diagnostic.py."
        ),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generations, model_frames = load_generations(args.scored_dir, args.rubric)
    print(f"Loaded {len(generations):,} generations from {len(model_frames)} models in {args.scored_dir}")

    reverse_summary, cells = reverse_threshold(generations)
    curve = pass_vs_output_cc(generations)
    kappa_map, mapping = passing_generation_mapping(generations, args.prompt_threshold)
    reverse_summary["passing_generation_mapping_at_prompt_threshold"] = {
        "prompt_threshold": args.prompt_threshold,
        "estimated_output_cyclomatic_complexity": round(mapping["estimated_output_cyclomatic_complexity"], 1),
    }
    combined = build_combined_df(model_frames)
    library = library_diagnostic(combined, load_prompt_frame(args.prompts, args.prompt_metadata))

    cells.to_csv(args.out_dir / "reverse_threshold_zero_pass_cells.csv", index=False)
    curve.to_csv(args.out_dir / "pass_vs_output_cc.csv", index=False)
    report = {
        "schema_version": 1,
        "script": "scripts/camera_ready/output_cc_diagnostics.py",
        "inputs": {
            "scored_dir": display_path(args.scored_dir),
            "rubric": display_path(args.rubric),
            "prompts": display_path(args.prompts),
            "prompt_metadata": display_path(args.prompt_metadata),
        },
        "panel": {
            "n_models": len(model_frames),
            "rows_per_model": {name: int(len(frame)) for name, frame in model_frames.items()},
            "n_generations": int(len(generations)),
            "n_prompts": int(generations["prompt_id"].nunique()),
            "mean_pass_rate": float(generations["pass_rate"].mean()),
        },
        "reverse_threshold": reverse_summary,
        "pass_vs_output_cc": {
            "csv": "pass_vs_output_cc.csv",
            "top_code": CC_TOP_CODE,
            "n_generations": int(curve["n"].sum()),
            "sem": "pandas sem of generation rows (ddof=1); no adjustment for repeated prompts or models",
        },
        "kappa_map": kappa_map,
        "passing_generation_mapping": mapping,
        "library_diagnostic": library,
    }
    (args.out_dir / "output_cc_diagnostics.json").write_text(
        json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")

    shares = reverse_summary["shares"]
    primary = library["primary"]
    print(f"Reverse threshold: {reverse_summary['n_scored_generations']:,} with CC, "
          f"{reverse_summary['n_missing_output_cyclomatic_complexity']:,} missing; "
          f"zero-pass {reverse_summary['n_zero_pass_complete_cases']:,} complete, "
          f"{reverse_summary['n_zero_pass_missing_output_cyclomatic_complexity']:,} missing CC")
    print("  cells: " + ", ".join(f"{row['prompt_group']} & {row['output_group']}: {row['n']:,}"
                                  for row in reverse_summary["zero_pass_cells"]))
    print("  shares: " + ", ".join(f"{key}={'n/a' if value is None else format(value, '.5f')}"
                                   for key, value in shares.items()))
    print(f"Passing mapping: n={kappa_map['n_passing']:,}, R2={kappa_map['r2']:.4f}, "
          f"{args.prompt_threshold:g} -> output CC {mapping['estimated_output_cyclomatic_complexity']:.2f}")
    if primary["coefficient"] is not None:
        print(f"Library diagnostic: n={primary['n']}, tagged={primary['n_tagged']}, "
              f"coef={primary['coefficient']:.4f}, robust SE={primary['robust_standard_error']:.4f}, "
              f"p={primary['p_value']:.3f}")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
