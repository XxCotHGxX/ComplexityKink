"""Assemble one outcome's camera-ready results into the files the figures read.

Run by scripts/camera_ready/run_reanalysis.sh after the individual analyses.
In --results-dir it writes:

* analysis_summary.json            (the headline combined fit, from combined/)
* per_model_bootstrap_summary.{json,csv}  (one row per model, from per_model/)
* robustness_summary.json          (the reviewed summary with every
                                    outcome-dependent section recomputed)

Sections that do not depend on the outcome (human calibration, paraphrase and
language checks) are carried over unchanged. The extension and repeated-sampling
checks use raw harness outcomes, because those runs did not save generated
code; the extension's matched-panel threshold is recomputed here on raw harness
outcomes for both sources.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def extension_threshold(data_root: Path) -> dict:
    """Matched five-model threshold with raw harness outcomes on both sources."""
    os.environ["CK_SCORED_DIR"] = str(data_root / "data" / "stage_d" / "scored_harness")
    from analyze_source_frame_sensitivity import prepare_threshold_design, threshold_curve
    from regenerate_tail_extension_tables import load_extension_metadata, load_matched_frame

    frame = load_matched_frame(data_root, load_extension_metadata(data_root)).reset_index(drop=True)
    design = prepare_threshold_design(frame, [])
    y = frame["pass_rate"].to_numpy(dtype=float)
    gamma, sup_wald = max(threshold_curve(y, design), key=lambda item: item[1])
    low = frame["composite"].to_numpy() <= gamma
    return {"outcome": "raw unit-test harness (extension generations were not saved, so not audited)",
            "combined_threshold": float(gamma), "sup_wald": float(sup_wald),
            "combined_pass_at_or_below": float(y[low].mean()), "combined_pass_above": float(y[~low].mean()),
            "n_prompts": int(len(frame))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT, help="Repository root holding data/.")
    ap.add_argument("--reviewed-summary", type=Path, default=ROOT / "results" / "robustness_summary.json")
    args = ap.parse_args()
    out = args.results_dir

    combined = json.loads((out / "combined" / "analysis_summary.json").read_text(encoding="utf-8"))
    (out / "analysis_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    rows = []
    for summary_path in sorted((out / "per_model").glob("*/analysis_summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        (model_id, fit), = [(k, v) for k, v in data.items() if not k.startswith("_")]
        rows.append({"model_id": model_id, "summary_path": str(summary_path.relative_to(ROOT)),
                     **{k: v for k, v in fit.items() if k != "model"}})
    if len(rows) != 21:
        raise SystemExit(f"expected 21 per-model summaries, found {len(rows)}")
    (out / "per_model_bootstrap_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "per_model_bootstrap_summary.csv", index=False)

    robustness = json.loads(args.reviewed_summary.read_text(encoding="utf-8"))
    breakpoints = json.loads((out / "breakpoints.json").read_text(encoding="utf-8"))
    frame_sens = json.loads((out / "source_frame_sensitivity.json").read_text(encoding="utf-8"))
    iv = json.loads((out / "iv_diagnostics.json").read_text(encoding="utf-8"))
    output_cc = json.loads((out / "output_cc_diagnostics.json").read_text(encoding="utf-8"))
    model_frames = json.loads((out / "model_specific_source_frame.json").read_text(encoding="utf-8"))

    c = combined["_combined"]
    robustness["outcome"] = {"label": out.name, "scored_dir": breakpoints["scored_dir"]}
    robustness["headline_combined_fit"] = {k: c.get(k) for k in (
        "kink_threshold", "kink_sup_wald", "kink_pval", "kink_ci_lower", "kink_ci_upper",
        "mean_pass_low", "mean_pass_high", "placebo_pval", "linear_bic", "cubic_bic",
        "piecewise_bic", "iv_fstat", "iv_j_pval")}
    robustness["pooling_robustness"] = {
        "leave_one_model_out_fits": len(breakpoints["leave_one_model_out"]),
        "leave_one_model_out_threshold_range": breakpoints["lomo_threshold_range"],
        "median_pool_threshold": breakpoints["pool_median"]["threshold"],
        "median_pool_pass_at_or_below": breakpoints["pool_median"]["mean_pass_low"],
        "median_pool_pass_above": breakpoints["pool_median"]["mean_pass_high"],
        "models_with_upward_regime_change": breakpoints["direction_tally"]["up"],
        "models_with_downward_regime_change": breakpoints["direction_tally"]["down"],
        "downward_models": [r["model"] for r in breakpoints["per_model"] if r["direction"] == "down"],
        "per_model_threshold_range": breakpoints["per_model_threshold_range"],
        "per_model_threshold_median": breakpoints["per_model_threshold_median"],
    }
    robustness["task_type_controls"] = {**robustness.get("task_type_controls", {}),
                                        **breakpoints["task_type_controls"]}
    robustness["source_frame_sensitivity"] = frame_sens
    robustness["model_specific_source_frame"] = model_frames
    robustness["overidentification"] = iv
    robustness["reverse_threshold"] = output_cc
    ext = robustness.setdefault("high_complexity_extension", {})
    ext.setdefault("matched_five_model", {}).update(extension_threshold(args.data_root))
    (out / "robustness_summary.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    print(f"assembled {out}: combined gamma {c['kink_threshold']}, "
          f"CI [{c['kink_ci_lower']}, {c['kink_ci_upper']}], extension gamma "
          f"{ext['matched_five_model']['combined_threshold']}")


if __name__ == "__main__":
    main()
