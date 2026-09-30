"""Assemble one outcome's camera-ready results into the files the figures read.

Run by scripts/camera_ready/run_reanalysis.sh after the individual analyses.
In --results-dir it writes:

* analysis_summary.json            (the headline combined fit, from combined/)
* per_model_bootstrap_summary.{json,csv}  (one row per model, from per_model/)
* robustness_summary.json          (the reviewed summary with every
                                    outcome-dependent section recomputed)

Sections that do not depend on the outcome (human calibration, paraphrase and
language checks) are carried over unchanged. Every outcome-dependent section is
rebuilt from this run's outputs; the few sections that cannot be recomputed for
this outcome are carried over with an explicit ``outcome`` label saying which
outcome they use (the repeated-sampling check uses raw harness outcomes because
those generations were not included in the independent audit; the extension
runs saved no code, so the extension's matched-panel threshold is recomputed
here on raw harness outcomes for both sources; the library-mention diagnostic
exists only for the reviewed outcome). The script fails if two sections
disagree on the headline breakpoint.
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
        rows.append({"model_id": model_id, "summary_path": str(summary_path.resolve().relative_to(ROOT)),
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
    reviewed_tt = robustness.get("task_type_controls", {})
    robustness["task_type_controls"] = {
        # Label reliability is a property of the prompts, not of the outcome.
        **{k: reviewed_tt[k] for k in ("n_prompts", "n_categories", "reliability_subsample_n",
                                       "krippendorff_alpha") if k in reviewed_tt},
        "uncontrolled": {"threshold": c["kink_threshold"], "sup_wald": c["kink_sup_wald"],
                         "regime_gap_percentage_points": 100 * (c["mean_pass_high"] - c["mean_pass_low"])},
        **breakpoints["task_type_controls"],
        "large_task_types_tested_for_overidentification": iv["within_task_type"]["categories_tested"],
        "large_task_types_rejecting": iv["within_task_type"]["categories_rejecting_at_0_05"],
    }
    robustness["task_type_controls"]["fixed_effects"]["specification"] = (
        "by_side: task-type effects may differ on each side of the break; see control_specs")
    robustness["source_frame_sensitivity"] = frame_sens
    robustness["model_specific_source_frame"] = model_frames
    robustness["overidentification"] = iv
    robustness["output_cc_regression_diagnostics"] = iv["output_cc_regressions"]
    robustness["reverse_threshold"] = output_cc
    for name, fname in (("control_specs", "control_specs.json"),
                        ("no_response_sensitivity", "no_response_sensitivity.json"),
                        ("auditor_agreement", "auditor_agreement.json"),
                        ("frame_decomposition_at_headline", "frame_decomposition.json")):
        path = out / fname
        if path.exists():
            robustness[name] = json.loads(path.read_text(encoding="utf-8"))
    robustness["pass_at_k"] = {**robustness.get("pass_at_k", {}), "outcome": (
        "raw unit-test harness: these 7,180 generations (code saved) were not included in the "
        "independent audit")}
    robustness["mechanism_diagnostic"] = {**robustness.get("mechanism_diagnostic", {}), "outcome": (
        "reviewed version only: the diagnostic sample includes prompts outside the benchmark and "
        "cannot be re-estimated under this outcome")}
    robustness["submitted_baseline"] = {**robustness.get("submitted_baseline", {}), "outcome": (
        "reviewed version (as submitted), kept for reference")}
    ext = robustness.setdefault("high_complexity_extension", {})
    matched = ext.setdefault("matched_five_model", {})
    matched.update(extension_threshold(args.data_root))
    replication = pd.read_csv(out / "tail_extension_replication.csv")
    for b in (15, 16):
        rows_b = replication[replication["bin"] == b].set_index("source")
        orig, extn = rows_b.loc["Original benchmark"], rows_b.loc["Audit-clean extension"]
        matched[f"bin_{b}"] = {"extension_pass_rate": float(extn["mean_pass"]),
                               "original_pass_rate": float(orig["mean_pass"]),
                               "difference": float(extn["mean_pass"] - orig["mean_pass"]),
                               "welch_p": float(orig["welch_p"])}
    # Consistency: every section that reports the headline breakpoint must agree.
    headline = {c["kink_threshold"], breakpoints["pool_mean"]["threshold"],
                robustness["task_type_controls"]["uncontrolled"]["threshold"],
                frame_sens["unadjusted_threshold"]["threshold"]}
    if len(headline) != 1:
        raise SystemExit(f"sections disagree on the headline breakpoint: {sorted(headline)}")
    (out / "robustness_summary.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    print(f"assembled {out}: combined gamma {c['kink_threshold']}, "
          f"CI [{c['kink_ci_lower']}, {c['kink_ci_upper']}], extension gamma "
          f"{ext['matched_five_model']['combined_threshold']}")


if __name__ == "__main__":
    main()
