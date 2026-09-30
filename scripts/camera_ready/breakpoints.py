"""Breakpoint analyses for the camera-ready, for any outcome definition.

Reads the per-model scored files from CK_SCORED_DIR (default
data/stage_d/scored_combined) with the same loaders as src/analyze_kink.py and
recomputes, with the manuscript's sup-Wald machinery
(scripts/analyze_source_frame_sensitivity.py):

1. mean-pooled, median-pooled, and 21 leave-one-model-out breakpoints;
2. every model-specific unadjusted breakpoint and its raw regime direction;
3. the task-type fixed-effects search and the predictive-fit decomposition.

It writes a pooling report in the schema of the reviewed
results/rebuttal/pooling_submitted/pooling_and_kappa_report.json (without the
kappa map, which scripts/camera_ready/output_cc_diagnostics.py produces), so
scripts/analyze_model_specific_source_frame.py can select the downward fits.

Usage:
    CK_SCORED_DIR=data/stage_d/scored_independent_audit \
        python scripts/camera_ready/breakpoints.py --out results/camera_ready/<outcome>/breakpoints.json
"""
from __future__ import annotations

import argparse
import json
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

from analyze_kink import discover_models, load_rubric_scores, load_scored_model  # noqa: E402
from analyze_source_frame_sensitivity import (  # noqa: E402
    piecewise_matrix,
    prepare_threshold_design,
    threshold_curve,
    wild_bootstrap_threshold,
)
from config import STAGE_C_EXCLUDED_MODELS  # noqa: E402


def load_panel(data_root: Path, scored_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Prompt x model pass-rate matrix and the prompt composite."""
    rubric = load_rubric_scores(data_root / "stage_d" / "ensemble_scores_current_aggregated.jsonl")
    columns, composite = {}, {}
    for model_name, model_path in discover_models(str(scored_dir)):
        if model_name in STAGE_C_EXCLUDED_MODELS:
            continue
        df = load_scored_model(model_path, rubric)
        columns[model_name] = df.set_index("prompt_id")["pass_rate"]
        composite.update(df.set_index("prompt_id")["composite"].to_dict())
    panel = pd.DataFrame(columns).sort_index()
    if panel.shape != (5000, 21) or panel.isna().any().any():
        raise RuntimeError(f"expected a complete 5,000 x 21 panel, got {panel.shape} "
                           f"with {int(panel.isna().sum().sum())} missing cells")
    return panel, pd.Series(composite).reindex(panel.index).astype(float)


def point_fit(y: np.ndarray, frame: pd.DataFrame, design) -> dict:
    """Manuscript sup-Wald breakpoint and raw regime means, no bootstrap."""
    gamma, sup_wald = max(threshold_curve(y, design), key=lambda item: item[1])
    low = frame["composite"].to_numpy() <= gamma
    return {"threshold": float(gamma), "sup_wald": float(sup_wald), "n": int(len(y)),
            "mean_pass_low": float(y[low].mean()), "mean_pass_high": float(y[~low].mean()),
            "n_low": int(low.sum()), "n_high": int((~low).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=300, help="Wild-bootstrap draws for the task-type search.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scored_dir = Path(os.environ.get("CK_SCORED_DIR", args.data_root / "stage_d" / "scored_combined"))
    panel, composite = load_panel(args.data_root, scored_dir)
    base = pd.DataFrame({"composite": composite.to_numpy()}, index=panel.index)
    design = prepare_threshold_design(base.reset_index(drop=True), [])

    report: dict = {"scored_dir": str(scored_dir), "n_models": int(panel.shape[1])}
    report["pool_mean"] = point_fit(panel.mean(axis=1).to_numpy(), base, design)
    report["pool_median"] = point_fit(panel.median(axis=1).to_numpy(), base, design)
    report["leave_one_model_out"] = [
        {**point_fit(panel.drop(columns=m).mean(axis=1).to_numpy(), base, design), "excluded_model": m}
        for m in panel.columns
    ]
    lomo = [r["threshold"] for r in report["leave_one_model_out"]]
    report["lomo_threshold_range"] = [min(lomo), max(lomo)]
    per_model = []
    for m in panel.columns:
        fit = point_fit(panel[m].to_numpy(), base, design)
        fit.update(model=m, direction="up" if fit["mean_pass_high"] > fit["mean_pass_low"] else "down")
        per_model.append(fit)
    report["per_model"] = per_model
    report["direction_tally"] = {d: sum(r["direction"] == d for r in per_model) for d in ("up", "down")}
    thresholds = sorted(r["threshold"] for r in per_model)
    report["per_model_threshold_range"] = [thresholds[0], thresholds[-1]]
    report["per_model_threshold_median"] = float(np.median(thresholds))

    # Task-type fixed effects on the mean-pooled outcome.
    meta = pd.read_parquet(args.data_root / "public_release" / "prompts", columns=["prompt_id", "task_type"])
    task = meta.set_index("prompt_id")["task_type"].reindex(panel.index)
    if task.isna().any():
        raise RuntimeError("task-type label missing for some prompts")
    dummies = pd.get_dummies(task, prefix="tt", drop_first=True, dtype=float)
    frame = pd.concat([base, dummies], axis=1).reset_index(drop=True)
    frame["pass_rate"] = panel.mean(axis=1).to_numpy()
    controlled = wild_bootstrap_threshold(frame, list(dummies.columns), args.n_boot, args.seed)
    gamma_tt = controlled["threshold"]
    y = frame["pass_rate"]
    r2 = lambda X: float(sm.OLS(y, sm.add_constant(X, has_constant="add")).fit().rsquared)  # noqa: E731
    piecewise = piecewise_matrix(frame, gamma_tt, ["composite"])
    report["task_type_controls"] = {
        "fixed_effects": controlled,
        "n_categories": int(task.nunique()),
        "piecewise_predictive_fit": {
            "piecewise_threshold": gamma_tt,
            "piecewise_composite_only_r2": r2(piecewise),
            "task_type_only_r2": r2(dummies.reset_index(drop=True)),
            "task_type_plus_piecewise_composite_r2": r2(pd.concat([piecewise, dummies.reset_index(drop=True)], axis=1)),
        },
    }
    fit = report["task_type_controls"]["piecewise_predictive_fit"]
    fit["incremental_r2_over_task_type"] = fit["task_type_plus_piecewise_composite_r2"] - fit["task_type_only_r2"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pm, med = report["pool_mean"], report["pool_median"]
    print(f"mean pool: gamma={pm['threshold']} supW={pm['sup_wald']:.2f} low={pm['mean_pass_low']:.4f} high={pm['mean_pass_high']:.4f}")
    print(f"median pool: gamma={med['threshold']} low={med['mean_pass_low']:.4f} high={med['mean_pass_high']:.4f}")
    print(f"LOMO range: {report['lomo_threshold_range']}; directions {report['direction_tally']}; "
          f"model thresholds {report['per_model_threshold_range']} median {report['per_model_threshold_median']}")
    print(f"task-type FE: gamma={gamma_tt} supW={controlled['sup_wald']:.2f} "
          f"low={controlled['mean_pass_low']:.4f} high={controlled['mean_pass_high']:.4f} "
          f"boot exc={controlled['wild_bootstrap']['exceedances']}/{args.n_boot}")
    print("R2:", {k: round(v, 5) for k, v in fit.items() if k != "piecewise_threshold"})
    print("downward:", [r["model"] for r in per_model if r["direction"] == "down"])


if __name__ == "__main__":
    main()
