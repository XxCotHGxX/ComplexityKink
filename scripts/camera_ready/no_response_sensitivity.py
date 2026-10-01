"""Sensitivity of the breakpoint results to generations with no model response.

Some generations came back from the model API with no response at all
(time-outs, cancelled operations, or empty responses). The primary analysis
scores them like any generation without code: pass rate 0 under the harness and
"incorrect" under the audit's empty-code rule. This script treats them as
missing instead. The pooled outcome averages over the models that responded,
and model-specific fits drop those rows. A generation whose response contained
no code (for example prose only) is a model failure and is kept.

    CK_SCORED_DIR=data/stage_d/scored_independent_audit python \
        scripts/camera_ready/no_response_sensitivity.py --out results/camera_ready/independent_audit/no_response_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (HERE, ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analyze_kink import discover_models, load_rubric_scores  # noqa: E402
from analyze_source_frame_sensitivity import prepare_threshold_design  # noqa: E402
from breakpoints import point_fit  # noqa: E402
from config import STAGE_C_EXCLUDED_MODELS  # noqa: E402


def load(scored_dir: Path, rubric: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prompt x model pass rates (missing pass rate coded 0, as the loaders do) and the no-response mask."""
    rates, masks = {}, {}
    for name, path in discover_models(str(scored_dir)):
        if name in STAGE_C_EXCLUDED_MODELS:
            continue
        r_col, m_col = {}, {}
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") not in rubric:
                continue
            r_col[rec["id"]] = float(rec.get("pass_rate") or 0.0)
            m_col[rec["id"]] = not (rec.get("output") or "").strip()
        rates[name], masks[name] = pd.Series(r_col), pd.Series(m_col)
    panel, mask = pd.DataFrame(rates).sort_index(), pd.DataFrame(masks).sort_index()
    if panel.shape != (5000, 21) or panel.isna().any().any():
        raise RuntimeError(f"expected a complete 5,000 x 21 panel, got {panel.shape}")
    return panel, mask.reindex_like(panel).fillna(False).astype(bool)


def fit_on(values: pd.Series, composite: pd.Series) -> dict:
    keep = values.notna()
    frame = pd.DataFrame({"composite": composite[keep].to_numpy()})
    design = prepare_threshold_design(frame, [])
    fit = point_fit(values[keep].to_numpy(dtype=float), frame, design)
    fit["direction"] = "up" if fit["mean_pass_high"] > fit["mean_pass_low"] else "down"
    return fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=["azure_kimi-k2.5", "google_gemini-3.1-pro-preview"],
                    help="Models whose later-frame fits are also reported.")
    args = ap.parse_args()

    scored_dir = Path(os.environ.get("CK_SCORED_DIR", args.data_root / "stage_d" / "scored_combined"))
    rubric = load_rubric_scores(str(args.data_root / "stage_d" / "ensemble_scores_current_aggregated.jsonl"))
    panel, mask = load(scored_dir, rubric)
    composite = pd.Series({pid: rubric[pid]["composite"] for pid in panel.index}).astype(float)
    frames = {}
    for line in open(args.data_root / "stage_d" / "stage_d_prompts.jsonl", encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            frames[rec["prompt_id"]] = rec.get("selection_source")
    later = pd.Series(frames).reindex(panel.index) == "stage_d_candidate"

    cells = mask.stack()
    flagged = cells[cells].index
    counts = {"total": int(mask.to_numpy().sum()),
              "by_model": {m: int(n) for m, n in mask.sum().items() if n},
              "later_frame": int(sum(later[p] for p, _ in flagged)),
              "composite_above_14": int(sum(composite[p] > 14 for p, _ in flagged))}

    masked = panel.where(~mask)
    report = {"scored_dir": str(scored_dir), "no_response_generations": counts, "variants": {}}
    for label, data in (("all_rows", panel), ("no_response_as_missing", masked)):
        pooled = fit_on(data.mean(axis=1), composite)
        per_model = {m: fit_on(data[m], composite) for m in panel.columns}
        later_fits = {m: fit_on(data[m].where(later), composite) for m in args.models}
        report["variants"][label] = {
            "mean_pooled": pooled,
            "per_model": per_model,
            "direction_tally": dict(Counter(f["direction"] for f in per_model.values())),
            "downward_models": sorted(m for m, f in per_model.items() if f["direction"] == "down"),
            "later_frame_fits": later_fits,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("no-response generations:", counts)
    for label, v in report["variants"].items():
        p = v["mean_pooled"]
        print(f"{label}: pooled gamma={p['threshold']} {p['mean_pass_low']:.4f}->{p['mean_pass_high']:.4f}; "
              f"directions {v['direction_tally']}; down {v['downward_models']}")
        for m, f in v["later_frame_fits"].items():
            print(f"   later frame {m}: gamma={f['threshold']} {f['mean_pass_low']:.4f}->{f['mean_pass_high']:.4f}")
        for m in args.models:
            f = v["per_model"][m]
            print(f"   all frames {m}: gamma={f['threshold']} {f['mean_pass_low']:.4f}->{f['mean_pass_high']:.4f}")


if __name__ == "__main__":
    main()
