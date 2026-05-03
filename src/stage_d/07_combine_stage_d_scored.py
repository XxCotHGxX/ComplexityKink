"""
Stage D step 7: combine retained Stage C scored rows with new Stage D scored rows.

This script never mutates Stage C data. It writes a Stage D-only scored panel
under data/stage_d/scored_combined/ by taking:

  - retained prompt rows from data/scored_corrected/
  - newly generated and test-scored rows from data/stage_d/scored_new/

The output is one JSONL per model with at most one row per Stage D prompt_id.
It also writes a coverage report so partial generation runs are easy to audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import STAGE_C_EXCLUDED_MODELS  # noqa: E402

DEFAULT_STAGE_D_PROMPTS = ROOT / "data" / "stage_d" / "stage_d_prompts.jsonl"
DEFAULT_STAGE_C_SCORED = ROOT / "data" / "scored_corrected"
DEFAULT_STAGE_D_SCORED_NEW = ROOT / "data" / "stage_d" / "scored_new"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "stage_d" / "scored_combined"
DEFAULT_REPORT = ROOT / "data" / "stage_d" / "scored_combined_coverage.csv"


def load_prompts(path: Path) -> dict[str, dict]:
    prompts: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                prompts[rec["prompt_id"]] = rec
    return prompts


def row_prompt_id(row: dict) -> str | None:
    return row.get("id") or row.get("prompt_id")


def load_rows_by_prompt(path: Path, allowed_ids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row_prompt_id(row)
            if pid in allowed_ids:
                rows[pid] = row
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def add_complexity_provenance(row: dict, prompt: dict | None) -> dict:
    row = dict(row)
    if row.get("kappa_cyclomatic") is not None:
        row.setdefault("kappa_cyclomatic_source", "lizard_on_generated_output")
    if prompt and "reference_cc" in prompt:
        row.setdefault("reference_cc", prompt.get("reference_cc"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine retained and newly scored rows for Stage D."
    )
    parser.add_argument("--stage-d-prompts", default=str(DEFAULT_STAGE_D_PROMPTS))
    parser.add_argument("--stage-c-scored-dir", default=str(DEFAULT_STAGE_C_SCORED))
    parser.add_argument("--stage-d-scored-new-dir", default=str(DEFAULT_STAGE_D_SCORED_NEW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    prompts = load_prompts(Path(args.stage_d_prompts))
    stage_d_ids = set(prompts)
    stage_c_dir = Path(args.stage_c_scored_dir)
    stage_d_new_dir = Path(args.stage_d_scored_new_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict] = []
    model_files = sorted(stage_c_dir.glob("*.jsonl"))
    for old_path in model_files:
        if old_path.name.startswith("_"):
            continue
        model_name = old_path.stem
        if model_name in STAGE_C_EXCLUDED_MODELS:
            continue

        old_rows = load_rows_by_prompt(old_path, stage_d_ids)
        new_rows = load_rows_by_prompt(stage_d_new_dir / old_path.name, stage_d_ids)

        combined = dict(old_rows)
        combined.update(new_rows)
        missing_ids = sorted(stage_d_ids - set(combined))

        ordered_rows = [
            add_complexity_provenance(combined[pid], prompts.get(pid))
            for pid in sorted(combined)
        ]
        write_jsonl(output_dir / old_path.name, ordered_rows)

        report_rows.append({
            "model_id": model_name,
            "stage_d_prompts": len(stage_d_ids),
            "retained_from_stage_c": len(old_rows),
            "new_scored_rows": len(new_rows),
            "combined_rows": len(combined),
            "missing_rows": len(missing_ids),
        })

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id", "stage_d_prompts", "retained_from_stage_c",
                "new_scored_rows", "combined_rows", "missing_rows",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("Stage D scored-panel combine")
    print(f"  Models: {len(report_rows)}")
    print(f"  Output: {output_dir}")
    print(f"  Report: {report}")
    if report_rows:
        min_rows = min(r["combined_rows"] for r in report_rows)
        max_rows = max(r["combined_rows"] for r in report_rows)
        print(f"  Combined row range: {min_rows:,} to {max_rows:,}")


if __name__ == "__main__":
    main()
