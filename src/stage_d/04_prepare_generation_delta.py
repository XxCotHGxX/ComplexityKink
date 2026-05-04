"""
Stage D step 4: prepare retained/new prompt manifests for model generation.

After rubric-balanced pruning, the final prompt set contains:
  - retained Stage C prompt_ids that already have model generations/scored rows
  - newly sourced prompt_ids that need fresh generations

This script writes:
  data/stage_d/generation_delta/
    stage_d_new_prompts.jsonl
    stage_d_retained_prompts.jsonl
    stage_d_pruned_prompt_ids.txt
    per_model_missing/<model>.jsonl
    per_model_retained_counts.csv

Use the new prompt file with the existing generator:

  python src/data_provenance/02_generate_solutions.py \
      --prompts data/stage_d/generation_delta/stage_d_new_prompts.jsonl \
      --models src/stage_d/models_stage_d_panel.json \
      --output-dir data/stage_d/generations

The existing Stage C data remains untouched. Downstream Stage D scoring can
combine retained old rows plus newly generated rows by prompt_id.
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

DEFAULT_STAGE_C_PROMPTS = ROOT / "data" / "experiment_prompts.jsonl"
DEFAULT_STAGE_D_PROMPTS = ROOT / "data" / "stage_d" / "stage_d_prompts.jsonl"
DEFAULT_SCORED_DIR = ROOT / "data" / "scored_corrected"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "stage_d" / "generation_delta"


def load_prompts(path: Path) -> dict[str, dict]:
    prompts = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompts[rec["prompt_id"]] = rec
    return prompts


def load_model_prompt_ids(path: Path) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id") or rec.get("prompt_id")
            if pid:
                ids.add(pid)
    return ids


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage D generation delta.")
    parser.add_argument("--stage-c-prompts", default=str(DEFAULT_STAGE_C_PROMPTS))
    parser.add_argument("--stage-d-prompts", default=str(DEFAULT_STAGE_D_PROMPTS))
    parser.add_argument("--scored-dir", default=str(DEFAULT_SCORED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    stage_c = load_prompts(Path(args.stage_c_prompts))
    stage_d = load_prompts(Path(args.stage_d_prompts))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_c_ids = set(stage_c)
    stage_d_ids = set(stage_d)
    retained_ids = sorted(stage_c_ids & stage_d_ids)
    new_ids = sorted(stage_d_ids - stage_c_ids)
    pruned_ids = sorted(stage_c_ids - stage_d_ids)

    retained_prompts = [stage_d[pid] for pid in retained_ids]
    new_prompts = [stage_d[pid] for pid in new_ids]
    write_jsonl(output_dir / "stage_d_retained_prompts.jsonl", retained_prompts)
    write_jsonl(output_dir / "stage_d_new_prompts.jsonl", new_prompts)
    (output_dir / "stage_d_pruned_prompt_ids.txt").write_text(
        "\n".join(pruned_ids) + ("\n" if pruned_ids else ""),
        encoding="utf-8",
    )

    missing_dir = output_dir / "per_model_missing"
    missing_dir.mkdir(parents=True, exist_ok=True)
    rows_for_csv = []

    scored_files = sorted(Path(args.scored_dir).glob("*.jsonl"))
    for scored_path in scored_files:
        if scored_path.name.startswith("_"):
            continue
        model_id = scored_path.stem
        if model_id in STAGE_C_EXCLUDED_MODELS:
            continue
        model_ids = load_model_prompt_ids(scored_path)
        retained_have = len(stage_d_ids & model_ids)
        missing_ids = sorted(stage_d_ids - model_ids)
        missing_prompts = [stage_d[pid] for pid in missing_ids]
        write_jsonl(missing_dir / f"{model_id}.jsonl", missing_prompts)
        rows_for_csv.append({
            "model_id": model_id,
            "stage_d_prompts": len(stage_d_ids),
            "already_available": retained_have,
            "missing_to_generate": len(missing_ids),
            "new_stage_d_prompts": len(new_ids),
        })

    with (output_dir / "per_model_retained_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id", "stage_d_prompts", "already_available",
                "missing_to_generate", "new_stage_d_prompts",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print("Stage D generation delta")
    print(f"  Stage C prompts: {len(stage_c_ids):,}")
    print(f"  Stage D prompts: {len(stage_d_ids):,}")
    print(f"  Retained:        {len(retained_ids):,}")
    print(f"  New:             {len(new_ids):,}")
    print(f"  Pruned:          {len(pruned_ids):,}")
    print(f"  Output:          {output_dir}")
    if rows_for_csv:
        max_missing = max(r["missing_to_generate"] for r in rows_for_csv)
        min_missing = min(r["missing_to_generate"] for r in rows_for_csv)
        print(f"  Per-model missing range: {min_missing:,} to {max_missing:,}")


if __name__ == "__main__":
    main()
