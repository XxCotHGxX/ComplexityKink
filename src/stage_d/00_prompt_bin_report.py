"""
Stage D utility: report prompt counts across rubric complexity bins.

This is intentionally read-only. It summarizes existing Stage C prompts,
candidate prompts, and an optional finalized Stage D prompt set against the
same rubric-bin specification used by 02_select_rubric_balanced_prompts.py.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from importlib import util


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXISTING_PROMPTS = ROOT / "data" / "experiment_prompts.jsonl"
DEFAULT_EXISTING_SCORES = ROOT / "data" / "complexity_rubric_scores.jsonl"
DEFAULT_CANDIDATE_PROMPTS = ROOT / "data" / "stage_d" / "candidate_prompts.jsonl"
DEFAULT_CANDIDATE_SCORES = ROOT / "data" / "stage_d" / "candidate_rubric_scores.jsonl"
DEFAULT_STAGE_D_PROMPTS = ROOT / "data" / "stage_d" / "stage_d_prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage_d" / "prompt_bin_report.json"


def _load_selector_module():
    path = ROOT / "src" / "stage_d" / "02_select_rubric_balanced_prompts.py"
    spec = util.spec_from_file_location("stage_d_prompt_selector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import selector helpers from {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SELECTOR = _load_selector_module()
parse_bins = _SELECTOR.parse_bins
rubric_bin = _SELECTOR.rubric_bin
target_counts = _SELECTOR.target_counts


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_scores(*paths: Path) -> dict[str, dict]:
    scores: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = rec.get("prompt_id")
                if pid and rec.get("composite") is not None and rec.get("scores"):
                    scores[pid] = rec
    return scores


def prompt_id(row: dict) -> str | None:
    return row.get("prompt_id") or row.get("id")


def summarize(
    name: str,
    prompt_rows: list[dict],
    scores: dict[str, dict],
    bins: list[tuple[str, float, float]],
) -> dict:
    labels = [b[0] for b in bins]
    counts: Counter[str] = Counter()
    missing_scores = 0
    out_of_range = 0
    seen: set[str] = set()
    duplicates = 0

    for row in prompt_rows:
        pid = prompt_id(row)
        if not pid:
            continue
        if pid in seen:
            duplicates += 1
            continue
        seen.add(pid)
        score = scores.get(pid)
        if score is None:
            missing_scores += 1
            continue
        label = rubric_bin(score.get("composite"), bins)
        if label is None:
            out_of_range += 1
            continue
        counts[label] += 1

    return {
        "name": name,
        "prompt_rows": len(prompt_rows),
        "unique_prompt_ids": len(seen),
        "scored_prompt_ids": sum(counts.values()),
        "missing_scores": missing_scores,
        "out_of_range_scores": out_of_range,
        "duplicate_prompt_ids": duplicates,
        "counts": {label: counts.get(label, 0) for label in labels},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage D rubric bins.")
    parser.add_argument("--existing-prompts", default=str(DEFAULT_EXISTING_PROMPTS))
    parser.add_argument("--existing-scores", default=str(DEFAULT_EXISTING_SCORES))
    parser.add_argument("--candidate-prompts", default=str(DEFAULT_CANDIDATE_PROMPTS))
    parser.add_argument("--candidate-scores", default=str(DEFAULT_CANDIDATE_SCORES))
    parser.add_argument("--stage-d-prompts", default=str(DEFAULT_STAGE_D_PROMPTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n-prompts", type=int, default=5000)
    parser.add_argument("--bins", default="0-3,4-6,7-9,10-12,13-15,16-24")
    args = parser.parse_args()

    bins = parse_bins(args.bins)
    labels = [b[0] for b in bins]
    targets = target_counts(args.n_prompts, labels)
    scores = load_scores(Path(args.existing_scores), Path(args.candidate_scores))

    existing = load_jsonl(Path(args.existing_prompts))
    candidates = load_jsonl(Path(args.candidate_prompts))
    stage_d = load_jsonl(Path(args.stage_d_prompts))

    reports = [
        summarize("stage_c_existing", existing, scores, bins),
        summarize("stage_d_candidates", candidates, scores, bins),
        summarize("combined_pool", candidates + existing, scores, bins),
    ]
    if stage_d:
        reports.append(summarize("stage_d_selected", stage_d, scores, bins))

    combined = reports[2]["counts"]
    shortages = {
        label: max(0, targets[label] - combined.get(label, 0))
        for label in labels
    }

    report = {
        "bins": [{"label": label, "low": lo, "high": hi} for label, lo, hi in bins],
        "n_prompts": args.n_prompts,
        "targets": targets,
        "reports": reports,
        "combined_pool_shortages_vs_target": shortages,
        "score_rows_loaded": len(scores),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Stage D prompt-bin report")
    print(f"  Output: {output}")
    print("  Targets:")
    for label in labels:
        print(f"    {label:>5}: {targets[label]}")
    for item in reports:
        print(f"  {item['name']}: {item['scored_prompt_ids']:,} scored")
        for label in labels:
            print(f"    {label:>5}: {item['counts'][label]}")
        if item["missing_scores"]:
            print(f"    missing_scores: {item['missing_scores']}")
    if any(shortages.values()):
        print("  Combined-pool shortages:")
        for label in labels:
            if shortages[label]:
                print(f"    {label:>5}: {shortages[label]}")


if __name__ == "__main__":
    main()
