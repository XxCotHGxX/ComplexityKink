"""
Stage D step 3: select a human calibration subsample.

The target is a compact but defensible human-scored set, usually 500 prompts,
balanced across rubric bins and enriched for ensemble-disagreement cases once
ensemble scores exist.

This script works before the ensemble is available too: it will balance only on
rubric_bin from the Stage D prompt file. If an ensemble score file is supplied,
it uses score variance to prioritize ambiguous prompts within each bin.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = ROOT / "data" / "stage_d" / "stage_d_prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage_d" / "human_calibration_prompts.jsonl"
DEFAULT_REPORT = ROOT / "data" / "stage_d" / "human_calibration_report.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_disagreement(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            if not pid:
                continue
            # Prefer explicit variance fields produced by future ensemble code.
            val = (
                rec.get("composite_variance")
                or rec.get("composite_std")
                or rec.get("mean_dimension_variance")
                or 0.0
            )
            out[pid] = float(val)
    return out


def target_counts(total: int, labels: list[str]) -> dict[str, int]:
    base = total // len(labels)
    rem = total % len(labels)
    return {label: base + (1 if i < rem else 0) for i, label in enumerate(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select human calibration prompts.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--ensemble-scores", default=None,
                        help="Optional ensemble score JSONL with disagreement fields.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--n-prompts", type=int, default=500)
    parser.add_argument("--high-disagreement-share", type=float, default=0.5,
                        help="Within each bin, this share comes from top-disagreement rows.")
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(Path(args.prompts))
    disagreements = load_disagreement(Path(args.ensemble_scores) if args.ensemble_scores else None)

    by_bin: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row.get("rubric_bin")
        if label:
            row = dict(row)
            row["calibration_priority"] = disagreements.get(row.get("prompt_id"), 0.0)
            by_bin[label].append(row)

    labels = sorted(by_bin.keys(), key=lambda x: float(x.split("-", 1)[0]))
    targets = target_counts(args.n_prompts, labels)

    selected = []
    shortages = {}
    for label in labels:
        pool = by_bin[label]
        target = targets[label]
        if len(pool) < target:
            selected.extend(pool)
            shortages[label] = target - len(pool)
            continue

        n_priority = int(round(target * args.high_disagreement_share))
        pool_sorted = sorted(pool, key=lambda r: r["calibration_priority"], reverse=True)
        priority = pool_sorted[:n_priority]
        priority_ids = {r["prompt_id"] for r in priority}
        rest = [r for r in pool if r["prompt_id"] not in priority_ids]
        rng.shuffle(rest)
        selected.extend(priority + rest[:target - len(priority)])

    rng.shuffle(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    realized = Counter(row["rubric_bin"] for row in selected)
    report = {
        "n_requested": args.n_prompts,
        "n_selected": len(selected),
        "targets": targets,
        "realized": dict(realized),
        "shortages": shortages,
        "used_disagreement_scores": bool(disagreements),
        "high_disagreement_share": args.high_disagreement_share,
        "output": str(output),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Human calibration sample")
    print(f"  Selected: {len(selected):,}/{args.n_prompts:,}")
    print(f"  Output:   {output}")
    print("  Counts:")
    for label in labels:
        print(f"    {label:>5}: {realized.get(label, 0):>3} / {targets[label]}")


if __name__ == "__main__":
    main()
