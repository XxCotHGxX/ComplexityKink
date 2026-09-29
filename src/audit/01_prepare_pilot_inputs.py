"""Step 1 of the independent-audit pilot: collect candidate prompts.

Joins the 5,000 benchmark prompts with their OpenCodeInstruct reference
solutions and four-judge ensemble composites, then draws a composite-stratified
candidate sample for the known-answer set. No code is executed here.

See docs/independent_audit_protocol.md.

Usage:
    python src/audit/01_prepare_pilot_inputs.py --data-root D:/path/to/data
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260928
ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def display_bin(composite: float) -> int:
    # Bin b contains composites in [b - 0.5, b + 0.5).
    return int(math.floor(composite + 0.5))


def find_references(source: Path, wanted: set[str]) -> dict[str, dict]:
    """Scan the source extraction for the reference solution of each prompt."""
    found: dict[str, dict] = {}
    with open(source, encoding="utf-8") as f:
        for line in f:
            head = line[:400]
            j = head.find('"id": "')
            pid = head[j + 7:j + 39] if j >= 0 else json.loads(line).get("id")
            if pid in wanted and pid not in found:
                rec = json.loads(line)
                found[pid] = {
                    "reference_code": rec.get("code_cleaned") or "",
                    "source_average_test_score": rec.get("average_test_score"),
                }
                if len(found) == len(wanted):
                    break
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-candidates", type=int, default=360,
                    help="Candidates to execute; more than the 150 kept, to survive filters.")
    args = ap.parse_args()
    data = args.data_root
    out_dir = args.out_dir or data / "independent_audit" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = {r["prompt_id"]: r for r in load_jsonl(data / "stage_d" / "stage_d_prompts.jsonl")}
    ensemble = {r["prompt_id"]: r for r in load_jsonl(data / "stage_d" / "ensemble_scores_current_aggregated.jsonl")}
    assert len(prompts) == 5000 and set(prompts) == set(ensemble), "benchmark/ensemble join must be 5,000"

    refs = find_references(data / "final_results_scored.jsonl", set(prompts))
    assert len(refs) == 5000, f"only {len(refs)} reference solutions found"

    by_bin: dict[int, list[str]] = defaultdict(list)
    rows = {}
    for pid, p in prompts.items():
        composite = float(sum(ensemble[pid]["scores_mean"].values()))
        ref = refs[pid]["reference_code"]
        if not ref.strip():
            continue
        rows[pid] = {
            "prompt_id": pid,
            "frame": p.get("selection_source"),
            "composite": round(composite, 4),
            "display_bin": display_bin(composite),
            "task": p["input"],
            "unit_tests": p["unit_tests"],
            "reference_code": ref,
            "source_average_test_score": refs[pid]["source_average_test_score"],
        }
        by_bin[rows[pid]["display_bin"]].append(pid)

    # Equal allocation across display bins, capped by bin support.
    rng = random.Random(SEED)
    bins = sorted(by_bin)
    per_bin = math.ceil(args.n_candidates / len(bins))
    chosen: list[str] = []
    for b in bins:
        ids = sorted(by_bin[b])
        rng.shuffle(ids)
        chosen.extend(ids[:per_bin])
    rng.shuffle(chosen)

    out = out_dir / "pilot_inputs.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for pid in chosen:
            f.write(json.dumps(rows[pid]) + "\n")
    counts = {b: min(len(by_bin[b]), per_bin) for b in bins}
    print(f"wrote {len(chosen)} candidates to {out}")
    print("per display bin:", counts)


if __name__ == "__main__":
    main()
