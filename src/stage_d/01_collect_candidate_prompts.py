"""
Stage D step 1: collect additional prompt candidates from the original dataset.

This script does not score prompts. It builds a quality-filtered candidate file
from data/final_results_scored.jsonl so the existing rubric scorer can be run
on the new prompts:

  python src/data_provenance/05_score_complexity_rubric.py \
      --prompts data/stage_d/candidate_prompts.jsonl \
      --output data/stage_d/candidate_rubric_scores.jsonl

The collector excludes prompt_ids already present in the Stage C prompt set and
optionally excludes candidates already written by earlier Stage D runs. The
default sampling is tail-heavy over reference cyclomatic-complexity bins because
the current 5,000-prompt set is badly under-covered in the high rubric tail.
Final balancing is still done from rubric scores, not reference CC.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "final_results_scored.jsonl"
DEFAULT_EXISTING_PROMPTS = ROOT / "data" / "experiment_prompts.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage_d" / "candidate_prompts.jsonl"

REFERENCE_BINS = ("1", "2", "3", "4", "5", "6-7", "8-10", "11-15", "16+")
TAIL_HEAVY_WEIGHTS = {
    "1": 0.5,
    "2": 0.5,
    "3": 0.75,
    "4": 0.75,
    "5": 1.0,
    "6-7": 1.5,
    "8-10": 2.0,
    "11-15": 2.5,
    "16+": 3.0,
}


def _load_stage_c_selection_module():
    path = ROOT / "src" / "data_provenance" / "01_select_prompts.py"
    spec = importlib.util.spec_from_file_location("stage_c_select_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import selection helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STAGE_C = _load_stage_c_selection_module()
compute_cc_quick = _STAGE_C.compute_cc_quick
clean_code_from_markdown = _STAGE_C.clean_code_from_markdown
is_trivial_test = _STAGE_C.is_trivial_test


def reference_bin(cc: int | None) -> str | None:
    if cc is None:
        return None
    if cc <= 5:
        return str(cc)
    if cc <= 7:
        return "6-7"
    if cc <= 10:
        return "8-10"
    if cc <= 15:
        return "11-15"
    return "16+"


def load_prompt_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("prompt_id") or rec.get("id")
            if pid:
                ids.add(pid)
    return ids


def allocate_targets(n_candidates: int, mode: str) -> dict[str, int]:
    if mode == "uniform":
        base = n_candidates // len(REFERENCE_BINS)
        remainder = n_candidates % len(REFERENCE_BINS)
        return {
            b: base + (1 if i < remainder else 0)
            for i, b in enumerate(REFERENCE_BINS)
        }

    total_weight = sum(TAIL_HEAVY_WEIGHTS.values())
    raw = {
        b: int(n_candidates * TAIL_HEAVY_WEIGHTS[b] / total_weight)
        for b in REFERENCE_BINS
    }
    short = n_candidates - sum(raw.values())
    for b in sorted(REFERENCE_BINS, key=lambda x: TAIL_HEAVY_WEIGHTS[x], reverse=True):
        if short <= 0:
            break
        raw[b] += 1
        short -= 1
    return raw


def allocate_custom_targets(n_candidates: int, requested_bins: list[str]) -> dict[str, int]:
    bad = [b for b in requested_bins if b not in REFERENCE_BINS]
    if bad:
        raise ValueError(f"Unknown reference bins: {bad}")
    base = n_candidates // len(requested_bins)
    remainder = n_candidates % len(requested_bins)
    targets = {b: 0 for b in REFERENCE_BINS}
    for i, b in enumerate(requested_bins):
        targets[b] = base + (1 if i < remainder else 0)
    return targets


def update_reservoir(
    reservoir: list[dict],
    target: int,
    seen_count: int,
    rec: dict,
    rng: random.Random,
) -> None:
    if target <= 0:
        return
    if len(reservoir) < target:
        reservoir.append(rec)
        return
    j = rng.randrange(seen_count)
    if j < target:
        reservoir[j] = rec


def valid_candidate_record(rec: dict) -> dict | None:
    tests = rec.get("unit_tests", "")
    if not tests or tests == "[]":
        return None
    if is_trivial_test(tests):
        return None

    exec_status = rec.get("tests_execution_status", [])
    if exec_status and any(s != "pass" for s in exec_status):
        return None

    prompt = rec.get("input", "")
    if not prompt or len(prompt.strip()) < 20:
        return None

    code = rec.get("code_cleaned", "")
    if not code:
        code = clean_code_from_markdown(rec.get("output", ""))
    cc = compute_cc_quick(code)
    b = reference_bin(cc)
    if b is None:
        return None

    return {
        "prompt_id": rec.get("id"),
        "input": prompt,
        "unit_tests": tests,
        "reference_cc": cc,
        "reference_bin": b,
        "lang": rec.get("lang", "python"),
        "selection_source": "stage_d_candidate_pool",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Stage D prompt candidates for rubric scoring."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--existing-prompts", default=str(DEFAULT_EXISTING_PROMPTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n-candidates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--scan-limit", type=int, default=0,
                        help="Records to scan from the source (0 = full file).")
    parser.add_argument("--reference-allocation", choices=("tail-heavy", "uniform"),
                        default="tail-heavy")
    parser.add_argument("--only-reference-bins", default=None,
                        help="Comma-separated reference bins to sample uniformly, e.g. 11-15,16+.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace output instead of appending new candidates.")
    args = parser.parse_args()

    source = Path(args.source)
    existing_prompts = Path(args.existing_prompts)
    output = Path(args.output)
    rng = random.Random(args.seed)

    if not source.exists():
        raise FileNotFoundError(source)

    excluded_ids = load_prompt_ids(existing_prompts)
    existing_candidate_ids: set[str] = set()
    if output.exists() and not args.overwrite:
        existing_candidate_ids = load_prompt_ids(output)
        excluded_ids |= existing_candidate_ids

    if args.only_reference_bins:
        requested = [b.strip() for b in args.only_reference_bins.split(",") if b.strip()]
        targets = allocate_custom_targets(args.n_candidates, requested)
        allocation_label = f"custom({','.join(requested)})"
    else:
        targets = allocate_targets(args.n_candidates, args.reference_allocation)
        allocation_label = args.reference_allocation
    reservoirs: dict[str, list[dict]] = {b: [] for b in REFERENCE_BINS}
    seen_by_bin: dict[str, int] = defaultdict(int)
    scanned = 0
    eligible = 0
    duplicate_or_existing = 0

    print("Stage D candidate collection")
    print(f"  Source:              {source}")
    print(f"  Existing prompt ids: {len(load_prompt_ids(existing_prompts))}")
    print(f"  Existing candidates: {len(existing_candidate_ids)}")
    print(f"  New candidates:      {args.n_candidates}")
    print(f"  Allocation:          {allocation_label}")
    print("  Reference-bin targets:")
    for b in REFERENCE_BINS:
        print(f"    {b:>5}: {targets[b]}")

    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if args.scan_limit and scanned >= args.scan_limit:
                break
            scanned += 1
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("id")
            if not pid or pid in excluded_ids:
                duplicate_or_existing += 1
                continue

            cand = valid_candidate_record(rec)
            if cand is None:
                continue

            eligible += 1
            b = cand["reference_bin"]
            seen_by_bin[b] += 1
            update_reservoir(
                reservoirs[b], targets[b], seen_by_bin[b], cand, rng,
            )

            if scanned % 100000 == 0:
                kept = sum(len(v) for v in reservoirs.values())
                print(f"  scanned={scanned:,} eligible={eligible:,} kept={kept:,}")

    selected: list[dict] = []
    for b in REFERENCE_BINS:
        selected.extend(reservoirs[b])
    rng.shuffle(selected)

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    with output.open(mode, encoding="utf-8") as out:
        for rec in selected:
            out.write(json.dumps(rec) + "\n")

    print("\nCollection complete")
    print(f"  Scanned:             {scanned:,}")
    print(f"  Eligible:            {eligible:,}")
    print(f"  Existing duplicates: {duplicate_or_existing:,}")
    print(f"  Written this run:    {len(selected):,}")
    print(f"  Output:              {output}")
    print("  Realized reference-bin counts:")
    for b in REFERENCE_BINS:
        short = max(0, targets[b] - len(reservoirs[b]))
        msg = f"    {b:>5}: {len(reservoirs[b])}"
        if short:
            msg += f" (short {short})"
        print(msg)


if __name__ == "__main__":
    main()
