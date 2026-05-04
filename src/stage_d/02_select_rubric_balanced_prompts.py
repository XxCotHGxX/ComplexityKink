"""
Stage D step 2: fill rubric-bin gaps and prune over-represented bins.

Inputs:
  - Existing Stage C prompts and rubric scores.
  - Newly collected candidate prompts and their rubric scores.

Output:
  - A final prompt file, normally 5,000 prompts, balanced across rubric bins
    as far as the scored candidate pool allows.
  - A JSON audit report with realized counts, shortages, and source mix.

Important: this script only uses rubric scores already written to disk. It does
not call any model APIs. Score candidates first with:

  python src/data_provenance/05_score_complexity_rubric.py \
      --prompts data/stage_d/candidate_prompts.jsonl \
      --output data/stage_d/candidate_rubric_scores.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXISTING_PROMPTS = ROOT / "data" / "experiment_prompts.jsonl"
DEFAULT_EXISTING_SCORES = ROOT / "data" / "complexity_rubric_scores.jsonl"
DEFAULT_CANDIDATE_PROMPTS = ROOT / "data" / "stage_d" / "candidate_prompts.jsonl"
DEFAULT_CANDIDATE_SCORES = ROOT / "data" / "stage_d" / "candidate_rubric_scores.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage_d" / "stage_d_prompts.jsonl"
DEFAULT_REPORT = ROOT / "data" / "stage_d" / "stage_d_prompt_balance_report.json"
DEFAULT_AUDIT_FLAGS = (
    ROOT / "data" / "stage_d" / "stage_c_unit_test_audit_flags.csv",
    ROOT / "data" / "stage_d" / "candidate_unit_test_audit_flags.csv",
)

DEFAULT_RUBRIC_BINS = (
    ("0-3", 0, 3),
    ("4-6", 4, 6),
    ("7-9", 7, 9),
    ("10-12", 10, 12),
    ("13-15", 13, 15),
    ("16-18", 16, 18),
    ("19-24", 19, 24),
)


def parse_bins(spec: str | None) -> list[tuple[str, float, float]]:
    if not spec:
        return list(DEFAULT_RUBRIC_BINS)
    bins = []
    for part in spec.split(","):
        label = part.strip()
        if not label:
            continue
        if "-" not in label:
            raise ValueError(f"Invalid bin {label!r}; expected low-high")
        lo_s, hi_s = label.split("-", 1)
        lo = float(lo_s)
        hi = float(hi_s)
        bins.append((label, lo, hi))
    if not bins:
        raise ValueError("No bins parsed")
    return bins


def rubric_bin(composite: float | int | None, bins: list[tuple[str, float, float]]) -> str | None:
    if composite is None:
        return None
    value = float(composite)
    for label, lo, hi in bins:
        if lo <= value <= hi:
            return label
    return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
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
                if not pid or rec.get("composite") is None or not rec.get("scores"):
                    continue
                scores[pid] = rec
    return scores


def load_audit_flags(paths: list[Path]) -> dict[str, set[str]]:
    flags: dict[str, set[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = row.get("prompt_id")
                if not pid:
                    continue
                row_flags = {
                    flag for flag in (row.get("flags") or "").split(";")
                    if flag
                }
                if row_flags:
                    flags.setdefault(pid, set()).update(row_flags)
    return flags


def load_excluded_prompt_ids(paths: list[Path]) -> set[str]:
    prompt_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    pid = row.get("prompt_id")
                    if pid:
                        prompt_ids.add(pid)
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    prompt_ids.add(line)
                    continue
                if isinstance(rec, dict):
                    pid = rec.get("prompt_id") or rec.get("id")
                    if pid:
                        prompt_ids.add(pid)
                elif isinstance(rec, str):
                    prompt_ids.add(rec)
    return prompt_ids


def parse_flag_spec(spec: str | None) -> set[str]:
    if not spec:
        return set()
    return {flag.strip() for flag in spec.split(",") if flag.strip()}


def has_excluded_flag(flags: set[str], exact: set[str], prefixes: set[str]) -> bool:
    return bool(flags & exact) or any(
        flag.startswith(prefix)
        for flag in flags
        for prefix in prefixes
    )


def target_counts(total: int, bin_labels: list[str]) -> dict[str, int]:
    base = total // len(bin_labels)
    remainder = total % len(bin_labels)
    return {
        label: base + (1 if i < remainder else 0)
        for i, label in enumerate(bin_labels)
    }


def normalize_prompt(rec: dict, source: str) -> dict:
    out = dict(rec)
    out["selection_source"] = source
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a final Stage D prompt set balanced by rubric bins."
    )
    parser.add_argument("--existing-prompts", default=str(DEFAULT_EXISTING_PROMPTS))
    parser.add_argument("--existing-scores", default=str(DEFAULT_EXISTING_SCORES))
    parser.add_argument("--candidate-prompts", default=str(DEFAULT_CANDIDATE_PROMPTS))
    parser.add_argument("--candidate-scores", default=str(DEFAULT_CANDIDATE_SCORES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--audit-flags", nargs="*", default=[str(p) for p in DEFAULT_AUDIT_FLAGS],
                        help="CSV files from 08_audit_unit_tests.py.")
    parser.add_argument("--exclude-flags",
                        default="contract_hidden_test_callable,contract_io_prompt_callable_tests,risk_external_fixture_or_global,weak_many_duplicate_tests,weak_some_duplicate_tests",
                        help="Comma-separated exact audit flags to exclude.")
    parser.add_argument("--exclude-flag-prefixes", default="hard_",
                        help="Comma-separated audit flag prefixes to exclude.")
    parser.add_argument("--exclude-prompt-id-files", nargs="*", default=[],
                        help="Text, JSONL, or CSV files listing prompt_ids to exclude.")
    parser.add_argument("--n-prompts", type=int, default=5000)
    parser.add_argument("--bins", default=None,
                        help="Comma-separated inclusive composite bins, e.g. 0-3,4-6,...")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--prefer-existing", action="store_true",
                        help="Fill each bin from existing prompts before candidates.")
    parser.add_argument("--allow-short", action="store_true",
                        help="Write fewer than n-prompts if high bins are still short.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bins = parse_bins(args.bins)
    labels = [b[0] for b in bins]
    targets = target_counts(args.n_prompts, labels)

    existing_rows = [
        normalize_prompt(r, "stage_c_existing")
        for r in load_jsonl(Path(args.existing_prompts))
    ]
    candidate_rows = [
        normalize_prompt(r, "stage_d_candidate")
        for r in load_jsonl(Path(args.candidate_prompts))
    ]
    scores = load_scores(Path(args.existing_scores), Path(args.candidate_scores))
    audit_flags = load_audit_flags([Path(p) for p in args.audit_flags])
    excluded_prompt_ids = load_excluded_prompt_ids([Path(p) for p in args.exclude_prompt_id_files])
    excluded_flags = parse_flag_spec(args.exclude_flags)
    excluded_prefixes = parse_flag_spec(args.exclude_flag_prefixes)

    pools: dict[str, list[dict]] = defaultdict(list)
    missing_scores = 0
    duplicate_ids: set[str] = set()
    excluded_by_prompt_id = 0
    excluded_by_audit = 0
    excluded_by_audit_flag_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    ordered_rows = existing_rows + candidate_rows if args.prefer_existing else candidate_rows + existing_rows
    for row in ordered_rows:
        pid = row.get("prompt_id")
        if not pid:
            continue
        if pid in excluded_prompt_ids:
            excluded_by_prompt_id += 1
            continue
        if pid in seen_ids:
            duplicate_ids.add(pid)
            continue
        seen_ids.add(pid)
        row_flags = audit_flags.get(pid, set())
        if row_flags and has_excluded_flag(row_flags, excluded_flags, excluded_prefixes):
            excluded_by_audit += 1
            for flag in row_flags:
                excluded_by_audit_flag_counts[flag] += 1
            continue
        score = scores.get(pid)
        if score is None:
            missing_scores += 1
            continue
        label = rubric_bin(score.get("composite"), bins)
        if label is None:
            continue
        out = dict(row)
        out["rubric_scores"] = score["scores"]
        out["rubric_composite"] = score["composite"]
        out["rubric_bin"] = label
        if row_flags:
            out["unit_test_audit_flags"] = sorted(row_flags)
        pools[label].append(out)

    for label in labels:
        rng.shuffle(pools[label])
        if args.prefer_existing:
            existing_pool = [
                row for row in pools[label]
                if row["selection_source"] == "stage_c_existing"
            ]
            candidate_pool = [
                row for row in pools[label]
                if row["selection_source"] != "stage_c_existing"
            ]
            pools[label] = existing_pool + candidate_pool

    selected: list[dict] = []
    shortages = {}
    for label in labels:
        take = min(targets[label], len(pools[label]))
        selected.extend(pools[label][:take])
        if take < targets[label]:
            shortages[label] = targets[label] - take

    if shortages and not args.allow_short:
        shortage_text = ", ".join(f"{k}: {v}" for k, v in shortages.items())
        raise SystemExit(
            "Rubric bins are still under-filled. Score more candidates or pass "
            f"--allow-short. Shortages: {shortage_text}"
        )

    rng.shuffle(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    realized = Counter(row["rubric_bin"] for row in selected)
    source_mix = Counter(row["selection_source"] for row in selected)
    pool_counts = {label: len(pools[label]) for label in labels}
    report = {
        "n_requested": args.n_prompts,
        "n_selected": len(selected),
        "bins": [{"label": label, "low": lo, "high": hi} for label, lo, hi in bins],
        "targets": targets,
        "realized": dict(realized),
        "shortages": shortages,
        "pool_counts": pool_counts,
        "source_mix": dict(source_mix),
        "missing_score_rows": missing_scores,
        "duplicate_prompt_ids_skipped": len(duplicate_ids),
        "excluded_prompt_ids": {
            "files": args.exclude_prompt_id_files,
            "unique_ids_loaded": len(excluded_prompt_ids),
            "excluded_rows": excluded_by_prompt_id,
        },
        "audit_filter": {
            "audit_flag_files": args.audit_flags,
            "exclude_flags": sorted(excluded_flags),
            "exclude_flag_prefixes": sorted(excluded_prefixes),
            "excluded_rows": excluded_by_audit,
            "excluded_flag_counts": dict(excluded_by_audit_flag_counts),
        },
        "output": str(output),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Stage D rubric-balanced selection")
    print(f"  Output:     {output}")
    print(f"  Selected:   {len(selected):,}/{args.n_prompts:,}")
    print(f"  Report:     {report_path}")
    print("  Realized counts:")
    for label in labels:
        print(
            f"    {label:>5}: {realized.get(label, 0):>4} "
            f"(target {targets[label]}, pool {pool_counts[label]})"
        )
    if shortages:
        print("  Shortages:")
        for label, n in shortages.items():
            print(f"    {label:>5}: {n}")
    if audit_flags:
        print(f"  Audit-excluded rows: {excluded_by_audit:,}")


if __name__ == "__main__":
    main()
