"""Audit LLM generation JSONL files for completeness and malformed responses.

Checks per file:
  - total lines / valid JSON
  - duplicate prompt_ids
  - empty or whitespace-only raw_response
  - empty code_cleaned
  - non-null error field
  - unique prompt coverage vs experiment_prompts.jsonl

Usage:
  python scripts/audit_generations.py                    # audit all
  python scripts/audit_generations.py <file.jsonl> ...   # specific files
  python scripts/audit_generations.py --fix-dupes        # write deduped .clean.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "generations"
PROMPTS = ROOT / "data" / "experiment_prompts.jsonl"


def load_expected_ids() -> set[str]:
    ids = set()
    if PROMPTS.exists():
        with PROMPTS.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    pass
    return ids


def audit_file(path: Path, expected: set[str], fix_dupes: bool = False) -> dict:
    total = 0
    bad_json = 0
    empty_raw = 0
    empty_code = 0
    errors = 0
    ids: list[str] = []
    seen_first: dict[str, dict] = {}  # prompt_id -> first record (for dedup)
    dup_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            pid = rec.get("prompt_id")
            if pid:
                ids.append(pid)
                if pid in seen_first:
                    dup_lines += 1
                else:
                    seen_first[pid] = rec
            raw = rec.get("raw_response") or ""
            code = rec.get("code_cleaned") or ""
            if not raw.strip():
                empty_raw += 1
            if not code.strip():
                empty_code += 1
            if rec.get("error"):
                errors += 1

    id_counts = Counter(ids)
    unique_ids = set(ids)
    dup_ids = {k: v for k, v in id_counts.items() if v > 1}
    missing = expected - unique_ids if expected else set()
    extra = unique_ids - expected if expected else set()

    result = {
        "file": path.name,
        "total_lines": total,
        "bad_json": bad_json,
        "unique_prompts": len(unique_ids),
        "duplicate_ids": len(dup_ids),
        "duplicate_lines": dup_lines,
        "empty_raw_response": empty_raw,
        "empty_code_cleaned": empty_code,
        "error_field_set": errors,
        "missing_vs_expected": len(missing),
        "extra_vs_expected": len(extra),
    }

    if fix_dupes and dup_lines > 0:
        out = path.with_suffix(".clean.jsonl")
        with out.open("w", encoding="utf-8") as f:
            for rec in seen_first.values():
                f.write(json.dumps(rec) + "\n")
        result["cleaned_written"] = str(out.name)
        result["cleaned_lines"] = len(seen_first)

    return result


def main() -> int:
    args = sys.argv[1:]
    fix = "--fix-dupes" in args
    args = [a for a in args if not a.startswith("--")]

    if args:
        files = [Path(a) if Path(a).exists() else GEN_DIR / a for a in args]
    else:
        files = sorted(GEN_DIR.glob("*.jsonl"))

    expected = load_expected_ids()
    print(f"Expected prompts: {len(expected)}")
    print(f"Auditing {len(files)} file(s)\n")

    header = f"{'file':<55} {'lines':>7} {'uniq':>6} {'dup_ln':>7} {'empty_r':>8} {'empty_c':>8} {'err':>5} {'miss':>6} {'extra':>6}"
    print(header)
    print("-" * len(header))

    issues: list[dict] = []
    for path in files:
        if not path.exists():
            print(f"  [missing] {path}")
            continue
        r = audit_file(path, expected, fix_dupes=fix)
        flag = ""
        if (r["bad_json"] or r["duplicate_lines"] or r["empty_raw_response"]
                or r["empty_code_cleaned"] or r["error_field_set"]
                or r["missing_vs_expected"]):
            flag = "  <-- ISSUES"
            issues.append(r)
        print(f"{r['file']:<55} {r['total_lines']:>7} {r['unique_prompts']:>6} "
              f"{r['duplicate_lines']:>7} {r['empty_raw_response']:>8} "
              f"{r['empty_code_cleaned']:>8} {r['error_field_set']:>5} "
              f"{r['missing_vs_expected']:>6} {r['extra_vs_expected']:>6}{flag}")
        if r.get("cleaned_written"):
            print(f"    -> wrote {r['cleaned_written']} ({r['cleaned_lines']} unique records)")

    print()
    if issues:
        print(f"{len(issues)} file(s) flagged. Re-run with --fix-dupes to write deduped .clean.jsonl files.")
    else:
        print("All clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
