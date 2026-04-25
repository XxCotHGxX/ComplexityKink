#!/usr/bin/env python3
"""Scan generation result files and remove erroneous/empty outputs.

Checks each record for:
  1. Explicit error field set (non-null)
  2. Empty or missing code_cleaned
  3. raw_response that looks like an error message rather than code
  4. Duplicate prompt_ids (keeps the last valid one)

Usage:
  python src/data_provenance/clean_results.py [--dir data/generations] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# Patterns that indicate an error response got recorded as an answer
ERROR_PATTERNS = [
    r"^(Error|ERROR|error):",
    r"^Request failed",
    r"^HTTP \d{3}",
    r"^status_code=\d+",
    r"^openai\..*Error",
    r"^httpx\.",
    r"^Connection(Error|Timeout|Refused)",
    r"^Traceback \(most recent call last\)",
    r"^rate limit",
    r"^quota exceeded",
    r"^EMPTY_CODE:",
]
ERROR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

# Minimum reasonable code length (a one-liner like "print(1)" is ~8 chars)
MIN_CODE_LENGTH = 5


def is_valid_record(rec):
    """Return (is_valid, reason) for a generation record."""
    # 1. Explicit error
    if rec.get("error") is not None:
        return False, f"error field: {str(rec['error'])[:80]}"

    # 2. Missing or empty code
    code = rec.get("code_cleaned") or ""
    if not code.strip():
        return False, "empty code_cleaned"

    # 3. Code too short to be real
    if len(code.strip()) < MIN_CODE_LENGTH:
        return False, f"code too short ({len(code.strip())} chars)"

    # 4. Raw response looks like an error
    raw = rec.get("raw_response") or ""
    if ERROR_RE.search(raw[:200]):
        # But if code_cleaned is valid, the extractor worked ,  keep it
        if len(code.strip()) >= MIN_CODE_LENGTH:
            return True, "ok (raw looks like error but code extracted)"
        return False, f"raw_response is error: {raw[:80]}"

    return True, "ok"


def clean_file(filepath, dry_run=False):
    """Clean a single JSONL results file. Returns (kept, removed, dupes) counts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  LINE {line_num}: malformed JSON, removing")
                continue
            records.append(rec)

    if not records:
        return 0, 0, 0

    # Deduplicate: keep the LAST valid record per prompt_id
    # If no valid record exists for a prompt_id, drop all copies
    seen = {}  # prompt_id -> (index, record, is_valid, reason)
    for i, rec in enumerate(records):
        pid = rec.get("prompt_id", f"unknown_{i}")
        valid, reason = is_valid_record(rec)
        prev = seen.get(pid)
        if prev is None:
            seen[pid] = (i, rec, valid, reason)
        else:
            # Prefer valid over invalid; if both valid, keep later one
            _, _, prev_valid, _ = prev
            if valid or not prev_valid:
                seen[pid] = (i, rec, valid, reason)

    kept = []
    removed = 0
    dupes = 0
    invalid_examples = []

    # Count duplicates
    pid_counts = {}
    for rec in records:
        pid = rec.get("prompt_id", "?")
        pid_counts[pid] = pid_counts.get(pid, 0) + 1

    for pid, count in pid_counts.items():
        if count > 1:
            dupes += count - 1

    for pid, (idx, rec, valid, reason) in sorted(seen.items(), key=lambda x: x[1][0]):
        if valid:
            kept.append(rec)
        else:
            removed += 1
            if len(invalid_examples) < 3:
                invalid_examples.append(f"    {pid[:12]}...: {reason}")

    if invalid_examples:
        for ex in invalid_examples:
            print(ex)
        if removed > 3:
            print(f"    ... and {removed - 3} more")

    if not dry_run and (removed > 0 or dupes > 0):
        # Write to temp file, then copy over (safe for Azure File Share / SMB)
        import shutil
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir=str(filepath.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                for rec in kept:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            shutil.copy2(tmp_path, str(filepath))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return len(kept), removed, dupes


def main():
    parser = argparse.ArgumentParser(description="Clean erroneous outputs from generation results")
    parser.add_argument("--dir", default="data/generations", help="Directory containing .jsonl result files")
    parser.add_argument("--dry-run", action="store_true", help="Report issues without modifying files")
    parser.add_argument("--file", help="Clean a single file instead of the whole directory")
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        gen_dir = Path(args.dir)
        if not gen_dir.exists():
            print(f"Directory not found: {gen_dir}")
            sys.exit(1)
        files = sorted(gen_dir.glob("*.jsonl"))

    if not files:
        print("No .jsonl files found.")
        sys.exit(0)

    total_kept = 0
    total_removed = 0
    total_dupes = 0
    files_with_issues = 0

    print(f"{'Mode: DRY RUN' if args.dry_run else 'Cleaning files'}...")
    print(f"Scanning {len(files)} files...\n")

    for filepath in files:
        kept, removed, dupes = clean_file(filepath, dry_run=args.dry_run)
        total = kept + removed
        status = ""
        if removed > 0 or dupes > 0:
            parts = []
            if removed: parts.append(f"{removed} invalid")
            if dupes: parts.append(f"{dupes} dupes")
            status = f" -> removed {', '.join(parts)}"
            files_with_issues += 1
        print(f"  {filepath.name}: {total} records, {kept} kept{status}")
        total_kept += kept
        total_removed += removed
        total_dupes += dupes

    print(f"\n{'=' * 60}")
    print(f"Total: {total_kept + total_removed} records across {len(files)} files")
    print(f"  Kept:    {total_kept}")
    print(f"  Removed: {total_removed} invalid")
    print(f"  Dupes:   {total_dupes} duplicates merged")
    print(f"  Files with issues: {files_with_issues}/{len(files)}")
    if args.dry_run:
        print("\n  (Dry run ,  no files modified. Remove --dry-run to apply.)")


if __name__ == "__main__":
    main()
