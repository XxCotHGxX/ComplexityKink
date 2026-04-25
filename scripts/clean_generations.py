"""Purge bad/duplicate records from a generations JSONL file in-place.

Keeps one record per prompt_id, preferring records that are:
  - error is None
  - code_cleaned is non-empty

Writes a .bak backup next to the original before overwriting.

Usage:
    python scripts/clean_generations.py <file.jsonl> [<file2.jsonl> ...]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def is_good(rec: dict) -> bool:
    if rec.get("error"):
        return False
    code = rec.get("code_cleaned") or ""
    return bool(code.strip())


def clean_file(path: Path) -> None:
    records_by_id: dict[str, dict] = {}
    total = 0
    bad_json = 0

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
            if not pid:
                continue
            existing = records_by_id.get(pid)
            if existing is None:
                records_by_id[pid] = rec
            elif not is_good(existing) and is_good(rec):
                records_by_id[pid] = rec

    kept = sum(1 for r in records_by_id.values() if is_good(r))
    dropped_bad = len(records_by_id) - kept
    # Only keep good records; purge bad-only prompt_ids so resume will retry them
    final = [r for r in records_by_id.values() if is_good(r)]

    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup: {bak.name}")

    with path.open("w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(rec) + "\n")

    print(f"  {path.name}: {total} in -> {len(final)} kept "
          f"(bad_json={bad_json}, purged_bad_only_ids={dropped_bad})")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        clean_file(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
