"""Apply the independent audit to produce the camera-ready outcome.

Mirrors src/data_provenance/07_apply_judge.py, but with one auditor applied to
all 105,000 generations (docs/independent_audit_protocol.md):

    verdict "correct"   -> pass_rate 1.0
    verdict "incorrect" -> pass_rate 0.0
    anything else       -> the unit-test harness pass rate

Writes data/stage_d/scored_independent_audit/<model>.jsonl with the same schema
as data/stage_d/scored_combined/, so the analysis scripts run unchanged with
CK_SCORED_DIR pointing there. Every row keeps the raw harness value and the
reviewed-version (o4-mini, earlier frame only) value for provenance.

Usage:
    python src/audit/06_apply_independent_audit.py --data-root /srv/ckr/data \
        --audit-file audit_mai_thinking_1.jsonl --auditor MAI-Thinking-1
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERDICT_TO_PASS = {"correct": 1.0, "incorrect": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--audit-file", required=True, help="File under independent_audit/production/.")
    ap.add_argument("--auditor", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="Refuse to write unless this share of rows has a final verdict record.")
    args = ap.parse_args()

    scored_dir = args.data_root / "stage_d" / "scored_combined"
    out_dir = args.data_root / "stage_d" / "scored_independent_audit"
    audit_path = args.data_root / "independent_audit" / "production" / args.audit_file
    prompts = {json.loads(line)["prompt_id"]: json.loads(line).get("selection_source")
               for line in open(args.data_root / "stage_d" / "stage_d_prompts.jsonl", encoding="utf-8")}

    verdicts: dict[str, dict] = {}
    unreadable = 0
    with open(audit_path, "rb") as f:
        for raw in f:
            text = raw.strip(b"\x00\r\n ")
            if not text:
                continue
            try:
                rec = json.loads(text)
            except ValueError:
                unreadable += 1
                continue
            if rec.get("status") in ("ok", "parse_error"):
                verdicts[rec["case_id"]] = rec  # last final record wins
    if unreadable:
        print(f"skipped {unreadable} unreadable line(s) in {audit_path.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts, by_frame = Counter(), Counter()
    sums = {"harness": 0.0, "reviewed": 0.0, "independent": 0.0}
    frame_sums: dict[str, dict[str, float]] = {}
    n_rows = 0
    files = sorted(p for p in scored_dir.glob("*.jsonl") if not p.name.startswith("_"))
    rows_by_file = {}
    for path in files:
        rows = []
        seen = set()
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in prompts or r["id"] in seen:
                continue
            seen.add(r["id"])
            case_id = f"{path.stem}:{r['id']}"
            harness = r.get("harness_pass_rate", r.get("pass_rate"))
            rec = verdicts.get(case_id)
            verdict = rec.get("verdict") if rec and rec.get("status") == "ok" else None
            new_pass = VERDICT_TO_PASS.get(verdict, harness)
            counts[verdict if rec else "missing"] += 1
            frame = prompts[r["id"]]
            by_frame[(frame, "changed" if new_pass != harness else "same")] += 1
            fs = frame_sums.setdefault(frame, {"n": 0, "harness": 0.0, "reviewed": 0.0, "independent": 0.0})
            fs["n"] += 1
            for key, value in (("harness", harness), ("reviewed", r.get("pass_rate")), ("independent", new_pass)):
                sums[key] += value
                fs[key] += value
            n_rows += 1
            r["pass_rate_reviewed_o4mini_audit"] = r.get("pass_rate")
            r["pass_rate"] = new_pass
            r["independent_audit_verdict"] = verdict
            r["independent_auditor"] = args.auditor
            rows.append(r)
        rows_by_file[path.name] = rows

    covered = n_rows - counts["missing"]
    coverage = covered / n_rows if n_rows else 0.0
    summary = {
        "auditor": args.auditor, "audit_file": args.audit_file, "rows": n_rows,
        "coverage": coverage, "verdicts": dict(counts),
        "changed_by_frame": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_frame.items())},
        "mean_pass": {k: v / n_rows for k, v in sums.items()},
        "mean_pass_by_frame": {f: {k: fs[k] / fs["n"] for k in ("harness", "reviewed", "independent")}
                               for f, fs in frame_sums.items()},
    }
    print(json.dumps(summary, indent=2))
    if coverage < args.min_coverage:
        raise SystemExit(f"coverage {coverage:.4f} < {args.min_coverage}; not writing outputs")

    for name, rows in rows_by_file.items():
        with open(out_dir / name, "w", encoding="utf-8", newline="\n") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")
    with open(out_dir / "_summary.json", "w", encoding="utf-8", newline="\n") as out:
        json.dump(summary, out, indent=2)
    print(f"wrote {len(rows_by_file)} files to {out_dir}")


if __name__ == "__main__":
    main()
