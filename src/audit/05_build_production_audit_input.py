"""Build the production audit input: every scored generation in the benchmark.

Reads the harness-scored rows in data/stage_d/scored_combined/ (21 models x
5,000 prompts) and writes one audit row per generation with the raw
unit-test harness pass rate, exactly the fields the original audit saw.
Also writes the seeded 5% subsample audited by the secondary auditor.

Usage:
    python src/audit/05_build_production_audit_input.py --data-root /mnt/ckr/data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20260928
EXPECTED_MODELS, EXPECTED_PROMPTS = 21, 5000
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--secondary-fraction", type=float, default=0.05)
    args = ap.parse_args()

    scored_dir = args.data_root / "stage_d" / "scored_combined"
    out_dir = args.data_root / "independent_audit" / "production"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = {json.loads(line)["prompt_id"] for line in
               open(args.data_root / "stage_d" / "stage_d_prompts.jsonl", encoding="utf-8")}

    files = sorted(p for p in scored_dir.glob("*.jsonl") if not p.name.startswith("_"))
    assert len(files) == EXPECTED_MODELS, f"expected {EXPECTED_MODELS} scored files, found {len(files)}"
    case_ids = []
    tmp = out_dir / "audit_input.jsonl.partial"
    with open(tmp, "w", encoding="utf-8", newline="
") as out:
        for path in files:
            model_key = path.stem
            seen = set()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    pid = r["id"]
                    if pid not in prompts or pid in seen:
                        continue
                    seen.add(pid)
                    case_id = f"{model_key}:{pid}"
                    case_ids.append(case_id)
                    out.write(json.dumps({
                        "case_id": case_id,
                        "model_key": model_key,
                        "prompt_id": pid,
                        "task": r.get("input") or "",
                        "code": r.get("code_cleaned") or "",
                        "unit_tests": r.get("unit_tests") or "",
                        "harness_pass_rate": r.get("harness_pass_rate", r.get("pass_rate")),
                    }) + "\n")
            assert seen == prompts, f"{model_key}: covers {len(seen)} of {EXPECTED_PROMPTS} prompts"
    assert len(case_ids) == EXPECTED_MODELS * EXPECTED_PROMPTS
    # Write via copy rather than rename: os.replace is unreliable on Azure Files.
    final = out_dir / "audit_input.jsonl"
    final.write_bytes(tmp.read_bytes())
    tmp.unlink()

    rng = random.Random(SEED)
    secondary = set(rng.sample(sorted(case_ids), round(len(case_ids) * args.secondary_fraction)))
    with open(final, encoding="utf-8") as f, \
            open(out_dir / "audit_input_secondary_5pct.jsonl", "w", encoding="utf-8", newline="
") as out:
        for line in f:
            if json.loads(line)["case_id"] in secondary:
                out.write(line)
    print(f"wrote {len(case_ids)} rows to {final}; secondary sample {len(secondary)} rows")


if __name__ == "__main__":
    main()
