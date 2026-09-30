"""Inter-auditor agreement on the seeded 5% production sample.

Compares the primary auditor's final verdicts with each second auditor's on the
cases where both return correct or incorrect. Verdicts count only under the
completeness rule (``final_verdict``: responses cut off by the token limit or a
content filter do not count). Generations with no code are marked incorrect by
rule without an auditor request, so they are excluded from the agreement
statistics (they would agree trivially); counts with them are reported too.

    python scripts/camera_ready/auditor_agreement.py \
        --out results/camera_ready/independent_audit/auditor_agreement.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "audit"))
from independent_audit import final_verdict, read_records  # noqa: E402

PRIMARY = ("MiMo-V2.6-Pro", "audit_mimo_v26_pro.jsonl")
SECOND = (("MAI-Thinking-1", "audit_mai_thinking_1.jsonl"),
          ("Phi-4-reasoning", "audit_phi4_reasoning.jsonl"))


def final_records(path: Path) -> dict[str, dict]:
    records, _ = read_records(path)
    out = {}
    for rec in records:
        if rec.get("status") in ("ok", "parse_error"):
            out[rec["case_id"]] = rec  # last final record wins, as in the apply step
    return out


def agreement(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    pa = sum(a == "correct" for a, _ in pairs) / n
    pb = sum(b == "correct" for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    po = agree / n
    table = {f"{a}|{b}": sum(1 for x in pairs if x == (a, b))
             for a in ("correct", "incorrect") for b in ("correct", "incorrect")}
    return {"n": n, "agreement": po, "cohen_kappa": (po - pe) / (1 - pe), "table": table}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-dir", type=Path, default=ROOT / "data" / "independent_audit" / "production")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sample = [json.loads(line)["case_id"]
              for line in open(args.production_dir / "audit_input_secondary_5pct.jsonl", encoding="utf-8")
              if line.strip()]
    primary = final_records(args.production_dir / PRIMARY[1])
    report = {"primary_auditor": PRIMARY[0], "sample_size": len(sample),
              "rule": "final_verdict (complete responses only); empty-code rule cases excluded"}
    for name, fname in SECOND:
        other = final_records(args.production_dir / fname)
        rule_cases = {c for c in sample
                      if (primary.get(c) or {}).get("parse_status") == "rule"
                      or (other.get(c) or {}).get("parse_status") == "rule"}
        both = [(final_verdict(primary.get(c)), final_verdict(other.get(c))) for c in sample]
        decisive = {"correct", "incorrect"}
        with_rule = [p for p in both if p[0] in decisive and p[1] in decisive]
        without_rule = [p for c, p in zip(sample, both)
                        if p[0] in decisive and p[1] in decisive and c not in rule_cases]
        report[name] = {"empty_code_rule_cases": len(rule_cases),
                        "excluding_rule_cases": agreement(without_rule),
                        "including_rule_cases": agreement(with_rule)}
        r = report[name]["excluding_rule_cases"]
        print(f"{name}: n={r['n']} agreement={r['agreement']:.4f} kappa={r['cohen_kappa']:.4f} "
              f"(rule cases excluded: {len(rule_cases)})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
