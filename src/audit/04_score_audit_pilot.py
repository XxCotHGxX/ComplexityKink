"""Step 4 of the independent-audit pilot: score auditors and apply the rule.

Implements the selection rule fixed in docs/independent_audit_protocol.md.
Unparseable responses and API errors count as not matching the ground truth
in every accuracy metric, and are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULE = {"max_error_rate": 0.02, "min_clean_accuracy": 0.95, "min_cosmetic_rescue": 0.90}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(cases: dict[str, dict], audits: list[dict]) -> dict:
    latest = {}
    for a in audits:
        if a["case_id"] in cases:
            latest[a["case_id"]] = a
    missing = sorted(set(cases) - set(latest))
    rows = []
    for cid, case in cases.items():
        a = latest.get(cid, {"status": "missing"})
        rows.append((case["category"], case["ground_truth"], a.get("verdict") if a["status"] == "ok" else None, a["status"]))

    def rate(pred, subset):
        return sum(1 for r in subset if pred(r)) / len(subset) if subset else float("nan")

    by_cat = {c: [r for r in rows if r[0] == c] for c in ("clean", "cosmetic", "bug")}
    tokens = [a.get("usage", {}).get("completion_tokens") for a in latest.values() if a.get("usage")]
    latency = [a.get("latency_s") for a in latest.values() if a.get("latency_s") is not None]
    out = {
        "n_cases": len(rows),
        "missing": len(missing),
        "error_rate": rate(lambda r: r[3] != "ok", rows),
        "uncertain_rate": rate(lambda r: r[2] == "uncertain", rows),
        "clean_accuracy": rate(lambda r: r[2] == "correct", by_cat["clean"]),
        "cosmetic_rescue": rate(lambda r: r[2] == "correct", by_cat["cosmetic"]),
        "bug_wrong_rescue": rate(lambda r: r[2] == "correct", by_cat["bug"]),
        "bug_detected": rate(lambda r: r[2] == "incorrect", by_cat["bug"]),
        "overall_accuracy": rate(lambda r: r[2] == r[1], rows),
        "median_completion_tokens": sorted(tokens)[len(tokens) // 2] if tokens else None,
        "median_latency_s": sorted(latency)[len(latency) // 2] if latency else None,
    }
    out["eligible"] = (out["error_rate"] <= RULE["max_error_rate"]
                       and out["clean_accuracy"] >= RULE["min_clean_accuracy"]
                       and out["cosmetic_rescue"] >= RULE["min_cosmetic_rescue"]
                       and out["missing"] == 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, default=ROOT / "data" / "independent_audit" / "pilot")
    ap.add_argument("--auditors", nargs="+", default=["Phi-4-reasoning=audit_phi4_reasoning.jsonl",
                                                     "MAI-Thinking-1=audit_mai_thinking_1.jsonl"])
    ap.add_argument("--ga-open-weights", default="Phi-4-reasoning")
    args = ap.parse_args()

    cases = {c["case_id"]: c for c in load_jsonl(args.pilot_dir / "known_answer_set.jsonl")}
    results = {}
    for spec in args.auditors:
        name, fname = spec.split("=", 1)
        results[name] = score(cases, load_jsonl(args.pilot_dir / fname))

    eligible = [n for n, r in results.items() if r["eligible"]]
    primary = None
    if eligible:
        best = min(results[n]["bug_wrong_rescue"] for n in eligible)
        tied = [n for n in eligible if results[n]["bug_wrong_rescue"] - best <= 0.01]
        tied.sort(key=lambda n: (-results[n]["overall_accuracy"], n != args.ga_open_weights))
        primary = tied[0]
    decision = {"rule": RULE, "results": results, "eligible": eligible, "primary_auditor": primary}
    (args.pilot_dir / "pilot_decision.json").write_text(json.dumps(decision, indent=2))

    keys = ["error_rate", "uncertain_rate", "clean_accuracy", "cosmetic_rescue", "bug_wrong_rescue",
            "bug_detected", "overall_accuracy", "median_completion_tokens", "median_latency_s", "eligible"]
    print(f"{'metric':26s}" + "".join(f"{n:>18s}" for n in results))
    for k in keys:
        cells = []
        for n in results:
            v = results[n][k]
            cells.append(f"{v:>18.3f}" if isinstance(v, float) else f"{str(v):>18s}")
        print(f"{k:26s}" + "".join(cells))
    print("eligible:", eligible, "| primary auditor:", primary)


if __name__ == "__main__":
    main()
