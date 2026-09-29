"""Step 3 of the independent-audit pilot: select the known-answer set.

Keeps prompts whose clean variant passes every test, whose cosmetic variant
does not, and which have at least one mutant the tests detect. For each kept
prompt the most subtle detected mutant (highest harness pass rate below 1.0)
is the bug case. Prompts are allocated round-robin across display bins.

See docs/independent_audit_protocol.md.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260928
GROUND_TRUTH = {"clean": "correct", "cosmetic": "correct", "bug": "incorrect"}
ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, default=ROOT / "data" / "independent_audit" / "pilot")
    ap.add_argument("--n-prompts", type=int, default=150)
    args = ap.parse_args()

    inputs = {r["prompt_id"]: r for r in load_jsonl(args.pilot_dir / "pilot_inputs.jsonl")}
    variants = load_jsonl(args.pilot_dir / "variants_executed.jsonl")
    by_prompt: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for v in variants:
        by_prompt[v["prompt_id"]][v["category"]].append(v)

    chosen: dict[str, dict] = {}
    for pid, cats in by_prompt.items():
        clean, cosmetic = cats.get("clean", []), cats.get("cosmetic", [])
        if not clean or clean[0]["harness_pass_rate"] != 1.0:
            continue
        if not cosmetic or cosmetic[0]["harness_pass_rate"] >= 1.0:
            continue
        killed = [m for m in cats.get("bug", []) if m["harness_pass_rate"] < 1.0]
        if not killed:
            continue
        bug = sorted(killed, key=lambda m: (-m["harness_pass_rate"], m["variant_id"]))[0]
        chosen[pid] = {"clean": clean[0], "cosmetic": cosmetic[0], "bug": bug}

    rng = random.Random(SEED)
    by_bin: dict[int, list[str]] = defaultdict(list)
    for pid in sorted(chosen):
        by_bin[inputs[pid]["display_bin"]].append(pid)
    for ids in by_bin.values():
        rng.shuffle(ids)
    selected: list[str] = []
    while len(selected) < args.n_prompts and any(by_bin.values()):
        for b in sorted(by_bin):
            if by_bin[b] and len(selected) < args.n_prompts:
                selected.append(by_bin[b].pop())
    assert len(selected) == args.n_prompts, f"only {len(selected)} eligible prompts"

    cases = []
    for pid in selected:
        p = inputs[pid]
        for category, v in chosen[pid].items():
            cases.append({
                "case_id": v["variant_id"],
                "prompt_id": pid,
                "category": category,
                "ground_truth": GROUND_TRUTH[category],
                "mutation": v["mutation"],
                "task": p["task"],
                "code": v["code"],
                "unit_tests": p["unit_tests"],
                "harness_pass_rate": v["harness_pass_rate"],
                "composite": p["composite"],
                "display_bin": p["display_bin"],
                "frame": p["frame"],
            })
    rng.shuffle(cases)

    out = args.pilot_dir / "known_answer_set.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    bugs = [c for c in cases if c["category"] == "bug"]
    print(f"eligible prompts: {len(chosen)}; selected {len(selected)}; cases {len(cases)} -> {out}")
    print("selected per display bin:", dict(sorted(Counter(inputs[p]["display_bin"] for p in selected).items())))
    print("selected per frame:", dict(Counter(inputs[p]["frame"] for p in selected)))
    print("bug mutation kinds:", dict(Counter(c["mutation"]["kind"] for c in bugs)))
    partial = sum(1 for c in bugs if c["harness_pass_rate"] > 0)
    print(f"bug harness pass rate: {partial} of {len(bugs)} partially pass (>0); "
          f"mean {sum(c['harness_pass_rate'] for c in bugs) / len(bugs):.3f}")
    print("cosmetic harness pass rates:", dict(Counter(round(c["harness_pass_rate"], 2)
                                                       for c in cases if c["category"] == "cosmetic")))


if __name__ == "__main__":
    main()
