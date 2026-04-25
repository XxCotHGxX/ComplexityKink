"""
Step 7: Apply judge verdicts to produce corrected pass rates.

For each row in data/scored/<model>.jsonl, look up the matching audit row
in data/audits/<model>.jsonl. If the judge said "correct", the corrected
pass_rate is 1.0. If "incorrect", it is 0.0. If "uncertain" or the row was
not audited (error), we fall back to the harness pass_rate.

Output is written to data/scored_corrected/<model>.jsonl with the same
schema as data/scored/, but with pass_rate replaced and a new field
judge_verdict recorded for provenance.

USAGE:
  python src/data_provenance/07_apply_judge.py
"""
import os
import json
import argparse


def load_audit(path):
    v = {}
    if not os.path.exists(path):
        return v
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("judge_verdict") in ("correct", "incorrect", "uncertain"):
                v[r["id"]] = r["judge_verdict"]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-dir", default="data/scored")
    ap.add_argument("--audit-dir", default="data/audits")
    ap.add_argument("--output-dir", default="data/scored_corrected")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fn in sorted(os.listdir(args.scored_dir)):
        if not fn.endswith(".jsonl") or fn.startswith("_") or fn.endswith(".bak"):
            continue
        scored_path = os.path.join(args.scored_dir, fn)
        audit_path = os.path.join(args.audit_dir, fn)
        out_path = os.path.join(args.output_dir, fn)

        verdicts = load_audit(audit_path)
        n_rows = 0
        n_judged = 0
        n_flipped = 0
        with open(scored_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_rows += 1
                v = verdicts.get(r.get("id"))
                orig_pr = r.get("pass_rate", 0.0)
                if v == "correct":
                    new_pr = 1.0
                elif v == "incorrect":
                    new_pr = 0.0
                else:
                    new_pr = orig_pr
                if v in ("correct", "incorrect"):
                    n_judged += 1
                if (orig_pr < 0.5) != (new_pr < 0.5):
                    n_flipped += 1
                r["harness_pass_rate"] = orig_pr
                r["pass_rate"] = new_pr
                r["judge_verdict"] = v
                fout.write(json.dumps(r) + "\n")

        print(f"  {fn:50s} rows={n_rows} judged={n_judged} flipped={n_flipped}")


if __name__ == "__main__":
    main()
