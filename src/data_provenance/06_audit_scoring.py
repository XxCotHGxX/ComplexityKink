"""
Step 6: Audit the test-harness verdict with an independent judge (o4-mini).

The harness in 03_execute_and_score.py literally runs the generated code
against the unit tests. That catches anything that runs wrong, but it also
fails *semantically correct* code for cosmetic reasons:

  - Generated function uses a different name than the test asserts on
  - Generated code is script-style (reads input(), prints result) but tests
    call a function
  - Minor I/O or type-coercion differences

Here we ask o4-mini, given the task, the code, the test assertions, and the
harness's pass/fail verdict, whether the verdict is *correct* on the merits.
We flag rows where judge disagrees with harness. The judge is the SAME
external model we already use for rubric complexity, so the "no in-panel
model grades itself" principle is preserved.

Output schema (one row per scored row audited):
  {"id": ..., "model_id": ..., "harness_pass_rate": ...,
   "judge_verdict": "correct" | "incorrect" | "uncertain",
   "judge_reason": "...",
   "agrees_with_harness": bool}

USAGE:
  python src/data_provenance/06_audit_scoring.py \\
      --scored-dir data/scored --output-dir data/audits --workers 8
  # or audit a single model:
  python src/data_provenance/06_audit_scoring.py \\
      --only anthropic_claude-opus-4.6 --limit 500
"""
import os
import sys
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    import load_keys  # type: ignore
except ImportError:
    load_keys = None

JUDGE_MODEL = "o4-mini"
AZURE_ENDPOINT = "https://datapipeline0.cognitiveservices.azure.com/"
AZURE_API_VERSION = "2025-01-01-preview"

SYSTEM_PROMPT = """You are auditing an automated test harness that runs generated Python code against unit tests.

You will be given:
1. The coding task description
2. The generated code
3. The unit test assertions the harness ran
4. The harness verdict (pass rate 0.0 to 1.0)

Decide whether the code is a SEMANTICALLY CORRECT solution to the task, ignoring cosmetic harness artifacts:
  - Function name mismatch (code defines `foo`, test calls `bar`) to still CORRECT if logic is right
  - Script-style I/O (reads input(), prints result) when test calls a function to still CORRECT if logic is right
  - Minor output formatting differences (newlines, whitespace) to still CORRECT if logic is right
  - Uses a different but equivalent algorithm to CORRECT

The code is INCORRECT if:
  - The algorithm is wrong or produces wrong answers
  - It crashes on legitimate inputs
  - It misunderstands the task

Respond with ONLY a JSON object:
{"verdict": "correct" | "incorrect" | "uncertain", "reason": "<one short sentence>"}"""


def audit_row(client, row):
    task = row.get("input") or ""
    code = row.get("code_cleaned") or ""
    tests = row.get("unit_tests") or ""
    pr = row.get("pass_rate", 0.0)

    if not code.strip():
        return row["id"], {"verdict": "incorrect", "reason": "empty code"}, None

    user = (
        f"TASK:\n{task}\n\n"
        f"CODE:\n```python\n{code}\n```\n\n"
        f"UNIT TESTS:\n{tests}\n\n"
        f"HARNESS PASS RATE: {pr:.2f}"
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=4000,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        obj = json.loads(raw)
        v = obj.get("verdict")
        if v not in ("correct", "incorrect", "uncertain"):
            return row["id"], None, f"bad verdict: {v}"
        return row["id"], {"verdict": v, "reason": obj.get("reason", "")}, None
    except json.JSONDecodeError as e:
        return row["id"], None, f"json err: {e} | {raw[:200]}"
    except Exception as e:
        return row["id"], None, f"api err: {e}"


def load_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("judge_verdict"):
                done.add(r["id"])
    return done


def audit_file(client, scored_path, out_path, limit, workers):
    rows = []
    with open(scored_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]

    done = load_done(out_path)
    pending = [r for r in rows if r["id"] not in done]
    print(f"  {os.path.basename(scored_path)}: {len(rows)} rows, "
          f"{len(done)} audited, {len(pending)} remaining")
    if not pending:
        return 0, 0

    audited, errors = 0, 0
    start = time.time()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(audit_row, client, r): r for r in pending}
            for fut in as_completed(futures):
                row = futures[fut]
                rid, result, err = fut.result()
                if result:
                    pr = row.get("pass_rate", 0.0)
                    harness_pass = pr >= 0.5
                    judge_pass = result["verdict"] == "correct"
                    rec = {
                        "id": rid,
                        "model_id": row.get("model_id", "unknown"),
                        "harness_pass_rate": pr,
                        "judge_verdict": result["verdict"],
                        "judge_reason": result["reason"],
                        "agrees_with_harness": harness_pass == judge_pass,
                    }
                    f_out.write(json.dumps(rec) + "\n")
                    f_out.flush()
                    audited += 1
                else:
                    rec = {"id": rid, "judge_verdict": None, "error": err}
                    f_out.write(json.dumps(rec) + "\n")
                    f_out.flush()
                    errors += 1
                total = audited + errors
                if total % 100 == 0:
                    rate = total / (time.time() - start) * 60
                    eta = (len(pending) - total) / rate if rate > 0 else 0
                    print(f"    [{total}/{len(pending)}] ok={audited} err={errors} "
                          f"{rate:.0f}/min ETA {eta:.1f}min")

    return audited, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored-dir", default="data/scored")
    ap.add_argument("--output-dir", default="data/audits")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only audit these model basenames (without .jsonl)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Audit first N rows per file (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--api", choices=["azure", "openai"], default="azure",
                    help="Which API endpoint to use")
    args = ap.parse_args()

    if load_keys is not None:
        load_keys.load()
    if args.api == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )

    files = sorted(f for f in os.listdir(args.scored_dir)
                   if f.endswith(".jsonl") and not f.startswith("_"))
    if args.only:
        files = [f for f in files if f.replace(".jsonl", "") in args.only]

    print(f"Auditing {len(files)} files with {JUDGE_MODEL}")
    os.makedirs(args.output_dir, exist_ok=True)

    totals = []
    for fn in files:
        scored_path = os.path.join(args.scored_dir, fn)
        out_path = os.path.join(args.output_dir, fn)
        print(f"\n== {fn}")
        audited, errors = audit_file(client, scored_path, out_path,
                                     args.limit, args.workers)
        totals.append((fn, audited, errors))

    print("\n" + "=" * 60)
    for fn, a, e in totals:
        print(f"  {fn:50s} audited={a} errors={e}")


if __name__ == "__main__":
    main()
