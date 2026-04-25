"""
Generate solutions via the Gemini CLI (gemini -p).

Uses the Gemini CLI's own auth instead of the Python SDK.
Outputs to the same JSONL format as 02_generate_solutions.py.

Usage:
    python scripts/generate_via_gemini_cli.py [--model gemini-3-flash]
"""
import json
import os
import subprocess
import time
import re
from datetime import datetime, timezone

GEMINI_CMD = r"C:\Users\herna\AppData\Roaming\npm\gemini.cmd"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_PATH = os.path.join(BASE, "data", "experiment_prompts.jsonl")
GENERATIONS_DIR = os.path.join(BASE, "data", "generations")

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a coding problem, write a "
    "complete Python solution. Output ONLY the Python code inside a single "
    "```python``` code block. Do not include any explanation, tests, or "
    "examples outside the code block."
)

RPM = 10
MODEL_ID = "google/gemini-3-flash-preview"


def clean_code_from_response(text):
    if not text:
        return ""
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    if "def " in text:
        return text.strip()
    return text.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash-preview-04-17",
                        help="Gemini CLI model name (default: gemini-2.5-flash-preview-04-17)")
    parser.add_argument("--model-id", default=MODEL_ID,
                        help="Model ID for output records")
    parser.add_argument("--rpm", type=int, default=RPM)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retries per prompt before skipping (no record written)")
    parser.add_argument("--retry-backoff", type=float, default=5.0,
                        help="Base seconds for exponential backoff between retries")
    args = parser.parse_args()

    api_model = args.model
    model_id = args.model_id
    min_interval = 60.0 / args.rpm if args.rpm > 0 else 0

    # Load prompts
    prompts = []
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts")

    # Output file
    safe_name = model_id.replace("/", "_").replace(" ", "_").replace(":", "_")
    outpath = os.path.join(GENERATIONS_DIR, f"{safe_name}.jsonl")
    os.makedirs(GENERATIONS_DIR, exist_ok=True)

    # Resume: load done IDs
    done_ids = set()
    if os.path.exists(outpath):
        with open(outpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = rec.get("code_cleaned") or ""
                if code.strip() and rec.get("error") is None:
                    done_ids.add(rec.get("prompt_id"))

    remaining = [p for p in prompts if p["prompt_id"] not in done_ids]
    print(f"Resuming: {len(done_ids)} done, {len(remaining)} remaining")

    if not remaining:
        print("All prompts complete!")
        return

    success = 0
    errors = 0
    skipped = []  # prompt_ids that failed all retries
    start = time.time()

    with open(outpath, "a", encoding="utf-8") as f_out:
        for i, prompt_rec in enumerate(remaining):
            t0 = time.time()

            full_prompt = SYSTEM_PROMPT + "\n\n" + prompt_rec["input"]
            raw_response = None
            code_cleaned = None
            error_msg = None

            for attempt in range(1, args.max_retries + 1):
                raw_response = None
                error_msg = None
                try:
                    result = subprocess.run(
                        [GEMINI_CMD, "-p", full_prompt, "--model", api_model, "--yolo", "--approval-mode", "yolo"],
                        capture_output=True, text=True, encoding="utf-8",
                        timeout=args.timeout,
                    )
                    if result.returncode == 0:
                        raw_response = result.stdout
                    else:
                        error_msg = f"Exit code {result.returncode}: {result.stderr.strip()[:500]}"
                except subprocess.TimeoutExpired:
                    error_msg = f"TIMEOUT: {args.timeout}s"
                except Exception as e:
                    error_msg = str(e)

                if error_msg is None:
                    code_cleaned = clean_code_from_response(raw_response or "")
                    if code_cleaned and code_cleaned.strip():
                        break
                    error_msg = "EMPTY_CODE: no extractable code block"
                    code_cleaned = None

                if attempt < args.max_retries:
                    backoff = args.retry_backoff * (2 ** (attempt - 1))
                    print(f"    retry {attempt}/{args.max_retries - 1} for {prompt_rec['prompt_id'][:8]}: "
                          f"{error_msg[:80]} | sleep {backoff:.1f}s")
                    time.sleep(backoff)

            if error_msg is None and code_cleaned and code_cleaned.strip():
                record = {
                    "prompt_id": prompt_rec["prompt_id"],
                    "model_id": model_id,
                    "raw_response": raw_response,
                    "code_cleaned": code_cleaned,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": None,
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()
                success += 1
            else:
                # All retries exhausted ,  do NOT write a record. Next run will retry.
                errors += 1
                skipped.append(prompt_rec["prompt_id"])
                print(f"    SKIP {prompt_rec['prompt_id'][:8]} after {args.max_retries} attempts: {error_msg[:80]}")

            total = success + errors
            elapsed = time.time() - start
            rate = total / elapsed if elapsed > 0 else 0
            eta_s = (len(remaining) - total) / rate if rate > 0 else 0
            if eta_s < 60:
                eta_str = f"{eta_s:.0f}s"
            elif eta_s < 3600:
                eta_str = f"{eta_s/60:.1f}m"
            else:
                eta_str = f"{eta_s/3600:.1f}h"

            print(f"  [{total}/{len(remaining)}] {success} ok, {errors} err | "
                  f"{rate:.1f}/s | ETA {eta_str}")

            # Rate limit (skip if CLI overhead already exceeds interval)
            if min_interval > 0:
                dt = time.time() - t0
                if dt < min_interval:
                    time.sleep(min_interval - dt)

    print(f"\nDone: {success} success, {errors} errors (skipped; not written)")
    if skipped:
        print(f"Skipped prompt_ids (will retry on next run): {len(skipped)}")
        skip_log = outpath + ".skipped.txt"
        with open(skip_log, "w", encoding="utf-8") as f:
            for pid in skipped:
                f.write(pid + "\n")
        print(f"  logged to {skip_log}")


if __name__ == "__main__":
    main()
