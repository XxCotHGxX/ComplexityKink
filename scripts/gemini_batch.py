"""
Gemini batch generation for Complexity Kink prompts.

Google's batch API runs requests at 50% of sync pricing with up-to-24h
turnaround. We split work into three phases so they can happen in
separate invocations (the batch may take hours):

  submit   Read experiment_prompts.jsonl + any already-completed
           generations, build a JSONL batch file with the leftover
           prompts, upload it, create a batch job, write the job name
           to data/batch_state/<model>.json.
  status   Query the job state. Prints progress; exits 0 when
           SUCCEEDED, non-zero while running.
  retrieve When SUCCEEDED: download result file (or read inlined
           responses), parse each response, append one row per prompt
           to data/generations/<model>.jsonl in the same schema as
           02_generate_solutions.py writes.

USAGE:
  python scripts/gemini_batch.py submit \\
      --model google/gemini-3-flash-preview \\
      --api-model gemini-3-flash
  python scripts/gemini_batch.py status \\
      --model google/gemini-3-flash-preview
  python scripts/gemini_batch.py retrieve \\
      --model google/gemini-3-flash-preview

Requires GOOGLE_API_KEY in the environment (or GEMINI_API_KEY).
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "experiment_prompts.jsonl"
GEN_DIR = ROOT / "data" / "generations"
STATE_DIR = ROOT / "data" / "batch_state"
REQUEST_DIR = ROOT / "data" / "batch_requests"
RESULT_DIR = ROOT / "data" / "batch_results"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write a single, complete Python "
    "solution that passes all unit tests. Respond with ONLY the Python code "
    "inside a ```python code block, no explanations before or after."
)


def _client():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in env")
    return genai.Client(api_key=key)


def _safe_name(model_id):
    return model_id.replace("/", "_").replace(":", "_").replace(" ", "_")


def _load_prompts(path):
    prompts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompts[r["prompt_id"]] = r
    return prompts


def _load_done(gen_path):
    """Return prompt_ids already generated (and not blank/error)."""
    done = set()
    if not gen_path.exists():
        return done
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            if not (r.get("code_cleaned") or "").strip():
                continue
            pid = r.get("prompt_id")
            if pid is not None:
                done.add(pid)
    return done


def _state_path(model_id, state_dir):
    return state_dir / f"{_safe_name(model_id)}.json"


def _gen_path(model_id, gen_dir):
    return gen_dir / f"{_safe_name(model_id)}.jsonl"


def _user_content(prompt_rec):
    return (
        f"Task:\n{prompt_rec['input']}\n\n"
        "Write a complete Python solution."
    )


# ── submit ─────────────────────────────────────────────────────────

def cmd_submit(args):
    prompts_path = Path(args.prompts)
    gen_dir = Path(args.gen_dir)
    state_dir = Path(args.state_dir)
    request_dir = Path(args.request_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(prompts_path)
    gen_path = _gen_path(args.model, gen_dir)
    done = _load_done(gen_path)
    remaining = [p for pid, p in prompts.items() if pid not in done]
    print(f"Prompts: {len(prompts)}  done: {len(done)}  remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    # Build a JSONL batch file. Each line is a GenerateContentRequest with
    # a stable `key` so we can map responses back to prompt_ids.
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_name(args.model)
    batch_file = request_dir / f"{slug}_{ts}.jsonl"

    with open(batch_file, "w", encoding="utf-8") as f:
        for p in remaining:
            req = {
                "key": str(p["prompt_id"]),
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": _user_content(p)}]}
                    ],
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "generation_config": {
                        "temperature": 0.2,
                        "max_output_tokens": args.max_tokens,
                    },
                },
            }
            f.write(json.dumps(req) + "\n")
    print(f"Wrote {len(remaining)} requests to {batch_file}")

    client = _client()
    print("Uploading batch file...")
    uploaded = client.files.upload(
        file=str(batch_file),
        config=types.UploadFileConfig(
            display_name=f"ck-{slug}-{ts}",
            mime_type="jsonl",
        ),
    )
    print(f"Uploaded: {uploaded.name}")

    print("Creating batch job...")
    batch = client.batches.create(
        model=args.api_model,
        src=uploaded.name,
        config=types.CreateBatchJobConfig(display_name=f"ck-{slug}-{ts}"),
    )
    print(f"Batch job: {batch.name}  state={batch.state}")

    state_path = _state_path(args.model, state_dir)
    state = {
        "model_id": args.model,
        "api_model": args.api_model,
        "batch_name": batch.name,
        "uploaded_file": uploaded.name,
        "batch_file_local": str(batch_file),
        "prompts": str(prompts_path),
        "generation_output": str(gen_path),
        "submitted_at": ts,
        "n_requests": len(remaining),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"State: {state_path}")


# ── status ─────────────────────────────────────────────────────────

def cmd_status(args):
    state_path = _state_path(args.model, Path(args.state_dir))
    if not state_path.exists():
        raise SystemExit(f"No state file at {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    client = _client()
    batch = client.batches.get(name=state["batch_name"])
    print(f"Model:       {state['model_id']}")
    print(f"Batch:       {batch.name}")
    print(f"State:       {batch.state}")
    if hasattr(batch, "create_time") and batch.create_time:
        print(f"Created:     {batch.create_time}")
    if hasattr(batch, "end_time") and batch.end_time:
        print(f"Finished:    {batch.end_time}")
    if hasattr(batch, "error") and batch.error:
        print(f"Error:       {batch.error}")

    state_str = str(batch.state)
    if "SUCCEEDED" in state_str:
        print("Ready to retrieve.")
        sys.exit(0)
    elif "FAILED" in state_str or "CANCELLED" in state_str or "EXPIRED" in state_str:
        sys.exit(2)
    else:
        sys.exit(1)  # still running


# ── retrieve ───────────────────────────────────────────────────────

def _extract_code(response_obj):
    """Pull text out of a GenerateContentResponse-shaped dict/object."""
    try:
        if hasattr(response_obj, "candidates"):
            cands = response_obj.candidates
        elif isinstance(response_obj, dict):
            cands = response_obj.get("candidates") or []
        else:
            return ""
        if not cands:
            return ""
        c0 = cands[0]
        if hasattr(c0, "content"):
            content = c0.content
        else:
            content = c0.get("content", {})
        parts = getattr(content, "parts", None) or content.get("parts", [])
        texts = []
        for p in parts:
            t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
            if t:
                texts.append(t)
        return "\n".join(texts)
    except Exception:
        return ""


def _clean_code(text):
    """Strip ```python fences; mirror clean_code_from_response in 02_generate_solutions.py."""
    if not text:
        return ""
    s = text.strip()
    if "```" in s:
        chunks = s.split("```")
        # pick the first fenced chunk; drop optional 'python' language tag
        for chunk in chunks[1::2]:
            body = chunk
            if body.lower().startswith("python"):
                body = body.split("\n", 1)[1] if "\n" in body else ""
            body = body.strip()
            if body:
                return body
    return s


def cmd_retrieve(args):
    state_path = _state_path(args.model, Path(args.state_dir))
    if not state_path.exists():
        raise SystemExit(f"No state file at {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    client = _client()
    batch = client.batches.get(name=state["batch_name"])
    state_str = str(batch.state)
    if "SUCCEEDED" not in state_str:
        raise SystemExit(f"Batch is {batch.state}, not SUCCEEDED. Run status.")

    # Results may be inline or in a file.
    rows = []
    dest = batch.dest
    if dest is None:
        raise SystemExit("Batch has no dest; cannot retrieve.")

    if getattr(dest, "inlined_responses", None):
        for item in dest.inlined_responses:
            key = getattr(item, "key", None)
            resp = getattr(item, "response", None)
            err = getattr(item, "error", None)
            rows.append((key, resp, err))
    elif getattr(dest, "file_name", None):
        print(f"Downloading result file {dest.file_name}...")
        raw = client.files.download(file=dest.file_name)
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append((obj.get("key"), obj.get("response"), obj.get("error")))
    else:
        raise SystemExit("Unknown batch dest shape.")

    print(f"Retrieved {len(rows)} responses.")

    gen_path = _gen_path(args.model, Path(args.gen_dir))
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().isoformat()
    seen = _load_done(gen_path)
    written, errors, empty, skipped = 0, 0, 0, 0
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = result_dir / f"{_safe_name(args.model)}_{state['batch_name'].replace('/', '_')}_results.jsonl"
    with open(raw_results_path, "w", encoding="utf-8") as raw_out:
        for key, resp, err in rows:
            raw_out.write(json.dumps({
                "key": key,
                "response": resp,
                "error": str(err) if err else None,
            }, default=str) + "\n")

    with open(gen_path, "a", encoding="utf-8") as f:
        for key, resp, err in rows:
            try:
                pid = int(key) if key and str(key).isdigit() else key
            except Exception:
                pid = key
            if str(pid) in {str(x) for x in seen}:
                skipped += 1
                continue
            if err:
                f.write(json.dumps({
                    "prompt_id": pid,
                    "model_id": args.model,
                    "raw_response": "",
                    "code_cleaned": "",
                    "timestamp": ts,
                    "error": str(err),
                }) + "\n")
                errors += 1
                continue
            raw = _extract_code(resp)
            code = _clean_code(raw)
            if not code.strip():
                empty += 1
            f.write(json.dumps({
                "prompt_id": pid,
                "model_id": args.model,
                "raw_response": raw,
                "code_cleaned": code,
                "timestamp": ts,
                "error": None,
            }) + "\n")
            written += 1

    state["retrieved_at"] = ts
    state["raw_results_path"] = str(raw_results_path)
    state["retrieve_counts"] = {
        "written": written,
        "errors": errors,
        "empty": empty,
        "skipped_existing": skipped,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Raw results: {raw_results_path}")
    print(
        f"Appended to {gen_path}: written={written} "
        f"errors={errors} empty={empty} skipped_existing={skipped}"
    )


# ── main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--gen-dir", default=str(GEN_DIR))
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--request-dir", default=str(REQUEST_DIR))
    ap.add_argument("--result-dir", default=str(RESULT_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="Build batch file and submit")
    s.add_argument("--model", required=True,
                   help="model_id used to name gen file (e.g. google/gemini-3-flash-preview)")
    s.add_argument("--api-model", required=True,
                   help="Gemini model name (e.g. gemini-3-flash)")
    s.add_argument("--max-tokens", type=int, default=2048)
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("status", help="Check batch job state")
    s.add_argument("--model", required=True)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("retrieve", help="Download results, append to generations")
    s.add_argument("--model", required=True)
    s.set_defaults(func=cmd_retrieve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
