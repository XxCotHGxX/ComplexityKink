"""
OpenAI batch generation for Complexity Kink prompts (GPT-5.4 etc.).

OpenAI's batch API runs at 50% of sync pricing with up-to-24h turnaround.
Three phases, same shape as scripts/gemini_batch.py:

  submit    Builds a JSONL of /v1/chat/completions requests for the
            remaining prompts, uploads it, creates a batch, saves state.
  status    Polls the batch; exits 0 on completed, non-zero otherwise.
  retrieve  Downloads output_file_id + error_file_id, parses each row,
            appends to data/generations/<model>.jsonl in the standard
            schema.

USAGE:
  python scripts/openai_batch.py submit \\
      --model openai/gpt-5.4 --api-model gpt-5.4
  python scripts/openai_batch.py status   --model openai/gpt-5.4
  python scripts/openai_batch.py retrieve --model openai/gpt-5.4

  # Stage D:
  python scripts/openai_batch.py \\
      --prompts data/stage_d/generation_delta/stage_d_new_prompts.jsonl \\
      --gen-dir data/stage_d/generations \\
      --state-dir data/stage_d/batch_state \\
      --request-dir data/stage_d/batch_requests \\
      submit --model openai/gpt-5.4 --api-model gpt-5.4 --reasoning

Requires OPENAI_API_KEY in the environment.
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

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
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in env")
    return OpenAI(api_key=key)


def _load_prompts(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def _load_done(gen_path):
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


def _safe_name(model_id):
    return model_id.replace("/", "_").replace(":", "_")


def _state_path(model_id, state_dir):
    return state_dir / f"{_safe_name(model_id)}.json"


def _gen_path(model_id, gen_dir):
    return gen_dir / f"{_safe_name(model_id)}.jsonl"


def _user_content(p):
    return f"Task:\n{p['input']}\n\nWrite a complete Python solution."


# -- submit ---------------------------------------------------------

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

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_name(args.model)
    batch_file = request_dir / f"{slug}_{ts}.jsonl"

    with open(batch_file, "w", encoding="utf-8") as f:
        for p in remaining:
            body = {
                "model": args.api_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_content(p)},
                ],
            }
            # Temperature: reasoning models (o-series, gpt-5) may only
            # accept default temperature; skip the field in that case.
            if args.reasoning:
                body["max_completion_tokens"] = args.max_tokens
            else:
                body["max_tokens"] = args.max_tokens
                body["temperature"] = 0.2
            f.write(json.dumps({
                "custom_id": str(p["prompt_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }) + "\n")
    print(f"Wrote {len(remaining)} requests to {batch_file}")

    client = _client()
    print("Uploading batch file...")
    uploaded = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    print(f"Uploaded: {uploaded.id}")

    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"display_name": f"ck-{slug}-{ts}"},
    )
    print(f"Batch: {batch.id}  status={batch.status}")

    state_path = _state_path(args.model, state_dir)
    state_path.write_text(json.dumps({
        "model_id": args.model,
        "api_model": args.api_model,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "batch_file_local": str(batch_file),
        "prompts": str(prompts_path),
        "generation_output": str(gen_path),
        "submitted_at": ts,
        "n_requests": len(remaining),
    }, indent=2), encoding="utf-8")
    print(f"State: {state_path}")


# -- status ---------------------------------------------------------

def cmd_status(args):
    state_path = _state_path(args.model, Path(args.state_dir))
    if not state_path.exists():
        raise SystemExit(f"No state file at {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    client = _client()
    batch = client.batches.retrieve(state["batch_id"])
    print(f"Model:    {state['model_id']}")
    print(f"Batch:    {batch.id}")
    print(f"Status:   {batch.status}")
    rc = batch.request_counts
    if rc is not None:
        print(f"Requests: total={rc.total} completed={rc.completed} failed={rc.failed}")
    if batch.output_file_id:
        print(f"Output:   {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Errors:   {batch.error_file_id}")

    if batch.status == "completed":
        print("Ready to retrieve.")
        sys.exit(0)
    elif batch.status in ("failed", "cancelled", "expired"):
        sys.exit(2)
    else:
        sys.exit(1)


# -- retrieve -------------------------------------------------------

def _clean_code(text):
    if not text:
        return ""
    s = text.strip()
    if "```" in s:
        for chunk in s.split("```")[1::2]:
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
    batch = client.batches.retrieve(state["batch_id"])
    if batch.status != "completed":
        raise SystemExit(f"Batch is {batch.status}, not completed.")

    gen_path = _gen_path(args.model, Path(args.gen_dir))
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().isoformat()
    written, errors, empty = 0, 0, 0

    with open(gen_path, "a", encoding="utf-8") as f_out:
        if batch.output_file_id:
            print(f"Downloading {batch.output_file_id}...")
            content = client.files.content(batch.output_file_id).read()
            text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else content
            result_dir = Path(args.result_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / f"{_safe_name(args.model)}_{batch.id}_output.jsonl").write_text(
                text,
                encoding="utf-8",
            )
            for line in text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id")
                try:
                    pid = int(cid) if cid and str(cid).isdigit() else cid
                except Exception:
                    pid = cid
                resp = obj.get("response") or {}
                body = resp.get("body") or {}
                choices = body.get("choices") or []
                raw = ""
                if choices:
                    msg = choices[0].get("message") or {}
                    raw = msg.get("content") or ""
                code = _clean_code(raw)
                if not code.strip():
                    empty += 1
                f_out.write(json.dumps({
                    "prompt_id": pid,
                    "model_id": args.model,
                    "raw_response": raw,
                    "code_cleaned": code,
                    "timestamp": ts,
                    "error": None,
                }) + "\n")
                written += 1

        if batch.error_file_id:
            print(f"Downloading errors {batch.error_file_id}...")
            content = client.files.content(batch.error_file_id).read()
            text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else content
            result_dir = Path(args.result_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / f"{_safe_name(args.model)}_{batch.id}_errors.jsonl").write_text(
                text,
                encoding="utf-8",
            )
            for line in text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id")
                try:
                    pid = int(cid) if cid and str(cid).isdigit() else cid
                except Exception:
                    pid = cid
                err = obj.get("error") or obj.get("response") or {}
                f_out.write(json.dumps({
                    "prompt_id": pid,
                    "model_id": args.model,
                    "raw_response": "",
                    "code_cleaned": "",
                    "timestamp": ts,
                    "error": json.dumps(err),
                }) + "\n")
                errors += 1

    print(f"Appended to {gen_path}: written={written} errors={errors} empty={empty}")


# -- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--gen-dir", default=str(GEN_DIR))
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--request-dir", default=str(REQUEST_DIR))
    ap.add_argument("--result-dir", default=str(RESULT_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--model", required=True,
                   help="model_id for naming (e.g. openai/gpt-5.4)")
    s.add_argument("--api-model", required=True,
                   help="OpenAI model name (e.g. gpt-5.4)")
    s.add_argument("--max-tokens", type=int, default=4096)
    s.add_argument("--reasoning", action="store_true",
                   help="Set for o-series / gpt-5.x models that reject temperature")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("status")
    s.add_argument("--model", required=True)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("retrieve")
    s.add_argument("--model", required=True)
    s.set_defaults(func=cmd_retrieve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
