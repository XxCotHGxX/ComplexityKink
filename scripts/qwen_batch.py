"""
Qwen Cloud / DashScope batch generation for Stage D prompts.

This submits the Stage D generation delta to Alibaba's OpenAI-compatible Batch
API and retrieves results into the same generation schema used by
src/data_provenance/02_generate_solutions.py.

USAGE:
  python scripts/qwen_batch.py submit
  python scripts/qwen_batch.py status
  python scripts/qwen_batch.py retrieve

Requires DASHSCOPE_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "data" / "stage_d" / "generation_delta" / "stage_d_new_prompts.jsonl"
DEFAULT_GEN_DIR = ROOT / "data" / "stage_d" / "generations"
DEFAULT_STATE_DIR = ROOT / "data" / "stage_d" / "batch_state"
DEFAULT_REQUEST_DIR = ROOT / "data" / "stage_d" / "batch_requests"
DEFAULT_RESULT_DIR = ROOT / "data" / "stage_d" / "batch_results"

MODEL_ID = "qwen/qwen3.6-plus"
API_MODEL = "qwen3.6-plus-2026-04-02"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
ENDPOINT = "/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a coding problem, write a "
    "complete Python solution. Output ONLY the Python code inside a single "
    "```python``` code block. Do not include any explanation, tests, or "
    "examples outside the code block."
)


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def client() -> OpenAI:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("Set DASHSCOPE_API_KEY in the environment.")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=1200.0)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_done_ids(gen_path: Path) -> set[str]:
    done: set[str] = set()
    if not gen_path.exists():
        return done
    with gen_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            if not (rec.get("code_cleaned") or "").strip():
                continue
            pid = rec.get("prompt_id") or rec.get("id")
            if pid:
                done.add(str(pid))
    return done


def state_path(state_dir: Path) -> Path:
    return state_dir / f"{safe_name(MODEL_ID)}.json"


def generation_path(gen_dir: Path) -> Path:
    return gen_dir / f"{safe_name(MODEL_ID)}.jsonl"


def user_content(prompt_rec: dict) -> str:
    return "/no_think\n" + prompt_rec["input"]


def clean_code(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    if "def " in text or "class " in text or "import " in text:
        return text.strip()
    return text.strip()


def parse_batch_response(line_obj: dict) -> tuple[str | None, str, str | None]:
    custom_id = line_obj.get("custom_id")
    response = line_obj.get("response") or {}
    status_code = response.get("status_code")
    body = response.get("body") or {}
    if status_code is not None and int(status_code) >= 400:
        return custom_id, "", json.dumps(body or response, ensure_ascii=False)
    choices = body.get("choices") or []
    raw = ""
    if choices:
        msg = choices[0].get("message") or {}
        raw = msg.get("content") or ""
    return custom_id, raw, None


def cmd_submit(args: argparse.Namespace) -> None:
    prompts_path = Path(args.prompts)
    gen_dir = Path(args.gen_dir)
    state_dir = Path(args.state_dir)
    request_dir = Path(args.request_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_jsonl(prompts_path)
    gen_path = generation_path(gen_dir)
    done = load_done_ids(gen_path)
    remaining = [row for row in prompts if str(row["prompt_id"]) not in done]
    print(f"Prompts: {len(prompts):,}  done: {len(done):,}  remaining: {len(remaining):,}")
    if not remaining:
        print("Nothing to submit.")
        return
    if len(remaining) > 50_000:
        raise SystemExit("Qwen batch supports up to 50,000 requests per file.")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = request_dir / f"{safe_name(MODEL_ID)}_{ts}.jsonl"
    with batch_file.open("w", encoding="utf-8") as f:
        for row in remaining:
            body = {
                "model": args.api_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content(row)},
                ],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "enable_thinking": False,
            }
            f.write(json.dumps({
                "custom_id": str(row["prompt_id"]),
                "method": "POST",
                "url": ENDPOINT,
                "body": body,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(remaining):,} requests to {batch_file}")

    qwen = client()
    print("Uploading batch file...")
    with batch_file.open("rb") as f:
        uploaded = qwen.files.create(file=f, purpose="batch")
    print(f"Uploaded: {uploaded.id}")

    print("Creating batch job...")
    batch = qwen.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=args.completion_window,
        metadata={
            "ds_name": f"ck-stage-d-{safe_name(MODEL_ID)}-{ts}",
            "ds_description": "ComplexityKink Stage D Qwen 3.6 Plus generation delta",
        },
    )
    print(f"Batch: {batch.id}  status={batch.status}")

    state = {
        "model_id": MODEL_ID,
        "api_model": args.api_model,
        "base_url": BASE_URL,
        "endpoint": ENDPOINT,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "batch_file_local": str(batch_file),
        "prompts": str(prompts_path),
        "generation_output": str(gen_path),
        "submitted_at": ts,
        "n_requests": len(remaining),
        "enable_thinking": False,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    out_state = state_path(state_dir)
    out_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"State: {out_state}")


def cmd_status(args: argparse.Namespace) -> None:
    path = state_path(Path(args.state_dir))
    if not path.exists():
        raise SystemExit(f"No state file at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    batch = client().batches.retrieve(state["batch_id"])
    print(f"Model:    {state['model_id']}")
    print(f"Batch:    {batch.id}")
    print(f"Status:   {batch.status}")
    errors = getattr(batch, "errors", None)
    if errors:
        print(f"Errors:   {errors}")
    counts = getattr(batch, "request_counts", None)
    if counts:
        print(
            "Requests: "
            f"total={counts.total} completed={counts.completed} failed={counts.failed}"
        )
    if getattr(batch, "output_file_id", None):
        print(f"Output:   {batch.output_file_id}")
    if getattr(batch, "error_file_id", None):
        print(f"Errors:   {batch.error_file_id}")
    if batch.status == "completed":
        print("Ready to retrieve.")
        sys.exit(0)
    if batch.status in {"failed", "expired", "cancelled"}:
        sys.exit(2)
    sys.exit(1)


def download_text(qwen: OpenAI, file_id: str, result_path: Path) -> str:
    content = qwen.files.content(file_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(content, "write_to_file"):
        content.write_to_file(result_path)
        return result_path.read_text(encoding="utf-8")
    raw = content.read()
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    result_path.write_text(text, encoding="utf-8")
    return text


def cmd_retrieve(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    result_dir = Path(args.result_dir)
    gen_dir = Path(args.gen_dir)
    path = state_path(state_dir)
    if not path.exists():
        raise SystemExit(f"No state file at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    qwen = client()
    batch = qwen.batches.retrieve(state["batch_id"])
    if batch.status != "completed":
        raise SystemExit(f"Batch is {batch.status}, not completed.")

    ts = dt.datetime.utcnow().isoformat()
    gen_path = generation_path(gen_dir)
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    errors = 0
    empty = 0

    with gen_path.open("a", encoding="utf-8") as out:
        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            print(f"Downloading {output_file_id}...")
            output_path = result_dir / f"{safe_name(MODEL_ID)}_{batch.id}_output.jsonl"
            text = download_text(qwen, output_file_id, output_path)
            for line in text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                pid, raw, err = parse_batch_response(obj)
                code = clean_code(raw)
                if err:
                    errors += 1
                elif not code.strip():
                    empty += 1
                    err = "EMPTY_CODE: batch response had no extractable code"
                else:
                    written += 1
                out.write(json.dumps({
                    "prompt_id": pid,
                    "model_id": MODEL_ID,
                    "raw_response": raw,
                    "code_cleaned": code if not err else None,
                    "timestamp": ts,
                    "error": err,
                }, ensure_ascii=False) + "\n")

        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            print(f"Downloading errors {error_file_id}...")
            error_path = result_dir / f"{safe_name(MODEL_ID)}_{batch.id}_errors.jsonl"
            text = download_text(qwen, error_file_id, error_path)
            for line in text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                pid = obj.get("custom_id")
                out.write(json.dumps({
                    "prompt_id": pid,
                    "model_id": MODEL_ID,
                    "raw_response": "",
                    "code_cleaned": None,
                    "timestamp": ts,
                    "error": json.dumps(obj, ensure_ascii=False),
                }, ensure_ascii=False) + "\n")
                errors += 1

    print(
        f"Appended to {gen_path}: written={written:,} "
        f"errors={errors:,} empty={empty:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--gen-dir", default=str(DEFAULT_GEN_DIR))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--request-dir", default=str(DEFAULT_REQUEST_DIR))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    sub = parser.add_subparsers(dest="cmd", required=True)

    submit = sub.add_parser("submit")
    submit.add_argument("--api-model", default=API_MODEL)
    submit.add_argument("--max-tokens", type=int, default=4096)
    submit.add_argument("--temperature", type=float, default=0.0)
    submit.add_argument("--completion-window", default="24h")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)

    retrieve = sub.add_parser("retrieve")
    retrieve.set_defaults(func=cmd_retrieve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
