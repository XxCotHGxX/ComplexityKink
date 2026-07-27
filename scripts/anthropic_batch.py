"""
Anthropic Message Batches for Stage D generation.

Submits one Anthropic model at a time and retrieves results into the same
generation schema used by src/data_provenance/02_generate_solutions.py.

USAGE:
  python scripts/anthropic_batch.py submit --model anthropic/claude-sonnet-4.6
  python scripts/anthropic_batch.py status --model anthropic/claude-sonnet-4.6
  python scripts/anthropic_batch.py retrieve --model anthropic/claude-sonnet-4.6

Requires ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import anthropic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "src" / "stage_d" / "models_stage_d_panel.json"
DEFAULT_GEN_DIR = ROOT / "data" / "stage_d" / "generations"
DEFAULT_STATE_DIR = ROOT / "data" / "stage_d" / "batch_state"
DEFAULT_REQUEST_DIR = ROOT / "data" / "stage_d" / "batch_requests"
DEFAULT_RESULT_DIR = ROOT / "data" / "stage_d" / "batch_results"
DEFAULT_QUEUE_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.7",
]

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a coding problem, write a "
    "complete Python solution. Output ONLY the Python code inside a single "
    "```python``` code block. Do not include any explanation, tests, or "
    "examples outside the code block."
)


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_").replace(" ", "_")


def client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY in the environment.")
    return anthropic.Anthropic(api_key=key, timeout=1200.0)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model_config(models_path: Path, model_id: str) -> dict[str, Any]:
    raw = json.loads(models_path.read_text(encoding="utf-8"))
    models = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    matches = [m for m in models if m.get("model_id") == model_id]
    if not matches:
        raise SystemExit(f"Model {model_id!r} not found in {models_path}")
    config = matches[0]
    if config.get("backend") != "anthropic":
        raise SystemExit(f"Model {model_id!r} is backend={config.get('backend')!r}, not anthropic.")
    return config


def default_prompts_path(model_id: str, prompt_dir: Path | None = None) -> Path:
    if prompt_dir is None:
        prompt_dir = ROOT / "data" / "stage_d" / "generation_delta" / "per_model_missing"
    return prompt_dir / f"{safe_name(model_id)}.jsonl"


def state_path(model_id: str, state_dir: Path) -> Path:
    return state_dir / f"{safe_name(model_id)}.json"


def generation_path(model_id: str, gen_dir: Path) -> Path:
    return gen_dir / f"{safe_name(model_id)}.jsonl"


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
            pid = rec.get("prompt_id")
            if pid is not None:
                done.add(str(pid))
    return done


def load_seen_ids(gen_path: Path) -> set[str]:
    seen: set[str] = set()
    if not gen_path.exists():
        return seen
    with gen_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("prompt_id")
            if pid is not None:
                seen.add(str(pid))
    return seen


def validate_prompt_ids(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        pid = str(row.get("prompt_id", ""))
        if not pid:
            raise SystemExit("Prompt row missing prompt_id.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pid):
            raise SystemExit(
                f"Anthropic custom_id must be 1-64 chars of letters, digits, _ or -; got {pid!r}."
            )
        if pid in seen:
            raise SystemExit(f"Duplicate prompt_id in batch input: {pid}")
        seen.add(pid)


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


def result_to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


def message_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if isinstance(block, dict):
            block_type = block.get("type")
            text = block.get("text")
        else:
            text = getattr(block, "text", None)
        if block_type == "text" and text:
            parts.append(text)
    return "\n".join(parts)


def request_params(row: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    max_tokens = args.max_tokens if args.max_tokens is not None else int(config.get("max_tokens", 4096))
    params: dict[str, Any] = {
        "model": config.get("api_model", config["model_id"]),
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": row["input"]}],
        "max_tokens": max_tokens,
    }
    if not (args.no_temperature or config.get("no_temperature")):
        params["temperature"] = args.temperature
    return params


def cmd_submit(args: argparse.Namespace) -> None:
    if not getattr(args, "approved", False):
        raise SystemExit(
            "Anthropic batch submission requires explicit prior approval. "
            "Pass --approved only after the project owner approves this Anthropic batch."
        )

    model_id = args.model
    models_path = Path(args.models)
    config = load_model_config(models_path, model_id)
    prompt_dir = Path(args.prompt_dir) if getattr(args, "prompt_dir", None) else None
    prompts_path = Path(args.prompts) if args.prompts else default_prompts_path(model_id, prompt_dir)
    gen_dir = Path(args.gen_dir)
    state_dir = Path(args.state_dir)
    request_dir = Path(args.request_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_jsonl(prompts_path)
    validate_prompt_ids(prompts)
    gen_path = generation_path(model_id, gen_dir)
    done = load_done_ids(gen_path)
    remaining = [row for row in prompts if str(row["prompt_id"]) not in done]
    print(f"Model: {model_id} ({config.get('api_model', model_id)})")
    print(f"Prompts: {len(prompts):,}  done: {len(done):,}  remaining: {len(remaining):,}")
    if not remaining:
        print("Nothing to submit.")
        return
    if len(remaining) > 100_000:
        raise SystemExit("Anthropic Message Batches support up to 100,000 requests.")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = request_dir / f"{safe_name(model_id)}_{ts}.jsonl"
    requests: list[dict[str, Any]] = []
    with batch_file.open("w", encoding="utf-8") as f:
        for row in remaining:
            req = {
                "custom_id": str(row["prompt_id"]),
                "params": request_params(row, config, args),
            }
            requests.append(req)
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    size_mb = batch_file.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(remaining):,} requests to {batch_file} ({size_mb:.2f} MB)")
    if size_mb > 256:
        raise SystemExit("Anthropic Message Batches support files up to 256 MB.")

    print("Creating Anthropic message batch...")
    batch = client().messages.batches.create(requests=requests)
    print(f"Batch: {batch.id}  status={batch.processing_status}")

    state = {
        "model_id": model_id,
        "api_model": config.get("api_model", model_id),
        "batch_id": batch.id,
        "batch_file_local": str(batch_file),
        "prompts": str(prompts_path),
        "generation_output": str(gen_path),
        "submitted_at": ts,
        "n_requests": len(remaining),
        "max_tokens": args.max_tokens if args.max_tokens is not None else int(config.get("max_tokens", 4096)),
        "temperature": None if (args.no_temperature or config.get("no_temperature")) else args.temperature,
    }
    out_state = state_path(model_id, state_dir)
    out_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"State: {out_state}")


def cmd_status(args: argparse.Namespace) -> None:
    path = state_path(args.model, Path(args.state_dir))
    if not path.exists():
        raise SystemExit(f"No state file at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    batch = client().messages.batches.retrieve(state["batch_id"])
    print(f"Model:    {state['model_id']}")
    print(f"Batch:    {batch.id}")
    print(f"Status:   {batch.processing_status}")
    counts = getattr(batch, "request_counts", None)
    if counts:
        print(
            "Requests: "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}"
        )
    if getattr(batch, "results_url", None):
        print(f"Results:  {batch.results_url}")
    if batch.processing_status == "ended":
        print("Ready to retrieve.")
        sys.exit(0)
    if batch.processing_status == "canceling":
        sys.exit(2)
    sys.exit(1)


def cmd_retrieve(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    result_dir = Path(args.result_dir)
    gen_dir = Path(args.gen_dir)
    path = state_path(args.model, state_dir)
    if not path.exists():
        raise SystemExit(f"No state file at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    model_id = state["model_id"]
    claude = client()
    batch = claude.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        raise SystemExit(f"Batch is {batch.processing_status}, not ended.")

    result_dir.mkdir(parents=True, exist_ok=True)
    gen_path = generation_path(model_id, gen_dir)
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    raw_results_path = result_dir / f"{safe_name(model_id)}_{batch.id}_results.jsonl"
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    seen_ids = load_done_ids(gen_path)
    written = 0
    errors = 0
    empty = 0
    skipped = 0

    with raw_results_path.open("w", encoding="utf-8") as raw_out, gen_path.open(
        "a", encoding="utf-8"
    ) as gen_out:
        for item in claude.messages.batches.results(batch.id):
            raw_out.write(json.dumps(result_to_jsonable(item), ensure_ascii=False) + "\n")
            if str(item.custom_id) in seen_ids:
                skipped += 1
                continue
            result = item.result
            result_type = getattr(result, "type", None)
            raw_response = ""
            error_msg = None
            if result_type == "succeeded":
                raw_response = message_text(getattr(result, "message", None))
                code = clean_code(raw_response)
                if not code.strip():
                    empty += 1
                    error_msg = "EMPTY_CODE: batch response had no extractable code"
                    code_cleaned = None
                else:
                    written += 1
                    code_cleaned = code
            else:
                errors += 1
                code_cleaned = None
                error_msg = json.dumps(result_to_jsonable(result), ensure_ascii=False)

            gen_out.write(json.dumps({
                "prompt_id": item.custom_id,
                "model_id": model_id,
                "raw_response": raw_response,
                "code_cleaned": code_cleaned,
                "timestamp": ts,
                "error": error_msg,
            }, ensure_ascii=False) + "\n")

    state["retrieved_at"] = ts
    state["raw_results_path"] = str(raw_results_path)
    state["retrieve_counts"] = {
        "written": written,
        "errors": errors,
        "empty": empty,
        "skipped_existing": skipped,
    }
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Raw results: {raw_results_path}")
    print(
        f"Appended to {gen_path}: written={written:,} "
        f"errors={errors:,} empty={empty:,} skipped_existing={skipped:,}"
    )


def command_namespace(args: argparse.Namespace, model_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        models=args.models,
        prompts=(
            args.prompts
            if getattr(args, "prompts", None)
            else (
                str(default_prompts_path(model_id, Path(args.prompt_dir)))
                if getattr(args, "prompt_dir", None)
                else None
            )
        ),
        prompt_dir=args.prompt_dir,
        gen_dir=args.gen_dir,
        state_dir=args.state_dir,
        request_dir=args.request_dir,
        result_dir=args.result_dir,
        model=model_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        no_temperature=args.no_temperature,
        approved=args.approved,
    )


def prompt_ids_for_model(model_id: str, prompt_dir: Path | None = None) -> set[str]:
    prompts_path = default_prompts_path(model_id, prompt_dir)
    if not prompts_path.exists():
        raise SystemExit(f"Missing prompt file for {model_id}: {prompts_path}")
    return {str(row["prompt_id"]) for row in load_jsonl(prompts_path)}


def generation_complete(model_id: str, gen_dir: Path, prompt_dir: Path | None = None) -> bool:
    target = prompt_ids_for_model(model_id, prompt_dir)
    done = load_done_ids(generation_path(model_id, gen_dir))
    return target.issubset(done)


def cmd_queue(args: argparse.Namespace) -> None:
    queue_models = args.queue_models or DEFAULT_QUEUE_MODELS
    claude = client()
    state_dir = Path(args.state_dir)
    gen_dir = Path(args.gen_dir)
    prompt_dir = Path(args.prompt_dir) if getattr(args, "prompt_dir", None) else None
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"Queue order: {', '.join(queue_models)}")

    for model_id in queue_models:
        ns = command_namespace(args, model_id)
        submitted_this_run = 0
        print(f"\n=== {model_id} ===")

        if generation_complete(model_id, gen_dir, prompt_dir):
            print("Generation file already covers all missing prompts; skipping.")
            continue

        while True:
            path = state_path(model_id, state_dir)
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

            if state and state.get("batch_id") and not state.get("retrieved_at"):
                batch = claude.messages.batches.retrieve(state["batch_id"])
                counts = getattr(batch, "request_counts", None)
                if counts:
                    print(
                        f"{dt.datetime.now().isoformat(timespec='seconds')} "
                        f"{model_id}: {batch.processing_status} "
                        f"processing={counts.processing} succeeded={counts.succeeded} "
                        f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}"
                    )
                else:
                    print(
                        f"{dt.datetime.now().isoformat(timespec='seconds')} "
                        f"{model_id}: {batch.processing_status}"
                    )
                if batch.processing_status == "ended":
                    cmd_retrieve(ns)
                    if generation_complete(model_id, gen_dir, prompt_dir):
                        break
                    print("Retrieved batch did not complete all prompts; submitting remaining prompts.")
                    if submitted_this_run >= args.max_batches_per_model:
                        raise SystemExit(
                            f"{model_id} still has missing prompts after "
                            f"{submitted_this_run} additional batches."
                        )
                    cmd_submit(ns)
                    submitted_this_run += 1
                    continue
                if batch.processing_status == "canceling":
                    raise SystemExit(f"Batch {state['batch_id']} for {model_id} is canceling.")
                time.sleep(args.poll_seconds)
                continue

            if state and state.get("retrieved_at") and generation_complete(model_id, gen_dir, prompt_dir):
                print("Retrieved and complete.")
                break

            if submitted_this_run >= args.max_batches_per_model:
                raise SystemExit(
                    f"{model_id} still has missing prompts after "
                    f"{submitted_this_run} additional batches."
                )
            cmd_submit(ns)
            submitted_this_run += 1
            new_state_path = state_path(model_id, state_dir)
            if not new_state_path.exists():
                if generation_complete(model_id, gen_dir, prompt_dir):
                    break
                raise SystemExit(f"Submit did not create state file for {model_id}.")

        print(f"Complete: {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", default=str(DEFAULT_MODELS))
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--gen-dir", default=str(DEFAULT_GEN_DIR))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--request-dir", default=str(DEFAULT_REQUEST_DIR))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--prompt-dir", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    submit = sub.add_parser("submit")
    submit.add_argument("--model", required=True)
    submit.add_argument("--approved", action="store_true")
    submit.add_argument("--max-tokens", type=int, default=None)
    submit.add_argument("--temperature", type=float, default=0.0)
    submit.add_argument("--no-temperature", action="store_true")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status")
    status.add_argument("--model", required=True)
    status.set_defaults(func=cmd_status)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--model", required=True)
    retrieve.set_defaults(func=cmd_retrieve)

    queue = sub.add_parser("queue")
    queue.add_argument("--queue-models", nargs="*", default=None)
    queue.add_argument("--poll-seconds", type=int, default=300)
    queue.add_argument("--max-batches-per-model", type=int, default=3)
    queue.add_argument("--max-tokens", type=int, default=None)
    queue.add_argument("--temperature", type=float, default=0.0)
    queue.add_argument("--no-temperature", action="store_true")
    queue.add_argument("--approved", action="store_true")
    queue.set_defaults(func=cmd_queue)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
