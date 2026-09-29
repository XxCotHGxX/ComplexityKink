"""Model-agnostic audit of unit-test harness verdicts.

Replaces the o4-mini-only audit (src/data_provenance/06_audit_scoring.py) with
an auditor chosen under docs/independent_audit_protocol.md. The system prompt
is loaded verbatim from 06_audit_scoring.py so every auditor sees the same
instructions the original audit used.

Input rows need: case_id, task, code, unit_tests, harness_pass_rate.
Credentials come from the Azure CLI at start-up (nothing is written to disk).

Usage:
    python src/audit/independent_audit.py --input known_answer_set.jsonl \
        --output audits_phi4.jsonl --account DataPipeline0 --deployment Phi-4-reasoning
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
API_VERSION = "2024-05-01-preview"
REMINDER = (
    'Respond with ONLY a JSON object of the form {"verdict": "correct" | "incorrect" | "uncertain", '
    '"reason": "<one short sentence>"} giving your verdict on the CODE above. '
    "Do not repeat the task, code, or tests."
)


def original_system_prompt() -> str:
    spec = importlib.util.spec_from_file_location(
        "audit_v1", ROOT / "src" / "data_provenance" / "06_audit_scoring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEM_PROMPT


def azure_credentials(account: str | None, resource_group: str) -> tuple[str, str]:
    """AUDIT_ENDPOINT/AUDIT_API_KEY (e.g. container secrets) win; else ask the Azure CLI."""
    env_endpoint, env_key = os.environ.get("AUDIT_ENDPOINT"), os.environ.get("AUDIT_API_KEY")
    if env_endpoint and env_key:
        return env_endpoint.rstrip("/"), env_key
    if not account:
        raise SystemExit("Set AUDIT_ENDPOINT and AUDIT_API_KEY, or pass --account.")

    def az(*args: str) -> str:
        return subprocess.run(["az", *args, "-o", "tsv"], check=True, capture_output=True,
                              text=True, shell=True).stdout.strip()
    endpoint = az("cognitiveservices", "account", "show", "-g", resource_group, "-n", account,
                  "--query", "properties.endpoint")
    key = az("cognitiveservices", "account", "keys", "list", "-g", resource_group, "-n", account,
             "--query", "key1")
    return endpoint.rstrip("/"), key


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_OBJECT = re.compile(r"\{[^{}]*\"verdict\"[^{}]*\}", re.DOTALL)
VERDICT_FIELD = re.compile(r"\"verdict\"\s*:\s*\"(correct|incorrect|uncertain)\"", re.IGNORECASE)


def parse_verdict(content: str) -> tuple[str | None, str, str]:
    """Return (verdict, reason, parse_status) from a model response."""
    text = THINK_BLOCK.sub("", content or "")
    for candidate in reversed(JSON_OBJECT.findall(text)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        verdict = str(obj.get("verdict", "")).lower()
        if verdict in ("correct", "incorrect", "uncertain"):
            return verdict, str(obj.get("reason", "")), "json"
    match = VERDICT_FIELD.findall(text)
    if match:
        return match[-1].lower(), "", "regex"
    return None, "", "parse_error"


def audit_one(row: dict, cfg: dict) -> dict:
    base = {"case_id": row["case_id"], "auditor": cfg["deployment"]}
    code = row.get("code") or ""
    if not code.strip():
        return {**base, "status": "ok", "verdict": "incorrect", "reason": "empty code",
                "parse_status": "rule"}
    user = (
        f"TASK:\n{row.get('task') or ''}\n\n"
        f"CODE:\n```python\n{code}\n```\n\n"
        f"UNIT TESTS:\n{row.get('unit_tests') or ''}\n\n"
        f"HARNESS PASS RATE: {float(row.get('harness_pass_rate') or 0.0):.2f}\n\n"
        f"{REMINDER}"
    )
    body = {
        "model": cfg["deployment"],
        "messages": [{"role": "system", "content": cfg["system_prompt"]},
                     {"role": "user", "content": user}],
        "max_completion_tokens": cfg["max_completion_tokens"],
    }
    url = f"{cfg['endpoint']}/models/chat/completions?api-version={API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": cfg["key"]}
    last_error = ""
    for attempt in range(6):
        t0 = time.time()
        try:
            if cfg["stream"]:
                status, text, content, finish, usage = _post_streaming(url, headers, body, cfg)
            else:
                status, text, content, finish, usage = _post_blocking(url, headers, body, cfg)
            if status in (429, 500, 502, 503, 504):
                last_error = f"http {status}"
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            if status != 200:
                return {**base, "status": "api_error", "error": f"http {status}: {text[:300]}"}
            verdict, reason, parse_status = parse_verdict(content)
            return {**base, "status": "ok" if verdict else "parse_error", "verdict": verdict,
                    "reason": reason, "parse_status": parse_status, "finish_reason": finish,
                    "usage": usage, "transport": "stream" if cfg["stream"] else "blocking",
                    "latency_s": round(time.time() - t0, 2), "raw_tail": content[-600:]}
        except (requests.RequestException, KeyError, ValueError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(60, 2 ** attempt * 3))
    return {**base, "status": "api_error", "error": last_error}


def _post_blocking(url, headers, body, cfg):
    resp = requests.post(url, headers=headers, json=body, timeout=cfg["request_timeout"])
    if resp.status_code != 200:
        return resp.status_code, resp.text, "", None, None
    data = resp.json()
    choice = data["choices"][0]
    return 200, "", choice["message"].get("content") or "", choice.get("finish_reason"), data.get("usage")


def _post_streaming(url, headers, body, cfg):
    """Stream the response. Transport only: the model computes the same thing, but the
    connection stays alive past the ~680 s cut-off Azure applies to blocking requests."""
    payload = {**body, "stream": True}
    if cfg.get("stream_usage", True):
        payload["stream_options"] = {"include_usage": True}
    deadline = time.time() + cfg["max_wall_s"]
    with requests.post(url, headers=headers, json=payload, stream=True,
                       timeout=(30, cfg["request_timeout"])) as resp:
        if resp.status_code != 200:
            text = resp.text
            if resp.status_code == 400 and "stream_options" in text and cfg.get("stream_usage", True):
                cfg["stream_usage"] = False  # endpoint rejects usage-on-stream; retry without it
                return 503, text, "", None, None
            return resp.status_code, text, "", None, None
        parts, finish, usage = [], None, None
        for raw in resp.iter_lines(decode_unicode=True):
            if time.time() > deadline:
                raise TimeoutError(f"stream exceeded {cfg['max_wall_s']} s")
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        return 200, "", "".join(parts), finish, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--account", default=None,
                    help="Azure AI Services account; not needed when AUDIT_ENDPOINT/AUDIT_API_KEY are set.")
    ap.add_argument("--resource-group", default="ComplexityKinkResearch")
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--max-completion-tokens", type=int, default=16000)
    ap.add_argument("--request-timeout", type=int, default=900,
                    help="Seconds to wait for one response; long reasoning chains can exceed 5 minutes.")
    ap.add_argument("--no-stream", action="store_true",
                    help="Use blocking requests (Azure drops these at ~680 s).")
    ap.add_argument("--max-wall-s", type=int, default=2400,
                    help="Abandon one streamed response after this many seconds.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    endpoint, key = azure_credentials(args.account, args.resource_group)
    cfg = {"endpoint": endpoint, "key": key, "deployment": args.deployment,
           "system_prompt": original_system_prompt(),
           "max_completion_tokens": args.max_completion_tokens,
           "request_timeout": args.request_timeout,
           "stream": not args.no_stream, "max_wall_s": args.max_wall_s}

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") in ("ok", "parse_error"):
                    done.add(r["case_id"])
    pending = [r for r in rows if r["case_id"] not in done]
    print(f"{args.deployment}: {len(rows)} rows, {len(done)} done, {len(pending)} pending", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    t0 = time.time()
    with open(args.output, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(audit_one, r, cfg) for r in pending]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if i % 50 == 0 or i == len(pending):
                rate = i / max(time.time() - t0, 1e-9) * 60
                print(f"  [{i}/{len(pending)}] {counts} {rate:.0f}/min", flush=True)


if __name__ == "__main__":
    main()
