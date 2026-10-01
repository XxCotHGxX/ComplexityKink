"""Model-agnostic audit of unit-test harness verdicts.

Replaces the o4-mini-only audit (src/data_provenance/06_audit_scoring.py) with
an auditor chosen under docs/independent_audit_protocol.md. The system prompt
is loaded verbatim from 06_audit_scoring.py so every auditor sees the same
instructions the original audit used.

Input rows need: case_id, task, code, unit_tests, harness_pass_rate.
Azure credentials come from the Azure CLI at start-up; OpenRouter uses OPENROUTER_API_KEY.
Nothing secret is written to disk.

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
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
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


def read_records(path: Path) -> tuple[list[dict], int]:
    """Parse a JSONL results file, skipping unreadable lines (for example a NUL-filled
    tail seen over SMB while another client is still appending)."""
    records, skipped = [], 0
    with open(path, "rb") as f:
        for raw in f:
            text = raw.strip(b"\x00\r\n ")
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except ValueError:
                skipped += 1
    return records, skipped


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_OBJECT = re.compile(r"\{[^{}]*\"verdict\"[^{}]*\}", re.DOTALL)
VERDICT_FIELD = re.compile(r"\"verdict\"\s*:\s*\"(correct|incorrect|uncertain)\"", re.IGNORECASE)
# A response cut off by the token limit or a content filter is not a final answer:
# a verdict-like string in it may come from unfinished reasoning (amendment 5).
INCOMPLETE_FINISH = ("length", "content_filter")


def final_verdict(rec: dict | None) -> str | None:
    """The verdict a stored record contributes, or None if it does not count.

    A record counts only if the request succeeded, a verdict was parsed, and the
    response was complete. Records written before amendment 5 may carry a verdict
    parsed from a truncated response, so every consumer applies this check
    rather than trusting ``status`` alone.
    """
    if not rec or rec.get("status") != "ok":
        return None
    if rec.get("finish_reason") in INCOMPLETE_FINISH:
        return None
    return rec.get("verdict")


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
        f"HARNESS PASS RATE: {float(row.get('harness_pass_rate') or 0.0):.2f}"
    )
    if cfg["reminder"]:
        user += f"\n\n{cfg['reminder']}"
    messages = [{"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": user}]
    url, headers, body = build_request(cfg, messages)
    last_error = ""
    for attempt in range(6):
        t0 = time.time()
        try:
            post = _post_streaming if cfg["stream"] else _post_blocking
            status, text, content, finish, usage, provider = post(url, headers, body, cfg)
            if status in (408, 429, 500, 502, 503, 504):
                last_error = f"http {status}: {text[:200]}"
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            if status != 200:
                return {**base, "status": "api_error", "error": f"http {status}: {text[:300]}"}
            verdict, reason, parse_status = parse_verdict(content)
            if finish in INCOMPLETE_FINISH:
                verdict, parse_status = None, f"incomplete:{finish}"
            return {**base, "status": "ok" if verdict else "parse_error", "verdict": verdict,
                    "reason": reason, "parse_status": parse_status, "finish_reason": finish,
                    "usage": usage, "transport": "stream" if cfg["stream"] else "blocking",
                    "backend": cfg["backend"], "provider": provider,
                    "latency_s": round(time.time() - t0, 2), "raw_tail": content[-600:],
                    "raw": content}  # full response, so the parse can be replayed
        except (requests.RequestException, KeyError, ValueError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(60, 2 ** attempt * 3))
    return {**base, "status": "api_error", "error": last_error}


def build_request(cfg: dict, messages: list[dict]) -> tuple[str, dict, dict]:
    """Endpoint, headers, and body for the configured backend."""
    if cfg["backend"] == "openrouter":
        body = {"model": cfg["deployment"], "messages": messages,
                "max_tokens": cfg["max_completion_tokens"], "usage": {"include": True}}
        if cfg.get("provider"):  # pin the serving provider so runs are comparable
            body["provider"] = {"order": [cfg["provider"]], "allow_fallbacks": False}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['key']}",
                   "X-Title": "ComplexityKink independent audit"}
        return OPENROUTER_URL, headers, body
    body = {"model": cfg["deployment"], "messages": messages,
            "max_completion_tokens": cfg["max_completion_tokens"]}
    url = f"{cfg['endpoint']}/models/chat/completions?api-version={API_VERSION}"
    return url, {"Content-Type": "application/json", "api-key": cfg["key"]}, body


def _post_blocking(url, headers, body, cfg):
    resp = requests.post(url, headers=headers, json=body, timeout=cfg["request_timeout"])
    if resp.status_code != 200:
        return resp.status_code, resp.text, "", None, None, None
    data = resp.json()
    if data.get("error"):
        raise ValueError(f"response error: {str(data['error'])[:200]}")
    choice = data["choices"][0]
    return (200, "", choice["message"].get("content") or "", choice.get("finish_reason"),
            data.get("usage"), data.get("provider"))


def _post_streaming(url, headers, body, cfg):
    """Stream the response. Transport only: the model computes the same thing, but the
    connection stays alive past the ~680 s cut-off Azure applies to blocking requests."""
    payload = {**body, "stream": True}
    if cfg["backend"] == "azure" and cfg.get("stream_usage", True):
        payload["stream_options"] = {"include_usage": True}
    deadline = time.time() + cfg["max_wall_s"]
    with requests.post(url, headers=headers, json=payload, stream=True,
                       timeout=(30, cfg["request_timeout"])) as resp:
        if resp.status_code != 200:
            text = resp.text
            if resp.status_code == 400 and "stream_options" in text and cfg.get("stream_usage", True):
                cfg["stream_usage"] = False  # endpoint rejects usage-on-stream; retry without it
                return 503, text, "", None, None, None
            return resp.status_code, text, "", None, None, None
        parts, finish, usage, provider = [], None, None, None
        # Split on b"\n" only: str.splitlines (the iter_lines default) also breaks on
        # U+2028, form feeds, and similar characters inside the JSON payload.
        for raw_bytes in resp.iter_lines(delimiter=b"\n"):
            raw = raw_bytes.decode("utf-8", errors="replace").rstrip("\r")
            if time.time() > deadline:
                raise TimeoutError(f"stream exceeded {cfg['max_wall_s']} s")
            if not raw or not raw.startswith("data:"):
                continue  # includes OpenRouter keep-alive comments
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):  # mid-stream failure reported in-band
                raise ValueError(f"stream error: {str(chunk['error'])[:200]}")
            provider = chunk.get("provider") or provider
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        return 200, "", "".join(parts), finish, usage, provider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--account", default=None,
                    help="Azure AI Services account; not needed when AUDIT_ENDPOINT/AUDIT_API_KEY are set.")
    ap.add_argument("--resource-group", default="ComplexityKinkResearch")
    ap.add_argument("--backend", choices=["azure", "openrouter"], default="azure",
                    help="openrouter reads its key from AUDIT_API_KEY or OPENROUTER_API_KEY.")
    ap.add_argument("--provider", default=None,
                    help="OpenRouter only: pin this serving provider (no fallbacks).")
    ap.add_argument("--deployment", required=True,
                    help="Azure deployment name, or OpenRouter model id (e.g. vendor/model:free).")
    ap.add_argument("--max-completion-tokens", type=int, default=16000)
    ap.add_argument("--request-timeout", type=int, default=900,
                    help="Seconds to wait for one response; long reasoning chains can exceed 5 minutes.")
    ap.add_argument("--no-reminder", action="store_true",
                    help="Send the original 06_audit_scoring.py user message with no format reminder "
                         "(replicates the reviewed o4-mini audit).")
    ap.add_argument("--no-stream", action="store_true",
                    help="Use blocking requests (Azure drops these at ~680 s).")
    ap.add_argument("--max-wall-s", type=int, default=2400,
                    help="Abandon one streamed response after this many seconds.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "openrouter":
        endpoint = OPENROUTER_URL
        key = os.environ.get("AUDIT_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("Set OPENROUTER_API_KEY (or AUDIT_API_KEY) for the openrouter backend.")
    else:
        endpoint, key = azure_credentials(args.account, args.resource_group)
    cfg = {"endpoint": endpoint, "key": key, "deployment": args.deployment,
           "backend": args.backend, "provider": args.provider,
           "system_prompt": original_system_prompt(),
           "max_completion_tokens": args.max_completion_tokens,
           "request_timeout": args.request_timeout,
           "stream": not args.no_stream, "max_wall_s": args.max_wall_s,
           "reminder": "" if args.no_reminder else REMINDER}

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if args.output.exists():
        records, skipped = read_records(args.output)
        done = {r["case_id"] for r in records if r.get("status") in ("ok", "parse_error")}
        if skipped:
            print(f"skipped {skipped} unreadable line(s) in {args.output.name}", flush=True)
        with open(args.output, "rb+") as f:  # a killed writer can leave a partial last line
            f.seek(0, os.SEEK_END)
            if f.tell():
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
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
