"""
Stage D step 5: score prompts with an out-of-panel rubric judge ensemble.

This intentionally reuses the exact Stage C rubric text from
src/data_provenance/05_score_complexity_rubric.py. The output is long-format:
one row per (prompt_id, scorer_id), including raw response, parsed scores,
rubric hash, model/deployment metadata, timestamp, and any error.

Typical usage:

  python src/stage_d/05_score_rubric_ensemble.py \
    --prompts data/stage_d/stage_d_prompts.jsonl \
    --scorers src/stage_d/scorers.json \
    --output data/stage_d/ensemble_scores_long.jsonl \
    --workers 12

For candidate triage, run this on candidate_prompts.jsonl with a single cheap
scorer first, then run the full ensemble on stage_d_prompts.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = ROOT / "data" / "stage_d" / "stage_d_prompts.jsonl"
DEFAULT_SCORERS = ROOT / "src" / "stage_d" / "scorers.json"
DEFAULT_OUTPUT = ROOT / "data" / "stage_d" / "ensemble_scores_long.jsonl"
RUBRIC_DIMS = [
    "branching", "iteration", "state",
    "data_structures", "edge_cases", "composition",
]
_SECRET_CACHE: dict[str, str] = {}

sys.path.insert(0, str(ROOT / "src" / "data_provenance"))
try:
    import load_keys  # type: ignore
except ImportError:
    load_keys = None


def load_stage_c_rubric() -> str:
    path = ROOT / "src" / "data_provenance" / "05_score_complexity_rubric.py"
    spec = importlib.util.spec_from_file_location("stage_c_rubric_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Stage C rubric from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SYSTEM_PROMPT


SYSTEM_PROMPT = load_stage_c_rubric()
RUBRIC_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def resolve_env(value):
    if isinstance(value, str) and value in _SECRET_CACHE:
        return _SECRET_CACHE[value]
    if isinstance(value, str) and value.startswith("$"):
        resolved = os.environ.get(value[1:], "")
        if not resolved:
            raise RuntimeError(f"Environment variable {value} is not set")
        _SECRET_CACHE[value] = resolved
        return resolved
    if isinstance(value, str) and value.startswith("azkey:"):
        # azkey:<resource_group>/<account_name>
        target = value[len("azkey:"):]
        try:
            resource_group, account_name = target.split("/", 1)
        except ValueError as exc:
            raise RuntimeError(
                "Azure key reference must be azkey:<resource_group>/<account_name>"
            ) from exc

        normalized_account = re.sub(r"[^A-Za-z0-9]+", "_", account_name).upper()
        fallback_envs = [
            f"AZURE_KEY_{normalized_account}",
            f"AZURE_COGSERVICES_KEY_{normalized_account}",
            "AZURE_OPENAI_API_KEY",
        ]
        for env_name in fallback_envs:
            key = os.environ.get(env_name)
            if key:
                _SECRET_CACHE[value] = key
                return key

        az_exe = shutil.which("az") or shutil.which("az.cmd") or "az.cmd"
        try:
            result = subprocess.run(
                [
                    az_exe, "cognitiveservices", "account", "keys", "list",
                    "-g", resource_group,
                    "-n", account_name,
                    "--query", "key1",
                    "-o", "tsv",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            hint = " or ".join(f"${name}" for name in fallback_envs)
            detail = (exc.stderr or str(exc)).strip().splitlines()[0]
            raise RuntimeError(
                f"Azure CLI key lookup failed for {target}. Refresh az login "
                f"or set {hint}. First error line: {detail}"
            ) from exc
        key = result.stdout.strip()
        if not key:
            raise RuntimeError(f"Azure CLI returned no key for {target}")
        _SECRET_CACHE[value] = key
        return key
    return value


def load_prompts(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_scorers(path: Path, only: set[str] | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    scorers = raw["scorers"] if isinstance(raw, dict) and "scorers" in raw else raw
    scorers = [s for s in scorers if not s.get("disabled")]
    if only:
        scorers = [s for s in scorers if s["scorer_id"] in only]
    if not scorers:
        raise RuntimeError("No scorers selected")
    return scorers


def resolve_scorer_runtime_config(scorer: dict) -> dict:
    """Resolve secrets/endpoints before worker threads start."""
    resolved = dict(scorer)
    if "api_key" in resolved:
        resolved["api_key"] = resolve_env(resolved["api_key"])
    if "azure_endpoint" in resolved:
        resolved["azure_endpoint"] = resolve_env(resolved["azure_endpoint"])
    return resolved


def parse_scores(raw: str) -> dict[str, int]:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    scores = {}
    for dim in RUBRIC_DIMS:
        val = parsed.get(dim)
        if isinstance(val, str) and val.strip().isdigit():
            val = int(val.strip())
        if isinstance(val, bool) or not isinstance(val, int) or val < 0 or val > 4:
            raise ValueError(f"Invalid score for {dim}: {val!r}")
        scores[dim] = val
    return scores


def existing_pairs(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("scores") and rec.get("composite") is not None and rec.get("rubric_hash") == RUBRIC_HASH:
                done.add((rec.get("prompt_id"), rec.get("scorer_id")))
    return done


def call_openai(prompt: str, scorer: dict) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=resolve_env(scorer["api_key"]), base_url=scorer.get("base_url"))
    kwargs = {
        "model": scorer.get("api_model", scorer["scorer_id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if "max_completion_tokens" in scorer:
        kwargs["max_completion_tokens"] = scorer["max_completion_tokens"]
    else:
        kwargs["max_tokens"] = scorer.get("max_tokens", 2000)
    if not scorer.get("no_temperature", False):
        kwargs["temperature"] = scorer.get("temperature", 0)
    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    finally:
        client.close()


def call_azure(prompt: str, scorer: dict) -> str:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=resolve_env(scorer["api_key"]),
        azure_endpoint=resolve_env(scorer["azure_endpoint"]),
        api_version=scorer.get("api_version", "2025-01-01-preview"),
    )
    kwargs = {
        "model": scorer["azure_deployment"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if "max_completion_tokens" in scorer:
        kwargs["max_completion_tokens"] = scorer["max_completion_tokens"]
    else:
        kwargs["max_tokens"] = scorer.get("max_tokens", 2000)
    if not scorer.get("no_temperature", False):
        kwargs["temperature"] = scorer.get("temperature", 0)
    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    finally:
        client.close()


def call_azure_inference(prompt: str, scorer: dict) -> str:
    from openai import OpenAI

    endpoint = resolve_env(scorer["azure_endpoint"])
    base = re.match(r"(https?://[^/]+)", endpoint)
    if base:
        base_url = base.group(1).rstrip("/") + "/models/"
    else:
        base_url = endpoint.rstrip("/") + "/models/"

    api_key = resolve_env(scorer["api_key"])
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"api-key": api_key},
        timeout=scorer.get("timeout", 180),
    )
    kwargs = {
        "model": scorer["azure_deployment"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if "max_completion_tokens" in scorer:
        kwargs["max_completion_tokens"] = scorer["max_completion_tokens"]
    else:
        kwargs["max_tokens"] = scorer.get("max_tokens", 2000)
    if not scorer.get("no_temperature", False):
        kwargs["temperature"] = scorer.get("temperature", 0)
    try:
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", None)
        return content or ""
    finally:
        client.close()


def call_anthropic(prompt: str, scorer: dict) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=resolve_env(scorer["api_key"]))
    kwargs = {
        "model": scorer.get("api_model", scorer["scorer_id"]),
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": scorer.get("max_tokens", 2000),
    }
    if not scorer.get("no_temperature", False):
        kwargs["temperature"] = scorer.get("temperature", 0)
    response = client.messages.create(**kwargs)
    if not response.content:
        return ""
    return response.content[0].text


def call_google(prompt: str, scorer: dict) -> str:
    import google.generativeai as genai

    genai.configure(api_key=resolve_env(scorer["api_key"]))
    model = genai.GenerativeModel(
        scorer.get("api_model", scorer["scorer_id"]),
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=scorer.get("temperature", 0),
            max_output_tokens=scorer.get("max_tokens", 2000),
        ),
        request_options={"timeout": scorer.get("timeout", 180)},
    )
    return response.text or ""


BACKENDS = {
    "openai": call_openai,
    "azure": call_azure,
    "azure_inference": call_azure_inference,
    "anthropic": call_anthropic,
    "google": call_google,
}


class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.next_time = 0.0
        self.lock = Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.time()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if delay:
            time.sleep(delay)


def score_one(prompt_rec: dict, scorer: dict, limiter: RateLimiter, max_retries: int) -> dict:
    prompt_id = prompt_rec["prompt_id"]
    scorer_id = scorer["scorer_id"]
    backend = scorer["backend"]
    raw = ""
    error = None
    scores = None
    composite = None

    for attempt in range(max_retries + 1):
        try:
            limiter.wait()
            raw = BACKENDS[backend](prompt_rec["input"], scorer)
            scores = parse_scores(raw)
            composite = sum(scores.values())
            error = None
            break
        except Exception as exc:
            error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt + random.random(), 30))

    return {
        "prompt_id": prompt_id,
        "scorer_id": scorer_id,
        "backend": backend,
        "api_model": scorer.get("api_model"),
        "azure_deployment": scorer.get("azure_deployment"),
        "rubric_hash": RUBRIC_HASH,
        "rubric_dims": RUBRIC_DIMS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "composite": composite,
        "raw_response": raw,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score rubric complexity with a scorer ensemble.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--scorers", default=str(DEFAULT_SCORERS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these scorer IDs.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Score only the first N remaining jobs; 0 = all.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if load_keys is not None:
        load_keys.load()

    only = set(args.only) if args.only else None
    prompts = load_prompts(Path(args.prompts))
    scorers = load_scorers(Path(args.scorers), only)

    for scorer in scorers:
        if scorer["backend"] not in BACKENDS:
            raise ValueError(f"Unsupported backend {scorer['backend']!r}")
    scorers = [resolve_scorer_runtime_config(scorer) for scorer in scorers]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else existing_pairs(output)
    limiters = {
        scorer["scorer_id"]: RateLimiter(float(scorer.get("rpm", 0)))
        for scorer in scorers
    }

    jobs = []
    for scorer in scorers:
        for prompt in prompts:
            pair = (prompt["prompt_id"], scorer["scorer_id"])
            if pair not in done:
                jobs.append((prompt, scorer))
    if args.limit:
        jobs = jobs[:args.limit]

    print("Stage D ensemble rubric scoring")
    print(f"  Prompts:     {len(prompts):,}")
    print(f"  Scorers:     {len(scorers):,}")
    print(f"  Existing:    {len(done):,}")
    print(f"  Jobs:        {len(jobs):,}")
    print(f"  Rubric hash: {RUBRIC_HASH[:16]}...")
    if not jobs:
        print("Nothing to do.")
        return

    lock = Lock()
    ok = 0
    err = 0
    start = time.time()
    with output.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(score_one, prompt, scorer, limiters[scorer["scorer_id"]], args.max_retries)
                for prompt, scorer in jobs
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if rec["error"] is None:
                        ok += 1
                    else:
                        err += 1
                    if i % 100 == 0 or i == len(jobs):
                        elapsed = max(time.time() - start, 1e-9)
                        rate = i / elapsed * 60
                        print(f"  {i:,}/{len(jobs):,} done | ok={ok:,} err={err:,} | {rate:.1f}/min")


if __name__ == "__main__":
    main()
