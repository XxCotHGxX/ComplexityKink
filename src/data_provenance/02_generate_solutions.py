"""
Step 2: Generate solutions from multiple LLMs for the selected prompt set.

Sends each prompt to a configurable list of models via their respective APIs
and records the raw response.  Each model's generations are stored in a
separate JSONL file under data/generations/.

SUPPORTED BACKENDS:
  - "openai"            : OpenAI-compatible API (GPT-5.x, GPT-4.1, etc.)
  - "anthropic"         : Anthropic API (Claude Sonnet/Opus/Haiku)
  - "google"            : Google Generative AI (Gemini, requires google-generativeai)
  - "copilot"           : GitHub Copilot (auto-loads from apps.json, zero config)
  - "gemini_cloudcode"  : Google cloudcode API (auto-loads from ~/.gemini/oauth_creds.json)
  - "azure"             : Azure OpenAI Service (requires api_key, azure_endpoint, azure_deployment)
  - "local"             : Local/self-hosted models via OpenAI-compatible endpoint

USAGE:
  python src/data_provenance/02_generate_solutions.py \\
      --prompts data/experiment_prompts.jsonl \\
      --models models.json \\
      --output-dir data/generations

  See models_example.json for the expected model configuration format.

REPRODUCIBILITY:
  - temperature = 0 (greedy decoding) for deterministic output
  - All prompts are processed in the same order for every model
  - Each output record includes: prompt_id, model_id, raw_response, timestamp
"""
import os
import sys
import json
import time
import random
import re
import shutil
import subprocess
import httpx
import uuid
import argparse
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Optional author-side auth helper.
#
# Public users configure API keys via environment variables (OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GOOGLE_API_KEY, AZURE_* etc.) ,  the standard way.
# ``load_keys`` is an author-only shim that reads CLI-tool auth stores
# (Copilot, Gemini CLI, Antigravity) so the same script can also drive those
# backends locally. Its absence must never break the public pipeline.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
try:
    import load_keys  # type: ignore
except ImportError:
    load_keys = None

_SECRET_CACHE = {}


# ---------------------------------------------------------------------------
# System prompt used for ALL models ,  identical to ensure fairness
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a coding problem, write a "
    "complete Python solution. Output ONLY the Python code inside a single "
    "```python``` code block. Do not include any explanation, tests, or "
    "examples outside the code block."
)
EMPTY_CODE_ERROR = "EMPTY_CODE: model returned reasoning only, no extractable code block"
EMPTY_CODE_RETRY_DELAY_SECONDS = 2


def clean_code_from_response(text):
    """Extract code from a markdown-fenced response, filtering out junk."""
    if not text:
        return ""
    import re
    # Strip thinking blocks from reasoning models
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Try to find the first python block specifically
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # If there are multiple blocks, take the LAST one.
        # Reasoning models often output a draft and then a final version.
        return matches[-1].strip()
    
    # If no markdown blocks, try to find a raw function definition or fallback
    if "def " in text:
        # Extremely basic fallback: if it looks like code but no markdown, take it all
        return text.strip()
        
    return text.strip()


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible API
# ---------------------------------------------------------------------------
def generate_openai(prompt, model_id, api_key, base_url=None, **kwargs):
    """Generate via OpenAI API (works for GPT-5.x, GPT-4.1, local endpoints)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client_kwargs = {"api_key": api_key, "timeout": 1200.0}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    user_content = prompt
    if kwargs.get("no_think"):
        user_content = "/no_think\n" + prompt
    create_kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    if "max_completion_tokens" in kwargs:
        create_kwargs["max_completion_tokens"] = kwargs["max_completion_tokens"]
    else:
        create_kwargs["max_tokens"] = kwargs.get("max_tokens", 4096)
    if not kwargs.get("no_temperature", False):
        create_kwargs["temperature"] = kwargs.get("temperature", 0.0)
    if "frequency_penalty" in kwargs:
        create_kwargs["frequency_penalty"] = kwargs["frequency_penalty"]
    if "presence_penalty" in kwargs:
        create_kwargs["presence_penalty"] = kwargs["presence_penalty"]
    extra_body = {}
    if "enable_thinking" in kwargs:
        extra_body["enable_thinking"] = kwargs["enable_thinking"]
    if "thinking_budget" in kwargs:
        extra_body["thinking_budget"] = kwargs["thinking_budget"]
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    if kwargs.get("stream"):
        create_kwargs["stream"] = True
        content_parts = []
        with client.chat.completions.create(**create_kwargs) as stream:
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                delta = choices[0].delta if choices else None
                if delta and delta.content:
                    content_parts.append(delta.content)
        return "".join(content_parts)

    response = client.chat.completions.create(**create_kwargs)
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("MALFORMED_RESPONSE: no choices returned")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("MALFORMED_RESPONSE: no message returned")
    return getattr(message, "content", None) or ""


# ---------------------------------------------------------------------------
# Backend: Ollama native API (uses /api/chat with think:false)
# ---------------------------------------------------------------------------
def generate_ollama(prompt, model_id, api_key, base_url=None, **kwargs):
    """Generate via Ollama native /api/chat endpoint. Disables thinking mode.
    Uses streaming to keep the connection alive past Azure's 240s ingress timeout."""
    import httpx
    # base_url is like "https://host/v1" ,  strip /v1 to get the base
    ollama_base = base_url.rstrip("/")
    if ollama_base.endswith("/v1"):
        ollama_base = ollama_base[:-3]
    url = f"{ollama_base}/api/chat"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": False,
        "options": {
            "num_predict": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.0),
        },
    }
    content_parts = []
    with httpx.stream("POST", url, json=payload, timeout=1200.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if msg.get("content"):
                content_parts.append(msg["content"])
            if chunk.get("done"):
                break
    return "".join(content_parts)


# ---------------------------------------------------------------------------
# Backend: Anthropic API
# ---------------------------------------------------------------------------
def generate_anthropic(prompt, model_id, api_key, **kwargs):
    """Generate via Anthropic API (Claude models)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    create_kwargs = {
        "model": model_id,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": kwargs.get("max_tokens", 4096),
    }
    if not kwargs.get("no_temperature"):
        create_kwargs["temperature"] = kwargs.get("temperature", 0)
    for _attempt in range(3):
        response = client.messages.create(**create_kwargs)
        if response.content:
            return response.content[0].text
    raise RuntimeError("Anthropic returned empty content after 3 attempts")


# ---------------------------------------------------------------------------
# Backend: Google Generative AI API
# ---------------------------------------------------------------------------
def generate_google(prompt, model_id, api_key, **kwargs):
    """Generate via Google Generative AI API (Gemini models)."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_id,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=kwargs.get("max_tokens", 4096)),
        request_options={"timeout": 180},
    )
    return response.text

# ---------------------------------------------------------------------------
# Backend: Google Vertex AI (region-selectable, separate quotas)
# ---------------------------------------------------------------------------
def generate_vertex(prompt, model_id, api_key=None, **kwargs):
    """Generate via Vertex AI API. Supports region selection to dodge capacity issues."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        raise ImportError("pip install google-cloud-aiplatform")

    location = kwargs.get("vertex_location", "us-central1")
    project = kwargs.get("vertex_project", None)
    vertexai.init(project=project, location=location)
    model = GenerativeModel(
        model_id,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=0, max_output_tokens=kwargs.get("max_tokens", 4096)),
    )
    return response.text


# ---------------------------------------------------------------------------
# Backend: Azure OpenAI Service
# ---------------------------------------------------------------------------
def generate_azure(prompt, model_id, api_key, azure_endpoint, azure_deployment, **kwargs):
    """Generate via Azure OpenAI Service."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("pip install openai")

    # Auto-fix: Azure AI Foundry endpoints often need /openai appended for standard SDK
    if ".services.ai.azure.com" in azure_endpoint and not azure_endpoint.endswith("/openai"):
        azure_endpoint = azure_endpoint.rstrip("/") + "/openai"

    client = AzureOpenAI(
        api_key=api_key,
        api_version=kwargs.get("api_version", "2024-05-01-preview"),
        azure_endpoint=azure_endpoint
    )

    response = client.chat.completions.create(
        model=azure_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 4096),
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Backend: Azure AI Foundry (Inference/MaaS - /models/ endpoint)
# ---------------------------------------------------------------------------
def generate_azure_inference(prompt, model_id, api_key, azure_endpoint, azure_deployment, **kwargs):
    """Generate via Azure AI Foundry using the /models/ OpenAI-compatible endpoint.
    
    The Foundry endpoint format is:
      https://<project>.services.ai.azure.com/models/chat/completions
    
    We use the OpenAI client with base_url pointing to the /models/ prefix.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    # Strip path down to just the base (schema + host), then add /models/
    # Handles both:
    #   https://proj.services.ai.azure.com/
    #   https://proj.services.ai.azure.com/models/chat/completions?...
    import re
    base = re.match(r'(https?://[^/]+)', azure_endpoint)
    if base:
        base_url = base.group(1).rstrip('/') + '/models/'
    else:
        base_url = azure_endpoint.rstrip('/') + '/models/'

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"api-key": api_key},  # Azure Foundry uses api-key header
    )

    response = client.chat.completions.create(
        model=azure_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 4096),
    )
    msg = response.choices[0].message
    # Reasoning models (e.g. Kimi-K2.5) return content in reasoning_content
    content = msg.content
    if not content:
        content = getattr(msg, "reasoning_content", None)
    return content


# ---------------------------------------------------------------------------
# Shared Google Cloud Code Backend (Robust with Streaming & Retries)
# ---------------------------------------------------------------------------
def _generate_google_cloudcode_base(prompt, model_id, token, project, user_agent="antigravity", retries=3, **kwargs):
    """Robust generation via Google cloudcode API with retries and thought filtering."""
    if not token or not project:
        raise RuntimeError(f"Missing credentials for {user_agent} backend")

    url = "https://cloudcode-pa.googleapis.com/v1internal:generateContent"

    gen_req = {
        "model": model_id,
        "project": project,
        "user_prompt_id": str(uuid.uuid4()),
        "request": {
            "contents": [{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": kwargs.get("max_tokens", 4096)},
            "session_id": str(uuid.uuid4()),
        },
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
        "X-Goog-Api-Client": "google-cloud-sdk-custom"
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=90.0) as client:
                res = client.post(url, json=gen_req, headers=headers)
                if res.status_code == 429 or res.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server busy ({res.status_code})",
                        request=res.request, response=res
                    )
                res.raise_for_status()
                result = res.json()

            # Handle list (streaming chunks) or single object
            if isinstance(result, list):
                candidates = []
                for chunk in result:
                    c = chunk.get("response", chunk).get("candidates", [])
                    if c:
                        candidates = c
                        break
            else:
                resp_obj = result.get("response", result)
                candidates = resp_obj.get("candidates", [])

            if not candidates:
                raise RuntimeError(f"No candidates returned for {model_id}. Result: {str(result)[:300]}")

            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = [p["text"] for p in parts if "text" in p]

            if not text_parts:
                raise RuntimeError(f"Empty text in candidates for {model_id}")

            return "\n".join(text_parts)

        except (httpx.HTTPError, httpx.NetworkError, RuntimeError) as e:
            last_err = e
            if attempt < retries:
                # Short fixed backoff: 5s, 10s, 15s... capped at 30s
                wait = min(5 * (attempt + 1), 30) + random.random()
                print(f"    [{model_id}] Retry {attempt+1}/{retries} after {wait:.1f}s: {e}")
                time.sleep(wait)
                continue
            break

    raise last_err if last_err else RuntimeError("Generation failed after all retries")

def _require_load_keys(backend_name):
    if load_keys is None:
        raise RuntimeError(
            f"Backend '{backend_name}' depends on the author-side load_keys "
            "helper, which is not shipped with the public repository. Use one "
            "of the API backends (openai, anthropic, google, azure, azure_inference) "
            "with environment-variable credentials instead."
        )


def generate_gemini_cloudcode(prompt, model_id, api_key=None, **kwargs):
    """Generate via Google cloudcode API using Gemini CLI (AI Pro) credentials."""
    _require_load_keys("gemini_cloudcode")
    token, project = load_keys.get_gemini_session()
    ua = f"GeminiCLI/0.29.7/{model_id} (win32; x64)"
    retries = int(kwargs.get("max_retries", 3))
    return _generate_google_cloudcode_base(prompt, model_id, token, project, user_agent=ua, retries=retries, **kwargs)

def generate_antigravity(prompt, model_id, api_key=None, **kwargs):
    """Generate via Google cloudcode API using Antigravity (Moltbot) credentials."""
    _require_load_keys("antigravity")
    token, project = load_keys.get_antigravity_session()
    return _generate_google_cloudcode_base(prompt, model_id, token, project, user_agent="antigravity/1.0")

# ---------------------------------------------------------------------------
# Backend: GitHub Copilot API (uses Copilot OAuth token exchange)
# ---------------------------------------------------------------------------
def generate_copilot(prompt, model_id, api_key=None, **kwargs):
    """Generate via GitHub Copilot API using session token from apps.json."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    _require_load_keys("copilot")
    token, endpoint = load_keys.get_copilot_session()
    if not token:
        raise RuntimeError("Could not obtain Copilot session token")

    client = OpenAI(
        api_key=token,
        base_url=f"{endpoint}",
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=4096,
        extra_headers={
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.96.0",
            "Editor-Plugin-Version": "copilot-chat/0.24.0",
        },
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Backend: Codex CLI API (uses ~/.codex/auth.json token)
# ---------------------------------------------------------------------------
def generate_codex(prompt, model_id, api_key=None, **kwargs):
    """Generate using the actual Codex CLI binary to handle restricted tokens."""
    import subprocess
    import tempfile
    import uuid
    import shutil
    
    if not shutil.which("codex"):
        raise RuntimeError("The 'codex' command line tool is not installed or not in PATH.")

    # Prevent LM Studio's OPENAI_BASE_URL from hijacking the codex CLI's internal requests
    env = os.environ.copy()
    env.pop("OPENAI_BASE_URL", None)
    
    # Create a unique temporary file to capture strictly the markdown response
    tmp_out = os.path.join(tempfile.gettempdir(), f"codex_out_{uuid.uuid4().hex}.md")
    
    cmd = [
        "codex", "exec", 
        "--output-last-message", tmp_out
    ]
    
    try:
        # Run codex exec, streaming the full prompt to stdin
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            env=env,
            timeout=240  # 4 minutes max per prompt
        )
        
        if not os.path.exists(tmp_out):
            error_msg = result.stderr.strip()
            raise RuntimeError(f"Codex CLI failed to produce output. Error: {error_msg}")
            
        with open(tmp_out, "r", encoding="utf-8") as f:
            content = f.read()
            
        return content
        
    finally:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
BACKENDS = {
    "openai": generate_openai,
    "anthropic": generate_anthropic,
    "google": generate_google,
    "local": generate_ollama,  # local uses native Ollama /api/chat (think:false)
    "copilot": generate_copilot,
    "gemini_cloudcode": generate_gemini_cloudcode,
    "antigravity": generate_antigravity,
    "vertex": generate_vertex,
    "azure": generate_azure,
    "azure_inference": generate_azure_inference,
    "codex": generate_codex,
}


class CircuitBreakerException(Exception):
    pass

def generate_for_model(prompts, model_config, output_dir, resume=True, max_workers=1, local_model_lock=None):
    """
    Generate solutions for all prompts using a single model configuration.

    Parameters
    ----------
    prompts : list[dict]
        List of prompt records from experiment_prompts.jsonl.
    model_config : dict
        Model configuration with keys:
          - model_id: str (display name / output file key)
          - api_model: str (actual API model name)
          - backend: str (one of BACKENDS keys)
          - api_key: str (or env var name prefixed with $)
          - base_url: str (optional, for local/custom endpoints)
    output_dir : str
        Directory to write output files.
    resume : bool
        If True, skip prompts already in the output file.
    max_workers: int
        Number of threads for concurrent generation.
    """
    model_id = model_config["model_id"]
    api_model = model_config.get("api_model", model_id)
    backend = model_config["backend"]
    api_key = model_config.get("api_key", "")
    base_url = model_config.get("base_url", None)

    # Allow env var BASE_URL_OVERRIDE to replace external URLs with internal ones
    # (useful in Azure Container Apps to bypass ingress timeout limits)
    base_url_override = os.environ.get("BASE_URL_OVERRIDE")
    if base_url_override and base_url:
        base_url = base_url_override

    # Resolve ALL env var references (values starting with $) in the model config
    def resolve_env(val):
        if isinstance(val, str) and val in _SECRET_CACHE:
            return _SECRET_CACHE[val]
        if isinstance(val, str) and val.startswith("$"):
            resolved = os.environ.get(val[1:], "")
            if not resolved:
                print(f"  WARNING: Env var {val} not set!")
            else:
                _SECRET_CACHE[val] = resolved
            return resolved
        if isinstance(val, str) and val.startswith("azkey:"):
            target = val[len("azkey:"):]
            try:
                resource_group, account_name = target.split("/", 1)
            except ValueError:
                print("  WARNING: Azure key reference must be azkey:<resource_group>/<account_name>")
                return ""

            normalized_account = re.sub(r"[^A-Za-z0-9]+", "_", account_name).upper()
            fallback_envs = [
                f"AZURE_KEY_{normalized_account}",
                f"AZURE_COGSERVICES_KEY_{normalized_account}",
            ]
            if account_name.lower() == "datapipeline0":
                fallback_envs.append("AZURE_OPENAI_API_KEY")
            for env_name in fallback_envs:
                resolved = os.environ.get(env_name)
                if resolved:
                    _SECRET_CACHE[val] = resolved
                    return resolved

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
                detail = (exc.stderr or str(exc)).strip().splitlines()[0]
                print(
                    f"  WARNING: Azure CLI key lookup failed for {target}; "
                    f"refresh az login or set one of {fallback_envs}. "
                    f"First error line: {detail}"
                )
                return ""
            resolved = result.stdout.strip()
            _SECRET_CACHE[val] = resolved
            return resolved
        return val

    api_key = resolve_env(api_key)
    # Local and Copilot backends don't strictly require an 'api_key' in the config
    if not api_key and backend not in ["local", "copilot", "antigravity", "gemini_cloudcode"]:
        print(f"  WARNING: API key env var {model_config.get('api_key')} not set, skipping.")
        return

    # Also resolve azure_endpoint if it's an env var reference
    if "azure_endpoint" in model_config:
        model_config = dict(model_config)  # don't mutate the original
        model_config["azure_endpoint"] = resolve_env(model_config["azure_endpoint"])

    gen_fn = BACKENDS.get(backend)
    if gen_fn is None:
        print(f"  ERROR: Unknown backend '{backend}' for model {model_id}")
        return

    # Output file (sanitize for Windows: no / or :)
    safe_name = model_id.replace("/", "_").replace(" ", "_").replace(":", "_")
    outpath = os.path.join(output_dir, f"{safe_name}.jsonl")
    # If primary file is missing (e.g. Azure SMB DeletePending), use _restore copy
    if not os.path.exists(outpath):
        restore_path = outpath.replace(".jsonl", "_restore.jsonl")
        if os.path.exists(restore_path):
            outpath = restore_path
            print(f"  Using restore file: {outpath}")

    # Resume: load already-completed prompt_ids
    # Only count a prompt as done if it has real, non-empty code.
    # Blank/error records are automatically re-queued on the next run.
    target_ids = {p["prompt_id"] for p in prompts}
    done_ids = set()
    blank_ids = set()
    if resume and os.path.exists(outpath):
        with open(outpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("prompt_id")
                if pid not in target_ids:
                    continue
                code = rec.get("code_cleaned") or ""
                if code.strip() and rec.get("error") is None:
                    done_ids.add(pid)
                else:
                    blank_ids.add(pid)
        requeue = len(blank_ids)
        print(f"  Resuming: {len(done_ids)} complete, {requeue} blank/error re-queued")


    remaining = [p for p in prompts if p["prompt_id"] not in done_ids]
    print(f"  {model_id}: {len(remaining)} prompts to generate ({len(done_ids)} done)")

    # NOTE: Cleanup of stale records is deferred to clean_results.py.
    # In-place cleanup during resume was removed because os.replace() and
    # file rewrites are unsafe on Azure File Share (SMB), causing
    # DeletePending races that destroy the output file.
    # Ensure output file exists (may have been lost during cleanup on network FS)
    if not os.path.exists(outpath):
        open(outpath, 'a', encoding='utf-8').close()

    rpm = model_config.get("rpm", 15)
    min_interval = 60.0 / rpm if rpm > 0 else 0
    max_retries = model_config.get("max_retries", 3)

    success = 0
    errors = 0
    consecutive_failures = 0
    start = time.time()
    last_request_time = 0

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Thread safety for file writes and progress tracking
    lock = threading.Lock()
    total_done = 0

    def process_prompt(prompt_rec):
        nonlocal last_request_time, consecutive_failures

        # Rate limit waiting (thread-safe check)
        if min_interval > 0:
            with lock:
                now = time.time()
                wait = min_interval - (now - last_request_time)
                if wait > 0:
                    last_request_time = now + wait
                else:
                    last_request_time = now
            if wait > 0:
                time.sleep(wait)

        raw_response = None
        error_msg = None
        code_cleaned = None
        is_quota_error = False
        
        for attempt in range(max_retries):
            try:
                # Prepare call kwargs, including Azure-specific ones
                call_kwargs = {
                    "base_url": base_url,
                    "max_tokens": model_config.get("max_tokens", 4096),
                    "temperature": model_config.get("temperature", 0.0),
                }
                if "max_completion_tokens" in model_config:
                    call_kwargs["max_completion_tokens"] = model_config["max_completion_tokens"]
                if model_config.get("no_think"):
                    call_kwargs["no_think"] = True
                if model_config.get("no_temperature"):
                    call_kwargs["no_temperature"] = True
                if model_config.get("stream"):
                    call_kwargs["stream"] = True
                if "enable_thinking" in model_config:
                    call_kwargs["enable_thinking"] = model_config["enable_thinking"]
                if "thinking_budget" in model_config:
                    call_kwargs["thinking_budget"] = model_config["thinking_budget"]
                # Some APIs (e.g. Grok) don't accept penalty params at all
                if not model_config.get("no_penalties", False):
                    call_kwargs["frequency_penalty"] = model_config.get("frequency_penalty", 0.0)
                    call_kwargs["presence_penalty"] = model_config.get("presence_penalty", 0.0)
                
                # Pass through Vertex AI fields
                if backend == "vertex":
                    call_kwargs["vertex_location"] = model_config.get("vertex_location", "us-central1")
                    call_kwargs["vertex_project"] = model_config.get("vertex_project")

                # Pass through Azure-specific fields if they exist
                if backend in ["azure", "azure_inference"]:
                    call_kwargs["azure_endpoint"] = model_config.get("azure_endpoint")
                    call_kwargs["azure_deployment"] = model_config.get("azure_deployment")
                    call_kwargs["api_version"] = model_config.get("api_version")

                raw_response = gen_fn(
                    prompt_rec["input"],
                    api_model,
                    api_key,
                    **call_kwargs
                )
                code_cleaned = clean_code_from_response(raw_response or "")
                if not code_cleaned or not code_cleaned.strip():
                    error_msg = EMPTY_CODE_ERROR
                    if attempt < max_retries - 1:
                        print(
                            f"      [!] {model_id} empty/code-less response; "
                            f"retrying ({attempt + 1}/{max_retries})"
                        )
                        time.sleep(EMPTY_CODE_RETRY_DELAY_SECONDS)
                        continue
                    break
                error_msg = None
                is_quota_error = False
                with lock:
                    consecutive_failures = 0
                break
            except Exception as e:
                error_msg = str(e)
                # PRINT DETAILED ERROR FOR DEBUGGING
                print(f"      [!] {model_id} error: {error_msg}")
                
                is_rate_limit = any(s in error_msg.lower()
                                   for s in ['rate limit', '429', 'quota', 'too many', '403', 'forbidden', 'insufficient_quota'])
                
                if is_rate_limit:
                    is_quota_error = True
                    if attempt < max_retries - 1:
                        backoff = min(15 * (2 ** attempt), 300)
                        print(f"    Rate limited/Quota, waiting {backoff}s before retry...")
                        time.sleep(backoff)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        if error_msg is not None:
            code_cleaned = None

        record = {
            "prompt_id": prompt_rec["prompt_id"],
            "model_id": model_id,
            "raw_response": raw_response,
            "code_cleaned": code_cleaned,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg,
        }
        
        return record, is_quota_error

    def run_prompts():
        nonlocal success, errors, consecutive_failures, total_done

        # Open file in append mode for the duration of this model's run
        with open(outpath, 'a', encoding='utf-8') as f_out:
            # Concurrent processing of remaining prompts for this model
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_prompt = {executor.submit(process_prompt, p): p for p in remaining}
                    
                    for future in as_completed(future_to_prompt):
                        record, is_quota_error = future.result()
                        
                        with lock:
                            if record["error"] is None:
                                success += 1
                                consecutive_failures = 0
                            else:
                                consecutive_failures += 1
                                errors += 1
                                
                                if is_quota_error:
                                    print(f"    [!] Hard quota/rate-limit hit ({record['error']})")
                                    raise CircuitBreakerException(f"API Quota exhausted or blocked ({record['error']})")
                                    
                                if consecutive_failures >= max(10, max_workers * 2):
                                    print(f"    [!] Circuit Breaker Tripped! ({record['error']})")
                                    raise CircuitBreakerException(f"Too many consecutive API failures ({record['error']})")

                            # Write to file and flush safely
                            f_out.write(json.dumps(record) + '\n')
                            f_out.flush()

                            total_done += 1
                            # Periodic progress update
                            if total_done % max(1, max_workers) == 0 or total_done == len(remaining):
                                elapsed = time.time() - start
                                rate = total_done / elapsed if elapsed > 0 else 0
                                eta = (len(remaining) - total_done) / rate if rate > 0 else 0
                                
                                # Human readable ETA
                                if eta < 60:
                                    eta_str = f"{eta:.0f}s"
                                elif eta < 3600:
                                    eta_str = f"{eta/60:.1f}m"
                                else:
                                    eta_str = f"{eta/3600:.1f}h"
                                    
                                print(f"    [{model_id}] Progress: {total_done}/{len(remaining)} "
                                      f"({total_done/len(remaining)*100:.1f}%) | "
                                      f"{success} ok, {errors} err | "
                                      f"{rate:.1f} req/s | ETA {eta_str}")
                                      
            except CircuitBreakerException:
                # Executor shutdown is handled by the context manager
                raise
            except Exception as e:
                print(f"    [!] Unexpected thread error: {e}")
                with lock:
                    errors += 1
                    total_done += 1

        elapsed = time.time() - start
        print(f"  [{model_id}] Done: {success} success, {errors} errors in {elapsed:.0f}s")
        
    if backend == "local" and local_model_lock is not None:
        print(f"  [{model_id}] Waiting for local model lock...")
        with local_model_lock:
            print(f"  [{model_id}] Acquired local model lock. Generating...")
            run_prompts()
    else:
        run_prompts()


def main():
    parser = argparse.ArgumentParser(description="Multi-model code generation")
    parser.add_argument("--prompts", default=os.path.join("data", "experiment_prompts.jsonl"),
                        help="Path to prompt set JSONL")
    parser.add_argument("--models", default=os.path.join("src", "data_provenance", "models.json"),
                        help="Path to model configuration JSON")
    parser.add_argument("--output-dir", default=os.path.join("data", "generations"),
                        help="Output directory for per-model generation files")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these model IDs (space-separated)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume from existing output files")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of models to process in parallel")
    parser.add_argument("--max-workers-per-model", type=int, default=5,
                        help="Number of concurrent API requests per model")
    args = parser.parse_args()

    # Public users rely on standard environment variables. The author-only
    # CLI-auth shim is optional; skip silently if it is absent.
    if load_keys is not None:
        print("Loading API keys...")
        load_keys.status()
        print()

    # Load prompts
    prompts = []
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Load model configs
    with open(args.models, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    models = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    print(f"Loaded {len(models)} model configurations")

    # Filter if --only specified
    if args.only:
        # Support both space-separated (from nargs=+) and comma-separated inputs
        target_ids = []
        for entry in args.only:
            target_ids.extend(entry.replace(",", " ").split())
        models = [m for m in models if m["model_id"] in target_ids]
        print(f"Filtered to {len(models)} models: {[m['model_id'] for m in models]}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    import concurrent.futures
    import threading

    print(f"\n{'='*60}")
    print(f"Starting generation. Max {args.workers} models active concurrently.")
    print("Each model single-threads its prompts (no double teaming).")
    print("Local models share a lock: only 1 local model active at a time.")
    print(f"{'='*60}")

    local_model_lock = threading.Lock()

    def run_model(mc):
        try:
            generate_for_model(
                prompts, 
                mc, 
                args.output_dir, 
                resume=not args.no_resume, 
                max_workers=args.max_workers_per_model,
                local_model_lock=local_model_lock
            )
        except CircuitBreakerException as e:
            print(f"\n[!] Skipping remaining prompts for {mc['model_id']}")
            print(f"[!] Reason: {e}")
        except Exception as e:
            print(f"\n[!] Unexpected error for {mc['model_id']}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_model, mc): mc for mc in models}
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print(f"\n{'='*60}")
    print("All models complete.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
