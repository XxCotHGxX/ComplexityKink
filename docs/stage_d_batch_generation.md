# Stage D Batch-First Generation Policy

Stage D generation should use provider batch APIs whenever the provider path
used in the experiment supports asynchronous batch inference. Realtime
generation is a fallback, not the default, for paid/API providers.

## Hard Rule

Do not run `src/data_provenance/02_generate_solutions.py` for a model whose
provider path has a working batch API unless the batch submission is rejected
for that exact model or account. If that happens, record the rejection reason
and use realtime generation only for the missing rows.

Anthropic is stricter: Claude/Sonnet/Opus panel generation is batch-only, and
Anthropic batches require explicit prior approval from the project owner before
submission. Do not use realtime Anthropic generation as a fallback for Stage D
or the 24-bin follow-up; retry with another Anthropic Message Batch instead.

## Batch-Capable Paths

| Provider path | Panel models | Batch helper | Status |
| --- | --- | --- | --- |
| OpenAI first-party API | `openai/gpt-5.4` | `scripts/openai_batch.py` | Use batch |
| Anthropic first-party API | `anthropic/claude-sonnet-4.6`, `anthropic/claude-opus-4.6`, `anthropic/claude-opus-4.7` | `scripts/anthropic_batch.py` | Batch only; no realtime fallback |
| Google Gemini first-party API | Gemini models when available through `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `scripts/gemini_batch.py` | Use batch if the exact model is available first-party |
| Alibaba DashScope / Qwen first-party API | `qwen/qwen3.6-plus` | `scripts/qwen_batch.py` | Try batch first; fallback only if DashScope rejects the model for batch |
| Azure OpenAI global batch deployments | Any future Azure OpenAI deployment with `globalbatch` SKU | Not currently implemented for Stage D | Supported by provider, but not used by the current Azure MaaS panel |

## Realtime-Only Paths In The Current Panel

| Provider path | Panel models | Reason |
| --- | --- | --- |
| GitHub Copilot | `gpt-4.1`, `gpt-5-mini` | Copilot API path used here does not expose a batch job API. |
| OpenRouter | `arcee-ai/trinity-large-preview:free`; Gemini models if routed through OpenRouter | OpenRouter's documented API path is realtime chat/completions, not provider batch jobs. Prefer first-party Gemini batch when available. |
| Local/A100 OpenAI-compatible endpoints | `gpt-oss-20b`, `qwen3.5-9b`, `ministral-3-14b-reasoning`, `mistral-small-2412`, `glm_4_7_flash_results` | These are self-hosted realtime endpoints. Keep one model active at a time and queue prompts continuously. |
| Azure AI Inference / MaaS serverless | `azure/kimi-k2.5`, `azure/deepseek-v3.2-speciale`, `azure/mistral-large-3`, `azure/gpt-oss-120b`, `azure/llama-3.3-70b`, `azure/grok-3` | The current configs use Azure AI Inference serverless endpoints, not Azure OpenAI `globalbatch` deployments. |

## Stage D Commands

OpenAI:

```powershell
python scripts/openai_batch.py `
  --prompts data/stage_d/generation_delta/stage_d_new_prompts.jsonl `
  --gen-dir data/stage_d/generations `
  --state-dir data/stage_d/batch_state `
  --request-dir data/stage_d/batch_requests `
  --result-dir data/stage_d/batch_results `
  submit --model openai/gpt-5.4 --api-model gpt-5.4 --reasoning
```

Anthropic 24-bin follow-up:

```powershell
python scripts/anthropic_batch.py `
  --gen-dir data/stage_d_24bin_equal/generations `
  --state-dir data/stage_d_24bin_equal/batch_state `
  --request-dir data/stage_d_24bin_equal/batch_requests `
  --result-dir data/stage_d_24bin_equal/batch_results `
  queue --approved --poll-seconds 300 --queue-models `
  anthropic/claude-opus-4.6 anthropic/claude-opus-4.7
```

Gemini first-party:

```powershell
python scripts/gemini_batch.py `
  --prompts data/stage_d/generation_delta/per_model_missing/google_gemini-3-flash-preview.jsonl `
  --gen-dir data/stage_d/generations `
  --state-dir data/stage_d/batch_state `
  --request-dir data/stage_d/batch_requests `
  --result-dir data/stage_d/batch_results `
  submit --model google/gemini-3-flash-preview --api-model gemini-3-flash
```

Qwen/DashScope:

```powershell
python scripts/qwen_batch.py submit
```

If a batch-capable provider rejects the exact panel model, keep the generated
request file and state/error output in `data/stage_d/batch_requests`,
`data/stage_d/batch_state`, or `data/stage_d/batch_results` so the fallback is
auditable.
