# Stage D Generation Status

Updated: 2026-05-01 13:32 America/Chicago.

Stage D was pivoted to maximize reuse of already-paid Stage C responses. The active prompt manifest is `data/stage_d/stage_d_prompts.jsonl`; the previous random-within-bin manifest and delta files were backed up to `data/stage_d/backups/reuse_pivot_20260430_134525/`.

Current prompt accounting:
- Total Stage D prompts: 5,000.
- Retained Stage C prompts: 2,246.
- New prompts requiring generation for paid/direct models: 2,754, except `google_gemini-3.1-pro-preview` has 2,759 missing because five retained Stage C rows are absent for that model.
- Selected prompt unit-test audit: 0 flagged prompts.

The local/open-weight generation delta is running from `data/stage_d/generation_delta/stage_d_new_prompts.jsonl` with one prompt at a time per model (`--max-workers-per-model 1`). Each model has its own A100-backed Azure Container Apps endpoint.

| Model ID | Endpoint app | Output file | PID | Log |
|---|---|---|---:|---|
| `qwen3.5-9b` | `qwen35-api` | `data/stage_d/generations/qwen3.5-9b.jsonl` | 116664 | `data/stage_d/logs/generate_stage_d_reuse_qwen3.5-9b.stdout.log` |
| `ministral-3-14b-reasoning` | `ministral-a100-api` | `data/stage_d/generations/ministral-3-14b-reasoning.jsonl` | 82148 | `data/stage_d/logs/generate_stage_d_reuse_ministral-3-14b-reasoning.stdout.log` |
| `mistral-small-2412` (`Devstral-Small-2505-Q4_K_M`) | `devstral-a100-api` | `data/stage_d/generations/mistral-small-2412.jsonl` | 116136 | `data/stage_d/logs/generate_stage_d_reuse_mistral-small-2412.stdout.log` |
| `gpt-oss-20b` | `gpt-oss20b-a100-api` | `data/stage_d/generations/gpt-oss-20b.jsonl` | 120192 | `data/stage_d/logs/generate_stage_d_reuse_gpt-oss-20b_retry.stdout.log` |
| `glm_4_7_flash_results` | `glm47-a100-api` | `data/stage_d/generations/glm_4_7_flash_results.jsonl` | 134712 | `data/stage_d/logs/generate_stage_d_reuse_glm_4_7_flash_results.stdout.log` |

Azure serverless generation was also restarted for the six no-direct-cost Azure study models, using 8 workers per model because prior rows are reusable and Azure is not charged directly to the project.

| Model ID | PID | Log |
|---|---:|---|
| `azure/kimi-k2.5` | 93756 | `data/stage_d/logs/generate_stage_d_reuse_azure_kimi-k2.5.stdout.log` |
| `azure/deepseek-v3.2-speciale` | 133204 | `data/stage_d/logs/generate_stage_d_reuse_azure_deepseek-v3.2-speciale.stdout.log` |
| `azure/mistral-large-3` | 137532 | `data/stage_d/logs/generate_stage_d_reuse_azure_mistral-large-3.stdout.log` |
| `azure/gpt-oss-120b` | 11836 | `data/stage_d/logs/generate_stage_d_reuse_azure_gpt-oss-120b.stdout.log` |
| `azure/llama-3.3-70b` | 116920 | `data/stage_d/logs/generate_stage_d_reuse_azure_llama-3.3-70b.stdout.log` |
| `azure/grok-3` | 103236 | `data/stage_d/logs/generate_stage_d_reuse_azure_grok-3.stdout.log` |

Copilot/subscription-backed generation was started for the two non-metered OpenAI models, using one worker per model. Both stopped at the GitHub Copilot 5-hour session limit before completing, then the 2026-05-01 restart hit the GitHub Copilot weekly rate limit. Those two remaining queues are now running through OpenAI Batch instead.

| Model ID | PID | Log |
|---|---:|---|
| `gpt-4.1` | stopped; OpenAI Batch fallback submitted | `data/stage_d/logs/generate_stage_d_copilot_restart_gpt-4.1_20260501_121634.stdout.log` |
| `gpt-5-mini` | stopped; OpenAI Batch fallback submitted | `data/stage_d/logs/generate_stage_d_copilot_restart_gpt-5-mini_20260501_121634.stdout.log` |
| `arcee-ai/trinity-large-preview:free` | complete | `data/stage_d/logs/generate_stage_d_paid_arcee-ai_trinity-large-preview_free.stdout.log` |
| `qwen/qwen3.6-plus` | complete | `data/stage_d/logs/generate_stage_d_dashscope_qwen_qwen3.6-plus.stdout.log` |
| `google/gemini-3-flash-preview` | complete | `data/stage_d/logs/generate_stage_d_openrouter_google_gemini-3-flash-preview*.stdout.log` |

OpenAI batch jobs:

| Model ID | Batch ID | State file |
|---|---|---|
| `openai/gpt-5.4` | `batch_69f40377679081908db7de85ac4b4d72` | completed and retrieved |
| `gpt-4.1` | `batch_69f4f19e05b881909e1fe52206d467e9` | `data/stage_d/batch_state/gpt-4.1.json` |
| `gpt-5-mini` | `batch_69f4f19e56c08190a12222487749ba59` | `data/stage_d/batch_state/gpt-5-mini.json` |

Anthropic batch jobs:

| Model ID | Batch ID | State |
|---|---|---|
| `anthropic/claude-sonnet-4.6` | `msgbatch_01PnDdFbD8oM751SPynq2KMb` | completed and retrieved |
| `anthropic/claude-opus-4.6` | `msgbatch_01WXdz3FLdYSmh82ZCTRAWkg` | completed and retrieved |
| `anthropic/claude-opus-4.7` | `msgbatch_01XdpYyiaEJjGoFLmsTTzpAb` | completed and retrieved |

Notes:
- Batch-first generation is now the rule for provider paths that support it; see `docs/stage_d_batch_generation.md`.
- `gpt-oss-20b` was relaunched after setting server-side reasoning off on the A100 app. It produced intermittent empty final responses, so `src/data_provenance/02_generate_solutions.py` now retries empty/code-less responses before writing an `EMPTY_CODE` error. Existing error rows remain in the JSONL file but are requeued on restart.
- `arcee-ai/trinity-large-preview:free` is kept as the internal model ID for continuity with Stage C filenames, but `api_model` now uses the paid OpenRouter route `arcee-ai/trinity-large-preview` because the `:free` endpoint returned `404 No endpoints found`. The paid route produced valid rows on restart. Existing `:free` 404 error rows remain in the JSONL file but are requeued because they have no usable code.
- `qwen/qwen3.6-plus` was briefly started through OpenRouter, then stopped before any rows were written. Alibaba/Qwen Batch was attempted through DashScope with `qwen3.6-plus` and `qwen3.6-plus-2026-04-02`; both failed validation with `model_not_found` before processing requests. The active run is now first-party DashScope realtime (`DASHSCOPE_API_KEY`, `qwen3.6-plus`) with `enable_thinking=false`.
- `openai/gpt-5.4` completed through OpenAI Batch in 2,754/2,754 requests with 0 failed. Results were retrieved to `data/stage_d/generations/openai_gpt-5.4.jsonl`; Stage D coverage is 5,000/5,000 with 0 empty/error batch rows.
- `gpt-4.1` Copilot fallback has 3,289/5,000 Stage D prompts covered before the OpenAI Batch fallback. The submitted batch contains 1,711 remaining prompts from `data/stage_d/generation_delta/stage_d_new_prompts.jsonl`.
- `gpt-5-mini` Copilot fallback has 2,920/5,000 Stage D prompts covered before the OpenAI Batch fallback. The submitted batch contains 2,080 remaining prompts from `data/stage_d/generation_delta/stage_d_new_prompts.jsonl`.
- `google/gemini-3-flash-preview` completed through OpenRouter. Several malformed/empty responses were recovered with targeted direct retries; final Stage D coverage is 5,000/5,000.
- `gpt-oss-20b` completed after multiple targeted direct non-streaming recovery passes for streamed empty-response rows. Final Stage D coverage is 5,000/5,000; recovery metadata is written on appended rows and summarized in `data/stage_d/logs/targeted_recovery_report.json`, `targeted_recovery_gpt_oss_remaining_report.json`, and `targeted_recovery_gpt_oss_final5_report.json`.
- The original `ministral-api` and `mistral-small-api` apps in `managedEnvironment-GPU` were scaled back to zero because they were degraded with `WorkLoad Profile Full`. The active Mistral-family runs are using the dedicated A100 apps above.
- All active queues append directly to `data/stage_d/generations/`; do not start duplicate generators for these model IDs unless the current PID is stopped first.
- `data/stage_d/generation_delta/per_model_retained_counts.csv` counts only reusable Stage C rows. For Azure and A100 models, additional partial Stage D rows already exist in `data/stage_d/generations/` and are skipped automatically by the generator when their prompt IDs are in the active delta file.
