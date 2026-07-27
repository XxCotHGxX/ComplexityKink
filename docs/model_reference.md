# Evaluated model panel

The locked Stage D analysis evaluates 21 model versions. Model identifiers are
recorded as experiment identifiers so that result files and generation metadata
can be joined without relying on display names.

| Experiment ID | Display name | Provider route |
| :-- | :-- | :-- |
| `anthropic_claude-opus-4.6` | Claude Opus 4.6 | Anthropic |
| `anthropic_claude-opus-4.7` | Claude Opus 4.7 | Anthropic |
| `anthropic_claude-sonnet-4.6` | Claude Sonnet 4.6 | Anthropic |
| `arcee-ai_trinity-large-preview_free` | Trinity-large | OpenRouter |
| `azure_deepseek-v3.2-speciale` | DeepSeek V3.2 | Azure AI |
| `azure_gpt-oss-120b` | GPT-OSS-120B | Azure AI |
| `azure_grok-3` | Grok-3 | Azure AI |
| `azure_kimi-k2.5` | Kimi K2.5 | Azure AI |
| `azure_llama-3.3-70b` | Llama 3.3-70B | Azure AI |
| `azure_mistral-large-3` | Mistral Large-3 | Azure AI |
| `glm_4_7_flash_results` | GLM 4.7-flash | Local or hosted worker |
| `google_gemini-3-flash-preview` | Gemini 3 Flash | Google |
| `google_gemini-3.1-pro-preview` | Gemini 3.1 Pro Preview | Google |
| `gpt-4.1` | GPT-4.1 | OpenAI |
| `gpt-5-mini` | GPT-5-mini | OpenAI |
| `gpt-oss-20b` | GPT-OSS-20B | Local or hosted worker |
| `ministral-3-14b-reasoning` | Ministral-3-14B-reasoning | Local or hosted worker |
| `mistral-small-2412` | Mistral Small 2412 | Local or hosted worker |
| `openai_gpt-5.4` | GPT-5.4 | OpenAI |
| `qwen_qwen3.6-plus` | Qwen 3.6 Plus | Alibaba DashScope |
| `qwen3.5-9b` | Qwen 3.5-9B | Local or hosted worker |

## Version discipline

The experiment IDs above are the panel used for the locked 5,000-prompt
analysis. A provider route or local deployment can change without changing the
model, but substituting a different model version changes the experiment.

Public configuration files therefore use environment-variable placeholders and
generic endpoints. The private run configuration, deployment names, and
credential discovery helpers are intentionally not part of the repository.

The exact-version frontier extension uses:

- Claude Opus 4.6
- GPT-5.4
- Gemini 3.1 Pro Preview

Those models were chosen because they are members of the submitted panel. The
extension does not silently replace them with newer versions.

## Judge separation

The four rubric judges are outside the 21-model evaluation panel. Their shared
job is to score intended solution structure from the prompt, before any
evaluated model generates code. Their scores are averaged at the
prompt-dimension level and retained separately for inter-rater diagnostics.
