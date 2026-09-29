# License and terms-of-use notes for the public release

These notes are for the authors' review before upload. They list what each part
of the release derives from and which third-party terms may apply. **Nothing
here is a legal conclusion.** The license in `README.md` (`cc-by-4.0`) and in
`croissant.json` is a placeholder until the authors decide.

## 1. Source data: OpenCodeInstruct

- Used for: `prompt_text`, `unit_tests`, `prompt_id` (the source `id`),
  `source_reference_avg_test_score`, and `reference_cc` (computed from the
  source reference solution; the solution itself is not redistributed), in
  `prompts`, `extension_prompts`, `paraphrase_prompts.original_prompt_text`, and
  (as input to rewriting) `paraphrase_prompts` / `cross_language_prompts`.
- License: **CC BY 4.0**. Verified on the Hugging Face dataset card
  (`nvidia/OpenCodeInstruct`, "This dataset is licensed under the Creative
  Commons Attribution 4.0 International License"), which also marks it ready for
  commercial and non-commercial use and asks users to cite Ahmad et al., 2025
  (arXiv:2504.04030). The manuscript's statement (CC BY 4.0) matches.
- CC BY 4.0 obligations to satisfy in the card: attribution to NVIDIA, a link
  to the license, and an indication of changes (we select a subset, parse
  `unit_tests` into a list, and add derived fields). The current `README.md`
  names the source and license; add an explicit "changes made" sentence if you
  keep CC BY 4.0 for the compilation.
- The dataset card says its content is LLM-generated but does not name the
  generating models, so no further upstream model terms are identified.

## 2. Tools

- Lizard 1.21.0 (MIT) computed `reference_cc` and `output_cc_lizard`. Only its
  numeric outputs are released.

## 3. Model outputs included in the release

Every model below contributed text or labels that appear in the release. The
"route" column is whose service terms governed the request, which can differ
from the model developer's own license. Please check, for each row, (a) the
route's terms on publishing outputs, (b) any model-license clauses that follow
outputs (for example attribution or naming requirements, or limits on using
outputs to train other models), and (c) whether the account type (API,
enterprise, free tier, consumer subscription) changes those terms.

| `model_key` | Model | Developer | Route(s) recorded | Released outputs |
|---|---|---|---|---|
| `anthropic_claude-opus-4.6` | Claude Opus 4.6 | Anthropic | Anthropic API (Message Batches for later frame) | generation code + outcomes |
| `anthropic_claude-opus-4.7` | Claude Opus 4.7 | Anthropic | Anthropic API (Message Batches) | generation code + outcomes |
| `anthropic_claude-sonnet-4.6` | Claude Sonnet 4.6 | Anthropic | Anthropic API (Message Batches) | generation code + outcomes |
| `arcee-ai_trinity-large-preview_free` | Trinity Large (preview) | Arcee AI | OpenRouter, free tier | generation code + outcomes |
| `azure_deepseek-v3.2-speciale` | DeepSeek V3.2 | DeepSeek | Azure AI Foundry | generation code + outcomes; extension and pass@k outcomes/code |
| `azure_gpt-oss-120b` | gpt-oss-120b | OpenAI (open weights) | Azure AI Foundry; OpenRouter also in older configs | generation code + outcomes; extension; pass@k |
| `azure_grok-3` | Grok 3 | xAI | Azure AI Foundry; OpenRouter also in older configs | generation code + outcomes |
| `azure_kimi-k2.5` | Kimi K2.5 | Moonshot AI | Azure AI Foundry | generation code + outcomes; extension; pass@k |
| `azure_llama-3.3-70b` | Llama 3.3 70B Instruct | Meta | Azure AI Foundry; OpenRouter also in older configs | generation code + outcomes; extension; pass@k |
| `azure_mistral-large-3` | Mistral Large 3 | Mistral AI | Azure AI Foundry; OpenRouter also in older configs | generation code + outcomes; extension |
| `glm_4_7_flash_results` | GLM-4.7-Flash (Q8_0 GGUF) | Zhipu AI (Z.ai) | Local inference | generation code + outcomes |
| `google_gemini-3-flash-preview` | Gemini 3 Flash (preview) | Google | Gemini API batch mode; OpenRouter | generation code + outcomes |
| `google_gemini-3.1-pro-preview` | Gemini 3.1 Pro (preview) | Google | Gemini API batch mode; OpenRouter | generation code + outcomes |
| `gpt-4.1` | GPT-4.1 | OpenAI | GitHub Copilot model endpoint (earlier frame, per config); OpenAI Batch API (later frame) | generation code + outcomes |
| `gpt-5-mini` | GPT-5 mini | OpenAI | GitHub Copilot model endpoint (earlier frame, per config); OpenAI Batch API (later frame) | generation code + outcomes |
| `gpt-oss-20b` | gpt-oss-20b (Q4_K_M GGUF) | OpenAI (open weights) | Local inference | generation code + outcomes |
| `ministral-3-14b-reasoning` | Ministral 3 14B Reasoning (Q4_K_M GGUF) | Mistral AI | Local inference | generation code + outcomes |
| `mistral-small-2412` | Served as Devstral-Small-2505 (Q4_K_M GGUF) | Mistral AI | Local inference | generation code + outcomes |
| `openai_gpt-5.4` | GPT-5.4 | OpenAI | OpenAI Batch API | generation code + outcomes |
| `qwen3.5-9b` | Qwen3.5 9B (Q4_K_M GGUF) | Alibaba (Qwen) | Local inference | generation code + outcomes |
| `qwen_qwen3.6-plus` | Qwen3.6 Plus | Alibaba Cloud | Alibaba Cloud Model Studio (DashScope) batch; OpenRouter in older configs | generation code + outcomes |
| `azure_grok-4-20-non-reasoning` | Grok 4.20 (non-reasoning) | xAI | Azure AI Foundry | extension outcomes only (no code) |
| `cli_claude-opus-4.6` | Claude Opus 4.6 | Anthropic | **Claude Code CLI on a consumer subscription** and OpenRouter | fixed-version outcomes only (no code) |
| `cli_gpt-5.4` | GPT-5.4 | OpenAI | **Codex CLI (ChatGPT subscription)** | fixed-version outcomes only (no code) |
| `cli_gemini-3.1-pro` | Gemini 3.1 Pro (preview) | Google | OpenRouter | fixed-version outcomes only (no code) |
| `judge:o4-mini` | o4-mini | OpenAI | Azure AI Foundry (Azure OpenAI) | rubric scores, preliminary scores, task-type labels, earlier-frame harness-audit verdicts |
| `judge:gpt-5.5` | GPT-5.5 | OpenAI | Azure AI Foundry (Azure OpenAI) | rubric scores, task-type labels |
| `judge:llama-4-maverick` | Llama 4 Maverick | Meta | Azure AI Foundry | rubric scores, task-type labels |
| `judge:command-a` | Command A | Cohere | Azure AI Foundry | rubric scores, task-type labels |
| `aux:deepseek-v3.2-rewriter` | DeepSeek V3.2 | DeepSeek | Azure AI Foundry | paraphrased prompts; Java and C++ re-expressions (full text) |

Items that deserve particular attention (flags, not conclusions):

1. **Consumer-subscription routes.** The fixed-version check used the Claude
   Code CLI on a subscription and the Codex CLI on a ChatGPT subscription. Only
   pass rates are released from these runs (no code), but confirm that the
   applicable consumer terms allow publishing results.
2. **GitHub Copilot route.** The run configuration lists a Copilot model
   endpoint for GPT-4.1 and GPT-5-mini in the earlier frame. Confirm which
   route actually produced those rows and whether Copilot's terms allow
   redistributing the generated code.
3. **Free-tier route.** Trinity Large was queried through an OpenRouter free
   tier; check OpenRouter's and Arcee's terms for free-tier outputs.
4. **Model-license clauses that travel with outputs.** Some open-weight
   licenses (for example the Llama community licenses for Llama 3.3 and Llama 4
   Maverick) contain attribution or naming conditions tied to use of outputs;
   check each open-weight license (DeepSeek, gpt-oss, Kimi, Mistral/Devstral,
   GLM, Qwen, Trinity, Command A) for similar clauses, and whether Command A's
   weights license (reported as non-commercial) is relevant to judge scores
   obtained through Azure.
5. **Restrictions on training competing models.** Several providers' terms
   restrict using outputs to develop competing models. The card already lists
   training on this data as out of scope; decide whether the release license
   should say so explicitly (for example, a notice that model outputs remain
   subject to the originating providers' terms).
6. **Full-text rewrites.** `paraphrase_prompts` and `cross_language_prompts`
   contain complete DeepSeek-V3.2 rewrites of CC BY 4.0 prompts; they are
   derivative text of both the source prompt and the model output.
7. **Human grades.** `human_calibration` contains the pseudonymous grades of a
   second research-team member (`grader_2`). Confirm their consent to public
   release; the source files identified graders by name and the release does
   not.
8. **Compilation license.** Options to consider include CC BY 4.0 for the
   compilation (matching the source) with a notice that model outputs remain
   subject to their providers' terms, or a more restrictive license if any
   provider term requires it. This choice must also be entered in the Croissant
   `license` field and the Dataverse terms tab.
