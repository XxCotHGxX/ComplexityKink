# Model Reference - Complexity Kink Cross-Validation Experiment

> **Experiment Date:** February 2026  
> **Prompt Set:** 5,000 stratified samples from the NVIDIA OpenCodeInstruct dataset  
> **Task:** Generate Python solutions to programming prompts, then measure cyclomatic complexity and pass-rate for two-stage instrumental-variables analysis.

---

## Summary Table

| Model ID | Provider | Tier | Architecture | Parameters | Context (tokens) | Release | SWE-bench Verified |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | OpenAI | Efficient | Dense Transformer | Undisclosed | 1,000,000 | Apr 2025 | 54.6% |
| gpt-5-mini | OpenAI | Efficient | Dense Transformer | Undisclosed | 272,000 | Aug 2025 | - |
| gpt-5.2 | OpenAI | Standard | Dense Transformer | Undisclosed | 400,000 | Dec 2025 | - |
| gpt-5.1-codex-max | OpenAI | Frontier | Dense Transformer | Undisclosed | 1,000,000+ | Nov 2025 | 77.9% |
| gpt-5.3-codex | OpenAI | Frontier | Dense Transformer | Undisclosed | 128,000 | Feb 2026 | - |
| claude-haiku-4.5 | Anthropic | Efficient | Dense Transformer | Undisclosed | 200,000 | Oct 2025 | - |
| claude-sonnet-4.6 | Anthropic | Frontier | Dense Transformer | Undisclosed | 200,000 (1M beta) | Feb 2026 | - |
| claude-opus-4.6 | Anthropic | Frontier | Dense Transformer | Undisclosed | 200,000 (1M beta) | Feb 2026 | - |
| claude-opus-4.6-fast | Anthropic | Frontier | Dense Transformer | Undisclosed | 200,000 (1M beta) | Feb 2026 | - |
| grok-code-fast-1 | xAI | Efficient | Sparse MoE | ~314B | 256,000 | Aug 2025 | 70.8% |
| gemini-3-pro-high | Google | Frontier | Dense Transformer | Undisclosed | 1,000,000 | Nov 2025 | - |
| gemini-3-flash | Google | Efficient | Dense Transformer | Undisclosed | 1,000,000 | Dec 2025 | - |
| gpt-oss-20b | OpenAI (OSS) | Open-Source | Dense Transformer | 20B | 32,768 | 2025 | - |
| qwen3-14b | Alibaba (Qwen) | Open-Source | Dense Transformer | 14.8B | 32,768 (131K ext.) | 2025 | - |
| ministral-3-14b-reasoning | Mistral AI | Open-Source | Dense Transformer | ~14B | 32,768 | 2025 | - |
| trinity-large-preview | Arcee AI | Open-Source | Sparse MoE (4/256) | 398B total / 13B active | 128,000 | 2026 | - |

---

## Detailed Model Profiles

### Closed-Source Frontier Models

#### GPT-4.1 - OpenAI
- **Released:** April 14, 2025
- **Context Window:** 1,000,000 tokens
- **Strengths:** First GPT model with 1M context; optimized for coding with 54.6% SWE-bench Verified; reduced extraneous code edits from 9% to 2% vs. GPT-4o.
- **Role in Experiment:** Legacy baseline. Provides a pre-GPT-5 comparison point at zero Copilot quota cost.

#### GPT-5-mini - OpenAI
- **Released:** August 7, 2025
- **Context Window:** 272,000 tokens
- **Strengths:** Faster, cost-efficient variant of GPT-5; supports streaming, function calling, structured outputs. Suitable for well-defined tasks with precise prompts.
- **Role in Experiment:** Efficient-tier GPT-5 representative.

#### GPT-5.2 - OpenAI
- **Released:** December 11, 2025
- **Context Window:** 400,000 input / 128,000 output tokens
- **Strengths:** Integrated reasoning with auto-adjusting depth; 97% tool-calling accuracy. Available in Instant, Standard, and Thinking variants.
- **Role in Experiment:** Standard-tier generation model. Represents the midpoint between efficiency and frontier capability.

#### GPT-5.1-Codex-Max - OpenAI
- **Released:** November 19, 2025
- **Context Window:** 1,000,000+ tokens (multi-context compaction)
- **SWE-bench Verified:** 77.9%
- **Strengths:** Agentic coding model for project-scale work. Native multi-context window processing via "compaction." Optimized for Windows environments.
- **Role in Experiment:** Top-tier OpenAI coding model. Tests whether peak capability shifts the complexity kink threshold.

#### GPT-5.3-Codex - OpenAI
- **Released:** February 5, 2026
- **Context Window:** 128,000 tokens
- **Strengths:** OpenAI's newest agentic coding model. Combines GPT-5.2-Codex coding performance with GPT-5.2 reasoning. ~25% faster than predecessor. Described by OpenAI as "instrumental in creating itself."
- **Role in Experiment:** Newest frontier model in the experiment. Tests the absolute cutting edge of code generation.

---

#### Claude Haiku 4.5 - Anthropic
- **Released:** October 15, 2025
- **Context Window:** 200,000 input / 64,000 output tokens
- **Strengths:** Fastest Anthropic model. First Haiku with extended thinking, computer use, and context awareness. Performance comparable to Sonnet 4 at 1/3 the cost and 2x the speed.
- **Role in Experiment:** Efficient-tier Anthropic representative. Tests whether speed-cost tradeoffs affect solution complexity.

#### Claude Sonnet 4.6 - Anthropic
- **Released:** February 17, 2026
- **Context Window:** 200,000 tokens (1M beta) / 64,000 output tokens
- **Strengths:** Default model for claude.ai. Adaptive thinking by default. Comprehensive upgrades across coding, computer use, long-context reasoning, and agent planning. Improved prompt injection resistance.
- **Role in Experiment:** Mid-frontier Anthropic model. Provides a contrast point between Haiku 4.5 and Opus 4.6.

#### Claude Opus 4.6 - Anthropic
- **Released:** February 4, 2026
- **Context Window:** 200,000 tokens (1M beta) / 128,000 output tokens
- **Strengths:** Anthropic's most powerful model. 83% improvement on ARC-AGI-2 reasoning tasks vs. Opus 4.5. Excels in large codebases, complex refactors, and multi-step debugging.
- **Role in Experiment:** Peak Anthropic capability. Direct comparison to GPT-5.3-Codex and Gemini-3-Pro.

#### Claude Opus 4.6 Fast - Anthropic
- **Released:** February 2026
- **Context Window:** 200,000 tokens (1M beta) / 128,000 output tokens
- **Strengths:** Speed-optimized variant of Opus 4.6 with similar capability but reduced latency.
- **Role in Experiment:** Tests whether latency optimization (which may involve internal routing) affects solution quality patterns.

---

#### Grok Code Fast 1 - xAI
- **Released:** Late August 2025
- **Architecture:** Sparse Mixture-of-Experts, ~314B parameters
- **Context Window:** 256,000 tokens
- **SWE-bench Verified:** 70.8%
- **Throughput:** ~90-100 tokens/second
- **Strengths:** Purpose-built for agentic coding. Proficient in TypeScript, Python, Java, Rust, C++, Go. Real-time reasoning trace visibility. Extremely cost-effective ($0.20/M input tokens).
- **Role in Experiment:** xAI's specialist coding model. Tests the hypothesis that coding-optimized architecture alters the complexity-performance relationship.

---

#### Gemini 3 Pro (High) - Google
- **Released:** November 17, 2025
- **Context Window:** 1,000,000 tokens / 64,000 output tokens
- **Strengths:** Natively multimodal. Dynamic thinking by default. Advanced reasoning across text, images, audio, and video. Pricing at $2.00/M input, $12.00/M output.
- **Role in Experiment:** Google frontier representative.

#### Gemini 3 Flash - Google
- **Released:** December 17, 2025
- **Context Window:** 1,000,000 tokens / 65,536 output tokens
- **Strengths:** Speed-optimized near-Pro performance. Configurable reasoning via "thinking levels." Superior to Gemini 3 Pro on SWE-bench Verified. Extremely cost-effective ($0.50/M input tokens).
- **Role in Experiment:** Google efficient-tier. Tests whether Flash's speed optimizations produce qualitatively different code patterns.

---

### Open-Source / Locally Hosted Models

#### GPT-OSS-20B - OpenAI (Open-Source)
- **Parameters:** 20 billion
- **Context Window:** 32,768 tokens
- **Hosting:** Local or self-hosted OpenAI-compatible endpoint
- **Role in Experiment:** Small open-weight OpenAI model. Tests whether the complexity kink exists even in compact architectures with no API quota constraints.

#### Qwen3-14B - Alibaba Cloud (Qwen Team)
- **Parameters:** 14.8B total (13.2B non-embedding)
- **Context Window:** 32,768 native / 131,072 with YaRN extension
- **Architecture:** Dense causal decoder-only transformer. Grouped Query Attention (40 Q heads, 8 KV heads). SwiGLU activations, RMSNorm. Hybrid thinking/non-thinking mode.
- **Training Data:** 119 languages, three-stage pretraining
- **Hosting:** Local via LM Studio
- **Role in Experiment:** Non-Western open-source model. Tests cross-vendor generalizability of the complexity kink.

#### Ministral-3-14B-Reasoning - Mistral AI
- **Parameters:** ~14B
- **Context Window:** 32,768 tokens
- **Architecture:** Dense causal decoder-only transformer with enhanced reasoning capabilities
- **Hosting:** Local via LM Studio
- **Role in Experiment:** European open-source model with reasoning focus. Provides vendor diversity alongside Qwen and GPT-OSS.

#### Trinity Large Preview - Arcee AI
- **Parameters:** 398B total / 13B active per token
- **Architecture:** Sparse Mixture-of-Experts (4-of-256 expert routing). Gated attention, interleaved local/global attention, SMEBU load balancing.
- **Context Window:** 128,000 tokens (preview API); native 512K, tested to 1M
- **Training Data:** 17 trillion tokens on 2,048 NVIDIA B300 GPUs
- **Hosting:** Remote via OpenRouter (free tier)
- **Role in Experiment:** Largest open-weight model in the experiment by total parameter count. Tests whether massive MoE capacity with few active parameters changes the kink dynamics.

---

## Authentication Pipeline

| Backend | Models | Auth Mechanism | Quota Pool |
|---|---|---|---|
| GitHub Copilot | GPT-4.1, GPT-5-mini, GPT-5.2, GPT-5.1-Codex-Max, GPT-5.3-Codex, Claude Haiku 4.5, Claude Sonnet 4.6, Claude Opus 4.6, Claude Opus 4.6 Fast, Grok Code Fast 1 | OAuth token exchange (ghu_ -> session token) | Shared Copilot pool |
| Google Antigravity | Gemini 3 Pro High, Gemini 3 Flash | Moltbot OAuth refresh -> Cloud Code Assist | Google quota |
| LM Studio (Local) | GPT-OSS-20B, Qwen3-14B, Ministral-3-14B-Reasoning | OpenAI-compatible endpoint (no auth) | Unlimited |
| OpenRouter | Trinity Large Preview | API key (via environment variable) | Free tier |

---

## Experimental Design Rationale

The model selection was designed to maximize **cross-vendor, cross-tier, cross-architecture diversity** while remaining practically feasible within available API quotas:

1. **Vendor Diversity (6 vendors):** OpenAI, Anthropic, Google, xAI, Alibaba/Qwen, Mistral AI, Arcee AI
2. **Tier Stratification:** Efficient -> Standard -> Frontier -> Open-Source
3. **Architecture Variation:** Dense transformers vs. Sparse MoE (Grok, Trinity)
4. **Scale Range:** 14B parameters (Qwen3, Ministral) to 398B parameters (Trinity)
5. **Temporal Coverage:** April 2025 (GPT-4.1) through February 2026 (GPT-5.3-Codex, Claude Sonnet 4.6)

This design enables the cross-model analysis (`04_cross_model_analysis.py`) to test whether the complexity kink threshold is a universal property of LLM code generation or an artifact specific to particular model families, scales, or architectures.
