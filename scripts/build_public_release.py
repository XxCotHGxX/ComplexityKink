#!/usr/bin/env python
"""Build the public Complexity Kink dataset release (local build only).

This script assembles the Hugging Face / Harvard Dataverse release of the data
behind "The Complexity Kink" (NeurIPS 2026 Evaluations & Datasets track) from
the retained raw bundle. It never uploads anything and never contacts a
network service. It reads only from ``<data-root>/data`` and writes:

  * the release tree (Parquet configs, dataset card, docs, local Croissant)
    to ``--out`` (default ``<data-root>/data/public_release``);
  * ``release/build_report.json`` (sanity checks, sanitization scan) and the
    auto-generated field tables inside ``release/README.md``.

Source-of-truth rules (see release/README.md, "How the data was built"):

  * Main benchmark = ``data/stage_d/scored_independent_audit/*.jsonl`` (the
    camera-ready primary outcome, written by
    ``src/audit/06_apply_independent_audit.py``; every row also carries the raw
    harness value and the reviewed version's value) joined to
    ``data/stage_d/ensemble_scores_current_aggregated.jsonl``. The stale
    ``ensemble_scores_aggregated.jsonl`` is never used; the join must yield
    exactly 5,000 prompts.
  * Audit records come from ``data/independent_audit/`` (known-answer sets and
    responses under ``pilot/`` and ``confirmation/``; production verdicts under
    ``production/``). Verdicts are counted with the same completeness rule as
    the analysis (``final_verdict`` in ``src/audit/independent_audit.py``).
  * The output-CC-mined 24-bin "equal-support" set
    (``data/stage_d_24bin_equal``) is excluded. The 365-prompt audit-clean
    extension is ``data/rebuttal/tail_topup/tail_topup_final.jsonl``.
  * Raw API responses, request/batch ids, deployment names, endpoints, keys,
    and local paths are never copied.

Usage (from the repository root, with the project virtual environment):
    python scripts/build_public_release.py --data-root <directory containing data/>

``--data-root`` is auto-detected when this checkout lives inside the directory
that holds the retained ``data/`` bundle.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = REPO_ROOT / "release"
sys.path.insert(0, str(REPO_ROOT / "src" / "audit"))
from independent_audit import INCOMPLETE_FINISH, final_verdict  # noqa: E402

PRIMARY_AUDITOR = "auditor:mimo-v2.6-pro"
# (auditor key, file under data/independent_audit/<set dir>/) for the known-answer sets.
KNOWN_ANSWER_RUNS = {
    "A_selection": ("pilot", [
        ("auditor:phi-4-reasoning", "audit_phi4_reasoning.jsonl"),
        ("auditor:mai-thinking-1", "audit_mai_thinking_1.jsonl"),
        ("auditor:nemotron-3-ultra", "audit_or_nemotron3_ultra_venice.jsonl"),
        ("auditor:laguna-s-2.1", "audit_or_laguna_s21_poolside.jsonl"),
        (PRIMARY_AUDITOR, "audit_or_mimo_v26_pro_gmicloud.jsonl"),
        ("judge:o4-mini", "audit_o4mini_reviewed_settings.jsonl"),
    ]),
    "B_confirmation": ("confirmation", [
        (PRIMARY_AUDITOR, "audit_or_mimo_v26_pro_gmicloud.jsonl"),
    ]),
}
PRODUCTION_RUNS = [
    (PRIMARY_AUDITOR, "audit_mimo_v26_pro.jsonl"),
    ("auditor:mai-thinking-1", "audit_mai_thinking_1.jsonl"),
    ("auditor:phi-4-reasoning", "audit_phi4_reasoning.jsonl"),
]

DIMS = ["branching", "iteration", "state", "data_structures", "edge_cases", "composition"]
EXPECTED_MAIN_PROMPTS = 5000
EXPECTED_MODELS = 21
EXPECTED_JUDGE_ROWS = 19997
EXPECTED_EXTENSION = 365
PARQUET_COMPRESSION = "zstd"
DATAVERSE_FILE_LIMIT = int(2.5 * 1024**3)
SPLIT = "test"

JUDGE_FILES = {
    "o4-mini": "ensemble_scores_current_azure_o4_mini.jsonl",
    "gpt-5.5": "ensemble_scores_current_azure_gpt_5_5.jsonl",
    "llama-4-maverick": "ensemble_scores_current_azure_llama_4_maverick.jsonl",
    "command-a": "ensemble_scores_current_azure_cohere_command_a.jsonl",
}
SCORER_TO_JUDGE = {
    "azure_o4_mini": "o4-mini",
    "azure_gpt_5_5": "gpt-5.5",
    "azure_llama_4_maverick": "llama-4-maverick",
    "azure_cohere_command_a": "command-a",
}
FRAME_NAMES = {
    "stage_c_existing": "earlier_retained",
    "stage_d_candidate": "later_candidate",
}
DISPLAY_NAMES = {
    "anthropic_claude-opus-4.6": "Claude Opus 4.6",
    "anthropic_claude-opus-4.7": "Claude Opus 4.7",
    "anthropic_claude-sonnet-4.6": "Claude Sonnet 4.6",
    "arcee-ai_trinity-large-preview_free": "Trinity-large",
    "azure_deepseek-v3.2-speciale": "DeepSeek V3.2",
    "azure_gpt-oss-120b": "GPT-OSS-120B",
    "azure_grok-3": "Grok-3",
    "azure_kimi-k2.5": "Kimi K2.5",
    "azure_llama-3.3-70b": "Llama 3.3-70B",
    "azure_mistral-large-3": "Mistral Large-3",
    "glm_4_7_flash_results": "GLM 4.7-flash",
    "google_gemini-3-flash-preview": "Gemini 3 Flash",
    "google_gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gpt-4.1": "GPT-4.1",
    "gpt-5-mini": "GPT-5-mini",
    "gpt-oss-20b": "GPT-OSS-20B",
    "ministral-3-14b-reasoning": "Ministral-3-14B-reasoning",
    "mistral-small-2412": "Devstral Small 2505",
    "openai_gpt-5.4": "GPT-5.4",
    "qwen3.5-9b": "Qwen 3.5-9B",
    "qwen_qwen3.6-plus": "Qwen 3.6 Plus",
    # Extension-only / fixed-version routes.
    "azure_grok-4-20-non-reasoning": "Grok 4.20 (non-reasoning)",
    "cli_claude-opus-4.6": "Claude Opus 4.6 (fixed-version check)",
    "cli_gpt-5.4": "GPT-5.4 (fixed-version check)",
    "cli_gemini-3.1-pro": "Gemini 3.1 Pro Preview (fixed-version check)",
}
MATCHED_FIVE = {
    "azure_deepseek-v3.2-speciale",
    "azure_gpt-oss-120b",
    "azure_kimi-k2.5",
    "azure_llama-3.3-70b",
    "azure_mistral-large-3",
}
KEYWORD_FEATURES = [
    "inst_tokens", "inst_if_count", "inst_conditional_count", "inst_loop_count",
    "inst_collection_count", "inst_class_count", "inst_func_count",
    "inst_logic_count", "inst_total_structural", "inst_avg_word_len",
]
KEYWORD_FLOAT = {"inst_avg_word_len"}

# Model metadata for the ``models`` config. Values come from the locked panel
# configuration (src/stage_d/models_stage_d_panel.json), the retained batch
# state records, and the rebuttal generation scripts. Routes are descriptive;
# account-specific deployment names and endpoints are deliberately omitted.
MODELS_META = [
    # model_key, display, developer, roles, routes, served names, weights, notes
    ("anthropic_claude-opus-4.6", "Anthropic", ["evaluated_panel"],
     "Anthropic API (later frame via Message Batches)", "claude-opus-4-6", "closed", ""),
    ("anthropic_claude-opus-4.7", "Anthropic", ["evaluated_panel"],
     "Anthropic API (later frame via Message Batches)", "claude-opus-4-7", "closed",
     "Temperature not sent (provider default) per panel config."),
    ("anthropic_claude-sonnet-4.6", "Anthropic", ["evaluated_panel"],
     "Anthropic API (later frame via Message Batches)", "claude-sonnet-4-6", "closed", ""),
    ("arcee-ai_trinity-large-preview_free", "Arcee AI", ["evaluated_panel"],
     "OpenRouter (free tier)", "arcee-ai/trinity-large-preview", "open", ""),
    ("azure_deepseek-v3.2-speciale", "DeepSeek", ["evaluated_panel", "extension", "passk"],
     "Azure AI Foundry serverless", "DeepSeek-V3.2", "open",
     "Experiment key says 'speciale'; the recorded served model name is DeepSeek-V3.2."),
    ("azure_gpt-oss-120b", "OpenAI", ["evaluated_panel", "extension", "passk"],
     "Azure AI Foundry serverless (an OpenRouter route also appears in older configs)",
     "gpt-oss-120b", "open", ""),
    ("azure_grok-3", "xAI", ["evaluated_panel"],
     "Azure AI Foundry serverless (an OpenRouter route also appears in older configs)",
     "grok-3", "closed", ""),
    ("azure_kimi-k2.5", "Moonshot AI", ["evaluated_panel", "extension", "passk"],
     "Azure AI Foundry serverless", "Kimi-K2.5", "open", ""),
    ("azure_llama-3.3-70b", "Meta", ["evaluated_panel", "extension", "passk"],
     "Azure AI Foundry serverless (an OpenRouter route also appears in older configs)",
     "Llama-3.3-70B-Instruct", "open", ""),
    ("azure_mistral-large-3", "Mistral AI", ["evaluated_panel", "extension"],
     "Azure AI Foundry serverless (an OpenRouter route also appears in older configs)",
     "Mistral-Large-3", "open", ""),
    ("glm_4_7_flash_results", "Zhipu AI (Z.ai)", ["evaluated_panel"],
     "Locally served quantized GGUF", "GLM-4.7-Flash-Q8_0", "open", "8-bit GGUF quantization."),
    ("google_gemini-3-flash-preview", "Google", ["evaluated_panel"],
     "Gemini API batch mode and OpenRouter", "gemini-3-flash-preview", "closed", ""),
    ("google_gemini-3.1-pro-preview", "Google", ["evaluated_panel"],
     "Gemini API batch mode and OpenRouter", "gemini-3.1-pro-preview", "closed", ""),
    ("gpt-4.1", "OpenAI", ["evaluated_panel"],
     "GitHub Copilot model endpoint (earlier frame, per config) and OpenAI Batch API (later frame)",
     "gpt-4.1", "closed", ""),
    ("gpt-5-mini", "OpenAI", ["evaluated_panel"],
     "GitHub Copilot model endpoint (earlier frame, per config) and OpenAI Batch API (later frame)",
     "gpt-5-mini", "closed", "Temperature not sent (reasoning model)."),
    ("gpt-oss-20b", "OpenAI", ["evaluated_panel"],
     "Locally served quantized GGUF", "gpt-oss-20b-Q4_K_M", "open", "4-bit GGUF quantization."),
    ("ministral-3-14b-reasoning", "Mistral AI", ["evaluated_panel"],
     "Locally served quantized GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M", "open",
     "4-bit GGUF quantization."),
    ("mistral-small-2412", "Mistral AI", ["evaluated_panel"],
     "Locally served quantized GGUF", "Devstral-Small-2505-Q4_K_M", "open",
     "4-bit GGUF quantization. The experiment key says 'mistral-small-2412', but the recorded "
     "served model is Devstral-Small-2505 (Q4_K_M), the name used in the paper."),
    ("openai_gpt-5.4", "OpenAI", ["evaluated_panel", "fixed_version_check"],
     "OpenAI Batch API", "gpt-5.4", "closed", "Temperature not sent (reasoning model)."),
    ("qwen3.5-9b", "Alibaba (Qwen)", ["evaluated_panel"],
     "Locally served quantized GGUF", "Qwen3.5-9B-Q4_K_M", "open",
     "4-bit GGUF quantization; thinking disabled."),
    ("qwen_qwen3.6-plus", "Alibaba Cloud (Qwen)", ["evaluated_panel"],
     "Alibaba Cloud Model Studio (DashScope) batch; an OpenRouter route appears in older configs",
     "qwen3.6-plus / qwen3.6-plus-2026-04-02", "closed", "Thinking disabled."),
    ("azure_grok-4-20-non-reasoning", "xAI", ["extension"],
     "Azure AI Foundry serverless", "grok-4-20-non-reasoning", "closed",
     "Extension run only; not part of the 21-model panel or the matched-five frame."),
    ("cli_claude-opus-4.6", "Anthropic", ["fixed_version_check"],
     "Claude Code CLI (subscription) and OpenRouter", "claude-opus-4.6", "closed",
     "Single-shot, no tools; per-row route not recorded."),
    ("cli_gpt-5.4", "OpenAI", ["fixed_version_check"],
     "Codex CLI (read-only sandbox, reasoning effort medium)", "gpt-5.4", "closed",
     "Single-shot, no tools."),
    ("cli_gemini-3.1-pro", "Google", ["fixed_version_check"],
     "OpenRouter (reasoning effort low)", "google/gemini-3.1-pro-preview", "closed",
     "Single-shot, no tools."),
    ("judge:o4-mini", "OpenAI", ["rubric_judge", "preliminary_rater", "task_type_labeler",
                                  "harness_auditor"],
     "Azure AI Foundry", "o4-mini", "closed",
     "Its harness audit defined the reviewed version's outcome (earlier frame only); it was "
     "rerun on known-answer set A as a diagnostic of that audit, not as a candidate."),
    (PRIMARY_AUDITOR, "Xiaomi", ["outcome_auditor", "auditor_candidate"],
     "OpenRouter, pinned to one provider (GMICloud, bf16), streamed", "xiaomi/mimo-v2.6-pro",
     "not recorded", "Adopted by author decision after missing the pre-declared clean-accuracy "
     "threshold by three cases on set A; met every threshold on set B."),
    ("auditor:mai-thinking-1", "Microsoft", ["auditor_candidate", "agreement_auditor"],
     "Azure AI Foundry", "MAI-Thinking-1", "closed", "Preview model (inference retires 2026-11-04)."),
    ("auditor:phi-4-reasoning", "Microsoft", ["auditor_candidate", "agreement_auditor"],
     "Azure AI Foundry", "Phi-4-reasoning", "open (MIT)", ""),
    ("auditor:nemotron-3-ultra", "NVIDIA", ["auditor_candidate"],
     "OpenRouter (Venice, fp8)", "nvidia/nemotron-3-ultra-550b-a55b", "open", ""),
    ("auditor:laguna-s-2.1", "Poolside", ["auditor_candidate"],
     "OpenRouter (Poolside, fp4)", "poolside/laguna-s-2.1", "not recorded", ""),
    ("judge:gpt-5.5", "OpenAI", ["rubric_judge", "task_type_labeler"],
     "Azure AI Foundry", "gpt-5.5", "closed", ""),
    ("judge:llama-4-maverick", "Meta", ["rubric_judge", "task_type_labeler"],
     "Azure AI Foundry", "Llama-4-Maverick", "open", ""),
    ("judge:command-a", "Cohere", ["rubric_judge", "task_type_labeler"],
     "Azure AI Foundry", "Command A", "open-weights (non-commercial license)", ""),
    ("aux:deepseek-v3.2-rewriter", "DeepSeek", ["paraphrase_rewriter", "cross_language_porter"],
     "Azure AI Foundry serverless", "DeepSeek-V3.2", "open",
     "Paraphrase: temperature 0.2, max 1200 tokens. Porting: temperature 0.2, max 1400 tokens."),
]

MODEL_SETTINGS_DOC = {
    "evaluated_panel": "One completion per prompt. Settings vary by route and frame; see "
                       "generations.gen_* for per-row values recovered from retained batch "
                       "request records (null where no request record was retained).",
    "extension": "Temperature 0.0, max 4096 output tokens, one completion per prompt "
                 "(src/rebuttal/22_tail_generate_and_score.py).",
    "passk": "Temperature 0.8, max 4096 output tokens, five draws per prompt "
             "(src/rebuttal/18_passk_generate.py).",
    "fixed_version_check": "Single completion, no tools; OpenRouter routes use temperature 0.0 "
                           "and max 16000 tokens; Codex CLI uses its defaults "
                           "(src/rebuttal/25_frontier_cli_generate.py).",
    "outcome_auditor": "One request per non-empty main-benchmark generation with the audit system "
                       "prompt (task, code, unit tests, harness pass rate); max 16,000 completion "
                       "tokens; provider-default sampling (src/audit/independent_audit.py).",
    "auditor_candidate": "Scored on the 450-case known-answer set A with the same prompt and "
                         "parser (docs/independent_audit_protocol.md).",
    "agreement_auditor": "Re-audited the seeded 5% production sample (5,250 generations).",
}

GEN_SYSTEM_PROMPTS = {
    "A_pipeline": (
        "You are an expert Python programmer. Given a coding problem, write a "
        "complete Python solution. Output ONLY the Python code inside a single "
        "```python``` code block. Do not include any explanation, tests, or "
        "examples outside the code block."
    ),
    "B_batch_scripts": (
        "You are an expert Python programmer. Write a single, complete Python solution "
        "that passes all unit tests. Respond with ONLY the Python code inside a "
        "```python code block, no explanations before or after."
    ),
    "C_extension_passk": (
        "You are an expert Python programmer. Read the programming task and write a "
        "complete, correct Python solution. Define exactly the function(s) or "
        "class(es) the task names, with the specified signature. Output only Python "
        "code in a single ```python code block, no explanation."
    ),
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}:{line_number}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def display_bin(x: float | None) -> int | None:
    """Half-open display bin: bin b holds [b - 0.5, b + 0.5)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(math.floor(float(x) + 0.5))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def inum(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_tests(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    tests = json.loads(raw) if raw else []
    require(isinstance(tests, list), "unit_tests is not a JSON list")
    return [str(x) for x in tests]


def pearson(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    def rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort(kind="mergesort")
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=ranks)
        return (sums / counts)[inv]

    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def icc_2_1(matrix: np.ndarray) -> float:
    """Shrout-Fleiss ICC(2,1): two-way random effects, absolute agreement, single rater."""
    n, k = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((matrix - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


# ---------------------------------------------------------------------------
# Config schemas (single source of truth for Parquet, README tables, Croissant)
# ---------------------------------------------------------------------------

S, I, F, B, LS = "string", "int", "float", "bool", "list<string>"
PA_TYPES = {S: pa.string(), I: pa.int32(), F: pa.float64(), B: pa.bool_(), LS: pa.list_(pa.string())}
CR_TYPES = {S: "sc:Text", I: "sc:Integer", F: "sc:Float", B: "sc:Boolean", LS: "sc:Text"}


def dim_fields(prefix: str, kind: str, what: str) -> list[tuple[str, str, str]]:
    return [(f"{prefix}{d}", kind, f"{what} for the {d.replace('_', ' ')} dimension (0-4 scale).")
            for d in DIMS]


JUDGE_SCORE_FIELDS = [
    ("prompt_id", S, "Prompt identifier (OpenCodeInstruct `id`)."),
    ("judge", S, "Rubric judge: o4-mini, gpt-5.5, llama-4-maverick, or command-a (all outside the evaluated panel)."),
    *dim_fields("", I, "Judge score"),
    ("composite", I, "Sum of the six dimension scores (0-24)."),
    ("rubric_sha256", S, "SHA-256 of the exact rubric system prompt (docs/rubric_prompt.txt)."),
    ("scored_at", S, "UTC timestamp of the scoring call (ISO 8601)."),
]

CONFIGS: dict[str, dict[str, Any]] = {
    "prompts": {
        "description": "The 5,000-prompt Python benchmark: prompt text, unit tests, source metadata, "
                       "preliminary single-rater sampling score, four-judge ensemble index, task type, "
                       "keyword features, and prompt-level mean pass rates over the 21-model panel.",
        "key": "prompt_id",
        "fields": [
            ("prompt_id", S, "OpenCodeInstruct `id` of the source record; primary key."),
            ("source_dataset", S, "Source dataset (nvidia/OpenCodeInstruct)."),
            ("language", S, "Programming language of prompt, tests, and execution (always python)."),
            ("prompt_text", S, "Task instruction given to every evaluated model (OpenCodeInstruct `input`)."),
            ("unit_tests", LS, "Assertion snippets from OpenCodeInstruct `unit_tests`; each is executed separately by the harness."),
            ("n_unit_tests", I, "Number of assertion snippets."),
            ("construction_frame", S, "earlier_retained (2,246 prompts kept from the earlier prefix-scan draw) or later_candidate (2,754 later candidates, including deliberate high reference-CC supplementation)."),
            ("source_reference_avg_test_score", F, "NVIDIA-supplied `average_test_score` of the OpenCodeInstruct reference solution on these tests (copied from the source record, not re-executed)."),
            ("reference_cc", I, "Lizard cyclomatic complexity of the OpenCodeInstruct reference solution. Used upstream to shape the candidate pool and for alignment checks; not the analysis index."),
            ("reference_cc_band", S, "Reference-CC band used when drawing later candidates (null for earlier_retained)."),
            *dim_fields("prelim_", I, "Preliminary single-rater (o4-mini) score used only for stratified sampling"),
            ("prelim_composite", I, "Preliminary o4-mini composite (0-24); sampling only."),
            ("prelim_band", S, "Preliminary sampling band: 0-3, 4-6, 7-9, 10-12, 13-15, or 16-24 (834/834/833/833/833/833 prompts)."),
            *dim_fields("ens_", F, "Four-judge ensemble mean"),
            ("ens_composite", F, "Analysis index C_i: sum of the six ensemble dimension means (0-24)."),
            ("ens_composite_sd", F, "Sample standard deviation (ddof=1) of the per-judge composites; a judge-disagreement measure."),
            ("ens_n_judges", I, "Number of judges with a valid score (4 for 4,998 prompts; 3 and 2 for one prompt each)."),
            ("display_bin", I, "Descriptive display bin floor(ens_composite + 0.5): bin b holds [b-0.5, b+0.5). Breakpoint estimation uses the unbinned index."),
            ("task_type", S, "Primary task category from the nine-category taxonomy (o4-mini labeler)."),
            ("task_type_confidence", F, "Labeler self-reported confidence for task_type."),
            ("names_external_library", B, "Labeler flag: prompt names an external library or framework."),
            *[(f"kw_{name}", F if name in KEYWORD_FLOAT else I,
               f"Keyword/lexical prompt feature `{name}` (pre-generation lexical baseline).")
              for name in KEYWORD_FEATURES],
            ("n_models", I, "Number of evaluated-panel generations for this prompt (21)."),
            ("mean_pass_rate", F, "Mean of generations.pass_rate (the audited primary outcome) over the 21 models: the prompt-level outcome of the camera-ready paper."),
            ("mean_pass_rate_reviewed", F, "Mean of generations.pass_rate_reviewed over the 21 models (the reviewed version's prompt-level outcome)."),
            ("mean_harness_pass_rate", F, "Mean of generations.harness_pass_rate over the 21 models (pure test-execution fraction)."),
            ("in_human_calibration", B, "Prompt was graded in the human calibration study."),
            ("in_passk_subset", B, "Prompt is in the 359-prompt repeated-sampling subset."),
            ("in_paraphrase_subset", B, "Prompt is in the 150-prompt paraphrase check."),
            ("in_cross_language_subset", B, "Prompt is in the 117-prompt Java/C++ re-expression check."),
            ("is_fixed_version_anchor", B, "Prompt is one of the 150 midrange anchors in the fixed-version three-model check."),
        ],
    },
    "judge_scores": {
        "description": "Per-judge rubric scores for the 5,000 main prompts (19,997 rows; four out-of-panel judges).",
        "fields": JUDGE_SCORE_FIELDS,
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "generations": {
        "description": "One generated solution per (model, prompt) for the 21-model panel on the 5,000 main prompts "
                       "(105,000 rows): cleaned code, per-test outcomes, three outcome definitions (the audited "
                       "primary outcome, the reviewed version's outcome, and the raw harness fraction), the "
                       "auditor's verdict, Lizard output CC, and recovered generation settings. Raw API "
                       "responses are not included.",
        "fields": [
            ("model_key", S, "Experiment identifier of the evaluated model (joins to the models config)."),
            ("model_display_name", S, "Display name used in the paper."),
            ("prompt_id", S, "Prompt identifier (joins to prompts)."),
            ("construction_frame", S, "Construction frame of the prompt (copied from prompts)."),
            ("code", S, "Cleaned generated Python code exactly as executed and measured (extracted from the model response). Empty string when no code could be extracted; a small number of rows retain markdown fences."),
            ("pass_rate", F, "Primary outcome of the camera-ready paper: 1.0 if the independent auditor (MiMo-V2.6-Pro) judged the code correct, 0.0 if incorrect or if there is no code (empty-code rule); otherwise (uncertain, unparseable, or a response cut off by the token limit or a content filter) harness_pass_rate."),
            ("pass_rate_source", S, "'independent_audit' when pass_rate comes from a correct/incorrect verdict or the empty-code rule, else 'harness_fallback'."),
            ("independent_audit_verdict", S, "Counted verdict of the independent auditor: correct, incorrect, uncertain, or null (unparseable or incomplete response)."),
            ("independent_audit_handling", S, "auditor_verdict, auditor_uncertain, unparseable, incomplete_response (cut off by the token limit or a content filter), or empty_code_rule (no code; marked incorrect without an auditor request)."),
            ("harness_pass_rate", F, "Fraction of unit-test assertions that passed when the code was executed (from test_status)."),
            ("pass_rate_reviewed", F, "Outcome of the reviewed (submitted) version: harness_pass_rate, except that for earlier_retained prompts an o4-mini audit verdict of correct or incorrect set it to 1.0 or 0.0."),
            ("reviewed_audit_verdict", S, "The reviewed version's o4-mini harness-audit verdict (earlier_retained rows only): correct, incorrect, uncertain, or null."),
            ("model_returned_no_response", B, "The model API returned no response at all (time-out, cancelled operation, or empty response). These 157 rows (156 Gemini 3.1 Pro Preview) count as failures in the primary analysis; the paper also reports results treating them as missing."),
            ("n_tests", I, "Number of executed assertions."),
            ("test_status", LS, "Per-assertion outcome ('pass'/'fail'), in unit_tests order."),
            ("output_cc_lizard", I, "Generated-output cyclomatic complexity: Lizard CC of `code`, summed over reported functions. Null when not computable (1,052 rows). Never the prompt index or reference CC."),
            ("generated_at", S, "UTC timestamp recorded when the generation was stored (ISO 8601)."),
            ("gen_settings_source", S, "'batch_request_record' when settings were recovered from a retained batch request for this prompt and model, else 'not_recorded_per_row' (see the models config for documented defaults)."),
            ("gen_api_model", S, "Model string sent in the retained batch request (null if not recorded)."),
            ("gen_temperature", F, "Temperature sent in the retained batch request; null if not recorded or not sent (provider default)."),
            ("gen_temperature_sent", B, "Whether a temperature parameter was present in the retained batch request (null if not recorded)."),
            ("gen_max_output_tokens", I, "Output-token cap sent in the retained batch request (null if not recorded)."),
            ("gen_system_prompt_variant", S, "Generation system prompt variant (A_pipeline, B_batch_scripts, see docs/generation_system_prompts.json); null if not recorded."),
        ],
        "references": {"prompt_id": "prompts/prompt_id", "model_key": "models/model_key"},
    },
    "extension_prompts": {
        "description": "The 365-prompt audit-clean high-complexity extension: selected on prompt-side scores only, "
                       "contract-audited, and reference-verified (every reference solution passes every test). "
                       "Disjoint from the main benchmark and from the excluded 24-bin set.",
        "key": "prompt_id",
        "fields": [
            ("prompt_id", S, "OpenCodeInstruct `id`; primary key."),
            ("source_dataset", S, "Source dataset (nvidia/OpenCodeInstruct)."),
            ("language", S, "Always python."),
            ("prompt_text", S, "Task instruction."),
            ("unit_tests", LS, "Assertion snippets."),
            ("n_unit_tests", I, "Number of assertion snippets."),
            ("construction_frame", S, "Always tail_extension."),
            ("source_reference_avg_test_score", F, "NVIDIA-supplied reference `average_test_score` (all 1.0)."),
            ("reference_local_pass_rate", F, "Pass rate of the reference solution when re-executed by the benchmark harness (all 1.0)."),
            ("reference_cc", I, "Lizard CC of the OpenCodeInstruct reference solution."),
            *dim_fields("prelim_", I, "Preliminary o4-mini candidate score (used only to rank candidates for ensemble scoring)"),
            ("prelim_composite", I, "Preliminary o4-mini composite (0-24)."),
            *dim_fields("ens_", F, "Four-judge ensemble mean"),
            ("ens_composite", F, "Four-judge ensemble composite (selection required >= 15)."),
            ("ens_composite_sd", F, "Sample standard deviation (ddof=1) of the per-judge composites, recomputed from extension_judge_scores (the source file stores the population SD)."),
            ("ens_n_judges", I, "Number of judges (always 4)."),
            ("display_bin", I, "floor(ens_composite + 0.5); 218/133/11/3 prompts at bins 15/16/17/18."),
        ],
    },
    "extension_judge_scores": {
        "description": "Per-judge rubric scores for the 365 extension prompts (1,460 rows).",
        "fields": JUDGE_SCORE_FIELDS,
        "references": {"prompt_id": "extension_prompts/prompt_id"},
    },
    "extension_generations": {
        "description": "Test outcomes for the 365 extension prompts across the six extension-run models (2,190 rows). "
                       "Generated code was not retained by the extension run, so no code or output CC is available.",
        "fields": [
            ("model_key", S, "Experiment identifier (joins to models)."),
            ("model_display_name", S, "Display name."),
            ("prompt_id", S, "Extension prompt identifier."),
            ("in_matched_five", B, "Model is one of the five models present in both the main panel and the extension run (the paper's matched frame)."),
            ("pass_rate", F, "Fraction of unit-test assertions passed by the single generated solution."),
            ("has_code", B, "Whether code could be extracted from the response."),
            ("gen_temperature", F, "Temperature (0.0)."),
            ("gen_max_output_tokens", I, "Output-token cap (4096)."),
        ],
        "references": {"prompt_id": "extension_prompts/prompt_id", "model_key": "models/model_key"},
    },
    "fixed_version_generations": {
        "description": "Fixed-version three-model check (Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro Preview): 150 main-benchmark "
                       "midrange anchors plus the 365 extension prompts per model (1,545 rows). No code retained.",
        "fields": [
            ("model_key", S, "Experiment identifier (joins to models)."),
            ("model_display_name", S, "Display name."),
            ("prompt_id", S, "Prompt identifier (anchor ids join to prompts; tail ids join to extension_prompts)."),
            ("group", S, "anchor (main-benchmark midrange prompt) or tail (extension prompt)."),
            ("ens_composite", F, "Ensemble composite of the prompt as recorded by the run."),
            ("pass_rate", F, "Fraction of unit-test assertions passed."),
            ("has_code", B, "Whether code could be extracted from the response."),
        ],
    },
    "passk_generations": {
        "description": "Repeated-sampling subset: 359 main prompts x 4 models x 5 draws at temperature 0.8 (7,180 rows), "
                       "with code and pass rate. These draws are separate from the main single-generation panel.",
        "fields": [
            ("model_key", S, "Experiment identifier (joins to models)."),
            ("model_display_name", S, "Display name."),
            ("prompt_id", S, "Prompt identifier (joins to prompts)."),
            ("draw", I, "Draw index 1-5."),
            ("code", S, "Cleaned generated code."),
            ("pass_rate", F, "Fraction of unit-test assertions passed."),
            ("gen_temperature", F, "Temperature (0.8)."),
            ("gen_max_output_tokens", I, "Output-token cap (4096)."),
        ],
        "references": {"prompt_id": "prompts/prompt_id", "model_key": "models/model_key"},
    },
    "human_calibration": {
        "description": "Blinded human rubric grades: 200 prompts by grader_1 (first author) and a 50-prompt overlap by "
                       "grader_2 (second research-team member); LLM scores hidden during grading (250 rows).",
        "fields": [
            ("grader", S, "grader_1 (first author, 200 prompts) or grader_2 (second research-team member, 50 prompts)."),
            ("prompt_id", S, "Prompt identifier (joins to prompts)."),
            ("worksheet_row", I, "Position in the grader's randomized worksheet (grader_2 received the first 50 rows of grader_1's order)."),
            *dim_fields("", I, "Human grade"),
            ("composite", I, "Sum of the six human grades (0-24)."),
            ("in_two_grader_overlap", B, "Prompt was graded by both graders."),
            ("sampling_band", S, "Ensemble-composite band used to stratify the 500-prompt calibration pool."),
            ("selection_stratum", S, "high_disagreement (top judge-disagreement prompts within band) or random, reproduced from the seeded selector; null if not reproducible."),
            ("calibration_priority", F, "Judge disagreement at selection (ensemble composite SD)."),
        ],
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "task_type_labels": {
        "description": "Nine-category task-type labels: o4-mini labels all 5,000 prompts; gpt-5.5, llama-4-maverick, "
                       "and command-a label a shared 500-prompt subset (6,500 rows).",
        "fields": [
            ("prompt_id", S, "Prompt identifier (joins to prompts)."),
            ("labeler", S, "Labeling model."),
            ("is_primary_labeler", B, "True for the o4-mini labels used as task-type fixed effects."),
            ("primary_type", S, "Task category."),
            ("names_external_library", B, "Prompt names an external library/framework."),
            ("confidence", F, "Labeler self-reported confidence."),
            ("taxonomy_sha256", S, "Hash of the fixed taxonomy prompt."),
        ],
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "paraphrase_prompts": {
        "description": "150 plain-language rewrites (DeepSeek-V3.2, instructed to preserve inputs, outputs, behavior, "
                       "constraints, and edge cases). Rewrites were not execution-verified.",
        "fields": [
            ("prompt_id", S, "Original prompt identifier (joins to prompts)."),
            ("sampling_bin", I, "Rounded original composite bin used for sampling (75% from bins 10-18)."),
            ("original_composite", F, "Original ensemble composite as recorded at sampling time."),
            ("original_prompt_text", S, "Original prompt text."),
            ("paraphrased_prompt_text", S, "Rewritten prompt text that the judges rescored."),
        ],
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "paraphrase_judge_scores": {
        "description": "Per-judge rubric scores of the 150 paraphrased prompts (600 rows).",
        "fields": JUDGE_SCORE_FIELDS,
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "cross_language_prompts": {
        "description": "117 prompts re-expressed in Java and in C++ by DeepSeek-V3.2 for rescoring only (234 rows); "
                       "no non-Python generation or execution.",
        "fields": [
            ("prompt_id", S, "Original Python prompt identifier (joins to prompts)."),
            ("target_language", S, "java or cpp."),
            ("original_composite", F, "Original Python ensemble composite as recorded."),
            ("ported_prompt_text", S, "Re-expressed prompt text."),
        ],
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "cross_language_judge_scores": {
        "description": "Per-judge rubric scores of the Java and C++ re-expressions (936 rows).",
        "fields": [("target_language", S, "java or cpp."), *JUDGE_SCORE_FIELDS],
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "audit_known_answer_cases": {
        "description": "Known-answer cases used to select (set A) and confirm (set B) the outcome auditor. For each of "
                       "150 benchmark prompts per set whose reference solution passes every unit test: the normalized "
                       "reference (clean, correct), a copy with the test-called names renamed so the harness fails it "
                       "(cosmetic, correct), and the most subtle single-AST mutation the tests detect (bug, "
                       "incorrect), all executed in the benchmark harness (900 rows). Labels come from the harness "
                       "and construction, not human review; some clean references violate task instructions the "
                       "tests do not check.",
        "key": "case_id",
        "fields": [
            ("case_id", S, "Case identifier: <prompt_id>:<variant>."),
            ("known_answer_set", S, "A_selection (auditor selection, seed 20260928) or B_confirmation (fresh prompts, seed 20260930)."),
            ("prompt_id", S, "Benchmark prompt identifier (joins to prompts)."),
            ("category", S, "clean, cosmetic, or bug."),
            ("ground_truth", S, "correct (clean, cosmetic) or incorrect (bug)."),
            ("mutation", S, "For bug cases, the AST mutation kind and site (JSON); null otherwise."),
            ("code", S, "The code shown to the auditors."),
            ("harness_pass_rate", F, "Fraction of unit tests the case passes in the benchmark harness."),
        ],
        "references": {"prompt_id": "prompts/prompt_id"},
    },
    "audit_known_answer_responses": {
        "description": "Every auditor response on the known-answer sets: five candidate auditors and the reviewed "
                       "version's o4-mini audit (diagnostic) on set A, and the adopted auditor on set B (3,150 rows).",
        "fields": [
            ("known_answer_set", S, "A_selection or B_confirmation."),
            ("auditor", S, "Auditor model key (joins to models)."),
            ("case_id", S, "Known-answer case (joins to audit_known_answer_cases)."),
            ("status", S, "ok, parse_error, or api_error as recorded by the audit client."),
            ("verdict", S, "Verdict parsed from the response (correct, incorrect, uncertain, or null)."),
            ("counted_verdict", S, "Verdict after the completeness rule (null if the response was cut off by the token limit or a content filter); used for every reported rate."),
            ("finish_reason", S, "Finish reason reported by the endpoint."),
            ("parse_status", S, "json, regex, parse_error, or rule."),
            ("reason", S, "The auditor's stated reason."),
            ("response_tail", S, "Last 600 characters of the response."),
            ("completion_tokens", I, "Completion tokens reported by the endpoint (may exclude reasoning tokens on some routes)."),
        ],
        "references": {"case_id": "audit_known_answer_cases/case_id", "auditor": "models/model_key"},
    },
    "audit_production_verdicts": {
        "description": "Outcome-audit verdicts for the main benchmark: the adopted auditor on all 105,000 generations "
                       "and two second auditors on the seeded 5% agreement sample (5,250 generations each), "
                       "115,500 rows. The last final record per generation is kept, as in the analysis.",
        "fields": [
            ("auditor", S, "Auditor model key (joins to models)."),
            ("model_key", S, "Evaluated model (joins to models)."),
            ("prompt_id", S, "Prompt identifier (joins to prompts)."),
            ("in_agreement_sample", B, "Generation is in the seeded 5% agreement sample."),
            ("status", S, "ok, parse_error, or api_error as recorded by the audit client."),
            ("verdict", S, "Verdict parsed from the response."),
            ("counted_verdict", S, "Verdict after the completeness rule; for the adopted auditor this is generations.independent_audit_verdict."),
            ("finish_reason", S, "Finish reason reported by the endpoint."),
            ("parse_status", S, "json, regex, parse_error, or rule (empty code: incorrect without a request)."),
            ("reason", S, "The auditor's stated reason."),
            ("response_tail", S, "Last 600 characters of the response."),
        ],
        "references": {"prompt_id": "prompts/prompt_id", "model_key": "models/model_key", "auditor": "models/model_key"},
    },
    "models": {
        "description": "Metadata for every model whose outputs or labels appear in the release: developer, role(s), "
                       "access route, served model string, and documented generation settings.",
        "key": "model_key",
        "fields": [
            ("model_key", S, "Experiment identifier (judge:/aux: prefixes for non-evaluated roles)."),
            ("display_name", S, "Display name."),
            ("developer", S, "Model developer."),
            ("roles", LS, "Role(s) in the release."),
            ("access_routes", S, "How outputs were obtained (no account or deployment identifiers)."),
            ("served_model", S, "Model string recorded in the run configuration."),
            ("weights", S, "open or closed weights (descriptive)."),
            ("documented_settings", S, "Documented generation settings for the role(s)."),
            ("notes", S, "Caveats (quantization, naming discrepancies)."),
        ],
    },
}


# ---------------------------------------------------------------------------
# Loading and building
# ---------------------------------------------------------------------------

class Builder:
    def __init__(self, data_root: Path, out_dir: Path, source_scan: bool) -> None:
        self.root = data_root
        self.data = data_root / "data"
        self.out = out_dir
        self.source_scan = source_scan
        self.report: dict[str, Any] = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        self.tables: dict[str, list[dict]] = {}
        self.files: dict[str, list[dict]] = {}

    # ---- main benchmark -------------------------------------------------
    def load_main(self) -> None:
        sd = self.data / "stage_d"
        self.prompts_raw = {r["prompt_id"]: r for r in read_jsonl(sd / "stage_d_prompts.jsonl")}
        require(len(self.prompts_raw) == EXPECTED_MAIN_PROMPTS, "stage_d_prompts.jsonl must have 5,000 prompts")

        self.ensemble = {}
        for r in read_jsonl(sd / "ensemble_scores_current_aggregated.jsonl"):
            require(r["prompt_id"] not in self.ensemble, "duplicate ensemble row")
            self.ensemble[r["prompt_id"]] = r
        require(set(self.ensemble) == set(self.prompts_raw), "current ensemble ids != prompt ids")

        stale_ids = {r["prompt_id"] for r in read_jsonl(sd / "ensemble_scores_aggregated.jsonl")}
        self.report["pitfall_stale_ensemble_overlap"] = len(stale_ids & set(self.prompts_raw))

        # Per-judge rows.
        rows = []
        seen = set()
        dropped = Counter()
        for judge, fname in JUDGE_FILES.items():
            for r in read_jsonl(sd / fname):
                if r.get("error") is not None or not r.get("scores"):
                    dropped[judge] += 1
                    continue
                key = (r["prompt_id"], judge)
                require(key not in seen, f"duplicate judge row {key}")
                seen.add(key)
                rows.append(self._judge_row(r, judge))
        require(len(rows) == EXPECTED_JUDGE_ROWS, f"expected 19,997 judge rows, found {len(rows)}")
        require({r["prompt_id"] for r in rows} == set(self.prompts_raw), "judge rows do not cover the 5,000 prompts")
        self.report["judge_rows_dropped_invalid"] = dict(dropped)
        self.tables["judge_scores"] = sorted(rows, key=lambda r: (r["prompt_id"], r["judge"]))

        # Consistency: aggregated means == mean of per-judge rows.
        by_pid = defaultdict(list)
        for r in rows:
            by_pid[r["prompt_id"]].append(r)
        max_dev = 0.0
        sd_match_pop = sd_match_samp = 0
        for pid, lst in by_pid.items():
            agg = self.ensemble[pid]
            for d in DIMS:
                max_dev = max(max_dev, abs(np.mean([x[d] for x in lst]) - agg["scores_mean"][d]))
            comps = np.array([x["composite"] for x in lst], dtype=float)
            if len(comps) > 1:
                sd_match_pop += abs(comps.std(ddof=0) - agg["composite_std"]) < 1e-9
                sd_match_samp += abs(comps.std(ddof=1) - agg["composite_std"]) < 1e-9
            require(agg["n_scorers"] == len(lst), f"n_scorers mismatch for {pid}")
        require(max_dev < 1e-9, f"aggregated means deviate from per-judge rows by {max_dev}")
        self.report["ensemble_sd_definition"] = {"matches_population_sd": sd_match_pop,
                                                 "matches_sample_sd": sd_match_samp}

    @staticmethod
    def _judge_row(r: dict, judge: str) -> dict:
        s = r["scores"]
        out = {"prompt_id": r["prompt_id"], "judge": judge}
        for d in DIMS:
            out[d] = int(s[d])
        out["composite"] = int(sum(int(s[d]) for d in DIMS))
        if r.get("composite") is not None:
            require(int(r["composite"]) == out["composite"], "stored composite != sum of dims")
        out["rubric_sha256"] = r.get("rubric_hash")
        out["scored_at"] = r.get("timestamp")
        return out

    def load_batch_records(self) -> None:
        """Map (model_key, prompt_id) -> list of (submit_time_utc, settings) from retained batch requests."""
        self.batch_records: dict[tuple[str, str], list[tuple[dt.datetime, dict]]] = defaultdict(list)
        summary = []
        for d in (self.data / "batch_requests", self.data / "stage_d" / "batch_requests"):
            for path in sorted(d.glob("*.jsonl")):
                m = re.match(r"^(?P<key>.+)_(?P<date>\d{8})_(?P<time>\d{6})\.jsonl$", path.name)
                if not m:
                    continue
                model_key = m.group("key")
                # File names use local time (US Central, CDT = UTC-5 for Apr-May 2026 files).
                submitted = dt.datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
                submitted = submitted.replace(tzinfo=dt.timezone(dt.timedelta(hours=-5)))
                n = 0
                for r in read_jsonl(path):
                    pid = r.get("custom_id") or r.get("key")
                    settings = self._extract_settings(r)
                    self.batch_records[(model_key, pid)].append((submitted, settings))
                    n += 1
                summary.append({"model_key": model_key, "submitted_local": m.group("date") + "_" + m.group("time"), "requests": n})
        self.report["batch_request_files"] = summary

    @staticmethod
    def _extract_settings(r: dict) -> dict:
        system = None
        if "params" in r:  # Anthropic Message Batches
            p = r["params"]
            system = p.get("system")
            return {"api_model": p.get("model"), "temperature_sent": "temperature" in p,
                    "temperature": fnum(p.get("temperature")), "max_tokens": inum(p.get("max_tokens")),
                    "system": system}
        if "body" in r:  # OpenAI-compatible batch (OpenAI, DashScope)
            b = r["body"]
            msgs = b.get("messages") or []
            if msgs and msgs[0].get("role") == "system":
                system = msgs[0].get("content")
            return {"api_model": b.get("model"), "temperature_sent": "temperature" in b,
                    "temperature": fnum(b.get("temperature")),
                    "max_tokens": inum(b.get("max_completion_tokens", b.get("max_tokens"))),
                    "system": system}
        req = r.get("request", {})  # Gemini API batch
        gc = req.get("generation_config") or req.get("generationConfig") or {}
        si = req.get("system_instruction") or req.get("systemInstruction") or {}
        parts = si.get("parts") if isinstance(si, dict) else None
        if parts:
            system = parts[0].get("text")
        return {"api_model": None, "temperature_sent": "temperature" in gc,
                "temperature": fnum(gc.get("temperature")),
                "max_tokens": inum(gc.get("max_output_tokens", gc.get("maxOutputTokens"))),
                "system": system}

    def _settings_for(self, model_key: str, pid: str, generated_at: str | None, api_default: str | None) -> dict:
        recs = self.batch_records.get((model_key, pid))
        empty = {"gen_settings_source": "not_recorded_per_row", "gen_api_model": None, "gen_temperature": None,
                 "gen_temperature_sent": None, "gen_max_output_tokens": None, "gen_system_prompt_variant": None}
        if not recs:
            return empty
        ts = None
        if generated_at:
            try:
                ts = dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
            except ValueError:
                ts = None
        eligible = [x for x in recs if ts is None or x[0] <= ts + dt.timedelta(hours=1)]
        if not eligible:
            return empty
        _, s = max(eligible, key=lambda x: x[0])
        variant = None
        if s.get("system"):
            for name, text in GEN_SYSTEM_PROMPTS.items():
                if s["system"].strip() == text.strip():
                    variant = name
                    break
            if variant is None:
                variant = "other:" + hashlib.sha256(s["system"].encode("utf-8")).hexdigest()[:12]
        return {"gen_settings_source": "batch_request_record",
                "gen_api_model": s.get("api_model") or api_default,
                "gen_temperature": s.get("temperature") if s.get("temperature_sent") else None,
                "gen_temperature_sent": bool(s.get("temperature_sent")),
                "gen_max_output_tokens": s.get("max_tokens"),
                "gen_system_prompt_variant": variant}

    def _audit_final_records(self, path: Path) -> dict[str, dict]:
        """Last final record per case, exactly as the apply step keeps them."""
        final = {}
        with path.open("rb") as handle:
            for raw in handle:
                text = raw.strip(b"\x00\r\n ")
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except ValueError:
                    continue
                if rec.get("status") in ("ok", "parse_error"):
                    final[rec["case_id"]] = rec
        return final

    @staticmethod
    def _handling(rec: dict | None) -> str:
        if rec is None:
            return "unparseable"
        if rec.get("parse_status") == "rule":
            return "empty_code_rule"
        counted = final_verdict(rec)
        if counted in ("correct", "incorrect"):
            return "auditor_verdict"
        if counted == "uncertain":
            return "auditor_uncertain"
        if rec.get("status") == "ok" and rec.get("finish_reason") in INCOMPLETE_FINISH:
            return "incomplete_response"
        return "unparseable"

    def build_generations(self) -> None:
        sc_dir = self.data / "stage_d" / "scored_independent_audit"
        paths = sorted(p for p in sc_dir.glob("*.jsonl") if not p.name.startswith("_"))
        require(len(paths) == EXPECTED_MODELS, f"expected 21 scored_independent_audit files, found {len(paths)}")
        self.primary_audit = self._audit_final_records(
            self.data / "independent_audit" / "production" / dict(PRODUCTION_RUNS)[PRIMARY_AUDITOR])
        gemini_api = {"google_gemini-3-flash-preview": "gemini-3-flash-preview",
                      "google_gemini-3.1-pro-preview": "gemini-3.1-pro-preview"}
        rows: list[dict] = []
        kw: dict[str, dict] = {}
        kw_conflicts = 0
        stats = Counter()
        for path in paths:
            model_key = path.stem
            require(model_key in DISPLAY_NAMES, f"unknown model {model_key}")
            seen = set()
            for r in read_jsonl(path):
                pid = r["id"]
                require(pid in self.prompts_raw, f"{model_key}: prompt {pid} not in benchmark")
                require(pid not in seen, f"{model_key}: duplicate {pid}")
                seen.add(pid)
                prompt = self.prompts_raw[pid]
                require(r.get("input") == prompt["input"], f"{model_key}/{pid}: input differs from prompt file")
                status = [str(s) for s in (r.get("status") or r.get("tests_execution_status") or [])]
                n_tests = len(status)
                # Rows with no executed assertions (no extractable code) score 0.0 in the pipeline.
                harness = (sum(1 for s in status if s == "pass") / n_tests) if n_tests else 0.0
                stats["no_tests_executed"] += n_tests == 0
                # Reviewed version's outcome (o4-mini audit, earlier frame only).
                pr_rev = r.get("pass_rate_reviewed_o4mini_audit")
                require(pr_rev is not None, f"{model_key}/{pid}: null reviewed pass_rate")
                verdict = r.get("judge_verdict")
                frame = FRAME_NAMES[prompt["selection_source"]]
                if verdict in ("correct", "incorrect"):
                    source = "o4mini_audit_override"
                    require(frame == "earlier_retained", "audit override outside earlier frame")
                    require(float(pr_rev) == (1.0 if verdict == "correct" else 0.0), "override value mismatch")
                else:
                    source = "harness"
                    require(harness is not None and abs(float(pr_rev) - harness) < 1e-9,
                            f"{model_key}/{pid}: harness pass_rate mismatch")
                if "harness_pass_rate" in r and r["harness_pass_rate"] is not None and harness is not None:
                    require(abs(float(r["harness_pass_rate"]) - harness) < 1e-9, "stored harness mismatch")
                if source == "o4mini_audit_override" and harness is not None and abs(float(pr_rev) - harness) > 1e-9:
                    stats["audit_override_changed_value"] += 1
                    stats[f"audit_override_changed_to_{verdict}"] += 1
                # Primary outcome (independent audit), checked against the audit record itself.
                pr = r.get("pass_rate")
                require(pr is not None, f"{model_key}/{pid}: null pass_rate")
                ind = r.get("independent_audit_verdict")
                rec = self.primary_audit.get(f"{model_key}:{pid}")
                require(final_verdict(rec) == ind, f"{model_key}/{pid}: stored verdict differs from the audit record")
                if ind in ("correct", "incorrect"):
                    ind_source = "independent_audit"
                    require(float(pr) == (1.0 if ind == "correct" else 0.0), "independent audit value mismatch")
                else:
                    ind_source = "harness_fallback"
                    require(abs(float(pr) - harness) < 1e-9, f"{model_key}/{pid}: fallback differs from harness")
                handling = self._handling(rec)
                stats[f"independent_audit_{handling}"] += 1
                no_response = bool(r.get("generation_returned_no_response"))
                stats["model_returned_no_response"] += no_response
                cc = r.get("kappa_cyclomatic")
                code = r.get("code_cleaned") or ""
                stats["code_empty"] += not code.strip()
                stats["code_with_fence"] += code.lstrip().startswith("```")
                row = {
                    "model_key": model_key,
                    "model_display_name": DISPLAY_NAMES[model_key],
                    "prompt_id": pid,
                    "construction_frame": frame,
                    "code": code,
                    "pass_rate": float(pr),
                    "pass_rate_source": ind_source,
                    "independent_audit_verdict": ind,
                    "independent_audit_handling": handling,
                    "harness_pass_rate": harness,
                    "pass_rate_reviewed": float(pr_rev),
                    "reviewed_audit_verdict": verdict,
                    "model_returned_no_response": no_response,
                    "n_tests": n_tests,
                    "test_status": status,
                    "output_cc_lizard": inum(cc),
                    "generated_at": r.get("generation_timestamp"),
                }
                row.update(self._settings_for(model_key, pid, r.get("generation_timestamp"), gemini_api.get(model_key)))
                stats["settings_" + row["gen_settings_source"]] += 1
                rows.append(row)
                feats = r.get("iv_features") or {}
                if pid in kw:
                    if any(kw[pid].get(k) != feats.get(k) for k in KEYWORD_FEATURES):
                        kw_conflicts += 1
                else:
                    kw[pid] = feats
            require(seen == set(self.prompts_raw), f"{model_key} covers {len(seen)} of 5,000 prompts")
            # Pitfall 1: the join of this model file with the current ensemble must be exactly 5,000.
            require(len(seen & set(self.ensemble)) == EXPECTED_MAIN_PROMPTS, "ensemble join != 5,000")
        require(len(rows) == EXPECTED_MAIN_PROMPTS * EXPECTED_MODELS, "generation row count != 105,000")
        self.report["generation_stats"] = dict(stats)
        self.report["keyword_feature_conflicts_across_models"] = kw_conflicts
        self.keyword = kw
        rows.sort(key=lambda r: (r["model_key"], r["prompt_id"]))
        self.tables["generations"] = rows

    def build_prompts(self) -> None:
        # Task types.
        labels = list(read_jsonl(self.data / "rebuttal" / "task_type_labels_long.jsonl"))
        primary = {}
        tt_rows = []
        seen = set()
        for r in labels:
            if r.get("error") is not None:
                continue
            labeler = SCORER_TO_JUDGE.get(r["scorer_id"], r["scorer_id"])
            key = (r["prompt_id"], labeler)
            require(key not in seen, f"duplicate task-type label {key}")
            seen.add(key)
            tt_rows.append({
                "prompt_id": r["prompt_id"], "labeler": labeler,
                "is_primary_labeler": labeler == "o4-mini",
                "primary_type": r.get("primary_type"),
                "names_external_library": r.get("names_external_library"),
                "confidence": fnum(r.get("confidence")),
                "taxonomy_sha256": r.get("taxonomy_hash"),
            })
            if labeler == "o4-mini":
                primary[r["prompt_id"]] = r
        require(set(primary) == set(self.prompts_raw), "o4-mini task-type labels must cover 5,000 prompts")
        self.tables["task_type_labels"] = sorted(tt_rows, key=lambda r: (r["prompt_id"], r["labeler"]))

        # Subset membership flags.
        passk_ids = {r["prompt_id"] for r in read_jsonl(self.data / "rebuttal" / "passk" / "draw_scores.jsonl")}
        para_ids = {r["prompt_id"] for r in read_jsonl(self.data / "rebuttal" / "paraphrase" / "paraphrases_full.jsonl")}
        xl_ids = {r["prompt_id"] for r in read_jsonl(self.data / "rebuttal" / "nonpython" / "ported_java_full.jsonl")}
        human_ids = set(json.loads((self.data / "rebuttal" / "human_calibration" / "filled_mh.json").read_text(encoding="utf-8"))["scores"])
        anchor_ids = set()
        for path in (self.data / "rebuttal" / "frontier_tail" / "scored").glob("*.jsonl"):
            anchor_ids |= {r["prompt_id"] for r in read_jsonl(path) if r.get("group") == "anchor"}
        for name, ids in (("passk", passk_ids), ("paraphrase", para_ids), ("cross_language", xl_ids),
                          ("human", human_ids), ("anchors", anchor_ids)):
            require(ids <= set(self.prompts_raw), f"{name} ids not all in the main benchmark")
        self.report["subset_sizes"] = {"passk": len(passk_ids), "paraphrase": len(para_ids),
                                       "cross_language": len(xl_ids), "human_graded": len(human_ids),
                                       "fixed_version_anchors": len(anchor_ids)}

        by_pid = defaultdict(list)
        for g in self.tables["generations"]:
            by_pid[g["prompt_id"]].append(g)

        rows = []
        for pid in sorted(self.prompts_raw):
            p = self.prompts_raw[pid]
            e = self.ensemble[pid]
            tests = parse_tests(p["unit_tests"])
            gens = by_pid[pid]
            src = self.source_meta.get(pid, {})
            row = {
                "prompt_id": pid,
                "source_dataset": "nvidia/OpenCodeInstruct",
                "language": p.get("lang", "python"),
                "prompt_text": p["input"],
                "unit_tests": tests,
                "n_unit_tests": len(tests),
                "construction_frame": FRAME_NAMES[p["selection_source"]],
                "source_reference_avg_test_score": src.get("average_test_score"),
                "reference_cc": inum(p.get("reference_cc")),
                "reference_cc_band": p.get("reference_bin"),
            }
            for d in DIMS:
                row[f"prelim_{d}"] = inum(p["rubric_scores"][d])
            row["prelim_composite"] = inum(p["rubric_composite"])
            row["prelim_band"] = p["rubric_bin"]
            for d in DIMS:
                row[f"ens_{d}"] = float(e["scores_mean"][d])
            row["ens_composite"] = float(e["composite_mean"])
            row["ens_composite_sd"] = fnum(e.get("composite_std"))
            row["ens_n_judges"] = int(e["n_scorers"])
            row["display_bin"] = display_bin(row["ens_composite"])
            t = primary[pid]
            row["task_type"] = t.get("primary_type")
            row["task_type_confidence"] = fnum(t.get("confidence"))
            row["names_external_library"] = t.get("names_external_library")
            feats = self.keyword.get(pid, {})
            for name in KEYWORD_FEATURES:
                v = feats.get(name)
                row[f"kw_{name}"] = fnum(v) if name in KEYWORD_FLOAT else inum(v)
            row["n_models"] = len(gens)
            row["mean_pass_rate"] = float(np.mean([g["pass_rate"] for g in gens]))
            row["mean_pass_rate_reviewed"] = float(np.mean([g["pass_rate_reviewed"] for g in gens]))
            hs = [g["harness_pass_rate"] for g in gens if g["harness_pass_rate"] is not None]
            row["mean_harness_pass_rate"] = float(np.mean(hs)) if hs else None
            row["in_human_calibration"] = pid in human_ids
            row["in_passk_subset"] = pid in passk_ids
            row["in_paraphrase_subset"] = pid in para_ids
            row["in_cross_language_subset"] = pid in xl_ids
            row["is_fixed_version_anchor"] = pid in anchor_ids
            rows.append(row)
        self.tables["prompts"] = rows

    # ---- source metadata (OpenCodeInstruct extraction) ----------------------
    def scan_source(self, wanted: dict[str, str]) -> None:
        """Stream data/final_results_scored.jsonl (the OpenCodeInstruct extraction) for wanted ids."""
        self.source_meta: dict[str, dict] = {}
        if not self.source_scan:
            self.report["source_scan"] = "skipped"
            return
        path = self.data / "final_results_scored.jsonl"
        pat = re.compile(rb'^\{"id": "([0-9a-f]+)"')
        tests_mismatch = 0
        with path.open("rb") as handle:
            for line in handle:
                m = pat.match(line)
                if not m:
                    continue
                pid = m.group(1).decode()
                if pid in wanted and pid not in self.source_meta:
                    r = json.loads(line)
                    self.source_meta[pid] = {"average_test_score": fnum(r.get("average_test_score"))}
                    if r.get("unit_tests") != wanted[pid]:
                        tests_mismatch += 1
        missing = set(wanted) - set(self.source_meta)
        self.report["source_scan"] = {"found": len(self.source_meta), "wanted": len(wanted),
                                      "missing": len(missing), "unit_tests_differ_from_source": tests_mismatch}

    # ---- extension --------------------------------------------------------
    def build_extension(self) -> None:
        t = self.data / "rebuttal" / "tail_topup"
        final = {r["prompt_id"]: r for r in read_jsonl(t / "tail_topup_final.jsonl")}
        require(len(final) == EXPECTED_EXTENSION, "extension must have 365 prompts")
        # Provenance checks for the 365: derived from the 1,129 ensemble-scored candidates by dropping
        # every prompt carrying an exclusion flag in the contract/test audit.
        pool = {r["prompt_id"] for r in read_jsonl(t / "tail_topup_prompts.jsonl")}
        flagged: dict[str, set[str]] = defaultdict(set)
        import csv
        with (t / "tail_audit_flags.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                flagged[row["prompt_id"]].update(x.strip() for x in re.split(r"[;,]", row.get("flags") or "") if x.strip())
        exclude = {"contract_hidden_test_callable", "contract_io_prompt_callable_tests",
                   "risk_external_fixture_or_global", "weak_many_duplicate_tests", "weak_some_duplicate_tests"}
        drop = {p for p, fs in flagged.items() if fs & exclude or any(x.startswith("hard_") for x in fs)}
        require(pool - drop == set(final), "365 extension != audited pool minus flagged prompts")
        refs = {r["id"]: r for r in read_jsonl(t / "tail_reference_rows.jsonl")}
        require(all(refs[p]["pass_rate"] == 1.0 for p in final), "extension reference solutions must pass")
        b24 = {r["prompt_id"] for r in read_jsonl(self.data / "stage_d_24bin_equal" / "prompts.jsonl")}
        require(not (set(final) & b24), "extension overlaps the excluded 24-bin set")
        require(not (set(final) & set(self.prompts_raw)), "extension overlaps the main benchmark")
        self.report["extension_provenance"] = {
            "candidates_scored_by_ensemble_with_composite_ge_15": len(pool),
            "dropped_by_contract_or_test_audit": len(pool & drop),
            "retained": len(final),
            "reference_rows_all_pass": True,
            "overlap_with_24bin_set": 0,
            "overlap_with_main_benchmark": 0,
            "24bin_set_size": len(b24),
            "24bin_overlap_with_main_benchmark": len(b24 & set(self.prompts_raw)),
        }

        prelim = {}
        for r in read_jsonl(self.data / "stage_d" / "candidate_rubric_scores.jsonl"):
            if r.get("prompt_id") in final and r.get("scores"):
                prelim.setdefault(r["prompt_id"], r)
        require(set(prelim) == set(final), "missing preliminary scores for extension prompts")

        jrows = []
        seen = set()
        for r in read_jsonl(t / "tail_ensemble_long.jsonl"):
            if r["prompt_id"] not in final or r.get("error") is not None or not r.get("scores"):
                continue
            judge = SCORER_TO_JUDGE[r["scorer_id"]]
            key = (r["prompt_id"], judge)
            require(key not in seen, f"duplicate extension judge row {key}")
            seen.add(key)
            jrows.append(self._judge_row(r, judge))
        require(len(jrows) == 4 * EXPECTED_EXTENSION, f"expected 1,460 extension judge rows, got {len(jrows)}")
        self.tables["extension_judge_scores"] = sorted(jrows, key=lambda r: (r["prompt_id"], r["judge"]))
        by = defaultdict(list)
        for r in jrows:
            by[r["prompt_id"]].append(r)

        rows = []
        for pid in sorted(final):
            f = final[pid]
            tests = parse_tests(f["unit_tests"])
            js = by[pid]
            ens = {d: float(np.mean([x[d] for x in js])) for d in DIMS}
            comp = float(sum(ens.values()))
            require(abs(comp - float(f["rubric_composite"])) < 1e-9, f"extension composite mismatch {pid}")
            for d in DIMS:
                require(abs(ens[d] - float(f["rubric_scores"][d])) < 1e-9, "extension dim mismatch")
            pr = prelim[pid]
            row = {
                "prompt_id": pid, "source_dataset": "nvidia/OpenCodeInstruct", "language": "python",
                "prompt_text": f["input"], "unit_tests": tests, "n_unit_tests": len(tests),
                "construction_frame": "tail_extension",
                "source_reference_avg_test_score": self.source_meta.get(pid, {}).get("average_test_score"),
                "reference_local_pass_rate": float(refs[pid]["pass_rate"]),
                "reference_cc": inum(f.get("reference_cc")),
            }
            for d in DIMS:
                row[f"prelim_{d}"] = inum(pr["scores"][d])
            row["prelim_composite"] = inum(pr["composite"])
            for d in DIMS:
                row[f"ens_{d}"] = ens[d]
            row["ens_composite"] = comp
            # The extension source file stores the population SD; recompute the sample SD (ddof=1)
            # so this column matches prompts.ens_composite_sd.
            comps = np.array([x["composite"] for x in js], dtype=float)
            require(abs(comps.std(ddof=0) - float(f["composite_std"])) < 1e-9, "extension SD definition changed")
            row["ens_composite_sd"] = float(comps.std(ddof=1))
            row["ens_n_judges"] = len(js)
            row["display_bin"] = display_bin(comp)
            rows.append(row)
        self.tables["extension_prompts"] = rows

        grows = []
        for path in sorted((t / "scored").glob("*.jsonl")):
            model_key = path.stem
            seen_p = set()
            for r in read_jsonl(path):
                pid = r["prompt_id"]
                if pid not in final or r.get("pass_rate") is None:
                    continue
                require(pid not in seen_p, f"duplicate extension generation {model_key}/{pid}")
                seen_p.add(pid)
                grows.append({"model_key": model_key, "model_display_name": DISPLAY_NAMES[model_key],
                              "prompt_id": pid, "in_matched_five": model_key in MATCHED_FIVE,
                              "pass_rate": float(r["pass_rate"]), "has_code": bool(r.get("has_code")),
                              "gen_temperature": 0.0, "gen_max_output_tokens": 4096})
            require(seen_p == set(final), f"{model_key} covers {len(seen_p)} of 365 extension prompts")
        self.tables["extension_generations"] = sorted(grows, key=lambda r: (r["model_key"], r["prompt_id"]))

        # Fixed-version three-model check (anchors + the 365 retained tail prompts only).
        frows = []
        for path in sorted((self.data / "rebuttal" / "frontier_tail" / "scored").glob("*.jsonl")):
            model_key = path.stem
            seen_p = set()
            for r in read_jsonl(path):
                if r.get("pass_rate") is None:
                    continue
                if r.get("group") == "tail" and r["prompt_id"] not in final:
                    continue
                pid = r["prompt_id"]
                require(pid not in seen_p, f"duplicate fixed-version row {model_key}/{pid}")
                seen_p.add(pid)
                frows.append({"model_key": model_key, "model_display_name": DISPLAY_NAMES[model_key],
                              "prompt_id": pid, "group": r.get("group"),
                              "ens_composite": fnum(r.get("rubric_composite")),
                              "pass_rate": float(r["pass_rate"]), "has_code": bool(r.get("has_code"))})
            require(len(seen_p) == 515, f"{model_key}: expected 515 fixed-version rows, got {len(seen_p)}")
        self.tables["fixed_version_generations"] = sorted(frows, key=lambda r: (r["model_key"], r["group"], r["prompt_id"]))

    # ---- auxiliary robustness sets ------------------------------------------
    def build_passk(self) -> None:
        pk = self.data / "rebuttal" / "passk"
        scores = {}
        for r in read_jsonl(pk / "draw_scores.jsonl"):
            key = (r["model"], r["prompt_id"], int(r["draw"]))
            require(key not in scores, f"duplicate pass@k score {key}")
            scores[key] = r
        code = {}
        for path in sorted((pk / "generations").glob("*.jsonl")):
            for r in read_jsonl(path):
                if r.get("error") is not None:
                    continue
                key = (path.stem, r["prompt_id"], int(r["draw"]))
                require(key not in code, f"duplicate pass@k generation {key}")
                code[key] = r.get("code_cleaned") or ""
        require(set(scores) == set(code), "pass@k scores and generations do not align")
        rows = [{"model_key": k[0], "model_display_name": DISPLAY_NAMES[k[0]], "prompt_id": k[1], "draw": k[2],
                 "code": code[k], "pass_rate": float(scores[k]["pass_rate"]),
                 "gen_temperature": 0.8, "gen_max_output_tokens": 4096} for k in sorted(scores)]
        require(len(rows) == 359 * 4 * 5, f"expected 7,180 pass@k rows, got {len(rows)}")
        self.tables["passk_generations"] = rows

    def build_human(self) -> None:
        hc = self.data / "rebuttal" / "human_calibration"
        sample = list(read_jsonl(hc / "calibration_sample_current.jsonl"))
        sample_by = {r["prompt_id"]: r for r in sample}
        stratum = self._replicate_calibration_selection(sample)
        rows = []
        graders = {"mh": "grader_1", "tian": "grader_2"}
        overlap = None
        filled = {}
        for raw_id, gid in graders.items():
            f = json.loads((hc / f"filled_{raw_id}.json").read_text(encoding="utf-8"))
            k = json.loads((hc / f"key_{raw_id}.json").read_text(encoding="utf-8"))
            require(f["completed"] == f["total"], f"{gid} worksheet incomplete")
            order = {r["prompt_id"]: r["row"] for r in k["rows"]}
            filled[gid] = f["scores"]
            require(set(f["scores"]) == set(order), f"{gid}: filled ids != key ids")
        overlap = set(filled["grader_1"]) & set(filled["grader_2"])
        require(len(filled["grader_1"]) == 200 and len(filled["grader_2"]) == 50 and len(overlap) == 50,
                "unexpected human calibration sizes")
        for raw_id, gid in graders.items():
            k = json.loads((hc / f"key_{raw_id}.json").read_text(encoding="utf-8"))
            for kr in k["rows"]:
                pid = kr["prompt_id"]
                s = filled[gid][pid]
                vals = {d: int(s[d]) for d in DIMS}
                require(all(0 <= v <= 4 for v in vals.values()), "human grade out of range")
                srow = sample_by[pid]
                rows.append({"grader": gid, "prompt_id": pid, "worksheet_row": int(kr["row"]), **vals,
                             "composite": sum(vals.values()), "in_two_grader_overlap": pid in overlap,
                             "sampling_band": srow.get("rubric_bin"),
                             "selection_stratum": stratum.get(pid) if stratum else None,
                             "calibration_priority": fnum(srow.get("calibration_priority"))})
        self.tables["human_calibration"] = sorted(rows, key=lambda r: (r["grader"], r["worksheet_row"]))

    def _replicate_calibration_selection(self, sample: list[dict]) -> dict[str, str] | None:
        """Re-run src/rebuttal/13_reselect_human_calibration.py's seeded selection to recover strata."""
        bins = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 24)]

        def label(c: float) -> str:
            c = round(c)
            for lo, hi in bins:
                if lo <= c <= hi:
                    return f"{lo}-{hi}"
            return "16-24"

        rng = random.Random(20260724)
        ens = {}
        for pid, e in self.ensemble.items():  # file order
            if pid in self.prompts_raw and e.get("composite_mean") is not None:
                ens[pid] = {"dis": float(e.get("composite_std") or 0.0), "bin": label(float(e["composite_mean"]))}
        by_bin = defaultdict(list)
        for pid, e in ens.items():
            by_bin[e["bin"]].append(pid)
        labels = [f"{lo}-{hi}" for lo, hi in bins]
        base, rem = divmod(500, len(labels))
        targets = {lab: base + (1 if i < rem else 0) for i, lab in enumerate(labels)}
        selected, strata = [], {}
        for lab in labels:
            pool = by_bin.get(lab, [])
            want = targets[lab]
            if len(pool) <= want:
                chosen = pool[:]
                for p in chosen:
                    strata[p] = "all_in_band"
            else:
                n_high = int(round(want * 0.5))
                ranked = sorted(pool, key=lambda p: ens[p]["dis"], reverse=True)
                high = ranked[:n_high]
                hs = set(high)
                remaining = [p for p in pool if p not in hs]
                rng.shuffle(remaining)
                low = remaining[: want - len(high)]
                chosen = high + low
                for p in high:
                    strata[p] = "high_disagreement"
                for p in low:
                    strata[p] = "random"
            selected.extend(chosen)
        rng.shuffle(selected)
        ok = selected == [r["prompt_id"] for r in sample]
        self.report["human_calibration_selection_replicated"] = ok
        return strata if ok else None

    def build_paraphrase_and_xl(self) -> None:
        pdir = self.data / "rebuttal" / "paraphrase"
        full = list(read_jsonl(pdir / "paraphrases_full.jsonl"))
        require(len(full) == 150 and all(r.get("paraphrase_error") is None for r in full), "paraphrase set")
        for r in full:
            require(r["original_input"] == self.prompts_raw[r["prompt_id"]]["input"], "paraphrase original text differs")
        self.tables["paraphrase_prompts"] = sorted(
            [{"prompt_id": r["prompt_id"], "sampling_bin": inum(r.get("orig_bin")),
              "original_composite": fnum(r.get("original_composite")),
              "original_prompt_text": r["original_input"],
              "paraphrased_prompt_text": r["paraphrased_input"]} for r in full], key=lambda r: r["prompt_id"])
        para_ids = {r["prompt_id"] for r in full}
        jrows, seen = [], set()
        for r in read_jsonl(pdir / "paraphrase_ensemble_long.jsonl"):
            if r.get("error") is not None or not r.get("scores"):
                continue
            require(r["prompt_id"] in para_ids, "paraphrase score for unknown prompt")
            judge = SCORER_TO_JUDGE[r["scorer_id"]]
            require((r["prompt_id"], judge) not in seen, "duplicate paraphrase score")
            seen.add((r["prompt_id"], judge))
            jrows.append(self._judge_row(r, judge))
        self.tables["paraphrase_judge_scores"] = sorted(jrows, key=lambda r: (r["prompt_id"], r["judge"]))

        ndir = self.data / "rebuttal" / "nonpython"
        prows, jrows, seen = [], [], set()
        for lang in ("java", "cpp"):
            for r in read_jsonl(ndir / f"ported_{lang}_full.jsonl"):
                require(r.get("port_error") is None, "port error present")
                require(r["input"] == self.prompts_raw[r["prompt_id"]]["input"], "ported source text differs")
                prows.append({"prompt_id": r["prompt_id"], "target_language": lang,
                              "original_composite": fnum(r.get("original_composite")),
                              "ported_prompt_text": r["ported_input"]})
            for r in read_jsonl(ndir / f"ensemble_{lang}_long.jsonl"):
                if r.get("error") is not None or not r.get("scores"):
                    continue
                src_id = r["prompt_id"].rsplit("__", 1)[0]
                judge = SCORER_TO_JUDGE[r["scorer_id"]]
                key = (src_id, lang, judge)
                require(key not in seen, f"duplicate cross-language score {key}")
                seen.add(key)
                row = self._judge_row(r, judge)
                row["prompt_id"] = src_id
                jrows.append({"target_language": lang, **row})
        self.tables["cross_language_prompts"] = sorted(prows, key=lambda r: (r["target_language"], r["prompt_id"]))
        self.tables["cross_language_judge_scores"] = sorted(jrows, key=lambda r: (r["target_language"], r["prompt_id"], r["judge"]))

    # ---- outcome audit ----------------------------------------------------------
    def build_audit(self) -> None:
        base = self.data / "independent_audit"
        cases, responses = [], []
        for set_name, (sub, runs) in KNOWN_ANSWER_RUNS.items():
            ka = list(read_jsonl(base / sub / "known_answer_set.jsonl"))
            require(len(ka) == 450, f"{set_name}: expected 450 known-answer cases, found {len(ka)}")
            ids = []
            for c in ka:
                require(c["prompt_id"] in self.prompts_raw, f"{set_name}: known-answer prompt outside the benchmark")
                ids.append(c["case_id"])
                cases.append({"case_id": c["case_id"], "known_answer_set": set_name, "prompt_id": c["prompt_id"],
                              "category": c["category"], "ground_truth": c["ground_truth"],
                              "mutation": json.dumps(c["mutation"], sort_keys=True) if c.get("mutation") else None,
                              "code": c["code"], "harness_pass_rate": fnum(c.get("harness_pass_rate"))})
            for auditor, fname in runs:
                final = self._audit_final_records(base / sub / fname)
                missing = [cid for cid in ids if cid not in final]
                require(not missing, f"{set_name}/{auditor}: {len(missing)} cases without a final record")
                for cid in ids:
                    rec = final[cid]
                    usage = rec.get("usage") or {}
                    responses.append({
                        "known_answer_set": set_name, "auditor": auditor, "case_id": cid,
                        "status": rec.get("status"), "verdict": rec.get("verdict"),
                        "counted_verdict": final_verdict(rec), "finish_reason": rec.get("finish_reason"),
                        "parse_status": rec.get("parse_status"), "reason": rec.get("reason") or None,
                        "response_tail": rec.get("raw_tail") or None,
                        "completion_tokens": inum(usage.get("completion_tokens")) if isinstance(usage, dict) else None})
        require(len({c["case_id"] for c in cases}) == len(cases), "known-answer case ids repeat across sets")
        self.tables["audit_known_answer_cases"] = cases
        self.tables["audit_known_answer_responses"] = responses

        sample = {r["case_id"] for r in read_jsonl(base / "production" / "audit_input_secondary_5pct.jsonl")}
        rows = []
        audit_report = {}
        for auditor, fname in PRODUCTION_RUNS:
            final = (self.primary_audit if auditor == PRIMARY_AUDITOR
                     else self._audit_final_records(base / "production" / fname))
            expected = EXPECTED_MAIN_PROMPTS * EXPECTED_MODELS if auditor == PRIMARY_AUDITOR else len(sample)
            require(len(final) == expected, f"{auditor}: {len(final)} final records, expected {expected}")
            for cid, rec in sorted(final.items()):
                model_key, pid = cid.split(":", 1)
                rows.append({"auditor": auditor, "model_key": model_key, "prompt_id": pid,
                             "in_agreement_sample": cid in sample, "status": rec.get("status"),
                             "verdict": rec.get("verdict"), "counted_verdict": final_verdict(rec),
                             "finish_reason": rec.get("finish_reason"), "parse_status": rec.get("parse_status"),
                             "reason": rec.get("reason") or None, "response_tail": rec.get("raw_tail") or None})
            audit_report[auditor] = {"final_records": len(final),
                                     "handling": dict(Counter(self._handling(r) for r in final.values()))}
        self.tables["audit_production_verdicts"] = rows
        self.report["audit"] = audit_report

    def build_models(self) -> None:
        rows = []
        for key, dev, roles, routes, served, weights, notes in MODELS_META:
            name = DISPLAY_NAMES.get(key) or key.split(":", 1)[-1]
            settings = "; ".join(MODEL_SETTINGS_DOC[r] for r in roles if r in MODEL_SETTINGS_DOC) or \
                "Judges/labelers: one scoring call per prompt with the fixed rubric or taxonomy prompt."
            rows.append({"model_key": key, "display_name": name, "developer": dev, "roles": roles,
                         "access_routes": routes, "served_model": served, "weights": weights,
                         "documented_settings": settings, "notes": notes or None})
        used = {r["model_key"] for t in ("generations", "extension_generations", "passk_generations",
                                         "fixed_version_generations") for r in self.tables[t]}
        require(used <= {r["model_key"] for r in rows}, "model metadata missing for some model_key")
        self.tables["models"] = rows

    # ---- writing ------------------------------------------------------------
    def write_all(self) -> None:
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        for name, spec in CONFIGS.items():
            rows = self.tables[name]
            names = [f[0] for f in spec["fields"]]
            for r in rows:
                extra = set(r) - set(names)
                require(not extra, f"{name}: unexpected columns {extra}")
            schema = pa.schema([pa.field(n, PA_TYPES[k]) for n, k, _ in spec["fields"]])
            table = pa.Table.from_pylist([{n: r.get(n) for n in names} for r in rows], schema=schema)
            cdir = self.out / name
            cdir.mkdir()
            path = cdir / f"{SPLIT}-00000-of-00001.parquet"
            pq.write_table(table, path, compression=PARQUET_COMPRESSION, row_group_size=10000)
            size = path.stat().st_size
            require(size < DATAVERSE_FILE_LIMIT, f"{path.name} exceeds the Dataverse per-file limit")
            self.files[name] = [{"path": path.relative_to(self.out).as_posix(), "rows": len(rows),
                                 "bytes": size, "sha256": sha256_file(path)}]

    def write_docs(self) -> None:
        docs = self.out / "docs"
        docs.mkdir()
        rubric = self._rubric_prompt()
        (docs / "rubric_prompt.txt").write_text(rubric + "\n", encoding="utf-8")
        rubric_hash = hashlib.sha256(rubric.encode("utf-8")).hexdigest()
        hashes = {r["rubric_sha256"] for r in self.tables["judge_scores"]}
        self.report["rubric_hash"] = {"computed": rubric_hash, "judge_rows": sorted(h for h in hashes if h)}
        require(hashes == {rubric_hash}, "judge rows use a rubric hash different from docs/rubric_prompt.txt")
        (docs / "generation_system_prompts.json").write_text(json.dumps({
            "_note": "System prompts observed for generation. A_pipeline: src/data_provenance/02_generate_solutions.py "
                     "(realtime and Anthropic/DashScope batch paths). B_batch_scripts: OpenAI Batch requests (user turn "
                     "prefixed with 'Task:'). Gemini batch requests carried the instruction in the user turn. "
                     "C_extension_passk: extension and pass@k runs. Per-row variants are recorded in "
                     "generations.gen_system_prompt_variant where a batch request record was retained.",
            **GEN_SYSTEM_PROMPTS}, indent=2) + "\n", encoding="utf-8")

    def _rubric_prompt(self) -> str:
        src = REPO_ROOT / "src" / "data_provenance" / "05_score_complexity_rubric.py"
        if not src.exists():
            src = self.root / "src" / "data_provenance" / "05_score_complexity_rubric.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SYSTEM_PROMPT" for t in node.targets):
                return ast.literal_eval(node.value)
        raise RuntimeError("SYSTEM_PROMPT not found in 05_score_complexity_rubric.py")

    # ---- sanity checks -------------------------------------------------------
    def sanity(self) -> None:
        P = self.tables["prompts"]
        G = self.tables["generations"]
        J = self.tables["judge_scores"]
        chk: dict[str, Any] = {}
        chk["main_prompts"] = len(P)
        chk["generation_rows"] = len(G)
        chk["models"] = len({g["model_key"] for g in G})
        chk["judge_score_rows"] = len(J)
        chk["judge_rows_by_judge"] = dict(Counter(r["judge"] for r in J))
        chk["judges_per_prompt"] = dict(Counter(Counter(r["prompt_id"] for r in J).values()))
        chk["extension_prompts"] = len(self.tables["extension_prompts"])
        chk["extension_display_bins"] = dict(sorted(Counter(r["display_bin"] for r in self.tables["extension_prompts"]).items()))

        # Pass rates.
        pr = np.array([g["pass_rate"] for g in G])
        hr = np.array([g["harness_pass_rate"] for g in G], dtype=float)
        chk["mean_pass_rate_generation_level"] = float(pr.mean())
        chk["mean_pass_rate_prompt_level"] = float(np.mean([p["mean_pass_rate"] for p in P]))
        chk["mean_harness_pass_rate_generation_level"] = float(np.nanmean(hr))
        comp = np.array([p["ens_composite"] for p in P])
        later = np.array([p["construction_frame"] == "later_candidate" for p in P])
        # Headline breakpoints: 14.0 for the primary outcome, 13.75 for the reviewed version's.
        for label, column, gamma in (("primary", "mean_pass_rate", 14.0),
                                     ("reviewed", "mean_pass_rate_reviewed", 13.75)):
            mp = np.array([p[column] for p in P])
            low = comp <= gamma
            chk[f"regime_{label}_at_{gamma}"] = {
                "n_low": int(low.sum()), "pass_low": float(mp[low].mean()),
                "n_high": int((~low).sum()), "pass_high": float(mp[~low].mean()),
                "later_frame_share_low": float(later[low].mean()),
                "later_frame_share_high": float(later[~low].mean())}
        chk["mean_pass_rate_reviewed_generation_level"] = float(np.mean([g["pass_rate_reviewed"] for g in G]))
        chk["independent_audit_handling"] = dict(Counter(g["independent_audit_handling"] for g in G))
        chk["model_returned_no_response"] = dict(Counter(g["model_key"] for g in G if g["model_returned_no_response"]))
        dbins = Counter(p["display_bin"] for p in P)
        chk["main_display_bin_counts"] = {str(b): dbins[b] for b in sorted(dbins)}
        frames = defaultdict(list)
        for p in P:
            frames[p["construction_frame"]].append(p)
        chk["frames"] = {k: {"n": len(v), "mean_composite": float(np.mean([x["ens_composite"] for x in v])),
                             "mean_pass_rate": float(np.mean([x["mean_pass_rate"] for x in v])),
                             "mean_harness_pass_rate": float(np.mean([x["mean_harness_pass_rate"] for x in v])),
                             "ref_avg_test_score_lt_1": sum(1 for x in v if (x["source_reference_avg_test_score"] is not None and x["source_reference_avg_test_score"] < 1.0)),
                             "ref_avg_test_score_eq_0": sum(1 for x in v if x["source_reference_avg_test_score"] == 0.0)}
                         for k, v in frames.items()}
        edges = [(-1e9, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 24)]
        chk["analyzed_index_band_counts"] = [int(((comp > lo) & (comp <= hi)).sum()) for lo, hi in edges]
        chk["prelim_band_counts"] = dict(Counter(p["prelim_band"] for p in P))
        chk["reference_cc_available"] = sum(1 for p in P if p["reference_cc"] is not None)

        # Lizard availability and reverse-threshold cell.
        cc_ok = [g for g in G if g["output_cc_lizard"] is not None]
        chk["generations_with_lizard_cc"] = len(cc_ok)
        comp_by = {p["prompt_id"]: p["ens_composite"] for p in P}
        zero = [g for g in cc_ok if g["pass_rate"] == 0.0]
        cell = [g for g in zero if comp_by[g["prompt_id"]] > 8 and g["output_cc_lizard"] <= 10]
        chk["zero_pass_complete_cases"] = len(zero)
        chk["reverse_threshold_cell"] = len(cell)
        chk["reverse_threshold_share"] = len(cell) / len(zero) if zero else None
        chk["zero_pass_without_cc"] = sum(1 for g in G if g["pass_rate"] == 0.0 and g["output_cc_lizard"] is None)

        # ICC(2,1) on complete four-judge composites.
        by = defaultdict(dict)
        for r in J:
            by[r["prompt_id"]][r["judge"]] = r["composite"]
        judges = sorted(JUDGE_FILES)
        complete = [[v[j] for j in judges] for v in by.values() if len(v) == 4]
        chk["icc_2_1_composite"] = {"n_complete": len(complete), "icc": icc_2_1(np.array(complete, dtype=float))}

        # Human calibration.
        H = self.tables["human_calibration"]
        ens = {p["prompt_id"]: p["ens_composite"] for p in P}
        g1 = {r["prompt_id"]: r["composite"] for r in H if r["grader"] == "grader_1"}
        g2 = {r["prompt_id"]: r["composite"] for r in H if r["grader"] == "grader_2"}
        ov = sorted(g2)
        chk["human"] = {
            "grader_1_n": len(g1), "grader_2_n": len(g2),
            "grader_1_vs_ensemble_pearson": pearson([g1[p] for p in g1], [ens[p] for p in g1]),
            "grader_1_vs_ensemble_spearman": spearman([g1[p] for p in g1], [ens[p] for p in g1]),
            "grader_1_vs_ensemble_icc_2_1": icc_2_1(np.array([[g1[p], ens[p]] for p in g1], dtype=float)),
            "grader_1_mean_offset": float(np.mean([g1[p] - ens[p] for p in g1])),
            "overlap_grader_1_vs_ensemble_pearson": pearson([g1[p] for p in ov], [ens[p] for p in ov]),
            "overlap_grader_2_vs_ensemble_pearson": pearson([g2[p] for p in ov], [ens[p] for p in ov]),
            "overlap_grader_1_vs_grader_2_pearson": pearson([g1[p] for p in ov], [g2[p] for p in ov]),
            "strata": {f"{a}|{b}": c for (a, b), c in
                       Counter((r["grader"], r["selection_stratum"]) for r in H).items()},
        }

        # Extension: matched-five means at bins 15/16 (extension vs original).
        E = self.tables["extension_prompts"]
        EG = self.tables["extension_generations"]
        ext_bin = {r["prompt_id"]: r["display_bin"] for r in E}
        m5 = defaultdict(list)
        for g in EG:
            if g["in_matched_five"]:
                m5[g["prompt_id"]].append(g["pass_rate"])
        orig = defaultdict(list)
        for g in G:  # raw harness on both sides: the extension runs saved no code to audit
            if g["model_key"] in MATCHED_FIVE:
                orig[g["prompt_id"]].append(g["harness_pass_rate"])
        pbin = {p["prompt_id"]: p["display_bin"] for p in P}
        ext_summary = {}
        for b in (15, 16, 17, 18):
            e_vals = [np.mean(v) for p, v in m5.items() if ext_bin[p] == b]
            o_vals = [np.mean(v) for p, v in orig.items() if pbin[p] == b]
            ext_summary[str(b)] = {"extension_n": len(e_vals), "extension_mean": float(np.mean(e_vals)) if e_vals else None,
                                   "original_n": len(o_vals), "original_mean": float(np.mean(o_vals)) if o_vals else None}
        chk["extension_matched_five"] = ext_summary
        FV = self.tables["fixed_version_generations"]
        fv = defaultdict(list)
        for r in FV:
            fv[(r["model_key"], display_bin(r["ens_composite"]))].append(r["pass_rate"])
        chk["fixed_version_bins_15_16"] = {f"{m}|{b}": {"n": len(v), "mean": float(np.mean(v))}
                                           for (m, b), v in sorted(fv.items()) if b in (15, 16)}

        # Pass@k, paraphrase, cross-language.
        chk["passk_rows"] = len(self.tables["passk_generations"])
        pj = defaultdict(list)
        for r in self.tables["paraphrase_judge_scores"]:
            pj[r["prompt_id"]].append(r)
        para_comp = {p: float(sum(np.mean([x[d] for x in v]) for d in DIMS)) for p, v in pj.items()}
        ids = sorted(para_comp)
        diffs = [para_comp[p] - ens[p] for p in ids]
        chk["paraphrase"] = {"n": len(ids), "judge_rows": len(self.tables["paraphrase_judge_scores"]),
                             "spearman_vs_current_ensemble": spearman([ens[p] for p in ids], [para_comp[p] for p in ids]),
                             "pearson_vs_current_ensemble": pearson([ens[p] for p in ids], [para_comp[p] for p in ids]),
                             "mean_shift": float(np.mean(diffs)),
                             "within_one_point": float(np.mean(np.abs(diffs) <= 1.0))}
        xl = defaultdict(list)
        for r in self.tables["cross_language_judge_scores"]:
            xl[(r["target_language"], r["prompt_id"])].append(r)
        xl_summary = {}
        for lang in ("java", "cpp"):
            keys = sorted(p for (l, p) in xl if l == lang)
            lc = [float(sum(np.mean([x[d] for x in xl[(lang, p)]]) for d in DIMS)) for p in keys]
            xl_summary[lang] = {"n": len(keys), "pearson_vs_python_ensemble": pearson([ens[p] for p in keys], lc)}
        chk["cross_language"] = xl_summary
        chk["task_type_counts_primary"] = dict(Counter(p["task_type"] for p in P))

        # Timeframe of generation / scoring timestamps.
        ts = sorted(t for t in (g["generated_at"] for g in G) if t)
        js = sorted(t for t in (r["scored_at"] for r in J) if t)
        chk["timeframe"] = {"generations": [ts[0], ts[-1]] if ts else None,
                            "main_judge_scores": [js[0], js[-1]] if js else None}
        self.report["sanity"] = chk

    # ---- sanitization ----------------------------------------------------------
    SANITIZE_PATTERNS = {
        "sk-": r"sk-[A-Za-z0-9_\-]{6,}",
        "key (case-insensitive, any)": r"(?i)key",
        "credential-like literal": r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{16,}['\"]",
        "herna": r"(?i)herna",
        "Bearer": r"Bearer",
        "azure.com": r"(?i)azure\.com",
        "openrouter": r"(?i)openrouter",
        "windows user path": r"(?i)[a-z]:[\\/]+users[\\/]",
        "ProgD": r"(?i)progd",
        "gmail": r"(?i)gmail\.com",
        "batch/request ids": r"(msgbatch_|batch_[0-9a-f]{20,}|chatcmpl-|req_[A-Za-z0-9]{16,})",
        "azure resource names": r"(?i)(datapipeline0|cognitiveservices|services\.ai\.azure)",
    }

    def sanitize_scan(self) -> None:
        pats = {k: re.compile(v) for k, v in self.SANITIZE_PATTERNS.items()}
        results: dict[str, dict] = {}
        for name, info in self.files.items():
            table = pq.read_table(self.out / info[0]["path"])
            for col in table.column_names:
                t = table.schema.field(col).type
                if not (pa.types.is_string(t) or pa.types.is_list(t)):
                    continue
                values = table.column(col).to_pylist()
                for v in values:
                    texts = v if isinstance(v, list) else [v]
                    for text in texts:
                        if not text:
                            continue
                        for pname, pat in pats.items():
                            hits = pat.findall(text) if pname != "key (case-insensitive, any)" else None
                            if pname == "key (case-insensitive, any)":
                                n = len(pat.findall(text))
                                if n:
                                    slot = results.setdefault(pname, {"matches": 0, "by_column": Counter(), "examples": []})
                                    slot["matches"] += n
                                    slot["by_column"][f"{name}.{col}"] += n
                                continue
                            if hits:
                                slot = results.setdefault(pname, {"matches": 0, "by_column": Counter(), "examples": []})
                                slot["matches"] += len(hits)
                                slot["by_column"][f"{name}.{col}"] += len(hits)
                                if len(slot["examples"]) < 8:
                                    m = pat.search(text)
                                    a, b = max(0, m.start() - 50), min(len(text), m.end() + 50)
                                    slot["examples"].append({"where": f"{name}.{col}", "context": text[a:b]})
        for path in self.out.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".json", ".txt"):
                text = path.read_text(encoding="utf-8")
                for pname, pat in pats.items():
                    if pname == "key (case-insensitive, any)":
                        continue
                    n = len(pat.findall(text))
                    if n:
                        slot = results.setdefault(pname, {"matches": 0, "by_column": Counter(), "examples": []})
                        slot["matches"] += n
                        slot["by_column"][path.relative_to(self.out).as_posix()] += n
        for v in results.values():
            v["by_column"] = dict(v["by_column"].most_common())
        self.report["sanitization_scan"] = {k: results.get(k, {"matches": 0}) for k in self.SANITIZE_PATTERNS}

    # ---- README field tables + Croissant --------------------------------------
    def fill_readme(self) -> None:
        readme = RELEASE_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        # Reverse-threshold numbers in the card come from the release itself.
        comp_by = {p["prompt_id"]: p["ens_composite"] for p in self.tables["prompts"]}
        zero = [g for g in self.tables["generations"] if g["output_cc_lizard"] is not None and g["pass_rate"] == 0.0]
        cell = [g for g in zero if comp_by[g["prompt_id"]] > 8 and g["output_cc_lizard"] <= 10]
        comp = np.array([p["ens_composite"] for p in self.tables["prompts"]])
        regimes = {}
        for token, column, gamma in (("__REG_PRIMARY__", "mean_pass_rate", 14.0),
                                     ("__REG_REVIEWED__", "mean_pass_rate_reviewed", 13.75)):
            mp = np.array([p[column] for p in self.tables["prompts"]])
            low = comp <= gamma
            regimes[token] = (f"{int(low.sum()):,} at {mp[low].mean():.3f} / "
                              f"{int((~low).sum()):,} at {mp[~low].mean():.3f}")
        # Matched-five raw-harness means at bins 15 and 16, extension vs original.
        ext_bin = {r["prompt_id"]: r["display_bin"] for r in self.tables["extension_prompts"]}
        pbin = {p["prompt_id"]: p["display_bin"] for p in self.tables["prompts"]}
        ext, orig = defaultdict(list), defaultdict(list)
        for g in self.tables["extension_generations"]:
            if g["in_matched_five"]:
                ext[g["prompt_id"]].append(g["pass_rate"])
        for g in self.tables["generations"]:
            if g["model_key"] in MATCHED_FIVE:
                orig[g["prompt_id"]].append(g["harness_pass_rate"])
        means = {(src, b): np.mean([np.mean(v) for p, v in d.items() if bins[p] == b])
                 for src, d, bins in (("ext", ext, ext_bin), ("orig", orig, pbin)) for b in (15, 16)}
        ext_text = "; ".join(f"{means[('ext', b)]:.3f} vs {means[('orig', b)]:.3f}" for b in (15, 16))
        for token, value in (("__RT_CELL__", f"{len(cell):,}"), ("__RT_ZERO__", f"{len(zero):,}"),
                             ("__RT_SHARE__", f"{100 * len(cell) / len(zero):.1f}%"),
                             ("__EXT_MATCHED__", ext_text), *regimes.items()):
            text = text.replace(token, value)
        parts = []
        for name, spec in CONFIGS.items():
            info = self.files[name][0]
            parts.append(f"#### `{name}` ({info['rows']:,} rows)\n\n{spec['description']}\n\n| Field | Type | Description |\n|---|---|---|")
            for fname, kind, desc in spec["fields"]:
                parts.append(f"| `{fname}` | {kind} | {desc.replace('|', '/')} |")
            parts.append("")
        block = "\n".join(parts)
        begin, end = "<!-- FIELDS:BEGIN -->", "<!-- FIELDS:END -->"
        require(begin in text and end in text, "README.md is missing the FIELDS markers")
        text = text.split(begin)[0] + begin + "\n" + block + "\n" + end + text.split(end, 1)[1]
        # Inventory table.
        ib, ie = "<!-- INVENTORY:BEGIN -->", "<!-- INVENTORY:END -->"
        if ib in text and ie in text:
            rows = ["| Config | Rows | File size |", "|---|---:|---:|"]
            for name in CONFIGS:
                info = self.files[name][0]
                rows.append(f"| `{name}` | {info['rows']:,} | {info['bytes'] / 1e6:.2f} MB |")
            text = text.split(ib)[0] + ib + "\n" + "\n".join(rows) + "\n" + ie + text.split(ie, 1)[1]
        readme.write_text(text, encoding="utf-8")
        shutil.copyfile(readme, self.out / "README.md")

    def write_croissant(self) -> None:
        rai_path = RELEASE_DIR / "croissant_rai_fields.json"
        rai = json.loads(rai_path.read_text(encoding="utf-8")) if rai_path.exists() else {}
        context = {
            "@language": "en", "@vocab": "https://schema.org/", "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
            "prov": "http://www.w3.org/ns/prov#", "dct": "http://purl.org/dc/terms/",
            "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract", "field": "cr:field", "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes",
            "isArray": "cr:isArray", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath",
            "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "recordSet": "cr:recordSet",
            "references": "cr:references", "regex": "cr:regex", "sdVersion": "cr:sdVersion",
            "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField",
            "transform": "cr:transform", "containedIn": "cr:containedIn",
            "arrayShape": "cr:arrayShape", "equivalentProperty": "cr:equivalentProperty",
            "fileProperty": "cr:fileProperty", "path": "cr:path", "repeated": "cr:repeated",
            "replace": "cr:replace", "samplingRate": "cr:samplingRate",
        }
        distribution, record_sets = [], []
        for name, spec in CONFIGS.items():
            info = self.files[name][0]
            fid = f"{name}-parquet"
            distribution.append({"@type": "cr:FileObject", "@id": fid, "name": fid,
                                 "description": f"Parquet file for the {name} config.",
                                 "contentUrl": info["path"], "encodingFormat": "application/x-parquet",
                                 "contentSize": f"{info['bytes']} B", "sha256": info["sha256"]})
            fields = []
            for fname, kind, desc in spec["fields"]:
                f = {"@type": "cr:Field", "@id": f"{name}/{fname}", "name": fname, "description": desc,
                     "dataType": CR_TYPES[kind],
                     "source": {"fileObject": {"@id": fid}, "extract": {"column": fname}}}
                if kind == LS:
                    f["isArray"] = True
                    f["arrayShape"] = "-1"
                ref = spec.get("references", {}).get(fname)
                if ref:
                    f["references"] = {"field": {"@id": ref}}
                fields.append(f)
            rs = {"@type": "cr:RecordSet", "@id": name, "name": name, "description": spec["description"],
                  "field": fields}
            if spec.get("key"):
                rs["key"] = {"@id": f"{name}/{spec['key']}"}
            record_sets.append(rs)
        doc = {
            "@context": context,
            "@type": "sc:Dataset",
            "conformsTo": "http://mlcommons.org/croissant/1.1",
            "name": "complexity-kink",
            "description": ("Prompt-side structural-complexity scores (four out-of-panel LLM judges and human "
                            "calibration), 105,000 generated Python solutions from 21 LLMs with unit-test outcomes, "
                            "an independent LLM audit of every test verdict (with its known-answer validation sets), "
                            "and Lizard output complexity, plus robustness subsets, for 5,000 OpenCodeInstruct "
                            "prompts and a 365-prompt audit-clean high-complexity extension."),
            "license": "https://creativecommons.org/licenses/by/4.0/",
            # The code repository links the dataset; replace with the dataset page once it is published.
            "url": "https://github.com/uwm-se/ComplexityKink",
            "version": "1.0.0",
            "datePublished": dt.date.today().isoformat(),
            "creator": [{"@type": "sc:Person", "name": "Michael Hernandez",
                         "affiliation": "University of Wisconsin-Milwaukee"},
                        {"@type": "sc:Person", "name": "Tian Zhao",
                         "affiliation": "University of Wisconsin-Milwaukee"}],
            "citeAs": ("@inproceedings{hernandez2026complexitykink, title={The Complexity Kink: LLM Rubric "
                       "Instruments for Causal Inference on Code Generation Reliability}, author={Hernandez, "
                       "Michael and Zhao, Tian}, booktitle={Advances in Neural Information Processing Systems, "
                       "Evaluations and Datasets Track}, year={2026}}"),
            "keywords": ["code generation", "LLM evaluation", "LLM-as-judge", "cyclomatic complexity",
                         "benchmark", "Python", "reliability breakpoints"],
            "inLanguage": "en",
            "isLiveDataset": False,
            "distribution": distribution,
            "recordSet": record_sets,
        }
        for k, v in rai.items():
            if k.startswith("_") or k == "@context":
                continue
            doc[k] = v
        (self.out / "croissant.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def finish(self) -> None:
        total = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        self.report["inventory"] = {name: info[0] for name, info in self.files.items()}
        self.report["release_total_bytes"] = total
        self.report["release_dir"] = "data/public_release"
        (RELEASE_DIR / "build_report.json").write_text(json.dumps(self.report, indent=2, default=str) + "\n", encoding="utf-8")


def find_data_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "data" / "stage_d" / "scored_combined").is_dir():
            return cand
    raise SystemExit("Could not locate data/stage_d/scored_combined; pass --data-root.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Directory containing the retained data/ bundle (auto-detected upward from this repo).")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default <data-root>/data/public_release).")
    ap.add_argument("--no-source-scan", action="store_true",
                    help="Skip streaming the 21 GB OpenCodeInstruct extraction (source_reference_avg_test_score becomes null).")
    args = ap.parse_args()
    data_root = (args.data_root or find_data_root(REPO_ROOT)).resolve()
    out = (args.out or data_root / "data" / "public_release").resolve()
    b = Builder(data_root, out, source_scan=not args.no_source_scan)
    print(f"data root: {data_root}\noutput:    {out}")
    b.load_main()
    print("loaded prompts, current ensemble, 19,997 judge rows")
    ext_ids = {r["prompt_id"]: r["unit_tests"] for r in read_jsonl(data_root / "data/rebuttal/tail_topup/tail_topup_final.jsonl")}
    wanted = {pid: r["unit_tests"] for pid, r in b.prompts_raw.items()}
    wanted.update(ext_ids)
    b.scan_source(wanted)
    print(f"source scan: {b.report['source_scan']}")
    b.load_batch_records()
    b.build_generations()
    print("built generations (105,000 rows)")
    b.build_prompts()
    b.build_extension()
    b.build_passk()
    b.build_human()
    b.build_paraphrase_and_xl()
    b.build_audit()
    print("built outcome-audit tables")
    b.build_models()
    b.write_all()
    b.write_docs()
    b.fill_readme()
    b.write_croissant()
    b.sanity()
    b.sanitize_scan()
    b.finish()
    s = b.report["sanity"]
    print(json.dumps({k: s[k] for k in ("main_prompts", "generation_rows", "judge_score_rows", "extension_prompts",
                                        "generations_with_lizard_cc", "icc_2_1_composite",
                                        "mean_pass_rate_prompt_level")}, indent=2))
    print(f"total release bytes: {b.report['release_total_bytes']:,}")
    print(f"report: {RELEASE_DIR / 'build_report.json'}")


if __name__ == "__main__":
    main()
