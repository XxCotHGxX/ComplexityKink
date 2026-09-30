---
pretty_name: "Complexity Kink: Prompt-Side Complexity Scores, LLM Code Generations, and Test Outcomes"
license: cc-by-4.0
language:
- en
tags:
- code
- code-generation
- python
- llm-evaluation
- llm-as-a-judge
- benchmark
- cyclomatic-complexity
task_categories:
- text-generation
size_categories:
- 100K<n<1M
annotations_creators:
- machine-generated
- expert-generated
source_datasets:
- nvidia/OpenCodeInstruct
configs:
- config_name: prompts
  default: true
  data_files:
  - split: test
    path: prompts/test-*.parquet
- config_name: judge_scores
  data_files:
  - split: test
    path: judge_scores/test-*.parquet
- config_name: generations
  data_files:
  - split: test
    path: generations/test-*.parquet
- config_name: extension_prompts
  data_files:
  - split: test
    path: extension_prompts/test-*.parquet
- config_name: extension_judge_scores
  data_files:
  - split: test
    path: extension_judge_scores/test-*.parquet
- config_name: extension_generations
  data_files:
  - split: test
    path: extension_generations/test-*.parquet
- config_name: fixed_version_generations
  data_files:
  - split: test
    path: fixed_version_generations/test-*.parquet
- config_name: passk_generations
  data_files:
  - split: test
    path: passk_generations/test-*.parquet
- config_name: human_calibration
  data_files:
  - split: test
    path: human_calibration/test-*.parquet
- config_name: task_type_labels
  data_files:
  - split: test
    path: task_type_labels/test-*.parquet
- config_name: paraphrase_prompts
  data_files:
  - split: test
    path: paraphrase_prompts/test-*.parquet
- config_name: paraphrase_judge_scores
  data_files:
  - split: test
    path: paraphrase_judge_scores/test-*.parquet
- config_name: cross_language_prompts
  data_files:
  - split: test
    path: cross_language_prompts/test-*.parquet
- config_name: cross_language_judge_scores
  data_files:
  - split: test
    path: cross_language_judge_scores/test-*.parquet
- config_name: audit_known_answer_cases
  data_files:
  - split: test
    path: audit_known_answer_cases/test-*.parquet
- config_name: audit_known_answer_responses
  data_files:
  - split: test
    path: audit_known_answer_responses/test-*.parquet
- config_name: audit_production_verdicts
  data_files:
  - split: test
    path: audit_production_verdicts/test-*.parquet
- config_name: models
  data_files:
  - split: test
    path: models/test-*.parquet
---

# Complexity Kink: prompt-side complexity, LLM code generations, and test outcomes

This dataset accompanies the NeurIPS 2026 Evaluations & Datasets track paper
*The Complexity Kink: LLM Rubric Instruments for Causal Inference on Code
Generation Reliability* (Michael Hernandez and Tian Zhao, University of
Wisconsin-Milwaukee). Code and analysis: <https://github.com/uwm-se/ComplexityKink>.

It scores the structural complexity of 5,000 Python programming tasks from the
prompt alone, never from a model's code or test results, and pairs that score with
105,000 generated solutions (21 LLMs x 5,000 prompts), their unit-test outcomes,
an independent LLM audit of every test verdict, and the cyclomatic complexity of
the generated code. It also contains the paper's robustness data: a 365-prompt
audit-clean high-complexity extension, a repeated-sampling subset, blinded human
rubric grades, task-type labels, paraphrase and Java/C++ rescoring sets, and the
known-answer sets used to choose and check the auditor.

The motivation is a measurement problem. Complexity computed from generated code
is failure-dependent: a hard prompt can yield a short failing program and land in
a low-complexity bin. Scoring the prompt instead avoids that problem. In this
data, 3,561 of 14,247 failed generations (outcome 0) with computable
output complexity (25.0%) pair a prompt composite above 8 with output
complexity of at most 10.

## Quick start

```python
from datasets import load_dataset

repo = "TODO-author/complexity-kink"  # TODO(author): the Hugging Face repo id, once created
prompts = load_dataset(repo, "prompts", split="test")
gens = load_dataset(repo, "generations", split="test")

# Prompt-level primary outcome: audited pass rate averaged over the 21 models.
df = prompts.to_pandas()[["prompt_id", "ens_composite", "mean_pass_rate",
                          "mean_pass_rate_reviewed", "mean_harness_pass_rate", "construction_frame"]]
```

All configs share `prompt_id` (the OpenCodeInstruct record `id`) and model
configs share `model_key`. Every config has a single `test` split: this is an
evaluation dataset and should not be used for training.

## Configs

<!-- INVENTORY:BEGIN -->
| Config | Rows | File size |
|---|---:|---:|
| `prompts` | 5,000 | 2.32 MB |
| `judge_scores` | 19,997 | 0.34 MB |
| `generations` | 105,000 | 32.93 MB |
| `extension_prompts` | 365 | 0.12 MB |
| `extension_judge_scores` | 1,460 | 0.03 MB |
| `extension_generations` | 2,190 | 0.01 MB |
| `fixed_version_generations` | 1,545 | 0.01 MB |
| `passk_generations` | 7,180 | 1.76 MB |
| `human_calibration` | 250 | 0.01 MB |
| `task_type_labels` | 6,500 | 0.11 MB |
| `paraphrase_prompts` | 150 | 0.07 MB |
| `paraphrase_judge_scores` | 600 | 0.01 MB |
| `cross_language_prompts` | 234 | 0.07 MB |
| `cross_language_judge_scores` | 936 | 0.02 MB |
| `audit_known_answer_cases` | 900 | 0.19 MB |
| `audit_known_answer_responses` | 3,150 | 0.34 MB |
| `audit_production_verdicts` | 115,500 | 13.96 MB |
| `models` | 35 | 0.01 MB |
<!-- INVENTORY:END -->

| Group | Configs | What it is |
|---|---|---|
| Main benchmark | `prompts`, `judge_scores`, `generations` | 5,000 prompts, 19,997 per-judge rubric rows, 105,000 generations |
| High-complexity extension | `extension_prompts`, `extension_judge_scores`, `extension_generations`, `fixed_version_generations` | 365 audit-clean, reference-verified prompts at ensemble composite >= 15 |
| Robustness checks | `passk_generations`, `human_calibration`, `task_type_labels`, `paraphrase_prompts`, `paraphrase_judge_scores`, `cross_language_prompts`, `cross_language_judge_scores` | Repeated sampling, human grades, task taxonomy, rewrites, language re-expressions |
| Outcome audit | `audit_known_answer_cases`, `audit_known_answer_responses`, `audit_production_verdicts` | Known-answer sets used to choose and confirm the auditor, every auditor's responses on them, and the production verdicts of the adopted auditor (all 105,000 generations) and two second auditors (5% sample) |
| Metadata | `models` | Developer, role, access route, and documented settings for every model |

Supporting files: `docs/rubric_prompt.txt` (the exact rubric given to every judge;
SHA-256 `3bbf9bb0...a8`), `docs/generation_system_prompts.json`, and
`croissant.json` (Croissant 1.1 metadata with Responsible-AI fields).

## Key definitions

- **Prompt-side index (`ens_composite`)**: four out-of-panel LLM judges (o4-mini,
  gpt-5.5, Llama 4 Maverick, Command A) score six dimensions (branching,
  iteration, state, data structures, edge cases, composition), each 0-4. The
  index is the per-dimension mean over judges, summed (0-24). This is the
  variable analyzed in the paper.
- **Preliminary score (`prelim_*`)**: a single o4-mini rating used only to
  stratify sampling (834/834/833/833/833/833 prompts across bands 0-3, 4-6, 7-9,
  10-12, 13-15, 16-24). It is not the analysis index.
- **Construction frame**: `earlier_retained` (2,246 prompts kept from an earlier
  prefix-scan draw) or `later_candidate` (2,754 later candidates, including
  deliberate high reference-complexity supplementation). The two frames differ
  sharply in mean index (7.92 vs 11.42) and mean pass rate (0.795 vs 0.897 on the
  primary outcome). Each model's two frames were also generated in separate runs
  (weeks apart, sometimes through different routes or token limits; see the
  `gen_*` fields and the `models` config), so the frame marks the generation run
  as well as the prompt source.
- **Display bin**: `floor(composite + 0.5)`, so bin *b* holds [b-0.5, b+0.5).
  Bins are for description only; breakpoints are estimated on the unbinned index.
- **Three outcome definitions** (all in `generations`):
  - `harness_pass_rate`: the fraction of unit-test assertions that passed when
    the generated code was executed.
  - `pass_rate` (**primary**, used in the camera-ready paper): an independent LLM
    auditor (MiMo-V2.6-Pro, from a vendor used nowhere else in the study) read
    each generation's task, code, tests, and harness result. A `correct` verdict
    sets `pass_rate` to 1.0 and `incorrect` to 0.0; uncertain, unparseable, or
    cut-off responses keep the harness value (`independent_audit_handling`
    says which). Generations with no code are 0.0 by rule. For 98.9% of rows the
    value is therefore a binary verdict. Mean over all generations: 0.851.
  - `pass_rate_reviewed`: the reviewed (submitted) version's outcome, in which an
    o4-mini audit set values to 1.0 or 0.0 in the earlier frame only (23,848 of
    47,166 earlier-frame rows changed). Mean: 0.820, against 0.792 for
    `harness_pass_rate`.
- **No-response generations**: 157 generations (156 from Gemini 3.1 Pro Preview,
  from one batch) came back from the model API with no response at all
  (`model_returned_no_response`). They count as failures in the primary
  analysis; the paper also reports results with them treated as missing.
- **Output complexity (`output_cc_lizard`)**: McCabe cyclomatic complexity of
  the generated code computed with Lizard 1.21.0 and summed over functions;
  available for 103,948 of 105,000 generations. It is never the prompt index and
  never reference-solution complexity (`reference_cc`).

## How the data was built

1. **Source prompts.** Python records from
   [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
   with non-trivial unit tests. The earlier frame came from a 200,000-record
   source-ordered prefix scan; later candidates added deliberate high
   reference-complexity supplementation and passed automated contract and
   test-quality filtering. Prompt text and tests are unchanged from the source.
2. **Stratified sampling** on the preliminary o4-mini rubric score.
3. **Ensemble scoring.** After the set was locked, four judges outside the
   evaluated panel rescored every prompt with the same rubric prompt: 19,997
   valid rows (4,998 prompts with four judges, one with three, one with two).
   Composite ICC(2,1) = 0.872 on the 4,998 complete prompts. The judges saw only
   the prompt text. This scoring took place after most generations already
   existed, but it never used them.
4. **Generation.** Each of 21 models produced one solution per prompt through
   provider APIs, batch APIs, OpenRouter, or locally served quantized builds
   (see the `models` config). Three system-prompt variants were used
   (`docs/generation_system_prompts.json`); per-row settings are in the `gen_*`
   fields where a request record was retained. Raw API responses are not
   released; `code` is the cleaned code that was executed and measured.
5. **Execution and measurement.** Each unit-test assertion was executed in a
   fresh Python process (isolated mode) with a 5-second timeout; Lizard measured
   the generated code.
6. **Outcome audit.** Before any auditor was scored, a protocol fixed a
   known-answer set and a selection rule (`audit_known_answer_*` configs). No
   candidate met the rule in full; the adopted auditor (MiMo-V2.6-Pro) missed the
   clean-accuracy threshold by three cases on set A and met every threshold on a
   fresh set B. It then audited all 105,000 generations; two other auditors
   audited a seeded 5% sample for agreement (`audit_production_verdicts`). The
   protocol, its amendments, and the code are in the GitHub repository
   (`docs/independent_audit_protocol.md`, `src/audit/`).
7. **Robustness data.** Task-type labels (nine categories; o4-mini on all
   prompts, three more labelers on 500), blinded human grades, 150 paraphrases
   and 117 Java/C++ re-expressions written by DeepSeek-V3.2 and rescored by the
   judges, and a 359-prompt x 4-model x 5-draw repeated-sampling subset at
   temperature 0.8 (with code; its outcomes are raw harness values, as these
   generations were not included in the independent audit).

### The 365-prompt audit-clean extension

The extension was selected on prompt-side information only. Unused
OpenCodeInstruct candidates (none in the main benchmark or in the excluded 24-bin
set) with preliminary o4-mini composite >= 15 were rescored by the four judges;
1,129 reached ensemble composite >= 15. A contract and test-quality audit then
removed every prompt carrying an exclusion flag (764 prompts; flags included
hidden test callables, I/O-style prompts with callable tests, external fixtures
or globals, and duplicate tests), leaving 365. All 365 reference solutions pass
every test when re-executed. Display bins 15/16/17/18 hold 218/133/11/3 prompts.
Six Azure-hosted models generated one solution each at temperature 0.0 with a
system prompt (variant C) that, unlike the main runs, tells the model to define
exactly the functions the task names (five of the models are also in the main
panel and form the paper's matched frame); the run did not retain generated
code, so `extension_generations` has raw harness outcomes only and could not be
audited. The
`fixed_version_generations` config holds the three exact-version frontier-model
runs on the 365 prompts plus 150 main-benchmark anchors. Candidate rows that did
not survive the audit are not released.

### What is excluded

- **24-bin "equal-support" tail set.** An earlier high-complexity tail set was
  mined on the cyclomatic complexity of *generated* solutions with the
  reference-test gate disabled. Because selection used the post-generation metric
  this study criticizes, the set is contaminated and none of its prompts,
  generations, or scores are included. (Main-benchmark prompts that were also
  reused in that set appear only as main-benchmark prompts.)
- **AuroraGPT-IT-v4.** A locally served run that covered only the earlier frame
  was excluded from the panel post hoc and is not released.
- **Raw API responses, request and batch identifiers, deployment names,
  endpoints, and credentials.**

## Fields

<!-- FIELDS:BEGIN -->
#### `prompts` (5,000 rows)

The 5,000-prompt Python benchmark: prompt text, unit tests, source metadata, preliminary single-rater sampling score, four-judge ensemble index, task type, keyword features, and prompt-level mean pass rates over the 21-model panel.

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | OpenCodeInstruct `id` of the source record; primary key. |
| `source_dataset` | string | Source dataset (nvidia/OpenCodeInstruct). |
| `language` | string | Programming language of prompt, tests, and execution (always python). |
| `prompt_text` | string | Task instruction given to every evaluated model (OpenCodeInstruct `input`). |
| `unit_tests` | list<string> | Assertion snippets from OpenCodeInstruct `unit_tests`; each is executed separately by the harness. |
| `n_unit_tests` | int | Number of assertion snippets. |
| `construction_frame` | string | earlier_retained (2,246 prompts kept from the earlier prefix-scan draw) or later_candidate (2,754 later candidates, including deliberate high reference-CC supplementation). |
| `source_reference_avg_test_score` | float | NVIDIA-supplied `average_test_score` of the OpenCodeInstruct reference solution on these tests (copied from the source record, not re-executed). |
| `reference_cc` | int | Lizard cyclomatic complexity of the OpenCodeInstruct reference solution. Used upstream to shape the candidate pool and for alignment checks; not the analysis index. |
| `reference_cc_band` | string | Reference-CC band used when drawing later candidates (null for earlier_retained). |
| `prelim_branching` | int | Preliminary single-rater (o4-mini) score used only for stratified sampling for the branching dimension (0-4 scale). |
| `prelim_iteration` | int | Preliminary single-rater (o4-mini) score used only for stratified sampling for the iteration dimension (0-4 scale). |
| `prelim_state` | int | Preliminary single-rater (o4-mini) score used only for stratified sampling for the state dimension (0-4 scale). |
| `prelim_data_structures` | int | Preliminary single-rater (o4-mini) score used only for stratified sampling for the data structures dimension (0-4 scale). |
| `prelim_edge_cases` | int | Preliminary single-rater (o4-mini) score used only for stratified sampling for the edge cases dimension (0-4 scale). |
| `prelim_composition` | int | Preliminary single-rater (o4-mini) score used only for stratified sampling for the composition dimension (0-4 scale). |
| `prelim_composite` | int | Preliminary o4-mini composite (0-24); sampling only. |
| `prelim_band` | string | Preliminary sampling band: 0-3, 4-6, 7-9, 10-12, 13-15, or 16-24 (834/834/833/833/833/833 prompts). |
| `ens_branching` | float | Four-judge ensemble mean for the branching dimension (0-4 scale). |
| `ens_iteration` | float | Four-judge ensemble mean for the iteration dimension (0-4 scale). |
| `ens_state` | float | Four-judge ensemble mean for the state dimension (0-4 scale). |
| `ens_data_structures` | float | Four-judge ensemble mean for the data structures dimension (0-4 scale). |
| `ens_edge_cases` | float | Four-judge ensemble mean for the edge cases dimension (0-4 scale). |
| `ens_composition` | float | Four-judge ensemble mean for the composition dimension (0-4 scale). |
| `ens_composite` | float | Analysis index C_i: sum of the six ensemble dimension means (0-24). |
| `ens_composite_sd` | float | Sample standard deviation (ddof=1) of the per-judge composites; a judge-disagreement measure. |
| `ens_n_judges` | int | Number of judges with a valid score (4 for 4,998 prompts; 3 and 2 for one prompt each). |
| `display_bin` | int | Descriptive display bin floor(ens_composite + 0.5): bin b holds [b-0.5, b+0.5). Breakpoint estimation uses the unbinned index. |
| `task_type` | string | Primary task category from the nine-category taxonomy (o4-mini labeler). |
| `task_type_confidence` | float | Labeler self-reported confidence for task_type. |
| `names_external_library` | bool | Labeler flag: prompt names an external library or framework. |
| `kw_inst_tokens` | int | Keyword/lexical prompt feature `inst_tokens` (pre-generation lexical baseline). |
| `kw_inst_if_count` | int | Keyword/lexical prompt feature `inst_if_count` (pre-generation lexical baseline). |
| `kw_inst_conditional_count` | int | Keyword/lexical prompt feature `inst_conditional_count` (pre-generation lexical baseline). |
| `kw_inst_loop_count` | int | Keyword/lexical prompt feature `inst_loop_count` (pre-generation lexical baseline). |
| `kw_inst_collection_count` | int | Keyword/lexical prompt feature `inst_collection_count` (pre-generation lexical baseline). |
| `kw_inst_class_count` | int | Keyword/lexical prompt feature `inst_class_count` (pre-generation lexical baseline). |
| `kw_inst_func_count` | int | Keyword/lexical prompt feature `inst_func_count` (pre-generation lexical baseline). |
| `kw_inst_logic_count` | int | Keyword/lexical prompt feature `inst_logic_count` (pre-generation lexical baseline). |
| `kw_inst_total_structural` | int | Keyword/lexical prompt feature `inst_total_structural` (pre-generation lexical baseline). |
| `kw_inst_avg_word_len` | float | Keyword/lexical prompt feature `inst_avg_word_len` (pre-generation lexical baseline). |
| `n_models` | int | Number of evaluated-panel generations for this prompt (21). |
| `mean_pass_rate` | float | Mean of generations.pass_rate (the audited primary outcome) over the 21 models: the prompt-level outcome of the camera-ready paper. |
| `mean_pass_rate_reviewed` | float | Mean of generations.pass_rate_reviewed over the 21 models (the reviewed version's prompt-level outcome). |
| `mean_harness_pass_rate` | float | Mean of generations.harness_pass_rate over the 21 models (pure test-execution fraction). |
| `in_human_calibration` | bool | Prompt was graded in the human calibration study. |
| `in_passk_subset` | bool | Prompt is in the 359-prompt repeated-sampling subset. |
| `in_paraphrase_subset` | bool | Prompt is in the 150-prompt paraphrase check. |
| `in_cross_language_subset` | bool | Prompt is in the 117-prompt Java/C++ re-expression check. |
| `is_fixed_version_anchor` | bool | Prompt is one of the 150 midrange anchors in the fixed-version three-model check. |

#### `judge_scores` (19,997 rows)

Per-judge rubric scores for the 5,000 main prompts (19,997 rows; four out-of-panel judges).

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Prompt identifier (OpenCodeInstruct `id`). |
| `judge` | string | Rubric judge: o4-mini, gpt-5.5, llama-4-maverick, or command-a (all outside the evaluated panel). |
| `branching` | int | Judge score for the branching dimension (0-4 scale). |
| `iteration` | int | Judge score for the iteration dimension (0-4 scale). |
| `state` | int | Judge score for the state dimension (0-4 scale). |
| `data_structures` | int | Judge score for the data structures dimension (0-4 scale). |
| `edge_cases` | int | Judge score for the edge cases dimension (0-4 scale). |
| `composition` | int | Judge score for the composition dimension (0-4 scale). |
| `composite` | int | Sum of the six dimension scores (0-24). |
| `rubric_sha256` | string | SHA-256 of the exact rubric system prompt (docs/rubric_prompt.txt). |
| `scored_at` | string | UTC timestamp of the scoring call (ISO 8601). |

#### `generations` (105,000 rows)

One generated solution per (model, prompt) for the 21-model panel on the 5,000 main prompts (105,000 rows): cleaned code, per-test outcomes, three outcome definitions (the audited primary outcome, the reviewed version's outcome, and the raw harness fraction), the auditor's verdict, Lizard output CC, and recovered generation settings. Raw API responses are not included.

| Field | Type | Description |
|---|---|---|
| `model_key` | string | Experiment identifier of the evaluated model (joins to the models config). |
| `model_display_name` | string | Display name used in the paper. |
| `prompt_id` | string | Prompt identifier (joins to prompts). |
| `construction_frame` | string | Construction frame of the prompt (copied from prompts). |
| `code` | string | Cleaned generated Python code exactly as executed and measured (extracted from the model response). Empty string when no code could be extracted; a small number of rows retain markdown fences. |
| `pass_rate` | float | Primary outcome of the camera-ready paper: 1.0 if the independent auditor (MiMo-V2.6-Pro) judged the code correct, 0.0 if incorrect or if there is no code (empty-code rule); otherwise (uncertain, unparseable, or a response cut off by the token limit or a content filter) harness_pass_rate. |
| `pass_rate_source` | string | 'independent_audit' when pass_rate comes from a correct/incorrect verdict or the empty-code rule, else 'harness_fallback'. |
| `independent_audit_verdict` | string | Counted verdict of the independent auditor: correct, incorrect, uncertain, or null (unparseable or incomplete response). |
| `independent_audit_handling` | string | auditor_verdict, auditor_uncertain, unparseable, incomplete_response (cut off by the token limit or a content filter), or empty_code_rule (no code; marked incorrect without an auditor request). |
| `harness_pass_rate` | float | Fraction of unit-test assertions that passed when the code was executed (from test_status). |
| `pass_rate_reviewed` | float | Outcome of the reviewed (submitted) version: harness_pass_rate, except that for earlier_retained prompts an o4-mini audit verdict of correct or incorrect set it to 1.0 or 0.0. |
| `reviewed_audit_verdict` | string | The reviewed version's o4-mini harness-audit verdict (earlier_retained rows only): correct, incorrect, uncertain, or null. |
| `model_returned_no_response` | bool | The model API returned no response at all (time-out, cancelled operation, or empty response). These 157 rows (156 Gemini 3.1 Pro Preview) count as failures in the primary analysis; the paper also reports results treating them as missing. |
| `n_tests` | int | Number of executed assertions. |
| `test_status` | list<string> | Per-assertion outcome ('pass'/'fail'), in unit_tests order. |
| `output_cc_lizard` | int | Generated-output cyclomatic complexity: Lizard CC of `code`, summed over reported functions. Null when not computable (1,052 rows). Never the prompt index or reference CC. |
| `generated_at` | string | UTC timestamp recorded when the generation was stored (ISO 8601). |
| `gen_settings_source` | string | 'batch_request_record' when settings were recovered from a retained batch request for this prompt and model, else 'not_recorded_per_row' (see the models config for documented defaults). |
| `gen_api_model` | string | Model string sent in the retained batch request (null if not recorded). |
| `gen_temperature` | float | Temperature sent in the retained batch request; null if not recorded or not sent (provider default). |
| `gen_temperature_sent` | bool | Whether a temperature parameter was present in the retained batch request (null if not recorded). |
| `gen_max_output_tokens` | int | Output-token cap sent in the retained batch request (null if not recorded). |
| `gen_system_prompt_variant` | string | Generation system prompt variant (A_pipeline, B_batch_scripts, see docs/generation_system_prompts.json); null if not recorded. |

#### `extension_prompts` (365 rows)

The 365-prompt audit-clean high-complexity extension: selected on prompt-side scores only, contract-audited, and reference-verified (every reference solution passes every test). Disjoint from the main benchmark and from the excluded 24-bin set.

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | OpenCodeInstruct `id`; primary key. |
| `source_dataset` | string | Source dataset (nvidia/OpenCodeInstruct). |
| `language` | string | Always python. |
| `prompt_text` | string | Task instruction. |
| `unit_tests` | list<string> | Assertion snippets. |
| `n_unit_tests` | int | Number of assertion snippets. |
| `construction_frame` | string | Always tail_extension. |
| `source_reference_avg_test_score` | float | NVIDIA-supplied reference `average_test_score` (all 1.0). |
| `reference_local_pass_rate` | float | Pass rate of the reference solution when re-executed by the benchmark harness (all 1.0). |
| `reference_cc` | int | Lizard CC of the OpenCodeInstruct reference solution. |
| `prelim_branching` | int | Preliminary o4-mini candidate score (used only to rank candidates for ensemble scoring) for the branching dimension (0-4 scale). |
| `prelim_iteration` | int | Preliminary o4-mini candidate score (used only to rank candidates for ensemble scoring) for the iteration dimension (0-4 scale). |
| `prelim_state` | int | Preliminary o4-mini candidate score (used only to rank candidates for ensemble scoring) for the state dimension (0-4 scale). |
| `prelim_data_structures` | int | Preliminary o4-mini candidate score (used only to rank candidates for ensemble scoring) for the data structures dimension (0-4 scale). |
| `prelim_edge_cases` | int | Preliminary o4-mini candidate score (used only to rank candidates for ensemble scoring) for the edge cases dimension (0-4 scale). |
| `prelim_composition` | int | Preliminary o4-mini candidate score (used only to rank candidates for ensemble scoring) for the composition dimension (0-4 scale). |
| `prelim_composite` | int | Preliminary o4-mini composite (0-24). |
| `ens_branching` | float | Four-judge ensemble mean for the branching dimension (0-4 scale). |
| `ens_iteration` | float | Four-judge ensemble mean for the iteration dimension (0-4 scale). |
| `ens_state` | float | Four-judge ensemble mean for the state dimension (0-4 scale). |
| `ens_data_structures` | float | Four-judge ensemble mean for the data structures dimension (0-4 scale). |
| `ens_edge_cases` | float | Four-judge ensemble mean for the edge cases dimension (0-4 scale). |
| `ens_composition` | float | Four-judge ensemble mean for the composition dimension (0-4 scale). |
| `ens_composite` | float | Four-judge ensemble composite (selection required >= 15). |
| `ens_composite_sd` | float | Sample standard deviation (ddof=1) of the per-judge composites, recomputed from extension_judge_scores (the source file stores the population SD). |
| `ens_n_judges` | int | Number of judges (always 4). |
| `display_bin` | int | floor(ens_composite + 0.5); 218/133/11/3 prompts at bins 15/16/17/18. |

#### `extension_judge_scores` (1,460 rows)

Per-judge rubric scores for the 365 extension prompts (1,460 rows).

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Prompt identifier (OpenCodeInstruct `id`). |
| `judge` | string | Rubric judge: o4-mini, gpt-5.5, llama-4-maverick, or command-a (all outside the evaluated panel). |
| `branching` | int | Judge score for the branching dimension (0-4 scale). |
| `iteration` | int | Judge score for the iteration dimension (0-4 scale). |
| `state` | int | Judge score for the state dimension (0-4 scale). |
| `data_structures` | int | Judge score for the data structures dimension (0-4 scale). |
| `edge_cases` | int | Judge score for the edge cases dimension (0-4 scale). |
| `composition` | int | Judge score for the composition dimension (0-4 scale). |
| `composite` | int | Sum of the six dimension scores (0-24). |
| `rubric_sha256` | string | SHA-256 of the exact rubric system prompt (docs/rubric_prompt.txt). |
| `scored_at` | string | UTC timestamp of the scoring call (ISO 8601). |

#### `extension_generations` (2,190 rows)

Test outcomes for the 365 extension prompts across the six extension-run models (2,190 rows). Generated code was not retained by the extension run, so no code or output CC is available.

| Field | Type | Description |
|---|---|---|
| `model_key` | string | Experiment identifier (joins to models). |
| `model_display_name` | string | Display name. |
| `prompt_id` | string | Extension prompt identifier. |
| `in_matched_five` | bool | Model is one of the five models present in both the main panel and the extension run (the paper's matched frame). |
| `pass_rate` | float | Fraction of unit-test assertions passed by the single generated solution. |
| `has_code` | bool | Whether code could be extracted from the response. |
| `gen_temperature` | float | Temperature (0.0). |
| `gen_max_output_tokens` | int | Output-token cap (4096). |

#### `fixed_version_generations` (1,545 rows)

Fixed-version three-model check (Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro Preview): 150 main-benchmark midrange anchors plus the 365 extension prompts per model (1,545 rows). No code retained.

| Field | Type | Description |
|---|---|---|
| `model_key` | string | Experiment identifier (joins to models). |
| `model_display_name` | string | Display name. |
| `prompt_id` | string | Prompt identifier (anchor ids join to prompts; tail ids join to extension_prompts). |
| `group` | string | anchor (main-benchmark midrange prompt) or tail (extension prompt). |
| `ens_composite` | float | Ensemble composite of the prompt as recorded by the run. |
| `pass_rate` | float | Fraction of unit-test assertions passed. |
| `has_code` | bool | Whether code could be extracted from the response. |

#### `passk_generations` (7,180 rows)

Repeated-sampling subset: 359 main prompts x 4 models x 5 draws at temperature 0.8 (7,180 rows), with code and pass rate. These draws are separate from the main single-generation panel.

| Field | Type | Description |
|---|---|---|
| `model_key` | string | Experiment identifier (joins to models). |
| `model_display_name` | string | Display name. |
| `prompt_id` | string | Prompt identifier (joins to prompts). |
| `draw` | int | Draw index 1-5. |
| `code` | string | Cleaned generated code. |
| `pass_rate` | float | Fraction of unit-test assertions passed. |
| `gen_temperature` | float | Temperature (0.8). |
| `gen_max_output_tokens` | int | Output-token cap (4096). |

#### `human_calibration` (250 rows)

Blinded human rubric grades: 200 prompts by grader_1 (first author) and a 50-prompt overlap by grader_2 (second research-team member); LLM scores hidden during grading (250 rows).

| Field | Type | Description |
|---|---|---|
| `grader` | string | grader_1 (first author, 200 prompts) or grader_2 (second research-team member, 50 prompts). |
| `prompt_id` | string | Prompt identifier (joins to prompts). |
| `worksheet_row` | int | Position in the grader's randomized worksheet (grader_2 received the first 50 rows of grader_1's order). |
| `branching` | int | Human grade for the branching dimension (0-4 scale). |
| `iteration` | int | Human grade for the iteration dimension (0-4 scale). |
| `state` | int | Human grade for the state dimension (0-4 scale). |
| `data_structures` | int | Human grade for the data structures dimension (0-4 scale). |
| `edge_cases` | int | Human grade for the edge cases dimension (0-4 scale). |
| `composition` | int | Human grade for the composition dimension (0-4 scale). |
| `composite` | int | Sum of the six human grades (0-24). |
| `in_two_grader_overlap` | bool | Prompt was graded by both graders. |
| `sampling_band` | string | Ensemble-composite band used to stratify the 500-prompt calibration pool. |
| `selection_stratum` | string | high_disagreement (top judge-disagreement prompts within band) or random, reproduced from the seeded selector; null if not reproducible. |
| `calibration_priority` | float | Judge disagreement at selection (ensemble composite SD). |

#### `task_type_labels` (6,500 rows)

Nine-category task-type labels: o4-mini labels all 5,000 prompts; gpt-5.5, llama-4-maverick, and command-a label a shared 500-prompt subset (6,500 rows).

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Prompt identifier (joins to prompts). |
| `labeler` | string | Labeling model. |
| `is_primary_labeler` | bool | True for the o4-mini labels used as task-type fixed effects. |
| `primary_type` | string | Task category. |
| `names_external_library` | bool | Prompt names an external library/framework. |
| `confidence` | float | Labeler self-reported confidence. |
| `taxonomy_sha256` | string | Hash of the fixed taxonomy prompt. |

#### `paraphrase_prompts` (150 rows)

150 plain-language rewrites (DeepSeek-V3.2, instructed to preserve inputs, outputs, behavior, constraints, and edge cases). Rewrites were not execution-verified.

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Original prompt identifier (joins to prompts). |
| `sampling_bin` | int | Rounded original composite bin used for sampling (75% from bins 10-18). |
| `original_composite` | float | Original ensemble composite as recorded at sampling time. |
| `original_prompt_text` | string | Original prompt text. |
| `paraphrased_prompt_text` | string | Rewritten prompt text that the judges rescored. |

#### `paraphrase_judge_scores` (600 rows)

Per-judge rubric scores of the 150 paraphrased prompts (600 rows).

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Prompt identifier (OpenCodeInstruct `id`). |
| `judge` | string | Rubric judge: o4-mini, gpt-5.5, llama-4-maverick, or command-a (all outside the evaluated panel). |
| `branching` | int | Judge score for the branching dimension (0-4 scale). |
| `iteration` | int | Judge score for the iteration dimension (0-4 scale). |
| `state` | int | Judge score for the state dimension (0-4 scale). |
| `data_structures` | int | Judge score for the data structures dimension (0-4 scale). |
| `edge_cases` | int | Judge score for the edge cases dimension (0-4 scale). |
| `composition` | int | Judge score for the composition dimension (0-4 scale). |
| `composite` | int | Sum of the six dimension scores (0-24). |
| `rubric_sha256` | string | SHA-256 of the exact rubric system prompt (docs/rubric_prompt.txt). |
| `scored_at` | string | UTC timestamp of the scoring call (ISO 8601). |

#### `cross_language_prompts` (234 rows)

117 prompts re-expressed in Java and in C++ by DeepSeek-V3.2 for rescoring only (234 rows); no non-Python generation or execution.

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Original Python prompt identifier (joins to prompts). |
| `target_language` | string | java or cpp. |
| `original_composite` | float | Original Python ensemble composite as recorded. |
| `ported_prompt_text` | string | Re-expressed prompt text. |

#### `cross_language_judge_scores` (936 rows)

Per-judge rubric scores of the Java and C++ re-expressions (936 rows).

| Field | Type | Description |
|---|---|---|
| `target_language` | string | java or cpp. |
| `prompt_id` | string | Prompt identifier (OpenCodeInstruct `id`). |
| `judge` | string | Rubric judge: o4-mini, gpt-5.5, llama-4-maverick, or command-a (all outside the evaluated panel). |
| `branching` | int | Judge score for the branching dimension (0-4 scale). |
| `iteration` | int | Judge score for the iteration dimension (0-4 scale). |
| `state` | int | Judge score for the state dimension (0-4 scale). |
| `data_structures` | int | Judge score for the data structures dimension (0-4 scale). |
| `edge_cases` | int | Judge score for the edge cases dimension (0-4 scale). |
| `composition` | int | Judge score for the composition dimension (0-4 scale). |
| `composite` | int | Sum of the six dimension scores (0-24). |
| `rubric_sha256` | string | SHA-256 of the exact rubric system prompt (docs/rubric_prompt.txt). |
| `scored_at` | string | UTC timestamp of the scoring call (ISO 8601). |

#### `audit_known_answer_cases` (900 rows)

Known-answer cases used to select (set A) and confirm (set B) the outcome auditor. For each of 150 benchmark prompts per set whose reference solution passes every unit test: the normalized reference (clean, correct), a copy with the test-called names renamed so the harness fails it (cosmetic, correct), and the most subtle single-AST mutation the tests detect (bug, incorrect), all executed in the benchmark harness (900 rows). Labels come from the harness and construction, not human review; some clean references violate task instructions the tests do not check.

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Case identifier: <prompt_id>:<variant>. |
| `known_answer_set` | string | A_selection (auditor selection, seed 20260928) or B_confirmation (fresh prompts, seed 20260930). |
| `prompt_id` | string | Benchmark prompt identifier (joins to prompts). |
| `category` | string | clean, cosmetic, or bug. |
| `ground_truth` | string | correct (clean, cosmetic) or incorrect (bug). |
| `mutation` | string | For bug cases, the AST mutation kind and site (JSON); null otherwise. |
| `code` | string | The code shown to the auditors. |
| `harness_pass_rate` | float | Fraction of unit tests the case passes in the benchmark harness. |

#### `audit_known_answer_responses` (3,150 rows)

Every auditor response on the known-answer sets: five candidate auditors and the reviewed version's o4-mini audit (diagnostic) on set A, and the adopted auditor on set B (3,150 rows).

| Field | Type | Description |
|---|---|---|
| `known_answer_set` | string | A_selection or B_confirmation. |
| `auditor` | string | Auditor model key (joins to models). |
| `case_id` | string | Known-answer case (joins to audit_known_answer_cases). |
| `status` | string | ok, parse_error, or api_error as recorded by the audit client. |
| `verdict` | string | Verdict parsed from the response (correct, incorrect, uncertain, or null). |
| `counted_verdict` | string | Verdict after the completeness rule (null if the response was cut off by the token limit or a content filter); used for every reported rate. |
| `finish_reason` | string | Finish reason reported by the endpoint. |
| `parse_status` | string | json, regex, parse_error, or rule. |
| `reason` | string | The auditor's stated reason. |
| `response_tail` | string | Last 600 characters of the response. |
| `completion_tokens` | int | Completion tokens reported by the endpoint (may exclude reasoning tokens on some routes). |

#### `audit_production_verdicts` (115,500 rows)

Outcome-audit verdicts for the main benchmark: the adopted auditor on all 105,000 generations and two second auditors on the seeded 5% agreement sample (5,250 generations each), 115,500 rows. The last final record per generation is kept, as in the analysis.

| Field | Type | Description |
|---|---|---|
| `auditor` | string | Auditor model key (joins to models). |
| `model_key` | string | Evaluated model (joins to models). |
| `prompt_id` | string | Prompt identifier (joins to prompts). |
| `in_agreement_sample` | bool | Generation is in the seeded 5% agreement sample. |
| `status` | string | ok, parse_error, or api_error as recorded by the audit client. |
| `verdict` | string | Verdict parsed from the response. |
| `counted_verdict` | string | Verdict after the completeness rule; for the adopted auditor this is generations.independent_audit_verdict. |
| `finish_reason` | string | Finish reason reported by the endpoint. |
| `parse_status` | string | json, regex, parse_error, or rule (empty code: incorrect without a request). |
| `reason` | string | The auditor's stated reason. |
| `response_tail` | string | Last 600 characters of the response. |

#### `models` (35 rows)

Metadata for every model whose outputs or labels appear in the release: developer, role(s), access route, served model string, and documented generation settings.

| Field | Type | Description |
|---|---|---|
| `model_key` | string | Experiment identifier (judge:/aux: prefixes for non-evaluated roles). |
| `display_name` | string | Display name. |
| `developer` | string | Model developer. |
| `roles` | list<string> | Role(s) in the release. |
| `access_routes` | string | How outputs were obtained (no account or deployment identifiers). |
| `served_model` | string | Model string recorded in the run configuration. |
| `weights` | string | open or closed weights (descriptive). |
| `documented_settings` | string | Documented generation settings for the role(s). |
| `notes` | string | Caveats (quantization, naming discrepancies). |

<!-- FIELDS:END -->

## Intended uses

- Reproducing and auditing the paper's analyses: pooled and model-specific
  breakpoint estimates, task-type and construction-frame sensitivity, the
  failure-dependent output-complexity diagnostic, and the rater, paraphrase,
  cross-language, repeated-sampling, and extension checks.
- Evaluating alternative prompt-side complexity measures against the four-judge
  index and the human grades.
- Studying LLM-as-judge reliability when judges score inputs rather than outputs.
- Studying how failed or unscorable generations bias output-side metrics, and
  auditing test harnesses and benchmark construction.

## Out-of-scope uses

- Treating any breakpoint as a universal, causal, or deployment-level failure
  threshold for code generators.
- Ranking models outside this exact protocol (prompt frame, settings, routes,
  tests), or reading results as current for models whose hosted versions change.
- Treating the six rubric dimensions as valid instruments: the overidentification
  restrictions reject.
- Training or fine-tuning models on these prompts, tests, or generations, which
  would contaminate future evaluation.
- Running the generated code outside a sandbox.

## Limitations and biases

- **Benchmark-conditional breakpoints.** Every estimate is conditional on the
  constructed frame, the scoring index, generation settings, and the unit-test
  protocol. The prompt distribution is neither the natural OpenCodeInstruct
  distribution nor balanced on the final index.
- **Construction-frame composition.** Later candidates make up 41.0% of prompts
  at or below composite 14.0 (the primary-outcome breakpoint) but 95.4% above it,
  so the pooled rebound is largely a composition effect. The frame also marks
  the generation run and test quality (below). Control for `construction_frame`
  in any pooled analysis.
- **Python only.** Java and C++ data are rescoring of re-expressed prompts; no
  non-Python code was generated or executed.
- **LLM-judge index with moderate human agreement.** On a disagreement-enriched
  200-prompt sample, first-grader agreement with the ensemble is Pearson r =
  0.408 and ICC(2,1) = 0.395 (mean offset -1.18 points); the two human graders
  correlate 0.561 on their 50-prompt overlap. Common-mode judge bias cannot be
  averaged away.
- **Unit-test quality.** Tests are inherited from OpenCodeInstruct. For 1,259 of
  the 2,246 `earlier_retained` prompts the source-supplied reference
  `average_test_score` is below 1.0 (491 at 0.0); every `later_candidate` and
  extension prompt has a fully passing reference. See
  `source_reference_avg_test_score`. The earlier frame was not passed through
  the contract and test-quality filter applied to later candidates.
- **Heterogeneous generation settings.** Temperature (0.0, 0.2, or provider
  default), output-token caps, system prompts, and access routes differ across
  models and frames. Per-row settings are released where a batch request record
  was retained (24,675 rows); otherwise see the `models` config. Several
  open-weight models were served as 4-bit or 8-bit GGUF quantizations, and the
  model whose experiment key is `mistral-small-2412` was served as
  Devstral-Small-2505 (Q4_K_M), the name used in the paper. Each model's two
  frames were generated in separate runs weeks apart, and the extension and
  pass@k runs used a system prompt (variant C) that names the functions to
  define, which the main runs did not.
- **Missing values.** `output_cc_lizard` is null for 1,052 generations; 208
  generations have no extractable code (scored 0.0; 157 of them because the
  model API returned no response); 129 `code` values retain
  markdown fences. Two prompts have fewer than four judge scores.
- **Sparse tails.** Main-benchmark display bins 0 and 17-19 hold 3, 45, 6, and 5
  prompts; the extension adds only 14 prompts above bin 16.
- **Single draw.** Apart from the pass@k subset, each model contributes one
  generation per prompt.
- **Post hoc robustness data.** Task types, construction-frame analysis, the
  extension, human calibration, paraphrase, language, and repeated-sampling data
  were added after the initial analysis and were not preregistered.

## Personal and sensitive information

No sensitive attributes are recorded about any person. Human grades are released
under pseudonymous IDs (`grader_1` = first author, `grader_2` = second
research-team member) without names or demographics. Prompts and code may contain
fictitious names, placeholder emails, and placeholder credentials from the
source dataset or model outputs; an automated scan of every released text field
found no real keys, tokens, account or deployment identifiers, or local paths.

## Safety

`code`, `passk_generations.code`, and any code reconstructed from these tasks
are unreviewed model outputs. Some generated programs open files, sockets, or
GUI windows. Execute them only in an isolated sandbox without network access.

## Licensing

<!-- TODO(author): choose the release license after reviewing release/LICENSE_NOTES.md; cc-by-4.0 in the header is a placeholder. -->
Prompts, unit tests, and reference-solution metadata derive from OpenCodeInstruct
(CC BY 4.0; attribution: NVIDIA). Generated code, judge scores, labels, and
rewrites are outputs of third-party models whose providers' terms may govern
redistribution; see the `models` config for the full list of models and routes.

## Citation

```bibtex
@inproceedings{complexitykink2026,
  title     = {The Complexity Kink: LLM Rubric Instruments for Causal Inference on Code Generation Reliability},
  author    = {Hernandez, Michael and Zhao, Tian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

Please also cite OpenCodeInstruct: W. U. Ahmad et al., "OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs," arXiv:2504.04030, 2025.

## Reproducing reported numbers

These values were recomputed from the released files by
`scripts/build_public_release.py` in the paper's code repository.

| Quantity | Manuscript | This release |
|---|---|---|
| Main prompts / generations / judge rows | 5,000 / 105,000 / 19,997 | 5,000 / 105,000 / 19,997 |
| Four-judge composite ICC(2,1), complete cases | 0.872 (n = 4,998) | 0.8719 (n = 4,998) |
| Generations with Lizard output CC | 103,948 | 103,948 |
| Prompts at or below / above 14.0 and mean pass (primary outcome) | 3,703 at 0.834 / 1,297 at 0.901 | 3,703 at 0.834 / 1,297 at 0.901 |
| Prompts at or below / above 13.75 and mean pass (reviewed outcome) | 3,617 at 0.799 / 1,383 at 0.876 | 3,617 at 0.799 / 1,383 at 0.876 |
| Failed complete cases in the reverse-threshold cell (primary outcome) | 3,561 of 14,247 (25.0%) | 3,561 of 14,247 (25.0%) |
| Human grader 1 vs ensemble (n = 200): Pearson / ICC(2,1) | 0.408 / 0.395 | 0.408 / 0.395 |
| Extension bins 15/16/17/18 | 218/133/11/3 | 218/133/11/3 |
| Matched-five raw-harness pass, bin 15 and 16 (extension vs original) | 0.880 vs 0.890; 0.799 vs 0.795 | 0.880 vs 0.890; 0.799 vs 0.795 |
