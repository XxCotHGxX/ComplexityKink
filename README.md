# The Complexity Kink

**Identifying a Hard Reliability Threshold in LLM Code Generation via Instrumental Variables**

Michael Hernandez, University of Wisconsin-Milwaukee

This repository contains the full analysis pipeline, data provenance scripts, and paper source for the Complexity Kink program: a study of how the reliability of frontier code-generating language models breaks down past a specific level of structural task complexity, and how that breakdown is hidden by the standard practice of measuring complexity from model output.

---

## What this repository implements

A two-stage least squares (2SLS) analysis on 21 frontier 2025 and 2026 language models, evaluated against 5,000 stratified Python coding tasks from NVIDIA's OpenCodeInstruct, using a six-dimension rubric scored by an out-of-panel LLM as the instrument for true task complexity.

The headline finding is a structural break (the "Complexity Kink") at rubric composite 8 on a 0 to 24 scale, beyond which mean pass rate drops from 48.9% to 37.0% across the 21-model panel. The break replicates at the 5% level in 19 of the 21 individual models, with per-model thresholds in the range 4 to 10.

## Background and the three-stage progression

The methodology developed in three stages, each correcting a weakness of the prior. The repository in its current state implements **Stage C**. Earlier stages remain available in the source tree and in the git history as `stage-b-final`.

**Stage A (baseline, not the contribution).** Compute cyclomatic complexity from generated code with a static analysis tool. This is the standard approach in code-generation benchmarks. It is biased by construction: when a model fails on a complex task, the broken or stub output has near-trivial cyclomatic complexity, regardless of the task's actual difficulty. Failed hard tasks are recorded as easy tasks, the complexity-performance relationship is flattened, and any structural break is hidden.

**Stage B (preprint).** Two-stage least squares with eight keyword-derived instruments fit through a Random Forest. The instrument vector counts branching keywords, loop keywords, class keywords, function keywords, and structural keywords in the prompt text, plus token count and average word length. Stage B identified the kink at predicted cyclomatic complexity 6.5 with a Hausman chi-squared of 108.5 and a first-stage F of 1,765. The Stage B preprint and its analysis code are preserved in `paper/stage_b_preprint/` and `src/run_stage2_iv.py`.

**Stage C (this repository).** Replace the keyword-based instrument with a rigid six-dimension rubric scored by a frontier reasoning model (o4-mini) that is excluded from the evaluated panel. Each prompt is scored 0 to 4 on six structural dimensions of the correct solution: branching, iteration, state management, data structures, edge cases, and algorithmic composition. The rubric design eliminates four weaknesses of the Stage B instrument:

1. *Survivorship bias:* the Stage B Random Forest was trained only on rows where the model passed (the only rows where observed cyclomatic complexity equals true target complexity). Stage C scores prompts directly and never trains on output.
2. *The generated regressor problem:* Stage B's predicted complexity was an estimate, requiring re-fitting of Stage 1 in every bootstrap iteration. Stage C's rubric scores are deterministic constants.
3. *Low-end discrimination:* Stage B's keyword features could not statistically separate cyclomatic complexity 1 from 2. The rubric separates them at p less than 1e-6.
4. *Non-monotonic instruments:* two of the eight Stage B instruments (token count, average word length) violate the IV monotonicity condition. All six Stage C dimensions are monotone in true complexity by construction.

## Headline results

| Quantity | Stage B (preprint) | Stage C (this repo) |
| :-- | --: | --: |
| Instrument | 8 keyword features via Random Forest | 6-dimension LLM rubric |
| First-stage F | 1,765 | 3,002 |
| Partial R-squared | 0.531 | 0.462 |
| Hausman chi-squared | 108.5 | 20.1 |
| Kink threshold | 6.5 (cyclomatic units) | 8.0 (rubric composite) |
| Below-kink pass rate | 40.4% | 48.9% |
| Above-kink pass rate | 11.8% | 37.0% |
| Bootstrap CI on threshold | not reported | [4.0, 11.0] |
| Sargan / Hansen J | not interpretable (non-monotone) | rejects (J = 154.3) |
| Placebo p-value | < 0.001 | < 0.002 |
| Sample | 35,499 generations, 5 languages, 1 model | 105,000 generations, Python, 21 models |

The smaller Hausman chi-squared in Stage C is expected. By scoring prompts rather than outputs, the rubric removes a large share of the endogeneity before 2SLS runs, so the gap between OLS and IV is smaller. The Stage C paper discusses this explicitly.

The Sargan / Hansen J test rejects under Stage C. We treat this honestly in the paper, not as positive validation. Two interpretations are consistent with the rejection. First, single-scorer idiosyncrasy in o4-mini may add a small direct channel from one or more rubric dimensions to pass rate. Second, the J test loses calibration in large samples; with N = 5,000 even a small residual correlation between an instrument and the structural error drives J into the rejection region. Stage D, scoped in the paper, addresses this directly with a four-scorer ensemble and a measured errors-in-variables correction.

## Repository structure

```
.
LICENSE
README.md                            (this file)
requirements.txt                     pinned dependencies
paper/
  complexity_kink_stage_c.tex        Stage C paper draft (current)
  stage_b_preprint/
    complexity_kink_2026.tex         Stage B paper LaTeX source
    Complexity_Kink_stage_b.pdf      Stage B preprint as published
src/
  analyze_kink.py                    Stage C primary analysis (per-model + combined 2SLS, Hansen, placebo)
  extract_paper_numbers.py           Reproduces the exact numbers cited in the paper
  config.py                          Central configuration: paths, hyperparameters, design choices
  data_loader.py                     Canonical loader: enforces no silent NaN-to-1 fallback on cyclomatic complexity
  feature_extractor_iv.py            Stage B keyword feature extraction (preserved)
  generate_viz.py                    Static publication figures
  generate_viz_plotly.py             Interactive HTML figures
  run_stage2_iv.py                   Stage B 2SLS implementation (preserved; provides shared threshold-grid helper)
  train_stage1_iv.py                 Stage B Random Forest training (preserved)
  verify_pipeline.py                 End-to-end smoke test on a 2,000-sample subset
  parsers/
    py_parser.py                     Tree-sitter AST parser used in feature extraction
  data_provenance/
    README.md                        Full pipeline narrative
    00_extract_from_source.py        Pull from OpenCodeInstruct on Hugging Face
    01_select_prompts.py             Stratified sampling across complexity bins
    02_generate_solutions.py         Multi-backend LLM generation
    03_execute_and_score.py          Sandboxed unit-test execution and scoring
    04_cross_model_analysis.py       Per-model 2SLS comparison
    05_score_complexity_rubric.py    Rubric scoring with o4-mini (Stage C)
    06_audit_scoring.py              Post-scoring audit: detect silent failures
    07_apply_judge.py                Apply LLM judge to ambiguous cases
figures/
  pipeline.png                       End-to-end pipeline diagram
  complexity_kink.png                Pass rate against rubric composite, with kink at gamma = 8
  per_model_kink_heatmap.png         Per-model Hansen sup-Wald surface across the 21-model panel
  per_model_threshold_forest.png     Forest plot of per-model threshold estimates
  dimension_performance_radar.png    Pass rate by rubric dimension, frontier vs open-source
  frontier_vs_open_source_gap.png    Performance gap widens past the kink
  sankey_misclassification.png       Output cyclomatic complexity to rubric composite, red flows are the bias
  threshold_sensitivity.png          Sensitivity of the threshold to MIN_REGIME and grid density
  ols_vs_iv_correction.png           Naive output-CC curve against rubric-instrumented curve
results/
  analysis_summary.json              Per-model and combined Stage C 2SLS results
  paper_numbers.json                 Specific numbers cited in the paper, with source provenance
docs/
  METHODOLOGY_NOTES.md               Detailed three-stage progression with rationale for each design choice
  model_reference.md                 Profile of every model in the evaluated panel
  predictions_and_rewrite_notes.md   Pre-registered predictions for the Stage C analysis
  reproduction_guide.md              How to reproduce every number in the paper from raw data
docker/
  Dockerfile.scorer                  Sandboxed unit-test execution environment
```

The full generated dataset (~22 GB), API keys, and large intermediate artifacts are excluded by `.gitignore`.

## Reproducing the results

The full pipeline takes about 36 hours of wall-clock time and roughly $400 in API spend across 21 models. Most consumers should rerun only the analysis stage, which finishes in under five minutes.

### Quick path: rerun the analysis from packaged scores

Requires only the rubric scores and per-model scored generations, which are produced by Stages 1 to 5 of the data provenance pipeline. If you have those files in `data/`:

```
pip install -r requirements.txt
python src/extract_paper_numbers.py     # writes results/paper_numbers.json
python src/analyze_kink.py              # writes results/analysis_summary.json plus interactive figures
```

`extract_paper_numbers.py` produces every numeric claim in the paper from the raw rubric and scored files. Re-running it is the fastest way to verify nothing has drifted from the paper's reported values.

### Full path: rebuild from raw OpenCodeInstruct

```
pip install -r requirements.txt
python src/data_provenance/00_extract_from_source.py
python src/data_provenance/01_select_prompts.py
python src/data_provenance/05_score_complexity_rubric.py
python src/data_provenance/02_generate_solutions.py
docker build -t kink-scorer -f docker/Dockerfile.scorer .
python src/data_provenance/03_execute_and_score.py
python src/analyze_kink.py
```

For Stage D and future large panel runs, use provider batch APIs before
realtime generation whenever the provider path supports it. The batch-first
policy and commands are in `docs/stage_d_batch_generation.md`.

API keys for the 21 panel models are loaded from environment variables. See `src/data_provenance/load_keys.py` for the expected names.

### Smoke test on a subset

```
python src/verify_pipeline.py
```

Runs the Stage B keyword pipeline on a 2,000-sample subset and validates outputs against assertion-based checks. Useful for catching regressions in shared infrastructure.

## Models in the evaluated panel

Twenty-one frontier models from 2025 and 2026 across nine providers. The o4-mini scorer is held out and excluded from the panel to preserve the IV exclusion restriction.

| Provider | Models |
| :-- | :-- |
| Anthropic | Claude Opus 4.6, Claude Opus 4.7, Claude Sonnet 4.6 |
| OpenAI | GPT-5.4, GPT-5-mini, GPT-4.1, GPT-OSS-20B, GPT-OSS-120B |
| Google | Gemini 3.1 Pro, Gemini 3 Flash |
| xAI | Grok-3 |
| DeepSeek | DeepSeek V3.2 |
| Moonshot | Kimi K2.5 |
| Alibaba | Qwen 3.6 Plus, Qwen 3.5-9B |
| Mistral | Mistral Large-3, Mistral Small, Ministral-3-14B-reasoning |
| Meta | Llama 3.3-70B |
| Zhipu | GLM 4.7-flash |
| Arcee | Trinity-large |

Full per-model specifications are in `docs/model_reference.md`.

## Limitations and Stage D

Stage C carries four open limitations, all explicitly discussed in the paper.

**Single-scorer rubric.** The instrument is built by one scorer (o4-mini). Any private bias of o4-mini propagates into every observation. Stage D moves to a four-scorer ensemble and reports inter-rater reliability.

**Sampling-layer endogeneity.** The 5,000-prompt sample was stratified into cyclomatic complexity bins computed on the Qwen2.5 reference solution. That measurement carries the same output-side endogeneity the main analysis corrects for. Failed reference solutions collapse to cyclomatic complexity 1 and get binned as easy, so the sample undersamples the structurally complex tail. Stage D restratifies on rubric composite directly, which is computed from the prompt and cannot be contaminated by any solver's failure mode.

**Python-only.** Stage B covered five languages. Stage C is Python only for tractable scoring and consistent unit-test execution. Cross-language replication with the rubric instrument is the most direct extension.

**Cross-sectional only.** We demonstrate the kink exists. We do not track its location across model generations. Whether frontier progress moves the threshold rightward over time is open.

Stage D, planned and scoped in the paper, has four components: rubric-stratified resampling, ensemble scoring, errors-in-variables correction using measured between-scorer variance, and a 500-prompt human-scored calibration subsample.

## Recognition

Second place, undergraduate research division, University of Wisconsin-Milwaukee College of Engineering and Applied Science Student Research Poster Competition, April 2026.

## Citation

```bibtex
@unpublished{hernandez2026complexitykink_c,
  title  = {The Complexity Kink, Revisited: LLM Rubric Instruments for Causal Inference on Code Generation Reliability},
  author = {Hernandez, Michael},
  year   = {2026},
  note   = {Working draft. See paper/complexity_kink_stage_c.tex.}
}

@unpublished{hernandez2026complexitykink_b,
  title  = {The Complexity Kink: Identifying a Hard Reliability Threshold in LLM Code Generation via Instrumental Variables},
  author = {Hernandez, Michael},
  year   = {2026},
  note   = {Stage B preprint. See paper/stage_b_preprint/.}
}
```

## License

MIT License. See `LICENSE`.
