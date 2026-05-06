# The Complexity Kink

**Prompt-side complexity measurement for code-generation reliability**

Anonymous authors. This repository is prepared for double-blind review.

This repository contains the code and artifact scaffolding for the Complexity Kink study. The project evaluates whether code-generation reliability changes regime as the intended structural complexity of a prompt increases, and whether output-derived complexity metrics hide that relationship when models fail.

## Overview

The experiment evaluates 21 frontier language models on 5,000 stratified Python coding prompts drawn from NVIDIA OpenCodeInstruct. Each model produces one answer per prompt, yielding 105,000 model-prompt generations. Generated answers are scored by unit-test execution, and generated-output cyclomatic complexity is computed with Lizard on cleaned generated Python code.

The key measurement change is prompt-side complexity scoring. Before any evaluated model generates code, each prompt is scored with a fixed six-dimension structural rubric:

- branching
- iteration
- state
- data structures
- edge cases
- algorithmic composition

Four out-of-panel LLM judges score the prompt set. Coverage is effectively complete: 4,998 prompts have four judge scores, one prompt has three, and one prompt has two, for 19,997 total judge-score rows. The prompt-level instrument is the ensemble mean of the six rubric dimensions.

## Current Results

The combined 21-model analysis detects a structural break in the prompt-side complexity relationship:

| Quantity | Value |
| :-- | --: |
| Prompts | 5,000 |
| Evaluated models | 21 |
| Model-prompt generations | 105,000 |
| Rubric judges | 4 out-of-panel LLMs |
| Composite inter-rater reliability | ICC = 0.872 |
| Combined threshold | gamma = 13.75 |
| 95% bootstrap CI | [7.75, 14.0] |
| Sup-Wald statistic | 121.70 |
| Placebo p-value | < 0.001 |
| Mean pass rate below threshold | 79.9% |
| Mean pass rate at/above threshold | 87.6% |

The final panel does not support a simple "harder means worse" linear story. Pass rates decline through a mid-complexity region and rebound in the high-complexity region. The stable result is the presence of a statistically strong nonlinear regime change, not a universal monotone collapse after one cutoff.

The reverse-threshold mechanism remains visible when conditioning on complete failures. Among zero-pass generations, 28.5% have prompt-side rubric composite above 8 while Lizard output cyclomatic complexity is at most 10. These are cases where an output-side metric would place a failed response into a low or moderate complexity bin even though the prompt was structurally nontrivial.

## Econometric Diagnostics

The analysis uses the six rubric dimensions as instruments for observed generated-output cyclomatic complexity in a 2SLS diagnostic model. The first stage is strong:

| Diagnostic | Value |
| :-- | --: |
| First-stage reduced-form F | 13,611.3 |
| Partial R-squared | 0.681 |
| Hausman chi-squared | 17.16 |
| Hausman p-value | 3.4e-5 |
| Sargan-Hansen J | 411.3 |
| Sargan-Hansen p-value | < machine precision |

The overidentification test rejects for the full six-instrument set. We treat this as a design caveat rather than as validation. Leave-one-out and leave-two-out checks do not isolate a single removable dimension that resolves the rejection, although data structures and composition account for the largest reductions in the J statistic. Smaller coherent subsets such as branching plus edge cases, and branching plus iteration plus edge cases, pass overidentification while retaining strong first stages; these are used as robustness diagnostics.

## Repository Structure

```text
.
LICENSE
README.md
requirements.txt
paper/
  Scratch-NeurIps.tex
  checklist.tex
  neurips_2026.sty
src/
  analyze_kink.py
  data_loader.py
  extract_paper_numbers.py
  feature_extractor_iv.py
  generate_viz.py
  generate_viz_plotly.py
  run_stage2_iv.py
  verify_pipeline.py
  parsers/
  data_provenance/
docs/
  model_reference.md
  reproduction_guide.md
docker/
  Dockerfile.scorer
scripts/
```

Large generated datasets, API keys, provider logs, and intermediate result bundles are excluded from the repository.

## Reproducing the Analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main analysis from scored model outputs and aggregated rubric scores. Replace the input paths with the packaged artifact paths for the scored generations, ensemble rubric scores, and prompt file:

```bash
python src/analyze_kink.py \
  --scored-dir path/to/scored_generations \
  --rubric path/to/ensemble_rubric_scores.jsonl \
  --prompts path/to/prompts.jsonl \
  --outdir results/analysis_current
```

For a faster smoke run, reduce the bootstrap counts:

```bash
python src/analyze_kink.py \
  --scored-dir path/to/scored_generations \
  --rubric path/to/ensemble_rubric_scores.jsonl \
  --prompts path/to/prompts.jsonl \
  --outdir results/analysis_smoke \
  --n-boot 100 \
  --n-ci-boot 100 \
  --n-placebo 100
```

For per-model bootstrap runs on a Windows workstation, use the parallel bootstrap helper under `scripts/`.

## Evaluated Model Panel

The evaluated panel contains 21 models from major hosted and open-weight providers:

| Provider | Models |
| :-- | :-- |
| Anthropic | Claude Opus 4.6, Claude Opus 4.7, Claude Sonnet 4.6 |
| OpenAI | GPT-5.4, GPT-5-mini, GPT-4.1, GPT-OSS-20B, GPT-OSS-120B |
| Google | Gemini 3.1 Pro, Gemini 3 Flash |
| xAI | Grok-3 |
| DeepSeek | DeepSeek V3.2 |
| Moonshot | Kimi K2.5 |
| Alibaba | Qwen 3.6 Plus, Qwen 3.5-9B |
| Mistral | Mistral Large-3, Mistral Small 2412, Ministral-3-14B-reasoning |
| Meta | Llama 3.3-70B |
| Zhipu | GLM 4.7-flash |
| Arcee | Trinity-large |

The rubric judges are excluded from the evaluated model panel.

## Notes for Review

This artifact is intended to support anonymous review. It contains code, prompt/rubric workflow definitions, and reproducibility scripts. Provider credentials, paid-generation logs, and large generated-output bundles are not committed.

## License

MIT License. See `LICENSE`.
