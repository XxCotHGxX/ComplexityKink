# Reproduction Guide

Step-by-step instructions for reproducing every numeric claim in the Stage C paper from raw OpenCodeInstruct.

## What you need

* Python 3.11 or newer
* Docker (for sandboxed test execution)
* About 22 GB of free disk space
* API access to the 21 panel models (or a subset)
* About $400 in expected API spend if running the full panel from scratch

If you do not need to regenerate the model panel and only want to verify the analysis numbers, skip to the "Analysis only" section. The packaged rubric scores and per-model scored generations are sufficient input for that path.

## Environment setup

```
git clone <anonymous-artifact-repository-url>
cd ComplexityKink
python -m venv .venv
.venv/Scripts/activate          # Windows
source .venv/bin/activate       # Unix
pip install -r requirements.txt
```

API keys are loaded from environment variables. The expected names are listed in `src/data_provenance/load_keys.py`. Set the keys for whichever providers you intend to query.

## Full pipeline from raw data

### Step 1: Extract from OpenCodeInstruct

```
python src/data_provenance/00_extract_from_source.py
```

Pulls a filtered subset from `nvidia/OpenCodeInstruct` on Hugging Face. Filters out tasks with trivial test suites and tasks where the reference solver fails its own tests. Output: `data/final_results_scored.jsonl` (about 12 GB).

### Step 2: Stratified prompt selection

```
python src/data_provenance/01_select_prompts.py
```

Draws 5,000 prompts stratified across cyclomatic complexity bins computed on the Qwen2.5 reference solution. Output: `data/experiment_prompts.jsonl`. The seed is fixed at 42 for reproducibility.

Note that this stratification carries an output-side endogeneity that the Stage C paper discusses as a limitation. Stage D restratifies on rubric composite to fix this.

### Step 3: Rubric scoring

```
python src/data_provenance/05_score_complexity_rubric.py
```

Scores every prompt on six structural dimensions using o4-mini through the Azure AI Foundry endpoint. The scorer is held out from the evaluated panel to preserve the IV exclusion restriction. Output: `data/complexity_rubric_scores.jsonl`.

Three of 5,000 prompts trigger the Azure content filter on benign content. These are scored manually against the published rubric and patched into the output file.

### Step 4: Multi-model generation

```
python src/data_provenance/02_generate_solutions.py --workers 30
```

Generates one solution per prompt for every model in the panel. Resume-safe: if interrupted, it picks up where it left off. Output: `data/generations/<model_id>.jsonl`, one file per model.

This step is the bulk of the wall-clock time and API spend. Plan for roughly 24 hours and $400 with the full panel running concurrently.

### Step 5: Sandboxed scoring

```
docker build -t kink-scorer -f docker/Dockerfile.scorer .
python src/data_provenance/03_execute_and_score.py
```

Runs each generation through the original NVIDIA test suite inside a Docker sandbox. Output: `data/scored/<model_id>.jsonl`. Pass rate is the fraction of test cases passed.

### Step 6: Audit and judge

```
python src/data_provenance/06_audit_scoring.py
python src/data_provenance/07_apply_judge.py
```

Detects silent scoring failures (zero-pass-rate rows that look like infrastructure problems rather than actual code failures) and applies an LLM judge to ambiguous outputs. Output: `data/scored_corrected/<model_id>.jsonl`.

### Step 7: Stage C analysis

```
python src/analyze_kink.py
python src/extract_paper_numbers.py
```

`analyze_kink.py` runs per-model and combined 2SLS with rubric instruments, the Hansen sup-Wald threshold detection with bootstrap inference, the placebo permutation test, and the cross-model comparisons. Output: `results/analysis_summary.json` plus interactive HTML figures.

`extract_paper_numbers.py` produces every specific numeric claim cited in the paper, including the per-dimension first-stage coefficients, the placebo distribution statistics, and the kink sample splits. Output: `results/paper_numbers.json`.

## Analysis only

If you have the rubric scores and scored generations from a prior run, the analysis stage finishes in under five minutes:

```
pip install -r requirements.txt
python src/analyze_kink.py
python src/extract_paper_numbers.py
```

This is the fastest path to verify the paper's reported values against the underlying data.

## Keyword-pilot reproduction

The unpublished keyword-pilot pipeline is reproducible from the same raw data.
It uses keyword features in place of the rubric and a Random Forest in place of
the LLM scorer.

```
python src/feature_extractor_iv.py
python src/train_stage1_iv.py
python src/run_stage2_iv.py
```

Output: keyword features, trained Random Forest weights, and keyword-pilot
threshold estimates.

## Smoke test

```
python src/verify_pipeline.py
```

Runs the keyword-pilot pipeline on a 2,000-sample subset with assertion-based output checks. Useful for catching regressions in shared infrastructure (data loader, parsers, configuration).

## Verifying the headline numbers

After running `extract_paper_numbers.py`, the key claims in the paper map to fields in `results/paper_numbers.json` as follows.

| Paper claim | JSON path |
| :-- | :-- |
| First-stage F = 3,002 | `iv.fstat` |
| Partial R-squared = 0.462 | `iv.partial_r2` |
| OLS coefficient on kappa = -0.00918 | `kappa_ols.coef` |
| 2SLS coefficient on kappa = -0.01310 | `iv.coef` |
| Sargan J = 154.3 | `iv.sargan_stat` |
| Below-kink pass rate = 48.9% | `kink_split.mean_below` |
| Above-kink pass rate = 37.0% | `kink_split.mean_above` |
| Placebo mean sup-Wald = 2.22 | `placebo_distribution.mean` |
| Placebo 95% range = [0.51, 5.22] | `placebo_distribution.p025`, `p975` |

If any of these drift from the paper after a rerun, that is the signal to investigate. The most common cause is an updated rubric score file or a regenerated scored output.

## Per-model numbers

`results/analysis_summary.json` contains one entry per panel model plus a `_combined` entry. Each entry includes the OLS and 2SLS coefficients, the first-stage F, the Hausman statistic, the kink threshold and bootstrap CI, the regime pass rates, and the placebo p-value. Table 4 of the paper is generated directly from this file.
