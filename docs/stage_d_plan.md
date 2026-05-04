# Stage D Plan

Stage D fixes the sampling-layer limitation in Stage C. Stage C balanced prompts
by reference cyclomatic complexity, but the realized rubric distribution is
middle-heavy and misses the high-complexity tail. Stage D keeps the same
rubric definition, goes back to the original OpenCodeInstruct extraction to
find additional prompts, scores those candidates, then prunes back to a final
rubric-balanced prompt set.

## Current Stage C Distribution

Reference cyclomatic-complexity bins are balanced by construction:

| Reference CC bin | Count |
|---:|---:|
| 1 | 556 |
| 2 | 556 |
| 3 | 556 |
| 4 | 556 |
| 5 | 555 |
| 6-7 | 556 |
| 8-10 | 555 |
| 11-15 | 555 |
| 16+ | 555 |

The same prompts are not balanced by rubric composite:

| Rubric bin | Count |
|---:|---:|
| 0-3 | 473 |
| 4-6 | 1,284 |
| 7-9 | 1,942 |
| 10-12 | 997 |
| 13-15 | 269 |
| 16-18 | 35 |
| 19-24 | 0 |

This is the gap Stage D is designed to fill.

## Stage D Prompt Set Built 2026-04-29

We use six rubric-composite bins for the final Stage D sample:

| Rubric bin | Target | Realized | Candidate pool |
|---:|---:|---:|---:|
| 0-3 | 834 | 834 | 932 |
| 4-6 | 834 | 834 | 2,888 |
| 7-9 | 833 | 833 | 6,576 |
| 10-12 | 833 | 833 | 9,968 |
| 13-15 | 833 | 833 | 4,351 |
| 16-24 | 833 | 833 | 847 |

The original 5,000 Stage C prompts contributed 636 retained prompts after the
unit-test quality gate and Azure-filter replacement pass. The remaining 4,364
prompts are newly sourced from the
original dataset. The high
tail is too sparse to support a standalone 19-24 bin; merging 16-24 preserves a
clean estimand while still forcing genuine tail coverage.

Candidate scoring summary:

- 41,996 total Stage D candidate prompts collected.
- 41,994 candidates scored successfully with the Stage C o4-mini rubric.
- 2 candidates repeatedly failed scorer calls and are excluded from the pool.
- 21,434 candidate/Stage-C rows were excluded by the unit-test audit gate.
- The final selected prompt set has zero unit-test audit flags.
- Two otherwise valid prompts, a Fibonacci task and a chemical-formula parser,
  were replaced after Azure content filtering rejected them in judge/generation
  runs. The replacements are an LCM task in `0-3` and a BST task in `13-15`.
- Final prompt set: `data/stage_d/stage_d_prompts.jsonl`.
- Balance report: `data/stage_d/stage_d_prompt_balance_report.json`.

## Prompt Pipeline

1. Collect extra candidates from the original extraction. The final run used a
   broad tail-heavy pool plus two high-reference-tail supplements:

```bash
python src/stage_d/01_collect_candidate_prompts.py \
  --n-candidates 10000 \
  --output data/stage_d/candidate_prompts.jsonl

python src/stage_d/01_collect_candidate_prompts.py \
  --n-candidates 16000 \
  --only-reference-bins 16+ \
  --output data/stage_d/candidate_prompts.jsonl

python src/stage_d/01_collect_candidate_prompts.py \
  --n-candidates 8000 \
  --only-reference-bins 16+ \
  --output data/stage_d/candidate_prompts.jsonl

python src/stage_d/01_collect_candidate_prompts.py \
  --n-candidates 16000 \
  --only-reference-bins 16+ \
  --output data/stage_d/candidate_prompts.jsonl
```

2. Score those candidates with the existing Stage C rubric scorer:

```bash
python src/data_provenance/05_score_complexity_rubric.py \
  --prompts data/stage_d/candidate_prompts.jsonl \
  --output data/stage_d/candidate_rubric_scores.jsonl \
  --workers 64 \
  --max-retries 5 \
  --retry-base-delay 5
```

3. Select the final balanced Stage D prompt set:

```bash
python src/stage_d/02_select_rubric_balanced_prompts.py \
  --bins 0-3,4-6,7-9,10-12,13-15,16-24 \
  --n-prompts 5000 \
  --audit-flags data/stage_d/candidate_unit_test_audit_flags.csv data/stage_d/stage_c_unit_test_audit_flags.csv \
  --exclude-flags contract_hidden_test_callable,contract_io_prompt_callable_tests,risk_external_fixture_or_global,weak_many_duplicate_tests,weak_some_duplicate_tests \
  --output data/stage_d/stage_d_prompts.jsonl \
  --report data/stage_d/stage_d_prompt_balance_report.json
```

4. Prepare retained/new/pruned manifests for model generation:

```bash
python src/stage_d/04_prepare_generation_delta.py
```

5. Generate only newly needed prompts. Use provider batch APIs first whenever
   the provider path supports them; see `docs/stage_d_batch_generation.md` for
   the exact policy and commands. The realtime generator is only for
   realtime-only paths or audited provider/model batch rejections:

```bash
python src/data_provenance/02_generate_solutions.py \
  --prompts data/stage_d/generation_delta/stage_d_new_prompts.jsonl \
  --models src/stage_d/models_stage_d_panel.json \
  --output-dir data/stage_d/generations
```

For exact per-model top-up, use the manifests in
`data/stage_d/generation_delta/per_model_missing/`. The current quality-gated
delta is 4,364 new prompts.

The Stage D model config uses environment-variable placeholders for Azure-backed
panel models. Set the relevant `AZURE_*` endpoint and key variables locally
before running generation or rubric scoring.

6. Execute unit tests for newly generated rows:

```bash
python src/data_provenance/03_execute_and_score.py \
  --prompts data/stage_d/stage_d_prompts.jsonl \
  --generations-dir data/stage_d/generations \
  --output-dir data/stage_d/scored_new
```

7. Combine retained Stage C scored rows with newly scored rows:

```bash
python src/stage_d/07_combine_stage_d_scored.py
```

## Ensemble Scoring Plan

After the final prompt set exists, rescore every Stage D prompt with multiple
out-of-panel judges. The first pass should keep the exact Stage C rubric text
so the measurement target stays fixed. The ensemble should write one row per
`(prompt_id, scorer_id)` plus a separate aggregated file with:

- mean score per dimension
- composite mean
- composite variance / standard deviation
- per-dimension variance
- inter-rater reliability metrics

Use the single o4-mini Stage C score as one scorer only if it remains
out-of-panel. Do not include any evaluated generation model as a scorer.

Current Azure holdout judge config:

```bash
python src/stage_d/05_score_rubric_ensemble.py \
  --prompts data/stage_d/stage_d_prompts.jsonl \
  --scorers src/stage_d/scorers_azure_holdout.json \
  --output data/stage_d/ensemble_scores_long.jsonl \
  --workers 64

python src/stage_d/06_aggregate_ensemble_scores.py \
  --input data/stage_d/ensemble_scores_long.jsonl
```

The configured judges are o4-mini, GPT-5.5, Llama-4-Maverick, and Cohere
Command A. Phi-4-reasoning is deployed but disabled after smoke-test timeouts.
Azure CLI auth must be fresh if a private copy of the config uses `azkey:` key
references. In the anonymized artifact, prefer endpoint/key environment
variables so cloud resource names do not appear in the repository.

Completed ensemble run:

- Successful judge rows: 20,000 / 20,000.
- Coverage: 4 judges for all 5,000 prompts.
- Composite ICC on complete cases: 0.8652 over 5,000 prompts.
- Aggregated output: `data/stage_d/ensemble_scores_aggregated.jsonl`.
- Reliability report: `data/stage_d/ensemble_reliability_report.json`.

## Azure Generation Status

The six Azure-backed study models have complete Stage D scored panels:

| Model | Retained Stage C rows | New Stage D rows | Combined rows |
|---|---:|---:|---:|
| azure/deepseek-v3.2-speciale | 636 | 4,364 | 5,000 |
| azure/gpt-oss-120b | 636 | 4,364 | 5,000 |
| azure/grok-3 | 636 | 4,364 | 5,000 |
| azure/kimi-k2.5 | 636 | 4,364 | 5,000 |
| azure/llama-3.3-70b | 636 | 4,364 | 5,000 |
| azure/mistral-large-3 | 636 | 4,364 | 5,000 |

The combined coverage report is
`data/stage_d/scored_combined_coverage.csv`.

## Human Calibration

Select 500 prompts with:

```bash
python src/stage_d/03_select_human_calibration_sample.py
```

Before ensemble scores exist, this balances across rubric bins. After ensemble
scores exist, pass the aggregated ensemble file with `--ensemble-scores`; the
script will reserve half of each bin for high-disagreement prompts.

## Speed and Scientific Quality Suggestions

- Score a candidate pool larger than needed. High-rubric prompts are rare; a
  10k candidate pool is a starting point, not a guarantee.
- Use a two-pass scoring strategy: one cheap scorer over many candidates to
  find likely tail prompts, then the full ensemble only on retained candidates.
- Keep the rubric text fixed across scorers. Prompt drift across judges creates
  measurement drift, not independence.
- Save raw scorer responses and parsed scores. Reproducibility requires both.
- Track scorer identity, model version/deployment, timestamp, and rubric hash.
- Prioritize human calibration on high-disagreement prompts, not only random
  prompts. Random coverage estimates average reliability; disagreement coverage
  diagnoses failure modes.
- Generate model answers only for new Stage D prompts. Reuse retained Stage C
  generations/scored rows by prompt id.
