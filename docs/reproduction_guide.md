# Reproduction guide

This guide covers the locked Stage D analysis. It does not reproduce the
post-submission generation runs from API calls. Their final aggregate values
are recorded separately in `results/robustness_summary.json`.

## Requirements

- Python 3.12 or newer
- Docker for executing generated code
- The packaged benchmark data bundle
- Provider credentials only if regenerating model outputs or rubric scores

Install dependencies from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment in the usual way for your shell.

## Expected Stage D inputs

Place the benchmark bundle under `data/stage_d/` with this layout:

```text
data/stage_d/
|-- stage_d_prompts.jsonl
|-- ensemble_scores_current_aggregated.jsonl
`-- scored_combined/
    `-- one scored JSONL file per evaluated model
```

The important file-name distinction is deliberate:

- `ensemble_scores_current_aggregated.jsonl` is the rubric aggregate paired
  with the locked 5,000-prompt panel.
- The older `ensemble_scores_aggregated.jsonl` came from a pre-replacement run
  and must not be paired with `scored_combined/`.

`src/analyze_kink.py` now warns if a rubric join retains less than 90% of a
scored model file, which usually signals a wrong-file pairing.

## Reproduce the submitted combined analysis

Run:

```bash
python src/analyze_kink.py \
  --scored-dir data/stage_d/scored_combined \
  --rubric data/stage_d/ensemble_scores_current_aggregated.jsonl \
  --prompts data/stage_d/stage_d_prompts.jsonl \
  --outdir results/reproduced_stage_d
```

The main machine-readable output is:

```text
results/reproduced_stage_d/analysis_summary.json
```

Compare its `_combined` object with `results/analysis_summary.json`. The key
fields are:

| Claim | JSON field |
| :-- | :-- |
| Number of prompts | `_combined.N` |
| Robust first-stage Wald chi-square(6) | `_combined.iv_fstat` |
| Sargan-Hansen p-value | `_combined.iv_j_pval` |
| Threshold | `_combined.kink_threshold` |
| Sup-Wald | `_combined.kink_sup_wald` |
| Bootstrap interval | `_combined.kink_ci_lower`, `_combined.kink_ci_upper` |
| Pass rate at or below threshold | `_combined.mean_pass_low` |
| Pass rate above threshold | `_combined.mean_pass_high` |

For a fast pipeline check:

```bash
python src/analyze_kink.py \
  --scored-dir data/stage_d/scored_combined \
  --rubric data/stage_d/ensemble_scores_current_aggregated.jsonl \
  --prompts data/stage_d/stage_d_prompts.jsonl \
  --outdir results/smoke \
  --combined-only \
  --skip-visualizations \
  --n-boot 100 \
  --n-ci-boot 100 \
  --n-placebo 100
```

The smoke run checks code paths and data joins. Its bootstrap values are not
expected to match the locked full run exactly.

## Reproduce a prompt subset

Write one prompt ID per line, then pass:

```bash
python src/analyze_kink.py \
  --scored-dir data/stage_d/scored_combined \
  --rubric data/stage_d/ensemble_scores_current_aggregated.jsonl \
  --prompts data/stage_d/stage_d_prompts.jsonl \
  --restrict-prompts path/to/prompt_ids.txt \
  --outdir results/subset
```

This is the supported route for source-restricted or other prompt-level
robustness checks.

## Regenerate paper figures

After reproducing the locked analysis, place its combined summary at:

```text
results/analysis_summary.json
```

The figure generator also expects:

```text
results/per_model_bootstrap_summary.csv
```

Then run:

```bash
python scripts/generate_stage_d_paper_figures.py
```

It writes the six filenames used by the revised manuscript:

- `paper/pipeline.png`
- `paper/complexity_kink.png`
- `paper/heatmap_E_per_model_kink.png`
- `paper/sankey.png`
- `paper/tail_extension.png`
- `paper/pass_vs_output_cc.png`

The script reads thresholds and statistics from result files or recomputes them
from scored inputs. Reported result values are not typed into the plotting code.
Descriptive prompt-composite bins use
$b=\lfloor C_i+0.5\rfloor$, equivalently
$C_i\in[b-0.5,b+0.5)$. Exact half-point boundaries therefore enter the higher
bin. Breakpoint estimation remains on the continuous, unbinned composite.

## Full benchmark construction

The tracked construction pipeline is under `src/stage_d/`:

1. `01_collect_candidate_prompts.py` collects eligible source prompts.
2. `02_select_rubric_balanced_prompts.py` locks the prompt set.
3. `04_prepare_generation_delta.py` identifies model-prompt rows that need new
   generations.
4. `05_score_rubric_ensemble.py` applies the fixed rubric with held-out judges.
5. `06_aggregate_ensemble_scores.py` builds prompt-level ensemble scores.
6. `07_combine_stage_d_scored.py` combines retained and newly scored rows.
7. `08_audit_unit_tests.py` checks the selected unit-test contracts.

The original 5,000-prompt source draw used `--scan-limit 200000`, a prefix
scan over source-ordered shards. Run
`scripts/check_sample_representativeness.py` against an independent full-pool
reference sample to audit that sampling-frame decision. Stage D then balanced
the collected candidate set using a preliminary single `o4-mini` rubric score.
After the prompt set was locked, the four-judge ensemble rescored it. Every
reported analysis uses the ensemble mean, not the preliminary sampling score.

Provider batch helpers are under `scripts/`. All credentials are read from
environment variables. Local credential loaders, cloud resource names, batch
request files, and provider logs are intentionally excluded from the release.

## Sandboxed execution

Build the scorer image:

```bash
docker build -t kink-scorer -f docker/Dockerfile.scorer .
```

Then score generated rows through:

```bash
python src/data_provenance/03_execute_and_score.py
```

Generated code should never be executed directly on the host. The scorer uses
timeouts and isolated working directories, but Docker is still a required
safety boundary for a full rerun.

## Post-submission checks

The additional checks include construction-frame and task-type controls,
overidentification subsamples, pooling sensitivity, pass@k, human calibration,
paraphrase stability, language transfer, and the audit-clean high-complexity
extension.

This repository publishes their consolidated outputs, not the internal response
drafts, grader keys, provider logs, or account-specific generation runners:

- Human-readable report: `docs/robustness_results.md`
- Machine-readable report: `results/robustness_summary.json`
- Construction-frame output: `results/source_frame_sensitivity.json`
- Mechanism diagnostic: `results/mechanism_diagnostic.json`

With the retained data bundle available, verify the three corrected binned
artifacts and the construction-frame sensitivity with:

```bash
python scripts/regenerate_tail_extension_tables.py \
  --data-root path/to/retained/repository \
  --check
python scripts/regenerate_mechanism_diagnostic.py \
  --data-root path/to/retained/repository \
  --check
python scripts/analyze_source_frame_sensitivity.py \
  --data-root path/to/retained/repository/data \
  --output results/source_frame_sensitivity.json \
  --n-boot 300 \
  --seed 42
```

The tail command checks all four source CSVs, including the fixed-version
three-model table. The construction-frame command validates complete 5,000
prompt mapping and 21 by 5,000 model coverage before fitting the sensitivity.

The raw benchmark extension and generated responses should be distributed in
the separate anonymized data artifact, where their licenses, hashes, and
provenance can be documented without mixing private review material into the
source repository.
