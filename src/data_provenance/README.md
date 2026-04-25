# Data Provenance Pipeline

Reproducible pipeline from raw NVIDIA OpenCodeInstruct data to scored,
cross-model experiment results.

## Quick Start (Reproduction)

```bash
# 1. Extract from NVIDIA's 5M-sample dataset (downloads ~22GB)
python src/data_provenance/00_extract_from_source.py

# 2. Select 5,000 stratified prompts with quality-filtered tests
python src/data_provenance/01_select_prompts.py --n-prompts 5000 --seed 42

# 3. Configure your models (copy the example, add your API keys)
cp src/data_provenance/models_example.json src/data_provenance/models.json
# Edit models.json with your keys

# 4. Generate solutions from each model
python src/data_provenance/02_generate_solutions.py

# 5. Execute tests and score
python src/data_provenance/03_execute_and_score.py

# 6. Cross-model analysis
python src/data_provenance/04_cross_model_analysis.py
```

## Pipeline Architecture

```
OpenCodeInstruct (5M Python, 50 parquet shards on HuggingFace)
        |
        v
+----------------------------+
| 00_extract_from_source     |  Download shards in deterministic order (0..49)
|                            |  Use NVIDIA's tests_execution_status (ground truth)
|                            |  Output: data/final_results_scored.jsonl
+----------------------------+
        |
        v
+----------------------------+
| 01_select_prompts          |  Stratified sample of 5,000 prompts by CC bin
|                            |  Filters: no trivial tests, reference must pass
|                            |  Output: data/experiment_prompts.jsonl
+----------------------------+
        |
        v
+----------------------------+
| 02_generate_solutions      |  Send prompts to each model via API
|                            |  Config: models.json (backends, keys)
|                            |  Output: data/generations/<model>.jsonl
+----------------------------+
        |
        v
+----------------------------+
| 03_execute_and_score       |  Run unit tests, compute CC via lizard
|                            |  Deduplicates per prompt_id (last record wins)
|                            |  Output: data/scored/<model>.jsonl
+----------------------------+
        |
        v
+----------------------------+
| 04_cross_model_analysis    |  Stage 1 RF + Stage 2 2SLS per model
|                            |  Output: output/cross_model_summary.csv
+----------------------------+
```

## Quality Filters (Built Into Step 01)

`01_select_prompts.py` automatically filters out:

1. **Trivial tests** - Rejects prompts where every assertion is `assert X == None`
   (these always pass regardless of code quality)
2. **Broken references** - Rejects prompts where NVIDIA's own solution fails its
   own tests (ground truth must be all-pass)
3. **Empty/missing tests** - Rejects prompts with no unit tests at all

## Robustness Features (Built Into Step 02)

- **Resume-safe** - Restart anytime; only processes prompts not yet completed
- **EMPTY_CODE detection** - Flags reasoning-only outputs (no code block) as errors
- **Auto-cleanup** - On resume, removes stale blank/error records before appending
- **Rate limiting** - Configurable RPM per model to avoid API throttling

## Model Configuration

Copy `models_example.json` to `models.json` and configure your backends:

| Backend | Description | Auth |
|---------|-------------|------|
| `copilot` | GitHub Copilot (GPT models) | Copilot subscription |
| `openai` | OpenAI-compatible API | API key in `api_key` |
| `gemini_cloudcode` | Google Gemini via CLI auth | `gemini-cli` login |
| `local` | LM Studio / llama.cpp | Local server URL |
| `azure` | Azure OpenAI | Endpoint + key |

## Key Design Decisions

- **Same system prompt** for all models (fairness)
- **Temperature = 0.0** for deterministic, reproducible output
- **Last code block extraction** - Reasoning models often output drafts then a
  final version; we take the last `\`\`\`python` block
- **CC computed on generated code** - The complexity metric (`kappa_cyclomatic`)
  is always computed on the model's own output, not the reference solution

## Historical Pipeline (SUPERSEDED)

The scripts `01_prepare_sampling.py` through `04_score_tests.py` document the
**original** data pipeline. They are retained for transparency but should NOT
be used for reproduction. See below for why.

### Known Issues (Historical Only)

1. **Go/Java/C++ tests were never executed** - pass/fail was fabricated
2. **CC defaults to 1 on failure** - conflates "unparsable" with "simple"
3. **Non-deterministic shard ordering** - different runs yield different samples
4. **No Python in language detection** - Python matched via different path
