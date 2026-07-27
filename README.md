# The Complexity Kink

**Prompt-side structural complexity and code-generation reliability**

Michael Hernandez, University of Wisconsin-Milwaukee

Faculty advisor and review-period collaborator: Tian Zhao,
University of Wisconsin-Milwaukee

This is Michael's named public research copy. The manuscript source remains
anonymous because it preserves the version submitted for double-blind review.
The reviewer-facing artifact should use its separate anonymous URL, not this
GitHub repository.

## What this project studies

Code-generation benchmarks often measure complexity from the code a model
produces. That creates a measurement problem: a failed answer to a difficult
prompt can be a short stub or partial program, so an output-side metric can make
the failure look artificially simple.

This project measures intended solution structure from the prompt before
generation. Each prompt is scored on six fixed dimensions:

- branching
- iteration
- state
- data structures
- edge cases
- algorithmic composition

Four out-of-panel LLM judges score the 5,000-prompt benchmark. The resulting
prompt-side composite is compared with unit-test pass rate across 21 evaluated
models.

## Artifact status

This snapshot separates the submitted analysis from the checks added during
review:

- `paper/Scratch-NeurIps.tex` is the anonymous submitted manuscript source. It
  is preserved as a submission snapshot rather than silently rewritten.
- `results/analysis_summary.json` and
  `results/per_model_bootstrap_summary.{csv,json}` contain the locked Stage D
  analysis used for the submitted results.
- `docs/robustness_results.md` and `results/robustness_summary.json` record the
  post-submission checks requested during review.

The old Stage C result JSON, duplicate Stage C manuscript, and stale derived
figures were removed from this snapshot. They remain recoverable from Git
history.

## Submitted result

| Quantity | Value |
| :-- | --: |
| Prompts | 5,000 |
| Evaluated models | 21 |
| Model-prompt generations | 105,000 |
| Rubric judges | 4 out-of-panel LLMs |
| Composite inter-rater reliability | ICC = 0.872 |
| Combined threshold | $\hat{\gamma}=13.75$ |
| 95% bootstrap interval | $[7.75,14.0]$ |
| Sup-Wald statistic | 121.70 |
| Placebo p-value | $p<0.001$ |
| Mean pass rate below threshold | 79.9% |
| Mean pass rate at or above threshold | 87.6% |

The stable finding is a strong nonlinear regime change, not a universal
"harder means worse" collapse. Pass rates fall through a mid-complexity region
and rebound in the better-supported part of the high-complexity region. The
direction and size of the change vary across models and task types.

## What changed after review

The additional checks make the interpretation narrower and more useful:

- The full six-dimension overidentification test rejects strongly. At
  $n=5{,}000$, $J=411.3$ and $J/N=0.082$. Random subsamples still reject in
  97.5% of draws at $n=250$, so this is not explained by sample size alone. We
  treat the composite primarily as a pre-generation complexity index. The 2SLS
  estimates are secondary diagnostics, not clean causal estimates.
- Task-type fixed effects reduce the sup-Wald statistic from 121.7 to 19.6 and
  the regime gap from 7.6 to 2.1 percentage points, while the break remains
  significant. Task composition explains a substantial part of the pooled
  shape, but not all of it.
- A contract-audited extension adds 365 prompt-side-selected prompts, with
  218/140/4/3 prompts in bins 15/16/17/18. At bins 15 and 16, matched-five-model
  pass rates are 0.88 and 0.80, compared with 0.89 and 0.81 in the original
  panel. The differences are both -0.010 and are not significant
  ($p=0.53$ and $p=0.73$). Evidence above bin 16 remains too sparse for a strong
  endpoint claim.
- Human calibration shows meaningful signal and meaningful disagreement. On a
  deliberately difficult 200-prompt sample, human-LLM Pearson correlation is
  0.41 and ICC(2,1) is 0.40. On the 50-prompt overlap, the two human graders
  correlate at 0.56.
- A five-draw check on 359 prompts gives single-draw versus five-draw
  correlation $r=0.960$, with the same estimated threshold of 14.25.
- Prompt paraphrases preserve the ordering well
  (Spearman $\rho=0.963$, 91% within one composite point).
- Java and C++ ports preserve the prompt-side score ordering
  ($r=0.992$ and $r=0.969$).

Full definitions, sample sizes, and limitations are in
`docs/robustness_results.md`.

## Repository layout

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- docker/
|-- docs/
|   |-- reproduction_guide.md
|   |-- robustness_results.md
|   `-- model_reference.md
|-- paper/
|   |-- README.md
|   |-- Scratch-NeurIps.tex
|   `-- *.png
|-- results/
|   |-- analysis_summary.json
|   |-- per_model_bootstrap_summary.csv
|   |-- per_model_bootstrap_summary.json
|   `-- robustness_summary.json
|-- scripts/
`-- src/
```

Large prompt bundles, generated model outputs, provider request logs, API keys,
and local cloud-resource configurations are not committed here.

## Quick analysis check

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the combined analysis with the scored model directory, aggregated rubric
scores, and prompt file from the packaged benchmark artifact:

```bash
python src/analyze_kink.py \
  --scored-dir path/to/scored_generations \
  --rubric path/to/ensemble_rubric_scores.jsonl \
  --prompts path/to/prompts.jsonl \
  --outdir results/analysis_current
```

For a faster smoke run, set smaller bootstrap counts:

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

See `docs/reproduction_guide.md` for the full pipeline and the input-to-output
map.

## Evaluated panel

| Provider | Models |
| :-- | :-- |
| Anthropic | Claude Opus 4.6, Claude Opus 4.7, Claude Sonnet 4.6 |
| OpenAI | GPT-5.4, GPT-5-mini, GPT-4.1, GPT-OSS-20B, GPT-OSS-120B |
| Google | Gemini 3.1 Pro Preview, Gemini 3 Flash |
| xAI | Grok-3 |
| DeepSeek | DeepSeek V3.2 |
| Moonshot | Kimi K2.5 |
| Alibaba | Qwen 3.6 Plus, Qwen 3.5-9B |
| Mistral | Mistral Large-3, Mistral Small 2412, Ministral-3-14B-reasoning |
| Meta | Llama 3.3-70B |
| Zhipu | GLM 4.7-flash |
| Arcee | Trinity-large |

The four rubric judges are excluded from the evaluated model panel.

## License

MIT License. See `LICENSE`.
