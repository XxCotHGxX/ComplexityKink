# The Complexity Kink

**Prompt-side structural complexity and code-generation reliability**

Michael Hernandez, University of Wisconsin-Milwaukee

Faculty advisor and review-period collaborator: Tian Zhao,
University of Wisconsin-Milwaukee

This is Michael's named public research copy. The
[official named Stage C paper](paper/Complexity_Kink_Official.pdf) is available
as a PDF. The active NeurIPS manuscript source remains anonymous because it is
under double-blind review. The reviewer-facing artifact should use its separate
anonymous URL, not this GitHub repository.

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

The prompt set was stratified across six bands of a preliminary single-rater
rubric score, then locked and rescored by four out-of-panel LLM judges. The
four-judge stage yields 19,997 score rows: 4,998 prompts have four ratings, one
has three, and one has two. All reported analyses use the ensemble composite,
which is compared with unit-test pass rate across 21 evaluated models.

## Artifact status

This snapshot contains the current manuscript revision and distinguishes
the locked submitted results from the checks added during review:

- `paper/Complexity_Kink_Official.pdf` is the named Stage C public paper.
- `paper/Scratch-NeurIps.tex` is the anonymous revised NeurIPS manuscript source.
- `results/analysis_summary.json` and
  `results/per_model_bootstrap_summary.{csv,json}` contain the locked Stage D
  analysis used for the submitted results.
- `docs/robustness_results.md` and `results/robustness_summary.json` record the
  additional robustness checks.

The old Stage C result JSON, duplicate Stage C manuscript, and stale derived
figures were removed from this snapshot.

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
| Wild bootstrap and placebo | 0 of 2,000 exceedances each; $p_{\mathrm{MC}}<0.001$ |
| Mean pass rate at or below threshold | 79.9% |
| Mean pass rate above threshold | 87.6% |

The unadjusted mean-pooled curve is nonlinear, not a universal "harder means
worse" collapse. Pass rates fall through a mid-complexity region and rebound in
the better-supported part of the high region. This is a descriptive feature of
the constructed benchmark. Its location, direction, and size vary across
construction frames, task controls, pooling choices, and models.

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
- Construction-frame sensitivity is more consequential. Adding an indicator
  for the 2,246 retained earlier prompts versus 2,754 later candidates moves the
  selected threshold to 8.50 and the raw regime gap to -3.48 points. The earlier
  frame has no significant breakpoint. The later frame selects 14.25, but its
  raw gap is only +0.87 points. Neither frame reproduces the pooled +7.6-point
  increase, so we treat the large rebound as source-composition-sensitive.
- A contract-audited extension adds 365 prompt-side-selected prompts, with
  218/133/11/3 prompts in display bins 15/16/17/18. At bins 15 and 16,
  matched-five-model pass rates are 0.880 and 0.799, compared with 0.894 and
  0.808 in the original panel. The differences are -0.014 and -0.009 and are
  not significant ($p=0.377$ and $p=0.765$). Evidence above bin 16 remains too
  sparse for a strong endpoint claim.
- Human calibration shows meaningful signal and meaningful disagreement. On a
  deliberately difficult 200-prompt sample, human-LLM Pearson correlation is
  0.41 and ICC(2,1) is 0.40. On the 50-prompt overlap, the two human graders
  correlate at 0.56.
- A five-draw check on 359 prompts gives single-draw versus five-draw
  correlation $r=0.960$, with the same estimated threshold of 14.25. This is a
  part-whole comparison because the first draw contributes to the mean.
- Prompt paraphrases preserve the ordering well
  (Spearman $\rho=0.963$, 91% within one composite point).
- Java and C++ prompt re-expressions preserve the prompt-side score ordering
  ($r=0.992$ and $r=0.969$); generation and execution remain Python-only.

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
|   |-- Complexity_Kink_Official.pdf
|   |-- Scratch-NeurIps.tex
|   `-- *.png
|-- results/
|   |-- analysis_summary.json
|   |-- mechanism_diagnostic.json
|   |-- pass_vs_output_cc.csv
|   |-- per_model_bootstrap_summary.csv
|   |-- per_model_bootstrap_summary.json
|   |-- reverse_threshold_zero_pass_cells.csv
|   |-- robustness_summary.json
|   |-- source_frame_sensitivity.json
|   |-- tail_extension_curve.csv
|   |-- tail_extension_fixed_version.csv
|   |-- tail_extension_replication.csv
|   `-- tail_extension_source_split.csv
|-- scripts/
`-- src/
```

Large prompt bundles, generated model outputs, provider request logs, API keys,
and local cloud-resource configurations are not committed here.

Descriptive integer bins use one explicit convention throughout: display bin
$b$ contains composite values in $[b-0.5,b+0.5)$, so exact half-point
boundaries enter the higher bin. The inferential threshold searches use the
continuous, unbinned composite.

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

One additional locally served, quantized AuroraGPT-IT-v4 run covered only the
earlier prompt frame and was not generated for the 2,754 newly added prompts.
It was excluded before the final panel analysis in a post hoc decision without
a prespecified eligibility rule. Its pass rate on the earlier frame was 20.6%,
and performance was not a documented exclusion criterion. All claims are
limited to the reported 21-model panel.

## License

Repository code is under the MIT License. See `LICENSE`. OpenCodeInstruct is
used under CC BY 4.0, Lizard under the MIT License, and linearmodels under the
NCSA License.
