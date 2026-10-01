# The Complexity Kink

**Prompt-side structural complexity and code-generation reliability**

Code and analysis artifact for *The Complexity Kink: LLM Rubric Instruments for
Causal Inference on Code Generation Reliability* by Michael Hernandez and Tian
Zhao (University of Wisconsin–Milwaukee), accepted at the NeurIPS 2026
Evaluations & Datasets track. Preprint:
[arXiv:2609.19616](https://arxiv.org/abs/2609.19616).

- Paper: [`paper/Scratch-NeurIps.pdf`](paper/Scratch-NeurIps.pdf) (camera-ready). The
  LaTeX source is not distributed with this repository.
- Benchmark data: the release (prompts, per-judge rubric scores, all 105,000
  generations with three outcome definitions, audit verdicts, and Lizard
  complexity, the outcome audit's known-answer sets, human calibration grades,
  and the 365-prompt extension, with Croissant metadata) will be linked here;
  its dataset card and builder are in [`release/`](release/) and
  [`scripts/build_public_release.py`](scripts/build_public_release.py).
- Outcome audit: [`docs/independent_audit_protocol.md`](docs/independent_audit_protocol.md)
  documents how every unit-test verdict was reviewed by an independent LLM
  auditor, the pre-declared selection rule, its amendments, and every
  departure from it.
- Reproduction: [`docs/reproduction_guide.md`](docs/reproduction_guide.md).


## What this project studies

Code-generation benchmarks often measure complexity from the code a model
produces. That creates a measurement problem: a failed answer to a difficult
prompt can be a short stub or partial program, so an output-side metric can make
the failure look artificially simple.

This project scores intended solution structure from the prompt alone, never
from generated code or test results. Each prompt is scored on six fixed
dimensions:

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
which is compared with pass rate across 21 evaluated models.

## Camera-ready result

The primary outcome is the unit-test pass rate as reviewed by an independent
LLM auditor (MiMo-V2.6-Pro): a `correct` verdict sets pass rate to 1.0 and
`incorrect` to 0.0; uncertain, unparseable, or cut-off responses keep the
harness value.

| Quantity | Value |
| :-- | --: |
| Prompts / evaluated models / generations | 5,000 / 21 / 105,000 |
| Composite inter-rater reliability | ICC = 0.872 |
| Mean pass: audited / reviewed version / raw harness | 0.851 / 0.820 / 0.792 |
| Pooled threshold | $\hat{\gamma}=14.0$ (sup-Wald 151.15) |
| 95% pairs-bootstrap percentile interval | $[10.5,14.25]$ |
| Wild bootstrap and placebo | 0 of 2,000 exceedances each; $p_{\mathrm{MC}}<0.001$ |
| Mean pass at or below / above the threshold | 83.4% / 90.1% |
| Task-adjusted jump at 14.0 (additive task fixed effects) | +0.4 points ($p=0.66$) |
| Later-frame break (change) / earlier frame | 14.25 (+1.8 points) / no significant break |
| Models with lower pass above their own break | 2 (Kimi K2.5; Gemini 3.1 Pro Preview, driven by 156 API no-response generations) |

The pooled curve is nonlinear, not a universal "harder means worse" collapse,
and its jump is mostly composition: task type removes nearly all of it, and the
two construction frames (which also differ in generation run and test quality)
explain the rebound. Kimi K2.5's later-frame decline at 14.25 holds under every
outcome definition. The pooled breakpoint is 11.25 on raw unit-test outcomes
and 13.75 under the reviewed version's partial audit.

## Robustness checks

- The six-dimension overidentification test rejects strongly ($J=552.2$ at
  $n=5{,}000$; every random subsample rejects at $n=250$), so the composite is
  treated as an index and the 2SLS estimates as diagnostics only.
- A 365-prompt audit-clean extension (raw harness outcomes on both sides,
  generated under a slightly different protocol) matches the original data at
  bins 15 and 16 (0.880 vs 0.890; 0.799 vs 0.795) but adds only 14 prompts
  above bin 16.
- Human calibration shows meaningful signal and meaningful disagreement: on a
  deliberately difficult 200-prompt sample, human-LLM Pearson correlation is
  0.41 and ICC(2,1) is 0.40; the two human graders correlate at 0.56 on their
  50-prompt overlap.
- A five-draw check on 359 prompts gives single-draw versus five-draw
  correlation $r=0.960$, with the same threshold of 14.25 (part-whole).
- Prompt paraphrases preserve the ordering (Spearman $\rho=0.963$, 91% within
  one point); Java and C++ re-expressions preserve the prompt-side scores
  ($r=0.992$ and $0.969$); generation and execution remain Python-only.

`docs/robustness_results.md` records the review-period values (reviewed
outcome). Camera-ready values are in `results/camera_ready/`.

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
|   `-- Scratch-NeurIps.pdf
|-- release/            (dataset card, Croissant RAI fields, license notes)
|-- results/
|   |-- camera_ready/    (every camera-ready result, one folder per outcome definition)
|   |-- independent_audit/
|   |-- analysis_summary.json   (reviewed version, kept for reference)
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
|   `-- camera_ready/    (reanalysis for any outcome definition)
`-- src/
    `-- audit/           (independent outcome audit)
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

Run the combined analysis with the scored model directory
(`data/stage_d/scored_independent_audit` for the camera-ready primary outcome),
aggregated rubric scores, and prompt file from the packaged benchmark artifact.
`scripts/camera_ready/run_reanalysis.sh` runs this and every other analysis
for one outcome definition:

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
| Mistral | Mistral Large-3, Devstral Small 2505, Ministral-3-14B-reasoning |
| Meta | Llama 3.3-70B |
| Zhipu | GLM 4.7-flash |
| Arcee | Trinity-large |

The four rubric judges are excluded from the evaluated model panel.

One additional locally served, quantized AuroraGPT-IT-v4 run covered only the
earlier prompt frame and was not generated for the 2,754 newly added prompts.
It was excluded before the final panel analysis in a post hoc decision without
a prespecified eligibility rule. Its pass rate was 20.6% on that run's 5,000
prompts (23.2% on the 2,246 retained earlier-frame prompts) under the reviewed
outcome, and performance was not a documented exclusion criterion. All claims are
limited to the reported 21-model panel.

## License

Repository code is under the MIT License. See `LICENSE`. OpenCodeInstruct is
used under CC BY 4.0, Lizard under the MIT License, and linearmodels under the
NCSA License.
