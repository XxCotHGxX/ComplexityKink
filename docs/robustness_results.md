# Post-submission robustness results

This note records the checks completed during the review period. It keeps the
submitted 5,000-prompt analysis separate from new evidence and states the
limits directly. The machine-readable values are in
`results/robustness_summary.json`.

## 1. Submitted baseline

The submitted Stage D panel contains 5,000 prompts and 21 evaluated models.
Four out-of-panel judges score each prompt against the same six-dimension
structural rubric. The combined threshold estimate is
$\hat{\gamma}=13.75$, with sup-Wald 121.70 and a 95% bootstrap interval of
$[7.75,14.0]$. Mean pass rate is 0.799 below the threshold and 0.876 at or
above it.

The pooled curve is non-monotone. It falls through a mid-complexity region and
then rises in the better-supported part of the high-complexity region. That
shape is not a universal failure threshold, and the direction varies across
models.

## 2. Instrument validity

The full six-instrument specification rejects the overidentifying
restrictions: $J=411.3$ on five degrees of freedom, with $J/N=0.082$.

We tested whether this was merely a large-sample rejection. Across 200 random
subsamples at each size from $n=250$ through $n=4{,}000$, $J/N$ changes only
modestly, from 0.102 to 0.082. At $n=250$, 97.5% of the draws still reject at
the 5% level. The restrictions are genuinely violated.

The dimension-level estimates show why. In just-identified fits,
`data_structures` has a positive coefficient, while `composition` and `state`
have negative coefficients. Overidentification should reject when the
dimensions imply different structural effects.

We therefore use the ensemble composite primarily as a prompt-side complexity
index. Coherent subsets that do not reject are limited IV diagnostics, not
proof that the exclusion restriction holds. Principal-component rotation does
not fix the problem: all overidentified leading-PC specifications reject, and
PC1 alone is just-identified.

## 3. Task-type composition

All 5,000 prompts were labeled with a nine-category taxonomy fixed before the
analysis. One out-of-panel judge labeled the full set. Three additional
out-of-panel judges labeled a shared 500-prompt reliability sample, giving
Krippendorff's $\alpha=0.787$.

Adding task-type fixed effects changes the result substantially:

| Quantity | No controls | Task-type fixed effects |
| :-- | --: | --: |
| Sup-Wald | 121.7 | 19.6 |
| Regime gap | +7.6 points | +2.1 points |
| Estimated threshold | 13.75 | 10.75 |

The controlled break remains significant under the wild bootstrap, but much of
the pooled shape is task composition. The composite threshold term adds
$R^2=0.053$ beyond task-type effects. Overidentification still rejects within
four of the five task types large enough to test, so the instrument disagreement
is not localized to one task category.

## 4. Clean high-complexity extension

The extension was selected from prompt-side scores, then passed through a full
contract audit and reference-execution gate. Every retained reference solution
passes every test.

The final extension contains 365 prompts:

| Rounded composite bin | Added prompts |
| --: | --: |
| 15 | 218 |
| 16 | 140 |
| 17 | 4 |
| 18 | 3 |

At the well-supported bins, matched-five-model pass rates closely reproduce the
original panel:

| Bin | Extension | Original matched panel | Difference | Welch p-value |
| --: | --: | --: | --: | --: |
| 15 | 0.88 | 0.89 | -0.010 | 0.53 |
| 16 | 0.80 | 0.81 | -0.010 | 0.73 |

Combining the extension with the original matched frame gives
$\hat{\gamma}=14.0$, with pass rate 0.783 below and 0.859 above. Support at bins
15 and 16 rises to 878 and 417 prompts.

Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro Preview each completed 801 of 801
requested generations before the audit filter. On the retained analysis frame,
all three score about 0.94 at bin 15 and 0.85 to 0.88 at bin 16.

Only seven audit-clean additions lie above bin 16. The extension supports the
high-regime pattern through bin 16, but the extreme tail remains unresolved.

## 5. Human calibration

Michael Hernandez and Tian Zhao graded prompts with the LLM scores hidden.
Michael scored 200 prompts. Tian scored a 50-prompt overlap. Half of the
200-prompt set was deliberately drawn from the highest-disagreement cases, so
this is a stress test rather than an average-case population estimate.

On the full 200 prompts:

- human-LLM Pearson correlation: 0.41
- human-LLM Spearman correlation: 0.37
- ICC(2,1): 0.40
- dimension-level quadratic-weighted kappa range: 0.22 to 0.44
- mean human minus LLM composite offset: -1.18 points

On the shared 50 prompts:

- Michael versus LLM Pearson: 0.61
- Tian versus LLM Pearson: 0.92
- human versus human correlation: 0.56
- Tian minus LLM composite offset: -1.35 points

The rubric carries signal, but neither a single human nor an LLM ensemble
should be treated as ground truth.

## 6. Repeated sampling

The pass@k check uses 359 prompts, four models, and five generations per
prompt-model pair, for 7,180 executed generations.

- Mean within-prompt-model standard deviation: 0.064
- First draw versus five-draw mean Pearson correlation: 0.960
- Threshold from the first draw: 14.25
- Threshold from the five-draw mean: 14.25

This is a part-whole comparison, but it shows that the estimated threshold is
not being set by a single unusual completion in this sample.

## 7. Wording sensitivity

We rewrote 150 threshold-region prompts in plainer language while preserving
their requirements. Median paraphrase length is 46% of the original.

- Original versus paraphrase Spearman correlation: 0.963
- Pearson correlation: 0.987
- Mean composite shift: -0.115 on the 0 to 24 scale
- Share within one point: 91.3%
- Share within two points: 99.3%

The score ordering is stable under a large reduction in wording and verbosity.

## 8. Language transfer

We re-expressed 117 prompts in Java and 117 in C++, then rescored intended
structure. This checks measurement transfer, not full execution-based
reliability in the other languages.

| Language | Correlation with Python | Inter-judge ICC |
| :-- | --: | --: |
| Python, same prompts | 1.000 | 0.912 |
| Java | 0.992 | 0.913 |
| C++ | 0.969 | 0.882 |

A full generation and execution rerun outside Python remains future work.

## 9. Output-side complexity

Generated-output cyclomatic complexity is computed with Lizard on cleaned model
output. It is not dataset metadata, a post-generation rubric score, or reference
solution complexity.

Among zero-pass generations, 28.5% have prompt-side composite above 8 but
generated-output cyclomatic complexity at most 10. The corresponding shares are
26.8%, 24.9%, and 24.1% when pass rate is at most 0.25, 0.50, and 0.65.

Using passing generations only, a fitted scale mapping places
$\hat{\gamma}=13.75$ at generated-output cyclomatic complexity about 22.9. That
mapping is descriptive. It is deliberately estimated outside the failure region
where output complexity is most contaminated.

## 10. Mechanism and model dependence

A 404-prompt diagnostic at bins 13 and 17 does not support the earlier
library/framework explanation. After controlling for ordinary prompt features,
the external-library tag has coefficient -0.045 with $p=0.67$. The mechanism is
therefore left open.

Leave-one-model-out threshold estimation returns 13.75 in all 21 fits. Median
pooling still favors the piecewise model, but moves the threshold to 10.75 and
narrows the regime gap. Across individual models, 16 move upward and five move
downward after their estimated breaks. The pooled break is stable, while its
size and direction are model-dependent.
