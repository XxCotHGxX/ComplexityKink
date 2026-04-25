# Methodology Notes: Evolution of Complexity Measurement

These notes document the three-stage progression of how task complexity is measured
in this study. They are preserved here as a reference for paper revisions and to
provide transparency about methodological decisions.

---

## Stage A: Naive Output-Based Complexity (Baseline)

**Method:** Compute cyclomatic complexity (CC) directly from the model's generated
code output using the `lizard` static analysis tool.

**The Endogeneity Problem:**

When a model fails on a complex task, the broken or incomplete output has
artificially low CC. A function that should have nested loops, conditionals, and
error handling instead produces a stub or syntactically invalid code that registers
as CC=1. This creates simultaneity bias: Cov(kappa_obs, epsilon) != 0.

- Failed code registers as "simple" regardless of true task complexity
- Hard tasks appear easy because their outputs are broken
- The complexity-performance curve is systematically flatter than reality
- The structural break (kink) is hidden entirely

This is the measurement approach used in NVIDIA's OpenCodeInstruct dataset and
most existing LLM code benchmarks.

## Stage B: IV/Random Forest Predicted Complexity (Original Contribution)

**Method:** Two-Stage Least Squares (2SLS) with keyword-derived instruments.

- Stage 1: Train a Random Forest on successful solutions (pass_rate=1.0) to predict
  CC from instruction keyword features (if_count, loop_count, class_count, etc.)
- Stage 2: Use predicted kappa-hat as instrument in 2SLS regression

**Improvements over Stage A:**

- Separates task complexity (from prompt keywords) from output quality
- Reveals the "Complexity Kink" at kappa ~ 6.5 that the naive approach hides
- Hausman test confirms endogeneity (chi-squared = 108.5, p < 0.001)
- First-stage F = 1,765 (strong instrument)
- Pass rate drops from 40.4% to 11.8% across the kink

**Known Weaknesses (identified via peer review):**

1. R-squared = 0.531 - almost half the variance unexplained
2. Survivorship bias: RF trained only on successful code; predictions for failures
   are extrapolation from a non-random subsample
3. Generated regressor problem: kappa-hat is estimated, so bootstrap must re-fit
   Stage 1 in every iteration (was not done initially)
4. CC=1 vs CC=2 misclassification: keyword features cannot statistically
   distinguish these low-complexity bins (p not significant)
5. Non-monotonic instruments: inst_tokens and inst_avg_word_len violate IV
   assumptions (longer prompts do not always imply more complex code)
6. Asymmetric out-of-fold predictions: successful samples get OOF predictions,
   failures get full-model predictions

## Stage C: LLM Rubric Scoring (Current Method)

**Method:** Use a frontier LLM (o4-mini, not included in the study) with a rigid
six-dimension rubric to score the structural complexity of each prompt. Each
dimension is scored 0-4, producing a composite range of 0-24.

**Rubric Dimensions:**

1. Branching - conditional paths required in a correct solution
2. Iteration - loops and recursion required
3. State - number of variables tracked simultaneously
4. Data Structures - complexity of data organization required
5. Edge Cases - boundary conditions the code must explicitly check
6. Composition - number of algorithmic steps chained together

**Scoring Model:** o4-mini deployed on Azure AI Foundry. Reasoning models produce
deterministic outputs. Temperature is not configurable (defaults to 1, but
reasoning chains are deterministic by design). The scoring model is explicitly
excluded from the cross-model experiment to avoid circularity.

**Improvements over Stage B:**

| Metric                        | Stage B (IV/RF)              | Stage C (Rubric)                  |
|-------------------------------|------------------------------|-----------------------------------|
| CC=1 vs CC=2 separation       | Not significant              | p < 0.000001                      |
| Correlation with reference CC | R-squared = 0.531 (trained)  | r = 0.60 (zero-shot)             |
| Survivorship bias             | Present                      | Eliminated (scores prompts only)  |
| Generated regressor problem   | Present                      | Eliminated (fixed scores)         |
| Number of instruments         | 8 keyword features           | 6 auditable dimensions            |
| Monotonicity                  | Violated (2 instruments)     | Each dimension monotonic          |
| Interpretability              | Black-box RF                 | Published rubric, reproducible    |

**Key Results:**

- 5,000/5,000 prompts scored successfully
- CC=1 vs CC=2 discrimination: composite difference = 0.79 (p < 0.000001)
- Pearson r = 0.6007 with reference CC (zero-shot, no training)
- Spearman rho = 0.6219
- Monotonically increasing across all CC bins: 4.56 (CC=1) to 12.42 (CC=21-45)

**Dimensions driving low-complexity separation:**
- Iteration: +0.32
- Composition: +0.19
- State: +0.13

## Paper Framing

The progression tells a methodological story:

1. "Here is the problem" - naive output-based CC is endogenous (Stage A)
2. "Here is a statistical fix" - IV/2SLS with keyword-predicted complexity (Stage B)
3. "Here is a better instrument" - LLM rubric scoring eliminates remaining
   weaknesses while preserving the IV framework (Stage C)

This positions the rubric as a natural evolution, not a rejection of the IV
framework. The 2SLS structure remains - only the instrument construction improves.

## Prompt Selection: Scan-Limit Decision

The 5,000-prompt experimental set was drawn by `01_select_prompts.py` with
`--scan-limit 200000`, i.e., stratified sampling over the first 200,000
records of the extracted OpenCodeInstruct pool rather than the full pool.

**Why this matters.** OpenCodeInstruct is a concatenation of multiple seed
corpora (Evol-Instruct, Magicoder, Glaive-derived, etc.), written into
HuggingFace parquet shards in source-grouped order. A prefix scan therefore
risks biasing the language mix, difficulty distribution, and test-quality
distribution relative to a full-pool draw.

**Why we did not re-sample.** The 5,000 prompts have already been evaluated
against ~18 frontier and open-source models at substantial cost. Re-drawing
would invalidate all collected generations. The prompts are locked as the
experimental artifact; the code default has been changed to full-pool scan
(`--scan-limit 0`, the new default) so anyone reproducing from scratch gets
an unbiased draw.

**Representativeness validation (planned).** We verify the locked 5,000-prompt
set is not materially unrepresentative of the full pool by drawing a second,
independent 5,000-prompt sample via full-pool stratified sampling and
comparing:
1. Language distribution (all Python by design, but seed-language of the
   original task may differ)
2. Reference CC distribution per bin
3. Instruction length distribution (tokens)
4. Test count per prompt

If the two samples are statistically indistinguishable on these axes
(Kolmogorov-Smirnov test, p > 0.05), the scan-limit decision introduced no
detectable bias. The validation sample is not evaluated against any model;
it exists only to characterize the sampling frame.

## Validation Still Required

- Inter-rater reliability: score a subset with 2-3 different LLMs, report ICC
- First-stage F-statistic with rubric dimensions as instruments
- Exclusion restriction argument: rubric measures code structure, not LLM-specific
  difficulty
- Sargan-Hansen J-test for over-identification with 6 instruments
- Polynomial alternative test (cubic vs. threshold, AIC/BIC comparison)

## File References

- Rubric scoring script: `src/data_provenance/05_score_complexity_rubric.py`
- Rubric output: `data/complexity_rubric_scores.jsonl`
- IV pipeline: `src/train_stage1_iv.py`, `src/run_stage2_iv.py`
- Pre-registered predictions and planned rewrite: `docs/predictions_and_rewrite_notes.md`
