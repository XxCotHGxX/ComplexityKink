# Predictions & Paper Rewrite Notes

**Created:** 2026-04-13
**Author:** Michael Hernandez
**Purpose:** Pre-register predictions before re-running Stage 2 on the new rubric-scored dataset, and track everything that must change in the paper after the numbers land.

---

## 1. Context: why the paper must be rewritten

The current paper (`paper/complexity_kink_2026.tex`) reports results from the **old methodology**:

- Complexity measured as **cyclomatic complexity (CC)** of generated code, via Lizard
- Endogenous: generated code contaminates the regressor (failed generations score CC≈1)
- Reported kink: **κ ≈ 6.5** (CC units), pass-rate collapse 40.4% to 11.8%
- Diagnostics: Hausman χ²=108.5, first-stage F=1,765, Hansen sup-Wald=257.4

The **new methodology** (Stage C, rubric-based):

- Complexity measured by an **LLM rubric** scoring 6 dimensions from the **instruction text alone** (branching, iteration, state, data structures, edge cases, composition)
- Composite score on **0-24 scale** (6 dims x 0-4 each)
- Rubric scorer: o4-mini, excluded from the 12+ model panel
- No output contamination ,  rubric score is pre-generation by construction
- N = 5,000 stratified prompts x 18+ models fully generated (as of 2026-04-13)

**These are different measurement instruments on different scales.** The old 6.5 CC threshold does not translate 1:1 to the rubric axis.

---

## 2. Pre-registered predictions

Recording before Stage 2 runs so the predictions cannot be adjusted post-hoc.

### 2.1 Primary prediction ,  kink location shifts right

**Claim:** The threshold on the 0-24 rubric axis will be in the range **11-13**, not near 6.5.

**Reasoning:**

1. **Scale rescaling.** Old axis was CC (typ. 1-15 for generated code); new axis is rubric composite (0-24). Even with identical underlying difficulty, the threshold coordinate moves right just from the wider range.
2. **Rubric captures dimensions CC misses.** Edge-case handling, state management, and solution composition are zero-weighted in CC but are first-class in the rubric. Prompts that were "simple CC, hard in practice" (e.g., subtle concurrency, careful bounds-checking) will now score higher on the rubric, pushing the failure mass right.
3. **No output contamination deflating κ.** Previously, failed generations collapsed to CC≈1, pulling the mean down and compressing the active range of the regressor. Rubric scores are instruction-derived and not affected, so the distribution expands.

**Falsification conditions:**

- Kink < 9 to prediction wrong; investigate whether rubric is capturing something CC already captured (possible redundancy) or whether scoring is saturating too early
- Kink > 15 to prediction wrong in direction but consistent with "rubric picks up more difficulty"; still meaningful
- No detectable kink to the strongest result; would require re-examining whether the kink was an artifact of CC endogeneity all along (see §3)

### 2.2 Secondary predictions

- **Pass-rate collapse magnitude:** expect similar or sharper drop (from ~45% to ~10%) because cleaner instrument should reveal the structural break more crisply
- **Heterogeneity by model tier:** frontier models (Opus 4.6, GPT-5-mini, Gemini 3) kink later (higher rubric threshold) than open-weight mid-tier (Llama 3.3, Mistral-Small). Smaller/older models kink earlier
- **Language heterogeneity:** Python likely shows sharpest kink (most training data, cleanest ceiling); Go/C++ kinks may be noisier due to smaller per-language subsamples
- **First-stage F:** likely lower than 1,765 (old F used endogenous controls); expect still >10 (rule-of-thumb weak-ID threshold) but could land in 50-500 range
- **Hausman test:** should remain highly significant (endogeneity was real); if it goes insignificant, that is evidence the rubric was cleaner than expected

### 2.3 Meta-prediction ,  what a null result would mean

If Stage 2 finds **no kink** on rubric-scored data, that is not a paper-killer ,  it reframes the paper:

> "The apparent complexity kink in prior work was an artifact of endogenous complexity measurement. When complexity is measured pre-generation from instructions alone, the pass-rate/complexity relationship is smooth, not discretely threshold-shaped."

This is still publishable and arguably a stronger methodological contribution. Document it honestly (ref: `feedback_research_philosophy.md` ,  null results are fine).

---

## 3. Paper sections that must change

### 3.1 Must-rewrite

| Section | Current | Needed |
|---|---|---|
| Abstract | κ=6.5, 40.4%to11.8% | New threshold, new magnitudes |
| §Methodology ,  Complexity measurement | Lizard CC on generated code | Rubric scoring (6 dims, o4-mini scorer, 0-24 scale), justify choice, include scorer validation (r=0.60 with reference CC) |
| §Methodology ,  Instruments | Instruction keyword counts | Keep instruments (these are fine) ,  but update discussion of why they work better with rubric κ than CC κ |
| §Results ,  Main table | Old 2SLS coefficients | Re-run from scratch |
| §Results ,  Threshold figure | Old Hansen sup-Wald plot | Regenerate from new data |
| §Diagnostics | Hausman 108.5, F=1,765, sup-Wald 257.4 | All new numbers |
| §Robustness | Current placebo + bootstrap | Must include Kleibergen-Paap, CC-vs-rubric comparison, fractional probit |
| §Discussion | Interpretation of κ=6.5 | Interpretation of whatever we find + explicit discussion of scale change |

### 3.2 New sections to add

1. **Measurement validation section.** Show CC (old) vs rubric (new) on the same prompts. Correlate them. Explicitly identify prompts where they disagree and argue the rubric is correct.
2. **Why the kink moved.** If the shift matches our §2.1 prediction, frame as "this is what we expected when endogeneity is removed, and here's why." If it lands elsewhere, honest post-hoc analysis.
3. **Robustness table** with all referee checks: Kleibergen-Paap rk Wald F, KP LM, CC-vs-LOC correlation, fractional probit, smooth polynomial vs threshold (AIC/BIC).

### 3.3 Sections that survive mostly intact

- Introduction & motivation (the endogeneity problem is still the core pitch)
- Literature review
- Dataset description (OpenCodeInstruct, language stratification, N=5,000xmodels)
- IV/2SLS theoretical framework
- Conclusion framework (just swap numbers)

---

## 4. Methodological design choices and planned robustness analyses

Design decisions and robustness checks to execute before the rewrite:

1. **Controls (`src/config.py:CONTROL_COLS`) held empty by design.** Any candidate control must be derivable pre-generation from the instruction or from fixed task metadata (language, source). Post-treatment variables derived from the generated code would reintroduce the endogeneity the IV strategy is built to remove (Angrist & Pischke 2009, §3.2.3). Rationale block documented in `config.py`.
2. **Full-pipeline bootstrap for threshold CIs** (`src/run_stage2_iv.py`). Refit Stage 1 inside each bootstrap iteration so uncertainty in the generated regressor propagates into the CI on the threshold.
3. **`MIN_REGIME_SIZE` sensitivity.** Run the sup-Wald test at 100, 250, and 500 and report the threshold and statistic at each to show the kink is not a degrees-of-freedom artifact.
4. **Robustness diagnostics:** Kleibergen-Paap rk Wald F (heteroskedasticity-robust weak-ID), KP LM underidentification, CC-vs-LOC correlation (rules out a length confound), smooth-polynomial alternative vs threshold (AIC/BIC), fractional-probit check for the bounded dependent variable.

---

## 5. Running log

### 2026-04-13 (entry created)

- Cleaned `google_gemini-3-flash-preview.jsonl`: 7,783 to 4,449 records. 551 prompt_ids missing, regenerating via updated `generate_via_gemini_cli.py` with inline retries (no more bad records written).
- Deleted scrap files: haiku-4.5 (4,299), sonnet-4.6_original (1,620), azure_grok-3_cloud (651), x-ai_grok-4 (185), azure_grok-4-fast-reasoning (0), openai_gpt-5.2 (292), openai_gpt-5.3-codex (0), qwen3.5-9b_cloud (1,501), qwen3.5-9b_restore (1,686).
- Partial files kept pending API funds: `openai_gpt-5.4.jsonl` (1), `google_gemini-3.1-pro-preview.jsonl` (839). Plan: top up to 5,000 when funds available, include as out-of-sample robustness check in appendix. If funds never come, paper proceeds with 18-model panel.
- Anthropic Opus 4.6 batch completed: 5,000/5,000 ✓
- Anthropic Sonnet 4.6: 5,000/5,000 ✓
- Current clean panel: 18 models x 5,000 prompts (with Gemini-flash in progress).

### (add entries below as work progresses)
