# NeurIPS Revision Handoff

Date: 2026-05-03

This note preserves the planned edits while `paper/Scratch-NeurIps.tex` is being overwritten by the advisor-edited copy.

## Advisor Direction

Dr. Zhao's latest guidance:

> Section 5 has some results that may be more interesting to statisticians or econometricians. Summarize the essence in the main paper and leave details in the appendix. Section 5 should mostly report results related to LLM testing.

Interpretation for the next edit pass:

- Keep the main Results section readable as an LLM evaluation paper.
- Keep econometric diagnostics in the main text only when they directly support the LLM reliability claim.
- Move detailed/statistician-facing diagnostics to the appendix.
- Do not delete rigor; relocate it.

## Main-Text Results to Keep

The Results section should foreground:

- Rubric validation at a high level: prompt-side composite increases with reference complexity.
- Low-end discrimination: rubric separates simple prompts better than the old keyword pilot.
- OLS vs IV headline: output-side complexity measurement understates the complexity penalty.
- Kink location: composite threshold around 8, with below/above pass-rate drop.
- Per-model replication summary: kink detected in most models; all models show declining pass rate with prompt complexity.
- Placebo/robustness summary in one paragraph, not a long diagnostic treatment.
- One or two figures only if they naturally support the narrative.

## Move to Appendix

Likely appendix material:

- Full first-stage coefficient table.
- Full first-stage instrument-strength discussion beyond the headline F/partial R2.
- Detailed Hansen/Sargan J discussion and large-sample caveat.
- Full per-model threshold table.
- Per-model heatmap if it slows the main narrative.
- Sparse high-complexity tail caveat and diagnostic details.
- Pilot keyword-instrument comparison table if not essential to the main claim.
- Detailed placebo distribution summary.
- Artifact, license, compute, safeguards, and LLM-usage details.
- NeurIPS checklist after all appendix material.

## Required Compliance Checks After Overwrite

Run these checks after receiving the final advisor-edited TeX:

- `\usepackage[eandd]{neurips_2026}` is active.
- No `[final]`, `[preprint]`, or `nonanonymous` option.
- No author names, advisor name, UWM, acknowledgments, or other identifying text in submission source.
- Abstract is one paragraph and under the current character limit.
- `\label{eq:structural}` exists if `\ref{eq:structural}` is used.
- Every `\ref{...}` has a matching `\label{...}`.
- Every `\cite{...}` has a matching `\bibitem{...}` or BibTeX entry.
- `linearmodels` is cited if `IV2SLS`/`linearmodels` is mentioned.
- Checklist is present after references and appendix.
- No checklist `TODO`, `answerTODO`, or placeholder text remains.
- Supplementary artifact and Croissant metadata are prepared for OpenReview.

## Hard Stop

Any magic number or hardcoded threshold in code, analysis scripts, tables, or manuscript text must be derivable, cited, or explained. If it cannot be explained, remove it before final submission.
