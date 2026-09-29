# Independent outcome audit: protocol

Written 2026-09-28, before any pilot results were observed.

## Why

The reviewed manuscript's pass rate for the 2,246 earlier-frame prompts used
an o4-mini audit (`src/data_provenance/06_audit_scoring.py`,
`07_apply_judge.py`) that replaced the unit-test harness verdict with 1.0 or
0.0 on 23,848 of 47,166 rows. The 2,754 later-frame prompts were never
audited, and o4-mini is also one of the four rubric judges. The camera-ready
therefore re-audits all 105,000 generations with a single auditor that is not
a rubric judge, not an evaluated model, and not from a model family used
elsewhere in the study.

## Candidate auditors

Only models already deployed in the project's Azure accounts were considered.
The two from families outside the evaluated panel and the rubric judges are:

| Auditor | Azure account | Status |
|---|---|---|
| `Phi-4-reasoning` (Microsoft) | DataPipeline0 | GA, open weights (MIT) |
| `MAI-Thinking-1` (Microsoft, 2026-06-01) | herna-mn98z78z-eastus2 | Preview; inference retires 2026-11-04 |

## Audit prompt

Both auditors receive the original audit system prompt verbatim (copied from
`06_audit_scoring.py`), the same user-message template (task, code, unit
tests, harness pass rate), `max_completion_tokens=16000` (see amendment), and provider-default
sampling. A single appended line reminds the model to answer with the JSON
object only; it is identical for both auditors. Responses are parsed by the
same tolerant parser (strip reasoning blocks, take the last JSON object with a
`verdict` key). Unparseable responses and API errors are recorded as
`parse_error` and, in the full run, fall back to the harness verdict exactly
like `uncertain`.

**Amendment before the pilot (2026-09-28, after a 4-case smoke test and
before any pilot case was scored).** The appended reminder originally read
"Respond with ONLY the JSON object described above." In the smoke test,
Phi-4-reasoning read this as a request to restate the inputs as JSON and
never gave a verdict (4/4 unparseable; one response hit the 8,000-token cap).
The reminder now states the exact verdict schema and says not to repeat the
inputs, and `max_completion_tokens` is 16,000. Both changes apply identically
to both auditors, and smoke outputs are excluded from all pilot metrics.

**Transport (operational, no effect on prompts or scoring).** First-party
Azure model endpoints close blocking (non-streaming) requests at roughly 680 s
and omit reasoning tokens from the usage report. The pilot used blocking
requests: one Phi-4-reasoning case exhausted its retries and eight more stalled
at the cut-off; those nine were re-requested with a longer client timeout, and
all 450 cases of both auditors completed. Production runs stream responses,
which keeps the connection open and reports reasoning tokens; the model
computes the same thing.

## Known-answer pilot set

Built by `src/audit/build_known_answer_set.py` from benchmark prompts whose
reference solution passes every unit test when run through the unmodified
harness (`score_solution` in `03_execute_and_score.py`) inside the `scorer`
container with no network. All code, including the reference, is normalized
through `ast.unparse` so that formatting does not reveal the category.

| Category | Construction | Ground truth |
|---|---|---|
| `clean` | normalized reference solution; harness pass rate 1.0 | correct |
| `cosmetic` | top-level functions/classes called by the tests renamed; harness pass rate < 1.0 | correct |
| `bug` | one AST mutation (comparison, arithmetic, boolean, integer-constant, or negated condition) that the tests detect; the most subtle detected mutant (highest harness pass rate below 1.0) is kept | incorrect |

150 prompts, stratified across prompt-composite display bins, contribute one
case to each category (450 cases). Seed: 20260928.

## Selection rule (fixed before the pilot)

An auditor is **eligible** if, on the pilot set:

1. parse/API error rate ≤ 2% of cases;
2. accuracy on `clean` cases (verdict `correct`) ≥ 95%;
3. rescue rate on `cosmetic` cases (verdict `correct`) ≥ 90%.

Among eligible auditors, the **primary auditor** is the one with the lowest
**wrong-rescue rate** on `bug` cases (verdict `correct` on incorrect code).
Ties within 1 percentage point go to the auditor with higher overall
accuracy, then to the GA/open-weights model. If no auditor is eligible, no
auditor is adopted and the pilot results are reported to the authors before
any further step.

The other auditor re-audits a random 5% of the 105,000 production rows
(seed 20260928) to report inter-auditor agreement (Cohen's kappa).

## Amendment 2: extended candidate pool (2026-09-29, after the first pilot)

**First pilot result.** Neither Azure candidate met the rule: Phi-4-reasoning
had clean accuracy 0.900 and cosmetic rescue 0.473; MAI-Thinking-1 had clean
accuracy 0.953 but cosmetic rescue 0.880 (threshold 0.90). Per the rule, no
auditor was adopted and the result went to the authors, who chose to extend the
candidate pool. The thresholds, prompt, parser, and known-answer set are
unchanged.

**Added candidates (OpenRouter).** Models from vendors not used anywhere in the
study (evaluated panel, rubric judges, paraphraser), excluding anonymous
"stealth" models, routers, safety/domain-specialized models, and models whose
card names a base model from an excluded family:

| Candidate | Vendor | Note |
|---|---|---|
| `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA | NVIDIA also released OpenCodeInstruct (task-exposure caveat) |
| `poolside/laguna-s-2.1` | Poolside | coding specialist |
| `thinkingmachines/inkling` | Thinking Machines | |
| `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | same caveat |
| `thinkingmachines/inkling-small` | Thinking Machines | |

Pilot runs may use the `:free` endpoints (daily request caps), in the order
listed. The reviewed version's auditor, o4-mini, is also run on the
known-answer set as a diagnostic of the reviewed outcome; it is not a
candidate (it is a rubric judge).

**Confirmation requirement.** Because this pool was chosen after the first
pilot, an auditor selected under the rule on the original 450 cases is adopted
only if it also meets the rule on a fresh known-answer set (seed 20260930,
prompts disjoint from the first set), run at the exact endpoint and provider
used for the production run.

## Amendment 3: MiMo-V2.6-Pro (2026-09-29, before its pilot)

**Amendment 2 result.** On the original 450 cases, Nemotron-3-Ultra (Venice,
fp8; clean 0.860, cosmetic 0.787, bug wrong-rescue 0.000) and Laguna-S-2.1
(Poolside, fp4; error rate 0.047, clean 0.887, cosmetic 0.747) did not meet the
rule. The reviewed o4-mini audit, run as a diagnostic with its original prompt
and settings, scored clean 0.880, cosmetic 0.820, bug wrong-rescue 0.000.

**Added candidate.** `xiaomi/mimo-v2.6-pro` (Xiaomi, released 2026-09-21;
vendor not used elsewhere in the study), pinned to the GMICloud bf16 endpoint.
Rule, prompt, parser, known-answer set, and the fresh-set confirmation
requirement are unchanged.

**Observation recorded before this pilot.** Several "clean" references pass
their unit tests yet violate an explicit instruction in the task (for example,
row-major instead of the required diagonal-major traversal), so the clean and
cosmetic labels are noisy for some prompts. Any label correction must come from
blind human review of the disputed prompts, applied identically to every
auditor; it is not made from auditor verdicts.

## Amendment 4: MiMo-V2.6-Pro adopted by author decision (2026-09-29)

**Amendment 3 result.** MiMo-V2.6-Pro (GMICloud, bf16): error rate 0.002,
clean 0.933, cosmetic 0.947, bug wrong-rescue 0.013, overall 0.956. That is the
highest overall accuracy of the six auditors tested and meets every criterion
except clean accuracy, where it falls three cases short (140 of 150 against the
required 143).

**Decision.** The authors adopt MiMo-V2.6-Pro as the primary auditor. This is a
departure from the rule, which no candidate met in full. The reasons: its clean
rejections name specific defects in reference solutions that the unit tests do
not catch, and at least one other auditor independently rejects 7 of its 9;
35 references are rejected by at least one of the six auditors, so the clean
labels are noisy and cap attainable clean accuracy; and, as a post hoc
sensitivity only, excluding the 11 references rejected by both o4-mini and
Nemotron gives MiMo clean 0.971 and cosmetic 0.986.

**Still to report.** The fresh-set confirmation (seed 20260930) runs at the
production endpoint and is reported whether or not it meets the thresholds. A
blind human review of the 35 disputed references may follow; if run, it is
reported whichever way it goes.

**Production.** All 105,000 generations are audited by
`xiaomi/mimo-v2.6-pro` through OpenRouter, pinned to GMICloud (bf16), with
streamed responses and the same prompt and parser as the pilot.
MAI-Thinking-1's audit of the seeded 5% sample is the second auditor for
inter-auditor agreement.

**Confirmation result (2026-09-29).** On the fresh known-answer set (150
prompts disjoint from the first set, seed 20260930), run at the production
endpoint (OpenRouter, GMICloud bf16), MiMo-V2.6-Pro met every threshold: error
rate 0.000, clean 0.953, cosmetic 0.973, bug wrong-rescue 0.007 (1 of 150),
overall 0.973. It therefore satisfies the amendment-2 confirmation requirement;
the selection-set shortfall on clean accuracy remains disclosed.

## Reporting

The camera-ready reports each auditor's pilot metrics, the selection, the
inter-auditor agreement, and the raw-harness outcome as a sensitivity
analysis. The known-answer set measures the audit on synthetic cosmetic and
injected-bug cases, which may be easier than real model failures; this is
stated as a limitation.
