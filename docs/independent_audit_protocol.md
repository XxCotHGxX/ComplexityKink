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

## Reporting

The camera-ready reports each auditor's pilot metrics, the selection, the
inter-auditor agreement, and the raw-harness outcome as a sensitivity
analysis. The known-answer set measures the audit on synthetic cosmetic and
injected-bug cases, which may be easier than real model failures; this is
stated as a limitation.
