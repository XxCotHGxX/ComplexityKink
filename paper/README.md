# Manuscript snapshot

`Scratch-NeurIps.tex` and its four PNG figures preserve the anonymous Stage D
manuscript snapshot used for submission. They are included so the repository
state can be compared with the locked analysis.

The post-submission checks in `docs/robustness_results.md` do not silently
rewrite this source. In particular, the submitted manuscript still contains
language that the review-period evidence narrows:

- The full overidentification rejection is not a harmless large-sample effect.
- Task-type controls explain a substantial part of the pooled regime gap.
- Human calibration is now complete.
- The clean high-complexity extension supports the pattern through bin 16, but
  leaves the extreme tail unresolved.
- The earlier library/framework explanation is not supported by its scoped
  diagnostic.

Those points belong in a revised or camera-ready manuscript after the review
process. Keeping them in a separate results note preserves a clear record of
what was submitted and what was learned afterward.

The old Stage C manuscript source was removed from the current tree because it
reported a different prompt set and threshold. It remains available in Git
history.
