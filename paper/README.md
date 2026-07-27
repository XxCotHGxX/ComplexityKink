# Revised manuscript source

`Scratch-NeurIps.tex` is the current anonymous manuscript revision. It
incorporates the additional analyses directly:

- The rubric composite is treated as a prompt-side index, not a validated
  causal instrument.
- Construction-frame and task-type controls, pooling sensitivity, human
  calibration, repeated sampling, paraphrase stability, and language rescoring
  are reported.
- The 365-prompt audit-clean extension is shown in a matched five-model frame.
- The generated-output CC figure states its complete-case denominator.
- A 459-prompt diagnostic provides no support for a library/framework
  explanation.

The paper directory contains six referenced PNG figures. The original submitted
source is not duplicated in this reviewer-facing snapshot. Aggregate numerical
inputs for the two new figures are under `results/`, and the plotting code is
in `scripts/generate_stage_d_paper_figures.py`. Rebuild the revised pipeline,
tail-extension, and output-CC figures with:

```powershell
python scripts/generate_stage_d_paper_figures.py --revision-only
```

The tail source tables use a matched five-model frame and first average pass
rate within prompt. `tail_extension_source_split.csv` covers displayed bins 9
through 18, not the full benchmark; `tail_extension_replication.csv` records
the bin-15 and bin-16 Welch comparisons, and
`tail_extension_fixed_version.csv` records the fixed-version three-model check.
Display bin $b$ contains composite values in $[b-0.5,b+0.5)$, so exact
half-point boundaries enter the higher bin. Threshold inference remains on the
continuous, unbinned composite. In `pass_vs_output_cc.csv`, bin 40 is top-coded
as 40 or greater. `reverse_threshold_zero_pass_cells.csv` contains 14,776
complete cases and excludes 977 zero-pass rows without computable output CC, as
stated in the manuscript.
