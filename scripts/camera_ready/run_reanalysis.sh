#!/usr/bin/env bash
# Regenerate every outcome-dependent result of the camera-ready for one outcome
# definition, from the repository root on a Linux host with the data at ./data.
#
#   scripts/camera_ready/run_reanalysis.sh <label> <scored-dir>
#
#   independent_audit  data/stage_d/scored_independent_audit   (primary outcome)
#   reviewed_outcome   data/stage_d/scored_combined            (reviewed version)
#   harness_outcome    data/stage_d/scored_harness             (raw unit tests)
#
# Outputs go to results/camera_ready/<label>/. The extension comparison always
# uses raw harness outcomes on both sides, because the extension runs did not
# save generated code and cannot be audited.
set -euo pipefail
LABEL="$1"
SCORED="$2"
OUT="results/camera_ready/$LABEL"
PY="${PY:-.venv/bin/python}"
RUBRIC=data/stage_d/ensemble_scores_current_aggregated.jsonl
PROMPTS=data/stage_d/stage_d_prompts.jsonl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/per_model" "$OUT/logs"
export CK_SCORED_DIR="$SCORED"

echo "[$(date -u +%T)] breakpoints, pooling, task type"
$PY scripts/camera_ready/breakpoints.py --out "$OUT/breakpoints.json" > "$OUT/logs/breakpoints.log" 2>&1

echo "[$(date -u +%T)] construction-frame sensitivity and downward-model frame audits"
$PY scripts/analyze_source_frame_sensitivity.py --data-root data --output "$OUT/source_frame_sensitivity.json" \
    --n-boot 300 --seed 42 > "$OUT/logs/source_frame.log" 2>&1 &
$PY scripts/analyze_model_specific_source_frame.py --pooling-report "$OUT/breakpoints.json" \
    --output "$OUT/model_specific_source_frame.json" > "$OUT/logs/model_specific.log" 2>&1 &

echo "[$(date -u +%T)] IV and output-complexity diagnostics"
$PY scripts/camera_ready/iv_diagnostics.py --scored-dir "$SCORED" --out "$OUT/iv_diagnostics.json" \
    > "$OUT/logs/iv.log" 2>&1 &
$PY scripts/camera_ready/output_cc_diagnostics.py --scored-dir "$SCORED" --out-dir "$OUT" \
    --prompt-threshold "$($PY -c "import json; print(json.load(open('$OUT/breakpoints.json'))['pool_mean']['threshold'])")" \
    > "$OUT/logs/output_cc.log" 2>&1 &

echo "[$(date -u +%T)] headline combined fit (2,000 wild, 1,000 pairs, 2,000 placebo)"
$PY src/analyze_kink.py --scored-dir "$SCORED" --rubric "$RUBRIC" --prompts "$PROMPTS" \
    --outdir "$OUT/combined" --combined-only --skip-visualizations \
    --n-boot 2000 --n-ci-boot 1000 --n-placebo 2000 > "$OUT/logs/combined.log" 2>&1 &

echo "[$(date -u +%T)] per-model fits (500 wild, 1,000 pairs, 500 placebo)"
for f in "$SCORED"/*.jsonl; do
  m="$(basename "$f" .jsonl)"
  case "$m" in _*) continue;; esac
  $PY src/analyze_kink.py --scored-dir "$SCORED" --rubric "$RUBRIC" --prompts "$PROMPTS" \
      --outdir "$OUT/per_model/$m" --min-rows 5000 --include-model "$m" --skip-combined \
      --skip-visualizations --n-boot 500 --n-ci-boot 1000 --n-placebo 500 \
      > "$OUT/logs/per_model_$m.log" 2>&1 &
done

echo "[$(date -u +%T)] extension tables (raw harness on both sides)"
CK_SCORED_DIR=data/stage_d/scored_harness $PY scripts/regenerate_tail_extension_tables.py \
    --data-root . --output-dir "$OUT" > "$OUT/logs/tail.log" 2>&1 &

wait
echo "[$(date -u +%T)] assemble summaries"
$PY scripts/camera_ready/assemble_results.py --results-dir "$OUT"
echo "[$(date -u +%T)] done: $OUT"
