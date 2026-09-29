#!/bin/sh
# Entry point for the ckr-audit Container Apps Job.
# Required env: AUDIT_DEPLOYMENT, AUDIT_INPUT, AUDIT_OUTPUT, AUDIT_ENDPOINT, AUDIT_API_KEY.
set -eu
PROD="$DATA_ROOT/independent_audit/production"
if [ ! -f "$PROD/audit_input.jsonl" ] || [ ! -f "$PROD/audit_input_secondary_5pct.jsonl" ]; then
  python /app/src/audit/05_build_production_audit_input.py --data-root "$DATA_ROOT"
fi
exec python /app/src/audit/independent_audit.py \
  --input "$PROD/$AUDIT_INPUT" \
  --output "$PROD/$AUDIT_OUTPUT" \
  --deployment "$AUDIT_DEPLOYMENT" \
  --workers "${AUDIT_WORKERS:-64}"
