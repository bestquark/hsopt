#!/bin/bash
set -euo pipefail

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${CAMPAIGN_PYTHON:?CAMPAIGN_PYTHON is required}"
: "${ITERATION:?ITERATION is required}"
: "${BATCH_ID:?BATCH_ID is required}"
: "${LANE_IDS_FILE:?LANE_IDS_FILE is required}"
: "${EVALUATOR_WORKERS:?EVALUATOR_WORKERS is required}"

test -d "${CAMPAIGN_ROOT}"
test -x "${CAMPAIGN_PYTHON}"
test -f "${LANE_IDS_FILE}"
test -f "${CAMPAIGN_ROOT}/runner/evaluate_worker.py"

cd "${CAMPAIGN_ROOT}"
exec "${CAMPAIGN_PYTHON}" runner/evaluate_worker.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --iteration "${ITERATION}" \
  --lane-ids-file "${LANE_IDS_FILE}" \
  --workers "${EVALUATOR_WORKERS}"
