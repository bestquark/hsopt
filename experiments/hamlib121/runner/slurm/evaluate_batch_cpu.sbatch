#!/bin/bash
#SBATCH --job-name=hamlib_pf_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?Set CAMPAIGN_ROOT}"
ITERATION="${ITERATION:?Set ITERATION}"
HAMLIB_DATA_ROOT="${HAMLIB_DATA_ROOT:?Set HAMLIB_DATA_ROOT}"
LANE_IDS_FILE="${LANE_IDS_FILE:?Set LANE_IDS_FILE}"
BATCH_ID="${BATCH_ID:?Set BATCH_ID}"
COMPUTE_BASELINE="${COMPUTE_BASELINE:-0}"
POLICY_SNAPSHOT="${POLICY_SNAPSHOT:-current}"

if [[ "$COMPUTE_BASELINE" == "1" ]]; then
  echo "CPU policy job cannot compute missing baselines; submit a GPU baseline batch instead" >&2
  exit 2
fi

cd "$CAMPAIGN_ROOT"
mkdir -p logs "iterations/iter_$(printf "%04d" "$ITERATION")/results" baselines

if [[ -n "${CAMPAIGN_UV_BASE_DIR:-}" ]]; then
  export XDG_CACHE_HOME="${CAMPAIGN_UV_BASE_DIR}/.cache"
  export XDG_DATA_HOME="${CAMPAIGN_UV_BASE_DIR}/.local/share"
fi
export UV_CACHE_DIR="${CAMPAIGN_UV_CACHE_DIR:-${XDG_CACHE_HOME:-${LUSTRE:-$HOME}/.cache}/uv}"
export UV_PYTHON_INSTALL_DIR="${CAMPAIGN_UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME:-${LUSTRE:-$HOME}/.local/share}/uv/python}"
export UV_PROJECT_ENVIRONMENT=.venv_cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CAMPAIGN_ARRAY_BACKEND=numpy
export CAMPAIGN_HDF5_SUBPROCESS=1
export CAMPAIGN_SYNTH_WORKERS="${CAMPAIGN_SYNTH_WORKERS:-30}"

export VIRTUAL_ENV="$PWD/.venv_cpu"
unset PYTHONHOME PYTHONPATH
SHA256SUM=/usr/bin/sha256sum
if [[ ! -x "$SHA256SUM" ]]; then
  echo "missing trusted $SHA256SUM" >&2
  exit 2
fi
EXPECTED_LOCK_LINE="$("$SHA256SUM" uv.lock)"
EXPECTED_LOCK_SHA256="${EXPECTED_LOCK_LINE%% *}"
RUNNER_LINE="$("$SHA256SUM" manifests/runner_provenance.json)"
EXPECTED_RUNNER_SHA256="${RUNNER_LINE%% *}"
IDENTITY_LINE="$("$SHA256SUM" manifests/campaign_identity.json)"
EXPECTED_IDENTITY_SHA256="${IDENTITY_LINE%% *}"
if [[ ! -x "$VIRTUAL_ENV/bin/python" \
      || ! -f "$VIRTUAL_ENV/.hsopt-uv-lock-sha256" \
      || ! -f "$VIRTUAL_ENV/.hsopt-runner-provenance-sha256" \
      || ! -f "$VIRTUAL_ENV/.hsopt-campaign-identity-sha256" ]] || \
   [[ "$(<"$VIRTUAL_ENV/.hsopt-uv-lock-sha256")" != "$EXPECTED_LOCK_SHA256" \
      || "$(<"$VIRTUAL_ENV/.hsopt-runner-provenance-sha256")" != "$EXPECTED_RUNNER_SHA256" \
      || "$(<"$VIRTUAL_ENV/.hsopt-campaign-identity-sha256")" != "$EXPECTED_IDENTITY_SHA256" ]]; then
  echo "missing/stale prebuilt locked CPU environment; run scripts/prebuild_cluster_environments.py" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "iteration=$ITERATION"
echo "batch_id=$BATCH_ID"
echo "campaign_root=$CAMPAIGN_ROOT"
echo "hamlib_data_root=$HAMLIB_DATA_ROOT"
echo "lane_ids_file=$LANE_IDS_FILE"
echo "compute_baseline=$COMPUTE_BASELINE"
echo "policy_snapshot=$POLICY_SNAPSHOT"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "started=$(date -Is)"
echo "campaign_array_backend=$CAMPAIGN_ARRAY_BACKEND"
echo "lane_count=$(grep -cve '^[[:space:]]*$' "$LANE_IDS_FILE")"

"$VIRTUAL_ENV/bin/python" scripts/evaluate_worker.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --iteration "$ITERATION" \
  --worker-id 0 \
  --num-workers 1 \
  --data-root "$HAMLIB_DATA_ROOT" \
  --lane-ids-file "$LANE_IDS_FILE" \
  --policy-snapshot "$POLICY_SNAPSHOT" \
  > "logs/eval_iter_${ITERATION}_batch_${BATCH_ID}.out" \
  2> "logs/eval_iter_${ITERATION}_batch_${BATCH_ID}.err"
echo "finished=$(date -Is)"
