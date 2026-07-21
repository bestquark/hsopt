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
ITERATION_PADDED="$(printf '%04d' "$ITERATION")"
mkdir -p "logs" "iterations/iter_${ITERATION_PADDED}/results"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CAMPAIGN_ARRAY_BACKEND=cupynumeric
export CAMPAIGN_LEGATE_FBMEM="${CAMPAIGN_LEGATE_FBMEM:-70000}"

VIRTUAL_ENV="$(dirname "$(dirname "$CAMPAIGN_PYTHON")")"
LEGATE="$(dirname "$CAMPAIGN_PYTHON")/legate"
test -x "$LEGATE"
SHA256SUM=/usr/bin/sha256sum
test -x "$SHA256SUM"
EXPECTED_LOCK_SHA256="$($SHA256SUM "${CAMPAIGN_ROOT}/runner/environment/uv.lock")"
EXPECTED_LOCK_SHA256="${EXPECTED_LOCK_SHA256%% *}"
EXPECTED_MANIFEST_SHA256="$($SHA256SUM "${CAMPAIGN_ROOT}/manifest.json")"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256%% *}"
if [[ ! -f "${VIRTUAL_ENV}/.hsopt-uv-lock-sha256" \
      || ! -f "${VIRTUAL_ENV}/.hsopt-manifest-sha256" \
      || "$(<"${VIRTUAL_ENV}/.hsopt-uv-lock-sha256")" != "$EXPECTED_LOCK_SHA256" \
      || "$(<"${VIRTUAL_ENV}/.hsopt-manifest-sha256")" != "$EXPECTED_MANIFEST_SHA256" ]]; then
  echo "missing/stale prebuilt reaction GPU environment; run prebuild_cluster_environment.py" >&2
  exit 2
fi
export VIRTUAL_ENV
SITE_PACKAGES="$("$CAMPAIGN_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="$SITE_PACKAGES"
NVIDIA_LIB_PATHS="$("$CAMPAIGN_PYTHON" - <<'PY'
from pathlib import Path
import site

paths = []
for site_dir in site.getsitepackages():
    root = Path(site_dir)
    paths.extend(str(path) for path in sorted((root / "nvidia").glob("*/lib")) if path.is_dir())
    paths.extend(
        str(path)
        for pattern in ("cutensor*/lib", "libucx*/lib")
        for path in sorted(root.glob(pattern))
        if path.is_dir()
    )
print(":".join(paths))
PY
)"
if [[ -n "$NVIDIA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

CAMPAIGN_NVIDIA_SMI=""
for candidate in /usr/bin/nvidia-smi /usr/local/nvidia/bin/nvidia-smi /usr/local/cuda/bin/nvidia-smi; do
  if [[ -x "$candidate" ]]; then
    resolved="$(/usr/bin/readlink -f "$candidate")"
    read -r owner mode < <(/usr/bin/stat -Lc '%u %a' "$resolved")
    if [[ "$owner" == "0" ]] && (( (8#$mode & 0022) == 0 )); then
      CAMPAIGN_NVIDIA_SMI="$resolved"
      break
    fi
  fi
done
if [[ -z "$CAMPAIGN_NVIDIA_SMI" ]]; then
  echo "no trusted NVIDIA system tool found" >&2
  exit 2
fi
export CAMPAIGN_NVIDIA_SMI

echo "host=$(hostname)"
echo "iteration=$ITERATION batch_id=$BATCH_ID lanes=$(grep -cve '^[[:space:]]*$' "$LANE_IDS_FILE")"
echo "slurm_job_id=${SLURM_JOB_ID:-} started=$(date -Is)"
"$CAMPAIGN_NVIDIA_SMI" --query-gpu=index,name,memory.total --format=csv,noheader

"$LEGATE" --gpus 1 --fbmem "$CAMPAIGN_LEGATE_FBMEM" --run-mode exec \
  "$CAMPAIGN_PYTHON" runner/evaluate_worker.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --iteration "${ITERATION}" \
  --lane-ids-file "${LANE_IDS_FILE}" \
  --workers "${EVALUATOR_WORKERS}"

echo "finished=$(date -Is)"
