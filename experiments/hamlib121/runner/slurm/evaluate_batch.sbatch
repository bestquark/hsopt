#!/bin/bash
#SBATCH --job-name=hamlib_pf_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=229G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?Set CAMPAIGN_ROOT}"
ITERATION="${ITERATION:?Set ITERATION}"
HAMLIB_DATA_ROOT="${HAMLIB_DATA_ROOT:?Set HAMLIB_DATA_ROOT}"
LANE_IDS_FILE="${LANE_IDS_FILE:?Set LANE_IDS_FILE}"
BATCH_ID="${BATCH_ID:?Set BATCH_ID}"
COMPUTE_BASELINE="${COMPUTE_BASELINE:-1}"
POLICY_SNAPSHOT="${POLICY_SNAPSHOT:-current}"

cd "$CAMPAIGN_ROOT"
mkdir -p logs "iterations/iter_$(printf "%04d" "$ITERATION")/results" baselines

export PATH="$HOME/.local/bin:$PATH"
if [[ -n "${CAMPAIGN_UV_BASE_DIR:-}" ]]; then
  export XDG_CACHE_HOME="${CAMPAIGN_UV_BASE_DIR}/.cache"
  export XDG_DATA_HOME="${CAMPAIGN_UV_BASE_DIR}/.local/share"
fi
export UV_CACHE_DIR="${CAMPAIGN_UV_CACHE_DIR:-${XDG_CACHE_HOME:-${LUSTRE:-$HOME}/.cache}/uv}"
export UV_PYTHON_INSTALL_DIR="${CAMPAIGN_UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME:-${LUSTRE:-$HOME}/.local/share}/uv/python}"
export UV_PROJECT_ENVIRONMENT=.venv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CAMPAIGN_ARRAY_BACKEND=cupynumeric
export CAMPAIGN_HDF5_SUBPROCESS=1
export CAMPAIGN_LEGATE_FBMEM="${CAMPAIGN_LEGATE_FBMEM:-70000}"
export CAMPAIGN_SYNTH_WORKERS="${CAMPAIGN_SYNTH_WORKERS:-30}"

uv sync --locked --extra gpu

export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
SITE_PACKAGES="$("$VIRTUAL_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
NVIDIA_LIB_PATHS="$("$VIRTUAL_ENV/bin/python" - <<'PY'
from pathlib import Path
import site

paths = []
for site_dir in site.getsitepackages():
    root = Path(site_dir)
    paths.extend(str(path) for path in sorted((root / "nvidia").glob("*/lib")) if path.is_dir())
    paths.extend(str(path) for pattern in ("cutensor*/lib", "libucx*/lib") for path in sorted(root.glob(pattern)) if path.is_dir())
print(":".join(paths))
PY
)"
if [[ -n "$NVIDIA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

LEGATE=.venv/bin/legate
if [[ ! -x "$LEGATE" ]]; then
  echo "missing $LEGATE after uv sync --extra gpu" >&2
  exit 2
fi
BASELINE_FLAG=()
if [[ "$COMPUTE_BASELINE" == "1" ]]; then
  BASELINE_FLAG=(--compute-baseline)
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
echo "legate=$("$LEGATE" --version 2>&1 | head -n 1 || true)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "lane_count=$(grep -cve '^[[:space:]]*$' "$LANE_IDS_FILE")"

nvidia-smi dmon -s pucm -d 5 -o DT > "logs/gpu_dmon_iter_${ITERATION}_batch_${BATCH_ID}.log" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

echo "worker=0 num_workers=1 cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
"$LEGATE" --gpus 1 --fbmem "$CAMPAIGN_LEGATE_FBMEM" scripts/evaluate_worker.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --iteration "$ITERATION" \
  --worker-id 0 \
  --num-workers 1 \
  --data-root "$HAMLIB_DATA_ROOT" \
  --lane-ids-file "$LANE_IDS_FILE" \
  --policy-snapshot "$POLICY_SNAPSHOT" \
  "${BASELINE_FLAG[@]}" \
  > "logs/eval_iter_${ITERATION}_batch_${BATCH_ID}.out" \
  2> "logs/eval_iter_${ITERATION}_batch_${BATCH_ID}.err"
echo "finished=$(date -Is)"
