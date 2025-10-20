#!/usr/bin/env bash
set -euo pipefail

# ---- Tunables (can override via environment) ----
# minimum compute capability, e.g. 80 => sm_80, 86 => sm_86
MIN_SM="${MIN_SM:-80}"
# minimum total memory (GiB) to accept a GPU
MIN_MEM_GB="${MIN_MEM_GB:-12}"

# Optional: restrict to first K matching GPUs (e.g., K=1 to pick a single device)
K="${K:-}"

# Query GPUs: index, compute_cap, total_mem(MiB)
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,compute_cap,memory.total \
                             --format=csv,noheader,nounits)

good=()
for g in "${GPUS[@]}"; do
  # line looks like: "0, 8.6, 49140"
  IFS=',' read -r idx cc mem_mib <<<"$g"
  idx="$(echo "$idx" | xargs)"
  cc_digits="$(echo "$cc" | tr -d ' .')"   # "8.6" -> "86"
  mem_gb=$(( $(echo "$mem_mib" | xargs) / 1024 ))

  if [[ "$cc_digits" -ge "$MIN_SM" && "$mem_gb" -ge "$MIN_MEM_GB" ]]; then
    good+=("$idx")
  fi
done

if ((${#good[@]} == 0)); then
  echo "No GPUs meet criteria (MIN_SM=$MIN_SM, MIN_MEM_GB=${MIN_MEM_GB}GiB)."
  echo "Detected:"
  printf '  %s\n' "${GPUS[@]}"
  exit 1
fi

# If K is set, take only the first K
if [[ -n "${K}" ]]; then
  good=("${good[@]:0:${K}}")
fi

export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${good[*]}")"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (MIN_SM=${MIN_SM}, MIN_MEM_GB=${MIN_MEM_GB}, K=${K:-all})"

# Exec the user command under this mask
exec "$@"