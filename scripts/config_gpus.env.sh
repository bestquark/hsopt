# NOTE: DO NOT use set -euo pipefail in a file you source into your shell.

# ---- Tunables ----
: "${MIN_SM:=80}"        # e.g., 80 -> sm_80, 86 -> sm_86
: "${MIN_MEM_GB:=12}"    # minimum GPU memory in GiB
: "${K:=}"               # optional: take first K GPUs

# Query GPUs safely (don't crash if nvidia-smi errs)
GPU_LINES=$(nvidia-smi --query-gpu=index,compute_cap,memory.total \
                       --format=csv,noheader,nounits 2>/dev/null || true)

if [[ -z "$GPU_LINES" ]]; then
  echo "[warn] nvidia-smi returned nothing; leaving CUDA_VISIBLE_DEVICES unchanged."
  return 0
fi

good=()
while IFS= read -r g; do
  # "0, 8.6, 49140"
  IFS=',' read -r idx cc mem_mib <<<"$g"
  idx="$(echo "$idx" | xargs)"
  cc_digits="$(echo "$cc" | tr -d ' .')"   # "8.6" -> "86"
  mem_gb=$(( $(echo "$mem_mib" | xargs) / 1024 ))
  if [[ "$cc_digits" -ge "$MIN_SM" && "$mem_gb" -ge "$MIN_MEM_GB" ]]; then
    good+=("$idx")
  fi
done <<< "$GPU_LINES"

if ((${#good[@]} == 0)); then
  echo "[warn] no GPUs meet criteria (MIN_SM=$MIN_SM, MIN_MEM_GB=${MIN_MEM_GB}GiB)."
  echo "Detected lines:"
  printf '  %s\n' $GPU_LINES
  return 0
fi

# Take first K if requested
if [[ -n "$K" ]]; then
  good=("${good[@]:0:$K}")
fi

export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${good[*]}")"
export OE_PROFILE_GPU=1
export N_GPUS="${#good[@]}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (MIN_SM=$MIN_SM, MIN_MEM_GB=$MIN_MEM_GB, K=${K:-all})"
echo "Selected GPUs: ${good[*]}"
echo "OE_PROFILE_GPU=$OE_PROFILE_GPU"
echo "N_GPUS=$N_GPUS"