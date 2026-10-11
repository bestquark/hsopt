# Rotation synthesis

HSOpt uses `cudaq.synth.gridsynth`, CUDA-Q's CPU implementation of Ross--Selinger,
with the policy's requested operator-norm epsilon, seed 0, fixed factoring/search
budgets, and no wall-clock synthesis timeout. Linux dependencies pin
`cuda-quantum-cu12==0.16.0.post1`. Dense unitary operations use the separately
selected NumPy or cuPyNumeric backend.

The literal gate string is reconstructed in matrix-multiplication order,
including scalar W phase gates. This preserves the phase needed for controlled
evolution and energy decoding. CUDA-Q's arbitrary-precision `rz_error` verifies
each unique synthesized rotation against its requested epsilon; the matrix is
then represented in complex128 for the dense evaluator.

## Apple silicon

The release's PyPI wheels are Linux wheels. Install the official 0.16.0 CPU
distribution outside the checkout; its Python bindings use Python 3.12.
After cloning HSOpt, run from the repository root:

```bash
uv sync --locked --python 3.12
HSOPT_CUDAQ_DIR="$(pwd)/../runtime/vendor/cudaq-0.16.0"
HSOPT_CUDAQ_TMP="$(mktemp -d)"
curl -fsSL https://github.com/NVIDIA/cuda-quantum/releases/download/0.16.0/install_cuda_quantum_darwin-0.16.0.arm64 -o "$HSOPT_CUDAQ_TMP/cudaq.run"
mkdir -p "$HSOPT_CUDAQ_DIR"
(
  cd "$HSOPT_CUDAQ_DIR"
  bash "$HSOPT_CUDAQ_TMP/cudaq.run" --tar xf ./cudaq/cudaq ./cudaq/lib ./cudaq/targets ./cudaq/LICENSE ./cudaq/LICENSES ./cudaq/NOTICE
)
.venv/bin/python - "$HSOPT_CUDAQ_DIR/cudaq" <<'PY'
import sys
import site
from pathlib import Path
sdk = Path(sys.argv[1]).resolve()
(Path(site.getsitepackages()[0]) / "cudaq-sdk.pth").write_text(str(sdk) + "\n")
PY
rm "$HSOPT_CUDAQ_TMP/cudaq.run"
rmdir "$HSOPT_CUDAQ_TMP"
uv run python -c 'from cudaq import synth; print(synth.gridsynth(0.2, 1e-5, seed=0))'
```

This extracts the upstream Python/runtime assets and their licenses without
running the installer or changing shell profiles. It installs the dependency
for rotation synthesis, rather than the full CUDA-Q development toolchain.

## Result provenance and migration

CUDA-Q results use `normalized_hamiltonian_qpe_t_count_cudaq_v3` and record
`rotation_synthesis` with the compiler build/version, algorithm, seed, search
budgets, epsilon convention and phase handling. Candidate queues and mutation
lanes reject an older scoring contract.

Exact Hamiltonian references from `normalized_hamiltonian_qpe_t_count_v2`
remain valid when their Hamiltonian/reaction hashes match: those references
contain no approximate rotation synthesis. Changing chemistry inputs still
requires a new reference.

Keep historical results immutable. Rescore baselines and archived candidates in
a new campaign root, then recompute acceptance against scores from the same
compiler. A replay of fixed proposals is a rescoring study; it is not an agent
trajectory generated with CUDA-Q feedback. If an acceptance decision changes,
a genuine CUDA-Q-driven continuation must restart from the last common
incumbent, with every compared score recomputed under CUDA-Q.

Generated initializer documentation now names CUDA-Q. Their PF schedules,
repetition counts and requested epsilons are unchanged by this migration;
external frozen historical policies retain their original bytes.

[NVIDIA synthesis documentation](https://nvidia.github.io/cuda-quantum/latest/using/examples/rotation_synthesis.html)
describes the upstream algorithm, error norm and global-phase conventions.

