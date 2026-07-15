# Product-Formula Benchmarks

`examples/pfs/` is the canonical home for the product-formula
Hamiltonian-simulation benchmarks in hsopt.

- `h2/`, `bh/`, `lih/`, `dmds/`: runnable baseline molecule folders.
- `n2/`, `co/`, `nh3/`, `hcn/`: histogram-derived larger molecule folders
  seeded for autoresearch with CAS(6e,5o) active spaces.
- `evaluate_energy.py`: T count with a chemical-accuracy constraint on the
  lowest resolved HF-reference spectral peak in the fixed-electron sector.
- `evaluate_fidelity.py`: T count with a final-state infidelity constraint
  against exact `exp(+i H t)`.
- `chemistry.py`: fixed molecule registry, cached Hamiltonian data, and shared
  product-formula compilation helpers.
- `array_backend.py`: opt-in NumPy/cuPyNumeric array backend selection for
  state propagation and evaluator diagnostics.
- `reactions/`: the fixed full-Hilbert-space compiled-unitary evaluator used
  by the ten-reaction paper experiment; its exact registry and runner live in
  `experiments/reactions10/`.

Run either evaluator directly:

```bash
uv run python examples/pfs/evaluate_energy.py --molecule h2 --script examples/pfs/h2/simple_pf.py
uv run python examples/pfs/evaluate_fidelity.py --molecule h2 --script examples/pfs/h2/simple_pf.py --epsilon 1e-3
```

To use NVIDIA cuPyNumeric for runtime array operations, set
`TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric` and launch under Legate on the cluster:

```bash
TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric legate --gpus 4 --fbmem 20000 \
  examples/pfs/evaluate_energy.py \
  --molecule n2 \
  --script examples/pfs/n2/simple_pf.py
```

For external `run_eval.py` / `campaign.py` runs, the molecule-local
`evaluate.py` wrapper re-executes itself under Legate when the cuPyNumeric
backend is selected:

```bash
export TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric
export TROVOLUT_PFS_LEGATE_GPUS=4
export TROVOLUT_PFS_LEGATE_FBMEM=20000
python3 campaign.py --agent codex --max-launches 100 --stall-launches 5 --sleep-seconds 10
```

Set `TROVOLUT_PFS_LEGATE_REEXEC=0` if the evaluator process is already being
started by `legate`. Set `TROVOLUT_PFS_LEGATE_ARGS` to override the generated
Legate arguments completely.

Problem-cache construction and `.npz` I/O remain NumPy-backed. The backend flag
is meant for the product-formula state evolution and evaluator linear algebra.

Use molecule folders directly with the helper CLI:

```bash
uv run hsopt status examples/pfs/h2
uv run hsopt prompt examples/pfs/h2 --evaluator fidelity --epsilon 1e-3
```

Extrapolate observed autoresearch T-gate gains to a FeMoco-sized active space:

```bash
uv run hsopt extrapolate-femoco --target-baseline-t-count 1e12 \
  --plot runs/femoco_extrapolation_uncertainty.svg
```

Pass `--gain-table` with records containing `molecule`, `baseline_t_count`, and
`optimized_t_count`, or pass `--campaign-root` for an external workspace that
contains campaign `events.jsonl` files. The command fits a Bayesian linear
regression for log gain factor, reports a posterior interval, and can write an
SVG uncertainty plot. Override FeMoco assumptions with flags such as
`--active-orbitals`, `--n-qubits`, and `--total-pauli-rotation-terms`.

## Repository Boundary

This tree contains benchmark definitions, fixed evaluators, and editable lane
policies. It does not contain campaign databases, mutation histories, scheduler
logs, telemetry, or agent transcripts.

The paper campaigns are:

- five independent autoresearch runs over the fixed 121-Hamiltonian set;
- the ten-reaction comparison between equal-share and reaction-aware error
  allocation.

Their runners and compact normalized results belong under
`experiments/hamlib121/` and `experiments/reactions10/`. Full campaign
workspaces stay external.

The active mutation target is only the lane-local `simple_pf.py`. Do not edit
`chemistry.py`, `evaluate_energy.py`, `evaluate_fidelity.py`, molecule
definitions, active spaces, geometries, basis sets, target times, or
localization as part of an optimization run.

Hamiltonian factorizations (SF, DF, THC) are not used in the product-formula
campaign. Factorization reduces the 1-norm, which benefits oracle-based methods
(qubitization, QSVT) but does not reduce the Pauli rotation count in
Trotterized circuits. Factorization-based resource estimation now belongs on the
`resource/classical-compute` branch.
