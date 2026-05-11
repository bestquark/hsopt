# Product-Formula Benchmarks

`examples/ft_protocols/pfs/` is the canonical home for the product-formula
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

Run either evaluator directly:

```bash
uv run python examples/ft_protocols/pfs/evaluate_energy.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py
uv run python examples/ft_protocols/pfs/evaluate_fidelity.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py --epsilon 1e-3
```

To use NVIDIA cuPyNumeric for runtime array operations, set
`TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric` and launch under Legate on the cluster:

```bash
TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric legate --gpus 4 --fbmem 20000 \
  examples/ft_protocols/pfs/evaluate_energy.py \
  --molecule n2 \
  --script examples/ft_protocols/pfs/n2/simple_pf.py
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
uv run t-autoresearch status examples/ft_protocols/pfs/h2
uv run t-autoresearch prompt examples/ft_protocols/pfs/h2 --evaluator fidelity --epsilon 1e-3
```

## Autoresearch Campaign

The experiment grid for the large-molecule benchmarks is:

- molecules: `n2`, `co`, `nh3`, `hcn`
- constraints: `energy`, `fidelity`
- target iterations per lane: `100`
- target simulation time: `t=1`, from each molecule's `MoleculeSpec`

That gives `4 * 2 = 8` independent hsopt/Codex autoresearch lanes.

The active mutation target is only the lane-local `simple_pf.py`. Do not edit
`chemistry.py`, `evaluate_energy.py`, `evaluate_fidelity.py`, molecule
definitions, active spaces, geometries, basis sets, target times, or
localization as part of an optimization run.

Each lane needs a `campaign_env.json` recording the fixed evaluator contract:

```json
{
  "TROVOLUT_PFS_EVALUATOR": "energy",
  "TROVOLUT_PFS_FIDELITY_EPSILON": null
}
```

Use `TROVOLUT_PFS_EVALUATOR=energy` for the energy constraint and
`TROVOLUT_PFS_EVALUATOR=fidelity` for the fidelity constraint. Leave
`TROVOLUT_PFS_FIDELITY_EPSILON` unset unless a specific fidelity budget is part
of the experiment.

On the cluster, add the cuPyNumeric/Legate settings before the campaign launch:

```bash
export TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric
export TROVOLUT_PFS_LEGATE_GPUS=4
export TROVOLUT_PFS_LEGATE_FBMEM=20000
```

Hamiltonian factorizations (SF, DF, THC) are not used in the product-formula
campaign. Factorization reduces the 1-norm, which benefits oracle-based methods
(qubitization, QSVT) but does not reduce the Pauli rotation count in
Trotterized circuits. Factorization-based resource estimation now belongs on the
`resource/hamiltonian-simulation-resources` branch.
