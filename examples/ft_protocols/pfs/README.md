# Product-Formula Benchmarks

`examples/ft_protocols/pfs/` is the canonical home for the product-formula
Hamiltonian-simulation benchmarks in trovolut.

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

For Trovolut `run_eval.py` / `campaign.py` runs, the molecule-local
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

## 32-Lane Factorization Campaign

This is the setup contract for the large-molecule histogram follow-up
campaign. The intended experiment grid is:

- molecules: `n2`, `co`, `nh3`, `hcn`
- constraints: `energy`, `fidelity`
- Hamiltonian factorization cost models: `unfactorized`, `sf`, `df`, `thc`
- target iterations per lane: `100`
- target simulation time: `t=1`, from each molecule's `MoleculeSpec`

That gives `4 * 2 * 4 = 32` independent Trovolut/Codex autoresearch lanes.
Name each lane directory exactly:

```text
{molecule}_{constraint}_{factorization}
```

For example:

```text
n2_energy_unfactorized
n2_energy_sf
n2_energy_df
n2_energy_thc
n2_fidelity_unfactorized
...
hcn_fidelity_thc
```

Keep campaign histories outside `examples/ft_protocols/pfs/`, for example:

```text
runs/t_autoresearch/pfs_histogram_factorized_<timestamp>/
```

Each lane must be scaffolded from the matching canonical molecule folder:

```text
examples/ft_protocols/pfs/{molecule}/
```

The active mutation target is only the lane-local `simple_pf.py`. Do not edit
`chemistry.py`, `evaluate_energy.py`, `evaluate_fidelity.py`, molecule
definitions, active spaces, geometries, basis sets, target times, localization,
or factorization cache semantics as part of an optimization run.

Each lane needs a `campaign_env.json` recording the fixed evaluator contract:

```json
{
  "TROVOLUT_PFS_EVALUATOR": "energy",
  "TROVOLUT_PFS_HAMILTONIAN_FACTORIZATION": "df",
  "TROVOLUT_PFS_FIDELITY_EPSILON": null
}
```

Use `TROVOLUT_PFS_EVALUATOR=energy` for the energy row and
`TROVOLUT_PFS_EVALUATOR=fidelity` for the fidelity row. Use one of
`unfactorized`, `sf`, `df`, or `thc` for
`TROVOLUT_PFS_HAMILTONIAN_FACTORIZATION`. Leave
`TROVOLUT_PFS_FIDELITY_EPSILON` unset unless a specific fidelity budget is part
of the experiment.

Before launching the full grid, run one cheap preflight from the repo root:

```bash
uv run python -m py_compile \
  examples/ft_protocols/pfs/chemistry.py \
  examples/ft_protocols/pfs/evaluate_energy.py \
  examples/ft_protocols/pfs/evaluate_fidelity.py \
  examples/ft_protocols/pfs/n2/simple_pf.py \
  examples/ft_protocols/pfs/co/simple_pf.py \
  examples/ft_protocols/pfs/nh3/simple_pf.py \
  examples/ft_protocols/pfs/hcn/simple_pf.py
```

A lane is launched from inside its run directory with:

```bash
export TROVOLUT_PFS_EVALUATOR=energy
export TROVOLUT_PFS_HAMILTONIAN_FACTORIZATION=df
python3 campaign.py --agent codex --max-launches 100 --stall-launches 5 --sleep-seconds 10
```

On the cluster, add the cuPyNumeric/Legate settings before the campaign launch:

```bash
export TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric
export TROVOLUT_PFS_LEGATE_GPUS=4
export TROVOLUT_PFS_LEGATE_FBMEM=20000
```

The lane-local `evaluate.py` wrapper will re-exec the scored evaluator under
`legate` when `TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric` is selected. If the
cluster scheduler already starts the evaluator under `legate`, set
`TROVOLUT_PFS_LEGATE_REEXEC=0`.

Keep each lane's autoresearch loop sequential: one mutation, one scored
evaluation, then inspect the result before the next mutation. The 4-GPU setting
parallelizes the linear algebra inside that scored evaluation. If the scheduler
allocates one GPU per job instead, use `TROVOLUT_PFS_LEGATE_GPUS=1` and run up
to four independent lane jobs concurrently on the node.

To launch all 32 lanes from a setup script, iterate over the Cartesian product
above, export each lane's `campaign_env.json`, and start one independent
campaign process per lane. On a shared machine or login node, prefer scheduler
array jobs over 32 local background processes.

After lanes have progress, render the requested `2 x 4` summary figure from the
repo root:

```bash
uv run python examples/ft_protocols/pfs/plot_factorization_autoresearch.py \
  runs/t_autoresearch/pfs_histogram_factorized_<timestamp>
```

The plotter expects the lane directory names shown above. It writes
`pfs_factorization_autoresearch.pdf` and `.png` into the run root. Each subplot
shows best valid T-gate count versus iteration for the four factorization
curves; columns are molecules and rows are `energy` / `fidelity`.

Campaign histories should be created by the external run harness. This folder
only stores the benchmark definitions, the molecule-local `simple_pf.py` files,
and fixed evaluator infrastructure.
