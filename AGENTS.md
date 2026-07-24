# hsopt Exact-Unitary Campaign Notes

The production paper workflow is the exact compiled-unitary pipeline in
`src/hsopt/compiled_unitary.py`,
`src/hsopt/exact_queue.py`, and
`src/hsopt/exact_loop.py`. All checked-in benchmark fixtures,
paper protocols, and support files live under `experiments/`.

Use `$hsopt` for product-formula T-gate optimization and `$ssh` for NVPark
submission or Slurm diagnosis.

## Immutable scientific ordering

Every production lane must follow these stages:

1. Prepare the paper baseline once for each distinct Hamiltonian. Compute the
   supplement's Eq. (19) double-commutator coefficient sum exactly for the
   Pauli decomposition and derive the Hamiltonian-dependent PF2 Trotter number
   without using `E_exact`. Separately construct and diagonalize
   `U_exact = exp(+i H t)` exactly once for evaluator reuse. Five HamLib
   replicas share one reference per instance; reaction references are
   assembled from the prepared species without another eigensolve.
2. Measure the generated paper policy unchanged as iteration 0 against that
   cached reference.
3. Permit OpenCode mutations only after iteration 0 passes. Mutations are
   iterations 1 through 50.

Candidate evaluation must load `exact_reference.json`; it must not reconstruct
or diagonalize `U_exact`. Analytical product-formula or synthesis error bounds
are not acceptance metrics.

## Immutable baseline

`experiments/generate_initial_pf.py` is the sole generator of `initial_pf.py`.
It requires sealed calibration data and embeds the
Reiher--Wiebe--Svore--Wecker--Troyer citation, lexicographic PF2/Strang
structure, Eq. (19) bound-derived `beta(H)` referenced to 0.1 mHa, `t=1`, the
resulting step count and step time, and the direct PyGridSynth tolerance
selected under the paper's Trotter/QPE/rotation-error relation with a
1 kcal/mol total target.
The exact reference is an evaluation oracle only; it must not affect
`beta(H)`, the step count, or the synthesis tolerance.

Every `experiments/abs_energy/<instance>/` and
`experiments/diff_energy/<reaction>/` directory contains the generated
`initial_pf.py` for that dataset element. These checked-in files must be
byte-identical to the external batch output. Never invent or hand-edit an
initializer. Never mutate `initial_pf.py`, and do not create an in-tree
`optimized_pf.py`. A lane's mutable `candidate.py` and `incumbent.py`, all
evaluations, and all transcripts live in its external campaign directory.

Do not change Hamiltonians, molecules, geometries, basis sets, active spaces,
target times, localization, electron sectors, or evaluator semantics during a
campaign.

## Production shape and resources

- HamLib exact subset: 91 instances through 14 qubits, five independent
  mutation lanes per instance, 50 mutation iterations per lane.
- Reaction set: 10 reactions, two policy lanes per reaction, 50 mutation
  iterations per lane.
- Total: 475 lanes and 23,750 mutation evaluations, excluding iteration 0.

The canonical mutation campaign uses cuPyNumeric on one A100 in the `polar4`
partition with `normal` QoS, two CPU cores, 16 GB system memory, and a 60000
MiB Legate framebuffer. PyGridSynth remains CPU-side within that allocation;
cuPyNumeric uses the A100 for compiled-unitary construction and dense
eigendecomposition. Batch tasks run sequentially so only one dense task is
resident on a GPU at a time.

The A100 campaign owns the single stochastic OpenCode mutation trace. Any CPU
validation campaign must use `--no-mutations` and replay the exact archived
A100 candidate files in order; never run a second independent agent pass.
Candidate hashes, per-attempt and confidence-adjusted T counts, and
accept/reject trajectories are required to match. CPU and GPU eigensolvers may
differ in the last floating-point bits, so compare energy observables at the
campaign's explicit numerical tolerance rather than claiming bitwise equality.

Do not increase CPUs, GPUs, or RAM merely to improve scheduling priority. A CPU
validation run must use the `cpu` partition and request no GPU resource.

OpenShell and OpenCode/Nemotron run persistently on Omni. Preserve raw
OpenCode/OpenShell logs plus redacted tool-action and metrics summaries.

## Fixed infrastructure

Treat these as fixed during a campaign unless the user explicitly changes the
benchmark framework:

- `src/hsopt/compiled_unitary.py`
- `src/hsopt/exact_queue.py`
- `src/hsopt/exact_loop.py`
- `src/hsopt/paper_baseline.py`
- `src/hsopt/paper_baseline_batch.py`
- `experiments/generate_initial_pf.py`

Campaign histories belong outside the checkout. After every lane completes,
run `hsopt-export-exact` for `abs_energy` and `diff_energy`, then run
`hsopt-reports` on both checksummed exports. The canonical A100 supervisor is
configured with all four final export/report paths so this completion gate runs
automatically and idempotently.
