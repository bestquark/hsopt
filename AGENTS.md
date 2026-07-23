# hsopt Exact-Unitary Campaign Notes

The production paper workflow is the exact compiled-unitary pipeline in
`src/hsopt/compiled_unitary.py`, `src/hsopt/exact_queue.py`, and
`src/hsopt/exact_loop.py`. All checked-in benchmark fixtures, paper protocols,
and support files live under `experiments/`.

Use `$hsopt` for product-formula T-gate optimization and `$ssh` for NVPark
submission or Slurm diagnosis.

## Immutable scientific ordering

Every production lane must follow these stages:

1. Construct and diagonalize `U_exact = exp(+i H t)` once for each distinct
   Hamiltonian reference and cache the resulting exact eigenphase energy and
   state. Five HamLib replicas share one reference per instance; the two
   policies for a reaction share one reaction reference.
2. Measure the generated paper policy unchanged as iteration 0 against that
   cached reference.
3. Permit OpenCode mutations only after iteration 0 passes. Mutations are
   iterations 1 through 50.

Candidate evaluation must load `exact_reference.json`; it must not reconstruct
or diagonalize `U_exact`. Analytical product-formula or synthesis error bounds
are not acceptance metrics.

## Immutable baseline

`experiments/generate_initial_pf.py` is the sole generator of `initial_pf.py`.
It embeds the Reiher--Wiebe--Svore--Wecker--Troyer citation, paper method, and
explicit benchmark-specific numerical choices, so there is no second baseline
file that can drift from the generator.

Never invent or hand-edit an initializer. Never mutate `initial_pf.py`, and do
not create an in-tree `optimized_pf.py`. A lane's mutable `candidate.py` and
`incumbent.py`, all evaluations, and all transcripts live in its external
campaign directory.

Do not change Hamiltonians, molecules, geometries, basis sets, active spaces,
target times, localization, electron sectors, or evaluator semantics during a
campaign.

## Production shape and resources

- HamLib exact subset: 91 instances through 14 qubits, five independent
  mutation lanes per instance, 50 mutation iterations per lane.
- Reaction set: 10 reactions, two policy lanes per reaction, 50 mutation
  iterations per lane.
- Total: 475 lanes and 23,750 mutation evaluations, excluding iteration 0.

Dense exact work uses cuPyNumeric on NVPark for both placements. PyGridSynth
remains CPU-side. Both placements evaluate the same uploaded source snapshot
and batch tasks sequentially so only one dense task is resident at a time:

- Canonical mutation campaign: `cpu` partition, `normal` QoS, zero GPUs,
  2 Slurm CPUs, 64 GB system memory, and the pinned cuPyNumeric 25.11 CPU
  build with Legate configured as one CPU worker plus runtime utility work.
- A100 validation/replay campaign: `polar4` partition, `normal` QoS, one GPU,
  2 CPUs, and 16 GB system memory.

The CPU campaign owns the stochastic OpenCode mutation trace. The A100
campaign must use `--no-mutations` and replay the exact archived CPU candidate
files in order; never run a second independent agent pass. Candidate hashes,
per-attempt and confidence-adjusted T counts, and accept/reject trajectory are
required to match; record and stop on any validation discrepancy. CPU and GPU
eigensolvers may differ in the last floating-point bits, so compare energy
observables at the campaign's explicit numerical tolerance rather than
claiming bitwise equality.

Do not increase CPUs or RAM merely to improve scheduling priority. A CPU run
must use the `cpu` partition and request no GPU resource.

OpenShell and OpenCode/Nemotron run persistently on Omni. Preserve raw
OpenCode/OpenShell logs plus redacted tool-action and metrics summaries.

## Fixed infrastructure

Treat these as fixed during a campaign unless the user explicitly changes the
benchmark framework:

- `src/hsopt/compiled_unitary.py`
- `src/hsopt/exact_queue.py`
- `src/hsopt/exact_loop.py`
- `experiments/generate_initial_pf.py`

Campaign histories belong outside the checkout. After every lane completes,
run `hsopt-export-exact` for `abs_energy` and `diff_energy`, then run
`hsopt-reports` on both checksummed exports. The canonical CPU supervisor is
configured with all four final export/report paths so this completion gate runs
automatically and idempotently.
