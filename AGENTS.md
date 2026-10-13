# hsopt Exact-Unitary Campaign Notes

The production paper workflow is the exact compiled-unitary pipeline in
`cli/src/hsopt/evaluation/compiled_unitary.py`,
`cli/src/hsopt/campaign/queue.py`, and
`cli/src/hsopt/campaign/loop.py`. All checked-in benchmark fixtures,
paper protocols, and support files live under `experiments/`.

Use `$hsopt` for product-formula T-gate optimization. Execution is provided by
the external runtime through `hsopt-handoff`; see `docs/runtime.md`.

## Immutable scientific ordering

Every production lane must follow these stages:

1. Prepare the selected paper baseline once for each distinct Hamiltonian.
   `loose` obtains the Hamiltonian-dependent PF2 Trotter number from the
   supplement's rigorous Eq. (19) Pauli double-commutator upper bound without
   using `E_exact`. `tight` obtains it from the first numerically passing PF2
   step count against the cached exact ground energy at the same 0.1 mHa
   reference error. Prepare the exact ground-phase reference once with the
   configured solver for calibration. Cache the lowest exact HF-accessible
   energy separately for candidate scoring; retain the ground energy as a
   diagnostic. Existing calibrated policies need not be regenerated. GPU
   Lanczos resolves H and maps its phases analytically to the exact unitary.
2. Measure the generated paper policy unchanged as iteration 0 against that
   cached reference.
3. Permit OpenCode mutations only after iteration 0 passes. Mutations are
   iterations 1 through the configured campaign budget.

Every reaction is evaluated in two independent scientific contracts:
`individual_errors` enforces `sum_s |nu_s (E_pf,s - E_exact,s)|`, while
`error_cancellation` enforces `|sum_s nu_s (E_pf,s - E_exact,s)|`. Both are
compared with the same chemical-accuracy threshold. Exact species references
may be shared, but candidate task IDs, lane manifests, mutations, incumbents,
exports, reports, and plots must remain contract-isolated.

Candidate evaluation must load `exact_reference.json`; it must not reconstruct
or diagonalize `U_exact`. Analytical product-formula or synthesis error bounds
are not acceptance metrics.

## Immutable baseline

`experiments/generate_initial_pf.py` defines the compact `ProductFormula` and is the sole generator of
`initial_pf_loose.py` and `initial_pf_tight.py`. Both require sealed calibration
data and embed the Reiher--Wiebe--Svore--Wecker--Troyer citation,
lexicographic PF2/Strang structure, `beta(H)` referenced to 0.1 mHa, the
norm-derived physical target time, and the CUDA-Q tolerance selected by
the same paper Eq. (34) Trotter/QPE/rotation-error allocation with a
1 kcal/mol total target. The exact reference must not affect the loose policy;
it intentionally selects the numeric beta used by the tight policy.
Calibration math and sealed-record preparation live in
`cli/src/hsopt/campaign/calibration.py`; the generator forwards its existing API.

Every `experiments/abs_energy/<instance>/` and
`experiments/diff_energy/<reaction>/` directory contains both generated
initializers for that dataset element. These checked-in files must be
byte-identical to the matching external batch output. Never invent or
hand-edit an initializer. A campaign must select one baseline explicitly and
copy it to the lane as `initial_pf.py`; the OpenShell workspace must never
expose the sibling baseline. Never mutate a generated initializer, and do not create an in-tree
`optimized_pf.py`. A lane's mutable `candidate.py` and `incumbent.py`, all
evaluations, and all transcripts live in its external campaign directory.

Do not change Hamiltonians, molecules, geometries, basis sets, active spaces,
target times, localization, electron sectors, or evaluator semantics during a
campaign.

## Production shape and resources

Dataset size, independent seeds, mutation budget, and compute resources are
selected by the deployment and recorded in each campaign manifest. Historical
campaign sizes are not defaults for a newly authorized campaign.

The deployment chooses evaluation resources and inference configuration.
Keep scheduler commands, credentials, resource/account defaults, and hardware
locks in external `runtime/` code. The scientific package must not import it.

A canonical campaign owns its stochastic mutation trace. Validation replays
archived candidates without another independent agent pass. Preserve candidate
hashes, scoring contracts, exact references, and numerical parity checks.
Loose/tight roots and reaction contracts remain separate on every backend.

## Fixed infrastructure

Treat these as fixed during a campaign unless the user explicitly changes the
benchmark framework:

- `cli/src/hsopt/evaluation/compiled_unitary.py`
- `cli/src/hsopt/campaign/queue.py`
- `cli/src/hsopt/campaign/loop.py`
- `cli/src/hsopt/campaign/baseline_batch.py`
- `experiments/generate_initial_pf.py` (the compact baseline policy and generator)
- `cli/src/hsopt/campaign/calibration.py` (baseline math and sealed calibration records)

Campaign histories belong outside the checkout. After every lane completes,
run `hsopt-export-exact` for `abs_energy` and `diff_energy`, then run
`hsopt-reports` on both checksummed exports. The external deployment must
invoke this completion gate automatically and idempotently.

## Shared protocol experiments

`experiments/abs_protocol/` and `experiments/diff_protocol/` own separate shared-PF
trajectories. One candidate is evaluated on every case in a sealed external suite;
the host binds per-case calibrated r, epsilon, and beta. Mutation prompts must
explicitly explore mixed-precision rotation synthesis. Use geometric-mean
baseline-relative T95 with a separate accuracy constraint for each selected case.
Keep reaction contracts, baselines and replicates isolated. GPU Lanczos is mandatory;
do not drop a case to improve the aggregate score or alter existing campaigns.

Shared absolute protocols additionally select `energy_target` explicitly.
`ground_branch` uses the cached exact ground input from tight calibration and
scores `|E_compiled,0 - E_0|`, with its compiled QPE peak probability in T95.
Ideal ground-state preparation cost is excluded. Its task IDs, evaluator version,
manifests and dashboard labels must remain separate from HF-accessible runs.
Do not regenerate beta calibrations when changing this selected scoring contract.
