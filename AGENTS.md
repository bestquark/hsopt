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

`baseline/generate_initial_pf.py` defines the compact `ProductFormula` and is the sole generator of
`initial_pf_loose.py` and `initial_pf_tight.py`. Both require sealed calibration
data and embed the Reiher--Wiebe--Svore--Wecker--Troyer citation,
lexicographic PF2/Strang structure, `beta(H)` referenced to 0.1 mHa, the
norm-derived physical target time, and the CUDA-Q tolerance selected by
the same paper Eq. (34) Trotter/QPE/rotation-error allocation with a
1 kcal/mol total target. The exact reference must not affect the loose policy;
it intentionally selects the numeric beta used by the tight policy.
Calibration math and sealed-record preparation live in
`cli/src/hsopt/campaign/calibration.py`; the generator forwards its existing API.

The separately selected `fermionic_nested` profile starts from
`baseline/initial_nested_pf.py`. `hsopt-elote prepare` calibrates its own
orbital-disjoint grouped PF2 and validates its unchanged compiled iteration 0;
never reinterpret historical Pauli calibrations as nested calibrations.
`--policy-profile fermionic_nested` selects this initializer for a NEW shared
experiment. The explicit ELOTE interface and reproduction boundaries are in
`docs/elote.md`. ELOTE names the learned protocol; HSOpt remains
the search framework. Keep running campaigns on their pinned source and inputs.

Input metadata lives only in `baseline/molecules.json` and `baseline/reactions.json`.
The single-instance experiment trees and generated baseline copies are archived on
`single_molecule`; do not recreate them in main. Use `hsopt-data export` for
external CSV/JSON preparation views and `hsopt-layout --check` for registry checks.
Generated calibrations and starting policies belong outside the checkout. Never
hand-edit a generated initializer. A campaign selects one baseline and exposes
only that policy as `initial_pf.py`; never expose its sibling in the mutation sandbox.
A lane's mutable `candidate.py`, `incumbent.py`, evaluations and transcripts remain
in its external campaign directory. Frozen selected releases may be checked in
with their source hashes, parameter rows and validation provenance.

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
- `baseline/generate_initial_pf.py` (the compact baseline policy and generator)
- `cli/src/hsopt/campaign/calibration.py` (baseline math and sealed calibration records)

Campaign histories belong outside the checkout. Preserve every round's candidate,
per-case results, aggregate decision, sealed inputs and evaluator identity. Replays
use the archived source and inputs, not newly generated substitutes. Existing
low-level `abs_energy`/`diff_energy` identifiers in external archives are historical
schema identifiers; they do not imply checked-in single-instance folders.

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

## Frozen manuscript artifacts

Current figure generators and released figures live in `figures/`; data processing
for manuscript figures belongs there. Keep only active manuscript generators and
their build/audit/table support in this directory. Numerical test helpers belong
in `cli/tests/helpers/`, and one-time exporters stay in their archived commit.
Frozen candidate/decision/case-metric exports and plotting inputs live under the
matching `experiments/*/results/` directories.
`experiments/manuscript.json` maps these inputs to their unchanged manuscript assets.
Never export credentials, raw sandbox messages or mutable dispatcher state. Preserve
all reported policy and source hashes; scientific aggregate values are not compiler
proxies. Older optional plotting frontends are on `manuscript-assets-archive-20261030`.
