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
   reference error. Construct and diagonalize `U_exact = exp(+i H t)` exactly
   once; both calibration modes and every lane reuse that reference.
2. Measure the generated paper policy unchanged as iteration 0 against that
   cached reference.
3. Permit OpenCode mutations only after iteration 0 passes. Mutations are
   iterations 1 through 50.

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

`experiments/generate_initial_pf.py` is the sole generator of
`initial_pf_loose.py` and `initial_pf_tight.py`. Both require sealed calibration
data and embed the Reiher--Wiebe--Svore--Wecker--Troyer citation,
lexicographic PF2/Strang structure, `beta(H)` referenced to 0.1 mHa, the
norm-derived physical target time, and the CUDA-Q tolerance selected by
the same paper Eq. (34) Trotter/QPE/rotation-error allocation with a
1 kcal/mol total target. The exact reference must not affect the loose policy;
it intentionally selects the numeric beta used by the tight policy.

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

- HamLib exact subset: 82 instances through 12 qubits, three independent
  mutation lanes per instance, 50 mutation iterations per lane.
- Reaction set: 10 reactions, one `individual_errors` lane and one
  `error_cancellation` lane per reaction, 50 mutation iterations per lane.
- Total: 266 lanes and 13,300 mutation evaluations, excluding iteration 0.

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
- `experiments/generate_initial_pf.py` (the sole baseline definition and policy generator)

Campaign histories belong outside the checkout. After every lane completes,
run `hsopt-export-exact` for `abs_energy` and `diff_energy`, then run
`hsopt-reports` on both checksummed exports. The external deployment must
invoke this completion gate automatically and idempotently.
