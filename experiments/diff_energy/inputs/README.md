# Reaction inputs

Two compact JSON files define the fixed ten-reaction campaign:

- `reactions.json`: reaction identities, stoichiometry, ordering, and the
  $1$ kcal/mol signed-energy target.
- `problem_metadata.json`: basis, active-space, term-count, qubit-count, and
  the norm-based phase-encoding contract for each unique species Hamiltonian.

The 21 JSON/NPZ Hamiltonian-cache pairs are intentionally external. They total
about 6.8 MiB and are verified before use against
`problem_cache.sha256`. Pass the `problem_cache/` directory itself explicitly
when preparing a campaign; it must contain exactly the 42 declared files and
no extras.

The sibling reaction directories expose this registry as one
`definition.json` per reaction, including the applicable cache checksums.
`uv run hsopt-layout --check` proves that view is current.

No old fixed-step or prevalidated initializer is authoritative. The Eq. (19)
paper bound is recomputed from every prepared Hamiltonian and rendered by
`experiments/generate_initial_pf.py`; its large calibration outputs stay in the
external campaign data root.
