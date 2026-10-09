# Reaction inputs

Two compact JSON files define the fixed ten-reaction campaign:

- `reactions.json`: reaction identities, stoichiometry, ordering, and the
  $1$ kcal/mol signed-energy target.
- `problem_metadata.json`: basis, active-space, term-count, qubit-count, and
  the norm-based phase-encoding contract for each unique species Hamiltonian.

`reaction_cache.tar.gz` contains the original 21 JSON/NPZ Hamiltonian-cache
pairs. They total about 6.8 MiB before compression and preserve the exact input
checksums. Materialize them outside the checkout with:

```bash
uv run hsopt-data reactions --output-root ../data/raw/reactions
```

Extraction verifies all 42 files against `problem_cache.sha256`. Pass that
output directory to `hsopt-campaign prepare-reactions`; it must contain exactly
the declared files and no extras.

The sibling reaction directories expose this registry as one
`definition.json` per reaction, including the applicable cache checksums.
`uv run hsopt-layout --check` proves that view is current.

No old fixed-step or prevalidated initializer is authoritative. The loose
Eq. (19) bound and tight exact-energy numeric beta are separately calibrated
from every prepared Hamiltonian and rendered by
`experiments/generate_initial_pf.py`; their large calibration outputs stay in
the external campaign data root.
