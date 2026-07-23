# Reaction seed inputs

These two JSON files are the validated starting grid for the fixed ten-reaction
paper campaign:

- `seed_manifest.json`: reaction identities, stoichiometry, and ordering.
- `seed_validation.json`: the twenty validated iteration-0 policies.

The 21 JSON/NPZ Hamiltonian-cache pairs are intentionally external. They total
about 6.8 MiB and are verified before use against
`problem_cache.sha256`. Pass the `problem_cache/` directory itself explicitly
when preparing a campaign; it must contain exactly the 42 declared files and
no extras.

The checked-in inputs are checksummed as:

- normalized seed manifest: `9fe91f88d3af88295d4d23ac5a6612d5a7e3eec97cc0c234a3c602001e708b1b`
- seed validation: `e2bee43bbf79fd5cfa8877c0be166a115443528ca39f2c53ec596b6239555fcc`
- external cache manifest: `cf521d0a98c7b9cb3478ef28349200f4ab98b171a79a915edca6459349f9c22f`

The seed manifest is normalized only to replace its old machine-local script
path with the repository-relative `experiments/generate_initial_pf.py`.

The July Omni iteration-13/14 checkpoint is preserved outside the repository
for audit only. It is incomplete and is not a paper result.
