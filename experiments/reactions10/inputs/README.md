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

The archived source payload is checksummed as:

- seed manifest: `7cd26b6d25117cbebcec93cd9b800c4ab473925adee895d147ddc3e135f45094`
- seed validation: `e2bee43bbf79fd5cfa8877c0be166a115443528ca39f2c53ec596b6239555fcc`
- original 44-file payload: `788908526ba4d5d4484418ed37d760d55cb76fd0f47a63588723ed93800c4206`
- original checksum manifest: `3111d725e00066147c885e169c6ee15e2b2b5b77a8219de11d33b5006fb3c863`

The July Omni iteration-13/14 checkpoint is preserved outside the repository
for audit only. It is incomplete and is not a paper result.
