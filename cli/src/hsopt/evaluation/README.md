# Evaluation methods

- **`exact` (default):** synthesize rotations, measure compiled-unitary energy error,
  and count the T gates needed for QPE. This is the production acceptance score.
- **`bch`:** estimate leading Trotter energy bias with sparse commutators and a
  small approximate reference state. It is a predictor, not a certified bound;
  it omits synthesis errors and T counts (`score: null`).

Choose `--method exact|bch` with `hsopt-evaluate` or either experiment family's
`evaluate.py`. Both read the same Hamiltonian JSON and `ProductFormula` policy.
Exact requires `--exact-reference`; BCH uses `--reference-size` (default 32).

```bash
uv run hsopt-evaluate --method bch --problem /path/to/hamiltonian.json \
  --policy-file /path/to/candidate.py --output /path/to/proxy.json
```

BCH supports symmetric, first-order-consistent schedules. Its reference cap
controls cost, not accuracy, and sparse expansions can still be expensive.
Reaction mode uses `--reaction` plus `--error-contract`. Production mutation
queues remain exact; proxy estimates alone do not accept a mutation.
