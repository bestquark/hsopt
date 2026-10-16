# Evaluation methods

The [ELOTE policy interface](../../../../docs/elote.md) exposes
commuting groups, splitting stages, repetitions, rotation angles, mixed synthesis
precision and optional exact Clifford basis changes. Resolved recipes and logical
Clifford cancellation/T-depth diagnostics are stored alongside the fixed score.
Depth is separate from total T95; grouping alone does not discount gate count.

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
Exact evaluations compile rotations with CUDA-Q Ross--Selinger on CPU, preserving
global phase and counting the returned T gates. The policy's epsilon is used
directly, with seed 0 and fixed search budgets recorded in every result.

GPU candidate evaluation selects `--backend cupy-lanczos`: CUDA-Q CPU synthesis,
fused complex128 matrix construction, and GPU Lanczos. It requires a cached exact
reference. Phase enclosure, selected Ritz residuals, and repeated HF-weight
checks must pass. An unresolved solve returns an error; this backend never
falls back to full-unitary diagonalization. Partial-spectrum diagnostics are
explicitly marked and do not claim to enumerate all HF-accessible peaks.
The same GPU backend prepares cached references through Hamiltonian Lanczos
and analytic unitary phases. Tight calibration tracks the cached ground target
with ideal rotations; it does not change the production HF-accessible score.

Shared `ground_branch` runs certify the cached ground input's dominant compiled
peak. If the first residual-qualified Lanczos start cannot be independently
confirmed, the evaluator automatically retries after stabilizing its energy and
overlap at larger Krylov checkpoints. The original fast path, residual/overlap
tolerances, candidate and energy target are preserved. Results record the
certifier version, retry diagnostics and total Krylov actions. Remaining
uncertified solves are inconclusive, with no measured accuracy violation implied.
