# Shared absolute-energy protocol

One short `initial_pf.py` is mutated once per round, then evaluated on **every
selected Hamiltonian**. The host binds the selected loose/tight calibration's
`r`, synthesis epsilon, and tuned beta for each molecule; it never recalibrates
or exposes the sibling baseline. Modify the shared rule for repetitions,
rotation ordering/angles, or **mixed-precision synthesis**.

The score is the equal-weight geometric mean
`exp(mean(log(T95_candidate / T95_iteration0)))`. Every molecule must satisfy
its fixed energy accuracy contract. `--energy-target hf_accessible` compares
the lowest compiled/exact peaks supported by the same HF determinant.
`--energy-target ground_branch` reuses the cached exact ground input from tight
calibration and compares `|E_compiled,0 - E_0|`. The latter selects the dominant
compiled QPE eigenspace by ground-input overlap, includes its probability in
T95, and excludes ideal ground-state preparation cost. GPU Lanczos must certify
majority overlap, residuals and independent-start agreement; there is no dense
fallback. Task IDs and scoring versions keep these experiments separate.
Ground-input evaluation applies the literal compiled circuit and its adjoint to
vectors through CUDA graphs. It never constructs the full unitary; Lanczos
starts near the fixed input and refines its peak with full reorthogonalization.
Clifford scaffolds and synthesis matrices are preserved, rather than replacing
compiled sequences with recovered rotation angles. Only its small Ritz matrix
is diagonalized; unresolved peaks fail explicitly.
Peak probabilities must agree within `1e-9`; evaluation rejects a case if that
uncertainty changes its integer QPE repetition count. Energy residuals and
independent-start energy agreement remain at `1e-10` Hartree.
Once feasible, accept only a
strict score reduction. Every iteration-0 case must pass before any mutation;
an inaccurate or uncertified baseline holds the suite. Zero T95 yields score zero; equal zero scores use
mean relative T95 as a tie-break. No case is silently dropped.

Create an external selection JSON; paths resolve relative to that file:

```json
{"cases": [{"id": "molecule_01", "source": "prepared/hamiltonian.json",
  "reference": "cached/exact_reference.json", "calibration": "cached/tight.json"}]}
```

Use `calibration_root` instead of `calibration` for a directory of sealed records.
List the entire chosen dataset (for example all 91 A100 molecules). References
must match the selected energy contract. Ground-input runs reuse the cached
ground vector, energy and Hamiltonian fingerprint; no HF-reference regeneration
or beta recalibration is needed. A missing or
uncertified reference needs resolution, not a diagonalization fallback.

```bash
uv run python experiments/abs_protocol/mutate.py prepare \
  --selection "$HSOPT_WORKDIR/data/abs-protocol-selection.json" \
  --run-dir "$HSOPT_WORKDIR/runs/abs_protocol/tight/seed-01" \
  --baseline tight --iterations 100 --replicate seed-01
uv run python experiments/abs_protocol/evaluate.py \
  --run-dir "$HSOPT_WORKDIR/runs/abs_protocol/tight/seed-01"
uv run python experiments/abs_protocol/mutate.py advance \
  --run-dir "$HSOPT_WORKDIR/runs/abs_protocol/tight/seed-01"
```

`prepare` requests iteration 0. Alternate evaluation and `advance` until complete.
Evaluation uses CUDA-Q synthesis and **GPU Lanczos only**. `--case-id` partitions
pending cases across workers; it never shrinks the acceptance suite. A site
runtime may dispatch the standard portable requests in `iterations/NNNN/requests.json`
and return each receipt with `mutate.py deliver --run-dir ... --case-id ... --receipt ...`.
Scheduling/batching stays outside this experiment.

Each replicate owns one `incumbent.py` and archives its shared candidate, full
`case_results.json`, aggregate `result.json`, acceptance decision, and OpenShell
traces per round. `advance --no-mutate` only reconciles results. Agent failures
use the existing bounded retry gates; operator recovery is `hsopt-openshell resume`.
Five replicates × 100 rounds produce **500 agent proposals and 45,500 evaluations**
for 91 molecules, plus baseline evaluations. Existing per-molecule campaigns
remain separate. Replicate names label independent stochastic trajectories;
they do not claim deterministic inference seeding.
