# Shared absolute-energy protocol

For a fresh grouped baseline, see [ELOTE](../../docs/elote.md) and select
`--policy-profile fermionic_nested` with its newly prepared calibration records.
The existing `pauli` profile retains the historical initializer.

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

Mutation preparation reads the complete compact suite summary plus small,
middle and large raw examples. All Hamiltonians remain available for inspection;
every candidate is still evaluated on every case. Failed reads and conclusively
blocked writes are audited and recoverable. Successful edits of protected files
and uncertain writes remain invalid.

## Train/test generalization

The default source registry is [`baseline/molecules.json`](../../baseline/molecules.json),
with all 115 molecular training Hamiltonians through 16 qubits. Materialize it with:

```bash
uv run python experiments/abs_protocol/prepare_inputs.py \
  --download --archive-root /work/data/hamlib --output /work/data/hamlib115
```

For the separately defined historical 100-case generalization experiment, export
that explicit selection instead of changing the default registry:

```bash
uv run hsopt-data export --selection generalization100 \
  --output-root /work/data/registry100
uv run hsopt-campaign prepare-hamlib \
  --active-manifest /work/data/registry100/molecules.csv \
  --checksum-manifest /work/data/registry100/hamlib_zip_sha256.txt \
  --data-root /work/data/hamlib --output-root /work/data/hamlib100 --max-qubits 16
```

Prepare cached ground references and the selected grouped baseline externally
before making the selection JSON; no test results select the dataset membership.

`generalize.py split` creates immutable, disjoint random selections. For a fixed
100-instance selection, use 80 training and 20 test instances and a different
recorded split seed per replicate:

```bash
uv run python experiments/abs_protocol/generalize.py split \
  --selection /work/data/selection100.json --output /work/data/splits/seed-01 \
  --seed 1 --train-size 80 --test-size 20
```

The scientific APIs in `hsopt.campaign.generalization` prepare a training run
from `train.json` alone. Its geometric-mean T95 objective and per-case accuracy
constraints use those 80 cases. Test Hamiltonians, references, baselines and
results stay outside the mutation sandbox and feedback. Splits are fixed for
all iterations, not resampled each round. Inputs and memberships are checksummed.

After all training rounds finish, `prepare_test` freezes that seed's final
training incumbent. `advance_test` emits ordinary portable GPU requests for the
20 test baselines and that frozen policy, without an agent or policy selection.
Its `generalization_report.json` reports the frozen policy even if test accuracy
fails or its T95 increases. It never substitutes the test baseline, tunes a
parameter from test performance, or feeds results back into training. Runtime
code dispatches these requests and triggers the final test automatically.

Five seeds × 100 rounds × 80 training cases require 40,000 mutation evaluations,
plus 400 training baselines, 100 test baselines and 100 final test evaluations.
These are random **Hamiltonian-instance** splits: related geometries/active
spaces may occur on both sides, so this does not claim unseen-species
extrapolation. Each case still receives its own host-calibrated starting
parameters; this measures transfer of the learned protocol relative to those
baselines, rather than eliminating baseline calibration on unseen cases.
