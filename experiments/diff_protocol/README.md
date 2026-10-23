# Shared reaction-energy protocol

For a fresh grouped baseline, see [ELOTE](../../docs/elote.md) and select
`--policy-profile fermionic_nested` with new calibrations for every species.
Reaction preparation apportions their budgets by stoichiometric weight; the
whole reaction iteration 0 must still pass before mutations can begin.

One short `initial_pf.py` is mutated per round and evaluated on every selected
reaction, using its species' existing calibrated `r`, epsilon, and tuned beta.
The objective is the geometric mean of reaction T95 relative to iteration 0,
with **every reaction required to pass accuracy**. Mutation prompts explicitly
ask for mixed-precision rotation-synthesis schemes as well as schedule/r changes.

The workflow and portable requests match [abs_protocol](../abs_protocol/README.md).
A selection case points to a reaction JSON, its cached exact difference reference,
and a `calibration_root` containing all species' sealed calibrations:

```json
{"cases": [{"id": "reaction_01", "source": "prepared/reaction.json",
  "reference": "cached/exact_difference_reference.json",
  "calibration_root": "cached/calibrations"}]}
```

```bash
uv run python experiments/diff_protocol/mutate.py prepare \
  --selection "$HSOPT_WORKDIR/data/diff-protocol-selection.json" \
  --run-dir "$HSOPT_WORKDIR/runs/diff_protocol/tight/individual_errors/seed-01" \
  --baseline tight --error-contract individual_errors --iterations 100 --replicate seed-01
uv run python experiments/diff_protocol/evaluate.py \
  --run-dir "$HSOPT_WORKDIR/runs/diff_protocol/tight/individual_errors/seed-01"
uv run python experiments/diff_protocol/mutate.py advance \
  --run-dir "$HSOPT_WORKDIR/runs/diff_protocol/tight/individual_errors/seed-01"
```

Run `error_cancellation` in a **different root**, with its own shared candidate,
trajectory, and aggregate score. Parameters are bound per reaction, preserving
its stoichiometric error allocation even when a species appears in several
reactions. CUDA-Q compilation and GPU Lanczos remain the fixed judge.
Select the suite explicitly; do not change its membership mid-trajectory. All
selected reactions must return before the next mutation.

For a reaction-difference dynamics comparison aligned with the molecular ELOTE
ground-input contract, select `--energy-target ground_branch`. This compares
each species' compiled QPE peak against its cached exact ground energy before
forming the signed reaction difference. It excludes ideal ground-state
preparation cost, as the molecular ground-input experiment does. It has a
separate evaluator version, task hashes, and run directory from HF-accessible
reaction scoring. Build its references from the cached exact-ground species
states with `prepare_ground_references.py`; pass `--include-held` only when the
new suite intentionally admits a previously held reaction such as HNO→NOH.
To test whether reaction-specific optimization improves a molecular ELOTE
policy, pass that policy as `initial_policy` to `protocol.prepare`. The run
seals its hash, evaluates it at iteration 0, and then counts 50 subsequent
mutations. Keep the tight PF2 baseline in a separate validation run for the
three-stage baseline → ELOTE → ELOTE-Delta comparison.
