# Shared reaction-energy protocol

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
Select the suite explicitly: the existing HNO→NOH hold is not resolved or
silently included by these new experiments. Do not change suite membership
mid-trajectory. All selected reactions must return before the next mutation.
