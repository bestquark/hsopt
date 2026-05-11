# Agent start

This page is the handoff surface for an agent. Start here when a human gives
you the hsopt repository or the docs URL and asks you to optimize a
Hamiltonian-simulation experiment.

## Copy/paste handoff

```text
You are working in the hsopt repository. Start by reading docs/agent-start.md,
then run `uv run hsopt list` and `uv run hsopt status <lane>`. For a direct
T-gate optimization, edit only the selected lane's `simple_pf.py`, keep the
fixed evaluator and chemistry contracts unchanged, run one evaluator after each
mutation, and preserve or restore the best valid state before stopping.
```

## First read

Use the repository as the source of truth. The runnable benchmark family today
is:

```text
examples/pfs/
```

The default task is to reduce compiled product-formula T count while preserving
the fixed evaluator accuracy contract. Do not redefine the chemistry problem or
the scoring rule unless the human explicitly asks for a benchmark-framework
change.

## Bootstrap

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync
uv run hsopt list
```

Pick a lane, then inspect its contract:

```bash
uv run hsopt status examples/pfs/h2
uv run hsopt prompt examples/pfs/h2 --evaluator energy
uv run hsopt evaluate examples/pfs/h2 --evaluator energy --print-only
```

The lane status prints the editable file, evaluator files, selected molecule,
and canonical evaluator command. The rendered prompt is the most direct
agent-facing instruction for that lane.

## Choose the work mode

| User intent | Start with |
| --- | --- |
| Reduce T gates for one molecule | `$hsopt`, then `uv run hsopt status <lane>` |
| Inspect available lanes | `uv run hsopt list` |
| Score the current implementation | `uv run hsopt evaluate <lane> --evaluator energy` |
| Optimize the agent prompt itself | `$hsgepa`, then `uv run hsgepa dataset --smoke ...` |
| Understand T accounting | [How T Gates Are Counted](experiments/t-gates.md) |

Use `energy` unless the human asks for final-state fidelity:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy
uv run hsopt evaluate examples/pfs/h2 --evaluator fidelity --epsilon 1e-3
```

## Mutation rules

For a direct hsopt optimization run:

- edit only the active lane's `simple_pf.py`;
- make one coherent precision-policy mutation at a time;
- run exactly one fixed evaluator after each mutation;
- read the JSON diagnostics before the next mutation;
- preserve campaign history outside `examples/pfs/`;
- restore or keep the best valid state before stopping.

Keep these fixed:

- molecule, geometry, basis, active space, target time, and localization;
- evaluator thresholds and score semantics;
- `examples/pfs/chemistry.py`;
- `examples/pfs/evaluate_energy.py`;
- `examples/pfs/evaluate_fidelity.py`.

## A minimal agent loop

```bash
uv run hsopt status examples/pfs/h2
uv run hsopt evaluate examples/pfs/h2 --evaluator energy
```

Then:

1. Read the JSON result and note `passes_accuracy_constraint`, `score`, and
   `t_gate_count`.
2. Edit `examples/pfs/h2/simple_pf.py` to change one precision policy.
3. Run the same evaluator command again.
4. Keep the change only if it gives a better valid result, or if it is a
   deliberate diagnostic step the human asked for.
5. Report the lane, evaluator, changed file, previous T count, new T count,
   validity, and any remaining risk.

## GEPA prompt optimization

Use this only when the object being optimized is the agent strategy prompt:

```bash
uv sync --extra gepa
uv run hsgepa seed-prompt > /tmp/hsgepa_seed_prompt.md
uv run hsgepa dataset --smoke --output runs/hsgepa/dataset_smoke.json
uv run hsgepa optimize \
  --dataset runs/hsgepa/dataset_smoke.json \
  --run-root runs/hsgepa/smoke \
  --max-metric-calls 6 \
  --no-parallel
```

GEPA trials should stay in `runs/hsgepa/` and should not directly mutate the
canonical benchmark lanes.

## Stop conditions

Stop and ask the human before continuing when:

- the requested molecule or method family is not listed by `uv run hsopt list`;
- evaluator dependencies are unavailable and cannot be installed locally;
- a change requires editing chemistry, evaluator scoring, or benchmark data;
- the best valid state is ambiguous after several invalid mutations.

When you stop, include the exact command that failed or the exact contract that
would need to change.
