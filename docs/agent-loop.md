# Agent experiment loop

Each hsopt exhibit is designed around an explicit experiment contract.

## Lane shape

| Concept | Requirement |
| --- | --- |
| Editable method file | the only file the agent mutates by default |
| Fixed evaluator | computes the scientific target and prints JSON |
| Diagnostics | enough fields for the next mutation to be evidence-driven |

The product-formula exhibit uses:

```text
examples/pfs/<molecule>/simple_pf.py
examples/pfs/evaluate_energy.py
examples/pfs/evaluate_fidelity.py
```

## The loop

1. Read the lane status and evaluator contract.
2. Inspect the previous scored result.
3. Make one coherent method-policy mutation.
4. Run one fixed evaluator command.
5. Read the JSON diagnostics before choosing the next mutation.
6. Keep or restore the best valid state before stopping.

The agent should not change molecule definitions, geometries, active spaces,
target times, evaluator thresholds, or score semantics unless the experiment
itself is being redefined.

## Evaluator output

The scorer should emit a JSON object with:

- `score`: lower-is-better objective.
- `passes_accuracy_constraint`: boolean validity gate.
- a primary cost such as `t_gate_count` or `query_count`.
- diagnostics that explain accuracy failures or cost regressions.

For product formulas, the lower-is-better score is the compiled T count plus a
large penalty when the selected accuracy metric exceeds its fixed budget.

## What `hsgepa` changes

`hsgepa` does not directly optimize a molecule. It optimizes the prompt that
drives the agent loop. The reward is valid fractional T-count reduction inside
an isolated trial workspace.
