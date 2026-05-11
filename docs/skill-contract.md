# Skill Contract

Every hsopt lane should make the agent's freedom explicit.

## Files

The lane owns three concepts:

| Concept | Requirement |
| --- | --- |
| Editable method file | the only file the agent mutates by default |
| Fixed evaluator | scores a candidate and prints JSON |
| Lane metadata | optional manifest for names, defaults, and support files |

Current product-formula lanes use:

```text
examples/pfs/<molecule>/simple_pf.py
examples/pfs/evaluate_energy.py
examples/pfs/evaluate_fidelity.py
```

## Evaluator JSON

The scorer should emit a JSON object with:

- `score`: lower-is-better objective.
- `passes_accuracy_constraint`: boolean validity gate.
- a primary cost such as `t_gate_count` or `query_count`.
- diagnostics that tell the next agent mutation where accuracy or cost failed.

## Agent policy

The agent should:

1. Read the lane status and previous result.
2. Make one coherent method mutation.
3. Run one fixed evaluator command.
4. Read the JSON output before the next mutation.
5. Restore or keep the best valid state before stopping.

The agent should not change the fixed benchmark target, evaluator contract, or
score semantics unless the user explicitly asks to change the benchmark
framework.

## Prompt optimization

`hsgepa` optimizes the strategy prompt that controls this loop. Its reward is a
valid fractional cost reduction in an isolated trial workspace, not the raw lane
score itself.
