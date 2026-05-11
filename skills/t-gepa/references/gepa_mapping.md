# GEPA Mapping for T-Gate Prompt Optimization

The local `t-gepa` evaluator maps GEPA concepts to hsopt as follows:

- `seed_candidate`: `{"prompt": <strategy text>}`
- `dataset`: JSON examples with `family`, `molecule`, `target_iterations`, and
  optional `max_agent_seconds`
- `evaluator(candidate, example)`: isolated T-gate trial
- GEPA metric: valid fractional T-count reduction
- GEPA side info: trial path, prompt, agent outputs, lane summary, and score
  components

Example dataset item:

```json
{
  "id": "pf1_h2",
  "family": "pf1",
  "molecule": "h2",
  "target_iterations": 1,
  "max_agent_seconds": 900
}
```

Use low budgets at first. Each metric call can launch an agent and run at least
one chemistry evaluator, so GEPA budget is materially more expensive than a
normal prompt-only benchmark.
