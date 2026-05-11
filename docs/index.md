# hsopt

Agent skills for Hamiltonian-simulation optimization.

`hsopt` turns a benchmark lane into an agent-editable optimization problem: one
method file, one fixed evaluator, one score, and a loop that preserves the
scientific contract while searching for lower cost.

<div class="hsopt-grid" markdown>

<div class="hsopt-card" markdown>
## Install the skills

```bash
npx skills add bestquark/hsopt
```

Install into Claude Code explicitly:

```bash
npx skills add bestquark/hsopt --agent claude-code --skill '*' -y
```
</div>

<div class="hsopt-card" markdown>
## Run a lane

```bash
uv run hsopt list
uv run hsopt status examples/pfs/h2
uv run hsopt prompt examples/pfs/h2
```

Use `$hsopt` inside Codex or `/hsopt` where the client exposes slash commands.
</div>

<div class="hsopt-card" markdown>
## Optimize the prompt

```bash
uv run hsgepa dataset --smoke --output runs/hsgepa/dataset_smoke.json
uv run hsgepa optimize --smoke --run-root runs/hsgepa/smoke
```

`hsgepa` uses GEPA to improve the agent strategy that drives hsopt lanes.
</div>

</div>

## What hsopt optimizes

The current runnable benchmark family is product-formula T-gate optimization in
`examples/pfs/`. Each lane exposes a `simple_pf.py` policy and fixed evaluator
entry points:

- `evaluate_energy.py`: minimize compiled T count subject to chemical accuracy
  on the HF-reference spectral peak.
- `evaluate_fidelity.py`: minimize compiled T count subject to final-state
  infidelity below the selected budget.

The same lane pattern is intended for other Hamiltonian-simulation methods:

| Method family | Agent edits | Evaluator holds fixed | Score direction |
| --- | --- | --- | --- |
| Product formulas | precision policy in `simple_pf.py` | molecule, active space, target time, accuracy test | lower T count |
| QSVT | polynomial degree, phase policy, block-encoding tolerances | block encoding, target function, approximation error | lower query or T cost |
| LCU | term grouping, coefficient buckets, PREPARE/SELECT tolerances | Hamiltonian terms, normalization, success/error test | lower T or query cost |

QSVT and LCU pages are blueprints for adding those lanes. They are not presented
as runnable folders until the corresponding `examples/` benchmarks exist.

## Agent loop

The important part is the contract, not the optimizer name:

1. Inspect the lane and fixed evaluator.
2. Mutate exactly the allowed method file.
3. Run one scored evaluation.
4. Read the JSON result.
5. Keep the best valid state and continue.

That makes hsopt a skill surface for repeated agent optimization, not just a
library API.
