# Product-Formula Optimization

Product-formula lanes are the runnable hsopt examples today.

The agent mutates one lane-local file:

```text
examples/pfs/<molecule>/simple_pf.py
```

The evaluator fixes the Hamiltonian instance, active space, target time,
localization, and accuracy criterion.

## Inspect the lane

```bash
uv run hsopt list
uv run hsopt status examples/pfs/h2
```

The status output includes the canonical energy and fidelity evaluator commands.

## Generate the agent prompt

```bash
uv run hsopt prompt examples/pfs/h2 --evaluator energy
```

In Codex, invoke the installed skill directly:

```text
$hsopt Run 20 sequential mutations on examples/pfs/h2 using the energy evaluator.
Bias toward broad epsilon policies over one-label tweaks.
```

In Claude Code, use the slash command when exposed by the client:

```text
/hsopt Run 20 sequential mutations on examples/pfs/h2 using the energy evaluator.
Bias toward broad epsilon policies over one-label tweaks.
```

## What the agent should try

Good mutations are precision-policy changes that remain easy to audit:

- group Pauli labels by family, support size, or coefficient magnitude
- relax synthesis epsilon for low-impact rotations first
- tighten only terms implicated by failed accuracy diagnostics
- keep `n_steps` fixed unless the evaluator shows product-formula error rather
  than synthesis error

The agent should not edit the evaluator, molecule definition, active space,
target time, geometry, basis, or scoring semantics.

## Score one candidate

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy
```

Or print the exact command without running it:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator fidelity --epsilon 1e-3 --print-only
```

## Optimize the strategy prompt

Use `hsgepa` when the search target is the agent policy rather than a direct
`simple_pf.py` mutation.

```bash
uv run hsgepa dataset --smoke --output runs/hsgepa/dataset_smoke.json
uv run hsgepa optimize \
  --dataset runs/hsgepa/dataset_smoke.json \
  --run-root runs/hsgepa/smoke \
  --max-metric-calls 6 \
  --no-parallel
```

Each GEPA trial copies the needed benchmark files into an isolated workspace,
runs an agent with the candidate strategy, restores the best valid state, and
scores valid fractional T-count reduction.
