# hsopt

Agent-run Hamiltonian-simulation optimization experiments.

This site is not package API documentation. It is a showroom for the experiments
in this repository: what is computed, what the agent is allowed to change, how
the evaluator scores a run, and how the reported quantum costs are counted.

<div class="hsopt-grid" markdown>

<div class="hsopt-card" markdown>
## Agent start

Give an agent this page first. It explains what to inspect, which commands to
run, what can be edited, and when to stop.

[Open the agent handoff](agent-start.md)
</div>

<div class="hsopt-card" markdown>
## Cost accounting

The current experiments compile Pauli rotations with pygridsynth and count the
`T` symbols in the synthesized single-qubit rotation sequence.

[Read the T-gate accounting](experiments/t-gates.md)
</div>

<div class="hsopt-card" markdown>
## Live exhibit

Product-formula T-count optimization is runnable today.

```bash
uv run hsopt status examples/pfs/h2
uv run hsopt prompt examples/pfs/h2
```
</div>

</div>

## What is being shown

The current tracked experiment surface on `main` is:

```text
examples/pfs/
```

Those product-formula lanes simulate fixed molecular Hamiltonians and ask a
simple question: how low can an agent push compiled T count without breaking the
fixed energy or fidelity constraint?

The larger ambition is to use the same showroom pattern for Hamiltonian
simulation methods beyond product formulas:

| Exhibit | Status | What the agent optimizes | What stays fixed |
| --- | --- | --- | --- |
| Product formulas | runnable now | synthesis precision policies in `simple_pf.py` | molecule, active space, target time, evaluator |
| QSVT | blueprint | polynomial degree, phase precision, block-encoding tolerances | target function, block encoding, error budget |
| LCU | blueprint | grouping and PREPARE/SELECT precision policies | Hamiltonian terms, normalization, success/error test |

The docs call QSVT and LCU directions "blueprints" until their benchmark
folders exist under `examples/`.

## Why agents

The point is not to expose a general package interface. The point is to make
small, auditable optimization experiments that an agent can repeat:

1. Start from a fixed scientific target.
2. Mutate only the intended method policy.
3. Run the exact scorer.
4. Archive the diagnostics.
5. Let the next mutation be informed by the previous result.

That is the role of the `hsopt` and `hsgepa` skills in this repo.
