# QSVT and LCU Blueprints

These are hsopt lane blueprints for Hamiltonian-simulation methods beyond the
current product-formula examples. They describe the agent/evaluator contract to
use when QSVT or LCU benchmark folders are added under `examples/`.

!!! note
    The repository currently ships runnable product-formula lanes in
    `examples/pfs/`. QSVT and LCU lanes should follow the same contract before
    they are advertised as runnable examples.

## QSVT lane shape

A QSVT lane should isolate a single editable policy file from the fixed
Hamiltonian simulation target:

```text
examples/qsvt/<problem>/
  simple_qsvt.py        # agent edits this
  evaluate.py           # fixed scorer
  .hsopt.json           # optional lane manifest
```

The evaluator should report JSON with at least:

| Field | Meaning |
| --- | --- |
| `score` | lower-is-better objective |
| `passes_accuracy_constraint` | boolean validity gate |
| `query_count` or `t_gate_count` | primary cost |
| `approximation_error` | QSVT approximation diagnostic |

Agent mutations should focus on:

- polynomial degree and parity choices
- phase synthesis precision
- block-encoding precision budgets
- amplitude amplification tolerances
- approximation-error slack allocation across subroutines

The evaluator should hold fixed:

- target Hamiltonian and normalization
- target function or time-evolution objective
- allowed approximation error
- block-encoding definition
- score semantics

Example agent instruction:

```text
$hsopt Run 15 mutations on examples/qsvt/h2_time_evolution.
Reduce query/T cost by reallocating polynomial and phase precision, but keep
the fixed approximation-error check valid.
```

## LCU lane shape

An LCU lane should expose one editable implementation policy and one fixed
scorer:

```text
examples/lcu/<problem>/
  simple_lcu.py         # agent edits this
  evaluate.py           # fixed scorer
  .hsopt.json           # optional lane manifest
```

The evaluator should report JSON with:

| Field | Meaning |
| --- | --- |
| `score` | lower-is-better objective |
| `passes_accuracy_constraint` | boolean validity gate |
| `lambda_norm` | fixed or reported LCU normalization |
| `prepare_cost` | PREPARE cost diagnostic |
| `select_cost` | SELECT cost diagnostic |
| `t_gate_count` or `query_count` | primary cost |

Agent mutations should focus on:

- coefficient bucketing
- term grouping and ordering
- PREPARE precision allocation
- SELECT synthesis precision
- oblivious amplitude amplification schedules
- truncation thresholds that remain inside the fixed error budget

The evaluator should hold fixed:

- Hamiltonian coefficients and target time
- normalization convention
- success probability/error contract
- scoring function

Example agent instruction:

```text
$hsopt Run 15 mutations on examples/lcu/lih_sparse_terms.
Lower the valid T count by changing grouping and PREPARE/SELECT precision
allocation only.
```

## Promotion checklist

Before a new method family becomes a public hsopt example:

1. Add a fixed `evaluate.py` that prints deterministic JSON.
2. Add one clearly named editable method file.
3. Add a smoke molecule or small Hamiltonian that runs in CI-scale time.
4. Document which file the agent may edit.
5. Add a status/prompt path so the skill can target the lane directly.
