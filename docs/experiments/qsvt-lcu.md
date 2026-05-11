# QSVT and LCU directions

QSVT and LCU are planned showroom directions for Hamiltonian simulation methods
beyond product formulas. They should become exhibits when their benchmark
folders exist under `examples/` with the same agent/evaluator contract.

!!! note
    The current runnable exhibit is product-formula T-count optimization in
    `examples/pfs/`. The QSVT and LCU sections below define the shape of future
    experiment lanes, not currently runnable folders.

## QSVT exhibit shape

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
- amplitude-amplification tolerances
- approximation-error allocation across subroutines

The evaluator should hold fixed:

- target Hamiltonian and normalization
- target function or time-evolution objective
- allowed approximation error
- block-encoding definition
- score semantics

Example exhibit prompt:

```text
$hsopt Run 15 mutations on examples/qsvt/h2_time_evolution.
Reduce query/T cost by reallocating polynomial and phase precision, but keep
the fixed approximation-error check valid.
```

## LCU exhibit shape

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
- oblivious amplitude-amplification schedules
- truncation thresholds that remain inside the fixed error budget

The evaluator should hold fixed:

- Hamiltonian coefficients and target time
- normalization convention
- success probability/error contract
- scoring function

Example exhibit prompt:

```text
$hsopt Run 15 mutations on examples/lcu/lih_sparse_terms.
Lower the valid T count by changing grouping and PREPARE/SELECT precision
allocation only.
```

## Promotion checklist

Before a new method family becomes a public exhibit:

1. Add a fixed `evaluate.py` that prints deterministic JSON.
2. Add one clearly named editable method file.
3. Add a smoke molecule or small Hamiltonian that runs in CI-scale time.
4. Document exactly which file the agent may edit.
5. Add a status/prompt path so `hsopt` can target the lane directly.
