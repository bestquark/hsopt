# How T gates are counted

The product-formula experiments report compiled T count, not an analytic
rotation-count proxy.

## Rotation compilation

Each non-identity Pauli rotation is reduced to a single-qubit `Rz(theta)` by a
Clifford scaffold:

1. Basis-change gates map `X` and `Y` support into the `Z` basis.
2. A CNOT ladder computes parity onto one pivot qubit.
3. The pivot receives the synthesized `Rz(theta)`.
4. The CNOT ladder and basis changes are undone.

The code uses:

```text
examples/pfs/chemistry.py::build_pauli_rotation_compiler
```

Only the synthesized `Rz` contributes to `t_gate_count`. The surrounding
Clifford scaffold affects `gate_count`, but not the T count.

## pygridsynth accounting

For each rounded angle and synthesis epsilon, the compiler calls pygridsynth to
produce a Clifford+T gate string for `Rz(theta)`.

The T count is:

```python
gate_string.count("T")
```

The evaluator tries multiple pygridsynth candidates, removes duplicate gate
strings, and picks the candidate with the smallest tuple:

```text
(t_gate_count, operator_error, gate_count, gate_string)
```

That makes T count the first optimization target, with operator error and total
gate string length used as tie breakers.

## Total experiment count

The product-formula method file applies many Pauli rotations. For every
nonzero rotation, the lane adds the compiled `t_gate_count` for that rotation to
the run total.

Important distinction:

- `t_gate_count`: total T gates spent over the full product-formula trajectory.
- `unique_compilation_count`: number of distinct `(label, angle, epsilon)`
  compilations seen by the cache.

Repeated rotations are counted each time they appear in the simulated circuit,
even when the compiled rotation was cached.

Zero-angle or skipped rotations add no T gates and are tracked separately as
`skipped_rotation_count`.

## Score and validity

The evaluator returns:

```text
score = t_gate_count + penalty
```

The penalty is zero when the run satisfies the fixed accuracy budget. When the
observed error exceeds the budget, the penalty scales as:

```text
1e9 * (excess_error / allowed_error)
```

If the accuracy metric is missing or non-finite, the evaluator adds a `1e9`
penalty.

For the energy evaluator, the accuracy metric is the absolute error of the
selected HF-reference spectral peak against the fixed-sector ground-state
energy. For the fidelity evaluator, the metric is final-state infidelity
against exact evolution.

## Output fields to inspect

Useful fields in evaluator JSON:

| Field | Meaning |
| --- | --- |
| `t_gate_count` | total compiled T gates for the trajectory |
| `score` | T count plus any accuracy penalty |
| `passes_accuracy_constraint` | whether the run is valid |
| `nonzero_rotation_count` | rotations that were applied |
| `skipped_rotation_count` | rotations skipped by the method policy |
| `synthesis_epsilon_histogram` | epsilon buckets used by the method |
| `unique_compilations` | per-rotation compilation diagnostics |
| `operator_error` | pygridsynth approximation error for a compiled rotation |

These diagnostics are what the next agent mutation should read before changing
the precision policy.
