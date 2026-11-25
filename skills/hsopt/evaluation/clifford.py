"""Exact signed-Pauli conjugation and auditable Clifford circuit cancellation."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
MATRICES = {
    "H": (PAULIS["X"] + PAULIS["Z"]) / math.sqrt(2),
    "S": np.diag([1, 1j]),
    "SDG": np.diag([1, -1j]),
    "CX": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}
INVERSE = {"H": "H", "S": "SDG", "SDG": "S", "CX": "CX", "X": "X"}


def _table():
    table = {}
    for gate, matrix in MATRICES.items():
        operators = (
            {a + b: np.kron(x, y) for a, x in PAULIS.items() for b, y in PAULIS.items()}
            if gate == "CX"
            else PAULIS
        )
        for label, operator in operators.items():
            rotated = matrix @ operator @ matrix.conj().T
            for target, candidate in operators.items():
                coefficient = np.trace(candidate.conj().T @ rotated).real / len(matrix)
                if abs(coefficient) > 0.5:
                    table[gate, label] = (1 if coefficient > 0 else -1, target)
                    break
    return table


CONJUGATION = _table()


def validate_frame(raw, n_qubits):
    if not isinstance(raw, (list, tuple)) or len(raw) > 16 * n_qubits:
        raise ValueError("basis_change() must return at most 16*n_qubits Clifford gates")
    gates = []
    for gate in raw:
        if not isinstance(gate, (list, tuple)) or not gate or gate[0] not in MATRICES:
            raise ValueError("basis gates are ('H'|'S'|'SDG', q) or ('CX', control, target)")
        name, *qubits = gate
        if len(qubits) != (2 if name == "CX" else 1):
            raise ValueError("wrong basis-gate arity")
        if any(type(q) is not int or q < 0 or q >= n_qubits for q in qubits):
            raise ValueError("invalid basis-gate qubit")
        if len(set(qubits)) != len(qubits):
            raise ValueError("basis-gate qubits must differ")
        gates.append(tuple(gate))
    return gates


def inverse(gates):
    return [(INVERSE[g[0]], *g[1:]) for g in reversed(gates)]


def conjugate(label, gates):
    """Return sign, Q with C P C† = sign Q; gates are in application order."""
    sign = 1
    result = list(label)
    for name, *qubits in gates:
        multiplier, target = CONJUGATION[name, "".join(result[q] for q in qubits)]
        sign *= multiplier
        for q, axis in zip(qubits, target, strict=True):
            result[q] = axis
    return sign, "".join(result)


def scaffold(label):
    """Application order of the same star-parity gadget used by the evaluator."""
    support = [q for q, axis in enumerate(label) if axis != "I"]
    gates = []
    for q in support:
        if label[q] == "Y":
            gates.append(("SDG", q))
        if label[q] in "XY":
            gates.append(("H", q))
    gates.extend(("CX", q, support[-1]) for q in support[:-1])
    return gates, support[-1]


def circuit_metrics(rotations, compiled, *, frame=()):
    """Cancel adjacent inverse Cliffords, never change the charged T words.

    This is an executable gate-list rewrite, not a LIQUi|> emulator. Fused
    simulation evaluates the exactly equivalent scaffold-conjugated matrices.
    Depths use an all-to-all logical circuit with unit-duration gates.
    """
    raw = list(frame)
    for (term, _angle), word in zip(rotations, compiled, strict=True):
        before, pivot = scaffold(term["label"])
        raw.extend(before)
        # CUDA-Q's word is a matrix product: its rightmost gate acts first.
        raw.extend((gate, pivot) for gate in reversed(word) if gate not in "IW")
        raw.extend(inverse(before))
    raw.extend(inverse(frame))
    reduced = []
    for gate in raw:
        if reduced and gate[0] in INVERSE and reduced[-1] == (INVERSE[gate[0]], *gate[1:]):
            reduced.pop()
        else:
            reduced.append(gate)
    count = Counter(g[0] for g in reduced)
    clocks, t_clocks = {}, {}
    for name, *qubits in reduced:
        clock = 1 + max((clocks.get(q, 0) for q in qubits), default=0)
        t_clock = (name == "T") + max((t_clocks.get(q, 0) for q in qubits), default=0)
        for q in qubits:
            clocks[q], t_clocks[q] = clock, t_clock
    return {
        "optimizer": "adjacent_inverse_cliffords_v1",
        "connectivity": "all_to_all_logical_qubits",
        "gate_counts": dict(sorted(count.items())),
        "gates_before_cancellation": len(raw),
        "gates_after_cancellation": len(reduced),
        "cancelled_clifford_gates": len(raw) - len(reduced),
        "one_step_logical_depth": max(clocks.values(), default=0),
        "one_step_t_depth": max(t_clocks.values(), default=0),
        "one_step_t_count": count["T"],
        "scalar_phases_retained_in_evaluation": True,
    }
