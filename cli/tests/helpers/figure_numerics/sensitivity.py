"""Numerical validation extracted from archived analyze_policy_sensitivity.py; no plotting frontend."""

from __future__ import annotations


import math


import numpy as np

from scipy.linalg import eigvals, schur


from hsopt.evaluation import compiled_unitary as ev


class PauliActions:
    """O(2**n) workspace, without a per-term index/phase cache."""

    def __init__(self, n):
        self.n = n
        self.indices = np.arange(1 << n, dtype=np.int64)
        self.parity = np.array([i.bit_count() % 2 for i in self.indices], dtype=np.int8)

    def apply(self, label, value):
        x = sum(1 << k for k, c in enumerate(reversed(label)) if c in "XY")
        z = sum(1 << k for k, c in enumerate(reversed(label)) if c in "YZ")
        source = self.indices ^ x
        phase = (1j ** label.count("Y")) * (1 - 2 * self.parity[source & z])
        if value.ndim == 2:
            phase = phase[:, None]
        return phase * value[source]

    def expectation(self, label, state):
        value = np.vdot(state, self.apply(label, state))
        if abs(value.imag) > 1e-10:
            raise ValueError("Hermitian Pauli expectation is not real")
        return float(value.real)


def propagate(actions, schedule, state, perturbation=None):
    """+i alpha P convention; an error is inserted AFTER the selected gadget."""
    value = state.copy()
    for j, (term, alpha) in enumerate(schedule):
        value = math.cos(alpha) * value + 1j * math.sin(alpha) * actions.apply(term["label"], value)
        if perturbation is not None and j == perturbation[0]:
            _, axis, delta = perturbation
            sign, label = ev._conjugated_pivot_pauli(term["label"], axis)
            value = math.cos(delta) * value + 1j * math.sin(delta) * sign * actions.apply(
                label, value
            )
    return value


def response(actions, schedule, state):
    """Return dimensionless signed eigenphase gradients at each rotation.

    Exact for an isolated step eigenvector. For a cached H eigenvector these
    are a ground-input approximation, rather than certified eigenphase gradients.
    X/Y/Z are errors on the compiled Rz pivot, conjugated by its real scaffold.
    A phase-aligned infinitesimal SU(2) error has norm ||delta||_2; its worst
    first-order phase change is ||b_j||_2 ||delta||_2.
    """
    value = state.copy()
    derivatives = []
    for term, alpha in schedule:
        value = math.cos(alpha) * value + 1j * math.sin(alpha) * actions.apply(term["label"], value)
        b = []
        for axis in "XYZ":
            sign, label = ev._conjugated_pivot_pauli(term["label"], axis)
            if abs(sign.imag) > 1e-12:
                raise ValueError("nonreal Hermitian scaffold sign")
            b.append(sign.real * actions.expectation(label, value))
        if np.linalg.norm(b) > 1 + 1e-9:
            raise ValueError("single-qubit Bloch response exceeds one")
        derivatives.append(b)
    return np.asarray(derivatives), value


def exact_step_target(actions, schedule, ground):
    """Only for small offline validation, never an evaluator fallback."""
    if actions.n > 8:
        raise ValueError("dense derivative validation is limited to eight qubits")
    unitary = propagate(actions, schedule, np.eye(ground.size, dtype=complex))
    triangular, vectors = schur(unitary, output="complex")
    spectrum = np.diag(triangular)
    weights = np.abs(vectors.conj().T @ ground) ** 2
    index = int(weights.argmax())
    phases = np.angle(spectrum * spectrum[index].conjugate())
    group = np.flatnonzero(np.abs(phases) < 1e-10)
    basis = vectors[:, group]
    target = basis @ (basis.conj().T @ ground)
    probability = float(np.vdot(target, target).real)
    target /= math.sqrt(probability)
    phase = float(np.angle(spectrum[index]))
    residual = float(np.linalg.norm(unitary @ target - np.exp(1j * phase) * target))
    return target, phase, probability, residual, len(group)


def validate_derivatives(actions, schedule, ground, approximate):
    target, phase, probability, residual, multiplicity = exact_step_target(
        actions, schedule, ground
    )
    exact, _ = response(actions, schedule, target)
    difference = approximate - exact
    result = {
        "qubits": actions.n,
        "step_eigenstate_ground_input_probability": probability,
        "step_eigenvector_residual": residual,
        "selected_phase_multiplicity": multiplicity,
        "ground_input_proxy_max_absolute_gradient_difference": float(abs(difference).max()),
        "ground_input_proxy_relative_gradient_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(exact), 1e-30)
        ),
        "angle_proxy_relative_gradient_l2_error": float(
            np.linalg.norm(difference[:, 2]) / max(np.linalg.norm(exact[:, 2]), 1e-30)
        ),
        "finite_differences": [],
    }
    # A degenerate eigenphase may split: its projected expectation is not a
    # unique branch derivative. Do not manufacture a finite-difference match.
    if multiplicity != 1:
        result["finite_difference_skip"] = "degenerate eigenphase requires subspace perturbation"
        return result
    selected = sorted({0, len(schedule) // 2})
    for j in selected:
        for axis_index, axis in enumerate("XYZ"):
            differences = []
            for delta in (1e-5, 1e-6):
                phases = []
                for sign in (-1, 1):
                    perturbed = propagate(
                        actions,
                        schedule,
                        np.eye(ground.size, dtype=complex),
                        (j, axis, sign * delta),
                    )
                    values = eigvals(perturbed)
                    offset = np.angle(values * np.exp(-1j * phase))
                    phases.append(float(offset[np.argmin(abs(offset))]))
                differences.append((phases[1] - phases[0]) / (2 * delta))
            analytic = float(exact[j, axis_index])
            result["finite_differences"].append(
                {
                    "rotation": j,
                    "axis": axis,
                    "analytic_phase_gradient": analytic,
                    "central_gradients": differences,
                    "absolute_error_at_1e_6": abs(differences[-1] - analytic),
                }
            )
    return result
