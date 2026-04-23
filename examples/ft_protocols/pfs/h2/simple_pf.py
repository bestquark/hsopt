"""
Editable product-formula Hamiltonian-simulation resource policy.

This method estimates direct real-time dynamics by multiplying Pauli rotations
for a fixed molecular Hamiltonian. Mutations should only change product-formula
order, step count, and synthesis precision policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str = "second_order_product_formula_baseline"
    order: int = 2
    trotter_steps: int = 50
    rotation_synthesis_epsilon: float = 1e-7
    synthesis_cache_round_digits: int = 16


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def _pauli_operator(label: str) -> np.ndarray:
    matrices = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }
    out = np.array([[1.0]], dtype=np.complex128)
    for char in label:
        out = np.kron(out, matrices[char])
    return out


def _product_formula_sequence(
    labels: Iterable[str],
    coefficients: Iterable[float],
    order: int,
) -> Tuple[Tuple[str, float, float], ...]:
    terms = tuple((str(label), float(coeff)) for label, coeff in zip(labels, coefficients))
    if int(order) == 1:
        return tuple((label, coeff, 1.0) for label, coeff in terms)
    if int(order) == 2:
        forward = tuple((label, coeff, 0.5) for label, coeff in terms)
        backward = tuple((label, coeff, 0.5) for label, coeff in reversed(terms))
        return forward + backward
    raise ValueError("order must be 1 or 2")


def _pauli_rotation(pauli: np.ndarray, theta: float) -> np.ndarray:
    identity = np.eye(pauli.shape[0], dtype=np.complex128)
    return np.cos(theta) * identity - 1.0j * np.sin(theta) * pauli


def _exact_centered_unitary(problem: Dict[str, object]) -> np.ndarray:
    hamiltonian = np.asarray(problem["hamiltonian"], dtype=np.complex128)
    centered = hamiltonian - float(problem["identity_coefficient"]) * np.eye(
        hamiltonian.shape[0],
        dtype=np.complex128,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    phases = np.exp(-1.0j * eigenvalues * float(problem["target_time"]))
    return (eigenvectors * phases) @ eigenvectors.conj().T


def _phase_insensitive_operator_error(approx: np.ndarray, exact: np.ndarray) -> float:
    phase = np.angle(np.trace(exact.conj().T @ approx))
    aligned = np.exp(-1.0j * phase) * approx
    return float(np.linalg.norm(aligned - exact, ord=2))


def _synthesized_rotation_t_count(theta: float, epsilon: float) -> int:
    if abs(theta) <= 1e-15:
        return 0
    from pygridsynth import gridsynth_gates, mpmath

    # A Pauli rotation exp(-i theta P) reduces to one synthesized Rz(2 theta)
    # plus Clifford basis changes and parity compute/uncompute.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates = gridsynth_gates(mpmath.mpf(str(2.0 * abs(theta))), mpmath.mpf(str(epsilon)))
    return int(gates.count("T"))


def estimate_product_formula(problem: Dict[str, object]) -> Dict[str, object]:
    config = get_method_config()
    steps = int(config.trotter_steps)
    if steps <= 0:
        raise ValueError("trotter_steps must be positive")

    labels = tuple(problem["pauli_labels"])
    coefficients = np.asarray(problem["pauli_coefficients"], dtype=np.float64)
    sequence = _product_formula_sequence(labels, coefficients, int(config.order))
    dt = float(problem["target_time"]) / float(steps)
    paulis = {label: _pauli_operator(label) for label in labels}

    step_unitary = np.eye(2 ** int(problem["n_qubits"]), dtype=np.complex128)
    synthesis_cache: Dict[float, int] = {}
    per_step_t = 0
    per_step_rotation_records = []
    for label, coefficient, factor in sequence:
        theta = coefficient * dt * factor
        step_unitary = _pauli_rotation(paulis[label], theta) @ step_unitary
        cache_key = round(theta, int(config.synthesis_cache_round_digits))
        if cache_key not in synthesis_cache:
            synthesis_cache[cache_key] = _synthesized_rotation_t_count(
                theta,
                float(config.rotation_synthesis_epsilon),
            )
        rotation_t = synthesis_cache[cache_key]
        per_step_t += rotation_t
        per_step_rotation_records.append(
            {
                "pauli": label,
                "theta": theta,
                "rotation_t_count": rotation_t,
            }
        )

    approx = np.linalg.matrix_power(step_unitary, steps)
    exact = _exact_centered_unitary(problem)
    product_formula_operator_error = _phase_insensitive_operator_error(approx, exact)
    rotation_count = steps * len(sequence)
    synthesis_operator_error = rotation_count * float(config.rotation_synthesis_epsilon)
    target_time = float(problem["target_time"])
    accuracy_proxy = (product_formula_operator_error + synthesis_operator_error) / max(target_time, 1e-15)

    return {
        "method_config": asdict(config),
        "target_time": target_time,
        "order": int(config.order),
        "trotter_steps": steps,
        "pauli_term_count": len(labels),
        "rotations_per_step": len(sequence),
        "rotation_count": rotation_count,
        "product_formula_operator_error": product_formula_operator_error,
        "rotation_synthesis_operator_error_bound": synthesis_operator_error,
        "product_formula_energy_error_hartree": product_formula_operator_error / max(target_time, 1e-15),
        "rotation_synthesis_energy_error_hartree": synthesis_operator_error / max(target_time, 1e-15),
        "accuracy_proxy_hartree": accuracy_proxy,
        "accuracy_proxy_terms": {
            "product_formula_energy_error_hartree": product_formula_operator_error / max(target_time, 1e-15),
            "rotation_synthesis_energy_error_hartree": synthesis_operator_error / max(target_time, 1e-15),
        },
        "count_status": "dense_exact_dynamics_with_gridsynth_proxy",
        "gate_counts": {
            "total_t_count": int(steps * per_step_t),
            "per_step_t_count": int(per_step_t),
            "unique_synthesized_rotation_count": len(synthesis_cache),
            "rotation_count": rotation_count,
        },
        "per_step_rotations": per_step_rotation_records,
        "t_gate_count": int(steps * per_step_t),
        "logical_qubits": int(problem["n_qubits"]),
    }
