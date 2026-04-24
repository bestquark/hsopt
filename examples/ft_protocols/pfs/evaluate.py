"""
Generic fixed-reference evaluator for small-molecule product-formula precision benchmarks.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict, Tuple

import numpy as np

from chemistry import CHEMICAL_ACCURACY_HARTREE, load_problem_data
from model_registry import get_spec


class StateAction:
    def __init__(self, apply_fn):
        self._apply_fn = apply_fn

    def __matmul__(self, state: np.ndarray) -> np.ndarray:
        return self._apply_fn(np.asarray(state, dtype=np.complex128))


def one_qubit_gate_matrix(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    factors = []
    identity = np.eye(2, dtype=np.complex128)
    for idx in range(n_qubits):
        factors.append(op if idx == qubit else identity)
    matrix = np.array([[1.0]], dtype=np.complex128)
    for factor in factors:
        matrix = np.kron(matrix, factor)
    return matrix


def cnot_matrix(control: int, target: int, n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for basis_state in range(dim):
        bits = [(basis_state >> (n_qubits - 1 - idx)) & 1 for idx in range(n_qubits)]
        if bits[control]:
            bits[target] ^= 1
        output_state = 0
        for bit in bits:
            output_state = (output_state << 1) | bit
        matrix[output_state, basis_state] = 1.0
    return matrix


def apply_single_qubit_gate_to_state(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def apply_cnot_to_state(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, (control, target), (0, 1))
    updated = moved.copy()
    updated[1, 0] = moved[1, 1]
    updated[1, 1] = moved[1, 0]
    updated = np.moveaxis(updated, (0, 1), (control, target))
    return updated.reshape(-1)


def load_method_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pf1_method", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load method file: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required_symbols = ["get_method_config", "run_product_formula"]
    for symbol in required_symbols:
        if not hasattr(module, symbol):
            raise AttributeError(f"method file is missing required symbol {symbol!r}")
    return module


def patch_pygridsynth_for_python39() -> None:
    import pygridsynth.mymath
    import pygridsynth.ring

    def ntz_compat(n: int) -> int:
        n = int(n)
        if n == 0:
            return 0
        lowbit = n & -n
        return lowbit.bit_length() - 1

    pygridsynth.mymath.ntz = ntz_compat
    pygridsynth.ring.ntz = ntz_compat


def import_pygridsynth_backend():
    try:
        patch_pygridsynth_for_python39()
        from pygridsynth import gridsynth_gates
        from pygridsynth.unitary import DOmegaUnitary
    except ImportError as exc:
        raise ImportError(
            "pygridsynth is required for compiler-backed T counting. "
            "Install it with `pip install pygridsynth` or `uv sync`."
        ) from exc
    return gridsynth_gates, DOmegaUnitary


def pygridsynth_matrix_from_gates(gate_string: str, d_omega_unitary_cls) -> np.ndarray:
    matrix = d_omega_unitary_cls.from_gates(gate_string).to_complex_matrix
    return np.array(
        [[complex(matrix[row, col]) for col in range(2)] for row in range(2)],
        dtype=np.complex128,
    )


def rz_matrix(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-0.5j * theta), 0.0],
            [0.0, np.exp(0.5j * theta)],
        ],
        dtype=np.complex128,
    )


def unitary_error_up_to_global_phase(target: np.ndarray, approx: np.ndarray) -> float:
    phase = np.vdot(target[:, 0], approx[:, 0])
    if abs(phase) <= 1e-30:
        phase = 1.0 + 0.0j
    else:
        phase = phase / abs(phase)
    return float(np.linalg.norm(target - (approx / phase), ord=2))


def build_pauli_rotation_compiler(n_qubits: int):
    gridsynth_gates, d_omega_unitary_cls = import_pygridsynth_backend()
    hadamard = (1.0 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    s_dagger = np.array([[1.0, 0.0], [0.0, -1j]], dtype=np.complex128)
    s_gate = np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128)
    named_single_qubit_gates = {
        "H": hadamard,
        "S": s_gate,
        "Sdg": s_dagger,
    }

    @functools.lru_cache(maxsize=None)
    def term_scaffold(label: str) -> Tuple[Tuple[Tuple[int, str], ...], Tuple[Tuple[int, str], ...], Tuple[Tuple[int, int], ...], int]:
        support = [idx for idx, ch in enumerate(label) if ch != "I"]
        if not support:
            raise ValueError("cannot compile an identity Pauli rotation")
        pivot = support[-1]

        basis_forward: list[Tuple[int, str]] = []
        basis_inverse: list[Tuple[int, str]] = []
        for qubit, pauli in enumerate(label):
            if pauli == "X":
                basis_forward.append((qubit, "H"))
                basis_inverse.append((qubit, "H"))
            elif pauli == "Y":
                basis_forward.extend([(qubit, "Sdg"), (qubit, "H")])
                basis_inverse.extend([(qubit, "H"), (qubit, "S")])

        cnot_pairs = tuple((qubit, pivot) for qubit in support[:-1])
        return tuple(basis_forward), tuple(basis_inverse), cnot_pairs, pivot

    @functools.lru_cache(maxsize=None)
    def compile_rz(theta: float, epsilon: float) -> Dict[str, object]:
        if epsilon <= 0.0:
            raise ValueError(f"synthesis epsilon must be positive, got {epsilon}")

        target_unitary = rz_matrix(theta)
        candidates = []
        seen_gate_strings = set()
        max_attempts = 48
        stale_attempts = 0
        stale_limit = 12

        for _ in range(max_attempts):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"pygridsynth is .*",
                    category=UserWarning,
                )
                gate_string = gridsynth_gates(theta, epsilon)
            if gate_string in seen_gate_strings:
                stale_attempts += 1
                if stale_attempts >= stale_limit:
                    break
                continue
            stale_attempts = 0
            seen_gate_strings.add(gate_string)
            rz_unitary = pygridsynth_matrix_from_gates(gate_string, d_omega_unitary_cls)
            candidates.append(
                {
                    "gate_string": gate_string,
                    "gate_count": len(gate_string),
                    "t_gate_count": gate_string.count("T"),
                    "rz_unitary": rz_unitary,
                    "operator_error": unitary_error_up_to_global_phase(target_unitary, rz_unitary),
                }
            )

        if not candidates:
            raise RuntimeError(f"pygridsynth returned no candidates for theta={theta:.16e}, epsilon={epsilon:.16e}")

        best = min(
            candidates,
            key=lambda item: (
                item["t_gate_count"],
                item["operator_error"],
                item["gate_count"],
                item["gate_string"],
            ),
        )
        return {
            "theta": theta,
            "epsilon": epsilon,
            "gate_string": best["gate_string"],
            "gate_count": best["gate_count"],
            "t_gate_count": best["t_gate_count"],
            "operator_error": best["operator_error"],
            "rz_unitary": best["rz_unitary"],
        }

    @functools.lru_cache(maxsize=None)
    def compile_pauli_rotation(label: str, alpha: float, epsilon: float) -> Dict[str, object]:
        alpha = float(round(alpha, 15))
        theta = float(round(-2.0 * alpha, 15))
        rz_compilation = compile_rz(theta, epsilon)
        rz_unitary = rz_compilation["rz_unitary"]
        gate_key = (
            rz_unitary[0, 0],
            rz_unitary[0, 1],
            rz_unitary[1, 0],
            rz_unitary[1, 1],
        )
        basis_forward, basis_inverse, cnot_pairs, pivot = term_scaffold(label)
        single_gate = np.array(
            [[gate_key[0], gate_key[1]], [gate_key[2], gate_key[3]]],
            dtype=np.complex128,
        )

        if rz_compilation["gate_count"] == 0:
            def apply_state_identity(state: np.ndarray) -> np.ndarray:
                return np.asarray(state, dtype=np.complex128)

            return {
                "cache_key": f"{label}|{alpha:.16e}|{epsilon:.16e}",
                "label": label,
                "alpha": alpha,
                "theta": theta,
                "epsilon": epsilon,
                "gate_string": rz_compilation["gate_string"],
                "gate_count": rz_compilation["gate_count"],
                "t_gate_count": rz_compilation["t_gate_count"],
                "operator_error": rz_compilation["operator_error"],
                "unitary": StateAction(apply_state_identity),
                "state_action": apply_state_identity,
            }

        def apply_state(state: np.ndarray) -> np.ndarray:
            updated = np.asarray(state, dtype=np.complex128)
            for qubit, gate_name in basis_forward:
                updated = apply_single_qubit_gate_to_state(updated, named_single_qubit_gates[gate_name], qubit, n_qubits)
            for control, target in cnot_pairs:
                updated = apply_cnot_to_state(updated, control, target, n_qubits)
            updated = apply_single_qubit_gate_to_state(updated, single_gate, pivot, n_qubits)
            for control, target in cnot_pairs:
                updated = apply_cnot_to_state(updated, control, target, n_qubits)
            for qubit, gate_name in basis_inverse:
                updated = apply_single_qubit_gate_to_state(updated, named_single_qubit_gates[gate_name], qubit, n_qubits)
            return updated

        return {
            "cache_key": f"{label}|{alpha:.16e}|{epsilon:.16e}",
            "label": label,
            "alpha": alpha,
            "theta": theta,
            "epsilon": epsilon,
            "gate_string": rz_compilation["gate_string"],
            "gate_count": rz_compilation["gate_count"],
            "t_gate_count": rz_compilation["t_gate_count"],
            "operator_error": rz_compilation["operator_error"],
            "unitary": StateAction(apply_state),
            "state_action": apply_state,
        }

    return compile_pauli_rotation


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    overlap = np.vdot(state_a, state_b)
    return float(abs(overlap) ** 2)


def unwrap_energy_near_reference(phase: float, target_time: float, reference_energy: float) -> float:
    principal_energy = float(phase / target_time)
    phase_period = float((2.0 * math.pi) / target_time)
    wrap_index = round((reference_energy - principal_energy) / phase_period)
    return principal_energy + wrap_index * phase_period


def estimate_ground_state_energy_from_overlap(
    reference_state: np.ndarray,
    propagated_state: np.ndarray,
    target_time: float,
    reference_energy: float,
    identity_shift: float = 0.0,
) -> Dict[str, object]:
    overlap = np.vdot(reference_state, propagated_state)
    overlap_magnitude = float(abs(overlap))
    if overlap_magnitude <= 1e-15:
        return {
            "phase_overlap_magnitude": overlap_magnitude,
            "energy_estimate_hartree": None,
            "energy_error_hartree": None,
        }

    centered_reference_energy = float(reference_energy - identity_shift)
    centered_energy_estimate = unwrap_energy_near_reference(float(np.angle(overlap)), target_time, centered_reference_energy)
    energy_estimate = centered_energy_estimate + identity_shift
    return {
        "phase_overlap_magnitude": overlap_magnitude,
        "energy_estimate_hartree": float(energy_estimate),
        "energy_error_hartree": float(abs(energy_estimate - reference_energy)),
    }


def build_problem(molecule: str) -> Dict[str, object]:
    spec = get_spec(molecule)
    payload = load_problem_data(molecule)
    metadata = payload["metadata"]
    hamiltonian = payload["hamiltonian"]
    pauli_terms = payload["pauli_terms"]
    initial_state = payload["initial_state"]
    exact_state = payload["exact_state"]
    ground_state = payload["ground_state"]

    return {
        "experiment": f"pf1_{molecule}",
        "molecule": molecule,
        "display_name": spec.display_name,
        "hamiltonian": hamiltonian,
        "identity": np.eye(hamiltonian.shape[0], dtype=np.complex128),
        "identity_coefficient": float(metadata["identity_coefficient"]),
        "pauli_terms": pauli_terms,
        "apply_exact_pauli_rotation": payload.get("apply_exact_pauli_rotation"),
        "initial_state": initial_state,
        "exact_state": exact_state,
        "ground_state": ground_state,
        "target_time": float(spec.target_time),
        "time_sign": 1.0,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_fidelity_floor": float(metadata["chemical_accuracy_fidelity_floor"]),
        "ground_state_energy_hartree": float(metadata["ground_state_energy_hartree"]),
        "centered_hamiltonian_norm": float(metadata["centered_hamiltonian_norm"]),
        "total_pauli_rotation_terms": int(metadata["total_pauli_rotation_terms"]),
        "max_abs_total_alpha": float(metadata["max_abs_total_alpha"]),
        "min_abs_total_alpha": float(metadata["min_abs_total_alpha"]),
        "n_qubits": int(metadata["n_qubits"]),
        "compile_pauli_rotation": build_pauli_rotation_compiler(int(metadata["n_qubits"])),
        "compilation_backend": "pygridsynth",
        "cas_label": metadata["cas_label"],
        "basis": metadata["basis"],
        "localization": metadata["localization"],
    }


def score_from_fidelity(t_gate_count: int, fidelity_floor: float, fidelity_value: float) -> float:
    allowed_infidelity = max(1e-15, 1.0 - fidelity_floor)
    actual_infidelity = 1.0 - fidelity_value
    excess_infidelity = max(0.0, actual_infidelity - allowed_infidelity)
    penalty = 1e9 * (excess_infidelity / allowed_infidelity)
    return float(t_gate_count) + penalty


def score_from_ground_energy_error(
    t_gate_count: int,
    allowed_error_hartree: float,
    actual_error_hartree: float | None,
) -> float:
    if actual_error_hartree is None or not math.isfinite(actual_error_hartree):
        return float(t_gate_count) + 1e9
    excess_error = max(0.0, actual_error_hartree - allowed_error_hartree)
    penalty = 1e9 * (excess_error / max(allowed_error_hartree, 1e-15))
    return float(t_gate_count) + penalty


def evaluate(molecule: str, script_path: Path) -> Dict[str, object]:
    start = time.perf_counter()
    module = load_method_module(script_path)
    problem = build_problem(molecule)

    quantized = module.run_product_formula(problem, exact_angles=False)
    exact_pf = module.run_product_formula(problem, exact_angles=True)
    ground_problem = dict(problem)
    ground_problem["initial_state"] = np.array(problem["ground_state"], dtype=np.complex128, copy=True)
    quantized_ground = module.run_product_formula(ground_problem, exact_angles=False)
    exact_pf_ground = module.run_product_formula(ground_problem, exact_angles=True)

    exact_state = np.array(problem["exact_state"], dtype=np.complex128, copy=False)
    quantized_state = np.array(quantized["state"], dtype=np.complex128, copy=False)
    exact_pf_state = np.array(exact_pf["state"], dtype=np.complex128, copy=False)
    ground_state = np.array(problem["ground_state"], dtype=np.complex128, copy=False)
    quantized_ground_state = np.array(quantized_ground["state"], dtype=np.complex128, copy=False)
    exact_pf_ground_state = np.array(exact_pf_ground["state"], dtype=np.complex128, copy=False)

    quantized_fidelity = fidelity(exact_state, quantized_state)
    exact_pf_fidelity = fidelity(exact_state, exact_pf_state)
    fidelity_floor = float(problem["chemical_accuracy_fidelity_floor"])
    passes_state_fidelity_diagnostic = quantized_fidelity >= fidelity_floor
    exact_ground_energy = float(problem["ground_state_energy_hartree"])
    identity_coefficient = float(problem["identity_coefficient"])
    quantized_ground_metrics = estimate_ground_state_energy_from_overlap(
        ground_state,
        quantized_ground_state,
        float(problem["target_time"]),
        exact_ground_energy,
        identity_coefficient,
    )
    exact_pf_ground_metrics = estimate_ground_state_energy_from_overlap(
        ground_state,
        exact_pf_ground_state,
        float(problem["target_time"]),
        exact_ground_energy,
        identity_coefficient,
    )
    ground_state_energy_error = quantized_ground_metrics["energy_error_hartree"]
    passes_accuracy_constraint = (
        ground_state_energy_error is not None
        and float(ground_state_energy_error) <= CHEMICAL_ACCURACY_HARTREE
    )

    hamiltonian = np.array(problem["hamiltonian"], dtype=np.complex128, copy=False)
    hf_energy = float(np.real(np.vdot(problem["initial_state"], hamiltonian @ problem["initial_state"])))
    exact_final_energy = float(np.real(np.vdot(exact_state, hamiltonian @ exact_state)))

    result = {
        "experiment": problem["experiment"],
        "molecule": molecule,
        "display_name": problem["display_name"],
        "script": str(script_path),
        "objective": "ground_state_phase_estimation",
        "score": score_from_ground_energy_error(
            int(quantized["t_gate_count"]),
            CHEMICAL_ACCURACY_HARTREE,
            ground_state_energy_error,
        ),
        "passes_accuracy_constraint": passes_accuracy_constraint,
        "passes_fidelity_constraint": passes_accuracy_constraint,
        "passes_state_fidelity_diagnostic": passes_state_fidelity_diagnostic,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_fidelity_floor": fidelity_floor,
        "target_time": float(problem["target_time"]),
        "time_sign": float(problem["time_sign"]),
        "ground_state_energy_hartree": exact_ground_energy,
        "ground_state_energy_estimate_hartree": quantized_ground_metrics["energy_estimate_hartree"],
        "ground_state_energy_error_hartree": ground_state_energy_error,
        "exact_pf1_ground_state_energy_estimate_hartree": exact_pf_ground_metrics["energy_estimate_hartree"],
        "exact_pf1_ground_state_energy_error_hartree": exact_pf_ground_metrics["energy_error_hartree"],
        "ground_state_phase_overlap_magnitude": quantized_ground_metrics["phase_overlap_magnitude"],
        "exact_pf1_ground_state_phase_overlap_magnitude": exact_pf_ground_metrics["phase_overlap_magnitude"],
        "accuracy_metric_name": "ground_state_energy_error_hartree",
        "accuracy_metric_value": ground_state_energy_error,
        "accuracy_threshold_name": "chemical_accuracy_hartree",
        "accuracy_threshold_value": CHEMICAL_ACCURACY_HARTREE,
        "fidelity": quantized_fidelity,
        "infidelity": 1.0 - quantized_fidelity,
        "exact_pf1_fidelity": exact_pf_fidelity,
        "exact_pf1_infidelity": 1.0 - exact_pf_fidelity,
        "precision_gain_over_exact_pf1": quantized_fidelity - exact_pf_fidelity,
        "t_gate_count": int(quantized["t_gate_count"]),
        "nonzero_rotation_count": int(quantized["nonzero_rotation_count"]),
        "skipped_rotation_count": int(quantized["skipped_rotation_count"]),
        "compilation_backend": problem["compilation_backend"],
        "synthesis_epsilon_histogram": quantized["synthesis_epsilon_histogram"],
        "unique_compilation_count": int(quantized["unique_compilation_count"]),
        "unique_compilations": quantized["unique_compilations"],
        "method_config": quantized["method_config"],
        "total_pauli_rotation_terms": int(problem["total_pauli_rotation_terms"]),
        "identity_coefficient": float(problem["identity_coefficient"]),
        "centered_hamiltonian_norm": float(problem["centered_hamiltonian_norm"]),
        "max_abs_total_alpha": float(problem["max_abs_total_alpha"]),
        "min_abs_total_alpha": float(problem["min_abs_total_alpha"]),
        "hf_energy": hf_energy,
        "exact_final_energy": exact_final_energy,
        "cas_label": problem["cas_label"],
        "basis": problem["basis"],
        "localization": problem["localization"],
        "evaluation_wall_seconds": time.perf_counter() - start,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one small-molecule PF1 precision policy.")
    parser.add_argument("--molecule", required=True, choices=tuple(sorted(name for name in get_spec.__globals__["MOLECULE_SPECS"])))
    parser.add_argument("--script", required=True, help="Path to the editable simple_pf.py file.")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = evaluate(args.molecule, Path(args.script).resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
