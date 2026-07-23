"""Exact compiled-unitary energy evaluation for small chemistry problems.

The scientific observable in this module is an eigenphase of the *compiled*
product-formula unitary.  PyGridSynth runs on the CPU.  Dense Hamiltonian and
unitary diagonalizations run through the selected NumPy-compatible array
backend, normally cuPyNumeric on an NVIDIA GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import warnings
from collections import Counter
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


CHEMICAL_ACCURACY_HARTREE = 1.0 / 627.5094740631
EXACT_GROUND_SUBSPACE_ATOL_HARTREE = 1.0e-9
TARGET_STATE_DEFINITION = "normalized_hf_projection_onto_exact_ground_eigenspace"
_PAULIS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}
_H = _PAULIS["X"] + _PAULIS["Z"]
_H /= math.sqrt(2.0)
_S_DAGGER = np.asarray([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)


def array_backend(name: str) -> tuple[str, Any]:
    """Return the requested dense-linear-algebra backend."""

    normalized = name.strip().lower()
    if normalized == "numpy":
        return "numpy", np
    if normalized != "cupynumeric":
        raise ValueError("backend must be 'numpy' or 'cupynumeric'")
    try:
        import cupynumeric as xp
    except ImportError as exc:
        raise RuntimeError(
            "cuPyNumeric is unavailable. Run this evaluator under Legate on an NVIDIA system."
        ) from exc
    return "cupynumeric", xp


def _fence(backend_name: str) -> None:
    if backend_name != "cupynumeric":
        return
    from legate.core import get_legate_runtime

    get_legate_runtime().issue_execution_fence(block=True)


def load_policy(path: Path) -> ModuleType:
    """Load one isolated product-formula policy module."""

    path = path.resolve()
    name = f"_hsopt_policy_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load policy: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for symbol in ("get_method_config", "choose_synthesis_epsilon"):
        if not callable(getattr(module, symbol, None)):
            raise AttributeError(f"{path.name} must define {symbol}()")
    return module


def load_problem(path: Path) -> dict[str, Any]:
    """Load and validate a compact Pauli Hamiltonian."""

    path = path.resolve()
    problem = json.loads(path.read_text(encoding="utf-8"))
    n_qubits = int(problem["n_qubits"])
    electrons = int(problem["electrons"])
    if n_qubits <= 0 or not 0 <= electrons <= n_qubits:
        raise ValueError(f"invalid problem dimensions in {path}")
    if n_qubits > 10:
        raise ValueError(
            f"{path.name} has {n_qubits} qubits; this exact dense evaluator is intentionally "
            "limited to 10 qubits"
        )
    terms = problem["terms"]
    if not isinstance(terms, list) or not terms:
        raise ValueError(f"{path.name} has no Pauli terms")
    for term in terms:
        label = str(term["label"])
        coefficient = float(term["coefficient"])
        if len(label) != n_qubits or set(label) - set("IXYZ"):
            raise ValueError(f"invalid Pauli label {label!r}")
        if not math.isfinite(coefficient):
            raise ValueError(f"non-finite coefficient for {label}")
        term["label"] = label
        term["coefficient"] = coefficient
        term["support_size"] = sum(char != "I" for char in label)
        term["x_count"] = label.count("X")
        term["y_count"] = label.count("Y")
        term["z_count"] = label.count("Z")
    problem["_path"] = str(path)
    return problem


def _pauli_left_action(label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return row destinations and phases for left multiplication by a Pauli."""

    source = np.arange(1 << len(label), dtype=np.int64)
    destination = source.copy()
    phase = np.ones(source.size, dtype=np.complex128)
    for qubit, pauli in enumerate(label):
        bit = 1 << (len(label) - qubit - 1)
        occupied = (source & bit) != 0
        if pauli == "X":
            destination ^= bit
        elif pauli == "Y":
            destination ^= bit
            phase *= np.where(occupied, -1.0j, 1.0j)
        elif pauli == "Z":
            phase *= np.where(occupied, -1.0, 1.0)
    return destination, phase


def _dense_hamiltonian(problem: Mapping[str, Any]) -> np.ndarray:
    dim = 1 << int(problem["n_qubits"])
    columns = np.arange(dim, dtype=np.int64)
    matrix = float(problem["identity_coefficient"]) * np.eye(dim, dtype=np.complex128)
    for term in problem["terms"]:
        rows, phase = _pauli_left_action(str(term["label"]))
        matrix[rows, columns] += float(term["coefficient"]) * phase
    return matrix


def _fixed_electron_hamiltonian(
    problem: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed-electron basis indices and dense Hamiltonian block."""

    n_qubits = int(problem["n_qubits"])
    electrons = int(problem["electrons"])
    indices_host = np.asarray(
        [index for index in range(1 << n_qubits) if index.bit_count() == electrons],
        dtype=np.int64,
    )
    full_host = _dense_hamiltonian(problem)
    sector_host = full_host[np.ix_(indices_host, indices_host)]
    return indices_host, sector_host


def _hartree_fock_state(problem: Mapping[str, Any], xp: Any) -> Any:
    n_qubits = int(problem["n_qubits"])
    basis_index = _hartree_fock_basis_index(problem)
    state = xp.zeros(1 << n_qubits, dtype=xp.complex128)
    state[basis_index] = 1.0
    return state


def _hartree_fock_basis_index(problem: Mapping[str, Any]) -> int:
    n_qubits = int(problem["n_qubits"])
    electrons = int(problem["electrons"])
    return sum(1 << (n_qubits - qubit - 1) for qubit in range(electrons))


def _hf_projected_ground_state(
    problem: Mapping[str, Any],
    *,
    indices_host: np.ndarray,
    eigenvalues: Any,
    eigenvectors: Any,
    xp: Any,
) -> tuple[float, Any, float, int]:
    """Return the unique ground component accessible from the HF state.

    A degenerate ground eigenspace has no unique eigenvector: different dense
    eigensolvers may return different orthonormal bases for the same subspace.
    Projecting the fixed HF state onto the complete ground eigenspace produces
    a basis-independent target for compiled-unitary eigenstate tracking.
    """

    eigenvalues_host = np.asarray(eigenvalues, dtype=np.float64)
    ground_energy = float(eigenvalues_host[0])
    ground_columns = np.flatnonzero(
        np.abs(eigenvalues_host - ground_energy) <= EXACT_GROUND_SUBSPACE_ATOL_HARTREE
    )
    if ground_columns.size == 0:
        raise RuntimeError("exact Hamiltonian diagonalization found no ground eigenspace")

    matches = np.flatnonzero(indices_host == _hartree_fock_basis_index(problem))
    if matches.size != 1:
        raise RuntimeError("Hartree--Fock determinant is absent from the fixed-electron sector")
    hf_sector = xp.zeros(indices_host.size, dtype=xp.complex128)
    hf_sector[int(matches[0])] = 1.0
    ground_basis = eigenvectors[:, ground_columns.tolist()]
    target = ground_basis @ (ground_basis.conj().T @ hf_sector)
    hf_ground_probability = float(np.asarray(xp.sum(xp.abs(target) ** 2)))
    if hf_ground_probability <= 1.0e-15:
        raise RuntimeError("Hartree--Fock state has zero overlap with the exact ground eigenspace")
    target /= math.sqrt(hf_ground_probability)
    return (
        ground_energy,
        target,
        hf_ground_probability,
        int(ground_columns.size),
    )


def _patch_pygridsynth() -> None:
    import pygridsynth.mymath
    import pygridsynth.ring

    def ntz_compat(value: int) -> int:
        value = int(value)
        return 0 if value == 0 else (value & -value).bit_length() - 1

    pygridsynth.mymath.ntz = ntz_compat
    pygridsynth.ring.ntz = ntz_compat


def _rz_matrix(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [np.exp(-0.5j * theta), 0.0],
            [0.0, np.exp(0.5j * theta)],
        ],
        dtype=np.complex128,
    )


def _operator_error(target: np.ndarray, approximate: np.ndarray) -> float:
    """Spectral-norm synthesis error after aligning an irrelevant global phase."""

    phase = np.vdot(target[:, 0], approximate[:, 0])
    phase = 1.0 + 0.0j if abs(phase) < 1.0e-30 else phase / abs(phase)
    return float(np.linalg.norm(target - approximate / phase, ord=2))


@lru_cache(maxsize=None)
def _compile_rz(theta: float, epsilon: float) -> dict[str, Any]:
    """Compile one Rz reproducibly on the CPU with PyGridSynth."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"synthesis epsilon must be positive, got {epsilon}")
    _patch_pygridsynth()
    from pygridsynth import gridsynth_gates
    from pygridsynth.unitary import DOmegaUnitary

    seed_bytes = hashlib.sha256(f"{theta:.17e}|{epsilon:.17e}".encode()).digest()[:8]
    previous_random_state = random.getstate()
    try:
        random.seed(int.from_bytes(seed_bytes, "big"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"pygridsynth is .*", category=UserWarning)
            gate_string = gridsynth_gates(theta, epsilon)
    finally:
        random.setstate(previous_random_state)

    exact_matrix = DOmegaUnitary.from_gates(gate_string).to_complex_matrix
    matrix = np.asarray(
        [[complex(exact_matrix[row, column]) for column in range(2)] for row in range(2)],
        dtype=np.complex128,
    )
    return {
        "gate_string": gate_string,
        "matrix": matrix,
        "t_count": gate_string.count("T"),
        "gate_count": len(gate_string),
        "operator_error": _operator_error(_rz_matrix(theta), matrix),
    }


def _basis_forward(term_pauli: str) -> np.ndarray:
    if term_pauli == "X":
        return _H
    if term_pauli == "Y":
        return _H @ _S_DAGGER
    return _PAULIS["I"]


def _identify_pauli(matrix: np.ndarray) -> tuple[complex, str]:
    coefficients = {
        label: complex(0.5 * np.trace(pauli.conj().T @ matrix)) for label, pauli in _PAULIS.items()
    }
    label = max(coefficients, key=lambda item: abs(coefficients[item]))
    coefficient = coefficients[label]
    if not np.allclose(matrix, coefficient * _PAULIS[label], atol=1.0e-12, rtol=0.0):
        raise ValueError("Clifford scaffold did not conjugate to a Pauli")
    return coefficient, label


def _conjugated_pivot_pauli(label: str, pivot_pauli: str) -> tuple[complex, str]:
    support = [index for index, char in enumerate(label) if char != "I"]
    if not support:
        raise ValueError("identity Pauli rotations are not compiled")
    pivot = support[-1]
    operator = ["I"] * len(label)
    if pivot_pauli in {"Y", "Z"}:
        for control in support[:-1]:
            operator[control] = "Z"
    operator[pivot] = pivot_pauli

    phase = 1.0 + 0.0j
    for qubit, operator_pauli in enumerate(operator):
        if operator_pauli == "I":
            continue
        basis = _basis_forward(label[qubit])
        local_phase, transformed = _identify_pauli(basis.conj().T @ _PAULIS[operator_pauli] @ basis)
        phase *= local_phase
        operator[qubit] = transformed
    return phase, "".join(operator)


def _apply_compiled_rotation(
    unitary: Any,
    *,
    label: str,
    compiled_rz: np.ndarray,
    actions: dict[str, tuple[Any, Any]],
    xp: Any,
) -> None:
    """Left-multiply a dense unitary by one compiled Pauli gadget."""

    decomposition = {
        pauli: complex(0.5 * np.trace(matrix.conj().T @ compiled_rz))
        for pauli, matrix in _PAULIS.items()
    }
    updated = decomposition["I"] * unitary
    for pivot_pauli in ("X", "Y", "Z"):
        coefficient = decomposition[pivot_pauli]
        if abs(coefficient) <= 1.0e-15:
            continue
        scaffold_phase, transformed_label = _conjugated_pivot_pauli(label, pivot_pauli)
        if transformed_label not in actions:
            rows, phases = _pauli_left_action(transformed_label)
            actions[transformed_label] = (
                xp.asarray(rows, dtype=xp.int64),
                xp.asarray(phases, dtype=xp.complex128),
            )
        rows, phases = actions[transformed_label]
        pauli_unitary = xp.empty_like(unitary)
        pauli_unitary[rows, :] = phases[:, None] * unitary
        updated = updated + coefficient * scaffold_phase * pauli_unitary
    unitary[:] = updated


def _ordered_terms(
    terms: Sequence[Mapping[str, Any]],
    policy: ModuleType,
    config: Any,
) -> list[dict[str, Any]]:
    copied = [dict(term) for term in terms]
    strategy = str(getattr(config, "term_order_strategy", "input"))
    if strategy == "input":
        return copied
    if strategy == "lexicographic":
        return sorted(copied, key=lambda term: term["label"])
    if strategy == "coefficient_desc":
        return sorted(copied, key=lambda term: (-abs(term["coefficient"]), term["label"]))
    if strategy == "coefficient_asc":
        return sorted(copied, key=lambda term: (abs(term["coefficient"]), term["label"]))
    if strategy == "support_desc":
        return sorted(
            copied,
            key=lambda term: (-term["support_size"], -abs(term["coefficient"]), term["label"]),
        )
    if strategy == "custom":
        ordered = policy.order_product_formula_terms([dict(term) for term in copied], config)
        if sorted(term["label"] for term in ordered) != sorted(term["label"] for term in copied):
            raise ValueError("custom term ordering must preserve every Pauli term exactly once")
        return [dict(term) for term in ordered]
    raise ValueError(f"unsupported term_order_strategy={strategy!r}")


def _step_rotations(
    terms: Sequence[Mapping[str, Any]],
    *,
    order: int,
    step_time: float,
) -> list[tuple[Mapping[str, Any], float]]:
    if order == 1:
        return [(term, float(term["coefficient"]) * step_time) for term in terms]
    if order == 2:
        outer = [(term, 0.5 * float(term["coefficient"]) * step_time) for term in terms[:-1]]
        return (
            outer
            + [(terms[-1], float(terms[-1]["coefficient"]) * step_time)]
            + list(reversed(outer))
        )
    raise ValueError("formula_order must be 1 or 2")


def _config_dict(config: Any) -> dict[str, Any]:
    values = asdict(config) if is_dataclass(config) else dict(vars(config))
    return {str(key): value for key, value in values.items()}


def _build_compiled_step(
    problem: Mapping[str, Any],
    policy: ModuleType,
    *,
    backend_name: str,
    xp: Any,
) -> tuple[Any, dict[str, Any]]:
    try:
        config = policy.get_method_config(problem)
    except TypeError:
        config = policy.get_method_config()
    order = int(config.formula_order)
    n_steps = int(config.n_steps)
    threshold = float(config.zero_angle_threshold)
    if n_steps <= 0 or threshold < 0.0:
        raise ValueError("n_steps must be positive and zero_angle_threshold nonnegative")
    terms = _ordered_terms(problem["terms"], policy, config)
    step_time = float(problem["target_time"]) / n_steps
    rotations = _step_rotations(terms, order=order, step_time=step_time)

    dim = 1 << int(problem["n_qubits"])
    unitary = xp.eye(dim, dtype=xp.complex128)
    actions: dict[str, tuple[Any, Any]] = {}
    t_count = 0
    skipped = 0
    epsilon_histogram: Counter[str] = Counter()
    compilations: dict[str, dict[str, Any]] = {}

    for term, alpha in rotations:
        if abs(alpha) <= threshold:
            skipped += 1
            continue
        epsilon = float(policy.choose_synthesis_epsilon(dict(term), alpha, 0, config))
        for step_index in range(1, n_steps):
            if (
                float(policy.choose_synthesis_epsilon(dict(term), alpha, step_index, config))
                != epsilon
            ):
                raise ValueError(
                    "exact step diagonalization requires a stationary synthesis policy; "
                    f"epsilon changed for {term['label']}"
                )
        theta = float(round(-2.0 * alpha, 15))
        compiled = _compile_rz(theta, epsilon)
        _apply_compiled_rotation(
            unitary,
            label=str(term["label"]),
            compiled_rz=compiled["matrix"],
            actions=actions,
            xp=xp,
        )
        _fence(backend_name)
        t_count += int(compiled["t_count"])
        epsilon_histogram[f"{epsilon:.3e}"] += n_steps
        key = f"{term['label']}|{theta:.17e}|{epsilon:.17e}"
        compilations[key] = {
            "label": term["label"],
            "theta": theta,
            "epsilon": epsilon,
            "gate_string_sha256": hashlib.sha256(compiled["gate_string"].encode()).hexdigest(),
            "gate_count": compiled["gate_count"],
            "t_count": compiled["t_count"],
            "operator_error": compiled["operator_error"],
        }

    return unitary, {
        "policy": _config_dict(config),
        "n_steps": n_steps,
        "formula_order": order,
        "step_time": step_time,
        "rotations_per_step": len(rotations),
        "scheduled_rotation_count": len(rotations) * n_steps,
        "skipped_rotation_count": skipped * n_steps,
        "one_step_t_count": t_count,
        "t_count": t_count * n_steps,
        "synthesis_epsilon_histogram": dict(sorted(epsilon_histogram.items())),
        "unique_compilations": sorted(
            compilations.values(),
            key=lambda row: (row["label"], row["theta"], row["epsilon"]),
        ),
    }


def _unwrap_energy(phase: float, time: float, reference: float) -> float:
    principal = phase / time
    period = 2.0 * math.pi / time
    return principal + round((reference - principal) / period) * period


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _serialize_state(state: Any) -> list[list[float]]:
    host = np.asarray(state, dtype=np.complex128)
    return [[float(value.real), float(value.imag)] for value in host]


def _reference_state(
    problem: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    xp: Any,
) -> tuple[float, Any]:
    if reference.get("observable") != "exact_unitary_ground_eigenphase":
        raise ValueError("reference is not an exact-unitary ground-eigenphase result")
    if reference.get("target_state_definition") != TARGET_STATE_DEFINITION:
        raise ValueError("reference does not use the HF-projected exact ground-eigenspace target")
    problem_path = Path(str(problem["_path"]))
    if reference.get("problem_sha256") != _sha256(problem_path):
        raise ValueError(f"exact reference does not match {problem_path}")
    encoded = reference.get("exact_ground_state")
    if not isinstance(encoded, list) or len(encoded) != 1 << int(problem["n_qubits"]):
        raise ValueError("exact reference has an invalid ground-state vector")
    state_host = np.asarray(
        [complex(float(value[0]), float(value[1])) for value in encoded],
        dtype=np.complex128,
    )
    state = xp.asarray(state_host, dtype=xp.complex128)
    state /= xp.linalg.norm(state)
    return float(reference["exact_unitary_ground_energy_hartree"]), state


def evaluate_exact_reference(
    problem_path: Path,
    *,
    backend: str = "cupynumeric",
) -> dict[str, Any]:
    """Construct and diagonalize ``U_exact = exp(+i H t)`` once."""

    problem = load_problem(problem_path)
    backend_name, xp = array_backend(backend)
    indices_host, sector_host = _fixed_electron_hamiltonian(problem)
    sector_hamiltonian = xp.asarray(sector_host, dtype=xp.complex128)
    h_eigenvalues, h_eigenvectors = xp.linalg.eigh(sector_hamiltonian)
    _fence(backend_name)
    (
        hamiltonian_energy,
        hamiltonian_target,
        hf_ground_probability,
        ground_subspace_dimension,
    ) = _hf_projected_ground_state(
        problem,
        indices_host=indices_host,
        eigenvalues=h_eigenvalues,
        eigenvectors=h_eigenvectors,
        xp=xp,
    )
    phases = xp.exp(1.0j * h_eigenvalues * float(problem["target_time"]))
    exact_unitary = (h_eigenvectors * phases[None, :]) @ h_eigenvectors.conj().T
    _fence(backend_name)
    eigenvalues, eigenvectors = xp.linalg.eig(exact_unitary)
    _fence(backend_name)
    eigenvectors /= xp.linalg.norm(eigenvectors, axis=0, keepdims=True)
    ground_overlaps = xp.abs(eigenvectors.conj().T @ hamiltonian_target) ** 2
    selected = int(np.asarray(xp.argmax(ground_overlaps)))
    selected_phase = float(np.asarray(xp.angle(eigenvalues[selected])))
    unitary_energy = _unwrap_energy(
        selected_phase,
        float(problem["target_time"]),
        hamiltonian_energy,
    )
    full_target_state = xp.zeros(1 << int(problem["n_qubits"]), dtype=xp.complex128)
    indices = xp.asarray(indices_host, dtype=xp.int64)
    full_target_state[indices] = hamiltonian_target
    consistency_error = abs(unitary_energy - hamiltonian_energy)
    if consistency_error > 1.0e-9:
        raise RuntimeError(
            "exact-unitary eigenphase does not reproduce the fixed-electron "
            f"Hamiltonian ground energy: {consistency_error}"
        )
    return {
        "status": "ok",
        "observable": "exact_unitary_ground_eigenphase",
        "unitary_definition": "U_exact = exp(+i H t)",
        "problem": problem["name"],
        "problem_path": str(Path(problem_path).resolve()),
        "problem_sha256": _sha256(Path(problem_path)),
        "backend": backend_name,
        "n_qubits": int(problem["n_qubits"]),
        "electrons": int(problem["electrons"]),
        "fixed_electron_sector_dimension": int(indices_host.size),
        "exact_ground_subspace_dimension": ground_subspace_dimension,
        "exact_ground_subspace_atol_hartree": EXACT_GROUND_SUBSPACE_ATOL_HARTREE,
        "target_state_definition": TARGET_STATE_DEFINITION,
        "target_time": float(problem["target_time"]),
        "hamiltonian_ground_energy_hartree": hamiltonian_energy,
        "exact_unitary_ground_energy_hartree": unitary_energy,
        "unitary_hamiltonian_consistency_error_hartree": consistency_error,
        "selected_eigen_index": selected,
        "selected_eigenphase_radians": selected_phase,
        "selected_ground_state_overlap": float(np.asarray(ground_overlaps[selected])),
        "qpe_success_probability_from_hf": hf_ground_probability,
        "max_eigenvalue_modulus_error": float(
            np.asarray(xp.max(xp.abs(xp.abs(eigenvalues) - 1.0)))
        ),
        "exact_ground_state": _serialize_state(full_target_state),
    }


def evaluate_exact_difference_reference(
    reaction_path: Path,
    *,
    backend: str = "cupynumeric",
) -> dict[str, Any]:
    """Compute immutable exact-unitary references for a signed energy difference."""

    reaction_path = reaction_path.resolve()
    reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
    species_results: dict[str, dict[str, Any]] = {}
    unitary_difference = 0.0
    hamiltonian_difference = 0.0
    for item in reaction["terms"]:
        species = str(item["species"])
        coefficient = float(item["coefficient"])
        problem_path = (reaction_path.parent / str(item["problem"])).resolve()
        result = evaluate_exact_reference(problem_path, backend=backend)
        species_results[species] = result
        unitary_difference += coefficient * result["exact_unitary_ground_energy_hartree"]
        hamiltonian_difference += coefficient * result["hamiltonian_ground_energy_hartree"]
    consistency_error = abs(unitary_difference - hamiltonian_difference)
    if consistency_error > 1.0e-9:
        raise RuntimeError(
            "exact-unitary signed difference does not reproduce the Hamiltonian "
            f"difference: {consistency_error}"
        )
    return {
        "status": "ok",
        "observable": "exact_unitary_signed_energy_difference",
        "unitary_definition": "U_exact = exp(+i H t) for each species",
        "reaction": reaction["name"],
        "reaction_path": str(reaction_path),
        "reaction_sha256": _sha256(reaction_path),
        "backend": backend,
        "hamiltonian_exact_difference_hartree": hamiltonian_difference,
        "exact_unitary_difference_hartree": unitary_difference,
        "unitary_hamiltonian_consistency_error_hartree": consistency_error,
        "species": species_results,
    }


def evaluate_molecule(
    problem_path: Path,
    policy_path: Path,
    *,
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
    reference_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact compiled-unitary ground-energy error for one molecule."""

    problem = load_problem(problem_path)
    policy = load_policy(policy_path)
    backend_name, xp = array_backend(backend)
    if reference_path is not None and reference_result is not None:
        raise ValueError("provide reference_path or reference_result, not both")
    if reference_path is not None:
        reference_result = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if reference_result is None:
        reference_result = evaluate_exact_reference(problem_path, backend=backend)
    exact_energy, exact_ground = _reference_state(problem, reference_result, xp=xp)
    hf_state = _hartree_fock_state(problem, xp)
    unitary_step, metrics = _build_compiled_step(
        problem,
        policy,
        backend_name=backend_name,
        xp=xp,
    )
    eigenvalues, eigenvectors = xp.linalg.eig(unitary_step)
    _fence(backend_name)
    eigenvectors /= xp.linalg.norm(eigenvectors, axis=0, keepdims=True)

    ground_overlaps = xp.abs(eigenvectors.conj().T @ exact_ground) ** 2
    selected = int(np.asarray(xp.argmax(ground_overlaps)))
    hf_probabilities = xp.abs(eigenvectors.conj().T @ hf_state) ** 2
    step_phase = float(np.asarray(xp.angle(eigenvalues[selected])))
    total_phase = metrics["n_steps"] * step_phase
    centered_reference = exact_energy - float(problem["identity_coefficient"])
    estimated_centered = _unwrap_energy(
        total_phase,
        float(problem["target_time"]),
        centered_reference,
    )
    estimated_energy = estimated_centered + float(problem["identity_coefficient"])
    signed_error = estimated_energy - exact_energy
    absolute_error = abs(signed_error)
    passes = absolute_error <= CHEMICAL_ACCURACY_HARTREE

    return {
        "status": "ok",
        "observable": "compiled_unitary_ground_eigenphase",
        "problem": problem["name"],
        "problem_path": str(Path(problem_path).resolve()),
        "policy_path": str(Path(policy_path).resolve()),
        "exact_reference_path": (
            str(Path(reference_path).resolve()) if reference_path is not None else None
        ),
        "exact_reference_problem_sha256": reference_result["problem_sha256"],
        "exact_ground_subspace_dimension": int(reference_result["exact_ground_subspace_dimension"]),
        "target_state_definition": reference_result["target_state_definition"],
        "backend": backend_name,
        "n_qubits": int(problem["n_qubits"]),
        "electrons": int(problem["electrons"]),
        "exact_ground_energy_hartree": exact_energy,
        "exact_unitary_ground_energy_hartree": exact_energy,
        "compiled_pf_energy_hartree": estimated_energy,
        "signed_energy_error_hartree": signed_error,
        "absolute_energy_error_hartree": absolute_error,
        "absolute_energy_error_mha": 1000.0 * absolute_error,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "passes_chemical_accuracy": passes,
        "selected_eigen_index": selected,
        "selected_ground_state_overlap": float(np.asarray(ground_overlaps[selected])),
        "qpe_success_probability_from_hf": float(np.asarray(hf_probabilities[selected])),
        "selected_step_eigenphase_radians": step_phase,
        "selected_total_eigenphase_radians": total_phase,
        "max_eigenvalue_modulus_error": float(
            np.asarray(xp.max(xp.abs(xp.abs(eigenvalues) - 1.0)))
        ),
        "t_count": int(metrics["t_count"]),
        "score": int(metrics["t_count"]) if passes else None,
        "evaluation_contract": (
            "minimize exact compiled T count subject to the compiled-unitary "
            "ground-eigenphase error being within chemical accuracy"
        ),
        **metrics,
    }


def evaluate_difference(
    reaction_path: Path,
    policy_path: Path,
    *,
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a signed energy difference from independently compiled unitaries."""

    reaction_path = reaction_path.resolve()
    reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
    reference = (
        json.loads(Path(reference_path).read_text(encoding="utf-8"))
        if reference_path is not None
        else evaluate_exact_difference_reference(reaction_path, backend=backend)
    )
    if reference.get("observable") != "exact_unitary_signed_energy_difference":
        raise ValueError("reference is not an exact-unitary signed-difference result")
    if reference.get("reaction_sha256") != _sha256(reaction_path):
        raise ValueError(f"exact reference does not match {reaction_path}")
    species_results: dict[str, dict[str, Any]] = {}
    exact_difference = 0.0
    compiled_difference = 0.0
    total_t_count = 0
    for item in reaction["terms"]:
        species = str(item["species"])
        coefficient = float(item["coefficient"])
        problem_path = (reaction_path.parent / str(item["problem"])).resolve()
        result = evaluate_molecule(
            problem_path,
            policy_path,
            backend=backend,
            reference_result=reference["species"][species],
        )
        species_results[species] = result
        exact_difference += coefficient * result["exact_ground_energy_hartree"]
        compiled_difference += coefficient * result["compiled_pf_energy_hartree"]
        total_t_count += int(result["t_count"])

    signed_error = compiled_difference - exact_difference
    absolute_error = abs(signed_error)
    passes = absolute_error <= CHEMICAL_ACCURACY_HARTREE
    return {
        "status": "ok",
        "observable": "compiled_unitary_signed_energy_difference",
        "reaction": reaction["name"],
        "reaction_path": str(reaction_path),
        "policy_path": str(Path(policy_path).resolve()),
        "exact_reference_path": (
            str(Path(reference_path).resolve()) if reference_path is not None else None
        ),
        "backend": backend,
        "exact_difference_hartree": exact_difference,
        "compiled_pf_difference_hartree": compiled_difference,
        "signed_difference_error_hartree": signed_error,
        "absolute_difference_error_hartree": absolute_error,
        "absolute_difference_error_mha": 1000.0 * absolute_error,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "passes_chemical_accuracy": passes,
        "t_count": total_t_count,
        "score": total_t_count if passes else None,
        "species": species_results,
        "evaluation_contract": (
            "minimize the sum of exact compiled species T counts subject to the "
            "signed compiled-unitary energy-difference error being within chemical accuracy"
        ),
    }
