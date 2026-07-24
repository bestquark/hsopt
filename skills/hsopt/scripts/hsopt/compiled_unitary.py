"""Exact compiled-unitary energy evaluation for small chemistry problems.

The scientific observable in this module is an eigenphase of the *compiled*
product-formula unitary.  PyGridSynth runs on the CPU.  Dense Hamiltonian and
unitary diagonalizations run through the selected NumPy-compatible array
backend, normally cuPyNumeric on an NVIDIA GPU.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import time
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


CHEMICAL_ACCURACY_HARTREE = 1.0 / 627.5094740631
EXACT_GROUND_SUBSPACE_ATOL_HARTREE = 1.0e-9
TARGET_ANCHOR_MIN_PROBABILITY = 1.0e-12
TARGET_STATE_DEFINITION = "normalized_anchor_projection_onto_exact_ground_eigenspace"
EVALUATION_CONTRACT_VERSION = "lowest_resolved_hf_qpe_peak_95pct_t_count_v1"
QPE_TARGET_CONFIDENCE = 0.95
# Match the established HF-reference spectral evaluator. This suppresses
# floating-point zero-weight eigenvectors; it is not a physical p_min or an
# energy-error bound. Resolved small weights pay their full repetition cost.
QPE_PEAK_WEIGHT_THRESHOLD = 1.0e-10
DENSE_EVALUATOR_MAX_QUBITS = 15
ROTATION_CHUNK_TARGET_BYTES = 1024**3
DEFERRED_ROTATION_TARGET_BYTES = 8 * 1024**3
MAX_ROTATIONS_BETWEEN_FENCES = 32
MAX_POLICY_ROTATIONS_PER_TERM = 16
_PAULIS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}
_H = _PAULIS["X"] + _PAULIS["Z"]
_H /= math.sqrt(2.0)
_S_DAGGER = np.asarray([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
_SAFE_POLICY_BUILTINS = {
    "__build_class__": __build_class__,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "object": object,
    "range": range,
    "reversed": reversed,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


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


def _rotation_fence_interval(dimension: int) -> int:
    """Bound deferred rotation temporaries while keeping the task queue fed."""

    matrix_bytes = dimension * dimension * np.dtype(np.complex128).itemsize
    worst_case_rotation_bytes = 4 * matrix_bytes
    return max(
        1,
        min(
            MAX_ROTATIONS_BETWEEN_FENCES,
            DEFERRED_ROTATION_TARGET_BYTES // worst_case_rotation_bytes,
        ),
    )


def validate_policy_source(path: Path) -> ast.Module:
    """Validate the pure-Python policy language before executing a candidate."""

    path = path.resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    product_formula_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ProductFormula"
    ]
    if len(product_formula_classes) != 1:
        raise ValueError(f"{path.name} must define exactly one ProductFormula class")

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if any(isinstance(child, (ast.Call, ast.Attribute)) for child in ast.walk(node)):
                raise ValueError(f"{path.name} top-level constants must be literal data")
            continue
        if isinstance(node, ast.ClassDef) and node.name == "ProductFormula":
            if node.bases or node.keywords or node.decorator_list:
                raise ValueError("ProductFormula cannot inherit from or decorate another class")
            continue
        raise ValueError(
            f"{path.name} may contain only a docstring, literal constants, "
            "and the ProductFormula class"
        )

    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
    )
    forbidden_names = {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "super",
        "type",
        "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"{path.name} contains unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(f"{path.name} uses forbidden name {node.id!r}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"{path.name} cannot access interpreter internals")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"{path.name} cannot access private attributes")

    policy_class = product_formula_classes[0]
    methods = {node.name for node in policy_class.body if isinstance(node, ast.FunctionDef)}
    required = {"__init__", "repetitions", "build_step", "synthesis_epsilon"}
    missing = sorted(required - methods)
    if missing:
        raise ValueError(f"ProductFormula is missing required methods: {missing}")
    return tree


def load_policy(path: Path) -> ModuleType:
    """Load one isolated ProductFormula policy with restricted builtins."""

    path = path.resolve()
    tree = validate_policy_source(path)
    name = f"_hsopt_policy_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    module = ModuleType(name)
    module.__dict__.update(
        {
            "__builtins__": dict(_SAFE_POLICY_BUILTINS),
            "__file__": str(path),
            "__name__": name,
        }
    )
    exec(compile(tree, str(path), "exec"), module.__dict__)
    product_formula = getattr(module, "ProductFormula", None)
    if not isinstance(product_formula, type):
        raise TypeError(f"{path.name} ProductFormula must be a class")
    for method in ("build_step", "synthesis_epsilon"):
        if not callable(getattr(product_formula, method, None)):
            raise AttributeError(f"{path.name} ProductFormula must define {method}()")
    return module


def load_problem(path: Path) -> dict[str, Any]:
    """Load and validate a compact Pauli Hamiltonian."""

    path = path.resolve()
    problem = json.loads(path.read_text(encoding="utf-8"))
    n_qubits = int(problem["n_qubits"])
    electrons = int(problem["electrons"])
    if n_qubits <= 0 or not 0 <= electrons <= n_qubits:
        raise ValueError(f"invalid problem dimensions in {path}")
    if n_qubits > DENSE_EVALUATOR_MAX_QUBITS:
        raise ValueError(
            f"{path.name} has {n_qubits} qubits; this exact dense evaluator is intentionally "
            f"limited to {DENSE_EVALUATOR_MAX_QUBITS} qubits"
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
    full_dimension = 1 << n_qubits
    indices_host = np.asarray(
        [index for index in range(full_dimension) if index.bit_count() == electrons],
        dtype=np.int64,
    )
    sector_dimension = indices_host.size
    sector_columns = np.arange(sector_dimension, dtype=np.int64)
    sector_positions = np.full(full_dimension, -1, dtype=np.int64)
    sector_positions[indices_host] = sector_columns
    sector_host = float(problem["identity_coefficient"]) * np.eye(
        sector_dimension,
        dtype=np.complex128,
    )
    for term in problem["terms"]:
        rows, phase = _pauli_left_action(str(term["label"]))
        destination = sector_positions[rows[indices_host]]
        retained = destination >= 0
        sector_host[destination[retained], sector_columns[retained]] += (
            float(term["coefficient"]) * phase[indices_host[retained]]
        )
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


def _anchored_ground_state(
    problem: Mapping[str, Any],
    *,
    indices_host: np.ndarray,
    eigenvalues: Any,
    eigenvectors: Any,
    xp: Any,
) -> tuple[float, Any, float, float, int, str, int]:
    """Return a deterministic exact-ground target and its diagnostics.

    A degenerate ground eigenspace has no unique eigenvector: different dense
    eigensolvers may return different orthonormal bases for the same subspace.
    Projecting a fixed computational-basis anchor onto the complete ground
    eigenspace produces a basis-independent target for compiled-unitary
    eigenstate tracking. The HF determinant is preferred; if it has no
    resolvable ground support, the lowest-index fixed-electron determinant with
    resolvable support is used instead.
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
    hf_position = int(matches[0])
    ground_basis = eigenvectors[:, ground_columns.tolist()]
    determinant_probabilities = np.asarray(
        xp.sum(xp.abs(ground_basis) ** 2, axis=1),
        dtype=np.float64,
    )
    hf_ground_probability = float(determinant_probabilities[hf_position])
    if hf_ground_probability >= TARGET_ANCHOR_MIN_PROBABILITY:
        anchor_position = hf_position
        anchor_kind = "hartree_fock"
    else:
        resolved = np.flatnonzero(determinant_probabilities >= TARGET_ANCHOR_MIN_PROBABILITY)
        if resolved.size == 0:
            raise RuntimeError(
                "exact ground eigenspace has no resolvable fixed-electron determinant anchor"
            )
        anchor_position = int(resolved[0])
        anchor_kind = "lowest_resolved_fixed_electron_determinant"

    anchor_probability = float(determinant_probabilities[anchor_position])
    target = ground_basis @ ground_basis[anchor_position, :].conj()
    target /= math.sqrt(anchor_probability)
    hf_target_probability = float(np.asarray(xp.abs(target[hf_position]) ** 2))
    return (
        ground_energy,
        target,
        hf_ground_probability,
        hf_target_probability,
        int(ground_columns.size),
        anchor_kind,
        int(indices_host[anchor_position]),
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"pygridsynth is .*", category=UserWarning)
        gate_string = gridsynth_gates(theta, epsilon)

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
    workspace: Any,
    *,
    label: str,
    compiled_rz: np.ndarray,
    xp: Any,
) -> tuple[Any, Any]:
    """Left-multiply using one reusable full-matrix workspace.

    ``unitary`` remains unchanged while ``workspace`` is filled.  The caller
    swaps the two arrays after every rotation, so no matrix-sized copy is
    needed.  Pauli permutations are accumulated in bounded row chunks to avoid
    hidden full-matrix expression temporaries.
    """

    decomposition = {
        pauli: complex(0.5 * np.trace(matrix.conj().T @ compiled_rz))
        for pauli, matrix in _PAULIS.items()
    }
    dimension = int(unitary.shape[0])
    row_chunk = max(
        1,
        min(
            dimension,
            ROTATION_CHUNK_TARGET_BYTES // (dimension * np.dtype(np.complex128).itemsize),
        ),
    )
    identity_coefficient = decomposition["I"]
    if abs(identity_coefficient) <= 1.0e-15:
        workspace.fill(0.0)
    else:
        for start in range(0, dimension, row_chunk):
            stop = min(start + row_chunk, dimension)
            workspace[start:stop, :] = identity_coefficient * unitary[start:stop, :]

    for pivot_pauli in ("X", "Y", "Z"):
        coefficient = decomposition[pivot_pauli]
        if abs(coefficient) <= 1.0e-15:
            continue
        scaffold_phase, transformed_label = _conjugated_pivot_pauli(label, pivot_pauli)
        source_for_destination, source_phases = _pauli_left_action(transformed_label)
        source_indices = xp.asarray(source_for_destination, dtype=xp.int64)
        destination_phases = xp.asarray(
            source_phases[source_for_destination],
            dtype=xp.complex128,
        )
        scale = coefficient * scaffold_phase
        for start in range(0, dimension, row_chunk):
            stop = min(start + row_chunk, dimension)
            workspace[start:stop, :] += (
                scale
                * destination_phases[start:stop, None]
                * xp.take(unitary, source_indices[start:stop], axis=0)
            )
    return workspace, unitary


def _step_rotations(
    terms: Sequence[Mapping[str, Any]],
    *,
    order: int,
    step_time: float,
) -> list[tuple[Mapping[str, Any], float]]:
    """Build a standard PF1/PF2 step for sealed baseline calibration only."""

    if order == 1:
        return [(term, float(term["coefficient"]) * step_time) for term in terms]
    if order == 2:
        outer = [(term, 0.5 * float(term["coefficient"]) * step_time) for term in terms[:-1]]
        return (
            outer
            + [(terms[-1], float(terms[-1]["coefficient"]) * step_time)]
            + list(reversed(outer))
        )
    raise ValueError("sealed baseline formula_order must be 1 or 2")


def _infer_term_order(
    terms: Sequence[Mapping[str, Any]],
    traversal: Sequence[Mapping[str, Any]],
) -> str:
    labels = [str(term["label"]) for term in traversal]
    strategies = {
        "lexicographic": [
            str(term["label"]) for term in sorted(terms, key=lambda term: term["label"])
        ],
        "input": [str(term["label"]) for term in terms],
        "coefficient_desc": [
            str(term["label"])
            for term in sorted(
                terms,
                key=lambda term: (-abs(float(term["coefficient"])), term["label"]),
            )
        ],
        "coefficient_asc": [
            str(term["label"])
            for term in sorted(
                terms,
                key=lambda term: (abs(float(term["coefficient"])), term["label"]),
            )
        ],
        "support_desc": [
            str(term["label"])
            for term in sorted(
                terms,
                key=lambda term: (
                    -int(term["support_size"]),
                    -abs(float(term["coefficient"])),
                    term["label"],
                ),
            )
        ],
    }
    for name, ordered_labels in strategies.items():
        if labels == ordered_labels:
            return name
    return "custom"


def _classify_rotation_schedule(
    terms: Sequence[Mapping[str, Any]],
    rotations: Sequence[tuple[Mapping[str, Any], float]],
    *,
    step_time: float,
) -> dict[str, Any]:
    """Infer standard PF1/PF2 structure; arbitrary schedules remain valid."""

    term_count = len(terms)
    labels = [str(term["label"]) for term, _angle in rotations]
    angles = [float(angle) for _term, angle in rotations]
    if len(rotations) == term_count and len(set(labels)) == term_count:
        traversal = [term for term, _angle in rotations]
        expected = [float(term["coefficient"]) * step_time for term in traversal]
        if all(
            math.isclose(actual, target, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for actual, target in zip(angles, expected, strict=True)
        ):
            return {
                "formula_name": "first_order_lie_trotter",
                "formula_order": 1,
                "term_order_strategy": _infer_term_order(terms, traversal),
            }

    if len(rotations) == 2 * term_count - 1 and labels == list(reversed(labels)):
        traversal = [term for term, _angle in rotations[:term_count]]
        traversal_labels = [str(term["label"]) for term in traversal]
        if len(set(traversal_labels)) == term_count:
            forward_angles = angles[:term_count]
            expected = [
                (0.5 if index < term_count - 1 else 1.0) * float(term["coefficient"]) * step_time
                for index, term in enumerate(traversal)
            ]
            if all(
                math.isclose(actual, target, rel_tol=1.0e-12, abs_tol=1.0e-15)
                for actual, target in zip(forward_angles, expected, strict=True)
            ) and all(
                math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)
                for left, right in zip(
                    angles[term_count:],
                    reversed(angles[: term_count - 1]),
                    strict=True,
                )
            ):
                return {
                    "formula_name": "symmetric_second_order_trotter_suzuki",
                    "formula_order": 2,
                    "term_order_strategy": _infer_term_order(terms, traversal),
                }

    return {
        "formula_name": "custom_stationary_product_formula",
        "formula_order": None,
        "term_order_strategy": "custom",
    }


def _product_formula_schedule(
    problem: Mapping[str, Any],
    policy: ModuleType,
) -> tuple[Any, list[tuple[dict[str, Any], float]], dict[str, Any]]:
    """Instantiate the candidate and validate its stationary rotation schedule."""

    candidate_problem = {
        key: value
        for key, value in problem.items()
        if key not in {"_path", "terms"}
    }
    candidate_problem["terms"] = [dict(term) for term in problem["terms"]]
    product_formula = policy.ProductFormula(candidate_problem)
    repetitions_method = getattr(product_formula, "repetitions", None)
    if not callable(repetitions_method):
        raise ValueError("ProductFormula must define repetitions()")
    n_steps = int(repetitions_method())
    if n_steps <= 0:
        raise ValueError("ProductFormula.repetitions() must be positive")
    step_time = float(problem["target_time"]) / n_steps
    trusted_terms = {str(term["label"]): dict(term) for term in problem["terms"]}
    raw_rotations = product_formula.build_step(
        [dict(term) for term in problem["terms"]],
        step_time,
    )
    if not isinstance(raw_rotations, (list, tuple)) or not raw_rotations:
        raise ValueError("ProductFormula.build_step() must return a nonempty list or tuple")
    if len(raw_rotations) > MAX_POLICY_ROTATIONS_PER_TERM * len(trusted_terms):
        raise ValueError("ProductFormula.build_step() returned too many rotations")

    rotations: list[tuple[dict[str, Any], float]] = []
    for index, item in enumerate(raw_rotations):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"rotation {index} must be a (term, angle) pair")
        raw_term, raw_angle = item
        if isinstance(raw_term, Mapping):
            label = str(raw_term.get("label", ""))
        else:
            label = str(raw_term)
        if label not in trusted_terms:
            raise ValueError(f"rotation {index} references unknown Pauli term {label!r}")
        angle = float(raw_angle)
        if not math.isfinite(angle):
            raise ValueError(f"rotation {index} has a non-finite angle")
        rotations.append((dict(trusted_terms[label]), angle))

    classification = _classify_rotation_schedule(
        list(problem["terms"]),
        rotations,
        step_time=step_time,
    )
    schedule_payload = [
        {"label": str(term["label"]), "angle": float(angle)} for term, angle in rotations
    ]
    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        product_formula,
        rotations,
        {
            "policy": {
                "class": "ProductFormula",
                "formula_name": classification["formula_name"],
                "formula_order": classification["formula_order"],
                "n_steps": n_steps,
                "term_order_strategy": classification["term_order_strategy"],
                "rotation_schedule_sha256": schedule_sha256,
            },
            "n_steps": n_steps,
            "formula_name": classification["formula_name"],
            "formula_order": classification["formula_order"],
            "term_order_strategy": classification["term_order_strategy"],
            "step_time": step_time,
            "rotations_per_step": len(rotations),
            "scheduled_rotation_count": len(rotations) * n_steps,
            "rotation_schedule_sha256": schedule_sha256,
        },
    )


def _build_compiled_step(
    problem: Mapping[str, Any],
    policy: ModuleType,
    *,
    backend_name: str,
    xp: Any,
) -> tuple[Any, dict[str, Any]]:
    product_formula, rotations, schedule_metrics = _product_formula_schedule(problem, policy)
    n_steps = int(schedule_metrics["n_steps"])

    dim = 1 << int(problem["n_qubits"])
    fence_interval = _rotation_fence_interval(dim)
    unitary = xp.eye(dim, dtype=xp.complex128)
    workspace = xp.empty_like(unitary)
    t_count = 0
    skipped = 0
    epsilon_histogram: Counter[str] = Counter()
    epsilon_values: set[float] = set()
    compilations: dict[str, dict[str, Any]] = {}

    applied_rotations = 0
    for term, alpha in rotations:
        epsilon = float(product_formula.synthesis_epsilon(dict(term), alpha))
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("ProductFormula.synthesis_epsilon() must return a value in (0, 1)")
        theta = float(round(-2.0 * alpha, 15))
        compiled = _compile_rz(theta, epsilon)
        unitary, workspace = _apply_compiled_rotation(
            unitary,
            workspace,
            label=str(term["label"]),
            compiled_rz=compiled["matrix"],
            xp=xp,
        )
        applied_rotations += 1
        if applied_rotations % fence_interval == 0:
            _fence(backend_name)
        t_count += int(compiled["t_count"])
        epsilon_histogram[f"{epsilon:.3e}"] += n_steps
        epsilon_values.add(epsilon)
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

    _fence(backend_name)
    del workspace
    policy_metrics = dict(schedule_metrics["policy"])
    policy_metrics.update(
        {
            "synthesis_policy": ("uniform" if len(epsilon_values) == 1 else "rotation_dependent"),
            "uniform_epsilon": (next(iter(epsilon_values)) if len(epsilon_values) == 1 else None),
        }
    )
    return unitary, {
        **schedule_metrics,
        "policy": policy_metrics,
        "skipped_rotation_count": skipped * n_steps,
        "one_step_t_count": t_count,
        "t_count": t_count * n_steps,
        "dense_matrix_dimension": dim,
        "dense_matrix_bytes": dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_workspace_bytes": dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_chunk_target_bytes": ROTATION_CHUNK_TARGET_BYTES,
        "rotation_fence_interval": fence_interval,
        "synthesis_epsilon_histogram": dict(sorted(epsilon_histogram.items())),
        "unique_compilations": sorted(
            compilations.values(),
            key=lambda row: (row["label"], row["theta"], row["epsilon"]),
        ),
    }


def _build_exact_product_formula_step(
    problem: Mapping[str, Any],
    policy: ModuleType,
    *,
    backend_name: str,
    xp: Any,
) -> tuple[Any, dict[str, Any]]:
    """Construct the same candidate schedule with ideal Pauli rotations."""

    _product_formula, rotations, schedule_metrics = _product_formula_schedule(problem, policy)
    dim = 1 << int(problem["n_qubits"])
    fence_interval = _rotation_fence_interval(dim)
    unitary = xp.eye(dim, dtype=xp.complex128)
    workspace = xp.empty_like(unitary)
    for index, (term, alpha) in enumerate(rotations, start=1):
        exact_rz = _rz_matrix(float(round(-2.0 * alpha, 15)))
        unitary, workspace = _apply_compiled_rotation(
            unitary,
            workspace,
            label=str(term["label"]),
            compiled_rz=exact_rz,
            xp=xp,
        )
        if index % fence_interval == 0:
            _fence(backend_name)
    _fence(backend_name)
    del workspace
    return unitary, {
        **schedule_metrics,
        "rotation_implementation": "ideal_pauli_rotations",
        "dense_matrix_dimension": dim,
        "dense_matrix_bytes": dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_workspace_bytes": dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_fence_interval": fence_interval,
    }


def _unwrap_energy(phase: float, time: float, reference: float) -> float:
    principal = phase / time
    period = 2.0 * math.pi / time
    return principal + round((reference - principal) / period) * period


def _qpe_repetitions(
    success_probability: float,
    *,
    target_confidence: float = QPE_TARGET_CONFIDENCE,
) -> int:
    """Return the attempts needed to observe a peak at the target confidence."""

    probability = min(1.0, max(0.0, float(success_probability)))
    confidence = float(target_confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("target_confidence must be strictly between zero and one")
    if probability <= 0.0:
        raise ValueError("QPE peak probability must be positive")
    if probability >= 1.0:
        return 1
    return max(
        1,
        int(math.ceil(math.log1p(-confidence) / math.log1p(-probability))),
    )


def _qpe_achieved_confidence(success_probability: float, repetitions: int) -> float:
    probability = min(1.0, max(0.0, float(success_probability)))
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if probability >= 1.0:
        return 1.0
    return -math.expm1(repetitions * math.log1p(-probability))


def _select_lowest_hf_accessible_peak(
    step_phases: Sequence[float],
    hf_probabilities: Sequence[float],
    *,
    n_steps: int,
    target_time: float,
    identity_coefficient: float,
    exact_energy: float,
) -> dict[str, Any]:
    """Select the lowest resolved PF energy that ideal QPE can sample from HF."""

    phases = np.asarray(step_phases, dtype=np.float64)
    probabilities = np.asarray(hf_probabilities, dtype=np.float64)
    if phases.ndim != 1 or probabilities.shape != phases.shape or phases.size == 0:
        raise ValueError("PF eigenphases and HF probabilities must be nonempty 1D arrays")
    if n_steps <= 0 or target_time <= 0.0:
        raise ValueError("n_steps and target_time must be positive")

    centered_reference = exact_energy - identity_coefficient
    total_phases = int(n_steps) * phases
    energies = np.asarray(
        [
            _unwrap_energy(float(phase), target_time, centered_reference) + identity_coefficient
            for phase in total_phases
        ],
        dtype=np.float64,
    )
    accessible = np.flatnonzero(probabilities >= QPE_PEAK_WEIGHT_THRESHOLD)
    if accessible.size == 0:
        raise RuntimeError(
            "compiled PF unitary has no numerically resolved eigenphase with HF support"
        )
    selected = min(
        (int(index) for index in accessible),
        key=lambda index: (float(energies[index]), -float(probabilities[index]), index),
    )
    selected_probability = min(1.0, max(0.0, float(probabilities[selected])))
    return {
        "selected_eigen_index": selected,
        "selected_step_eigenphase_radians": float(phases[selected]),
        "selected_total_eigenphase_radians": float(total_phases[selected]),
        "compiled_pf_energy_hartree": float(energies[selected]),
        "qpe_success_probability_from_hf": selected_probability,
        "hf_accessible_peak_count": int(accessible.size),
        "hf_accessible_probability_mass": float(
            min(1.0, max(0.0, probabilities[accessible].sum()))
        ),
        "phase_unwrap_energy_window_hartree": [
            float(exact_energy - math.pi / target_time),
            float(exact_energy + math.pi / target_time),
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _serialize_state(state: Any) -> list[list[float]]:
    host = np.asarray(state, dtype=np.complex128)
    return [[float(value.real), float(value.imag)] for value in host]


def _reference_energy(
    problem: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> float:
    """Validate a cached exact reference without loading its ground-state vector."""

    if reference.get("observable") != "exact_unitary_ground_eigenphase":
        raise ValueError("reference is not an exact-unitary ground-eigenphase result")
    if reference.get("evaluation_contract_version") != EVALUATION_CONTRACT_VERSION:
        raise ValueError("reference uses a different evaluator contract")
    problem_path = Path(str(problem["_path"]))
    if reference.get("problem_sha256") != _sha256(problem_path):
        raise ValueError(f"exact reference does not match {problem_path}")
    return float(reference["exact_unitary_ground_energy_hartree"])


def evaluate_exact_reference(
    problem_path: Path,
    *,
    backend: str = "cupynumeric",
) -> dict[str, Any]:
    """Construct and diagonalize ``U_exact = exp(+i H t)`` once."""

    evaluation_started = time.perf_counter()
    problem = load_problem(problem_path)
    backend_name, xp = array_backend(backend)

    sector_started = time.perf_counter()
    indices_host, sector_host = _fixed_electron_hamiltonian(problem)
    sector_hamiltonian = xp.asarray(sector_host, dtype=xp.complex128)
    _fence(backend_name)
    sector_build_seconds = time.perf_counter() - sector_started

    hamiltonian_eigensolve_started = time.perf_counter()
    h_eigenvalues, h_eigenvectors = xp.linalg.eigh(sector_hamiltonian)
    _fence(backend_name)
    hamiltonian_eigensolve_seconds = time.perf_counter() - hamiltonian_eigensolve_started
    (
        hamiltonian_energy,
        hamiltonian_target,
        hf_ground_probability,
        hf_target_probability,
        ground_subspace_dimension,
        target_anchor_kind,
        target_anchor_basis_index,
    ) = _anchored_ground_state(
        problem,
        indices_host=indices_host,
        eigenvalues=h_eigenvalues,
        eigenvectors=h_eigenvectors,
        xp=xp,
    )
    exact_unitary_started = time.perf_counter()
    phases = xp.exp(1.0j * h_eigenvalues * float(problem["target_time"]))
    exact_unitary = (h_eigenvectors * phases[None, :]) @ h_eigenvectors.conj().T
    _fence(backend_name)
    exact_unitary_build_seconds = time.perf_counter() - exact_unitary_started

    unitary_eigensolve_started = time.perf_counter()
    eigenvalues, eigenvectors = xp.linalg.eig(exact_unitary)
    _fence(backend_name)
    unitary_eigensolve_seconds = time.perf_counter() - unitary_eigensolve_started

    postprocess_started = time.perf_counter()
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
    selected_overlap = float(np.asarray(ground_overlaps[selected]))
    max_modulus_error = float(np.asarray(xp.max(xp.abs(xp.abs(eigenvalues) - 1.0))))
    serialized_target_state = _serialize_state(full_target_state)
    _fence(backend_name)
    postprocess_seconds = time.perf_counter() - postprocess_started
    consistency_error = abs(unitary_energy - hamiltonian_energy)
    if consistency_error > 1.0e-9:
        raise RuntimeError(
            "exact-unitary eigenphase does not reproduce the fixed-electron "
            f"Hamiltonian ground energy: {consistency_error}"
        )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    return {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
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
        "target_anchor_min_probability": TARGET_ANCHOR_MIN_PROBABILITY,
        "target_anchor_kind": target_anchor_kind,
        "target_anchor_basis_index": target_anchor_basis_index,
        "target_time": float(problem["target_time"]),
        "hamiltonian_ground_energy_hartree": hamiltonian_energy,
        "exact_unitary_ground_energy_hartree": unitary_energy,
        "unitary_hamiltonian_consistency_error_hartree": consistency_error,
        "selected_eigen_index": selected,
        "selected_eigenphase_radians": selected_phase,
        "selected_ground_state_overlap": selected_overlap,
        "hf_ground_subspace_probability": hf_ground_probability,
        "qpe_success_probability_from_hf": hf_target_probability,
        "max_eigenvalue_modulus_error": max_modulus_error,
        "exact_ground_state": serialized_target_state,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "timing_seconds": {
            "fixed_electron_hamiltonian_build": sector_build_seconds,
            "hamiltonian_eigensolve": hamiltonian_eigensolve_seconds,
            "exact_unitary_build": exact_unitary_build_seconds,
            "exact_unitary_eigensolve": unitary_eigensolve_seconds,
            "postprocess": postprocess_seconds,
            "total": evaluation_wall_seconds,
        },
    }


def assemble_exact_difference_reference(
    reaction_path: Path,
    species_references: Mapping[str, Mapping[str, Any]],
    *,
    backend: str,
) -> dict[str, Any]:
    """Assemble a reaction reference from already-computed species references."""

    assembly_started = time.perf_counter()
    reaction_path = reaction_path.resolve()
    reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
    species_results: dict[str, dict[str, Any]] = {}
    unitary_difference = 0.0
    hamiltonian_difference = 0.0
    for item in reaction["terms"]:
        species = str(item["species"])
        coefficient = float(item["coefficient"])
        if species not in species_references:
            raise ValueError(f"exact reference is missing reaction species {species!r}")
        problem_path = (reaction_path.parent / str(item["problem"])).resolve()
        problem = load_problem(problem_path)
        result = dict(species_references[species])
        unitary_energy = _reference_energy(problem, result)
        hamiltonian_energy = float(result["hamiltonian_ground_energy_hartree"])
        species_results[species] = result
        unitary_difference += coefficient * unitary_energy
        hamiltonian_difference += coefficient * hamiltonian_energy
    consistency_error = abs(unitary_difference - hamiltonian_difference)
    if consistency_error > 1.0e-9:
        raise RuntimeError(
            "exact-unitary signed difference does not reproduce the Hamiltonian "
            f"difference: {consistency_error}"
        )
    assembly_seconds = time.perf_counter() - assembly_started
    return {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
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
        "evaluation_wall_seconds": assembly_seconds,
        "timing_seconds": {
            "species_exact_reference_total": sum(
                float(result["evaluation_wall_seconds"]) for result in species_results.values()
            ),
            "assembly": assembly_seconds,
            "total": assembly_seconds,
        },
        "reference_provenance": "precomputed_unique_hamiltonian_references",
    }


def evaluate_exact_difference_reference(
    reaction_path: Path,
    *,
    backend: str = "cupynumeric",
) -> dict[str, Any]:
    """Compute immutable exact-unitary references for a signed energy difference."""

    evaluation_started = time.perf_counter()
    reaction_path = reaction_path.resolve()
    reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
    species_results: dict[str, dict[str, Any]] = {}
    for item in reaction["terms"]:
        species = str(item["species"])
        if species not in species_results:
            problem_path = (reaction_path.parent / str(item["problem"])).resolve()
            species_results[species] = evaluate_exact_reference(
                problem_path,
                backend=backend,
            )
    assembled = assemble_exact_difference_reference(
        reaction_path,
        species_results,
        backend=backend,
    )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    assembled["evaluation_wall_seconds"] = evaluation_wall_seconds
    assembled["timing_seconds"]["assembly"] = evaluation_wall_seconds - float(
        assembled["timing_seconds"]["species_exact_reference_total"]
    )
    assembled["timing_seconds"]["total"] = evaluation_wall_seconds
    assembled["reference_provenance"] = "computed_by_reaction_reference_evaluator"
    return assembled


def evaluate_molecule(
    problem_path: Path,
    policy_path: Path,
    *,
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
    reference_result: Mapping[str, Any] | None = None,
    target_confidence: float = QPE_TARGET_CONFIDENCE,
) -> dict[str, Any]:
    """Evaluate the lowest HF-accessible compiled-PF energy for one molecule."""

    evaluation_started = time.perf_counter()
    problem = load_problem(problem_path)
    policy = load_policy(policy_path)
    backend_name, xp = array_backend(backend)
    if reference_path is not None and reference_result is not None:
        raise ValueError("provide reference_path or reference_result, not both")
    if reference_path is not None:
        reference_result = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if reference_result is None:
        reference_result = evaluate_exact_reference(problem_path, backend=backend)

    reference_started = time.perf_counter()
    exact_energy = _reference_energy(problem, reference_result)
    hf_state = _hartree_fock_state(problem, xp)
    _fence(backend_name)
    reference_load_seconds = time.perf_counter() - reference_started

    compiled_unitary_started = time.perf_counter()
    unitary_step, metrics = _build_compiled_step(
        problem,
        policy,
        backend_name=backend_name,
        xp=xp,
    )
    compiled_unitary_build_seconds = time.perf_counter() - compiled_unitary_started

    eigensolve_started = time.perf_counter()
    eigenvalues, eigenvectors = xp.linalg.eig(unitary_step)
    _fence(backend_name)
    eigenvectors /= xp.linalg.norm(eigenvectors, axis=0, keepdims=True)
    _fence(backend_name)
    compiled_unitary_eigensolve_seconds = time.perf_counter() - eigensolve_started

    postprocess_started = time.perf_counter()
    hf_probabilities = np.asarray(
        xp.abs(eigenvectors.conj().T @ hf_state) ** 2,
        dtype=np.float64,
    )
    peak = _select_lowest_hf_accessible_peak(
        np.asarray(xp.angle(eigenvalues), dtype=np.float64),
        hf_probabilities,
        n_steps=int(metrics["n_steps"]),
        target_time=float(problem["target_time"]),
        identity_coefficient=float(problem["identity_coefficient"]),
        exact_energy=exact_energy,
    )
    estimated_energy = float(peak["compiled_pf_energy_hartree"])
    signed_error = estimated_energy - exact_energy
    absolute_error = abs(signed_error)
    passes = absolute_error <= CHEMICAL_ACCURACY_HARTREE
    qpe_success_probability = float(peak["qpe_success_probability_from_hf"])
    qpe_repetitions = _qpe_repetitions(
        qpe_success_probability,
        target_confidence=target_confidence,
    )
    qpe_achieved_confidence = _qpe_achieved_confidence(
        qpe_success_probability,
        qpe_repetitions,
    )
    per_attempt_t_count = int(metrics["t_count"])
    confidence_adjusted_t_count = qpe_repetitions * per_attempt_t_count
    max_modulus_error = float(np.asarray(xp.max(xp.abs(xp.abs(eigenvalues) - 1.0))))
    _fence(backend_name)
    postprocess_seconds = time.perf_counter() - postprocess_started
    evaluation_wall_seconds = time.perf_counter() - evaluation_started

    return {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "observable": "compiled_unitary_lowest_hf_accessible_eigenphase",
        "problem": problem["name"],
        "problem_path": str(Path(problem_path).resolve()),
        "policy_path": str(Path(policy_path).resolve()),
        "exact_reference_path": (
            str(Path(reference_path).resolve()) if reference_path is not None else None
        ),
        "exact_reference_problem_sha256": reference_result["problem_sha256"],
        "exact_ground_subspace_dimension": int(reference_result["exact_ground_subspace_dimension"]),
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
        **peak,
        "pf_peak_selection": "lowest_energy_peak_with_resolved_hf_weight",
        "qpe_peak_weight_threshold": QPE_PEAK_WEIGHT_THRESHOLD,
        "qpe_target_confidence": float(target_confidence),
        "qpe_achieved_confidence": qpe_achieved_confidence,
        "qpe_repetitions": qpe_repetitions,
        "max_eigenvalue_modulus_error": max_modulus_error,
        "per_attempt_t_count": per_attempt_t_count,
        "confidence_adjusted_t_count": confidence_adjusted_t_count,
        "t_count": per_attempt_t_count,
        "score": confidence_adjusted_t_count if passes else None,
        "evaluation_contract": (
            "minimize 95%-confidence QPE T cost subject to chemical accuracy "
            "of the lowest resolved HF-reference compiled-PF eigenphase"
        ),
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "timing_seconds": {
            "reference_load": reference_load_seconds,
            "compiled_unitary_build": compiled_unitary_build_seconds,
            "compiled_unitary_eigensolve": compiled_unitary_eigensolve_seconds,
            "postprocess": postprocess_seconds,
            "total": evaluation_wall_seconds,
        },
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

    evaluation_started = time.perf_counter()
    reaction_path = reaction_path.resolve()
    reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
    reference = (
        json.loads(Path(reference_path).read_text(encoding="utf-8"))
        if reference_path is not None
        else evaluate_exact_difference_reference(reaction_path, backend=backend)
    )
    if reference.get("observable") != "exact_unitary_signed_energy_difference":
        raise ValueError("reference is not an exact-unitary signed-difference result")
    if reference.get("evaluation_contract_version") != EVALUATION_CONTRACT_VERSION:
        raise ValueError("reference uses a different evaluator contract")
    if reference.get("reaction_sha256") != _sha256(reaction_path):
        raise ValueError(f"exact reference does not match {reaction_path}")
    species_results: dict[str, dict[str, Any]] = {}
    exact_difference = 0.0
    compiled_difference = 0.0
    total_t_count = 0
    confidence_adjusted_t_count = 0
    unique_species = {str(item["species"]) for item in reaction["terms"]}
    if not unique_species:
        raise ValueError("reaction must contain at least one species")
    # Give each unique species an equal share of the five-percent failure
    # budget. The union bound therefore guarantees at least 95% probability
    # that every required species peak has been observed.
    species_target_confidence = 1.0 - ((1.0 - QPE_TARGET_CONFIDENCE) / len(unique_species))
    for item in reaction["terms"]:
        species = str(item["species"])
        coefficient = float(item["coefficient"])
        if species not in species_results:
            problem_path = (reaction_path.parent / str(item["problem"])).resolve()
            result = evaluate_molecule(
                problem_path,
                policy_path,
                backend=backend,
                reference_result=reference["species"][species],
                target_confidence=species_target_confidence,
            )
            species_results[species] = result
            total_t_count += int(result["t_count"])
            confidence_adjusted_t_count += int(result["confidence_adjusted_t_count"])
        result = species_results[species]
        exact_difference += coefficient * result["exact_ground_energy_hartree"]
        compiled_difference += coefficient * result["compiled_pf_energy_hartree"]

    signed_error = compiled_difference - exact_difference
    absolute_error = abs(signed_error)
    passes = absolute_error <= CHEMICAL_ACCURACY_HARTREE
    union_bound_confidence = max(
        0.0,
        1.0
        - sum(
            1.0 - float(result["qpe_achieved_confidence"]) for result in species_results.values()
        ),
    )
    independent_joint_confidence = math.prod(
        float(result["qpe_achieved_confidence"]) for result in species_results.values()
    )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    return {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
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
        "qpe_joint_target_confidence": QPE_TARGET_CONFIDENCE,
        "qpe_species_target_confidence": species_target_confidence,
        "qpe_confidence_allocation": "equal_union_bound_failure_budget",
        "qpe_joint_achieved_confidence_union_bound": union_bound_confidence,
        "qpe_joint_achieved_confidence_independent": independent_joint_confidence,
        "per_attempt_t_count": total_t_count,
        "confidence_adjusted_t_count": confidence_adjusted_t_count,
        "t_count": total_t_count,
        "score": confidence_adjusted_t_count if passes else None,
        "species": species_results,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "timing_seconds": {
            "species_total": sum(
                float(result["evaluation_wall_seconds"]) for result in species_results.values()
            ),
            "total": evaluation_wall_seconds,
        },
        "evaluation_contract": (
            "minimize the summed confidence-adjusted species QPE T cost with at "
            "least 95% union-bound confidence, subject to signed reaction-energy "
            "error being within chemical accuracy"
        ),
    }
