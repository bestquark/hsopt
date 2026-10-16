"""Exact compiled-unitary energy evaluation for small chemistry problems.

The scientific observable in this module is an eigenphase of the *compiled*
product-formula unitary. CUDA-Q Ross--Selinger synthesis runs on the CPU. Dense Hamiltonian and
unitary diagonalizations run through the selected NumPy-compatible array
backend, normally cuPyNumeric on an NVIDIA GPU.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


CHEMICAL_ACCURACY_HARTREE = 1.0 / 627.5094740631
EXACT_GROUND_SUBSPACE_ATOL_HARTREE = 1.0e-9
TARGET_ANCHOR_MIN_PROBABILITY = 1.0e-12
TARGET_STATE_DEFINITION = "normalized_anchor_projection_onto_exact_ground_eigenspace"
PHASE_ENCODING_SCHEMA = 1
PHASE_ENCODING_METHOD = "centered_nonidentity_pauli_one_norm"
QPE_DIMENSIONLESS_EVOLUTION_TIME = 1.0
QPE_PHASE_ESTIMATION_ALPHA = 8.0 * math.pi
QPE_SINGLE_RUN_PHASE_SUCCESS_LOWER_BOUND = 0.5
EVALUATION_CONTRACT_VERSION = "normalized_hamiltonian_hf_qpe_t_count_cudaq_elote_v5"
REFERENCE_TARGET = "lowest_exact_hf_accessible_energy"
# Exact Hamiltonian references contain no approximate rotation synthesis.
EXACT_REFERENCE_CONTRACT_VERSIONS = (
    "normalized_hamiltonian_qpe_t_count_v2",
    "normalized_hamiltonian_qpe_t_count_cudaq_v3",
    "normalized_hamiltonian_hf_qpe_t_count_cudaq_v4",
    EVALUATION_CONTRACT_VERSION,
)
SYNTHESIS_OPTIONS = {
    "seed": 0,
    "max_factoring_iterations": 500_000,
    "max_candidate_iterations": 2_000_000,
    "max_factoring_restarts": 8,
    "max_odgp_scan_steps": 65_536,
    "timeout_ms": None,
}
DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL = "individual_errors"
DIFFERENCE_ERROR_CONTRACT_CANCELLATION = "error_cancellation"
DIFFERENCE_ERROR_CONTRACTS = (
    DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL,
    DIFFERENCE_ERROR_CONTRACT_CANCELLATION,
)
DIFFERENCE_ERROR_CONTRACT_VERSION = "reaction_error_contracts_v1"
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


def _host_array(value, *args, **kwargs):
    if type(value).__module__ == "cupy" or type(value).__module__.startswith("cupy."):
        value = value.get()
    return np.asarray(value, *args, **kwargs)


_PAULIS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": _host_array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": _host_array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": _host_array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}
_H = _PAULIS["X"] + _PAULIS["Z"]
_H /= math.sqrt(2.0)
_S_DAGGER = _host_array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)


def _safe_policy_import(
    name: str,
    globals: Mapping[str, Any] | None = None,
    locals: Mapping[str, Any] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    """Support only the explicitly allowed ``import math`` policy statement."""

    _ = (globals, locals)
    if name != "math" or fromlist or level != 0:
        raise ImportError("ProductFormula policies may import only math")
    return math


_SAFE_POLICY_BUILTINS = {
    "__build_class__": __build_class__,
    "__import__": _safe_policy_import,
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
    if normalized in {"cupy-lanczos"}:
        import cupy as xp

        return normalized, xp
    if normalized != "cupynumeric":
        raise ValueError("backend must be numpy, cupynumeric, or cupy-lanczos")
    try:
        import cupynumeric as xp
    except ImportError as exc:
        raise RuntimeError(
            "cuPyNumeric is unavailable. Run this evaluator under Legate on an NVIDIA system."
        ) from exc
    return "cupynumeric", xp


def _fence(backend_name: str) -> None:
    if backend_name.startswith("cupy") and backend_name != "cupynumeric":
        import cupy

        cupy.cuda.get_current_stream().synchronize()
        return
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
        if isinstance(node, ast.Import):
            if (
                len(node.names) == 1
                and node.names[0].name == "math"
                and node.names[0].asname in {None, "math"}
            ):
                continue
            raise ValueError(f"{path.name} may import only math")
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
    identity_coefficient = float(problem["identity_coefficient"])
    if not math.isfinite(identity_coefficient):
        raise ValueError(f"non-finite identity coefficient in {path}")
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
    expected_encoding = qpe_phase_encoding(identity_coefficient, terms)
    supplied_encoding = problem.get("phase_encoding")
    if not isinstance(supplied_encoding, Mapping):
        raise ValueError(f"{path.name} is missing the required normalized-QPE phase_encoding")
    _validate_phase_encoding(
        supplied_encoding,
        expected=expected_encoding,
        path=path,
    )
    target_time = float(problem["target_time"])
    expected_time = float(expected_encoding["physical_evolution_time_hartree_inverse"])
    if not math.isclose(target_time, expected_time, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError(
            f"{path.name} target_time={target_time} does not match the normalized-QPE "
            f"time {expected_time}"
        )
    problem["identity_coefficient"] = identity_coefficient
    problem["target_time"] = target_time
    problem["phase_encoding"] = expected_encoding
    problem["_path"] = str(path)
    return problem


def qpe_phase_encoding(
    identity_coefficient: float,
    terms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the rigorous centered Hamiltonian encoding used by QPE.

    Let ``H = c0 I + sum_j h_j P_j`` and
    ``lambda = sum_j |h_j|``.  The encoded unitary is

    ``U = exp(+i (H - c0 I) / lambda)``.

    Since ``||(H - c0 I)|| <= lambda``, every encoded eigenphase lies in
    ``[-1, 1]`` radians.  The principal phase therefore maps back to a unique
    physical energy without consulting the exact energy:

    ``E = c0 + lambda * phase``.
    """

    shift = float(identity_coefficient)
    scale = math.fsum(abs(float(term["coefficient"])) for term in terms)
    if not math.isfinite(shift):
        raise ValueError("phase-encoding energy shift must be finite")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("phase-encoding nonidentity Pauli one-norm must be positive")
    physical_time = QPE_DIMENSIONLESS_EVOLUTION_TIME / scale
    return {
        "schema": PHASE_ENCODING_SCHEMA,
        "method": PHASE_ENCODING_METHOD,
        "dimensionless_evolution_time": QPE_DIMENSIONLESS_EVOLUTION_TIME,
        "energy_shift_hartree": shift,
        "energy_scale_hartree": scale,
        "physical_evolution_time_hartree_inverse": physical_time,
        "centered_spectral_bound_hartree": [-scale, scale],
        "encoded_phase_bound_radians": [
            -QPE_DIMENSIONLESS_EVOLUTION_TIME,
            QPE_DIMENSIONLESS_EVOLUTION_TIME,
        ],
        "energy_recovery": "E_hartree = energy_shift_hartree + energy_scale_hartree * phase",
    }


def _validate_phase_encoding(
    supplied: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    path: Path,
) -> None:
    """Require checked-in encoding metadata to match the Hamiltonian exactly."""

    for key in ("schema", "method", "energy_recovery"):
        if supplied.get(key) != expected[key]:
            raise ValueError(f"{path.name} has stale phase_encoding field {key!r}")
    for key in (
        "dimensionless_evolution_time",
        "energy_shift_hartree",
        "energy_scale_hartree",
        "physical_evolution_time_hartree_inverse",
    ):
        if not math.isclose(
            float(supplied.get(key, math.nan)),
            float(expected[key]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"{path.name} has stale phase_encoding field {key!r}")
    for key in ("centered_spectral_bound_hartree", "encoded_phase_bound_radians"):
        actual_values = supplied.get(key)
        expected_values = expected[key]
        if not isinstance(actual_values, list) or len(actual_values) != 2:
            raise ValueError(f"{path.name} has malformed phase_encoding field {key!r}")
        if not all(
            math.isclose(
                float(actual),
                float(wanted),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            for actual, wanted in zip(actual_values, expected_values, strict=True)
        ):
            raise ValueError(f"{path.name} has stale phase_encoding field {key!r}")


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
    indices_host = _host_array(
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

    eigenvalues_host = _host_array(eigenvalues, dtype=np.float64)
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
    determinant_probabilities = _host_array(
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
    hf_target_probability = float(_host_array(xp.abs(target[hf_position]) ** 2))
    return (
        ground_energy,
        target,
        hf_ground_probability,
        hf_target_probability,
        int(ground_columns.size),
        anchor_kind,
        int(indices_host[anchor_position]),
    )


@lru_cache(maxsize=1)
def _cudaq_synth() -> tuple[Any, str]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The CUDA-Q `sample` and `observe`", category=FutureWarning
        )
        try:
            import cudaq
            from cudaq import synth
        except ImportError as exc:
            raise RuntimeError(
                "CUDA-Q 0.16 rotation synthesis is required; install the locked Linux "
                "dependencies or the macOS CPU distribution described in README.md."
            ) from exc
    return synth, str(cudaq.__version__)


def _synthesis_metadata() -> dict[str, Any]:
    _, version = _cudaq_synth()
    return {
        "compiler": "cudaq.synth.gridsynth",
        "version": version,
        "algorithm": "Ross-Selinger",
        "epsilon_convention": "requested_operator_norm_tolerance",
        "global_phase": "retained",
        **SYNTHESIS_OPTIONS,
    }


def _gate_sequence_matrix(gates: str) -> np.ndarray:
    """Reconstruct CUDA-Q's literal multiplication order, including scalar W."""
    omega = np.exp(0.25j * math.pi)
    alphabet = {
        "I": _PAULIS["I"],
        "H": _H,
        "S": np.diag([1.0, 1.0j]),
        "T": np.diag([1.0, omega]),
        "X": _PAULIS["X"],
        "W": omega * _PAULIS["I"],
    }
    matrix = np.eye(2, dtype=np.complex128)
    for gate in gates:
        matrix = matrix @ alphabet[gate]
    return matrix


def _rz_matrix(theta: float) -> np.ndarray:
    return _host_array(
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
    """Compile one Rz reproducibly on the CPU with CUDA-Q Ross--Selinger."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"synthesis epsilon must be positive, got {epsilon}")
    synth, _ = _cudaq_synth()
    sequence = synth.gridsynth(theta, epsilon, **SYNTHESIS_OPTIONS)
    gate_string = str(sequence)
    matrix = _gate_sequence_matrix(gate_string)
    operator_error = float(synth.rz_error(theta, sequence))
    if not math.isfinite(operator_error) or operator_error > epsilon:
        raise RuntimeError("CUDA-Q synthesis exceeded the requested operator-norm tolerance")
    return {
        "gate_string": gate_string,
        "matrix": matrix,
        "t_count": int(sequence.t_count),
        "gate_count": len(sequence),
        "operator_error": operator_error,
    }


def _synthesis_worker_count() -> int:
    """Use available CPUs; deployments can cap concurrency for shared workers."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    requested = os.environ.get("HSOPT_SYNTHESIS_WORKERS")
    if requested is None:
        return max(1, available)
    try:
        workers = int(requested)
    except ValueError as exc:
        raise ValueError("HSOPT_SYNTHESIS_WORKERS must be a positive integer") from exc
    if workers <= 0:
        raise ValueError("HSOPT_SYNTHESIS_WORKERS must be a positive integer")
    return max(1, min(available, workers))


def _compile_rotations(
    rotations: Sequence[tuple[Mapping[str, Any], float]],
    product_formula: Any,
) -> tuple[list[tuple[float, float]], dict[tuple[float, float], dict[str, Any]], dict[str, Any]]:
    """Compile distinct rotations concurrently, then consume them in policy order.

    CUDA-Q releases the GIL during its native synthesis. No GPU arrays or
    candidate methods enter worker threads. Per-call seeds/search budgets stay
    fixed, so scheduling does not alter circuits, tolerances or T counts.
    """
    keys = []
    for term, alpha in rotations:
        epsilon = float(product_formula.synthesis_epsilon(dict(term), alpha))
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("ProductFormula.synthesis_epsilon() must return a value in (0, 1)")
        keys.append((float(round(-2.0 * alpha, 15)), epsilon))
    unique = list(dict.fromkeys(keys))
    workers = min(_synthesis_worker_count(), len(unique))
    # Initialize bindings on the calling thread before concurrent native calls.
    _cudaq_synth()
    started = time.perf_counter()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hsopt-synthesis") as pool:
            values = list(pool.map(lambda key: _compile_rz(*key), unique))
    else:
        values = [_compile_rz(*key) for key in unique]
    return (
        keys,
        dict(zip(unique, values, strict=True)),
        {
            "worker_threads": workers,
            "unique_rotation_count": len(unique),
            "wall_seconds": time.perf_counter() - started,
        },
    )


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


def _conjugated_pivot_pauli(label: str, pivot_pauli: str, frame=()) -> tuple[complex, str]:
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
    if frame:
        from .clifford import conjugate, inverse
        sign, transformed = conjugate("".join(operator), inverse(frame))
        return phase * sign, transformed
    return phase, "".join(operator)


def _apply_compiled_rotation(
    unitary: Any,
    workspace: Any,
    *,
    label: str,
    compiled_rz: np.ndarray,
    xp: Any,
    frame=(),
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
        scaffold_phase, transformed_label = _conjugated_pivot_pauli(label, pivot_pauli, frame)
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

    from . import clifford, elote

    candidate_problem = elote.candidate_problem(problem, policy)
    decomposition = copy.deepcopy(candidate_problem.get("fermionic_decomposition"))
    product_formula = policy.ProductFormula(candidate_problem)
    repetitions_method = getattr(product_formula, "repetitions", None)
    if not callable(repetitions_method):
        raise ValueError("ProductFormula must define repetitions()")
    raw_steps = repetitions_method()
    if type(raw_steps) is not int or raw_steps <= 0:
        raise ValueError("ProductFormula.repetitions() must be a positive integer")
    n_steps = raw_steps
    step_time = float(problem["target_time"]) / n_steps
    basis_method = getattr(product_formula, "basis_change", None)
    frame = clifford.validate_frame(basis_method() if basis_method else [], int(problem["n_qubits"]))
    terms = elote.transformed_terms(problem, frame)
    trusted_terms = {str(term["label"]): dict(term) for term in terms}
    # Decomposing a fermionic block can introduce Pauli contributions that
    # cancel in the full H. They are admitted only from the verified host map.
    if decomposition:
        for block in decomposition["blocks"]:
            for term in block["terms"]:
                sign, label = clifford.conjugate(term["label"], frame)
                if label not in trusted_terms:
                    extra = {**elote.term_fields(label, 0.0),
                             "original_label": term["label"], "frame_sign": sign}
                    trusted_terms[label] = extra
                    terms.append(extra)
    raw_rotations = product_formula.build_step([dict(term) for term in terms], step_time)
    rotations, recipe = elote.schedule(
        raw_rotations, trusted_terms,
        limit=MAX_POLICY_ROTATIONS_PER_TERM * len(problem["terms"]),
    )
    recipe.update({"basis_change": frame, "repetitions": n_steps, "step_time": step_time,
                   "policy_sha256": (_sha256(Path(policy.__file__)) if getattr(policy, "__file__", None) else None),
                   "problem_sha256": (_sha256(Path(problem["_path"])) if problem.get("_path") else None)})
    if decomposition:
        recipe["fermionic_decomposition_sha256"] = decomposition["sha256"]
    classification = _classify_rotation_schedule(
        terms,
        rotations,
        step_time=step_time,
    )
    classification = elote.classify_groups(recipe, terms, step_time) or classification
    recipe["formula_name"] = classification["formula_name"]
    recipe["formula_order"] = classification["formula_order"]
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
            "elote": recipe,
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
    matrix_free: bool = False,
) -> tuple[Any, dict[str, Any]]:
    product_formula, rotations, schedule_metrics = _product_formula_schedule(problem, policy)
    n_steps = int(schedule_metrics["n_steps"])
    frame = schedule_metrics["elote"]["basis_change"]
    rotation_keys, compiled_rotations, synthesis_execution = _compile_rotations(
        rotations, product_formula
    )

    dim = 1 << int(problem["n_qubits"])
    fence_interval = _rotation_fence_interval(dim)
    if matrix_free:
        if backend_name != "cupy-lanczos":
            raise ValueError("compiled circuit graphs require GPU Lanczos")
        from .gpu_circuit import CompiledCircuit

        unitary, workspace = CompiledCircuit(dim), None
    else:
        unitary = xp.eye(dim, dtype=xp.complex128)
        workspace = xp.empty_like(unitary)
    t_count = 0
    skipped = 0
    epsilon_histogram: Counter[str] = Counter()
    epsilon_values: set[float] = set()
    compilations: dict[str, dict[str, Any]] = {}

    updater = _apply_compiled_rotation
    gpu_metrics = {}
    if backend_name in {"cupy-lanczos"}:
        from .gpu_rotations import RotationUpdater
        from .gpu_spectrum import phase_enclosure

        if not matrix_free and not frame:
            updater = RotationUpdater(optimized=True)
        gpu_metrics["compiled_phase_enclosure"] = phase_enclosure(
            (compiled_rotations[key]["matrix"] for key in rotation_keys), n_steps
        )
        gpu_metrics["matrix_construction"] = (
            "matrix_free_cuda_graph_literal_circuit" if matrix_free else "fused_complex128_paired_or_generic"
        )
    applied_rotations = 0
    for (term, _alpha), (theta, epsilon) in zip(rotations, rotation_keys, strict=True):
        compiled = compiled_rotations[(theta, epsilon)]
        if matrix_free:
            unitary.add(str(term["label"]), compiled["matrix"], frame=frame)
        else:
            unitary, workspace = updater(
                unitary, workspace, label=str(term["label"]),
                compiled_rz=compiled["matrix"], xp=xp,
                **({"frame": frame} if frame else {}),
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
    if matrix_free:
        unitary.finish()
    del workspace
    if hasattr(updater, "maps"):
        updater.maps.clear()
    policy_metrics = dict(schedule_metrics["policy"])
    policy_metrics.update(
        {
            "synthesis_policy": ("uniform" if len(epsilon_values) == 1 else "rotation_dependent"),
            "uniform_epsilon": (next(iter(epsilon_values)) if len(epsilon_values) == 1 else None),
        }
    )
    from .clifford import circuit_metrics
    from .elote import digest

    recipe = schedule_metrics["elote"]
    recipe["synthesis_tolerances"] = [epsilon for _theta, epsilon in rotation_keys]
    recipe["resolved_recipe_sha256"] = digest(recipe)
    logical_circuit = circuit_metrics(
        rotations, [compiled_rotations[key]["gate_string"] for key in rotation_keys], frame=frame,
    )
    if logical_circuit["one_step_t_count"] != t_count:
        raise RuntimeError("literal gate ledger disagrees with CUDA-Q T count")
    return unitary, {
        **schedule_metrics,
        **gpu_metrics,
        "logical_circuit": logical_circuit,
        "policy": policy_metrics,
        "skipped_rotation_count": skipped * n_steps,
        "one_step_t_count": t_count,
        "t_count": t_count * n_steps,
        "dense_matrix_dimension": dim,
        "dense_matrix_bytes": 0 if matrix_free else dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_workspace_bytes": dim * np.dtype(np.complex128).itemsize if matrix_free else dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_chunk_target_bytes": ROTATION_CHUNK_TARGET_BYTES,
        "rotation_fence_interval": fence_interval,
        "synthesis_epsilon_histogram": dict(sorted(epsilon_histogram.items())),
        "rotation_synthesis": _synthesis_metadata(),
        "rotation_synthesis_execution": synthesis_execution,
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
    matrix_free: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Construct the same candidate schedule with ideal Pauli rotations."""

    _product_formula, rotations, schedule_metrics = _product_formula_schedule(problem, policy)
    dim = 1 << int(problem["n_qubits"])
    fence_interval = _rotation_fence_interval(dim)
    frame = schedule_metrics["elote"]["basis_change"]
    if matrix_free:
        if backend_name != "cupy-lanczos":
            raise ValueError("matrix-free ideal PF requires GPU Lanczos")
        from .gpu_circuit import CompiledCircuit
        unitary, workspace = CompiledCircuit(dim), None
    else:
        unitary = xp.eye(dim, dtype=xp.complex128)
        workspace = xp.empty_like(unitary)
    matrices = []
    for index, (term, alpha) in enumerate(rotations, start=1):
        exact_rz = _rz_matrix(float(round(-2.0 * alpha, 15)))
        matrices.append(exact_rz)
        if matrix_free:
            unitary.add(str(term["label"]), exact_rz, frame=frame)
        else:
            unitary, workspace = _apply_compiled_rotation(
                unitary, workspace, label=str(term["label"]), compiled_rz=exact_rz, xp=xp,
                frame=frame,
            )
        if index % fence_interval == 0:
            _fence(backend_name)
    _fence(backend_name)
    del workspace
    if matrix_free:
        from .gpu_spectrum import phase_enclosure
        unitary.finish()
        schedule_metrics["compiled_phase_enclosure"] = phase_enclosure(matrices, schedule_metrics["n_steps"])
    return unitary, {
        **schedule_metrics,
        "rotation_implementation": "ideal_pauli_rotations",
        "dense_matrix_dimension": dim,
        "dense_matrix_bytes": 0 if matrix_free else dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_workspace_bytes": dim * np.dtype(np.complex128).itemsize if matrix_free else dim * dim * np.dtype(np.complex128).itemsize,
        "rotation_fence_interval": fence_interval,
    }


def _principal_phase(phase: float) -> float:
    """Wrap a real phase into ``[-pi, pi]`` without a spectral reference."""

    return math.atan2(math.sin(float(phase)), math.cos(float(phase)))


def _decode_energy(
    phase: float,
    *,
    target_time: float,
    identity_coefficient: float,
) -> float:
    """Map a normalized principal phase back to an energy in Hartree."""

    if target_time <= 0.0:
        raise ValueError("target_time must be positive")
    return float(identity_coefficient) + _principal_phase(phase) / float(target_time)


def _qpe_full_evolution_queries(
    *,
    target_time: float,
    energy_precision_hartree: float,
) -> int:
    """Return the paper's rigorous full-evolution query bound for QPE."""

    if target_time <= 0.0 or energy_precision_hartree <= 0.0:
        raise ValueError("QPE time and energy precision must be positive")
    phase_precision = target_time * energy_precision_hartree
    return max(1, math.ceil(QPE_PHASE_ESTIMATION_ALPHA / phase_precision))


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


def _qpe_cost_metrics(
    *,
    peak_probability: float,
    target_confidence: float,
    target_time: float,
    energy_precision_hartree: float,
    compiled_evolution_t_count: int,
) -> dict[str, Any]:
    """Return full QPE cost, including precision queries and peak repetitions."""

    success_probability = float(peak_probability) * QPE_SINGLE_RUN_PHASE_SUCCESS_LOWER_BOUND
    qpe_repetitions = _qpe_repetitions(
        success_probability,
        target_confidence=target_confidence,
    )
    qpe_achieved_confidence = _qpe_achieved_confidence(
        success_probability,
        qpe_repetitions,
    )
    full_evolution_queries = _qpe_full_evolution_queries(
        target_time=target_time,
        energy_precision_hartree=energy_precision_hartree,
    )
    per_qpe_run_t_count = int(compiled_evolution_t_count) * full_evolution_queries
    return {
        "qpe_target_confidence": float(target_confidence),
        "qpe_achieved_confidence": qpe_achieved_confidence,
        "qpe_repetitions": qpe_repetitions,
        "qpe_single_run_phase_success_lower_bound": (QPE_SINGLE_RUN_PHASE_SUCCESS_LOWER_BOUND),
        "qpe_success_probability_per_attempt": success_probability,
        "qpe_energy_precision_hartree": float(energy_precision_hartree),
        "qpe_phase_precision_radians": float(target_time) * float(energy_precision_hartree),
        "qpe_phase_estimation_alpha": QPE_PHASE_ESTIMATION_ALPHA,
        "qpe_full_evolution_queries": full_evolution_queries,
        "compiled_evolution_t_count": int(compiled_evolution_t_count),
        "per_attempt_t_count": per_qpe_run_t_count,
        "per_qpe_run_t_count": per_qpe_run_t_count,
        "confidence_adjusted_t_count": qpe_repetitions * per_qpe_run_t_count,
    }


def _select_lowest_hf_accessible_peak(
    step_phases: Sequence[float],
    hf_probabilities: Sequence[float],
    *,
    n_steps: int,
    target_time: float,
    identity_coefficient: float,
) -> dict[str, Any]:
    """Select the lowest resolved PF energy that ideal QPE can sample from HF."""

    phases = _host_array(step_phases, dtype=np.float64)
    probabilities = _host_array(hf_probabilities, dtype=np.float64)
    if phases.ndim != 1 or probabilities.shape != phases.shape or phases.size == 0:
        raise ValueError("PF eigenphases and HF probabilities must be nonempty 1D arrays")
    if n_steps <= 0 or target_time <= 0.0:
        raise ValueError("n_steps and target_time must be positive")

    total_phases = int(n_steps) * phases
    energies = _host_array(
        [
            _decode_energy(
                float(phase),
                target_time=target_time,
                identity_coefficient=identity_coefficient,
            )
            for phase in total_phases
        ],
        dtype=np.float64,
    )
    # QPE measures an energy eigenspace, not an arbitrary eigenvector chosen
    # inside a degenerate subspace. Sum its HF weights before thresholding.
    groups: list[list[int]] = []
    for index in sorted(range(energies.size), key=lambda i: float(energies[i])):
        if groups and abs(float(energies[index] - energies[groups[-1][0]])) <= 1.0e-10:
            groups[-1].append(index)
        else:
            groups.append([index])
    accessible_groups = [
        group for group in groups if float(probabilities[group].sum()) >= QPE_PEAK_WEIGHT_THRESHOLD
    ]
    accessible = np.asarray([index for group in accessible_groups for index in group])
    if accessible.size == 0:
        raise RuntimeError(
            "compiled PF unitary has no numerically resolved eigenphase with HF support"
        )
    selected_group = accessible_groups[0]
    selected = selected_group[0]
    selected_probability = min(1.0, max(0.0, float(probabilities[selected_group].sum())))
    return {
        "selected_eigen_index": selected,
        "selected_eigen_indices": selected_group,
        "selected_step_eigenphase_radians": float(phases[selected]),
        "selected_total_eigenphase_radians": float(total_phases[selected]),
        "selected_principal_eigenphase_radians": _principal_phase(total_phases[selected]),
        "compiled_pf_energy_hartree": float(energies[selected]),
        "qpe_success_probability_from_hf": selected_probability,
        "hf_accessible_peak_count": int(accessible.size),
        "hf_accessible_probability_mass": float(
            min(1.0, max(0.0, probabilities[accessible].sum()))
        ),
        "phase_decoding_energy_window_hartree": [
            float(identity_coefficient - math.pi / target_time),
            float(identity_coefficient + math.pi / target_time),
        ],
        "phase_branch_selection": "principal_phase_without_exact_energy_reference",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _serialize_state(state: Any) -> list[list[float]]:
    host = _host_array(state, dtype=np.complex128)
    return [[float(value.real), float(value.imag)] for value in host]


def _reference_energy(
    problem: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> float:
    """Validate the fixed-HF reference; ground energy is only a diagnostic."""

    if reference.get("observable") != "exact_unitary_ground_eigenphase":
        raise ValueError("reference is not an exact-unitary ground-eigenphase result")
    if reference.get("evaluation_contract_version") not in EXACT_REFERENCE_CONTRACT_VERSIONS:
        raise ValueError("reference uses a different evaluator contract")
    problem_path = Path(str(problem["_path"]))
    if reference.get("problem_sha256") != _sha256(problem_path):
        raise ValueError(f"exact reference does not match {problem_path}")
    if reference.get("reference_target") != REFERENCE_TARGET:
        raise ValueError("cached reference requires HF-accessible reference preparation")
    if reference.get("hf_basis_index") != _hartree_fock_basis_index(problem):
        raise ValueError("cached reference uses a different initial HF determinant")
    if reference.get("qpe_peak_weight_threshold") != QPE_PEAK_WEIGHT_THRESHOLD:
        raise ValueError("cached reference uses a different HF accessibility threshold")
    energy = float(reference["exact_hf_accessible_energy_hartree"])
    if not math.isfinite(energy):
        raise ValueError("cached HF-accessible reference energy must be finite")
    return energy


def prepare_hf_reference(
    problem_path: Path,
    ground_reference: Mapping[str, Any],
    *,
    backend: str = "numpy",
) -> dict[str, Any]:
    """Add the fixed-HF QPE target without recalibrating an initial policy.

    Reuse a certified HF-supported ground cluster when available. Otherwise
    resolve the lowest HF-supported Hamiltonian cluster with partial Lanczos.
    Exact evolution shares H's eigenvectors, so U_exact is never constructed.
    """
    started = time.perf_counter()
    problem = load_problem(problem_path)
    result = dict(ground_reference)
    if result.get("observable") != "exact_unitary_ground_eigenphase":
        raise ValueError("HF reference preparation requires a cached ground reference")
    if result.get("problem_sha256") != _sha256(problem_path):
        raise ValueError("cached ground reference does not match the Hamiltonian")
    if result.get("reference_target") == REFERENCE_TARGET:
        _reference_energy(problem, result)
        return result
    probability = float(result["hf_ground_subspace_probability"])
    if probability >= QPE_PEAK_WEIGHT_THRESHOLD:
        energy = float(result["exact_unitary_ground_energy_hartree"])
        diagnostics = {"method": "cached_hf_supported_ground_subspace"}
    else:
        backend_name, xp = array_backend(backend)
        indices, matrix = _fixed_electron_hamiltonian(problem)
        hf_index = int(np.flatnonzero(indices == _hartree_fock_basis_index(problem))[0])
        from .gpu_spectrum import reference_hf_eigensolve

        energy, probability, diagnostics = reference_hf_eigensolve(
            xp.asarray(matrix, dtype=xp.complex128),
            hf_index=hf_index,
            identity_coefficient=float(problem["identity_coefficient"]),
            target_time=float(problem["target_time"]),
            backend=backend_name,
        )
    ground_energy = float(result["exact_unitary_ground_energy_hartree"])
    result.update(
        evaluation_contract_version=EVALUATION_CONTRACT_VERSION,
        reference_target=REFERENCE_TARGET,
        hf_basis_index=_hartree_fock_basis_index(problem),
        qpe_peak_weight_threshold=QPE_PEAK_WEIGHT_THRESHOLD,
        exact_hf_accessible_energy_hartree=energy,
        exact_hf_accessible_probability=probability,
        exact_hf_accessible_ground_gap_hartree=energy - ground_energy,
        hf_reference_solver=diagnostics,
        hf_reference_preparation_seconds=time.perf_counter() - started,
    )
    return result


def evaluate_exact_reference(
    problem_path: Path,
    *,
    backend: str = "cupynumeric",
) -> dict[str, Any]:
    """Cache the exact ground phase once, using the selected reference solver."""

    evaluation_started = time.perf_counter()
    problem = load_problem(problem_path)
    backend_name, xp = array_backend(backend)

    sector_started = time.perf_counter()
    indices_host, sector_host = _fixed_electron_hamiltonian(problem)
    sector_hamiltonian = xp.asarray(sector_host, dtype=xp.complex128)
    _fence(backend_name)
    sector_build_seconds = time.perf_counter() - sector_started

    hamiltonian_eigensolve_started = time.perf_counter()
    if backend_name == "cupy-lanczos":
        from .gpu_spectrum import reference_eigensolve

        h_eigenvalues, h_eigenvectors = reference_eigensolve(
            sector_hamiltonian, ground_atol=EXACT_GROUND_SUBSPACE_ATOL_HARTREE
        )
    else:
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
    identity_coefficient = float(problem["identity_coefficient"])
    centered_h_eigenvalues = h_eigenvalues - identity_coefficient
    phases = xp.exp(1.0j * centered_h_eigenvalues * float(problem["target_time"]))
    # H and exp(+i(H-c0)t) share eigenvectors exactly. The Lanczos reference
    # decodes the converged Hamiltonian Ritz phases without constructing U_exact.
    exact_unitary = None
    if backend_name != "cupy-lanczos":
        exact_unitary = (h_eigenvectors * phases[None, :]) @ h_eigenvectors.conj().T
    _fence(backend_name)
    exact_unitary_build_seconds = time.perf_counter() - exact_unitary_started

    unitary_eigensolve_started = time.perf_counter()
    if backend_name == "cupy-lanczos":
        eigenvalues, eigenvectors = phases, h_eigenvectors
    else:
        eigenvalues, eigenvectors = xp.linalg.eig(exact_unitary)
    _fence(backend_name)
    unitary_eigensolve_seconds = time.perf_counter() - unitary_eigensolve_started

    postprocess_started = time.perf_counter()
    eigenvectors /= xp.linalg.norm(eigenvectors, axis=0, keepdims=True)
    ground_overlaps = xp.abs(eigenvectors.conj().T @ hamiltonian_target) ** 2
    selected = int(_host_array(xp.argmax(ground_overlaps)))
    selected_phase = float(_host_array(xp.angle(eigenvalues[selected])))
    unitary_energy = _decode_energy(
        selected_phase,
        target_time=float(problem["target_time"]),
        identity_coefficient=identity_coefficient,
    )
    full_target_state = xp.zeros(1 << int(problem["n_qubits"]), dtype=xp.complex128)
    indices = xp.asarray(indices_host, dtype=xp.int64)
    full_target_state[indices] = hamiltonian_target
    selected_overlap = float(_host_array(ground_overlaps[selected]))
    max_modulus_error = float(_host_array(xp.max(xp.abs(xp.abs(eigenvalues) - 1.0))))
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
    result = {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "observable": "exact_unitary_ground_eigenphase",
        "reference_solver": (
            "gpu_hamiltonian_lanczos_analytic_unitary_phase"
            if backend_name == "cupy-lanczos"
            else "dense_exact_unitary"
        ),
        "unitary_definition": "U_exact = exp(+i (H - c0 I) / lambda)",
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
        "phase_encoding": dict(problem["phase_encoding"]),
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
    return prepare_hf_reference(problem_path, result, backend=backend)


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
    ground_difference = 0.0
    for item in reaction["terms"]:
        species = str(item["species"])
        coefficient = float(item["coefficient"])
        if species not in species_references:
            raise ValueError(f"exact reference is missing reaction species {species!r}")
        problem_path = (reaction_path.parent / str(item["problem"])).resolve()
        problem = load_problem(problem_path)
        result = dict(species_references[species])
        unitary_energy = _reference_energy(problem, result)
        hamiltonian_energy = float(result["exact_hf_accessible_energy_hartree"])
        ground_difference += coefficient * float(result["hamiltonian_ground_energy_hartree"])
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
        "reference_target": REFERENCE_TARGET,
        "unitary_definition": ("U_exact = exp(+i (H - c0 I) / lambda) for each species"),
        "reaction": reaction["name"],
        "reaction_path": str(reaction_path),
        "reaction_sha256": _sha256(reaction_path),
        "backend": backend,
        "hamiltonian_exact_difference_hartree": hamiltonian_difference,
        "exact_unitary_difference_hartree": unitary_difference,
        "exact_ground_difference_hartree": ground_difference,
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
    energy_target: str = "hf_accessible",
    matrix_free: bool | None = None,
) -> dict[str, Any]:
    """Evaluate one explicitly selected, versioned compiled-PF energy contract."""

    if energy_target not in {"hf_accessible", "ground_branch"}:
        raise ValueError("energy_target must be hf_accessible or ground_branch")
    ground_input = energy_target == "ground_branch"
    if matrix_free is None:
        matrix_free = ground_input
    if matrix_free and not ground_input:
        raise ValueError("matrix-free ground targeting requires the ground-input contract")
    if ground_input and (backend != "cupy-lanczos" or (reference_path is None and reference_result is None)):
        raise ValueError("ground-input scoring requires a cached reference and GPU Lanczos; no fallback")

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
    if ground_input:
        from . import ground_branch

        exact_energy, cached_state = ground_branch.cached_reference(problem, reference_result)
        input_state = xp.asarray(cached_state, dtype=xp.complex128)
    else:
        exact_energy = _reference_energy(problem, reference_result)
        input_state = _hartree_fock_state(problem, xp)
    _fence(backend_name)
    reference_load_seconds = time.perf_counter() - reference_started

    compiled_unitary_started = time.perf_counter()
    unitary_step, metrics = _build_compiled_step(
        problem,
        policy,
        backend_name=backend_name,
        xp=xp,
        **({"matrix_free": True} if matrix_free else {}),
    )
    compiled_unitary_build_seconds = time.perf_counter() - compiled_unitary_started

    eigensolve_started = time.perf_counter()
    spectral_solver = {"method": "full_spectrum", "spectrum_complete": True}
    if backend_name in {"cupy-lanczos"}:
        from .gpu_spectrum import eigensolve

        eigenvalues, eigenvectors, spectral_solver = eigensolve(
            unitary_step, problem, metrics, iterative=backend_name == "cupy-lanczos",
            **({"target_state": input_state} if ground_input else {}),
        )
    else:
        eigenvalues, eigenvectors = xp.linalg.eig(unitary_step)
    _fence(backend_name)
    eigenvectors /= xp.linalg.norm(eigenvectors, axis=0, keepdims=True)
    _fence(backend_name)
    compiled_unitary_eigensolve_seconds = time.perf_counter() - eigensolve_started

    postprocess_started = time.perf_counter()
    input_probabilities = _host_array(
        xp.abs(eigenvectors.conj().T @ input_state) ** 2,
        dtype=np.float64,
    )
    selector = ground_branch.select_peak if ground_input else _select_lowest_hf_accessible_peak
    peak = selector(
        _host_array(xp.angle(eigenvalues), dtype=np.float64),
        input_probabilities,
        n_steps=int(metrics["n_steps"]),
        target_time=float(problem["target_time"]),
        identity_coefficient=float(problem["identity_coefficient"]),
    )
    if not ground_input and not spectral_solver["spectrum_complete"]:
        # These are full-spectrum diagnostics and cannot be inferred from Ritz pairs.
        peak["hf_accessible_peak_count"] = None
        peak["hf_accessible_probability_mass"] = None
    estimated_energy = float(peak["compiled_pf_energy_hartree"])
    signed_error = estimated_energy - exact_energy
    absolute_error = abs(signed_error)
    passes = absolute_error < CHEMICAL_ACCURACY_HARTREE
    qpe_success_probability = float(peak[
        "qpe_success_probability_from_ground_input" if ground_input else "qpe_success_probability_from_hf"
    ])
    if matrix_free:
        from .gpu_circuit import certify_qpe_repetitions

        certify_qpe_repetitions(qpe_success_probability, target_confidence)
    compiled_evolution_t_count = int(metrics["t_count"])
    qpe_energy_precision = (
        CHEMICAL_ACCURACY_HARTREE - absolute_error if passes else CHEMICAL_ACCURACY_HARTREE
    )
    qpe_cost = _qpe_cost_metrics(
        peak_probability=qpe_success_probability,
        target_confidence=target_confidence,
        target_time=float(problem["target_time"]),
        energy_precision_hartree=qpe_energy_precision,
        compiled_evolution_t_count=compiled_evolution_t_count,
    )
    max_modulus_error = float(_host_array(xp.max(xp.abs(xp.abs(eigenvalues) - 1.0))))
    _fence(backend_name)
    postprocess_seconds = time.perf_counter() - postprocess_started
    evaluation_wall_seconds = time.perf_counter() - evaluation_started

    result = {
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
        "spectral_solver": spectral_solver,
        "n_qubits": int(problem["n_qubits"]),
        "electrons": int(problem["electrons"]),
        "reference_target": REFERENCE_TARGET,
        "exact_reference_energy_hartree": exact_energy,
        "exact_hf_accessible_energy_hartree": exact_energy,
        "exact_hf_accessible_probability": reference_result.get("exact_hf_accessible_probability"),
        "exact_hf_accessible_ground_gap_hartree": reference_result.get(
            "exact_hf_accessible_ground_gap_hartree"
        ),
        "exact_ground_energy_hartree": reference_result["exact_unitary_ground_energy_hartree"],
        "exact_unitary_ground_energy_hartree": reference_result[
            "exact_unitary_ground_energy_hartree"
        ],
        "ground_state_energy_error_hartree": abs(
            estimated_energy - float(reference_result["exact_unitary_ground_energy_hartree"])
        ),
        "compiled_pf_energy_hartree": estimated_energy,
        "signed_energy_error_hartree": signed_error,
        "absolute_energy_error_hartree": absolute_error,
        "absolute_energy_error_mha": 1000.0 * absolute_error,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "passes_chemical_accuracy": passes,
        "target_time": float(problem["target_time"]),
        "phase_encoding": dict(problem["phase_encoding"]),
        **peak,
        "pf_peak_selection": "lowest_energy_peak_with_resolved_hf_weight",
        "qpe_peak_weight_threshold": QPE_PEAK_WEIGHT_THRESHOLD,
        "max_eigenvalue_modulus_error": max_modulus_error,
        **qpe_cost,
        "t_count": compiled_evolution_t_count,
        "score": qpe_cost["confidence_adjusted_t_count"] if passes else None,
        "evaluation_contract": (
            "minimize the paper-bound full QPE T cost, including phase-resolution "
            "queries and enough HF-state attempts for 95% peak confidence, subject "
            "to one-kcal/mol simulation error relative to exact evolution from "
            "the same HF determinant after decoding the normalized Hamiltonian phase"
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
    if ground_input:
        result.update(
            evaluation_contract_version=ground_branch.CONTRACT_VERSION,
            energy_target=energy_target,
            reference_target=ground_branch.REFERENCE_TARGET,
            observable="compiled_unitary_exact_ground_input_eigenphase",
            pf_peak_selection="largest_exact_ground_input_eigenspace_weight",
            initial_state_definition=reference_result["target_state_definition"],
            exact_ground_input_sha256=hashlib.sha256(cached_state.tobytes()).hexdigest(),
            evaluation_contract=(
                "minimize full QPE T95 for the dominant compiled eigenspace sampled from the "
                "cached exact ground input; compare E_compiled,0 - E_0 within one kcal/mol. "
                "Lanczos certifies a majority-weight peak and never uses dense fallback. "
                "Ideal ground-state preparation cost is excluded from this dynamics benchmark."
            ),
        )
        for name in ("exact_hf_accessible_energy_hartree", "exact_hf_accessible_probability",
                     "exact_hf_accessible_ground_gap_hartree", "qpe_peak_weight_threshold"):
            result.pop(name, None)
    return result


def evaluate_difference(
    reaction_path: Path,
    policy_path: Path,
    *,
    error_contract: str,
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a reaction under one explicit, immutable error contract.

    ``individual_errors`` enforces the conservative triangle-inequality metric
    ``sum_s |nu_s (E_pf,s - E_exact,s)|``. ``error_cancellation`` instead
    enforces the physically aggregated signed reaction error
    ``|sum_s nu_s (E_pf,s - E_exact,s)|``.
    """

    if error_contract not in DIFFERENCE_ERROR_CONTRACTS:
        raise ValueError(
            f"error_contract must be one of {DIFFERENCE_ERROR_CONTRACTS}, got {error_contract!r}"
        )

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
    if reference.get("evaluation_contract_version") not in EXACT_REFERENCE_CONTRACT_VERSIONS:
        raise ValueError("reference uses a different evaluator contract")
    if reference.get("reference_target") != REFERENCE_TARGET:
        raise ValueError("reaction reference requires HF-accessible reference preparation")
    if reference.get("reaction_sha256") != _sha256(reaction_path):
        raise ValueError(f"exact reference does not match {reaction_path}")
    species_results: dict[str, dict[str, Any]] = {}
    exact_difference = 0.0
    compiled_difference = 0.0
    total_t_count = 0
    species_coefficients: dict[str, float] = {}
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
        species_coefficients[species] = species_coefficients.get(species, 0.0) + coefficient
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
        result = species_results[species]
        exact_difference += coefficient * result["exact_reference_energy_hartree"]
        compiled_difference += coefficient * result["compiled_pf_energy_hartree"]

    signed_error = compiled_difference - exact_difference
    absolute_error = abs(signed_error)
    signed_species_contributions = {
        species: species_coefficients[species]
        * (
            float(result["compiled_pf_energy_hartree"])
            - float(result["exact_reference_energy_hartree"])
        )
        for species, result in species_results.items()
    }
    sum_absolute_species_error = math.fsum(
        abs(value) for value in signed_species_contributions.values()
    )
    individual_errors_pass = sum_absolute_species_error < CHEMICAL_ACCURACY_HARTREE
    error_cancellation_pass = absolute_error < CHEMICAL_ACCURACY_HARTREE
    if error_contract == DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL:
        contract_error = sum_absolute_species_error
        passes = individual_errors_pass
    else:
        contract_error = absolute_error
        passes = error_cancellation_pass
    reaction_qpe_precision = (
        CHEMICAL_ACCURACY_HARTREE - contract_error if passes else CHEMICAL_ACCURACY_HARTREE
    )
    confidence_adjusted_t_count = 0
    per_attempt_t_count = 0
    for species, result in species_results.items():
        weight = abs(species_coefficients[species])
        if weight <= 0.0:
            raise ValueError(f"reaction has zero net coefficient for species {species!r}")
        species_precision = reaction_qpe_precision / (len(unique_species) * weight)
        species_qpe_cost = _qpe_cost_metrics(
            peak_probability=float(result["qpe_success_probability_from_hf"]),
            target_confidence=species_target_confidence,
            target_time=float(result["phase_encoding"]["physical_evolution_time_hartree_inverse"]),
            energy_precision_hartree=species_precision,
            compiled_evolution_t_count=int(result["t_count"]),
        )
        result.update(species_qpe_cost)
        result["qpe_energy_precision_allocation"] = (
            "equal_absolute_reaction_contribution_under_triangle_inequality"
        )
        result["reaction_coefficient"] = species_coefficients[species]
        result["signed_reaction_error_contribution_hartree"] = signed_species_contributions[species]
        result["absolute_reaction_error_contribution_hartree"] = abs(
            signed_species_contributions[species]
        )
        confidence_adjusted_t_count += int(species_qpe_cost["confidence_adjusted_t_count"])
        per_attempt_t_count += int(species_qpe_cost["per_attempt_t_count"])
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
        "difference_error_contract": error_contract,
        "difference_error_contract_version": DIFFERENCE_ERROR_CONTRACT_VERSION,
        "observable": "compiled_unitary_signed_energy_difference",
        "reference_target": REFERENCE_TARGET,
        "exact_ground_difference_hartree": reference["exact_ground_difference_hartree"],
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
        "sum_absolute_species_error_hartree": sum_absolute_species_error,
        "sum_absolute_species_error_mha": 1000.0 * sum_absolute_species_error,
        "contract_error_hartree": contract_error,
        "contract_error_mha": 1000.0 * contract_error,
        "error_cancellation_gain_hartree": sum_absolute_species_error - absolute_error,
        "error_cancellation_fraction": (
            1.0 - absolute_error / sum_absolute_species_error
            if sum_absolute_species_error > 0.0
            else 0.0
        ),
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "passes_chemical_accuracy": passes,
        "passes_individual_errors_contract": individual_errors_pass,
        "passes_error_cancellation_contract": error_cancellation_pass,
        "qpe_energy_precision_hartree": reaction_qpe_precision,
        "qpe_energy_precision_allocation": (
            "equal_absolute_species_contribution_under_triangle_inequality"
        ),
        "qpe_joint_target_confidence": QPE_TARGET_CONFIDENCE,
        "qpe_species_target_confidence": species_target_confidence,
        "qpe_confidence_allocation": "equal_union_bound_failure_budget",
        "qpe_joint_achieved_confidence_union_bound": union_bound_confidence,
        "qpe_joint_achieved_confidence_independent": independent_joint_confidence,
        "compiled_evolution_t_count": total_t_count,
        "per_attempt_t_count": per_attempt_t_count,
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
            "least 95% union-bound confidence, subject to the explicitly selected "
            f"{error_contract} reaction-energy metric being within chemical accuracy"
        ),
    }


def evaluate_difference_individual_errors(
    reaction_path: Path,
    policy_path: Path,
    *,
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Enforce ``sum_s |nu_s (E_pf,s - E_exact,s)| < chemical accuracy``."""

    return evaluate_difference(
        reaction_path,
        policy_path,
        error_contract=DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL,
        backend=backend,
        reference_path=reference_path,
    )


def evaluate_difference_error_cancellation(
    reaction_path: Path,
    policy_path: Path,
    *,
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Enforce ``|sum_s nu_s (E_pf,s - E_exact,s)| < chemical accuracy``."""

    return evaluate_difference(
        reaction_path,
        policy_path,
        error_contract=DIFFERENCE_ERROR_CONTRACT_CANCELLATION,
        backend=backend,
        reference_path=reference_path,
    )
