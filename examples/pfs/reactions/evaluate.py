"""Fixed scientific evaluator for the ten-reaction benchmark.

This module owns the expensive, chemistry-facing part of the reaction
experiment.  Reaction definitions and stoichiometry intentionally live in
``experiments/reactions10/benchmark.py`` so there is only one registry.

The implementation is adapted from the archived Omni evaluator with SHA-256
``b81d8e936f71c0cbab76ba8f0b134b4ecbf2c610d95aaca04136b7aeb5309425``.
It reuses only chemistry kernels that are invariant in the cleaned repository;
the 1 kcal/mol conversion and reaction contract remain local to this benchmark.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np

from pfs.chemistry import (
    MoleculeSpec,
    _active_space_integrals,
    build_pauli_rotation_compiler,
    fixed_electron_sector_eigensystem,
    hartree_fock_state,
    import_pygridsynth_backend,
    pygridsynth_matrix_from_gates,
    qubit_problem_from_integrals,
    unwrap_energy_near_reference,
)


ARCHIVED_EVALUATOR_SHA256 = "b81d8e936f71c0cbab76ba8f0b134b4ecbf2c610d95aaca04136b7aeb5309425"
CACHE_FORMAT_VERSION = 1
HARTREE_PER_KCAL_PER_MOL = 1.0 / 627.5094740631
REACTION_TARGET_HARTREE = HARTREE_PER_KCAL_PER_MOL
REACTION_TARGET_MHA = 1000.0 * REACTION_TARGET_HARTREE
SPECTRAL_PEAK_WEIGHT_THRESHOLD = 1.0e-10


def _cache_paths(spec: MoleculeSpec, cache_root: Path) -> tuple[Path, Path]:
    root = Path(cache_root).expanduser().resolve()
    return root / f"{spec.key}.json", root / f"{spec.key}.npz"


def _metadata_matches_spec(metadata: Mapping[str, object], spec: MoleculeSpec) -> bool:
    """Accept the archived cache format only when all Hamiltonian inputs match."""

    return (
        metadata.get("cache_format_version") == CACHE_FORMAT_VERSION
        and metadata.get("molecule") == spec.key
        and metadata.get("geometry") == spec.geometry
        and metadata.get("basis") == spec.basis
        and metadata.get("charge") == spec.charge
        and metadata.get("multiplicity") == spec.multiplicity
        and metadata.get("active_electrons") == spec.active_electrons
        and metadata.get("active_orbitals") == spec.active_orbitals
        and metadata.get("target_time") == spec.target_time
        and metadata.get("localization") == spec.localization
    )


def _validate_cache_payload(
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    spec: MoleculeSpec,
) -> None:
    required_arrays = {
        "hamiltonian_real",
        "hamiltonian_imag",
        "ground_state_real",
        "ground_state_imag",
        "sector_indices",
        "pauli_labels",
        "pauli_coefficients",
        "pauli_support_sizes",
    }
    missing = sorted(required_arrays - arrays.keys())
    if missing:
        raise ValueError(f"cache for {spec.key} is missing arrays: {', '.join(missing)}")

    n_qubits = int(metadata.get("n_qubits", -1))
    if n_qubits != 2 * spec.active_orbitals:
        raise ValueError(
            f"cache for {spec.key} has n_qubits={n_qubits}; expected {2 * spec.active_orbitals}"
        )
    dimension = 1 << n_qubits
    matrix_shape = (dimension, dimension)
    for name in ("hamiltonian_real", "hamiltonian_imag"):
        if np.asarray(arrays[name]).shape != matrix_shape:
            raise ValueError(
                f"cache array {name} for {spec.key} has shape "
                f"{np.asarray(arrays[name]).shape}; expected {matrix_shape}"
            )
    for name in ("ground_state_real", "ground_state_imag"):
        if np.asarray(arrays[name]).shape != (dimension,):
            raise ValueError(
                f"cache array {name} for {spec.key} has shape "
                f"{np.asarray(arrays[name]).shape}; expected {(dimension,)}"
            )

    sector_indices = np.asarray(arrays["sector_indices"], dtype=np.int64)
    if sector_indices.ndim != 1 or sector_indices.size == 0:
        raise ValueError(f"cache for {spec.key} has invalid sector_indices")
    if np.any(sector_indices < 0) or np.any(sector_indices >= dimension):
        raise ValueError(f"cache for {spec.key} has out-of-range sector indices")
    if np.unique(sector_indices).size != sector_indices.size:
        raise ValueError(f"cache for {spec.key} has duplicate sector indices")

    labels = np.asarray(arrays["pauli_labels"])
    coefficients = np.asarray(arrays["pauli_coefficients"], dtype=np.float64)
    support_sizes = np.asarray(arrays["pauli_support_sizes"], dtype=np.int64)
    expected_terms = int(metadata.get("total_pauli_rotation_terms", -1))
    if not (
        labels.ndim == coefficients.ndim == support_sizes.ndim == 1
        and labels.size == coefficients.size == support_sizes.size == expected_terms
    ):
        raise ValueError(f"cache for {spec.key} has inconsistent Pauli arrays")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"cache for {spec.key} has non-finite Pauli coefficients")
    for raw_label, support_size in zip(labels.tolist(), support_sizes.tolist(), strict=True):
        label = str(raw_label)
        if len(label) != n_qubits or set(label) - set("IXYZ"):
            raise ValueError(f"cache for {spec.key} has invalid Pauli label {label!r}")
        if int(support_size) != sum(char != "I" for char in label):
            raise ValueError(f"cache for {spec.key} has wrong support size for {label}")

    ground_state = np.asarray(
        arrays["ground_state_real"] + 1j * arrays["ground_state_imag"],
        dtype=np.complex128,
    )
    if not np.all(np.isfinite(ground_state)) or not math.isclose(
        float(np.linalg.norm(ground_state)),
        1.0,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"cache for {spec.key} has an invalid ground state")
    for key in ("identity_coefficient", "ground_state_energy_hartree", "target_time"):
        if not math.isfinite(float(metadata.get(key, math.nan))):
            raise ValueError(f"cache for {spec.key} has invalid metadata field {key}")


def _build_problem_cache(spec: MoleculeSpec, cache_root: Path) -> dict[str, object]:
    Path(cache_root).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    one_body, two_body, constant = _active_space_integrals(spec)
    hamiltonian, identity_coefficient, pauli_terms = qubit_problem_from_integrals(
        one_body,
        two_body,
        constant,
    )
    n_qubits = 2 * spec.active_orbitals
    initial_state = hartree_fock_state(spec.active_electrons, spec.active_orbitals)
    sector_eigenvalues, sector_eigenvectors, sector_indices = fixed_electron_sector_eigensystem(
        hamiltonian,
        n_qubits,
        spec.active_electrons,
    )
    ground_state = np.asarray(sector_eigenvectors[:, 0], dtype=np.complex128)
    ground_state_energy = float(sector_eigenvalues[0])
    hf_ground_state_overlap = float(abs(np.vdot(ground_state, initial_state)) ** 2)

    identity = np.eye(hamiltonian.shape[0], dtype=np.complex128)
    centered_hamiltonian = hamiltonian - (np.trace(hamiltonian) / hamiltonian.shape[0]) * identity
    centered_norm = float(np.linalg.norm(centered_hamiltonian, ord=2))
    abs_alphas = [abs(float(term["coefficient"])) * float(spec.target_time) for term in pauli_terms]

    metadata: dict[str, object] = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "molecule": spec.key,
        "display_name": spec.display_name,
        "basis": spec.basis,
        "charge": spec.charge,
        "multiplicity": spec.multiplicity,
        "active_electrons": spec.active_electrons,
        "active_orbitals": spec.active_orbitals,
        "cas_label": spec.cas_label,
        "target_time": spec.target_time,
        "chemical_accuracy_kcal_per_mol": 1.0,
        "chemical_accuracy_hartree": REACTION_TARGET_HARTREE,
        "ground_state_energy_hartree": ground_state_energy,
        "ground_state_sector": "fixed_electron_number",
        "hf_ground_state_overlap": hf_ground_state_overlap,
        "spectral_peak_weight_threshold": SPECTRAL_PEAK_WEIGHT_THRESHOLD,
        "centered_hamiltonian_norm": centered_norm,
        "identity_coefficient": float(identity_coefficient),
        "total_pauli_rotation_terms": len(pauli_terms),
        "max_abs_total_alpha": max(abs_alphas),
        "min_abs_total_alpha": min(abs_alphas),
        "n_qubits": n_qubits,
        "localization": spec.localization,
        "geometry": spec.geometry,
    }
    arrays = {
        "one_body": np.asarray(one_body, dtype=np.float64),
        "two_body": np.asarray(two_body, dtype=np.float64),
        "hamiltonian_real": np.real(hamiltonian),
        "hamiltonian_imag": np.imag(hamiltonian),
        "initial_state_real": np.real(initial_state),
        "initial_state_imag": np.imag(initial_state),
        "ground_state_real": np.real(ground_state),
        "ground_state_imag": np.imag(ground_state),
        "sector_eigenvalues": np.asarray(sector_eigenvalues, dtype=np.float64),
        "sector_eigenvectors_real": np.real(sector_eigenvectors),
        "sector_eigenvectors_imag": np.imag(sector_eigenvectors),
        "sector_indices": np.asarray(sector_indices, dtype=np.int64),
        "pauli_labels": np.asarray(
            [item["label"] for item in pauli_terms],
            dtype=f"<U{n_qubits}",
        ),
        "pauli_coefficients": np.asarray(
            [item["coefficient"] for item in pauli_terms],
            dtype=np.float64,
        ),
        "pauli_support_sizes": np.asarray(
            [item["support_size"] for item in pauli_terms],
            dtype=np.int64,
        ),
    }

    json_path, npz_path = _cache_paths(spec, cache_root)
    token = f"{os.getpid()}-{time.time_ns()}"
    json_tmp = json_path.with_name(f".{json_path.name}.{token}.tmp")
    npz_tmp = npz_path.with_name(f".{npz_path.name}.{token}.tmp")
    try:
        json_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with npz_tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        npz_tmp.replace(npz_path)
        json_tmp.replace(json_path)
    finally:
        json_tmp.unlink(missing_ok=True)
        npz_tmp.unlink(missing_ok=True)
    _validate_cache_payload(metadata, arrays, spec)
    return {"metadata": metadata, "arrays": arrays}


def _load_problem_data(
    spec: MoleculeSpec,
    cache_root: Path,
    *,
    rebuild_cache: bool = False,
) -> dict[str, object]:
    json_path, npz_path = _cache_paths(spec, cache_root)
    if rebuild_cache:
        return _build_problem_cache(spec, cache_root)
    missing = [str(path) for path in (json_path, npz_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"paper evaluation requires a prebuilt cache for {spec.key}; "
            f"missing: {', '.join(missing)}. Use rebuild_cache=True only for "
            "an explicit diagnostic cache build."
        )
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read cache metadata {json_path}: {exc}") from exc
    if not isinstance(metadata, dict) or not _metadata_matches_spec(metadata, spec):
        raise ValueError(
            f"cache metadata does not match fixed MoleculeSpec for {spec.key}: {json_path}"
        )
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"could not read cache arrays {npz_path}: {exc}") from exc
    _validate_cache_payload(metadata, arrays, spec)
    return {"metadata": metadata, "arrays": arrays}


def build_reaction_problem(
    spec: MoleculeSpec,
    cache_root: Path,
    *,
    include_compiler: bool = True,
    rebuild_cache: bool = False,
) -> dict[str, object]:
    """Load or build one explicitly specified reaction Hamiltonian."""

    payload = _load_problem_data(spec, cache_root, rebuild_cache=rebuild_cache)
    metadata = payload["metadata"]
    arrays = payload["arrays"]
    assert isinstance(metadata, Mapping)
    assert isinstance(arrays, Mapping)

    ground_state = np.asarray(
        arrays["ground_state_real"] + 1j * arrays["ground_state_imag"],
        dtype=np.complex128,
    )
    pauli_terms: list[dict[str, object]] = []
    for raw_label, raw_coefficient, raw_support_size in zip(
        arrays["pauli_labels"].tolist(),
        arrays["pauli_coefficients"].tolist(),
        arrays["pauli_support_sizes"].tolist(),
        strict=True,
    ):
        label = str(raw_label)
        pauli_terms.append(
            {
                "label": label,
                "coefficient": float(raw_coefficient),
                "support_size": int(raw_support_size),
                "x_count": label.count("X"),
                "y_count": label.count("Y"),
                "z_count": label.count("Z"),
            }
        )

    n_qubits = int(metadata["n_qubits"])
    return {
        "species": spec.key,
        "display_name": str(metadata["display_name"]),
        "identity_coefficient": float(metadata["identity_coefficient"]),
        "pauli_terms": tuple(pauli_terms),
        "ground_state": ground_state,
        "sector_indices": np.asarray(arrays["sector_indices"], dtype=np.int64),
        "target_time": float(metadata["target_time"]),
        "time_sign": 1.0,
        "ground_state_energy_hartree": float(metadata["ground_state_energy_hartree"]),
        "hf_ground_state_overlap": float(metadata["hf_ground_state_overlap"]),
        "total_pauli_rotation_terms": int(metadata["total_pauli_rotation_terms"]),
        "n_qubits": n_qubits,
        "cas_label": str(metadata["cas_label"]),
        "basis": str(metadata["basis"]),
        "localization": str(metadata["localization"]),
        "geometry": str(metadata["geometry"]),
        "compile_pauli_rotation": (
            build_pauli_rotation_compiler(n_qubits) if include_compiler else None
        ),
    }


def _one_step_rotations(
    pauli_terms: Sequence[Mapping[str, object]],
    *,
    formula_order: int,
    time_sign: float,
    step_time: float,
) -> list[tuple[Mapping[str, object], float]]:
    if formula_order == 1:
        return [(term, time_sign * float(term["coefficient"]) * step_time) for term in pauli_terms]
    if formula_order == 2:
        if not pauli_terms:
            return []
        outer_terms = pauli_terms[:-1]
        middle_term = pauli_terms[-1]
        rotations = [
            (term, 0.5 * time_sign * float(term["coefficient"]) * step_time) for term in outer_terms
        ]
        rotations.append(
            (
                middle_term,
                time_sign * float(middle_term["coefficient"]) * step_time,
            )
        )
        rotations.extend(
            (
                term,
                0.5 * time_sign * float(term["coefficient"]) * step_time,
            )
            for term in reversed(outer_terms)
        )
        return rotations
    raise ValueError(f"formula_order must be 1 or 2, got {formula_order}")


def _single_qubit_paulis() -> dict[str, np.ndarray]:
    return {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }


_PAULIS = _single_qubit_paulis()
_HADAMARD = (1.0 / np.sqrt(2.0)) * np.asarray(
    [[1.0, 1.0], [1.0, -1.0]],
    dtype=np.complex128,
)
_S_DAGGER = np.asarray([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)


def _basis_forward_matrix(term_pauli: str) -> np.ndarray:
    if term_pauli == "X":
        return _HADAMARD
    if term_pauli == "Y":
        return _HADAMARD @ _S_DAGGER
    return _PAULIS["I"]


def _identify_pauli(matrix: np.ndarray) -> tuple[complex, str]:
    coefficients = {
        label: 0.5 * np.trace(pauli.conj().T @ matrix) for label, pauli in _PAULIS.items()
    }
    label = max(coefficients, key=lambda item: abs(coefficients[item]))
    coefficient = complex(coefficients[label])
    if not np.allclose(matrix, coefficient * _PAULIS[label], atol=1.0e-12, rtol=0.0):
        raise ValueError("Clifford conjugation did not produce a Pauli operator")
    return coefficient, label


def _local_basis_conjugation(
    term_pauli: str,
    operator_pauli: str,
) -> tuple[complex, str]:
    basis = _basis_forward_matrix(term_pauli)
    transformed = basis.conj().T @ _PAULIS[operator_pauli] @ basis
    return _identify_pauli(transformed)


def _conjugated_pivot_pauli(label: str, pivot_pauli: str) -> tuple[complex, str]:
    support = [index for index, char in enumerate(label) if char != "I"]
    if not support:
        raise ValueError("cannot conjugate an identity Pauli gadget")
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
        local_phase, transformed_pauli = _local_basis_conjugation(
            label[qubit],
            operator_pauli,
        )
        phase *= local_phase
        operator[qubit] = transformed_pauli
    return phase, "".join(operator)


def _pauli_left_action(label: str) -> tuple[np.ndarray, np.ndarray]:
    n_qubits = len(label)
    source = np.arange(1 << n_qubits, dtype=np.int64)
    destination = source.copy()
    phase = np.ones(source.size, dtype=np.complex128)
    for qubit, pauli in enumerate(label):
        if pauli == "I":
            continue
        bit_mask = 1 << (n_qubits - qubit - 1)
        bit_is_one = (source & bit_mask) != 0
        if pauli == "X":
            destination ^= bit_mask
        elif pauli == "Y":
            destination ^= bit_mask
            phase *= np.where(bit_is_one, -1.0j, 1.0j)
        elif pauli == "Z":
            phase *= np.where(bit_is_one, -1.0, 1.0)
        else:
            raise ValueError(f"invalid Pauli label character {pauli!r}")
    return destination, phase


def _apply_compiled_rotation_to_unitary(
    unitary: np.ndarray,
    *,
    label: str,
    single_qubit_unitary: np.ndarray,
    action_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    decomposition = {
        pauli: complex(0.5 * np.trace(matrix.conj().T @ single_qubit_unitary))
        for pauli, matrix in _PAULIS.items()
    }
    updated = decomposition["I"] * unitary
    for pivot_pauli in ("X", "Y", "Z"):
        coefficient = decomposition[pivot_pauli]
        if abs(coefficient) <= 1.0e-15:
            continue
        scaffold_phase, transformed_label = _conjugated_pivot_pauli(
            label,
            pivot_pauli,
        )
        action = action_cache.get(transformed_label)
        if action is None:
            action = _pauli_left_action(transformed_label)
            action_cache[transformed_label] = action
        destination, pauli_phase = action
        pauli_unitary = np.empty_like(unitary)
        pauli_unitary[destination, :] = pauli_phase[:, None] * unitary
        updated += coefficient * scaffold_phase * pauli_unitary
    unitary[:] = updated


def _config_dict(config: object) -> dict[str, object]:
    if is_dataclass(config):
        payload = asdict(config)
    else:
        payload = dict(getattr(config, "__dict__", {}))
    return {str(key): _json_scalar(value) for key, value in payload.items()}


def _json_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"method config value is not JSON-safe: {type(value).__name__}")


def _validated_epsilon(
    policy: ModuleType,
    term: Mapping[str, object],
    alpha: float,
    step_index: int,
    config: object,
    n_terms: int,
) -> float:
    epsilon = float(
        policy.choose_synthesis_epsilon(
            dict(term),
            alpha,
            step_index,
            config,
            n_terms,
        )
    )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            f"choose_synthesis_epsilon returned invalid epsilon {epsilon} "
            f"for {term['label']} at step {step_index}"
        )
    return epsilon


def _build_compiled_one_step_unitary(
    problem: Mapping[str, object],
    policy: ModuleType,
    config: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compile and apply one stationary PF step in the full Hilbert space."""

    formula_order = int(getattr(config, "formula_order"))
    n_steps = int(getattr(config, "n_steps"))
    zero_angle_threshold = float(getattr(config, "zero_angle_threshold"))
    if formula_order not in {1, 2}:
        raise ValueError(f"formula_order must be 1 or 2, got {formula_order}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if not math.isfinite(zero_angle_threshold) or zero_angle_threshold < 0.0:
        raise ValueError(
            f"zero_angle_threshold must be finite and nonnegative, got {zero_angle_threshold}"
        )

    pauli_terms = tuple(problem["pauli_terms"])
    step_time = float(problem["target_time"]) / float(n_steps)
    rotations = _one_step_rotations(
        pauli_terms,
        formula_order=formula_order,
        time_sign=float(problem["time_sign"]),
        step_time=step_time,
    )
    compile_pauli_rotation = problem.get("compile_pauli_rotation")
    if compile_pauli_rotation is None:
        raise ValueError("compiled evaluation requires a PyGridSynth compiler")

    _gridsynth_gates, d_omega_unitary_cls = import_pygridsynth_backend()
    matrix_cache: dict[str, np.ndarray] = {}
    action_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dim = 1 << int(problem["n_qubits"])
    unitary = np.eye(dim, dtype=np.complex128)
    one_step_t_gate_count = 0
    one_step_nonzero = 0
    one_step_skipped = 0
    one_step_skipped_abs_angle_sum = 0.0
    one_step_skipped_error_bound = 0.0
    max_skipped_abs_alpha = 0.0
    epsilon_histogram: dict[str, int] = {}
    unique_compilations: dict[str, dict[str, object]] = {}

    for term, alpha in rotations:
        abs_alpha = abs(float(alpha))
        if abs_alpha <= zero_angle_threshold:
            one_step_skipped += 1
            one_step_skipped_abs_angle_sum += abs_alpha
            one_step_skipped_error_bound += 2.0 * abs(math.sin(0.5 * abs_alpha))
            max_skipped_abs_alpha = max(max_skipped_abs_alpha, abs_alpha)
            continue

        epsilon = _validated_epsilon(
            policy,
            term,
            alpha,
            0,
            config,
            len(pauli_terms),
        )
        # A one-step eigenspectrum is valid only for a stationary PF policy.
        # Validate every requested step instead of silently scaling a step-0
        # policy that changes with ``step_index``.
        for step_index in range(1, n_steps):
            later_epsilon = _validated_epsilon(
                policy,
                term,
                alpha,
                step_index,
                config,
                len(pauli_terms),
            )
            if later_epsilon != epsilon:
                raise ValueError(
                    "full-space one-step diagonalization requires a stationary "
                    "synthesis policy; epsilon changed for "
                    f"{term['label']} between steps 0 and {step_index}"
                )

        compiled = compile_pauli_rotation(str(term["label"]), float(alpha), epsilon)
        gate_string = str(compiled["gate_string"])
        single_qubit_unitary = matrix_cache.get(gate_string)
        if single_qubit_unitary is None:
            single_qubit_unitary = np.asarray(
                pygridsynth_matrix_from_gates(gate_string, d_omega_unitary_cls),
                dtype=np.complex128,
            )
            matrix_cache[gate_string] = single_qubit_unitary
        _apply_compiled_rotation_to_unitary(
            unitary,
            label=str(term["label"]),
            single_qubit_unitary=single_qubit_unitary,
            action_cache=action_cache,
        )

        one_step_t_gate_count += int(compiled["t_gate_count"])
        one_step_nonzero += 1
        cache_key = str(compiled["cache_key"])
        unique_compilations[cache_key] = {
            "label": str(compiled["label"]),
            "alpha": float(compiled["alpha"]),
            "theta": float(compiled["theta"]),
            "epsilon": float(compiled["epsilon"]),
            "t_gate_count": int(compiled["t_gate_count"]),
            "gate_count": int(compiled["gate_count"]),
            "operator_error": float(compiled["operator_error"]),
        }
        epsilon_key = f"{float(compiled['epsilon']):.3e}"
        epsilon_histogram[epsilon_key] = epsilon_histogram.get(epsilon_key, 0) + n_steps

    return unitary, {
        "method_config": _config_dict(config),
        "formula_order": formula_order,
        "n_steps": n_steps,
        "rotations_per_step": len(rotations),
        "scheduled_rotation_count": len(rotations) * n_steps,
        "nonzero_rotation_count": one_step_nonzero * n_steps,
        "skipped_rotation_count": one_step_skipped * n_steps,
        "one_step_nonzero_rotation_count": one_step_nonzero,
        "one_step_skipped_rotation_count": one_step_skipped,
        "zero_angle_threshold": zero_angle_threshold,
        "one_step_skipped_absolute_angle_sum": one_step_skipped_abs_angle_sum,
        "skipped_absolute_angle_sum": one_step_skipped_abs_angle_sum * n_steps,
        "one_step_skipped_rotation_operator_error_bound": (one_step_skipped_error_bound),
        "skipped_rotation_operator_error_bound": one_step_skipped_error_bound * n_steps,
        "max_skipped_absolute_angle": max_skipped_abs_alpha,
        "one_step_t_gate_count": one_step_t_gate_count,
        "t_gate_count": one_step_t_gate_count * n_steps,
        "unique_compilation_count": len(unique_compilations),
        "unique_compilations": sorted(
            unique_compilations.values(),
            key=lambda item: (str(item["label"]), float(item["epsilon"])),
        ),
        "synthesis_epsilon_histogram": dict(sorted(epsilon_histogram.items())),
        "step_time": step_time,
        "full_hilbert_dimension": dim,
        "unitary_construction": (
            "compiled_one_step_product_formula_applied_and_diagonalized_in_full_hilbert_space"
        ),
        "phase_matching": "maximum_ground_state_eigenvector_overlap",
        "stationary_step_policy_verified": True,
        "t_count_semantics": (
            "exact_count_of_T_characters_in_selected_PyGridSynth_gate_strings_times_n_steps"
        ),
    }


def evaluate_species(
    spec: MoleculeSpec,
    policy: ModuleType,
    config: object,
    cache_root: Path,
) -> dict[str, object]:
    """Evaluate one species with compiled full-space one-step diagonalization."""

    started = time.perf_counter()
    problem = build_reaction_problem(
        spec,
        Path(cache_root),
        include_compiler=True,
        rebuild_cache=False,
    )
    unitary_step, run_metrics = _build_compiled_one_step_unitary(
        problem,
        policy,
        config,
    )
    eigenvalues, eigenvectors = np.linalg.eig(unitary_step)
    target_state = np.asarray(problem["ground_state"], dtype=np.complex128)
    target_state /= np.linalg.norm(target_state)
    eigenvector_norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    if np.any(eigenvector_norms <= 0.0):
        raise ValueError("compiled one-step unitary produced a zero-norm eigenvector")
    eigenvectors = eigenvectors / eigenvector_norms
    overlaps = np.abs(eigenvectors.conj().T @ target_state) ** 2
    selected = int(np.argmax(overlaps))

    identity_shift = float(problem["identity_coefficient"])
    exact_energy = float(problem["ground_state_energy_hartree"])
    target_time = float(problem["target_time"])
    reference_centered_energy = exact_energy - identity_shift
    step_phase = float(np.angle(eigenvalues[selected]))
    total_phase = int(getattr(config, "n_steps")) * step_phase
    centered_estimate = unwrap_energy_near_reference(
        total_phase,
        target_time,
        reference_centered_energy,
    )
    energy_estimate = float(centered_estimate + identity_shift)
    signed_error = energy_estimate - exact_energy

    return {
        "species": spec.key,
        "display_name": spec.display_name,
        "exact_energy_hartree": exact_energy,
        "pf_energy_estimate_hartree": energy_estimate,
        "signed_energy_error_hartree": signed_error,
        "absolute_energy_error_hartree": abs(signed_error),
        "signed_energy_error_mha": 1000.0 * signed_error,
        "absolute_energy_error_mha": 1000.0 * abs(signed_error),
        "selected_compiled_unitary_eigen_index": selected,
        "selected_compiled_unitary_ground_overlap": float(overlaps[selected]),
        "selected_compiled_step_eigenphase_radians": step_phase,
        "selected_compiled_total_eigenphase_radians": total_phase,
        "max_eigenvalue_modulus_error": float(np.max(np.abs(np.abs(eigenvalues) - 1.0))),
        "n_qubits": int(problem["n_qubits"]),
        "cas_label": str(problem["cas_label"]),
        "basis": str(problem["basis"]),
        "localization": str(problem["localization"]),
        "total_pauli_rotation_terms": int(problem["total_pauli_rotation_terms"]),
        "evaluation_wall_seconds": time.perf_counter() - started,
        **run_metrics,
    }


def validate_policy_module(policy: ModuleType, species_keys: Sequence[str]) -> None:
    required = (
        "set_active_species",
        "get_method_config",
        "choose_synthesis_epsilon",
    )
    missing = [name for name in required if not callable(getattr(policy, name, None))]
    if missing:
        raise ValueError(f"policy is missing required callables: {', '.join(missing)}")
    for species_key in species_keys:
        policy.set_active_species(species_key)
        config = policy.get_method_config()
        if int(getattr(config, "formula_order")) not in {1, 2}:
            raise ValueError(f"{species_key}: formula_order must be 1 or 2")
        if int(getattr(config, "n_steps")) <= 0:
            raise ValueError(f"{species_key}: n_steps must be positive")
        threshold = float(getattr(config, "zero_angle_threshold"))
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"{species_key}: zero_angle_threshold must be finite and nonnegative")
        sample_term: dict[str, object] = {
            "label": "IXYZ",
            "coefficient": 0.03125,
            "support_size": 3,
            "x_count": 1,
            "y_count": 1,
            "z_count": 1,
        }
        _validated_epsilon(
            policy,
            sample_term,
            1.0e-4,
            0,
            config,
            12,
        )


__all__ = [
    "ARCHIVED_EVALUATOR_SHA256",
    "CACHE_FORMAT_VERSION",
    "HARTREE_PER_KCAL_PER_MOL",
    "REACTION_TARGET_HARTREE",
    "REACTION_TARGET_MHA",
    "build_reaction_problem",
    "evaluate_species",
    "validate_policy_module",
]
