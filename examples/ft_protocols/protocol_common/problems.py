from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = EXAMPLE_ROOT / "problem_data"
CHEMICAL_ACCURACY_HARTREE = 1e-3


def molecule_choices() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in DATA_ROOT.glob("*.json")))


def load_protocol_problem(molecule: str) -> Dict[str, object]:
    """Load fixed molecule data for the fault-tolerant protocol examples."""

    json_path = DATA_ROOT / f"{molecule}.json"
    npz_path = DATA_ROOT / f"{molecule}.npz"
    if not json_path.exists() or not npz_path.exists():
        raise KeyError(f"unknown molecule key: {molecule}")

    metadata = json.loads(json_path.read_text())
    arrays = np.load(npz_path, allow_pickle=False)
    chemical_accuracy = float(metadata.get("chemical_accuracy_hartree", CHEMICAL_ACCURACY_HARTREE))

    return {
        "molecule": str(metadata["molecule"]),
        "display_name": str(metadata["display_name"]),
        "metadata": metadata,
        "one_body": np.asarray(arrays["one_body"], dtype=np.float64),
        "two_body": np.asarray(arrays["two_body"], dtype=np.float64),
        "hamiltonian": np.asarray(arrays["hamiltonian_real"], dtype=np.float64)
        + 1j * np.asarray(arrays["hamiltonian_imag"], dtype=np.float64),
        "initial_state": np.asarray(arrays["initial_state_real"], dtype=np.float64)
        + 1j * np.asarray(arrays["initial_state_imag"], dtype=np.float64),
        "exact_state": np.asarray(arrays["exact_state_real"], dtype=np.float64)
        + 1j * np.asarray(arrays["exact_state_imag"], dtype=np.float64),
        "pauli_labels": tuple(str(label) for label in arrays["pauli_labels"]),
        "pauli_coefficients": np.asarray(arrays["pauli_coefficients"], dtype=np.float64),
        "identity_coefficient": float(metadata["identity_coefficient"]),
        "target_time": float(metadata["target_time"]),
        "chemical_accuracy_hartree": chemical_accuracy,
        "ground_state_energy_hartree": float(metadata["ground_state_energy_hartree"]),
        "n_qubits": int(metadata["n_qubits"]),
        "total_pauli_rotation_terms": int(metadata["total_pauli_rotation_terms"]),
        "cas_label": str(metadata["cas_label"]),
        "basis": str(metadata["basis"]),
        "localization": str(metadata["localization"]),
    }


def score_from_accuracy_proxy(
    t_gate_count: int,
    allowed_error_hartree: float,
    actual_error_hartree: float | None,
) -> float:
    if actual_error_hartree is None or not np.isfinite(actual_error_hartree):
        return float(t_gate_count) + 1e9
    excess = max(0.0, float(actual_error_hartree) - float(allowed_error_hartree))
    penalty = 1e9 * (excess / max(float(allowed_error_hartree), 1e-15))
    return float(t_gate_count) + penalty
