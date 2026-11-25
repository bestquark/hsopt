"""Versioned ideal-ground-input QPE contract for shared protocol experiments.

Reuse the exact ground vector already used for tight calibration. The compiled
QPE peak is the eigenspace with the largest probability from that SAME vector;
partial Lanczos must certify more than half its norm in this peak. This measures
simulation error, without attributing HF state-preparation error to the protocol.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ENERGY_TARGET = "ground_branch"
CONTRACT_VERSION = "normalized_hamiltonian_ground_input_qpe_t_count_cudaq_elote_v3"
DIFFERENCE_CONTRACT_VERSION = "normalized_hamiltonian_ground_input_signed_difference_cudaq_elote_v1"
REFERENCE_TARGET = "exact_ground_state_branch"


def cached_reference(problem: Mapping[str, Any], reference: Mapping[str, Any]):
    """Validate the cached ground input; never recompute H or its spectrum."""
    from . import compiled_unitary as ev

    if (
        reference.get("observable") != "exact_unitary_ground_eigenphase"
        or reference.get("evaluation_contract_version") not in ev.EXACT_REFERENCE_CONTRACT_VERSIONS
        or reference.get("problem_sha256") != ev._sha256(Path(problem["_path"]))
        or reference.get("target_state_definition") != ev.TARGET_STATE_DEFINITION
        or reference.get("n_qubits") != problem["n_qubits"]
        or reference.get("electrons") != problem["electrons"]
        or reference.get("target_time") != problem["target_time"]
    ):
        raise ValueError("ground-input reference does not match the Hamiltonian/calibration target")
    energy = float(reference["exact_unitary_ground_energy_hartree"])
    serialized = np.asarray(reference["exact_ground_state"], dtype=np.float64)
    if serialized.shape != (1 << problem["n_qubits"], 2) or not np.isfinite(serialized).all():
        raise ValueError("cached exact ground input has invalid shape or amplitudes")
    state = serialized[:, 0] + 1j * serialized[:, 1]
    if not math.isfinite(energy) or abs(float(np.vdot(state, state).real) - 1.0) > 1e-9:
        raise ValueError("cached exact ground input must be normalized with finite energy")
    outside = [i for i in range(state.size) if i.bit_count() != problem["electrons"]]
    if float(np.sum(np.abs(state[outside]) ** 2)) > 1e-12:
        raise ValueError("cached ground input lies outside the fixed-electron sector")
    return energy, state


def select_peak(phases, probabilities, *, n_steps, target_time, identity_coefficient):
    """Basis-independent highest-weight ground-input QPE peak."""
    from . import compiled_unitary as ev

    phases, probabilities = (ev._host_array(a, dtype=np.float64) for a in (phases, probabilities))
    if phases.ndim != 1 or probabilities.shape != phases.shape or not phases.size:
        raise ValueError("ground-input phases and probabilities must be matching nonempty vectors")
    if (
        not np.isfinite(phases).all()
        or not np.isfinite(probabilities).all()
        or min(probabilities) < 0
    ):
        raise ValueError("ground-input spectral probabilities must be finite and nonnegative")
    energies = [
        ev._decode_energy(
            n_steps * p, target_time=target_time, identity_coefficient=identity_coefficient
        )
        for p in phases
    ]
    groups = []
    for i in sorted(range(len(energies)), key=energies.__getitem__):
        if groups and abs(energies[i] - energies[groups[-1][0]]) <= 1e-10:
            groups[-1].append(i)
        else:
            groups.append([i])
    selected = max(groups, key=lambda g: float(probabilities[g].sum()))
    probability = min(1.0, float(probabilities[selected].sum()))
    # A partial spectrum cannot hide a stronger competing peak when this peak
    # already owns a majority of the normalized input. Never guess otherwise.
    if probability <= 0.5:
        raise RuntimeError("Lanczos has not certified a dominant ground-input QPE peak")
    i = selected[0]
    return {
        "selected_eigen_index": i,
        "selected_eigen_indices": selected,
        "selected_step_eigenphase_radians": float(phases[i]),
        "selected_total_eigenphase_radians": float(n_steps * phases[i]),
        "selected_principal_eigenphase_radians": ev._principal_phase(n_steps * phases[i]),
        "compiled_pf_energy_hartree": float(energies[i]),
        "qpe_success_probability_from_ground_input": probability,
        "selected_exact_ground_overlap": probability,
        "phase_branch_selection": "principal_phase_without_exact_energy_reference",
    }
