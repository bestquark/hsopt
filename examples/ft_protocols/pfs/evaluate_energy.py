"""
HF-reference spectral-energy evaluator for product-formula Hamiltonian simulation.

The score is the compiled T count for the Hartree-Fock/reference trajectory plus
a large penalty when the lowest resolved spectral peak misses the fixed-sector
ground-state energy by more than chemical accuracy.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from .chemistry import (
        ACTIVE_MOLECULES,
        CHEMICAL_ACCURACY_HARTREE,
        build_problem,
        estimate_reference_peak_energy_from_state,
        fidelity,
        load_method_module,
        score_from_error_budget,
        t_gate_metrics,
    )
except ImportError:
    from chemistry import (
        ACTIVE_MOLECULES,
        CHEMICAL_ACCURACY_HARTREE,
        build_problem,
        estimate_reference_peak_energy_from_state,
        fidelity,
        load_method_module,
        score_from_error_budget,
        t_gate_metrics,
    )


def evaluate(molecule: str, script_path: Path) -> Dict[str, object]:
    start = time.perf_counter()
    module = load_method_module(script_path)
    problem = build_problem(molecule)

    reference_state = np.array(problem["initial_state"], dtype=np.complex128, copy=True)
    reference_problem = dict(problem)
    reference_problem["initial_state"] = reference_state
    reference_run = module.run_product_formula(reference_problem, exact_angles=False)

    exact_state = np.array(problem["exact_state"], dtype=np.complex128, copy=False)
    diagnostic_state = np.array(reference_run["state"], dtype=np.complex128, copy=False)

    exact_ground_energy = float(problem["ground_state_energy_hartree"])
    phase_metrics = estimate_reference_peak_energy_from_state(
        reference_state,
        diagnostic_state,
        np.array(problem["sector_eigenvalues"], dtype=np.float64, copy=False),
        np.array(problem["sector_eigenvectors"], dtype=np.complex128, copy=False),
        float(problem["target_time"]),
        float(problem["identity_coefficient"]),
        float(problem["spectral_peak_weight_threshold"]),
    )
    energy_estimate = phase_metrics["energy_estimate_hartree"]
    energy_error = None if energy_estimate is None else float(abs(float(energy_estimate) - exact_ground_energy))
    passes_constraint = (
        energy_error is not None
        and float(energy_error) <= CHEMICAL_ACCURACY_HARTREE
    )

    state_fidelity = fidelity(exact_state, diagnostic_state)
    state_infidelity = max(0.0, 1.0 - state_fidelity)
    hamiltonian = np.array(problem["hamiltonian"], dtype=np.complex128, copy=False)
    hf_energy = float(np.real(np.vdot(problem["initial_state"], hamiltonian @ problem["initial_state"])))
    exact_final_energy = float(np.real(np.vdot(exact_state, hamiltonian @ exact_state)))

    t_count = int(reference_run["t_gate_count"])
    result = {
        "experiment": problem["experiment"],
        "molecule": molecule,
        "display_name": problem["display_name"],
        "script": str(script_path),
        "objective": "hf_reference_spectral_energy",
        "spectral_estimator": "lowest_resolved_peak_projection",
        "reference_state": "hartree_fock",
        "score": score_from_error_budget(t_count, CHEMICAL_ACCURACY_HARTREE, energy_error),
        "passes_accuracy_constraint": passes_constraint,
        "passes_energy_constraint": passes_constraint,
        "accuracy_metric_name": "ground_state_energy_error_hartree",
        "accuracy_metric_value": energy_error,
        "accuracy_threshold_name": "chemical_accuracy_hartree",
        "accuracy_threshold_value": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_fidelity_floor": float(problem["chemical_accuracy_fidelity_floor"]),
        "target_time": float(problem["target_time"]),
        "time_sign": float(problem["time_sign"]),
        "ground_state_sector": problem["ground_state_sector"],
        "ground_state_energy_hartree": exact_ground_energy,
        "ground_state_energy_estimate_hartree": energy_estimate,
        "ground_state_energy_error_hartree": energy_error,
        "hf_ground_state_overlap": float(problem["hf_ground_state_overlap"]),
        "resolved_spectral_peak_count": phase_metrics["resolved_peak_count"],
        "selected_spectral_peak_index": phase_metrics["peak_index"],
        "selected_spectral_peak_energy_hartree": phase_metrics["peak_energy_hartree"],
        "selected_spectral_peak_weight": phase_metrics["peak_weight"],
        "selected_spectral_peak_phase_error_hartree": phase_metrics["energy_error_hartree"],
        "spectral_peak_phase_overlap_magnitude": phase_metrics["phase_overlap_magnitude"],
        "spectral_peak_weight_threshold": float(problem["spectral_peak_weight_threshold"]),
        "fidelity": state_fidelity,
        "infidelity": state_infidelity,
        "state_fidelity_to_exact_evolution": state_fidelity,
        "state_infidelity_to_exact_evolution": state_infidelity,
        "state_diagnostic_t_gate_count": int(reference_run["t_gate_count"]),
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
        "compilation_backend": problem["compilation_backend"],
        "evaluation_wall_seconds": time.perf_counter() - start,
    }
    result.update(t_gate_metrics(reference_run))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PF dynamics with an HF-reference spectral energy constraint.")
    parser.add_argument("--molecule", required=True, choices=tuple(sorted(ACTIVE_MOLECULES)))
    parser.add_argument("--script", required=True, help="Path to the editable simple_pf.py file.")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = evaluate(args.molecule, Path(args.script).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
