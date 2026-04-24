"""
Final-state fidelity evaluator for product-formula Hamiltonian simulation.

The score is the compiled T count plus a large penalty when the product-formula
state is farther than the allowed infidelity budget from exact exp(+i H t).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict

try:
    from .array_backend import array_backend_name, np
    from .chemistry import (
        ACTIVE_MOLECULES,
        build_problem,
        fidelity,
        load_method_module,
        score_from_error_budget,
        t_gate_metrics,
    )
except ImportError:
    from array_backend import array_backend_name, np
    from chemistry import (
        ACTIVE_MOLECULES,
        build_problem,
        fidelity,
        load_method_module,
        score_from_error_budget,
        t_gate_metrics,
    )


def evaluate(molecule: str, script_path: Path, epsilon: float | None = None) -> Dict[str, object]:
    start = time.perf_counter()
    module = load_method_module(script_path)
    problem = build_problem(molecule)

    if epsilon is not None and epsilon <= 0.0:
        raise ValueError(f"fidelity epsilon must be positive, got {epsilon}")
    fidelity_floor = float(problem["chemical_accuracy_fidelity_floor"])
    default_error_budget = max(1e-15, 1.0 - fidelity_floor)
    error_budget = float(default_error_budget if epsilon is None else epsilon)

    quantized = module.run_product_formula(problem, exact_angles=False)
    exact_state = np.asarray(problem["exact_state"], dtype=np.complex128)
    quantized_state = np.asarray(quantized["state"], dtype=np.complex128)
    fidelity_value = fidelity(exact_state, quantized_state)
    infidelity = max(0.0, 1.0 - fidelity_value)
    passes_constraint = infidelity <= error_budget

    hamiltonian = np.asarray(problem["hamiltonian"], dtype=np.complex128)
    hf_energy = float(np.real(np.vdot(problem["initial_state"], hamiltonian @ problem["initial_state"])))
    exact_final_energy = float(np.real(np.vdot(exact_state, hamiltonian @ exact_state)))
    quantized_final_energy = float(np.real(np.vdot(quantized_state, hamiltonian @ quantized_state)))

    t_count = int(quantized["t_gate_count"])
    result = {
        "experiment": problem["experiment"],
        "molecule": molecule,
        "display_name": problem["display_name"],
        "script": str(script_path),
        "objective": "final_state_fidelity",
        "score": score_from_error_budget(t_count, error_budget, infidelity),
        "passes_accuracy_constraint": passes_constraint,
        "passes_fidelity_constraint": passes_constraint,
        "accuracy_metric_name": "infidelity",
        "accuracy_metric_value": infidelity,
        "accuracy_threshold_name": "fidelity_error_budget",
        "accuracy_threshold_value": error_budget,
        "fidelity": fidelity_value,
        "infidelity": infidelity,
        "fidelity_error_budget": error_budget,
        "fidelity_error_budget_source": "chemical_accuracy_proxy" if epsilon is None else "cli_epsilon",
        "chemical_accuracy_fidelity_floor": fidelity_floor,
        "chemical_accuracy_hartree": float(problem["chemical_accuracy_hartree"]),
        "target_time": float(problem["target_time"]),
        "time_sign": float(problem["time_sign"]),
        "ground_state_energy_hartree": float(problem["ground_state_energy_hartree"]),
        "total_pauli_rotation_terms": int(problem["total_pauli_rotation_terms"]),
        "hamiltonian_factorization": problem["hamiltonian_factorization"],
        "factorization_cost_terms": int(problem["factorization_cost_terms"]),
        "factorization_metadata": problem["factorization_metadata"],
        "identity_coefficient": float(problem["identity_coefficient"]),
        "centered_hamiltonian_norm": float(problem["centered_hamiltonian_norm"]),
        "max_abs_total_alpha": float(problem["max_abs_total_alpha"]),
        "min_abs_total_alpha": float(problem["min_abs_total_alpha"]),
        "hf_energy": hf_energy,
        "exact_final_energy": exact_final_energy,
        "quantized_final_energy": quantized_final_energy,
        "cas_label": problem["cas_label"],
        "basis": problem["basis"],
        "localization": problem["localization"],
        "compilation_backend": problem["compilation_backend"],
        "array_backend": array_backend_name(),
        "evaluation_wall_seconds": time.perf_counter() - start,
    }
    result.update(t_gate_metrics(quantized))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PF dynamics with a final-state fidelity constraint.")
    parser.add_argument("--molecule", required=True, choices=tuple(sorted(ACTIVE_MOLECULES)))
    parser.add_argument("--script", required=True, help="Path to the editable simple_pf.py file.")
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Allowed final-state infidelity. Defaults to the chemistry-derived fidelity proxy.",
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = evaluate(args.molecule, Path(args.script).resolve(), epsilon=args.epsilon)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
