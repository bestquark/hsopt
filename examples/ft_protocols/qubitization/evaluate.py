"""
Qubitization-QPE evaluator with DF/THC factorization, effective-Hamiltonian
fidelity, and paper-exact Toffoli counting.

The method file exports ``get_method_config()`` returning a frozen dataclass
with fields: factorization, beth_per_leaf (DF) or beth_per_mu (THC), aleph,
epsilon_qpe.  The evaluator performs:

1.  Factorize two-body integrals (DF or THC).
2.  Build the discretized effective Hamiltonian H_tilde.
3.  Compute exact vs. approximate ground-state energies.
4.  Compute time-evolution fidelity |<e^{-iH_tilde t} psi0 | e^{-iHt} psi0>|^2.
5.  Estimate Toffoli / T-gate counts from the cost formulas.
6.  Assemble and return a comprehensive result dict.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Dict

import numpy as np
from scipy.linalg import expm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protocol_common.cost_formulas import (  # noqa: E402
    df_logical_qubits,
    df_walk_operator_cost,
    qubitization_qpe_cost,
    thc_logical_qubits,
    thc_walk_operator_cost,
)
from protocol_common.factorization import (  # noqa: E402
    build_effective_hamiltonian_df,
    build_effective_hamiltonian_thc,
    compute_e_core,
    df_lambda,
    double_factorization,
    fidelity,
    fixed_electron_sector_eigensystem,
    thc_factorization,
    thc_lambda,
)
from protocol_common.problems import (  # noqa: E402
    load_protocol_problem,
    molecule_choices,
)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def load_method_module(script_path: Path) -> ModuleType:
    """Dynamically load an editable method file and verify it exports
    ``get_method_config``."""
    spec = importlib.util.spec_from_file_location("qubitization_method", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load method file: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "get_method_config"):
        raise AttributeError(
            "method file is missing required symbol 'get_method_config'"
        )
    return module


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

_FIDELITY_BUDGET = 0.01  # 1 % infidelity budget for the score penalty


def evaluate(molecule: str, script_path: Path) -> Dict[str, object]:
    start = time.perf_counter()

    # -- load method config and problem data --------------------------------
    module = load_method_module(script_path)
    config = module.get_method_config()
    problem = load_protocol_problem(molecule)

    one_body = np.asarray(problem["one_body"], dtype=np.float64)
    two_body = np.asarray(problem["two_body"], dtype=np.float64)
    hamiltonian = np.asarray(problem["hamiltonian"], dtype=np.complex128)
    initial_state = np.asarray(problem["initial_state"], dtype=np.complex128)
    n_qubits = int(problem["n_qubits"])
    n_orb = one_body.shape[0]
    n_electrons = int(problem["metadata"]["active_electrons"])
    constant = compute_e_core(one_body, two_body, float(problem["identity_coefficient"]), n_qubits)
    target_time = float(problem["target_time"])
    chemical_accuracy = float(problem["chemical_accuracy_hartree"])
    exact_gs_energy = float(problem["ground_state_energy_hartree"])
    factorization_type = str(config.factorization)
    aleph = int(config.aleph)
    epsilon_qpe = float(config.epsilon_qpe)

    # -- factorize two-body integrals ---------------------------------------
    if factorization_type == "df":
        df_leaves = double_factorization(two_body)
        n_leaves = len(df_leaves)
        xi_per_leaf = [leaf["eigenvalues"].shape[0] for leaf in df_leaves]

        beth_per_leaf = list(config.beth_per_leaf)
        if len(beth_per_leaf) < n_leaves:
            last = beth_per_leaf[-1] if beth_per_leaf else 10
            beth_per_leaf.extend([last] * (n_leaves - len(beth_per_leaf)))
        beth_per_leaf = beth_per_leaf[:n_leaves]

        # effective Hamiltonian
        H_tilde = build_effective_hamiltonian_df(
            one_body, two_body, constant, df_leaves, beth_per_leaf, aleph, n_qubits,
        )

        # lambda (1-norm)
        lam = df_lambda(one_body, df_leaves)

        # walk operator cost
        walk_data = df_walk_operator_cost(
            n_orb, n_leaves, xi_per_leaf, beth_per_leaf, aleph,
        )
        walk_cost_per_step = int(walk_data["total_per_step"])

        # QPE cost
        qpe_data = qubitization_qpe_cost(walk_cost_per_step, lam, epsilon_qpe)
        total_toffoli = int(qpe_data["total_toffoli"])
        total_t_gates = int(qpe_data["total_t_gates"])
        n_iterations = int(qpe_data["n_iterations"])

        # logical qubits
        logical_qubits = df_logical_qubits(
            n_orb,
            n_leaves,
            int(walk_data["total_coefficients"]),
            int(walk_data["beth_max"]),
            aleph,
        )

        factorization_detail = {
            "type": "df",
            "n_leaves": n_leaves,
            "xi_per_leaf": xi_per_leaf,
            "beth_per_leaf": beth_per_leaf,
            "aleph": aleph,
            "lambda": lam,
        }

    elif factorization_type == "thc":
        thc_rank = getattr(config, "thc_rank", None)
        thc_data = thc_factorization(two_body, rank=thc_rank)
        rank = int(thc_data["rank"])

        beth_per_mu = list(config.beth_per_mu)
        if len(beth_per_mu) < rank:
            last = beth_per_mu[-1] if beth_per_mu else 10
            beth_per_mu.extend([last] * (rank - len(beth_per_mu)))
        beth_per_mu = beth_per_mu[:rank]

        # effective Hamiltonian
        H_tilde = build_effective_hamiltonian_thc(
            one_body, two_body, constant, thc_data, beth_per_mu, aleph, n_qubits,
        )

        # lambda (1-norm)
        lam = thc_lambda(one_body, thc_data)

        # walk operator cost
        walk_data = thc_walk_operator_cost(n_orb, rank, beth_per_mu, aleph)
        walk_cost_per_step = int(walk_data["total_per_step"])

        # QPE cost
        qpe_data = qubitization_qpe_cost(walk_cost_per_step, lam, epsilon_qpe)
        total_toffoli = int(qpe_data["total_toffoli"])
        total_t_gates = int(qpe_data["total_t_gates"])
        n_iterations = int(qpe_data["n_iterations"])

        # logical qubits
        d = int(walk_data["d"])
        beth_max = int(walk_data["beth_max"])
        logical_qubits = thc_logical_qubits(
            n_orb, rank, d, beth_max, aleph, n_iterations,
        )

        factorization_detail = {
            "type": "thc",
            "rank": rank,
            "beth_per_mu": beth_per_mu,
            "aleph": aleph,
            "lambda": lam,
            "thc_relative_error": float(thc_data["relative_error"]),
        }

    else:
        raise ValueError(
            f"unsupported factorization type: {factorization_type!r} "
            "(expected 'df' or 'thc')"
        )

    # -- ground-state energies in the fixed-electron sector -----------------
    eigvals_exact, eigvecs_exact = fixed_electron_sector_eigensystem(
        hamiltonian, n_qubits, n_electrons,
    )
    gs_energy_exact = float(eigvals_exact[0])

    eigvals_approx, eigvecs_approx = fixed_electron_sector_eigensystem(
        H_tilde, n_qubits, n_electrons,
    )
    gs_energy_approx = float(eigvals_approx[0])

    hamiltonian_energy_error = abs(gs_energy_approx - gs_energy_exact)
    total_energy_error = hamiltonian_energy_error + epsilon_qpe

    # -- time-evolution fidelity --------------------------------------------
    # |<e^{-i H_tilde t} psi0 | e^{-i H t} psi0>|^2
    exact_evolved = expm(-1j * hamiltonian * target_time) @ initial_state
    approx_evolved = expm(-1j * H_tilde * target_time) @ initial_state
    state_fidelity = fidelity(exact_evolved, approx_evolved)
    state_infidelity = max(0.0, 1.0 - state_fidelity)

    # -- score --------------------------------------------------------------
    passes_energy = total_energy_error <= chemical_accuracy
    fidelity_budget = _FIDELITY_BUDGET
    energy_penalty = (
        1e9 * max(0.0, total_energy_error - chemical_accuracy) / chemical_accuracy
    )
    fidelity_penalty = (
        1e9 * max(0.0, state_infidelity - fidelity_budget) / fidelity_budget
    )
    score = float(total_toffoli) + energy_penalty + fidelity_penalty

    # -- result dict --------------------------------------------------------
    return {
        "experiment": f"qubitization_qpe_{molecule}",
        "protocol": "qubitization_qpe",
        "molecule": molecule,
        "display_name": problem["display_name"],
        "script": str(script_path),
        "objective": "df_thc_qubitization_qpe_resource",
        "score": score,
        "passes_accuracy_constraint": passes_energy,
        "accuracy_metric_name": "total_energy_error_hartree",
        "accuracy_metric_value": total_energy_error,
        "accuracy_threshold_name": "chemical_accuracy_hartree",
        "accuracy_threshold_value": chemical_accuracy,
        "chemical_accuracy_hartree": chemical_accuracy,
        "ground_state_energy_exact_hartree": gs_energy_exact,
        "ground_state_energy_approx_hartree": gs_energy_approx,
        "ground_state_energy_reference_hartree": exact_gs_energy,
        "hamiltonian_energy_error_hartree": hamiltonian_energy_error,
        "epsilon_qpe_hartree": epsilon_qpe,
        "total_energy_error_hartree": total_energy_error,
        "fidelity": state_fidelity,
        "infidelity": state_infidelity,
        "fidelity_budget": fidelity_budget,
        "passes_fidelity_constraint": state_infidelity <= fidelity_budget,
        "target_time": target_time,
        "lambda_1norm": lam,
        "n_iterations_qpe": n_iterations,
        "total_toffoli": total_toffoli,
        "t_gate_count": total_t_gates,
        "logical_qubits": logical_qubits,
        "walk_cost_per_step": walk_cost_per_step,
        "factorization": factorization_detail,
        "method_config": asdict(config),
        "walk_operator_cost": {k: v for k, v in walk_data.items()},
        "qpe_cost": {k: v for k, v in qpe_data.items()},
        "n_orbitals": n_orb,
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "cas_label": problem["cas_label"],
        "basis": problem["basis"],
        "localization": problem["localization"],
        "evaluation_wall_seconds": time.perf_counter() - start,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a DF/THC qubitization-QPE resource policy."
    )
    parser.add_argument("--molecule", required=True, choices=molecule_choices())
    parser.add_argument(
        "--script", required=True,
        help="Path to the editable simple_qubitization.py method file.",
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = evaluate(args.molecule, Path(args.script).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
