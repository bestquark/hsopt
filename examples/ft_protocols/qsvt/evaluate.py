"""
Numerical QSVT dynamics evaluator for fault-tolerant chemistry protocols.

QSVT approximates e^{-iHt} by applying a polynomial transformation of the
block-encoded H/lambda.  The polynomial is the Jacobi-Anger expansion
truncated to degree d:

    e^{-i tau x} ~ J_0(tau) + 2 sum_{k=1}^{d} (-i)^k J_k(tau) T_k(x)

where tau = lambda * t, x = eigenvalue / lambda, and T_k is the k-th
Chebyshev polynomial of the first kind.

The evaluator:
  1. Loads a method config (factorization type, discretization widths,
     polynomial precision) from the user-editable script.
  2. Computes the DF or THC factorization and builds H_tilde (the effective
     Hamiltonian from discretized factorization).
  3. Computes lambda_tilde (1-norm of H_tilde's LCU decomposition).
  4. Determines the QSVT polynomial degree from the Jacobi-Anger bound.
  5. Simulates the QSVT output state numerically via eigen-decomposition
     of H_tilde and Jacobi-Anger evaluation on each eigenvalue.
  6. Computes fidelity against the exact evolution state.
  7. Computes energy error (Hamiltonian discretization + polynomial approx).
  8. Tallies Toffoli counts via walk-operator + qsvt_cost formulas.
  9. Returns score = toffoli_count + penalty.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.linalg import expm
from scipy.special import jv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protocol_common.cost_formulas import (  # noqa: E402
    df_logical_qubits,
    df_walk_operator_cost,
    qsvt_cost,
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
    score_from_accuracy_proxy,
)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_method_module(script_path: Path) -> ModuleType:
    """Import the user-editable method script and validate required symbols."""
    spec = importlib.util.spec_from_file_location("qsvt_method", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load method file: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "get_method_config"):
        raise AttributeError("method file is missing required symbol 'get_method_config'")
    return module


# ---------------------------------------------------------------------------
# Jacobi-Anger polynomial evaluation
# ---------------------------------------------------------------------------


def jacobi_anger_polynomial(x: np.ndarray, tau: float, degree: int) -> np.ndarray:
    """Evaluate the Jacobi-Anger polynomial approximation to e^{-i tau x}.

    Uses the Chebyshev expansion:
        e^{-i tau x} ~ J_0(tau) + 2 sum_{k=1}^{d} (-i)^k J_k(tau) T_k(x)

    Parameters
    ----------
    x : np.ndarray
        Points in [-1, 1] at which to evaluate the polynomial.
    tau : float
        Scaling parameter (lambda_tilde * target_time).
    degree : int
        Truncation degree d of the Jacobi-Anger expansion.

    Returns
    -------
    np.ndarray
        Complex array of shape x.shape with the polynomial values.
    """
    # Build Chebyshev coefficient vector: c[0] = J_0(tau), c[k] = 2*(-i)^k * J_k(tau)
    coeffs = np.zeros(degree + 1, dtype=np.complex128)
    coeffs[0] = jv(0, tau)
    for k in range(1, degree + 1):
        coeffs[k] = 2.0 * ((-1j) ** k) * jv(k, tau)

    # chebval evaluates sum_k c[k] * T_k(x)
    return chebval(x, coeffs)


def jacobi_anger_max_error(tau: float, degree: int) -> float:
    """Upper bound on the Jacobi-Anger truncation error.

    The tail is bounded by 2 * sum_{k=d+1}^{inf} |J_k(tau)|.
    We compute a generous finite partial sum as the bound.
    """
    tail_sum = 0.0
    for k in range(degree + 1, degree + 200):
        term = abs(float(jv(k, tau)))
        tail_sum += term
        if term < 1e-18:
            break
    return 2.0 * tail_sum


# ---------------------------------------------------------------------------
# Factorization dispatch
# ---------------------------------------------------------------------------


def _run_df_factorization(
    problem: Dict[str, object],
    config: object,
) -> Dict[str, object]:
    """Perform DF factorization, build H_tilde, compute lambda_tilde and costs."""
    one_body = np.asarray(problem["one_body"], dtype=np.float64)
    two_body = np.asarray(problem["two_body"], dtype=np.float64)
    n_qubits = int(problem["n_qubits"])
    constant = compute_e_core(
        np.asarray(problem["one_body"], dtype=np.float64),
        np.asarray(problem["two_body"], dtype=np.float64),
        float(problem["identity_coefficient"]),
        int(problem["n_qubits"]),
    )

    beth_per_leaf = list(config.beth_per_leaf)
    aleph = int(config.aleph)

    df_leaves = double_factorization(two_body)
    n_leaves = len(df_leaves)

    # Pad or trim beth_per_leaf to match the actual number of leaves
    if len(beth_per_leaf) < n_leaves:
        last = beth_per_leaf[-1] if beth_per_leaf else 10
        beth_per_leaf.extend([last] * (n_leaves - len(beth_per_leaf)))
    beth_per_leaf = beth_per_leaf[:n_leaves]

    h_tilde = build_effective_hamiltonian_df(
        one_body, two_body, constant, df_leaves,
        beth_per_leaf=beth_per_leaf,
        aleph=aleph,
        n_qubits=n_qubits,
    )

    lambda_tilde = df_lambda(one_body, df_leaves)

    xi_per_leaf = [len(leaf["eigenvalues"]) for leaf in df_leaves]

    walk_data = df_walk_operator_cost(
        n_orb=one_body.shape[0],
        n_leaves=n_leaves,
        xi_per_leaf=xi_per_leaf,
        beth_per_leaf=beth_per_leaf,
        aleph=aleph,
    )

    logical_qubit_count = df_logical_qubits(
        n_orb=one_body.shape[0],
        n_leaves=n_leaves,
        total_coefficients=walk_data["total_coefficients"],
        beth_max=walk_data["beth_max"],
        aleph=aleph,
    )

    return {
        "h_tilde": h_tilde,
        "lambda_tilde": lambda_tilde,
        "walk_cost_per_step": walk_data["total_per_step"],
        "logical_qubits": logical_qubit_count,
        "n_leaves": n_leaves,
        "xi_per_leaf": xi_per_leaf,
        "beth_per_leaf": beth_per_leaf,
        "walk_data": walk_data,
    }


def _run_thc_factorization(
    problem: Dict[str, object],
    config: object,
) -> Dict[str, object]:
    """Perform THC factorization, build H_tilde, compute lambda_tilde and costs."""
    one_body = np.asarray(problem["one_body"], dtype=np.float64)
    two_body = np.asarray(problem["two_body"], dtype=np.float64)
    n_qubits = int(problem["n_qubits"])
    constant = compute_e_core(
        np.asarray(problem["one_body"], dtype=np.float64),
        np.asarray(problem["two_body"], dtype=np.float64),
        float(problem["identity_coefficient"]),
        int(problem["n_qubits"]),
    )

    beth_per_mu = list(config.beth_per_mu)
    aleph = int(config.aleph)

    thc_data = thc_factorization(two_body)
    thc_rank = int(thc_data["rank"])

    # Pad or trim beth_per_mu to match the actual THC rank
    if len(beth_per_mu) < thc_rank:
        last = beth_per_mu[-1] if beth_per_mu else 10
        beth_per_mu.extend([last] * (thc_rank - len(beth_per_mu)))
    beth_per_mu = beth_per_mu[:thc_rank]

    h_tilde = build_effective_hamiltonian_thc(
        one_body, two_body, constant, thc_data,
        beth_per_mu=beth_per_mu,
        aleph=aleph,
        n_qubits=n_qubits,
    )

    lambda_tilde = thc_lambda(one_body, thc_data)

    walk_data = thc_walk_operator_cost(
        n_orb=one_body.shape[0],
        thc_rank=thc_rank,
        beth_per_mu=beth_per_mu,
        aleph=aleph,
    )

    # For THC logical qubits we need n_iterations (degree for QSVT).
    # We pass a placeholder; the caller will recompute if needed.
    # d and n_iterations will be set after we know the polynomial degree.
    logical_qubit_count = thc_logical_qubits(
        n_orb=one_body.shape[0],
        thc_rank=thc_rank,
        d=walk_data["d"],
        beth_max=walk_data["beth_max"],
        aleph=aleph,
        n_iterations=1,
    )

    return {
        "h_tilde": h_tilde,
        "lambda_tilde": lambda_tilde,
        "walk_cost_per_step": walk_data["total_per_step"],
        "logical_qubits": logical_qubit_count,
        "thc_rank": thc_rank,
        "beth_per_mu": beth_per_mu,
        "walk_data": walk_data,
    }


# ---------------------------------------------------------------------------
# QSVT numerical simulation
# ---------------------------------------------------------------------------


def simulate_qsvt_state(
    h_tilde: np.ndarray,
    lambda_tilde: float,
    target_time: float,
    degree: int,
    initial_state: np.ndarray,
) -> np.ndarray:
    """Simulate the QSVT output state numerically.

    Diagonalize H_tilde = V diag(eigenvalues) V^dag, then for each
    eigenvalue e_k compute the Jacobi-Anger polynomial
    p(e_k / lambda_tilde) which approximates e^{-i * e_k * t}.

    The QSVT state is V @ diag(p(e_k/lambda_tilde)) @ V^dag @ psi_0.
    """
    eigenvalues, V = np.linalg.eigh(h_tilde)

    tau = lambda_tilde * target_time
    x = eigenvalues / lambda_tilde

    # Clamp to [-1, 1] for numerical safety with Chebyshev evaluation
    x = np.clip(x, -1.0, 1.0)

    poly_values = jacobi_anger_polynomial(x, tau, degree)

    # QSVT state = V @ diag(poly_values) @ V^dag @ psi_0
    coeffs_in_eigenbasis = V.conj().T @ initial_state
    evolved_coeffs = poly_values * coeffs_in_eigenbasis
    qsvt_state = V @ evolved_coeffs

    # Normalize (the Jacobi-Anger polynomial is not exactly unitary at finite d)
    norm = np.linalg.norm(qsvt_state)
    if norm > 1e-15:
        qsvt_state = qsvt_state / norm

    return qsvt_state


def compute_energy_error(
    h_tilde: np.ndarray,
    hamiltonian: np.ndarray,
    qsvt_state: np.ndarray,
    lambda_tilde: float,
    target_time: float,
    degree: int,
    n_qubits: int,
    n_electrons: int,
) -> Dict[str, float]:
    """Compute the energy error from both Hamiltonian discretization and
    polynomial approximation.

    Returns a dict with the component errors and total.
    """
    # Ground-state energy of H_tilde in the fixed-electron sector
    eigvals_tilde, _ = fixed_electron_sector_eigensystem(h_tilde, n_qubits, n_electrons)
    e0_tilde = float(eigvals_tilde[0])

    # Ground-state energy of exact H in the fixed-electron sector
    eigvals_exact, _ = fixed_electron_sector_eigensystem(hamiltonian, n_qubits, n_electrons)
    e0_exact = float(eigvals_exact[0])

    # Hamiltonian discretization error
    discretization_error = abs(e0_tilde - e0_exact)

    # Polynomial approximation error bound from Jacobi-Anger truncation
    tau = lambda_tilde * target_time
    poly_error_bound = jacobi_anger_max_error(tau, degree)

    # Expectation value of H on the QSVT state
    qsvt_energy = float(np.real(np.vdot(qsvt_state, hamiltonian @ qsvt_state)))
    qsvt_energy_error = abs(qsvt_energy - e0_exact)

    # The total accuracy proxy combines discretization and polynomial errors.
    # We use the direct energy error as the primary metric, with the
    # component breakdown for diagnostics.
    accuracy_proxy = discretization_error + poly_error_bound

    return {
        "discretization_error_hartree": discretization_error,
        "polynomial_error_bound": poly_error_bound,
        "accuracy_proxy_hartree": accuracy_proxy,
        "ground_state_energy_exact_hartree": e0_exact,
        "ground_state_energy_tilde_hartree": e0_tilde,
        "qsvt_expectation_energy_hartree": qsvt_energy,
        "qsvt_energy_error_hartree": qsvt_energy_error,
    }


# ---------------------------------------------------------------------------
# Main evaluate
# ---------------------------------------------------------------------------


def evaluate(molecule: str, script_path: Path) -> Dict[str, object]:
    start = time.perf_counter()

    module = load_method_module(script_path)
    config = module.get_method_config()
    problem = load_protocol_problem(molecule)

    factorization_type = str(config.factorization).lower()
    target_time = float(problem["target_time"])
    n_qubits = int(problem["n_qubits"])
    hamiltonian = np.asarray(problem["hamiltonian"], dtype=np.complex128)
    initial_state = np.asarray(problem["initial_state"], dtype=np.complex128)
    polynomial_precision = float(config.polynomial_precision)

    # -- Step 1: Factorization and H_tilde construction --
    if factorization_type == "df":
        fact_result = _run_df_factorization(problem, config)
    elif factorization_type == "thc":
        fact_result = _run_thc_factorization(problem, config)
    else:
        raise ValueError(f"unsupported factorization type: {factorization_type!r}")

    h_tilde = fact_result["h_tilde"]
    lambda_tilde = fact_result["lambda_tilde"]
    walk_cost_per_step = fact_result["walk_cost_per_step"]
    logical_qubit_count = fact_result["logical_qubits"]

    # The block encoding operates on H_centered = H_tilde - Tr(H_tilde)/dim * I.
    # Sector eigenvalues of H_centered are bounded by lambda_tilde.
    dim = 2 ** n_qubits
    h_tilde_id_coeff = float(np.real(np.trace(h_tilde))) / dim
    h_tilde_centered = h_tilde - h_tilde_id_coeff * np.eye(dim, dtype=h_tilde.dtype)

    # -- Step 2: QSVT polynomial degree from Jacobi-Anger bound --
    qsvt_data = qsvt_cost(
        walk_cost_per_step=walk_cost_per_step,
        lam=lambda_tilde,
        target_time=target_time,
        epsilon_poly=polynomial_precision,
    )
    degree = int(qsvt_data["degree"])
    total_toffoli = int(qsvt_data["total_toffoli"])
    total_t_gates = int(qsvt_data["total_t_gates"])

    # -- Step 3: Numerical QSVT simulation --
    qsvt_state = simulate_qsvt_state(
        h_tilde=h_tilde_centered,
        lambda_tilde=lambda_tilde,
        target_time=target_time,
        degree=degree,
        initial_state=initial_state,
    )

    # -- Step 4: Fidelity --
    exact_evolved = expm(-1j * hamiltonian * target_time) @ initial_state
    fidelity_value = fidelity(exact_evolved, qsvt_state)

    # -- Step 5: Energy error --
    n_electrons = int(problem["metadata"]["active_electrons"])
    energy_data = compute_energy_error(
        h_tilde=h_tilde,
        hamiltonian=hamiltonian,
        qsvt_state=qsvt_state,
        lambda_tilde=lambda_tilde,
        target_time=target_time,
        degree=degree,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
    )

    accuracy_proxy = energy_data["accuracy_proxy_hartree"]

    # -- Step 6: Score --
    allowed = float(problem["chemical_accuracy_hartree"])
    passes = accuracy_proxy <= allowed
    score = score_from_accuracy_proxy(total_t_gates, allowed, accuracy_proxy)

    # -- Build factorization-specific metadata --
    fact_metadata: Dict[str, object] = {"factorization": factorization_type}
    if factorization_type == "df":
        fact_metadata.update({
            "n_leaves": fact_result["n_leaves"],
            "xi_per_leaf": fact_result["xi_per_leaf"],
            "beth_per_leaf": fact_result["beth_per_leaf"],
        })
    elif factorization_type == "thc":
        fact_metadata.update({
            "thc_rank": fact_result["thc_rank"],
            "beth_per_mu": fact_result["beth_per_mu"],
        })

    return {
        "experiment": f"qsvt_dynamics_{molecule}",
        "protocol": "qsvt_dynamics",
        "molecule": molecule,
        "display_name": problem["display_name"],
        "script": str(script_path),
        "objective": "qsvt_dynamics_resource_estimate",
        "score": score,
        "passes_accuracy_constraint": passes,
        "accuracy_metric_name": "accuracy_proxy_hartree",
        "accuracy_metric_value": accuracy_proxy,
        "accuracy_threshold_name": "chemical_accuracy_hartree",
        "accuracy_threshold_value": allowed,
        "t_gate_count": total_t_gates,
        "toffoli_count": total_toffoli,
        "logical_qubits": logical_qubit_count,
        "qualtran_count_status": "numerical_simulation",
        "fidelity": fidelity_value,
        "infidelity": max(0.0, 1.0 - fidelity_value),
        "qsvt_degree": degree,
        "lambda_tilde": lambda_tilde,
        "tau": float(qsvt_data["tau"]),
        "target_time": target_time,
        "polynomial_precision": polynomial_precision,
        "walk_cost_per_step": walk_cost_per_step,
        "energy_metrics": energy_data,
        "factorization_metadata": fact_metadata,
        "method_config": {
            "name": getattr(config, "name", "unknown"),
            "factorization": factorization_type,
            "aleph": int(config.aleph),
            "polynomial_precision": polynomial_precision,
        },
        "resource_estimate": {
            "qsvt": dict(qsvt_data),
            "walk_operator": dict(fact_result["walk_data"]),
        },
        "cas_label": problem["cas_label"],
        "basis": problem["basis"],
        "localization": problem["localization"],
        "evaluation_wall_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one QSVT dynamics resource policy with numerical simulation."
    )
    parser.add_argument("--molecule", required=True, choices=molecule_choices())
    parser.add_argument("--script", required=True, help="Path to the editable simple_qsvt.py file.")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = evaluate(args.molecule, Path(args.script).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
