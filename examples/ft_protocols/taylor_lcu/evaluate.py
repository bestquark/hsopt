"""Evaluate a Taylor/LCU Hamiltonian-simulation resource policy.

The Taylor LCU protocol approximates e^{-iHt} using a truncated Taylor series:

    e^{-iH tau} ~ sum_{k=0}^{K} (-iH tau)^k / k!

applied in S segments of duration tau = t/S.

This evaluator:
  1. Loads a method config (frozen dataclass with factorization parameters,
     Taylor order, error budget).
  2. Computes the DF or THC factorization and builds H_tilde (effective
     Hamiltonian from the discretized factorization).
  3. Simulates the Taylor LCU output state via eigendecomposition.
  4. Computes fidelity, energy accuracy, Toffoli cost, and a composite score.
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
from scipy.linalg import expm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protocol_common.problems import (  # noqa: E402
    load_protocol_problem,
    molecule_choices,
    score_from_accuracy_proxy,
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
from protocol_common.cost_formulas import (  # noqa: E402
    df_walk_operator_cost,
    thc_walk_operator_cost,
    taylor_lcu_cost,
    df_logical_qubits,
    thc_logical_qubits,
)


# ---------------------------------------------------------------------------
# Method module loader
# ---------------------------------------------------------------------------

def load_method_module(script_path: Path) -> ModuleType:
    """Import a user-provided method file that exports ``get_method_config``."""
    spec = importlib.util.spec_from_file_location("taylor_lcu_method", script_path)
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
# Taylor propagation via eigendecomposition
# ---------------------------------------------------------------------------

def _compute_segments_needed(
    lam_tilde: float,
    target_time: float,
    taylor_order: int,
    epsilon_taylor: float,
) -> int:
    """Find the minimum number of segments S such that the total Taylor
    truncation error across all segments is below *epsilon_taylor*.

    Per-segment remainder: |e^x - T_K(x)| <= e^|x| * |x|^{K+1} / (K+1)!
    where |x| = lam_tilde * t / S.  Total error = S * per-segment remainder.
    """
    K = taylor_order
    tau_total = lam_tilde * target_time
    S = 1
    while S <= 100_000:
        x = tau_total / S
        per_segment = math.exp(x) * (x ** (K + 1)) / math.factorial(K + 1)
        if S * per_segment <= epsilon_taylor:
            return S
        S += 1
    return S


def _taylor_propagate(
    H_tilde: np.ndarray,
    initial_state: np.ndarray,
    target_time: float,
    taylor_order: int,
    segments: int,
) -> np.ndarray:
    """Apply S segments of the truncated Taylor series to *initial_state*
    via eigendecomposition of H_tilde.

    For each eigenvalue e_j the per-segment factor is:
        f_j = sum_{k=0}^{K} (-i * e_j * tau)^k / k!
    where tau = target_time / segments.
    The composed factor over S segments is f_j^S.
    """
    eigvals, eigvecs = np.linalg.eigh(H_tilde)
    tau = target_time / segments
    K = taylor_order

    taylor_per_segment = np.zeros(len(eigvals), dtype=np.complex128)
    for k in range(K + 1):
        taylor_per_segment += (-1j * eigvals * tau) ** k / math.factorial(k)

    overall = taylor_per_segment ** segments

    coeffs = eigvecs.conj().T @ initial_state
    propagated_coeffs = overall * coeffs
    taylor_state = eigvecs @ propagated_coeffs

    norm = np.linalg.norm(taylor_state)
    if norm > 1e-30:
        taylor_state = taylor_state / norm
    return taylor_state


def _taylor_truncation_error_eigval(
    eigvals: np.ndarray,
    target_time: float,
    taylor_order: int,
    segments: int,
) -> float:
    """Compute the maximum spectral-norm Taylor truncation error.

    For each eigenvalue, compare the exact phase e^{-i e tau} with the
    composed truncated Taylor (sum_{k=0}^K (-i e tau)^k / k!)^S.
    """
    tau = target_time / segments
    K = taylor_order

    exact_per_segment = np.exp(-1j * eigvals * tau)
    taylor_per_segment = np.zeros(len(eigvals), dtype=np.complex128)
    for k in range(K + 1):
        taylor_per_segment += (-1j * eigvals * tau) ** k / math.factorial(k)

    exact_overall = exact_per_segment ** segments
    taylor_overall = taylor_per_segment ** segments

    return float(np.max(np.abs(exact_overall - taylor_overall)))


# ---------------------------------------------------------------------------
# Factorization dispatch
# ---------------------------------------------------------------------------

def _run_df_pipeline(
    problem: Dict[str, object],
    config,
) -> Dict[str, object]:
    """Run the DF factorization, build H_tilde, compute lambda, walk cost."""
    one_body = np.asarray(problem["one_body"], dtype=np.float64)
    two_body = np.asarray(problem["two_body"], dtype=np.float64)
    n_qubits = int(problem["n_qubits"])
    constant = compute_e_core(one_body, two_body, float(problem["identity_coefficient"]), n_qubits)

    df_leaves = double_factorization(two_body)
    beth = list(config.beth_per_leaf)
    if len(beth) < len(df_leaves):
        beth = beth + [beth[-1]] * (len(df_leaves) - len(beth))

    H_tilde = build_effective_hamiltonian_df(
        one_body, two_body, constant, df_leaves,
        beth_per_leaf=beth,
        aleph=config.aleph,
        n_qubits=n_qubits,
    )

    lam_tilde = df_lambda(one_body, df_leaves)
    xi_per_leaf = [len(leaf["eigenvalues"]) for leaf in df_leaves]

    walk_data = df_walk_operator_cost(
        n_orb=n_qubits // 2,
        n_leaves=len(df_leaves),
        xi_per_leaf=xi_per_leaf,
        beth_per_leaf=beth,
        aleph=config.aleph,
    )

    n_orb = n_qubits // 2
    logical_q = df_logical_qubits(
        n_orb=n_orb,
        n_leaves=len(df_leaves),
        total_coefficients=walk_data["total_coefficients"],
        beth_max=walk_data["beth_max"],
        aleph=config.aleph,
    )

    return {
        "H_tilde": H_tilde,
        "lam_tilde": lam_tilde,
        "walk_data": walk_data,
        "logical_qubits": logical_q,
        "n_leaves": len(df_leaves),
        "xi_per_leaf": xi_per_leaf,
    }


def _run_thc_pipeline(
    problem: Dict[str, object],
    config,
) -> Dict[str, object]:
    """Run the THC factorization, build H_tilde, compute lambda, walk cost."""
    one_body = np.asarray(problem["one_body"], dtype=np.float64)
    two_body = np.asarray(problem["two_body"], dtype=np.float64)
    n_qubits = int(problem["n_qubits"])
    constant = compute_e_core(one_body, two_body, float(problem["identity_coefficient"]), n_qubits)

    thc_data = thc_factorization(two_body)
    thc_rank = int(thc_data["rank"])

    beth = list(config.beth_per_mu)
    if len(beth) < thc_rank:
        beth = beth + [beth[-1]] * (thc_rank - len(beth))

    H_tilde = build_effective_hamiltonian_thc(
        one_body, two_body, constant, thc_data,
        beth_per_mu=beth,
        aleph=config.aleph,
        n_qubits=n_qubits,
    )

    lam_tilde = thc_lambda(one_body, thc_data)

    walk_data = thc_walk_operator_cost(
        n_orb=n_qubits // 2,
        thc_rank=thc_rank,
        beth_per_mu=beth,
        aleph=config.aleph,
    )

    logical_q = thc_logical_qubits(
        n_orb=n_qubits // 2,
        thc_rank=thc_rank,
        d=walk_data["d"],
        beth_max=walk_data["beth_max"],
        aleph=config.aleph,
        n_iterations=1,
    )

    return {
        "H_tilde": H_tilde,
        "lam_tilde": lam_tilde,
        "walk_data": walk_data,
        "logical_qubits": logical_q,
        "thc_rank": thc_rank,
    }


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(molecule: str, script_path: Path) -> Dict[str, object]:
    start = time.perf_counter()

    module = load_method_module(script_path)
    config = module.get_method_config()
    problem = load_protocol_problem(molecule)

    n_qubits = int(problem["n_qubits"])
    n_electrons = int(problem["metadata"]["active_electrons"])
    target_time = float(problem["target_time"])
    initial_state = np.asarray(problem["initial_state"], dtype=np.complex128)
    hamiltonian = np.asarray(problem["hamiltonian"], dtype=np.complex128)
    # --- Step 1: factorization and effective Hamiltonian ---
    factorization = str(config.factorization).lower()
    if factorization == "df":
        pipeline = _run_df_pipeline(problem, config)
    elif factorization == "thc":
        pipeline = _run_thc_pipeline(problem, config)
    else:
        raise ValueError(
            f"unsupported factorization {config.factorization!r}; "
            f"use 'df' or 'thc'"
        )

    H_tilde = pipeline["H_tilde"]
    lam_tilde = pipeline["lam_tilde"]
    walk_data = pipeline["walk_data"]
    logical_qubits = int(pipeline["logical_qubits"])

    # --- Step 2: ground-state energy of H_tilde vs H ---
    exact_eigvals, _ = fixed_electron_sector_eigensystem(
        hamiltonian, n_qubits, n_electrons,
    )
    ground_truth_energy = float(exact_eigvals[0])

    tilde_eigvals, tilde_eigvecs = fixed_electron_sector_eigensystem(
        H_tilde, n_qubits, n_electrons,
    )
    e0_tilde = float(tilde_eigvals[0])
    factorization_energy_error = abs(e0_tilde - ground_truth_energy)

    # --- Step 3: Taylor propagation ---
    # The block encoding operates on H_centered = H_tilde - Tr(H_tilde)/dim * I.
    # Sector eigenvalues of H_centered are bounded by lambda_tilde.
    # The identity coefficient contributes a global phase irrelevant for fidelity.
    K = int(config.taylor_order)
    eps_taylor = float(config.epsilon_taylor)
    dim = 2 ** n_qubits
    h_tilde_id_coeff = float(np.real(np.trace(H_tilde))) / dim
    H_tilde_centered = H_tilde - h_tilde_id_coeff * np.eye(dim, dtype=H_tilde.dtype)

    segments = _compute_segments_needed(lam_tilde, target_time, K, eps_taylor)

    taylor_state = _taylor_propagate(
        H_tilde_centered, initial_state, target_time, K, segments,
    )

    # --- Step 4: fidelity ---
    exact_evolved = expm(-1j * hamiltonian * target_time) @ initial_state
    taylor_fidelity = fidelity(taylor_state, exact_evolved)

    # --- Step 5: Taylor truncation error (spectral) ---
    tilde_sector_eigvals_centered = tilde_eigvals - h_tilde_id_coeff
    spectral_taylor_error = _taylor_truncation_error_eigval(
        tilde_sector_eigvals_centered, target_time, K, segments,
    )

    # --- Step 6: accuracy proxy ---
    accuracy_proxy = factorization_energy_error + spectral_taylor_error

    # --- Step 7: resource cost ---
    prepare_cost = walk_data.get("prepare_total", walk_data.get("prepare_cost", 0))
    select_cost = walk_data.get("select_total", walk_data.get("select_cost", 0))

    cost_data = taylor_lcu_cost(
        prepare_cost=int(prepare_cost),
        select_cost=int(select_cost),
        lam=lam_tilde,
        target_time=target_time,
        taylor_order=K,
        epsilon_taylor=eps_taylor,
    )

    toffoli_count = int(cost_data["total_toffoli"])
    t_gate_count = int(cost_data["total_t_gates"])

    # --- Step 8: score ---
    allowed = float(problem["chemical_accuracy_hartree"])
    passes = accuracy_proxy <= allowed
    score = score_from_accuracy_proxy(t_gate_count, allowed, accuracy_proxy)

    wall_seconds = time.perf_counter() - start

    return {
        "experiment": f"taylor_lcu_{molecule}",
        "protocol": "taylor_lcu",
        "molecule": molecule,
        "display_name": problem["display_name"],
        "script": str(script_path),
        "objective": "taylor_lcu_resource_estimate",
        "score": score,
        "passes_accuracy_constraint": passes,
        "accuracy_metric_name": "accuracy_proxy_hartree",
        "accuracy_metric_value": accuracy_proxy,
        "accuracy_threshold_name": "chemical_accuracy_hartree",
        "accuracy_threshold_value": allowed,
        "t_gate_count": t_gate_count,
        "toffoli_count": toffoli_count,
        "logical_qubits": logical_qubits,
        "qualtran_count_status": "analytic",
        "resource_estimate": {
            "method_config": {
                "name": getattr(config, "name", "taylor_lcu"),
                "factorization": factorization,
                "taylor_order": K,
                "epsilon_taylor": eps_taylor,
                "aleph": int(config.aleph),
            },
            "factorization": factorization,
            "lambda_tilde": lam_tilde,
            "segments": int(cost_data["segments"]),
            "taylor_order": K,
            "target_time": target_time,
            "factorization_energy_error_hartree": factorization_energy_error,
            "e0_tilde_hartree": e0_tilde,
            "e0_exact_hartree": ground_truth_energy,
            "spectral_taylor_error": spectral_taylor_error,
            "accuracy_proxy_hartree": accuracy_proxy,
            "taylor_fidelity": taylor_fidelity,
            "toffoli_count": toffoli_count,
            "t_gate_count": t_gate_count,
            "cost_breakdown": {
                "prepare_cost": int(prepare_cost),
                "select_cost": int(select_cost),
                "walk_total_per_step": int(walk_data["total_per_step"]),
                "cost_per_segment": int(cost_data["cost_per_segment"]),
                "segments": int(cost_data["segments"]),
                "actual_taylor_error_bound": float(cost_data["actual_taylor_error"]),
            },
            "logical_qubits": logical_qubits,
        },
        "cas_label": problem["cas_label"],
        "basis": problem["basis"],
        "localization": problem["localization"],
        "evaluation_wall_seconds": wall_seconds,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one Taylor/LCU dynamics resource policy."
    )
    parser.add_argument(
        "--molecule", required=True, choices=molecule_choices(),
    )
    parser.add_argument(
        "--script", required=True,
        help="Path to the editable method file (e.g. simple_taylor_lcu.py).",
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = evaluate(args.molecule, Path(args.script).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
