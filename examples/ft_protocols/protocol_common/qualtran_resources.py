from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def require_qualtran() -> None:
    try:
        import qualtran  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Qualtran is required for this benchmark family. "
            "Install it with `uv sync --extra ft-protocols`."
        ) from exc


def build_sparse_chem_walk(
    one_body: np.ndarray,
    two_body: np.ndarray,
    *,
    num_bits_state_prep: int,
    num_bits_rot_aa: int,
):
    require_qualtran()
    from qualtran.bloqs.chemistry.sparse.walk_operator import get_walk_operator_for_sparse_chem_ham

    return get_walk_operator_for_sparse_chem_ham(
        np.asarray(one_body, dtype=np.float64),
        np.asarray(two_body, dtype=np.float64),
        num_bits_state_prep=int(num_bits_state_prep),
        num_bits_rot_aa=int(num_bits_rot_aa),
    )


def sparse_lcu_lambda(one_body: np.ndarray, two_body: np.ndarray) -> float:
    return float(np.sum(np.abs(one_body)) + 0.5 * np.sum(np.abs(two_body)))


def gate_counts_to_dict(counts, *, ts_per_rotation: int) -> Dict[str, int]:
    total_t = int(counts.total_t_count(ts_per_rotation=int(ts_per_rotation)))
    total_no_rot = int(counts.total_t_count(ts_per_rotation=0))
    data = {key: int(value) for key, value in counts.asdict().items()}
    data.update(
        {
            "total_t_count": total_t,
            "total_t_count_without_rotations": total_no_rot,
            "rotation_synthesis_t_count": total_t - total_no_rot,
        }
    )
    return data


def cost_bloq(bloq, *, ts_per_rotation: int) -> Tuple[Dict[str, int], int]:
    require_qualtran()
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    counts = get_cost_value(bloq, QECGatesCost())
    logical_qubits = int(get_cost_value(bloq, QubitCount()))
    return gate_counts_to_dict(counts, ts_per_rotation=ts_per_rotation), logical_qubits


def safe_cost_bloq(bloq, *, ts_per_rotation: int) -> Dict[str, object]:
    try:
        gate_counts, logical_qubits = cost_bloq(bloq, ts_per_rotation=ts_per_rotation)
        return {
            "status": "exact",
            "gate_counts": gate_counts,
            "logical_qubits": logical_qubits,
            "error": None,
        }
    except Exception as exc:  # Qualtran is experimental; preserve useful partial estimates.
        return {
            "status": "fallback",
            "gate_counts": None,
            "logical_qubits": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def coefficient_loading_error(alpha: float, num_bits_state_prep: int, num_bits_rot_aa: int) -> Dict[str, float]:
    state_prep_error = float(alpha) * 2.0 ** (-int(num_bits_state_prep))
    rotation_arithmetic_error = float(alpha) * 2.0 ** (-int(num_bits_rot_aa))
    return {
        "state_preparation_error_hartree": state_prep_error,
        "rotation_arithmetic_error_hartree": rotation_arithmetic_error,
    }


def qpe_phase_epsilon(alpha: float, qpe_energy_error_hartree: float) -> float:
    return float(qpe_energy_error_hartree) / (float(alpha) * math.sqrt(2.0))


def gqsp_degree(alpha: float, target_time: float, precision: float) -> int:
    require_qualtran()
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        degree_jacobi_anger_approximation,
    )

    return int(degree_jacobi_anger_approximation(float(alpha) * float(target_time), precision=float(precision)))
