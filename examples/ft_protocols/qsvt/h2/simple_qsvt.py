"""
Editable Qualtran QSVT/QSP Hamiltonian-simulation resource policy.

This method estimates direct real-time dynamics, i.e. a block-encoding of
exp(-i H t), using a sparse chemistry qubitization walk and GQSP/QSVT. It is a
different simulation route from product formulas: the repeated object is the
qubitized walk, not per-term Pauli rotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from protocol_common.qualtran_resources import (
    build_sparse_chem_walk,
    coefficient_loading_error,
    cost_bloq,
    safe_cost_bloq,
)


@dataclass(frozen=True)
class QSVTConfig:
    name: str = "sparse_qsvt_dynamics_baseline"
    num_bits_state_prep: int = 14
    num_bits_rot_aa: int = 15
    simulation_error_fraction: float = 0.45
    ts_per_rotation: int = 11


def get_method_config() -> QSVTConfig:
    return QSVTConfig()


def _fallback_gqsp_counts(walk, degree: int, *, ts_per_rotation: int) -> Dict[str, object]:
    walk_counts, walk_qubits = cost_bloq(walk, ts_per_rotation=ts_per_rotation)
    prepare_counts, prepare_qubits = cost_bloq(walk.prepare, ts_per_rotation=ts_per_rotation)
    select_counts, select_qubits = cost_bloq(walk.block_encoding.select, ts_per_rotation=ts_per_rotation)

    # HamiltonianSimulationByGQSP uses PREPARE, PREPARE†, 2*d controlled walk
    # applications, and 2*d+1 arbitrary SU(2) rotations. Qualtran 0.7 can fail
    # to fully count the controlled adjoint walk; this fallback uses the
    # uncontrolled walk as a stable transparent proxy.
    total_t = (
        2 * int(prepare_counts["total_t_count"])
        + 2 * int(degree) * int(walk_counts["total_t_count"])
        + (2 * int(degree) + 1) * int(ts_per_rotation)
    )
    return {
        "gate_counts": {
            "total_t_count": int(total_t),
            "gqsp_degree": int(degree),
            "walk_total_t_count": int(walk_counts["total_t_count"]),
            "prepare_total_t_count": int(prepare_counts["total_t_count"]),
            "select_total_t_count": int(select_counts["total_t_count"]),
            "su2_rotation_count": 2 * int(degree) + 1,
            "su2_rotation_synthesis_t_count": (2 * int(degree) + 1) * int(ts_per_rotation),
        },
        "logical_qubits": max(int(walk_qubits) + 1, int(prepare_qubits), int(select_qubits)),
        "resource_breakdown": {
            "walk": walk_counts,
            "prepare": prepare_counts,
            "select": select_counts,
        },
    }


def estimate_qsvt_dynamics(problem: Dict[str, object]) -> Dict[str, object]:
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )

    config = get_method_config()
    walk = build_sparse_chem_walk(
        problem["one_body"],
        problem["two_body"],
        num_bits_state_prep=config.num_bits_state_prep,
        num_bits_rot_aa=config.num_bits_rot_aa,
    )
    alpha = float(walk.block_encoding.alpha)
    target_time = float(problem["target_time"])
    operator_precision = (
        float(problem["chemical_accuracy_hartree"])
        * target_time
        * float(config.simulation_error_fraction)
    )
    sim = HamiltonianSimulationByGQSP(walk, t=target_time, precision=operator_precision)
    direct_cost = safe_cost_bloq(sim, ts_per_rotation=config.ts_per_rotation)
    degree = int(sim.degree)

    if direct_cost["status"] == "exact":
        gate_counts = direct_cost["gate_counts"]
        logical_qubits = int(direct_cost["logical_qubits"])
        resource_breakdown = {"gqsp": gate_counts}
    else:
        fallback = _fallback_gqsp_counts(walk, degree, ts_per_rotation=config.ts_per_rotation)
        gate_counts = fallback["gate_counts"]
        logical_qubits = int(fallback["logical_qubits"])
        resource_breakdown = fallback["resource_breakdown"]

    loading_errors = coefficient_loading_error(
        alpha,
        config.num_bits_state_prep,
        config.num_bits_rot_aa,
    )
    simulation_energy_error = operator_precision / max(target_time, 1e-15)
    accuracy_proxy = simulation_energy_error + sum(loading_errors.values())

    return {
        "method_config": asdict(config),
        "alpha": alpha,
        "target_time": target_time,
        "gqsp_degree": degree,
        "operator_precision": operator_precision,
        "simulation_energy_error_hartree": simulation_energy_error,
        "accuracy_proxy_hartree": accuracy_proxy,
        "accuracy_proxy_terms": {
            "simulation_energy_error_hartree": simulation_energy_error,
            **loading_errors,
        },
        "qualtran_count_status": str(direct_cost["status"]),
        "qualtran_count_error": direct_cost["error"],
        "gate_counts": gate_counts,
        "resource_breakdown": resource_breakdown,
        "t_gate_count": int(gate_counts["total_t_count"]),
        "logical_qubits": logical_qubits,
        "simulation_bloq": str(sim),
        "walk_bloq": str(walk),
    }

