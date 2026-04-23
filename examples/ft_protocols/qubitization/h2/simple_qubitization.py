"""
Editable Qualtran qubitization-QPE resource policy.

The evaluator supplies fixed chemistry integrals and a fixed chemical-accuracy
threshold. Mutations should only change resource/precision policy parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from protocol_common.qualtran_resources import (
    build_sparse_chem_walk,
    coefficient_loading_error,
    cost_bloq,
    qpe_phase_epsilon,
)


@dataclass(frozen=True)
class QubitizationQPEConfig:
    name: str = "sparse_qubitization_qpe_baseline"
    num_bits_state_prep: int = 14
    num_bits_rot_aa: int = 15
    qpe_error_fraction: float = 0.45
    ts_per_rotation: int = 11


def get_method_config() -> QubitizationQPEConfig:
    return QubitizationQPEConfig()


def estimate_qubitization_qpe(problem: Dict[str, object]) -> Dict[str, object]:
    from qualtran.bloqs.phase_estimation import LPResourceState, QubitizationQPE

    config = get_method_config()
    walk = build_sparse_chem_walk(
        problem["one_body"],
        problem["two_body"],
        num_bits_state_prep=config.num_bits_state_prep,
        num_bits_rot_aa=config.num_bits_rot_aa,
    )
    alpha = float(walk.block_encoding.alpha)
    qpe_energy_error = float(problem["chemical_accuracy_hartree"]) * float(config.qpe_error_fraction)
    qpe_eps = qpe_phase_epsilon(alpha, qpe_energy_error)
    qpe = QubitizationQPE(walk, LPResourceState.from_standard_deviation_eps(qpe_eps))

    gate_counts, logical_qubits = cost_bloq(qpe, ts_per_rotation=config.ts_per_rotation)
    loading_errors = coefficient_loading_error(
        alpha,
        config.num_bits_state_prep,
        config.num_bits_rot_aa,
    )
    accuracy_proxy = qpe_energy_error + sum(loading_errors.values())

    return {
        "method_config": asdict(config),
        "alpha": alpha,
        "qpe_phase_epsilon": qpe_eps,
        "qpe_energy_error_hartree": qpe_energy_error,
        "accuracy_proxy_hartree": accuracy_proxy,
        "accuracy_proxy_terms": {
            "qpe_energy_error_hartree": qpe_energy_error,
            **loading_errors,
        },
        "qualtran_count_status": "exact",
        "gate_counts": gate_counts,
        "t_gate_count": int(gate_counts["total_t_count"]),
        "logical_qubits": logical_qubits,
        "qpe_bloq": str(qpe),
        "walk_bloq": str(walk),
    }

