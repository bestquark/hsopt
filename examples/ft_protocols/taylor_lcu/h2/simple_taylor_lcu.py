"""
Editable Taylor/LCU Hamiltonian-simulation resource policy.

This method estimates direct real-time dynamics through a segmented truncated
Taylor expansion. Qualtran supplies the chemistry SELECT/PREPARE resource
counts; the segment count, Taylor order, and amplification factor are explicit
policy knobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict

from protocol_common.qualtran_resources import (
    build_sparse_chem_walk,
    coefficient_loading_error,
    cost_bloq,
)


@dataclass(frozen=True)
class TaylorLCUConfig:
    name: str = "sparse_taylor_lcu_baseline"
    num_bits_state_prep: int = 14
    num_bits_rot_aa: int = 15
    taylor_order: int = 9
    segment_count: int = 5
    amplification_factor: int = 3
    ts_per_rotation: int = 11


def get_method_config() -> TaylorLCUConfig:
    return TaylorLCUConfig()


def _tail_bound(alpha_t_over_segments: float, taylor_order: int, segment_count: int) -> float:
    x = float(alpha_t_over_segments)
    order = int(taylor_order)
    if x <= 0.0:
        return 0.0
    segment_tail = math.exp(x) * (x ** (order + 1)) / math.factorial(order + 1)
    return float(segment_count) * segment_tail


def estimate_taylor_lcu(problem: Dict[str, object]) -> Dict[str, object]:
    config = get_method_config()
    walk = build_sparse_chem_walk(
        problem["one_body"],
        problem["two_body"],
        num_bits_state_prep=config.num_bits_state_prep,
        num_bits_rot_aa=config.num_bits_rot_aa,
    )
    alpha = float(walk.block_encoding.alpha)
    target_time = float(problem["target_time"])
    segment_count = int(config.segment_count)
    taylor_order = int(config.taylor_order)
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    if taylor_order <= 0:
        raise ValueError("taylor_order must be positive")

    prepare_counts, prepare_qubits = cost_bloq(walk.prepare, ts_per_rotation=config.ts_per_rotation)
    select_counts, select_qubits = cost_bloq(walk.block_encoding.select, ts_per_rotation=config.ts_per_rotation)
    x = alpha * target_time / float(segment_count)
    truncation_operator_error = _tail_bound(x, taylor_order, segment_count)

    # A segmented Taylor-LCU step prepares Taylor/term registers, applies up to
    # K SELECTs, and unprepares. The amplification factor accounts for the
    # oblivious-amplitude-amplification/reflection wrapper at this proxy level.
    prepare_t = int(prepare_counts["total_t_count"])
    select_t = int(select_counts["total_t_count"])
    per_segment_t = int(config.amplification_factor) * (
        2 * prepare_t + taylor_order * (prepare_t + select_t)
    )
    total_t = segment_count * per_segment_t

    loading_errors = coefficient_loading_error(
        alpha,
        config.num_bits_state_prep,
        config.num_bits_rot_aa,
    )
    truncation_energy_error = truncation_operator_error / max(target_time, 1e-15)
    accuracy_proxy = truncation_energy_error + sum(loading_errors.values())

    return {
        "method_config": asdict(config),
        "alpha": alpha,
        "target_time": target_time,
        "segment_count": segment_count,
        "taylor_order": taylor_order,
        "alpha_t_over_segments": x,
        "truncation_operator_error": truncation_operator_error,
        "truncation_energy_error_hartree": truncation_energy_error,
        "accuracy_proxy_hartree": accuracy_proxy,
        "accuracy_proxy_terms": {
            "truncation_energy_error_hartree": truncation_energy_error,
            **loading_errors,
        },
        "qualtran_count_status": "proxy",
        "gate_counts": {
            "total_t_count": int(total_t),
            "per_segment_t_count": int(per_segment_t),
            "prepare_total_t_count": prepare_t,
            "select_total_t_count": select_t,
            "prepare_call_count_proxy": segment_count * int(config.amplification_factor) * (taylor_order + 2),
            "select_call_count_proxy": segment_count * int(config.amplification_factor) * taylor_order,
        },
        "resource_breakdown": {
            "prepare": prepare_counts,
            "select": select_counts,
        },
        "t_gate_count": int(total_t),
        "logical_qubits": max(int(prepare_qubits), int(select_qubits)) + max(1, taylor_order.bit_length()),
        "walk_bloq": str(walk),
    }

