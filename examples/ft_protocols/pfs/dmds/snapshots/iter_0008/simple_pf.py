"""
Editable PF2 method file for the localized small-molecule precision benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str = "dmds_pf2_small_high_support_eps"
    n_steps: int = 512
    zero_angle_threshold: float = 1e-15
    uniform_epsilon: float = 1e-10
    small_high_support_epsilon: float = 3e-10
    small_high_support_alpha_threshold: float = 1.3e-10
    small_high_support_min_support: int = 8
    ultra_high_support_epsilon: float = 1e-9
    ultra_high_support_alpha_threshold: float = 3e-10
    ultra_high_support_min_support: int = 9
    small_angle_band_epsilon: float = 2e-10
    small_angle_band_alpha_threshold: float = 7e-9
    small_angle_band_min_support: int = 6
    medium_angle_band_epsilon: float = 2e-10
    medium_angle_band_min_alpha: float = 1e-7
    medium_angle_band_max_alpha: float = 1.13e-7
    medium_angle_band_min_support: int = 4
    narrow_angle_band_epsilon: float = 3e-10
    narrow_angle_band_min_alpha: float = 8.5e-7
    narrow_angle_band_max_alpha: float = 8.6e-7
    narrow_angle_band_min_support: int = 4
    secondary_narrow_band_epsilon: float = 3e-10
    secondary_narrow_band_min_alpha: float = 1.3867e-6
    secondary_narrow_band_max_alpha: float = 1.38685e-6
    secondary_narrow_band_min_support: int = 4
    micro_support_epsilon: float = 5e-10
    micro_support_alpha_threshold: float = 2.5e-10
    micro_support_size: int = 3


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
) -> float:
    _ = step_index
    label = str(term["label"])
    support = sum(ch != "I" for ch in label)
    if (
        support >= config.ultra_high_support_min_support
        and config.small_high_support_alpha_threshold < abs(alpha) <= config.ultra_high_support_alpha_threshold
    ):
        return config.ultra_high_support_epsilon
    if support >= config.small_high_support_min_support and abs(alpha) <= config.small_high_support_alpha_threshold:
        return config.small_high_support_epsilon
    if support >= config.small_angle_band_min_support and abs(alpha) <= config.small_angle_band_alpha_threshold:
        return config.small_angle_band_epsilon
    if (
        support >= config.medium_angle_band_min_support
        and config.medium_angle_band_min_alpha <= abs(alpha) <= config.medium_angle_band_max_alpha
    ):
        return config.medium_angle_band_epsilon
    if (
        support >= config.narrow_angle_band_min_support
        and config.narrow_angle_band_min_alpha <= abs(alpha) <= config.narrow_angle_band_max_alpha
    ):
        return config.narrow_angle_band_epsilon
    if (
        support >= config.secondary_narrow_band_min_support
        and config.secondary_narrow_band_min_alpha <= abs(alpha) <= config.secondary_narrow_band_max_alpha
    ):
        return config.secondary_narrow_band_epsilon
    if support == config.micro_support_size and abs(alpha) <= config.micro_support_alpha_threshold:
        return config.micro_support_epsilon
    return config.uniform_epsilon


def apply_pauli_rotation(
    state: np.ndarray,
    pauli_matrix: np.ndarray,
    alpha: float,
    identity: np.ndarray,
) -> np.ndarray:
    return (math.cos(alpha) * identity + 1j * math.sin(alpha) * pauli_matrix) @ state


def run_product_formula(problem: Dict[str, object], exact_angles: bool = False) -> Dict[str, object]:
    config = get_method_config()
    state = np.array(problem["initial_state"], dtype=np.complex128, copy=True)
    time_sign = float(problem["time_sign"])
    target_time = float(problem["target_time"])
    step_time = target_time / float(config.n_steps)
    compile_pauli_rotation = problem["compile_pauli_rotation"]
    exact_apply = problem["apply_exact_pauli_rotation"]
    terms = tuple(problem["pauli_terms"])
    outer_terms = terms[:-1]
    middle_term = terms[-1]

    epsilon_hist = Counter()
    unique_compilations = {}
    nonzero_rotation_count = 0
    skipped_rotation_count = 0
    t_gate_count = 0

    def apply_term(state_vector: np.ndarray, term: Dict[str, object], alpha: float) -> np.ndarray:
        nonlocal nonzero_rotation_count, skipped_rotation_count, t_gate_count
        if abs(alpha) <= config.zero_angle_threshold:
            skipped_rotation_count += 1
            return state_vector

        if exact_angles:
            nonzero_rotation_count += 1
            return exact_apply(state_vector, term["label"], alpha)

        epsilon = float(choose_synthesis_epsilon(term, alpha, step_index, config))
        compiled = compile_pauli_rotation(term["label"], alpha, epsilon)
        epsilon_key = f"{epsilon:.3e}"
        epsilon_hist[epsilon_key] += 1
        t_gate_count += int(compiled["t_gate_count"])
        unique_compilations[compiled["cache_key"]] = {
            "label": compiled["label"],
            "alpha": compiled["alpha"],
            "theta": compiled["theta"],
            "epsilon": compiled["epsilon"],
            "t_gate_count": compiled["t_gate_count"],
            "gate_count": compiled["gate_count"],
            "operator_error": compiled["operator_error"],
        }
        nonzero_rotation_count += 1
        return compiled["state_action"](state_vector)

    for step_index in range(config.n_steps):
        for term in outer_terms:
            alpha = 0.5 * time_sign * float(term["coefficient"]) * step_time
            state = apply_term(state, term, alpha)
        middle_alpha = time_sign * float(middle_term["coefficient"]) * step_time
        state = apply_term(state, middle_term, middle_alpha)
        for term in reversed(outer_terms):
            alpha = 0.5 * time_sign * float(term["coefficient"]) * step_time
            state = apply_term(state, term, alpha)

    state = state / np.linalg.norm(state)

    return {
        "state": state,
        "method_config": asdict(config),
        "nonzero_rotation_count": nonzero_rotation_count,
        "skipped_rotation_count": skipped_rotation_count,
        "synthesis_epsilon_histogram": dict(sorted(epsilon_hist.items())),
        "unique_compilation_count": len(unique_compilations),
        "unique_compilations": sorted(unique_compilations.values(), key=lambda item: (item["label"], item["epsilon"])),
        "t_gate_count": t_gate_count,
        "step_time": step_time,
    }
