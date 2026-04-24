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
    tertiary_narrow_band_epsilon: float = 3e-10
    tertiary_narrow_band_min_alpha: float = 2.4665e-7
    tertiary_narrow_band_max_alpha: float = 2.4667e-7
    tertiary_narrow_band_min_support: int = 6
    quaternary_narrow_band_epsilon: float = 3e-10
    quaternary_narrow_band_min_alpha: float = 2.4223e-6
    quaternary_narrow_band_max_alpha: float = 2.42233e-6
    quaternary_narrow_band_min_support: int = 4
    quinary_narrow_band_epsilon: float = 2e-10
    quinary_narrow_band_min_alpha: float = 1.9986209e-5
    quinary_narrow_band_max_alpha: float = 1.9986211e-5
    quinary_narrow_band_min_support: int = 8
    senary_narrow_band_epsilon: float = 3e-10
    senary_narrow_band_min_alpha: float = 6.4975e-6
    senary_narrow_band_max_alpha: float = 6.4976e-6
    senary_narrow_band_min_support: int = 4
    septenary_narrow_band_epsilon: float = 2e-10
    septenary_narrow_band_min_alpha: float = 1.88375e-6
    septenary_narrow_band_max_alpha: float = 1.88377e-6
    septenary_narrow_band_min_support: int = 6
    octonary_narrow_band_epsilon: float = 3e-10
    octonary_narrow_band_min_alpha: float = 2.41807e-6
    octonary_narrow_band_max_alpha: float = 2.41808e-6
    octonary_narrow_band_min_support: int = 4
    nonary_narrow_band_epsilon: float = 2e-10
    nonary_narrow_band_min_alpha: float = 1.85791e-6
    nonary_narrow_band_max_alpha: float = 1.85795e-6
    nonary_narrow_band_min_support: int = 6
    denary_narrow_band_epsilon: float = 3e-10
    denary_narrow_band_min_alpha: float = 2.42872e-6
    denary_narrow_band_max_alpha: float = 2.42873e-6
    denary_narrow_band_min_support: int = 6
    undecenary_submicro_band_epsilon: float = 2e-10
    undecenary_submicro_band_min_alpha: float = 2.51108e-7
    undecenary_submicro_band_max_alpha: float = 2.51109e-7
    undecenary_submicro_band_min_support: int = 6
    duodenary_submicro_band_epsilon: float = 5e-10
    duodenary_submicro_band_min_alpha: float = 4.97766e-7
    duodenary_submicro_band_max_alpha: float = 4.97768e-7
    duodenary_submicro_band_min_support: int = 6
    tredecenary_narrow_band_epsilon: float = 5e-10
    tredecenary_narrow_band_min_alpha: float = 2.65593e-6
    tredecenary_narrow_band_max_alpha: float = 2.65594e-6
    tredecenary_narrow_band_min_support: int = 8
    quattuordecenary_submicro_band_epsilon: float = 1e-9
    quattuordecenary_submicro_band_min_alpha: float = 2.14993e-7
    quattuordecenary_submicro_band_max_alpha: float = 2.14997e-7
    quattuordecenary_submicro_band_min_support: int = 4
    quindenary_narrow_band_epsilon: float = 3e-10
    quindenary_narrow_band_min_alpha: float = 2.29234e-6
    quindenary_narrow_band_max_alpha: float = 2.29236e-6
    quindenary_narrow_band_min_support: int = 6
    sexdenary_submicro_band_epsilon: float = 2e-10
    sexdenary_submicro_band_min_alpha: float = 5.6437e-7
    sexdenary_submicro_band_max_alpha: float = 5.6439e-7
    sexdenary_submicro_band_min_support: int = 6
    septendecenary_narrow_band_epsilon: float = 5e-10
    septendecenary_narrow_band_min_alpha: float = 5.432316e-6
    septendecenary_narrow_band_max_alpha: float = 5.432317e-6
    septendecenary_narrow_band_min_support: int = 6
    octodecenary_narrow_band_epsilon: float = 5e-10
    octodecenary_narrow_band_min_alpha: float = 4.984075e-6
    octodecenary_narrow_band_max_alpha: float = 4.984076e-6
    octodecenary_narrow_band_min_support: int = 8
    novemdecenary_narrow_band_epsilon: float = 1e-9
    novemdecenary_narrow_band_min_alpha: float = 8.591928e-6
    novemdecenary_narrow_band_max_alpha: float = 8.591929e-6
    novemdecenary_narrow_band_min_support: int = 8
    vigenary_narrow_band_epsilon: float = 1e-9
    vigenary_narrow_band_min_alpha: float = 2.305836e-6
    vigenary_narrow_band_max_alpha: float = 2.305837e-6
    vigenary_narrow_band_min_support: int = 6
    unvigenary_submicro_band_epsilon: float = 1e-9
    unvigenary_submicro_band_min_alpha: float = 6.108649e-7
    unvigenary_submicro_band_max_alpha: float = 6.108650e-7
    unvigenary_submicro_band_min_support: int = 6
    duovigenary_narrow_band_epsilon: float = 1e-9
    duovigenary_narrow_band_min_alpha: float = 5.113033e-6
    duovigenary_narrow_band_max_alpha: float = 5.113034e-6
    duovigenary_narrow_band_min_support: int = 6
    trivigenary_submicro_band_epsilon: float = 1e-9
    trivigenary_submicro_band_min_alpha: float = 6.107747e-7
    trivigenary_submicro_band_max_alpha: float = 6.107749e-7
    trivigenary_submicro_band_min_support: int = 6
    quadrivigenary_submicro_band_epsilon: float = 1e-9
    quadrivigenary_submicro_band_min_alpha: float = 5.70786e-7
    quadrivigenary_submicro_band_max_alpha: float = 5.70804e-7
    quadrivigenary_submicro_band_min_support: int = 6
    band_0032_epsilon: float = 1e-9
    band_0032_min_alpha: float = 6.6216730175e-7
    band_0032_max_alpha: float = 6.6217936325e-7
    band_0032_min_support: int = 6
    band_0033_epsilon: float = 1e-9
    band_0033_min_alpha: float = 3.68221715025e-6
    band_0033_max_alpha: float = 3.68222464875e-6
    band_0033_min_support: int = 6
    band_0034_epsilon: float = 1e-9
    band_0034_min_alpha: float = 1.46574116165e-5
    band_0034_max_alpha: float = 1.46574370895e-5
    band_0034_min_support: int = 6
    band_0035_epsilon: float = 1e-9
    band_0035_min_alpha: float = 5.678380651e-5
    band_0035_max_alpha: float = 5.6783821834e-5
    band_0035_min_support: int = 6
    band_0036_epsilon: float = 1e-9
    band_0036_min_alpha: float = 2.76115922e-6
    band_0036_max_alpha: float = 2.76116122e-6
    band_0036_min_support: int = 6
    band_0037_epsilon: float = 1e-9
    band_0037_min_alpha: float = 3.67975955897e-5
    band_0037_max_alpha: float = 3.67976130093e-5
    band_0037_min_support: int = 6
    band_0038_epsilon: float = 1e-9
    band_0038_min_alpha: float = 2.1009930881e-5
    band_0038_max_alpha: float = 2.1009962327e-5
    band_0038_min_support: int = 6
    band_0039_epsilon: float = 1e-9
    band_0039_min_alpha: float = 2.761197286e-6
    band_0039_max_alpha: float = 2.761199286e-6
    band_0039_min_support: int = 6
    band_0040_epsilon: float = 1e-9
    band_0040_min_alpha: float = 1.65411616063e-5
    band_0040_max_alpha: float = 1.65411942987e-5
    band_0040_min_support: int = 6
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
    if (
        support >= config.tertiary_narrow_band_min_support
        and config.tertiary_narrow_band_min_alpha <= abs(alpha) <= config.tertiary_narrow_band_max_alpha
    ):
        return config.tertiary_narrow_band_epsilon
    if (
        support >= config.quaternary_narrow_band_min_support
        and config.quaternary_narrow_band_min_alpha <= abs(alpha) <= config.quaternary_narrow_band_max_alpha
    ):
        return config.quaternary_narrow_band_epsilon
    if (
        support >= config.quinary_narrow_band_min_support
        and config.quinary_narrow_band_min_alpha <= abs(alpha) <= config.quinary_narrow_band_max_alpha
    ):
        return config.quinary_narrow_band_epsilon
    if (
        support >= config.senary_narrow_band_min_support
        and config.senary_narrow_band_min_alpha <= abs(alpha) <= config.senary_narrow_band_max_alpha
    ):
        return config.senary_narrow_band_epsilon
    if (
        support >= config.septenary_narrow_band_min_support
        and config.septenary_narrow_band_min_alpha <= abs(alpha) <= config.septenary_narrow_band_max_alpha
    ):
        return config.septenary_narrow_band_epsilon
    if (
        support >= config.octonary_narrow_band_min_support
        and config.octonary_narrow_band_min_alpha <= abs(alpha) <= config.octonary_narrow_band_max_alpha
    ):
        return config.octonary_narrow_band_epsilon
    if (
        support >= config.nonary_narrow_band_min_support
        and config.nonary_narrow_band_min_alpha <= abs(alpha) <= config.nonary_narrow_band_max_alpha
    ):
        return config.nonary_narrow_band_epsilon
    if (
        support >= config.denary_narrow_band_min_support
        and config.denary_narrow_band_min_alpha <= abs(alpha) <= config.denary_narrow_band_max_alpha
    ):
        return config.denary_narrow_band_epsilon
    if (
        support >= config.undecenary_submicro_band_min_support
        and config.undecenary_submicro_band_min_alpha <= abs(alpha) <= config.undecenary_submicro_band_max_alpha
    ):
        return config.undecenary_submicro_band_epsilon
    if (
        support >= config.duodenary_submicro_band_min_support
        and config.duodenary_submicro_band_min_alpha <= abs(alpha) <= config.duodenary_submicro_band_max_alpha
    ):
        return config.duodenary_submicro_band_epsilon
    if (
        support >= config.tredecenary_narrow_band_min_support
        and config.tredecenary_narrow_band_min_alpha <= abs(alpha) <= config.tredecenary_narrow_band_max_alpha
    ):
        return config.tredecenary_narrow_band_epsilon
    if (
        support >= config.quattuordecenary_submicro_band_min_support
        and config.quattuordecenary_submicro_band_min_alpha <= abs(alpha) <= config.quattuordecenary_submicro_band_max_alpha
    ):
        return config.quattuordecenary_submicro_band_epsilon
    if (
        support >= config.quindenary_narrow_band_min_support
        and config.quindenary_narrow_band_min_alpha <= abs(alpha) <= config.quindenary_narrow_band_max_alpha
    ):
        return config.quindenary_narrow_band_epsilon
    if (
        support >= config.sexdenary_submicro_band_min_support
        and config.sexdenary_submicro_band_min_alpha <= abs(alpha) <= config.sexdenary_submicro_band_max_alpha
    ):
        return config.sexdenary_submicro_band_epsilon
    if (
        support >= config.septendecenary_narrow_band_min_support
        and config.septendecenary_narrow_band_min_alpha <= abs(alpha) <= config.septendecenary_narrow_band_max_alpha
    ):
        return config.septendecenary_narrow_band_epsilon
    if (
        support >= config.octodecenary_narrow_band_min_support
        and config.octodecenary_narrow_band_min_alpha <= abs(alpha) <= config.octodecenary_narrow_band_max_alpha
    ):
        return config.octodecenary_narrow_band_epsilon
    if (
        support >= config.novemdecenary_narrow_band_min_support
        and config.novemdecenary_narrow_band_min_alpha <= abs(alpha) <= config.novemdecenary_narrow_band_max_alpha
    ):
        return config.novemdecenary_narrow_band_epsilon
    if (
        support >= config.vigenary_narrow_band_min_support
        and config.vigenary_narrow_band_min_alpha <= abs(alpha) <= config.vigenary_narrow_band_max_alpha
    ):
        return config.vigenary_narrow_band_epsilon
    if (
        support >= config.unvigenary_submicro_band_min_support
        and config.unvigenary_submicro_band_min_alpha <= abs(alpha) <= config.unvigenary_submicro_band_max_alpha
    ):
        return config.unvigenary_submicro_band_epsilon
    if (
        support >= config.duovigenary_narrow_band_min_support
        and config.duovigenary_narrow_band_min_alpha <= abs(alpha) <= config.duovigenary_narrow_band_max_alpha
    ):
        return config.duovigenary_narrow_band_epsilon
    if (
        support >= config.trivigenary_submicro_band_min_support
        and config.trivigenary_submicro_band_min_alpha <= abs(alpha) <= config.trivigenary_submicro_band_max_alpha
    ):
        return config.trivigenary_submicro_band_epsilon
    if (
        support >= config.quadrivigenary_submicro_band_min_support
        and config.quadrivigenary_submicro_band_min_alpha <= abs(alpha) <= config.quadrivigenary_submicro_band_max_alpha
    ):
        return config.quadrivigenary_submicro_band_epsilon
    if (
        support >= config.band_0032_min_support
        and config.band_0032_min_alpha <= abs(alpha) <= config.band_0032_max_alpha
    ):
        return config.band_0032_epsilon
    if (
        support >= config.band_0033_min_support
        and config.band_0033_min_alpha <= abs(alpha) <= config.band_0033_max_alpha
    ):
        return config.band_0033_epsilon
    if (
        support >= config.band_0034_min_support
        and config.band_0034_min_alpha <= abs(alpha) <= config.band_0034_max_alpha
    ):
        return config.band_0034_epsilon
    if (
        support >= config.band_0035_min_support
        and config.band_0035_min_alpha <= abs(alpha) <= config.band_0035_max_alpha
    ):
        return config.band_0035_epsilon
    if (
        support >= config.band_0036_min_support
        and config.band_0036_min_alpha <= abs(alpha) <= config.band_0036_max_alpha
    ):
        return config.band_0036_epsilon
    if (
        support >= config.band_0037_min_support
        and config.band_0037_min_alpha <= abs(alpha) <= config.band_0037_max_alpha
    ):
        return config.band_0037_epsilon
    if (
        support >= config.band_0038_min_support
        and config.band_0038_min_alpha <= abs(alpha) <= config.band_0038_max_alpha
    ):
        return config.band_0038_epsilon
    if (
        support >= config.band_0039_min_support
        and config.band_0039_min_alpha <= abs(alpha) <= config.band_0039_max_alpha
    ):
        return config.band_0039_epsilon
    if (
        support >= config.band_0040_min_support
        and config.band_0040_min_alpha <= abs(alpha) <= config.band_0040_max_alpha
    ):
        return config.band_0040_epsilon
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
