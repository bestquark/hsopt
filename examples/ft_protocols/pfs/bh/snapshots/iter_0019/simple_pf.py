"""
Editable PF1 method file for the localized small-molecule precision benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str = "bh_pf1_structured_eps"
    n_steps: int = 192
    zero_angle_threshold: float = 1e-15
    base_epsilon: float = 1e-7
    contiguous_zz_epsilon: float = 2e-7
    medium_support1_alpha_cutoff: float = 1e-3
    medium_support1_epsilon: float = 1e-6
    large_support1_alpha_cutoff: float = 1.5e-3
    large_support1_epsilon: float = 7e-7
    split_support2_mid_alpha_floor: float = 6.2e-4
    split_support2_mid_alpha_cutoff: float = 6.5e-4
    split_support2_mid_epsilon: float = 5e-7
    split_support2_alpha_cutoff: float = 7e-4
    split_support2_epsilon: float = 4e-7
    small_support4_alpha_cutoff: float = 5e-5
    small_support4_epsilon: float = 1e-6


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
) -> float:
    _ = (alpha, step_index)
    label = str(term["label"])
    support_size = int(term["support_size"])
    abs_alpha = abs(alpha)

    if support_size == 1 and abs_alpha >= config.large_support1_alpha_cutoff:
        return config.large_support1_epsilon
    if support_size == 1 and abs_alpha >= config.medium_support1_alpha_cutoff:
        return config.medium_support1_epsilon
    if support_size == 2 and label.count("Z") == 2 and "ZZ" in label:
        return config.contiguous_zz_epsilon
    if (
        support_size == 2
        and label.count("Z") == 2
        and "ZZ" not in label
        and config.split_support2_mid_alpha_floor <= abs_alpha <= config.split_support2_mid_alpha_cutoff
    ):
        return config.split_support2_mid_epsilon
    if support_size == 2 and label.count("Z") == 2 and "ZZ" not in label and abs_alpha <= config.split_support2_alpha_cutoff:
        return config.split_support2_epsilon
    if support_size == 4 and abs_alpha <= config.small_support4_alpha_cutoff:
        return config.small_support4_epsilon
    return config.base_epsilon


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
    identity = np.array(problem["identity"], dtype=np.complex128, copy=False)
    time_sign = float(problem["time_sign"])
    target_time = float(problem["target_time"])
    step_time = target_time / float(config.n_steps)
    compile_pauli_rotation = problem["compile_pauli_rotation"]

    epsilon_hist = Counter()
    unique_compilations = {}
    nonzero_rotation_count = 0
    skipped_rotation_count = 0
    t_gate_count = 0

    for step_index in range(config.n_steps):
        for term in problem["pauli_terms"]:
            coefficient = float(term["coefficient"])
            alpha = time_sign * coefficient * step_time

            if exact_angles:
                if abs(alpha) <= config.zero_angle_threshold:
                    skipped_rotation_count += 1
                    continue
                state = apply_pauli_rotation(state, term["matrix"], alpha, identity)
                nonzero_rotation_count += 1
            else:
                epsilon = float(choose_synthesis_epsilon(term, alpha, step_index, config))
                compiled = compile_pauli_rotation(term["label"], alpha, epsilon)
                state = compiled["unitary"] @ state
                nonzero_rotation_count += 1
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
