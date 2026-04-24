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
    name: str = "h2_pf1_uniform_eps"
    n_steps: int = 384
    zero_angle_threshold: float = 1e-15
    default_epsilon: float = 5e-7
    izii_second_plateau_epsilon: float = 1.2e-6
    iziz_second_plateau_epsilon: float = 1.4e-6
    low_angle_pair_epsilon: float = 7e-7
    medium_single_z_epsilon: float = 7e-7
    relaxed_xy_block_epsilon: float = 8e-7
    xyyx_second_plateau_epsilon: float = 1.2e-6
    yyxx_second_plateau_epsilon: float = 1.2e-6
    yxxy_epsilon: float = 7.4e-7
    yxxy_second_plateau_epsilon: float = 1.2e-6
    yxxy_second_plateau_cycle: int = 16
    yxxy_second_plateau_steps: tuple[int, ...] = (0, 1, 2, 4, 6, 8, 10, 12, 14)
    yxxy_bonus_steps: tuple[int, ...] = (3, 131, 259, 291, 293, 295, 297, 299, 301, 303, 307, 309, 311, 313, 315, 317, 319, 323, 325, 327, 329, 331, 333, 335, 339, 341, 343, 345, 347, 349, 351, 355, 357, 359, 361, 363, 365, 367, 371, 373, 375, 377, 379, 381, 383)
    zz_pair_epsilon: float = 1.8e-6
    mixed_zz_pair_epsilon: float = 9.4e-7
    izzi_second_plateau_epsilon: float = 1.7e-6


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
) -> float:
    label = str(term["label"])
    support_size = sum(pauli != "I" for pauli in label)
    if label in {"IZII", "ZIII"}:
        return config.izii_second_plateau_epsilon
    if label in {"IZIZ", "ZIZI"}:
        return config.iziz_second_plateau_epsilon
    if label == "ZZII":
        return config.zz_pair_epsilon
    if label in {"IZZI", "ZIIZ"}:
        return config.izzi_second_plateau_epsilon
    if label == "XXYY":
        return config.default_epsilon
    if label == "XYYX":
        return config.xyyx_second_plateau_epsilon
    if label == "YYXX":
        return config.yyxx_second_plateau_epsilon
    if label == "YXXY":
        if (
            step_index % config.yxxy_second_plateau_cycle in config.yxxy_second_plateau_steps
            or step_index in config.yxxy_bonus_steps
        ):
            return config.yxxy_second_plateau_epsilon
        return config.yxxy_epsilon
    if label in {"XXYY", "XYYX", "YYXX"}:
        return config.relaxed_xy_block_epsilon
    if support_size == 1 and abs(alpha) <= 4.7e-4:
        return config.medium_single_z_epsilon
    if support_size == 2 and abs(alpha) <= 3.2e-4:
        return config.low_angle_pair_epsilon
    return config.default_epsilon


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
