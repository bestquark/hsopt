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
    uniform_epsilon: float = 1e-7


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
) -> float:
    _ = (term, alpha, step_index)
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
