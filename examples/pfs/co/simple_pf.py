"""
Editable PF2 method file for the CO product-formula precision benchmark.

Synthesis precision follows Reiher et al. (2017): a total synthesis error
budget is split uniformly across all rotations.  The budget is derived from
chemical accuracy (1.6 mHa) with ~1/3 allocated to synthesis error.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict

from pfs.array_backend import np


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str = "co_pf2_reiher_uniform"
    n_steps: int = 512
    zero_angle_threshold: float = 1e-15
    synthesis_error_budget_hartree: float = 5e-4


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
    n_terms: int,
) -> float:
    """Reiher-style uniform synthesis precision: total budget / total rotations."""
    _ = (term, alpha, step_index)
    rotations_per_step = 2 * n_terms - 1
    total_rotations = config.n_steps * rotations_per_step
    return config.synthesis_error_budget_hartree / total_rotations


def run_product_formula(problem: Dict[str, object], exact_angles: bool = False) -> Dict[str, object]:
    config = get_method_config()
    state = np.asarray(problem["initial_state"], dtype=np.complex128).copy()
    time_sign = float(problem["time_sign"])
    target_time = float(problem["target_time"])
    step_time = target_time / float(config.n_steps)
    compile_pauli_rotation = problem["compile_pauli_rotation"]
    exact_apply = problem["apply_exact_pauli_rotation"]
    pauli_terms = tuple(problem["pauli_terms"])
    n_terms = len(pauli_terms)

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

        epsilon = float(choose_synthesis_epsilon(term, alpha, step_index, config, n_terms))
        compiled = compile_pauli_rotation(term["label"], alpha, epsilon)
        epsilon_key = f"{float(compiled['epsilon']):.3e}"
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
        outer_terms = pauli_terms[:-1]
        middle_term = pauli_terms[-1]
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
