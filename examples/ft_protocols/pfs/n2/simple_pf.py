"""
Editable PF2 method file for the N2 product-formula precision benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict

from pfs.array_backend import np


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str = "n2_pf2_uniform_eps"
    n_steps: int = 512
    zero_angle_threshold: float = 1e-15
    uniform_epsilon: float = 1e-10


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


def run_product_formula(problem: Dict[str, object], exact_angles: bool = False) -> Dict[str, object]:
    config = get_method_config()
    state = np.asarray(problem["initial_state"], dtype=np.complex128).copy()
    time_sign = float(problem["time_sign"])
    target_time = float(problem["target_time"])
    step_time = target_time / float(config.n_steps)
    compile_pauli_rotation = problem["compile_pauli_rotation"]
    exact_apply = problem["apply_exact_pauli_rotation"]
    simulation_terms = tuple(problem["pauli_terms"])
    cost_terms = tuple(problem.get("cost_terms", simulation_terms))
    count_simulation_terms = str(problem.get("hamiltonian_factorization", "pauli")) == "pauli"

    epsilon_hist = Counter()
    unique_compilations = {}
    nonzero_rotation_count = 0
    skipped_rotation_count = 0
    t_gate_count = 0

    def record_compilation(compiled: Dict[str, object]) -> None:
        nonlocal t_gate_count
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

    def apply_simulation_term(state_vector: np.ndarray, term: Dict[str, object], alpha: float) -> np.ndarray:
        nonlocal nonzero_rotation_count, skipped_rotation_count, t_gate_count
        if abs(alpha) <= config.zero_angle_threshold:
            skipped_rotation_count += 1
            return state_vector

        if exact_angles:
            nonzero_rotation_count += 1
            return exact_apply(state_vector, term["label"], alpha)

        epsilon = float(choose_synthesis_epsilon(term, alpha, step_index, config))
        compiled = compile_pauli_rotation(term["label"], alpha, epsilon)
        if count_simulation_terms:
            record_compilation(compiled)
        nonzero_rotation_count += 1
        return compiled["state_action"](state_vector)

    def count_cost_term(term: Dict[str, object], alpha: float) -> None:
        nonlocal nonzero_rotation_count, skipped_rotation_count
        if abs(alpha) <= config.zero_angle_threshold:
            skipped_rotation_count += 1
            return
        epsilon = float(choose_synthesis_epsilon(term, alpha, step_index, config))
        compiled = compile_pauli_rotation(term["label"], alpha, epsilon)
        record_compilation(compiled)
        nonzero_rotation_count += 1

    for step_index in range(config.n_steps):
        simulation_outer_terms = simulation_terms[:-1]
        simulation_middle_term = simulation_terms[-1]
        for term in simulation_outer_terms:
            alpha = 0.5 * time_sign * float(term["coefficient"]) * step_time
            state = apply_simulation_term(state, term, alpha)
        middle_alpha = time_sign * float(simulation_middle_term["coefficient"]) * step_time
        state = apply_simulation_term(state, simulation_middle_term, middle_alpha)
        for term in reversed(simulation_outer_terms):
            alpha = 0.5 * time_sign * float(term["coefficient"]) * step_time
            state = apply_simulation_term(state, term, alpha)

        if not count_simulation_terms and not exact_angles:
            cost_outer_terms = cost_terms[:-1]
            cost_middle_term = cost_terms[-1]
            for term in cost_outer_terms:
                alpha = 0.5 * time_sign * float(term["coefficient"]) * step_time
                count_cost_term(term, alpha)
            middle_alpha = time_sign * float(cost_middle_term["coefficient"]) * step_time
            count_cost_term(cost_middle_term, middle_alpha)
            for term in reversed(cost_outer_terms):
                alpha = 0.5 * time_sign * float(term["coefficient"]) * step_time
                count_cost_term(term, alpha)

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
