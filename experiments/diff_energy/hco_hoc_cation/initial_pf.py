"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 6de81d0aaf29ad67c82bc0799c0ce65635e07570ae61ed8afc333befe9a7da2e
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: 7a4b1f09cfd9592e636660a4da63c24f667b826ac52dfdbc3b2a31044e192897
Calibration scientific SHA-256: 1fce03fe8f4745a69d6063a697decd659f6356afbefa0beafadb6981f6de4f7a
Generator: experiments/generate_initial_pf.py
Paper: Reiher et al., PNAS 114, 7555-7560 (2017), DOI 10.1073/pnas.1619152114

Paper structure: PF2/Strang, lexicographic order, Eq. (19) bound beta(H),
the paper's Trotter/QPE/synthesis error relation, and uniform Clifford+T
rotation tolerance.

Benchmark adaptation: beta is referenced to 0.1 mHa; the full error target is
1 kcal/mol; error allocation follows the supplement's Eq. (34) calculus using
the rigorous-row gamma=4 and delta_0=11. PyGridSynth is not run here: iteration
0 compiles the generated policy and records its realized T count. Reaction
initializers conservatively give each signed species term an equal absolute
share of the reaction-error target.

beta(H) is obtained from the paper's rigorous Eq. (19) double-commutator coefficient bound for this Pauli decomposition. The separately cached exact energy is used only by the evaluator and does not enter beta, the Trotter step count, or the synthesis tolerance.

This file is measured unchanged as iteration 0. Scientific acceptance is based
exclusively on exact diagonalization of the compiled PF unitary against the
separately cached U_exact reference.
"""

from dataclasses import dataclass


BASELINES = {
    "Reaction seed hco_plus": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 435,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "7a4b1f09cfd9592e636660a4da63c24f667b826ac52dfdbc3b2a31044e192897",
        "double_commutator_sum_upper": 8189.897984430174,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202577,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 5618,
        "nonidentity_terms": 435,
        "paper_eq34_cost_proxy": 39190662188126.0,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 165.65351876092,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005191437302317956,
        "problem_sha256": "0ec664ac77df3aa9e0499906f3f4bda452e69eec7f870b4f33dd3c560ba4fab7",
        "qpe_fraction": 0.6515352182418198,
        "qpe_full_evolution_queries": 48412,
        "qpe_step_queries": 271978616,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 9049.805514169999,
        "rotation_fraction": 0.02269717263727023,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.8085123472564408e-05,
        "scheduled_rotations": 4882042,
        "scheduled_rotations_per_step": 869,
        "species": "hco_plus",
        "step_repetitions": 5618,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.000177999288002848,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.0002595718651158978,
        "trotter_fraction": 0.3257676091209099,
        "uniform_epsilon": 3.70441783838902e-12,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed hoc_plus": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 435,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "1fce03fe8f4745a69d6063a697decd659f6356afbefa0beafadb6981f6de4f7a",
        "double_commutator_sum_upper": 6803.944886165211,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 5120,
        "nonidentity_terms": 435,
        "paper_eq34_cost_proxy": 35605650456656.14,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 165.09968411462236,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005191042062958232,
        "problem_sha256": "bd1cd5269620e6d65b9159922d5ef4b8b353ce6bf1d3d1ed299c30ef9a005f95",
        "qpe_fraction": 0.65148561495327,
        "qpe_full_evolution_queries": 48416,
        "qpe_step_queries": 247889920,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 8248.60284300633,
        "rotation_fraction": 0.022771577570094948,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.8144409376522916e-05,
        "scheduled_rotations": 4449280,
        "scheduled_rotations_per_step": 869,
        "species": "hoc_plus",
        "step_repetitions": 5120,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0001953125,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.0002595521031479116,
        "trotter_fraction": 0.325742807476635,
        "uniform_epsilon": 4.078055185675641e-12,
        "zero_angle_threshold": 0.0,
    },
}


@dataclass(frozen=True)
class ProductFormulaConfig:
    formula_order: int
    n_steps: int
    zero_angle_threshold: float
    uniform_epsilon: float
    term_order_strategy: str
    baseline_source: str


def get_method_config(problem=None) -> ProductFormulaConfig:
    key = next(iter(BASELINES)) if problem is None else str(problem["name"])
    row = BASELINES[key]
    return ProductFormulaConfig(
        formula_order=int(row["formula_order"]),
        n_steps=int(row["n_steps"]),
        zero_angle_threshold=float(row["zero_angle_threshold"]),
        uniform_epsilon=float(row["uniform_epsilon"]),
        term_order_strategy=str(row["term_order_strategy"]),
        baseline_source=str(row["baseline_source"]),
    )


def choose_synthesis_epsilon(term, alpha, step_index, config) -> float:
    _ = (term, alpha, step_index)
    return config.uniform_epsilon
