"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: a438a4e4caba0a93689b82e6599c8b87e4b39f7dc567016dbcb9ebcbfd83df72
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: a390a0e40f07017621ab91825cb0fb4ec1506d6aecd459a30b12b54e4e2d90e8
Calibration scientific SHA-256: 2fc2427ce26d2e6b062f56addeeeefc16bfd2a124d53350d8b771d5531516c68
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
    "Reaction seed acetylene": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 195,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "a390a0e40f07017621ab91825cb0fb4ec1506d6aecd459a30b12b54e4e2d90e8",
        "double_commutator_sum_upper": 3129.064101886328,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202579,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 3474,
        "nonidentity_terms": 195,
        "paper_eq34_cost_proxy": 10373243281255.24,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 157.97976771745346,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005185720053892379,
        "problem_sha256": "f5ba3897cce3e6cc2e048f50a63a356ae217a63108d2d7ae7bde1c07a7a009dc",
        "qpe_fraction": 0.6508176927312954,
        "qpe_full_evolution_queries": 48466,
        "qpe_step_queries": 168370884,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5593.803805896599,
        "rotation_fraction": 0.02377346090305686,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.8942710736401003e-05,
        "scheduled_rotations": 1351386,
        "scheduled_rotations_per_step": 389,
        "species": "acetylene",
        "step_repetitions": 3474,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0002878526194588371,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00025928600269461895,
        "trotter_fraction": 0.3254088463656477,
        "uniform_epsilon": 1.4017246542735389e-11,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed vinylidene": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 251,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "2fc2427ce26d2e6b062f56addeeeefc16bfd2a124d53350d8b771d5531516c68",
        "double_commutator_sum_upper": 2351.6492682507155,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202579,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 3012,
        "nonidentity_terms": 251,
        "paper_eq34_cost_proxy": 11621670050267.031,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 158.6358528914577,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005186229975079434,
        "problem_sha256": "84721d874a02255a9518ae164cf94789b848af3c296325a9f5b3561076750af4",
        "qpe_fraction": 0.6508816888064759,
        "qpe_full_evolution_queries": 48461,
        "qpe_step_queries": 145964532,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 4849.380649372366,
        "rotation_fraction": 0.02367746679028614,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.886622255834278e-05,
        "scheduled_rotations": 1509012,
        "scheduled_rotations_per_step": 501,
        "species": "vinylidene",
        "step_repetitions": 3012,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00033200531208499334,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.0002593114987539717,
        "trotter_fraction": 0.32544084440323795,
        "uniform_epsilon": 1.2502367481731608e-11,
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
