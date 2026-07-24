"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 67fc570923800dc9f400e1cd65b09330e3d28a20eacd6941a9754df5ca438007
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: 6922794f3b3d91828ffde8824b2acba0de0767e639133d01bf56a2bc6c0d8dda
Calibration scientific SHA-256: f9577437ab9302d538392bbbaac2e2cdefc9025b612643b5a53162e6c8e2d888
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
    "HamLib H2 linear R=0.7 STO-6G Jordan-Wigner": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 14,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "6922794f3b3d91828ffde8824b2acba0de0767e639133d01bf56a2bc6c0d8dda",
        "double_commutator_sum_upper": 0.41861518626301725,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 41,
        "nonidentity_terms": 14,
        "paper_eq34_cost_proxy": 6368814384.974663,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 115.27624966210077,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005140449343435017,
        "problem_sha256": "7e741dc5e8e4d37c3cb8e17dc6bfe970da921bcd0242e2d50940161f1f312d4a",
        "qpe_fraction": 0.6451361327893831,
        "qpe_full_evolution_queries": 48893,
        "qpe_step_queries": 2004613,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 64.7004780711099,
        "rotation_fraction": 0.03229580081592552,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 2.5733317305005322e-05,
        "scheduled_rotations": 1107,
        "scheduled_rotations_per_step": 27,
        "species": "h2_r0_7",
        "step_repetitions": 41,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.024390243902439025,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00025702246717175086,
        "trotter_fraction": 0.32256806639469154,
        "uniform_epsilon": 2.3245995758812398e-08,
        "zero_angle_threshold": 0.0,
    },
    "HamLib H2 linear R=0.8 STO-6G Jordan-Wigner": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 14,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "f9577437ab9302d538392bbbaac2e2cdefc9025b612643b5a53162e6c8e2d888",
        "double_commutator_sum_upper": 0.3302852845790427,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 36,
        "nonidentity_terms": 14,
        "paper_eq34_cost_proxy": 5623560806.565018,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 114.55750146026256,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005139407955867465,
        "problem_sha256": "35cd94cfa48d4069a68d1720bbac306e5a91add8e800fc60df4b79083f0ca213",
        "qpe_fraction": 0.645005436676421,
        "qpe_full_evolution_queries": 48903,
        "qpe_step_queries": 1760508,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 57.470451936542375,
        "rotation_fraction": 0.03249184498536861,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 2.5889525440138098e-05,
        "scheduled_rotations": 972,
        "scheduled_rotations_per_step": 27,
        "species": "h2_r0_8",
        "step_repetitions": 36,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.027777777777777776,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00025697039779337325,
        "trotter_fraction": 0.3225027183382105,
        "uniform_epsilon": 2.6635314238825205e-08,
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
