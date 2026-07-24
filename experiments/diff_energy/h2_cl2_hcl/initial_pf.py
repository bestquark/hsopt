"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: fb1531f232e5ec2ff2668479ec619e66367871ff883f385c823c932f64142117
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: 824d9587adb641cdbc7af4477f2c499af03ce07ce71b55be89c350a9eaf7586f
Calibration scientific SHA-256: f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7
Calibration scientific SHA-256: 54d82a7c27c5b45e5619f7bc3d1457fecb095337384be8cde96ec505eb0ffed1
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
    "Reaction seed cl2": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 435,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "824d9587adb641cdbc7af4477f2c499af03ce07ce71b55be89c350a9eaf7586f",
        "double_commutator_sum_upper": 2635.6783875656065,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0005312004792135053,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 3903,
        "nonidentity_terms": 435,
        "paper_eq34_cost_proxy": 40614135982746.23,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 164.6890982872684,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0003460498250124585,
        "problem_sha256": "c92cd6851e87a1b847956647aa9b6184dc25262a3e1038f23f02c6925cc5bad4",
        "qpe_fraction": 0.6514486310795867,
        "qpe_full_evolution_queries": 72628,
        "qpe_step_queries": 283467084,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5133.885845600394,
        "rotation_fraction": 0.02282705338061985,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.212574169481753e-05,
        "scheduled_rotations": 3391707,
        "scheduled_rotations_per_step": 869,
        "species": "cl2",
        "step_repetitions": 3903,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00025621316935690495,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0005312004792135053,
        "trotter_error_hartree": 0.00017302491250622924,
        "trotter_fraction": 0.32572431553979336,
        "uniform_epsilon": 3.575114741579249e-12,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed h2": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 14,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7",
        "double_commutator_sum_upper": 28.98819825384078,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0005312004792135053,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 411,
        "nonidentity_terms": 14,
        "paper_eq34_cost_proxy": 109631457356.37128,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 130.539213554326,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0003439955859914337,
        "problem_sha256": "ab6b66733db61c6962cf9b3537ca4b37bdbef0a5b693cbc870efa51dfb1fbc0e",
        "qpe_fraction": 0.6475814677365372,
        "qpe_full_evolution_queries": 73062,
        "qpe_step_queries": 30028482,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 538.4068931007549,
        "rotation_fraction": 0.028627798395194123,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.5207100226354734e-05,
        "scheduled_rotations": 11097,
        "scheduled_rotations_per_step": 27,
        "species": "h2",
        "step_repetitions": 411,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0024330900243309003,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0005312004792135053,
        "trotter_error_hartree": 0.00017199779299571685,
        "trotter_fraction": 0.3237907338682686,
        "uniform_epsilon": 1.3703794022127365e-09,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed hcl": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 275,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "54d82a7c27c5b45e5619f7bc3d1457fecb095337384be8cde96ec505eb0ffed1",
        "double_commutator_sum_upper": 4929.411054746109,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.00026560023960675263,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 7547,
        "nonidentity_terms": 275,
        "paper_eq34_cost_proxy": 101220564476581.61,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 167.9601755801937,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.000173101867373537,
        "problem_sha256": "89146d3e777be4893c4056489167ec7e68eb2b131d8979bf7f6d75d4345ff80b",
        "qpe_fraction": 0.651738370529452,
        "qpe_full_evolution_queries": 145191,
        "qpe_step_queries": 1095756477,
        "reaction_coefficient": 2.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 7020.97646680724,
        "rotation_fraction": 0.02239244420582187,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 5.947438546447128e-06,
        "scheduled_rotations": 4143303,
        "scheduled_rotations_per_step": 549,
        "species": "hcl",
        "step_repetitions": 7547,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00013250298131707962,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.00026560023960675263,
        "trotter_error_hartree": 8.65509336867685e-05,
        "trotter_fraction": 0.325869185264726,
        "uniform_epsilon": 1.4354341322483844e-12,
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
