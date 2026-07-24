"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 84d028ffb77b88fd67fd5bfc57b10567458fc92586d9c4f3998b3dcb2665c031
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: 54c2ec1a1920ca3bcf4312ba563fa4c60aef3a891fe866620f12ea30fb879b41
Calibration scientific SHA-256: e11a3110730d91aff7db7f5f4aad0570e92cc8bb1e7f876da9c452c8f47dd21d
Calibration scientific SHA-256: f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7
Calibration scientific SHA-256: 380459e0917ba69111ad7645463ad3021de3ff36e884ea94afe2df6b364a8f59
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
    "Reaction seed co": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 435,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "54c2ec1a1920ca3bcf4312ba563fa4c60aef3a891fe866620f12ea30fb879b41",
        "double_commutator_sum_upper": 6988.968367593207,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0003984003594101289,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 7337,
        "nonidentity_terms": 435,
        "paper_eq34_cost_proxy": 104587769939903.36,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 169.3194662574334,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0002596994861329105,
        "problem_sha256": "d437a0ff99dcf2073bc5e3b033603cb035908195082c010e31e35f87453ce401",
        "qpe_fraction": 0.6518555518308801,
        "qpe_full_evolution_queries": 96777,
        "qpe_step_queries": 710052849,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 4,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 8360.005004539893,
        "rotation_fraction": 0.022216672253680063,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 8.851130210763176e-06,
        "scheduled_rotations": 6375853,
        "scheduled_rotations_per_step": 869,
        "species": "co",
        "step_repetitions": 7337,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0001362954886193267,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0003984003594101289,
        "trotter_error_hartree": 0.00012984974306645526,
        "trotter_fraction": 0.32592777591544003,
        "uniform_epsilon": 1.388226831886365e-12,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed co2": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 195,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "e11a3110730d91aff7db7f5f4aad0570e92cc8bb1e7f876da9c452c8f47dd21d",
        "double_commutator_sum_upper": 3090.0149066957038,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0003984003594101289,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 4881,
        "nonidentity_terms": 195,
        "paper_eq34_cost_proxy": 29887521884921.09,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 162.0882119184835,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0002594423387761888,
        "problem_sha256": "bb92e95e4b56fd1459407232ec5a5feb465d7319c3cff9ed5de045f8983e3365",
        "qpe_fraction": 0.6512101022205874,
        "qpe_full_evolution_queries": 96873,
        "qpe_step_queries": 472837113,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 4,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5558.790252110349,
        "rotation_fraction": 0.02318484666911891,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 9.236851245845704e-06,
        "scheduled_rotations": 1898709,
        "scheduled_rotations_per_step": 389,
        "species": "co2",
        "step_repetitions": 4881,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0002048760499897562,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0003984003594101289,
        "trotter_error_hartree": 0.0001297211693880944,
        "trotter_fraction": 0.3256050511102937,
        "uniform_epsilon": 4.864806163475133e-12,
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
        "error_constraint_value_hartree": 0.0003984003594101289,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 474,
        "nonidentity_terms": 14,
        "paper_eq34_cost_proxy": 170934656131.34277,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 132.27335411908555,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0002580935567599025,
        "problem_sha256": "ab6b66733db61c6962cf9b3537ca4b37bdbef0a5b693cbc870efa51dfb1fbc0e",
        "qpe_fraction": 0.647824608245925,
        "qpe_full_evolution_queries": 97379,
        "qpe_step_queries": 46157646,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 4,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 538.4068931007549,
        "rotation_fraction": 0.0282630876311125,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.1260024270275189e-05,
        "scheduled_rotations": 12798,
        "scheduled_rotations_per_step": 27,
        "species": "h2",
        "step_repetitions": 474,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.002109704641350211,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0003984003594101289,
        "trotter_error_hartree": 0.00012904677837995124,
        "trotter_fraction": 0.3239123041229625,
        "uniform_epsilon": 8.798268690635403e-10,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed h2o": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 259,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "380459e0917ba69111ad7645463ad3021de3ff36e884ea94afe2df6b364a8f59",
        "double_commutator_sum_upper": 11963.295455608451,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.00039840035941012897,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 9600,
        "nonidentity_terms": 259,
        "paper_eq34_cost_proxy": 80778725044632.86,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 167.8282292085484,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0002596482299752012,
        "problem_sha256": "1bb40d490a7e1610b416c538427efc6be1ac96988237e771e7d4fd243687c01a",
        "qpe_fraction": 0.6517268969326133,
        "qpe_full_evolution_queries": 96796,
        "qpe_step_queries": 929241600,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 4,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 10937.685063855355,
        "rotation_fraction": 0.022409654601080135,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 8.928014447327174e-06,
        "scheduled_rotations": 4963200,
        "scheduled_rotations_per_step": 517,
        "species": "h2o",
        "step_repetitions": 9600,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00010416666666666667,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0003984003594101289,
        "trotter_error_hartree": 0.0001298241149876006,
        "trotter_fraction": 0.32586344846630666,
        "uniform_epsilon": 1.7988423693035088e-12,
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
