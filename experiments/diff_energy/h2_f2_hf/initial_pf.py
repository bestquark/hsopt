"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 61a302f0aded9ac3dd63743ef3dba5c674704f1d51ca852a0c4500b6ef372d79
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: 2303a70ed784ef28a5e8cefe670e5bcf01c842a806fc96af1a5d80742f10836d
Calibration scientific SHA-256: f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7
Calibration scientific SHA-256: a7b608a5171005f7b139a885fc3d61967dee9bb86efbcfe37fcd1e4f3258ce52
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
    "Reaction seed f2": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 367,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "2303a70ed784ef28a5e8cefe670e5bcf01c842a806fc96af1a5d80742f10836d",
        "double_commutator_sum_upper": 6521.885266247272,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0005312004792135053,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 6139,
        "nonidentity_terms": 367,
        "paper_eq34_cost_proxy": 54435160365457.42,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 166.37999515980533,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.000346130123111645,
        "problem_sha256": "3849835ca30c23e3f26770f17fb09328b6391fc18799ff0568a51e3493f0a752",
        "qpe_fraction": 0.6515997945335532,
        "qpe_full_evolution_queries": 72611,
        "qpe_step_queries": 445758929,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 8075.819008773829,
        "rotation_fraction": 0.022600308199670405,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.2005294546037831e-05,
        "scheduled_rotations": 4499887,
        "scheduled_rotations_per_step": 733,
        "species": "f2",
        "step_repetitions": 6139,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00016289297931259162,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0005312004792135053,
        "trotter_error_hartree": 0.0001730650615558225,
        "trotter_fraction": 0.3257998972667766,
        "uniform_epsilon": 2.667910226642987e-12,
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
    "Reaction seed hf": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 275,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "a7b608a5171005f7b139a885fc3d61967dee9bb86efbcfe37fcd1e4f3258ce52",
        "double_commutator_sum_upper": 24011.951323908255,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.00026560023960675263,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 16652,
        "nonidentity_terms": 275,
        "paper_eq34_cost_proxy": 229474629407198.3,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 172.68527297312016,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0001732079941312274,
        "problem_sha256": "f28e7fb5454bf01f9e2e94f4df08b372bd05061109757d4ce2c03dfd5ab95e9e",
        "qpe_fraction": 0.6521379438048661,
        "qpe_full_evolution_queries": 145102,
        "qpe_step_queries": 2416238504,
        "reaction_coefficient": 2.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 15495.790177950996,
        "rotation_fraction": 0.02179308429270099,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 5.78824840991154e-06,
        "scheduled_rotations": 9141948,
        "scheduled_rotations_per_step": 549,
        "species": "hf",
        "step_repetitions": 16652,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 6.005284650492433e-05,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.00026560023960675263,
        "trotter_error_hartree": 8.66039970656137e-05,
        "trotter_fraction": 0.32606897190243306,
        "uniform_epsilon": 6.331526289486157e-13,
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
