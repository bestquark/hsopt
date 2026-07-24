"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 909f2ce3a0c0983513eeb8a8b9f65a994492d5da3cdc1a816cbb9463f71685d7
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7
Calibration scientific SHA-256: e15dc0fd40ffe3ae7d848e7765c01c2e547df86bbef0da6eb15c7f5488e8348b
Calibration scientific SHA-256: aa715d31c244336583641268066a22d2fa8bccb33a603d9285c320dd98d507e7
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
    "Reaction seed h2": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 14,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7",
        "double_commutator_sum_upper": 28.98819825384078,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.00017706682640450174,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 711,
        "nonidentity_terms": 14,
        "paper_eq34_cost_proxy": 597299181347.5089,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 137.156444224525,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.00011482378681781733,
        "problem_sha256": "ab6b66733db61c6962cf9b3537ca4b37bdbef0a5b693cbc870efa51dfb1fbc0e",
        "qpe_fraction": 0.6484771266838386,
        "qpe_full_evolution_queries": 218881,
        "qpe_step_queries": 155624391,
        "reaction_coefficient": -3.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 538.4068931007549,
        "rotation_fraction": 0.027284309974242138,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 4.831146177775748e-06,
        "scheduled_rotations": 19197,
        "scheduled_rotations_per_step": 27,
        "species": "h2",
        "step_repetitions": 711,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0014064697609001407,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.00017706682640450174,
        "trotter_error_hartree": 5.7411893408908665e-05,
        "trotter_fraction": 0.3242385633419193,
        "uniform_epsilon": 2.516615188714772e-10,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed n2": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 391,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "e15dc0fd40ffe3ae7d848e7765c01c2e547df86bbef0da6eb15c7f5488e8348b",
        "double_commutator_sum_upper": 3249.173455831248,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0005312004792135053,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 4334,
        "nonidentity_terms": 391,
        "paper_eq34_cost_proxy": 40529793793649.04,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 164.67709701602314,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.000346049249331566,
        "problem_sha256": "abef09574fb9793daef30f20272c1fdba7835acbb19ee37c8614c009e82a7ccc",
        "qpe_fraction": 0.6514475473439445,
        "qpe_full_evolution_queries": 72628,
        "qpe_step_queries": 314769752,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5700.152152207209,
        "rotation_fraction": 0.02282867898408325,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.2126605216156298e-05,
        "scheduled_rotations": 3384854,
        "scheduled_rotations_per_step": 781,
        "species": "n2",
        "step_repetitions": 4334,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00023073373327180433,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0005312004792135053,
        "trotter_error_hartree": 0.000173024624665783,
        "trotter_fraction": 0.32572377367197225,
        "uniform_epsilon": 3.5826080581780773e-12,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed nh3": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 443,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "aa715d31c244336583641268066a22d2fa8bccb33a603d9285c320dd98d507e7",
        "double_commutator_sum_upper": 7366.172098145875,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.00026560023960675263,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 9223,
        "nonidentity_terms": 443,
        "paper_eq34_cost_proxy": 203964568515445.28,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 172.00495987063707,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.00017319306560060758,
        "problem_sha256": "6fdbad15c26fc1fd10f2d6df0e40101bc68a2e758ce734c9c9087b116c50e174",
        "qpe_fraction": 0.6520817370384794,
        "qpe_full_evolution_queries": 145115,
        "qpe_step_queries": 1338395645,
        "reaction_coefficient": 2.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 3,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 8582.640676473573,
        "rotation_fraction": 0.021877394442280984,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 5.810641205841267e-06,
        "scheduled_rotations": 8162355,
        "scheduled_rotations_per_step": 885,
        "species": "nh3",
        "step_repetitions": 9223,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00010842459069717012,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.00026560023960675263,
        "trotter_error_hartree": 8.659653280030379e-05,
        "trotter_fraction": 0.3260408685192397,
        "uniform_epsilon": 7.11882931560961e-13,
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
