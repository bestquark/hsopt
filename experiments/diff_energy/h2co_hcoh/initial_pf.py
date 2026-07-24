"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 0ebd4c2de5a2883c93b6fd05ec6a3a6f1b23e9e55d7f231dc33f64fa45925a13
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: fdf572d2c032412d3195510e728e8009d371d771e49a5ba4d2421951af341a25
Calibration scientific SHA-256: 67c5587ad29220ea73cd9799ecd758d7470d2c0229f2909d00ca9d6d1fd8f6ff
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
    "Reaction seed h2co": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 251,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "fdf572d2c032412d3195510e728e8009d371d771e49a5ba4d2421951af341a25",
        "double_commutator_sum_upper": 4316.436278090629,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202579,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 4080,
        "nonidentity_terms": 251,
        "paper_eq34_cost_proxy": 15918971668681.281,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 160.45234540819007,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005187620539075195,
        "problem_sha256": "ba8f96e09e8fa71b9e7d8d7a1c3af93820ac3535f80152a24e3b51070bcf077d",
        "qpe_fraction": 0.6510562072228021,
        "qpe_full_evolution_queries": 48448,
        "qpe_step_queries": 197667840,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 6569.959115619083,
        "rotation_fraction": 0.023415689165796828,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.8657637958978634e-05,
        "scheduled_rotations": 2044080,
        "scheduled_rotations_per_step": 501,
        "species": "h2co",
        "step_repetitions": 4080,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00024509803921568627,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00025938102695375974,
        "trotter_fraction": 0.32552810361140105,
        "uniform_epsilon": 9.127645668945753e-12,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed hcoh": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 443,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "67c5587ad29220ea73cd9799ecd758d7470d2c0229f2909d00ca9d6d1fd8f6ff",
        "double_commutator_sum_upper": 3118.4729361196037,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202577,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 3467,
        "nonidentity_terms": 443,
        "paper_eq34_cost_proxy": 24229300446773.254,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 162.87737394400153,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005189429722807776,
        "problem_sha256": "244c96ebd7e3eb691afa41587146ec9e9194cd0a4ac0b80a9b642e941af8d387",
        "qpe_fraction": 0.6512832632093052,
        "qpe_full_evolution_queries": 48431,
        "qpe_step_queries": 167910277,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5584.328908758512,
        "rotation_fraction": 0.02307510518604206,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.838626039909137e-05,
        "scheduled_rotations": 3068295,
        "scheduled_rotations_per_step": 885,
        "species": "hcoh",
        "step_repetitions": 3467,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00028843380444188056,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.0002594714861403888,
        "trotter_fraction": 0.3256416316046526,
        "uniform_epsilon": 5.992337894202276e-12,
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
