"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 20fc0c78b38b369e5ee92227a62b12943f7519ca54d4773a2157f2c1034d79d1
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: 563706a26f236bbfb0dee688cd787d167b7f2d6c6af9b2839891e8f8536a955c
Calibration scientific SHA-256: da92a48a76639474361c4dd80e3b44e1ab66063cc9491feb5e987e59aa0ee96b
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
    "Reaction seed hcn": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 435,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "563706a26f236bbfb0dee688cd787d167b7f2d6c6af9b2839891e8f8536a955c",
        "double_commutator_sum_upper": 5409.810876961251,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 4566,
        "nonidentity_terms": 435,
        "paper_eq34_cost_proxy": 31621731658110.426,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 164.4146510309886,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005190549595548789,
        "problem_sha256": "8e4e93f38a2132f2b2bee7372f5f37a5d7aa175335e5a27a7e3087a7edc89f75",
        "qpe_fraction": 0.6514238093602515,
        "qpe_full_evolution_queries": 48421,
        "qpe_step_queries": 221090286,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 7355.1416553056615,
        "rotation_fraction": 0.022864285959622855,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.821827948793942e-05,
        "scheduled_rotations": 3967854,
        "scheduled_rotations_per_step": 869,
        "species": "hcn",
        "step_repetitions": 4566,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0002190100744634253,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00025952747977743947,
        "trotter_fraction": 0.32571190468012573,
        "uniform_epsilon": 4.591469214325784e-12,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed hnc": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 435,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "da92a48a76639474361c4dd80e3b44e1ab66063cc9491feb5e987e59aa0ee96b",
        "double_commutator_sum_upper": 4830.506561885685,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202577,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 4315,
        "nonidentity_terms": 435,
        "paper_eq34_cost_proxy": 29821316227985.71,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 164.07622520371754,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005190304819820993,
        "problem_sha256": "6d775f158107f452007b1a7ea688a2b8acc0515d0026cd6b1fac91b407190cb6",
        "qpe_fraction": 0.6513930895426088,
        "qpe_full_evolution_queries": 48423,
        "qpe_step_queries": 208945245,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 6950.18457444526,
        "rotation_fraction": 0.022910365686086728,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.8254995847108875e-05,
        "scheduled_rotations": 3749735,
        "scheduled_rotations_per_step": 869,
        "species": "hnc",
        "step_repetitions": 4315,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0002317497103128621,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00025951524099104964,
        "trotter_fraction": 0.3256965447713044,
        "uniform_epsilon": 4.868342922128864e-12,
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
