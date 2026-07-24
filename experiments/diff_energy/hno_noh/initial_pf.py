"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 03a85698dc830e54b7149d29c8321b1c304e3aff33e01274083a4555214aab21
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: d3aca45afe5b37eccacc36e48523f792da6ce38dc7d3d589c680512da887f618
Calibration scientific SHA-256: 0ed58599d72483a0eddfe0c2c46f71ae00c146aa1a553ac361a609bf995f7338
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
    "Reaction seed hno": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 443,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "d3aca45afe5b37eccacc36e48523f792da6ce38dc7d3d589c680512da887f618",
        "double_commutator_sum_upper": 4985.246263696364,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 4383,
        "nonidentity_terms": 443,
        "paper_eq34_cost_proxy": 30889231670096.51,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 164.27934714332127,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0005190451851546742,
        "problem_sha256": "aa44d183414ea97621a72985f797ae74cd5b11ff3dd84853d8dab6772e240054",
        "qpe_fraction": 0.651411542302788,
        "qpe_full_evolution_queries": 48422,
        "qpe_step_queries": 212233626,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 7060.627637608687,
        "rotation_fraction": 0.02288268654581805,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.8232941088246466e-05,
        "scheduled_rotations": 3878955,
        "scheduled_rotations_per_step": 885,
        "species": "hno",
        "step_repetitions": 4383,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.00022815423226100844,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.0002595225925773371,
        "trotter_fraction": 0.325705771151394,
        "uniform_epsilon": 4.700477599829455e-12,
        "zero_angle_threshold": 0.0,
    },
    "Reaction seed noh": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 443,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "0ed58599d72483a0eddfe0c2c46f71ae00c146aa1a553ac361a609bf995f7338",
        "double_commutator_sum_upper": 3081.0425305143017,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 3446,
        "nonidentity_terms": 443,
        "paper_eq34_cost_proxy": 24078299873136.934,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 162.8412822988892,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.00051894031826642,
        "problem_sha256": "6079a7775578d714e5cb577b6307a330cdd73c515ae24d313422784c0f2d5c87",
        "qpe_fraction": 0.6512799323709979,
        "qpe_full_evolution_queries": 48431,
        "qpe_step_queries": 166893226,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5550.713945533765,
        "rotation_fraction": 0.02308010144350314,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 1.839024142062777e-05,
        "scheduled_rotations": 3049710,
        "scheduled_rotations_per_step": 885,
        "species": "noh",
        "step_repetitions": 3446,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0002901915264074289,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00025947015913321,
        "trotter_fraction": 0.32563996618549895,
        "uniform_epsilon": 6.0301607105684705e-12,
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
