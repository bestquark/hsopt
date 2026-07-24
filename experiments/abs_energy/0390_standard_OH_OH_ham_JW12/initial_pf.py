"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: hamiltonian.json
Source SHA-256: 0ce1710c6c3c5c8183a010059ddcfe92e77d6d426d554d4d32b752013e307800
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: a983afd6e5c4c1ea393cd5a69f6ad47a96d90d3d9bac054a8227b1148d2d7594
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
    "HamLib 0390_standard_OH_OH_ham_JW12": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 630,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "a983afd6e5c4c1ea393cd5a69f6ad47a96d90d3d9bac054a8227b1148d2d7594",
        "double_commutator_sum_upper": 929.1636779471336,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0015936014376405159,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 1339,
        "nonidentity_terms": 630,
        "paper_eq34_cost_proxy": 6426675094251.83,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 157.2165443614503,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0010370243271463468,
        "problem_sha256": "0ce1710c6c3c5c8183a010059ddcfe92e77d6d426d554d4d32b752013e307800",
        "qpe_fraction": 0.6507425901182442,
        "qpe_full_evolution_queries": 24236,
        "qpe_step_queries": 32452004,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 3048.218623962418,
        "rotation_fraction": 0.02388611482263369,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 3.806494692099548e-05,
        "scheduled_rotations": 1685801,
        "scheduled_rotations_per_step": 1259,
        "step_repetitions": 1339,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0007468259895444362,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0015936014376405157,
        "trotter_error_hartree": 0.0005185121635731734,
        "trotter_fraction": 0.3253712950591221,
        "uniform_epsilon": 2.2579739198752097e-11,
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
