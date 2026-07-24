"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: hamiltonian.json
Source SHA-256: e93b0191424d0204624e97834666e8bb3cbb9eae87adc17ac4c735575c5e6cea
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: 68cf860674e7659cff15891e2173cdbe29d39fb5dca7e59b259e5a1d2a08f3b8
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
    "HamLib 0291_hydrogen_data_H6_ES_H6_linear_R0.8_sto-6g_ham_JW": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 918,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "68cf860674e7659cff15891e2173cdbe29d39fb5dca7e59b259e5a1d2a08f3b8",
        "double_commutator_sum_upper": 1431.9749994886756,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0015936014376405159,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 1662,
        "nonidentity_terms": 918,
        "paper_eq34_cost_proxy": 11878293573822.129,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 160.76279768701787,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.00103757102523009,
        "problem_sha256": "e93b0191424d0204624e97834666e8bb3cbb9eae87adc17ac4c735575c5e6cea",
        "qpe_fraction": 0.6510856483452453,
        "qpe_full_evolution_queries": 24223,
        "qpe_step_queries": 40258626,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 3784.144552588703,
        "rotation_fraction": 0.02337152748213222,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 3.7244899795380726e-05,
        "scheduled_rotations": 3049770,
        "scheduled_rotations_per_step": 1835,
        "step_repetitions": 1662,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.0006016847172081829,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0015936014376405157,
        "trotter_error_hartree": 0.000518785512615045,
        "trotter_fraction": 0.32554282417262265,
        "uniform_epsilon": 1.2212363488191152e-11,
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
