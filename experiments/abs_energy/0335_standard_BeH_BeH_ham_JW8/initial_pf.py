"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: hamiltonian.json
Source SHA-256: e7b7df6d977d2c3c97a28cb62ceda0ab140a39924de763193be55e8f979c0379
Paper contract SHA-256: 9d74067aa7507dde7c4af016beb1b54288ba97de9927b6cff4f00e2af63e6a4f
Calibration scientific SHA-256: a60ff9d17a8a7bc46a703112771feedd1e18e9a7f3735fabc19cba8200671433
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
    "HamLib 0335_standard_BeH_BeH_ham_JW8": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 104,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "a60ff9d17a8a7bc46a703112771feedd1e18e9a7f3735fabc19cba8200671433",
        "double_commutator_sum_upper": 1.6384642241242175,
        "error_allocation": "supplement_eq34_rigorous_additive_analytic_optimum",
        "error_constraint_value_hartree": 0.0015936014376405157,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "n_steps": 57,
        "nonidentity_terms": 104,
        "paper_eq34_cost_proxy": 36388775150.17571,
        "paper_error_constraint": "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total",
        "paper_modeled_t_per_rotation": 127.34268013086788,
        "paper_phase_estimation_alpha": 25.132741228718345,
        "paper_synthesis_delta_0": 11.0,
        "paper_synthesis_gamma": 4.0,
        "phase_estimation_error_hartree": 0.0010312456945782687,
        "problem_sha256": "e7b7df6d977d2c3c97a28cb62ceda0ab140a39924de763193be55e8f979c0379",
        "qpe_fraction": 0.6471164434346456,
        "qpe_full_evolution_queries": 24372,
        "qpe_step_queries": 1389204,
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 128.00250873026738,
        "rotation_fraction": 0.029325334848031574,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "rotation_synthesis_energy_error_hartree": 4.673289577311263e-05,
        "scheduled_rotations": 11799,
        "scheduled_rotations_per_step": 207,
        "step_repetitions": 57,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.017543859649122806,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0015936014376405157,
        "trotter_error_hartree": 0.0005156228472891343,
        "trotter_fraction": 0.3235582217173228,
        "uniform_epsilon": 3.960750552853006e-09,
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
