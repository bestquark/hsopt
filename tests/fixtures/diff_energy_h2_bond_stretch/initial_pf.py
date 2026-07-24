"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 67fc570923800dc9f400e1cd65b09330e3d28a20eacd6941a9754df5ca438007
Paper contract SHA-256: 857c98ae69f93598087e5131c4c46b3a775aa6a69be128e32b7e9b4127e3bb39
Calibration scientific SHA-256: 96636fada0df16a6e25b9f809b553539df29b047f009198ef678629e3da13c1b
Calibration scientific SHA-256: 3463f7292f4b09ebbf3d05a6f79edb2b517964cbf60d176c330365e16e71c5b8
Generator: experiments/generate_initial_pf.py
Paper: Reiher et al., PNAS 114, 7555-7560 (2017), DOI 10.1073/pnas.1619152114

Paper structure: PF2/Strang, lexicographic order, Eq. (19) bound beta(H),
the paper's Trotter/QPE/synthesis error relation, and uniform Clifford+T
rotation tolerance.

Benchmark adaptation: beta is referenced to 0.1 mHa; the full error target is
1 kcal/mol; allocation cost uses direct PyGridSynth T counts instead of the
paper's gamma/delta synthesis fit. Reaction initializers conservatively give
each signed species term an equal absolute share of the reaction-error target.

beta(H) is obtained from the paper's rigorous Eq. (19) double-commutator coefficient bound for this Pauli decomposition. The separately cached exact energy is used only by the evaluator and does not enter beta, the Trotter step count, or the synthesis tolerance.

This file is measured unchanged as iteration 0. Scientific acceptance is based
exclusively on exact diagonalization of the compiled PF unitary against the
separately cached U_exact reference.
"""

from dataclasses import dataclass


BASELINES = {
    "HamLib H2 linear R=0.7 STO-6G Jordan-Wigner": {
        "allocation_candidates_evaluated": 33,
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_bound_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 14,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "96636fada0df16a6e25b9f809b553539df29b047f009198ef678629e3da13c1b",
        "double_commutator_sum_upper": 0.41861518626301725,
        "error_allocation": "direct_pygridsynth_t_count_optimization_under_paper_variance_constraint",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "full_pf_t_count": 78036,
        "n_steps": 42,
        "nonidentity_terms": 14,
        "one_step_gate_count": 4787,
        "one_step_t_count": 1858,
        "phase_estimation_error_hartree": 0.0005111959450904679,
        "problem_sha256": "7e741dc5e8e4d37c3cb8e17dc6bfe970da921bcd0242e2d50940161f1f312d4a",
        "qpe_full_evolution_queries": 3073,
        "qpe_step_queries": 129066,
        "qpe_t_count_proxy": 239804628,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 64.7004780711099,
        "rotation_radial_fraction": 0.4,
        "rotation_synthesis": "clifford_plus_t_pygridsynth",
        "rotation_synthesis_energy_error_hartree": 0.00022310420126967222,
        "scheduled_rotations": 1134,
        "scheduled_rotations_per_step": 27,
        "species": "h2_r0_7",
        "step_repetitions": 42,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.023809523809523808,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00023904021564607734,
        "trotter_fraction": 0.3,
        "uniform_epsilon": 1.9674091822722415e-07,
        "zero_angle_threshold": 0.0,
    },
    "HamLib H2 linear R=0.8 STO-6G Jordan-Wigner": {
        "allocation_candidates_evaluated": 33,
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_eq19_bound_pf2",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_term_count": 14,
        "calibration_mode": "hamiltonian_bound",
        "calibration_sha256": "3463f7292f4b09ebbf3d05a6f79edb2b517964cbf60d176c330365e16e71c5b8",
        "double_commutator_sum_upper": 0.3302852845790427,
        "error_allocation": "direct_pygridsynth_t_count_optimization_under_paper_variance_constraint",
        "error_constraint_value_hartree": 0.0007968007188202577,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "full_pf_t_count": 66430,
        "n_steps": 35,
        "nonidentity_terms": 14,
        "one_step_gate_count": 4962,
        "one_step_t_count": 1898,
        "phase_estimation_error_hartree": 0.0004940646369098993,
        "problem_sha256": "35cd94cfa48d4069a68d1720bbac306e5a91add8e800fc60df4b79083f0ca213",
        "qpe_full_evolution_queries": 3180,
        "qpe_step_queries": 111300,
        "qpe_t_count_proxy": 211247400,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 57.470451936542375,
        "rotation_radial_fraction": 0.3,
        "rotation_synthesis": "clifford_plus_t_pygridsynth",
        "rotation_synthesis_energy_error_hartree": 0.00015537614016995026,
        "scheduled_rotations": 945,
        "scheduled_rotations_per_step": 27,
        "species": "h2_r0_8",
        "step_repetitions": 35,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.02857142857142857,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.0002788802515870902,
        "trotter_fraction": 0.35,
        "uniform_epsilon": 1.6441919594703731e-07,
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
