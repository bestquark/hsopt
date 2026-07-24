"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: hamiltonian.json
Source SHA-256: 7e741dc5e8e4d37c3cb8e17dc6bfe970da921bcd0242e2d50940161f1f312d4a
Paper contract SHA-256: 857c98ae69f93598087e5131c4c46b3a775aa6a69be128e32b7e9b4127e3bb39
Calibration scientific SHA-256: 96636fada0df16a6e25b9f809b553539df29b047f009198ef678629e3da13c1b
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
        "error_constraint_value_hartree": 0.0015936014376405157,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "full_pf_t_count": 53220,
        "n_steps": 30,
        "nonidentity_terms": 14,
        "one_step_gate_count": 4551,
        "one_step_t_count": 1774,
        "phase_estimation_error_hartree": 0.001064139217959783,
        "problem_sha256": "7e741dc5e8e4d37c3cb8e17dc6bfe970da921bcd0242e2d50940161f1f312d4a",
        "qpe_full_evolution_queries": 1477,
        "qpe_step_queries": 44310,
        "qpe_t_count_proxy": 78605940,
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 64.7004780711099,
        "rotation_radial_fraction": 0.3,
        "rotation_synthesis": "clifford_plus_t_pygridsynth",
        "rotation_synthesis_energy_error_hartree": 0.0003346563019045083,
        "scheduled_rotations": 810,
        "scheduled_rotations_per_step": 27,
        "step_repetitions": 30,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.03333333333333333,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0015936014376405157,
        "trotter_error_hartree": 0.0004780804312921547,
        "trotter_fraction": 0.3,
        "uniform_epsilon": 4.1315592827717075e-07,
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
