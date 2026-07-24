"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: reaction.json
Source SHA-256: 67fc570923800dc9f400e1cd65b09330e3d28a20eacd6941a9754df5ca438007
Paper contract SHA-256: a5be559277d03a9e7876978ad18dc169998678b3ac0a228a4daf0e052f160a31
Calibration scientific SHA-256: a483c26af8dbbc4e2ed6b7ba5d1561b97c7b8f2db8b0b992b8f3e7d80fc7cac3
Calibration scientific SHA-256: 5f0d4c5bce021b9c6b4fd79811286d813b55fabaa82df04c0c0569f0281f1e46
Generator: experiments/generate_initial_pf.py
Paper: Reiher et al., PNAS 114, 7555-7560 (2017), DOI 10.1073/pnas.1619152114

Paper structure: PF2/Strang, lexicographic order, molecule-specific beta(H),
the paper's Trotter/QPE/synthesis error relation, and uniform Clifford+T
rotation tolerance.

Benchmark adaptation: beta is referenced to 0.1 mHa; the full error target is
1 kcal/mol; allocation cost uses direct PyGridSynth T counts instead of the
paper's gamma/delta synthesis fit. Reaction initializers conservatively give
each signed species term an equal absolute share of the reaction-error target.

The default beta(H) is an empirical exact-energy calibration, as used for the paper's tractable small-molecule observations. It is an evaluation-oracle baseline, not a scalable unknown-energy procedure. The optional Hamiltonian-bound calibration carries a different mode and provenance in the generated policy.

This file is measured unchanged as iteration 0. Scientific acceptance is based
exclusively on exact diagonalization of the compiled PF unitary against the
separately cached U_exact reference.
"""

from dataclasses import dataclass


BASELINES = {
    "HamLib H2 linear R=0.7 STO-6G Jordan-Wigner": {
        "allocation_candidates_evaluated": 33,
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_empirical_pf2",
        "calibration_mode": "empirical_exact",
        "calibration_sha256": "a483c26af8dbbc4e2ed6b7ba5d1561b97c7b8f2db8b0b992b8f3e7d80fc7cac3",
        "error_allocation": "direct_pygridsynth_t_count_optimization_under_paper_variance_constraint",
        "error_constraint_value_hartree": 0.0007968007188202577,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "full_pf_t_count": 4806,
        "n_steps": 3,
        "nonidentity_terms": 14,
        "one_step_gate_count": 4257,
        "one_step_t_count": 1602,
        "phase_estimation_error_hartree": 0.0004940646369098993,
        "problem_sha256": "7e741dc5e8e4d37c3cb8e17dc6bfe970da921bcd0242e2d50940161f1f312d4a",
        "qpe_full_evolution_queries": 3180,
        "qpe_step_queries": 9540,
        "qpe_t_count_proxy": 15283080,
        "reaction_coefficient": -1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5.0,
        "rotation_radial_fraction": 0.3,
        "rotation_synthesis": "clifford_plus_t_pygridsynth",
        "rotation_synthesis_energy_error_hartree": 0.00015537614016995026,
        "scheduled_rotations": 81,
        "scheduled_rotations_per_step": 27,
        "species": "h2_r0_7",
        "step_repetitions": 3,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.3333333333333333,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.0002788802515870902,
        "trotter_fraction": 0.35,
        "uniform_epsilon": 1.918223952715435e-06,
        "zero_angle_threshold": 0.0,
    },
    "HamLib H2 linear R=0.8 STO-6G Jordan-Wigner": {
        "allocation_candidates_evaluated": 33,
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_empirical_pf2",
        "calibration_mode": "empirical_exact",
        "calibration_sha256": "5f0d4c5bce021b9c6b4fd79811286d813b55fabaa82df04c0c0569f0281f1e46",
        "error_allocation": "direct_pygridsynth_t_count_optimization_under_paper_variance_constraint",
        "error_constraint_value_hartree": 0.0007968007188202578,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": 2,
        "full_pf_t_count": 6576,
        "n_steps": 4,
        "nonidentity_terms": 14,
        "one_step_gate_count": 4325,
        "one_step_t_count": 1644,
        "phase_estimation_error_hartree": 0.0006171991828473914,
        "problem_sha256": "35cd94cfa48d4069a68d1720bbac306e5a91add8e800fc60df4b79083f0ca213",
        "qpe_full_evolution_queries": 2546,
        "qpe_step_queries": 10184,
        "qpe_t_count_proxy": 16742496,
        "reaction_coefficient": 1.0,
        "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
        "reaction_species_count": 2,
        "reaction_total_error_target_hartree": 0.0015936014376405157,
        "reference_beta_error_hartree": 0.0001,
        "reference_error_hartree": 0.0001,
        "reference_trotter_number_beta": 5.0,
        "rotation_radial_fraction": 0.25,
        "rotation_synthesis": "clifford_plus_t_pygridsynth",
        "rotation_synthesis_energy_error_hartree": 0.00015936014376405157,
        "scheduled_rotations": 108,
        "scheduled_rotations_per_step": 27,
        "species": "h2_r0_8",
        "step_repetitions": 4,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": 0.25,
        "target_time": 1.0,
        "term_order_strategy": "lexicographic",
        "total_error_target_hartree": 0.0007968007188202578,
        "trotter_error_hartree": 0.00015936014376405157,
        "trotter_fraction": 0.2,
        "uniform_epsilon": 1.4755568867041813e-06,
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
