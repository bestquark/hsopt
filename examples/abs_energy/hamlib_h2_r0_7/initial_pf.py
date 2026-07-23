"""Generated standard PF initializer. Do not edit by hand.

Source: hamiltonian.json
Source SHA-256: 9b5e4fbd215ad3492c7a3b626792894629e399cbad30d90e72237ea8e2f5c7d1
Paper contract: reiher_wiebe_svore_2017.json
Paper contract SHA-256: 8300ffe6936a1e5c531f82717ba6d6d083261b157e0f1a9bf8b109aa15963578
Generator: examples/generate_initial_pf.py
Paper: Reiher et al., PNAS 114, 7555-7560 (2017), DOI 10.1073/pnas.1619152114

This file is measured unchanged as iteration 0. Scientific acceptance is based
exclusively on exact diagonalization of the compiled PF unitary against the
separately cached U_exact reference.
"""

from dataclasses import dataclass


BASELINES = {
    "HamLib H2 linear R=0.7 STO-6G Jordan-Wigner": {
        "baseline_source": "reiher_wiebe_svore_wecker_troyer_2017_pf2",
        "formula_order": 2,
        "n_steps": 512,
        "nonidentity_terms": 14,
        "scheduled_rotations": 13824,
        "synthesis_error_budget_hartree": 0.0005312004792135052,
        "term_order_strategy": "lexicographic",
        "uniform_epsilon": 3.8425960591254716e-08,
    }
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
        zero_angle_threshold=0.0,
        uniform_epsilon=float(row["uniform_epsilon"]),
        term_order_strategy=str(row["term_order_strategy"]),
        baseline_source=str(row["baseline_source"]),
    )


def choose_synthesis_epsilon(term, alpha, step_index, config) -> float:
    _ = (term, alpha, step_index)
    return config.uniform_epsilon
