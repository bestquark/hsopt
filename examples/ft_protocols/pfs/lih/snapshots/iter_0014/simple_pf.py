"""
Editable PF1 method file for the localized small-molecule precision benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Dict

import numpy as np


RELAXED_DIAGONAL_LABELS = {
    "ZIIIIIII",
    "IZIIIIII",
    "IIZIIZII",
    "IIZIIIIZ",
    "IIIZZIII",
    "IIIZIIZI",
}

MID_DIAGONAL_LABELS = {
    "IIZIIIII",
    "IIIZIIII",
    "IIIIIIIZ",
    "IIIIIIZI",
}

SPARSE_DIAGONAL_RULES = {
    "IIZIIIII": (256, "quinary_diagonal_epsilon"),
    "IIIZIIII": (256, "quinary_diagonal_epsilon"),
    "IIIIZIII": (256, "quinary_diagonal_epsilon"),
    "IIIIIZII": (256, "quinary_diagonal_epsilon"),
    "IIZZIIII": (256, "senary_diagonal_epsilon"),
    "IIZIIIIZ": (256, "quinary_diagonal_epsilon"),
    "IIZIIZII": (256, "quinary_diagonal_epsilon"),
    "IIIZIIZI": (256, "quinary_diagonal_epsilon"),
    "IIIZZIII": (256, "quinary_diagonal_epsilon"),
    "IIZIIIZI": (256, "quinary_diagonal_epsilon"),
    "IIZIZIII": (256, "quinary_diagonal_epsilon"),
    "IIIZIZII": (256, "quinary_diagonal_epsilon"),
    "IIIIIZZI": (256, "quinary_diagonal_epsilon"),
    "IZZIIIII": (256, "quinary_diagonal_epsilon"),
    "IZIZIIII": (256, "quinary_diagonal_epsilon"),
    "ZIIZIIII": (256, "quinary_diagonal_epsilon"),
    "ZZIIIIII": (256, "senary_diagonal_epsilon"),
    "ZIIIIIIZ": (256, "senary_diagonal_epsilon"),
    "ZIIIIIZI": (16, "quaternary_diagonal_epsilon"),
    "ZIIIIZII": (16, "quinary_diagonal_epsilon"),
    "ZIIIZIII": (32, "quaternary_diagonal_epsilon"),
    "IZIIIIIZ": (32, "quaternary_diagonal_epsilon"),
    "IZIIIZII": (32, "quaternary_diagonal_epsilon"),
    "IZIIIIZI": (64, "quinary_diagonal_epsilon"),
    "IZIIZIII": (64, "quinary_diagonal_epsilon"),
    "IIIIZIIZ": (64, "quinary_diagonal_epsilon"),
    "IIIZIIIZ": (256, "quinary_diagonal_epsilon"),
}

RELAXED_OFFDIAGONAL_LABELS = {
    "IIIIXXYY",
    "IIIIXYYX",
    "IIIIYXXY",
    "IIIIYYXX",
    "YYIIXXII",
    "YYIIIIXX",
    "YXIIXYII",
    "YXIIIIXY",
}

SECONDARY_OFFDIAGONAL_LABELS = {
    "YZZYXXII",
    "YZZYIIXX",
    "YZZXXYII",
    "YZZXIIXY",
    "XZZYYXII",
    "XZZYIIYX",
    "XZZXYYII",
    "XZZXIIYY",
}

GLOBAL_LABEL_EPSILON_OVERRIDES = {
    "ZIIIIIIZ": "tertiary_diagonal_epsilon",
    "XIXIIIII": "tertiary_offdiag_epsilon",
    "YIYIIIII": "quaternary_offdiag_epsilon",
    "XYIIIIYX": "quinary_offdiag_epsilon",
}

SPARSE_OFFDIAGONAL_RULES = {
    "XYIIYXII": (4, "senary_offdiag_epsilon"),
    "IXZXIIII": (32, "targeted_exchange_offdiag_epsilon"),
    "IYZYIIII": (32, "targeted_exchange_offdiag_epsilon"),
    "XZXIIIII": (32, "targeted_exchange_offdiag_epsilon"),
    "YZYIIIII": (32, "targeted_exchange_offdiag_epsilon"),
    "IXXIIIXX": (32, "relaxed_offdiag_epsilon"),
    "IXXIXXII": (32, "relaxed_offdiag_epsilon"),
    "IXYIIIYX": (32, "relaxed_offdiag_epsilon"),
    "IXYIYXII": (32, "relaxed_offdiag_epsilon"),
    "IYXIIIXY": (32, "relaxed_offdiag_epsilon"),
    "IYXIXYII": (32, "relaxed_offdiag_epsilon"),
    "IYYIIIYY": (32, "relaxed_offdiag_epsilon"),
    "IYYIYYII": (32, "relaxed_offdiag_epsilon"),
    "IXIXIIII": (32, "quaternary_offdiag_epsilon"),
    "IYIYIIII": (32, "quaternary_offdiag_epsilon"),
    "XZXZIIII": (32, "quaternary_offdiag_epsilon"),
    "YZYZIIII": (32, "quaternary_offdiag_epsilon"),
    "XXIIIIYY": (32, "targeted_exchange_offdiag_epsilon"),
    "XXIIYYII": (32, "targeted_exchange_offdiag_epsilon"),
}


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str = "lih_pf1_selected_diag_eps"
    n_steps: int = 256
    zero_angle_threshold: float = 1e-15
    diagonal_epsilon: float = 9.5e-8
    relaxed_diagonal_epsilon: float = 1.0e-7
    mid_diagonal_epsilon: float = 1.05e-7
    tertiary_diagonal_epsilon: float = 1.5e-7
    quaternary_diagonal_epsilon: float = 1.5e-7
    quinary_diagonal_epsilon: float = 1.5e-7
    senary_diagonal_epsilon: float = 2.0e-7
    offdiag_epsilon: float = 1.5e-7
    relaxed_offdiag_epsilon: float = 2.5e-7
    secondary_offdiag_epsilon: float = 2.5e-7
    tertiary_offdiag_epsilon: float = 5.0e-7
    quaternary_offdiag_epsilon: float = 4.0e-7
    quinary_offdiag_epsilon: float = 5.0e-7
    senary_offdiag_epsilon: float = 2.0e-7
    targeted_exchange_offdiag_epsilon: float = 1.75e-7


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
) -> float:
    _ = alpha
    label = str(term["label"])

    sparse_rule = SPARSE_DIAGONAL_RULES.get(label)
    if sparse_rule is not None and (step_index % sparse_rule[0] == 0):
        return getattr(config, sparse_rule[1])

    if label in RELAXED_DIAGONAL_LABELS:
        return config.relaxed_diagonal_epsilon
    if label in MID_DIAGONAL_LABELS:
        return config.mid_diagonal_epsilon
    if label in GLOBAL_LABEL_EPSILON_OVERRIDES:
        return getattr(config, GLOBAL_LABEL_EPSILON_OVERRIDES[label])

    if set(label) <= {"I", "Z"}:
        return config.diagonal_epsilon

    if label in RELAXED_OFFDIAGONAL_LABELS:
        return config.relaxed_offdiag_epsilon
    if label in SECONDARY_OFFDIAGONAL_LABELS:
        return config.secondary_offdiag_epsilon

    sparse_rule = SPARSE_OFFDIAGONAL_RULES.get(label)
    if sparse_rule is not None and (step_index % sparse_rule[0] == 0):
        return getattr(config, sparse_rule[1])

    return config.offdiag_epsilon


def apply_pauli_rotation(
    state: np.ndarray,
    pauli_matrix: np.ndarray,
    alpha: float,
    identity: np.ndarray,
) -> np.ndarray:
    return (math.cos(alpha) * identity + 1j * math.sin(alpha) * pauli_matrix) @ state


def run_product_formula(problem: Dict[str, object], exact_angles: bool = False) -> Dict[str, object]:
    config = get_method_config()
    state = np.array(problem["initial_state"], dtype=np.complex128, copy=True)
    identity = np.array(problem["identity"], dtype=np.complex128, copy=False)
    time_sign = float(problem["time_sign"])
    target_time = float(problem["target_time"])
    step_time = target_time / float(config.n_steps)
    compile_pauli_rotation = problem["compile_pauli_rotation"]

    epsilon_hist = Counter()
    unique_compilations = {}
    nonzero_rotation_count = 0
    skipped_rotation_count = 0
    t_gate_count = 0

    for step_index in range(config.n_steps):
        for term in problem["pauli_terms"]:
            coefficient = float(term["coefficient"])
            alpha = time_sign * coefficient * step_time

            if exact_angles:
                if abs(alpha) <= config.zero_angle_threshold:
                    skipped_rotation_count += 1
                    continue
                state = apply_pauli_rotation(state, term["matrix"], alpha, identity)
                nonzero_rotation_count += 1
            else:
                epsilon = float(choose_synthesis_epsilon(term, alpha, step_index, config))
                compiled = compile_pauli_rotation(term["label"], alpha, epsilon)
                state = compiled["unitary"] @ state
                nonzero_rotation_count += 1
                epsilon_key = f"{epsilon:.3e}"
                epsilon_hist[epsilon_key] += 1
                t_gate_count += int(compiled["t_gate_count"])
                unique_compilations[compiled["cache_key"]] = {
                    "label": compiled["label"],
                    "alpha": compiled["alpha"],
                    "theta": compiled["theta"],
                    "epsilon": compiled["epsilon"],
                    "t_gate_count": compiled["t_gate_count"],
                    "gate_count": compiled["gate_count"],
                    "operator_error": compiled["operator_error"],
                }

    state = state / np.linalg.norm(state)

    return {
        "state": state,
        "method_config": asdict(config),
        "nonzero_rotation_count": nonzero_rotation_count,
        "skipped_rotation_count": skipped_rotation_count,
        "synthesis_epsilon_histogram": dict(sorted(epsilon_hist.items())),
        "unique_compilation_count": len(unique_compilations),
        "unique_compilations": sorted(unique_compilations.values(), key=lambda item: (item["label"], item["epsilon"])),
        "t_gate_count": t_gate_count,
        "step_time": step_time,
    }
