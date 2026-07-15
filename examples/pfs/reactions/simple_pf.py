"""Minimal stationary synthesis policy for the ten-reaction benchmark.

Autoresearch copies this file into a lane and mutates only the copy.  The
scientific evaluator and reaction registry stay fixed.

Adapted from archived policy source SHA-256
``1311648c568e25b460e7555c8f7200d320dbcdab091749f38b07b0e24269cb0f``.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Dict


HARTREE_PER_KCAL_PER_MOL = 1.0 / 627.5094740631
DEFAULT_SYNTHESIS_ERROR_BUDGET_HARTREE = 0.2 * HARTREE_PER_KCAL_PER_MOL
_ACTIVE_SPECIES = ""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default) if raw is None or not raw.strip() else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default) if raw is None or not raw.strip() else float(raw)


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str
    species: str
    formula_order: int
    n_steps: int
    zero_angle_threshold: float
    synthesis_error_budget_hartree: float


def set_active_species(species: str) -> None:
    global _ACTIVE_SPECIES
    if not str(species).strip():
        raise ValueError("species must be nonempty")
    _ACTIVE_SPECIES = str(species)


def get_method_config() -> ProductFormulaConfig:
    if not _ACTIVE_SPECIES:
        raise RuntimeError("set_active_species() must be called first")
    formula_order = _env_int("HSOPT_REACTIONS_FORMULA_ORDER", 1)
    if formula_order not in {1, 2}:
        raise ValueError(f"formula order must be 1 or 2, got {formula_order}")
    n_steps = _env_int("HSOPT_REACTIONS_PF_N_STEPS", 512)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    return ProductFormulaConfig(
        name=f"reaction_{_ACTIVE_SPECIES}_pf{formula_order}_uniform",
        species=_ACTIVE_SPECIES,
        formula_order=formula_order,
        n_steps=n_steps,
        zero_angle_threshold=_env_float(
            "HSOPT_REACTIONS_ZERO_ANGLE_THRESHOLD",
            1.0e-15,
        ),
        synthesis_error_budget_hartree=_env_float(
            "HSOPT_REACTIONS_SYNTHESIS_ERROR_BUDGET_HARTREE",
            DEFAULT_SYNTHESIS_ERROR_BUDGET_HARTREE,
        ),
    )


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
    n_terms: int,
) -> float:
    _ = (term, alpha, step_index)
    rotations_per_step = n_terms if config.formula_order == 1 else 2 * n_terms - 1
    scheduled_rotations = max(1, config.n_steps * rotations_per_step)
    epsilon = config.synthesis_error_budget_hartree / float(scheduled_rotations)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"invalid synthesis epsilon {epsilon}")
    return max(epsilon, 1.0e-15)


def describe_policy() -> Dict[str, object]:
    return asdict(get_method_config())
