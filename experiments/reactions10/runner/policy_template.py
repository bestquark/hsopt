"""Editable mixed-precision product-formula policy for __LANE_ID__."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


LANE_ID = "__LANE_ID__"
_ACTIVE_SPECIES = ""

# Each species starts from the independently validated uniform-grid winner.
# Autoresearch may change these values and add deterministic, term-aware logic
# below. Every policy decision must remain in this single file.
SPECIES_POLICIES = __SPECIES_POLICIES__  # noqa: F821


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str
    species: str
    formula_order: int
    n_steps: int
    zero_angle_threshold: float
    synthesis_error_budget_hartree: float
    schedule_strategy: str = "stationary_trotter"
    term_order_strategy: str = "input"
    qdrift_samples: int = 4096


def set_active_species(species: str) -> None:
    global _ACTIVE_SPECIES
    if species not in SPECIES_POLICIES:
        raise KeyError(f"unsupported species {species!r}")
    _ACTIVE_SPECIES = species


def get_method_config() -> ProductFormulaConfig:
    if not _ACTIVE_SPECIES:
        raise RuntimeError("set_active_species() must be called before get_method_config()")
    policy = SPECIES_POLICIES[_ACTIVE_SPECIES]
    return ProductFormulaConfig(
        name=f"{LANE_ID}_{_ACTIVE_SPECIES}_mixed_precision",
        species=_ACTIVE_SPECIES,
        formula_order=int(policy["formula_order"]),
        n_steps=int(policy["n_steps"]),
        zero_angle_threshold=float(policy.get("zero_angle_threshold", 1.0e-15)),
        synthesis_error_budget_hartree=float(policy["synthesis_error_budget_hartree"]),
        schedule_strategy=str(policy.get("schedule_strategy", "stationary_trotter")),
        term_order_strategy=str(policy.get("term_order_strategy", "input")),
        qdrift_samples=int(policy.get("qdrift_samples", 4096)),
    )


def choose_synthesis_epsilon(
    term: dict[str, object],
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


def order_product_formula_terms(
    terms: list[dict[str, object]],
    config: ProductFormulaConfig,
) -> list[dict[str, object]]:
    """Return any exact permutation when ``term_order_strategy == 'custom'``."""

    _ = config
    return list(terms)


def build_product_formula_schedule(
    terms: list[dict[str, object]],
    target_time: float,
    evaluator_seed: int,
    config: ProductFormulaConfig,
) -> list[dict[str, object]]:
    """Return explicit term_index/alpha[/epsilon] items for a custom method."""

    _ = (terms, target_time, evaluator_seed, config)
    raise NotImplementedError("define a deterministic schedule before selecting 'custom'")


def describe_policy() -> dict[str, Any]:
    return asdict(get_method_config())
