"""Rotation-angle and synthesis-tolerance diagnostics for evaluated policies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compiled_unitary import _product_formula_schedule, load_policy, load_problem


@dataclass(frozen=True)
class RotationSamples:
    """One product-formula policy represented by its physical Rz rotations."""

    angles: tuple[float, ...]
    tolerances: tuple[float, ...]
    weights: tuple[float, ...]
    scheduled_rotations: float
    components: tuple[dict[str, Any], ...]


def _source_problems(source_path: Path) -> list[tuple[str, dict[str, Any], float]]:
    source_path = source_path.expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("n_qubits"), int):
        return [(str(payload.get("name") or source_path.stem), load_problem(source_path), 1.0)]

    terms = payload.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError(f"{source_path} is neither a Hamiltonian nor a reaction definition")
    problems = []
    for index, term in enumerate(terms):
        if not isinstance(term, dict) or not isinstance(term.get("problem"), str):
            raise ValueError(f"{source_path}: reaction term {index} has no problem path")
        problem_path = (source_path.parent / str(term["problem"])).resolve()
        if not problem_path.is_relative_to(source_path.parent):
            raise ValueError(f"{source_path}: reaction term {index} escapes its input directory")
        weight = abs(float(term.get("coefficient", 1.0)))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"{source_path}: reaction term {index} has an invalid coefficient")
        problems.append(
            (
                str(term.get("species") or problem_path.stem),
                load_problem(problem_path),
                weight,
            )
        )
    return problems


def rotation_samples(policy_path: Path, source_path: Path) -> RotationSamples:
    """Extract the exact stationary schedule used by the fixed evaluator.

    The evaluator's restricted policy loader and schedule validator are reused,
    so this diagnostic cannot silently interpret a policy differently from a
    scored candidate. Weights include product-formula repetitions and absolute
    reaction stoichiometry, but deliberately exclude QPE query repetitions.
    """

    policy_path = policy_path.expanduser().resolve()
    policy = load_policy(policy_path)
    angles: list[float] = []
    tolerances: list[float] = []
    weights: list[float] = []
    components: list[dict[str, Any]] = []
    scheduled_rotations = 0.0
    for label, problem, stoichiometric_weight in _source_problems(source_path):
        product_formula, rotations, metrics = _product_formula_schedule(problem, policy)
        n_steps = int(metrics["n_steps"])
        rotation_weight = stoichiometric_weight * n_steps
        for term, alpha in rotations:
            theta = abs(float(round(-2.0 * alpha, 15)))
            epsilon = float(product_formula.synthesis_epsilon(dict(term), alpha))
            if not math.isfinite(theta) or theta <= 0.0:
                continue
            if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
                raise ValueError(
                    f"{policy_path}: synthesis epsilon must lie in (0, 1), found {epsilon}"
                )
            angles.append(theta)
            tolerances.append(epsilon)
            weights.append(rotation_weight)
        component_rotations = float(len(rotations)) * rotation_weight
        scheduled_rotations += component_rotations
        components.append(
            {
                "label": label,
                "n_qubits": int(problem["n_qubits"]),
                "n_steps": n_steps,
                "rotations_per_step": len(rotations),
                "weighted_scheduled_rotations": component_rotations,
            }
        )
    if not angles:
        raise ValueError(f"{policy_path}: policy produced no nonzero rotations")
    return RotationSamples(
        angles=tuple(angles),
        tolerances=tuple(tolerances),
        weights=tuple(weights),
        scheduled_rotations=scheduled_rotations,
        components=tuple(components),
    )


def _common_histograms(
    initial_values: tuple[float, ...],
    initial_weights: tuple[float, ...],
    current_values: tuple[float, ...],
    current_weights: tuple[float, ...],
    *,
    bins: int,
) -> dict[str, list[float]]:
    if bins < 8:
        raise ValueError("at least eight histogram bins are required")
    initial_log = np.log10(np.asarray(initial_values, dtype=float))
    current_log = np.log10(np.asarray(current_values, dtype=float))
    minimum = float(min(np.min(initial_log), np.min(current_log)))
    maximum = float(max(np.max(initial_log), np.max(current_log)))
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1.0e-12):
        minimum -= 0.5
        maximum += 0.5
    else:
        padding = 0.04 * (maximum - minimum)
        minimum -= padding
        maximum += padding
    edges = np.linspace(minimum, maximum, bins + 1)
    initial, _ = np.histogram(
        initial_log,
        bins=edges,
        weights=np.asarray(initial_weights, dtype=float),
    )
    current, _ = np.histogram(
        current_log,
        bins=edges,
        weights=np.asarray(current_weights, dtype=float),
    )

    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= np.sum(kernel)

    def normalized(values: np.ndarray) -> list[float]:
        smoothed = np.convolve(values, kernel, mode="same")
        maximum_density = float(np.max(smoothed))
        if maximum_density > 0.0:
            smoothed /= maximum_density
        return [float(value) for value in smoothed]

    return {
        "log10_edges": [float(value) for value in edges],
        "initial_density": normalized(initial),
        "current_density": normalized(current),
    }


def comparison_histograms(
    initial: RotationSamples,
    current: RotationSamples,
    *,
    bins: int = 40,
) -> dict[str, Any]:
    """Return compact common-bin distributions suitable for violins."""

    return {
        "schema": 1,
        "weighting": "formula_repetitions_times_absolute_reaction_stoichiometry",
        "qpe_repetitions_included": False,
        "initial": {
            "unique_schedule_entries": len(initial.angles),
            "weighted_scheduled_rotations": initial.scheduled_rotations,
            "components": list(initial.components),
        },
        "current": {
            "unique_schedule_entries": len(current.angles),
            "weighted_scheduled_rotations": current.scheduled_rotations,
            "components": list(current.components),
        },
        "angles": _common_histograms(
            initial.angles,
            initial.weights,
            current.angles,
            current.weights,
            bins=bins,
        ),
        "tolerances": _common_histograms(
            initial.tolerances,
            initial.weights,
            current.tolerances,
            current.weights,
            bins=bins,
        ),
    }
