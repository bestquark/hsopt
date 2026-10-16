"""Count-only PF resource analysis; never construct a state or solve an energy.

Large-case tight allocations are explicitly extrapolations, not calibrated
baselines. A count, including a literal CUDA-Q count, establishes neither an
energy error nor QPE success probability. All counts refer to one complete
physical evolution (r stationary steps), not T95 or an entire QPE circuit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hsopt.campaign.calibration import (
    BENCHMARK_TOTAL_ERROR_HARTREE,
    PAPER_BETA_REFERENCE_ERROR_HARTREE,
    optimize_paper_error_allocation,
)
from hsopt.evaluation import compiled_unitary as ev


FEATURE_NAMES = ("log_n_qubits", "log_nonidentity_terms", "log_one_norm")
MODEL_SCHEMA = "count-only-tight-beta-log-ridge-v1"


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _features(row: Mapping[str, Any]) -> list[float]:
    count = row.get("nonidentity_terms")
    if count is None:
        count = len(row["terms"])
    norm = row.get("one_norm")
    if norm is None:
        norm = 1.0 / _positive(row["target_time"], "target_time")
    return [
        math.log(_positive(row["n_qubits"], "n_qubits")),
        math.log(_positive(count, "nonidentity_terms")),
        math.log(_positive(norm, "one_norm")),
    ]


def _check_profile(row: Mapping[str, Any], profile: str) -> None:
    actual = row.get("policy_profile", row.get("profile"))
    if actual is not None and actual != profile:
        raise ValueError(f"calibration profile {actual!r} does not match {profile!r}")
    if row.get("baseline", "tight") != "tight":
        raise ValueError("tight-beta resource extrapolation requires tight calibration rows")


def _fit(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, ...]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    # Penalize slopes in standardized feature coordinates, never the intercept.
    penalty = math.sqrt(ridge) * np.diag([0.0, 1.0, 1.0, 1.0])
    coefficients = np.linalg.lstsq(
        np.vstack((design, penalty)), np.concatenate((y, np.zeros(4))), rcond=None
    )[0]
    return mean, scale, coefficients


def _predict(x: np.ndarray, fit: tuple[np.ndarray, ...]) -> np.ndarray:
    mean, scale, coefficients = fit
    return coefficients[0] + ((x - mean) / scale) @ coefficients[1:]


def _diagnostics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    return {
        "log_rmse": float(np.sqrt(np.mean(errors**2))),
        "median_multiplicative_error": float(np.exp(np.median(np.abs(errors)))),
        "max_multiplicative_error": float(np.exp(np.max(np.abs(errors)))),
    }


def fit_beta_model(
    rows: Sequence[Mapping[str, Any]], *, ridge: float = 1e-3, profile: str = "pauli"
) -> dict[str, Any]:
    """Fit log(beta * target_time) using only supplied small-case calibrations.

    Validation is leave-one-row-out and, when molecule labels permit it,
    leave-one-molecule-out. Neither diagnostic certifies a larger-case error.
    The caller owns the selected training scope; its exact rows are hashed.
    """
    if len(rows) < 3:
        raise ValueError("at least three small calibrated rows are needed")
    if not profile or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("a profile and finite nonnegative ridge are required")
    records = []
    for row in rows:
        _check_profile(row, profile)
        records.append(
            {
                "name": str(row.get("name", row.get("case_id", ""))),
                "molecule": str(row.get("molecule", "")),
                "features": _features(row),
                "target_time": _positive(row["target_time"], "target_time"),
                "beta": _positive(row["reference_trotter_number_beta"], "beta"),
            }
        )
    x = np.asarray([r["features"] for r in records])
    y = np.asarray([math.log(r["beta"]) + math.log(r["target_time"]) for r in records])
    fitted = _fit(x, y, ridge)
    loo = np.zeros_like(y)
    for index in range(len(rows)):
        keep = np.arange(len(rows)) != index
        loo[index] = _predict(x[index : index + 1], _fit(x[keep], y[keep], ridge))[0]
    groups = [r["molecule"] for r in records]
    grouped_diagnostics = None
    if all(groups) and len(set(groups)) > 1:
        predicted = np.zeros_like(y)
        for molecule in sorted(set(groups)):
            held = np.asarray([name == molecule for name in groups])
            predicted[held] = _predict(x[held], _fit(x[~held], y[~held], ridge))
        grouped_diagnostics = _diagnostics(y, predicted)
    mean, scale, coefficients = fitted
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema": MODEL_SCHEMA,
        "policy_profile": profile,
        "baseline": "tight",
        "fit_target": "log(reference_trotter_number_beta * target_time)",
        "time_convention": "target_time=1/nonidentity_Pauli_one_norm_in_inverse_Hartree",
        "beta_convention": "reference_step_count/target_time_at_0.1_mHa",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge": ridge,
        "training_count": len(rows),
        "training_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "training_names": [r["name"] for r in records],
        "training_feature_min": x.min(axis=0).tolist(),
        "training_feature_max": x.max(axis=0).tolist(),
        "in_sample": _diagnostics(y, _predict(x, fitted)),
        "leave_one_out": _diagnostics(y, loo),
        "leave_one_molecule_out": grouped_diagnostics,
        "accuracy_certified": False,
        "scope": "empirical resource extrapolation only; no large-case energy calibration",
    }


def predict_beta(
    problem: Mapping[str, Any], model: Mapping[str, Any], *, profile: str = "pauli"
) -> dict[str, Any]:
    """Predict beta for resource accounting without solving H or an ideal PF."""
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported beta extrapolation model")
    _check_profile(model, profile)
    features = np.asarray(_features(problem))
    fit = tuple(np.asarray(model[key]) for key in ("feature_mean", "feature_scale", "coefficients"))
    log_steps = float(_predict(features[None, :], fit)[0])
    if not math.isfinite(log_steps) or log_steps > 700:
        raise ValueError("non-finite or overflowing beta prediction")
    reference_steps = math.exp(max(0.0, log_steps))
    outside = [
        name
        for name, value, low, high in zip(
            FEATURE_NAMES,
            features,
            model["training_feature_min"],
            model["training_feature_max"],
            strict=True,
        )
        if value < low - 1e-12 or value > high + 1e-12
    ]
    return {
        "reference_trotter_number_beta": reference_steps
        / _positive(problem["target_time"], "target_time"),
        "predicted_reference_steps": reference_steps,
        "step_floor_applied": log_steps < 0.0,
        "outside_training_features": outside,
        "training_sha256": model["training_sha256"],
        "accuracy_certified": False,
    }


def _validate_parameters(row: Mapping[str, Any]) -> tuple[int, float, float]:
    steps = row["n_steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("n_steps must be a positive integer")
    epsilon = _positive(row["uniform_epsilon"], "uniform_epsilon")
    if epsilon >= 1:
        raise ValueError("uniform_epsilon must be in (0, 1)")
    beta = _positive(row["reference_trotter_number_beta"], "beta")
    return steps, epsilon, beta


def allocate_parameters(
    problem: Mapping[str, Any],
    *,
    sealed: Mapping[str, Any] | None = None,
    beta_model: Mapping[str, Any] | None = None,
    profile: str = "pauli",
    allocation_terms: Sequence[Mapping[str, Any]] | None = None,
    rotations_per_step: int | None = None,
) -> dict[str, Any]:
    """Prefer sealed small-case parameters; otherwise apply Eq.34 to predicted beta.

    The Pauli default uses PF2's rotation count. A nested profile requires its
    own beta fit, trusted forward-decomposition terms and actual rotation count,
    matching ``hsopt.campaign.elote.prepare``. Neither path calls calibration.
    """
    if sealed is not None:
        _check_profile(sealed, profile)
        _validate_parameters(sealed)
        return {
            **copy.deepcopy(dict(sealed)),
            "policy_profile": profile,
            "allocation_source": "sealed_calibration",
            "allocation_extrapolated": False,
        }
    if beta_model is None:
        raise ValueError("provide sealed parameters or an explicit beta_model")
    prediction = predict_beta(problem, beta_model, profile=profile)
    if profile != "pauli" and (allocation_terms is None or rotations_per_step is None):
        raise ValueError("nested extrapolation needs allocation_terms and rotations_per_step")
    if rotations_per_step is not None and (
        type(rotations_per_step) is not int or rotations_per_step < 1
    ):
        raise ValueError("rotations_per_step must be a positive integer")
    if allocation_terms is not None and not allocation_terms:
        raise ValueError("allocation_terms cannot be empty")
    allocation_problem = (
        {**problem, "terms": allocation_terms} if allocation_terms is not None else problem
    )
    parameters, derivation = optimize_paper_error_allocation(
        allocation_problem,
        beta=prediction["reference_trotter_number_beta"],
        reference_error_hartree=PAPER_BETA_REFERENCE_ERROR_HARTREE,
        total_error_hartree=BENCHMARK_TOTAL_ERROR_HARTREE,
        rotations_per_step=rotations_per_step,
    )
    _validate_parameters(parameters)
    return {
        **parameters,
        "policy_profile": profile,
        "allocation_source": "extrapolated_tight_beta",
        "allocation_extrapolated": True,
        "beta_prediction": prediction,
        "allocation_derivation": derivation,
    }


def count_policy(
    problem: Mapping[str, Any],
    policy_path: Path,
    allocation: Mapping[str, Any],
    *,
    method: str = "cudaq",
    gamma: float = 3.0,
    offset: float = 0.0,
) -> dict[str, Any]:
    """Count the trusted stationary schedule, without allocating 2**n objects.

    ``cudaq`` counts literal compiler outputs; ``analytic_ross_selinger`` uses
    max(0, gamma*log2(1/epsilon)+offset) per rotation and is only a cost model.
    This same schedule interface supports grouped and Clifford-frame ELOTEs.
    """
    if method not in {"cudaq", "analytic_ross_selinger"}:
        raise ValueError("method must be cudaq or analytic_ross_selinger")
    if not math.isfinite(gamma) or gamma <= 0 or not math.isfinite(offset):
        raise ValueError("analytic gamma must be positive and offset finite")
    n_qubits = problem["n_qubits"]
    if type(n_qubits) is not int or n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer")
    _positive(problem["target_time"], "target_time")
    if not problem["terms"]:
        raise ValueError("a nonempty nonidentity Hamiltonian is required")
    for term in problem["terms"]:
        label = term["label"]
        if len(label) != n_qubits or set(label) - set("IXYZ") or set(label) == {"I"}:
            raise ValueError("invalid nonidentity Pauli label")
        if not math.isfinite(float(term["coefficient"])):
            raise ValueError("Hamiltonian coefficients must be finite")
    steps, epsilon, beta = _validate_parameters(allocation)
    module = ev.load_policy(Path(policy_path))
    name = str(problem["name"])
    module.ProductFormula = type(
        "ProductFormula",
        (module.ProductFormula,),
        {
            "baseline_parameters": {name: (steps, epsilon, beta)},
            "parameters": {name: (steps, epsilon)},
        },
    )
    started = time.perf_counter()
    formula, rotations, metrics = ev._product_formula_schedule(problem, module)
    # Use the evaluator's rounding and tolerance contract even in proxy mode.
    if method == "cudaq":
        keys, compiled, synthesis = ev._compile_rotations(rotations, formula)
        counts = [compiled[key]["t_count"] for key in keys]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("CUDA-Q returned an invalid T count")
        one_step_count = sum(counts)
        actual_count, estimate = one_step_count * metrics["n_steps"], None
    else:
        keys = []
        for term, angle in rotations:
            tolerance = float(formula.synthesis_epsilon(dict(term), angle))
            if not math.isfinite(tolerance) or not 0 < tolerance < 1:
                raise ValueError("ProductFormula.synthesis_epsilon() must return a value in (0, 1)")
            keys.append((float(round(-2 * angle, 15)), tolerance))
        one_step_count = math.fsum(max(0.0, gamma * math.log2(1 / e) + offset) for _, e in keys)
        actual_count, estimate = None, one_step_count * metrics["n_steps"]
        synthesis = {"unique_rotation_count": len(set(keys)), "wall_seconds": 0.0}
    tolerances = np.asarray([epsilon for _, epsilon in keys])
    resolved = {
        "recipe": metrics["elote"],
        "compiler_keys": [[theta, tolerance] for theta, tolerance in keys],
    }
    schedule_bytes = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "name": name,
        "n_qubits": n_qubits,
        "target_time": float(problem["target_time"]),
        "nonidentity_terms": len(problem["terms"]),
        "count_method": method,
        "exact_compiled_count": method == "cudaq",
        "analytic_estimate": method != "cudaq",
        "compiled_evolution_t_count": actual_count,
        "estimated_evolution_t_count": estimate,
        "count_scope": "one_complete_evolution_excluding_QPE_and_state_preparation",
        "energy_evaluated": False,
        "accuracy_certified": False,
        "allocation_source": allocation.get("allocation_source", "sealed_calibration"),
        "allocation_extrapolated": allocation.get("allocation_extrapolated", False),
        "policy_profile": allocation.get("policy_profile", "pauli"),
        "bound_parameters": {
            "n_steps": steps,
            "uniform_epsilon": epsilon,
            "reference_trotter_number_beta": beta,
        },
        "n_steps": metrics["n_steps"],
        "rotations_per_step": len(rotations),
        "scheduled_rotation_count": len(rotations) * metrics["n_steps"],
        "unique_rotation_count": synthesis["unique_rotation_count"],
        "formula_name": metrics["formula_name"],
        "formula_order": metrics["formula_order"],
        "grouping": metrics["elote"]["grouping"],
        "basis_change": metrics["elote"]["basis_change"],
        "policy_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
        "problem_sha256": hashlib.sha256(
            json.dumps(
                {key: value for key, value in problem.items() if key != "_path"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "rotation_schedule_sha256": metrics["rotation_schedule_sha256"],
        "resolved_schedule_sha256": hashlib.sha256(schedule_bytes.encode()).hexdigest(),
        "synthesis_options": dict(ev.SYNTHESIS_OPTIONS) if method == "cudaq" else None,
        "analytic_cost_model": {"gamma": gamma, "offset": offset} if method != "cudaq" else None,
        "tolerance": {
            "minimum": float(tolerances.min()),
            "maximum": float(tolerances.max()),
            "geometric_mean": float(np.exp(np.log(tolerances).mean())),
        },
        "synthesis_seconds": synthesis["wall_seconds"],
        "wall_seconds": time.perf_counter() - started,
    }
