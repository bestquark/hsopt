"""Calibrate and instantiate the Reiher et al. PF2 paper baseline.

The paper uses two different accuracy quantities that must not be conflated:

* ``0.1 mHa`` is the reference error used to define the molecule-dependent
  Trotter number beta in the supplement's Figure 4.
* This benchmark uses ``1 kcal/mol`` as its total absolute-energy accuracy
  target.

The default calibration is empirical and intentionally oracular: it computes
the exact ground energy once, then finds the PF2 step count whose unsynthesized
eigenphase reaches the paper's reference error.  The optional Hamiltonian-only
mode estimates the paper's double-commutator bound without using E0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hsopt.compiled_unitary import (
    CHEMICAL_ACCURACY_HARTREE,
    _apply_compiled_rotation,
    _compile_rz,
    _fence,
    _step_rotations,
    _unwrap_energy,
    array_backend,
    evaluate_exact_reference,
    load_problem,
)


PAPER_BETA_REFERENCE_ERROR_HARTREE = 1.0e-4
BENCHMARK_TOTAL_ERROR_HARTREE = CHEMICAL_ACCURACY_HARTREE
PAPER_FORMULA_ORDER = 2
PAPER_TERM_ORDER = "lexicographic"
PAPER_PHASE_ESTIMATION_ALPHA = math.pi / 2.0
PAPER_ERROR_CONSTRAINT = "epsilon_TS + sqrt(epsilon_QPE^2 + epsilon_Rot^2) <= epsilon_total"
PAPER_TROTTER_SELECTION = (
    "empirical_exact_ground_branch_at_0.1_mHa_then_beta_sqrt_reference_over_allocated_trotter_error"
)
HAMILTONIAN_BOUND_SELECTION = (
    "paper_double_commutator_bound_from_pauli_coefficients_without_exact_energy"
)
CALIBRATION_SCHEMA = 2
ALLOCATION_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _calibration_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only deterministic scientific inputs and selected outputs."""

    reference = payload.get("reference", {})
    if not isinstance(reference, Mapping):
        reference = {}
    reference_signature = {
        key: reference[key]
        for key in (
            "mode",
            "definition",
            "reference_error_hartree",
            "reference_step_count",
            "reference_step_time",
            "reference_trotter_number_beta",
            "exact_ground_energy_hartree",
            "double_commutator_sum_upper",
            "bound_method",
            "bound_samples",
            "bound_seed",
            "bound_confidence",
        )
        if key in reference
    }
    return {
        "schema": payload.get("schema"),
        "problem": payload.get("problem"),
        "problem_sha256": payload.get("problem_sha256"),
        "backend": payload.get("backend"),
        "calibration_mode": payload.get("calibration_mode"),
        "paper": payload.get("paper"),
        "benchmark_adaptation": payload.get("benchmark_adaptation"),
        "reference": reference_signature,
        "instantiated_policy": payload.get("instantiated_policy"),
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_calibration_signature(payload))).hexdigest()


def seal_calibration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying a checksum over every other field."""

    sealed = dict(payload)
    sealed["calibration_sha256"] = _payload_sha256(sealed)
    return sealed


def verify_calibration(
    payload: Mapping[str, Any],
    *,
    problem_path: Path | None = None,
) -> None:
    """Validate a calibration before it can generate an immutable policy."""

    if int(payload.get("schema", -1)) != CALIBRATION_SCHEMA:
        raise ValueError("unsupported paper-baseline calibration schema")
    expected = str(payload.get("calibration_sha256", ""))
    if not expected or expected != _payload_sha256(payload):
        raise ValueError("paper-baseline calibration checksum mismatch")
    if problem_path is not None:
        actual = hashlib.sha256(problem_path.resolve().read_bytes()).hexdigest()
        if str(payload.get("problem_sha256", "")) != actual:
            raise ValueError(f"calibration does not match {problem_path}")
    policy = payload.get("instantiated_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("calibration is missing its instantiated policy")
    if int(policy.get("formula_order", -1)) != PAPER_FORMULA_ORDER:
        raise ValueError("paper baseline must use PF2/Strang")
    if str(policy.get("term_order_strategy", "")) != PAPER_TERM_ORDER:
        raise ValueError("paper baseline must use lexicographic term ordering")
    mode = str(payload.get("calibration_mode", ""))
    exact_reference = payload.get("exact_reference")
    if mode == "empirical_exact":
        if not isinstance(exact_reference, Mapping):
            raise ValueError("empirical calibration is missing its reusable exact reference")
        if exact_reference.get("observable") != "exact_unitary_ground_eigenphase":
            raise ValueError("empirical calibration has the wrong exact-reference observable")
        if exact_reference.get("problem_sha256") != payload.get("problem_sha256"):
            raise ValueError("empirical calibration exact reference has the wrong problem hash")
        exact_energy = float(exact_reference["exact_unitary_ground_energy_hartree"])
        calibrated_energy = float(payload["reference"]["exact_ground_energy_hartree"])
        if not math.isclose(exact_energy, calibrated_energy, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("empirical calibration exact-reference energy mismatch")
    elif mode == "hamiltonian_bound":
        if exact_reference is not None:
            raise ValueError("Hamiltonian-only calibration must not contain an exact reference")
    else:
        raise ValueError("unknown paper-baseline calibration mode")


def _exact_pf_energy(
    problem: Mapping[str, Any],
    *,
    exact_energy: float,
    exact_target: Any,
    n_steps: int,
    backend_name: str,
    xp: Any,
) -> dict[str, Any]:
    """Diagonalize one exact-rotation lexicographic Strang step."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    target_time = float(problem["target_time"])
    step_time = target_time / int(n_steps)
    terms = sorted(problem["terms"], key=lambda term: str(term["label"]))
    rotations = _step_rotations(
        terms,
        order=PAPER_FORMULA_ORDER,
        step_time=step_time,
    )
    dimension = 1 << int(problem["n_qubits"])
    unitary = xp.eye(dimension, dtype=xp.complex128)
    workspace = xp.empty_like(unitary)
    for term, alpha in rotations:
        theta = -2.0 * float(alpha)
        exact_rz = np.asarray(
            [
                [np.exp(-0.5j * theta), 0.0],
                [0.0, np.exp(0.5j * theta)],
            ],
            dtype=np.complex128,
        )
        unitary, workspace = _apply_compiled_rotation(
            unitary,
            workspace,
            label=str(term["label"]),
            compiled_rz=exact_rz,
            xp=xp,
        )
    _fence(backend_name)
    del workspace

    eigenvalues, eigenvectors = xp.linalg.eig(unitary)
    _fence(backend_name)
    eigenvectors /= xp.linalg.norm(eigenvectors, axis=0, keepdims=True)
    overlaps = xp.abs(eigenvectors.conj().T @ exact_target) ** 2
    selected = int(np.asarray(xp.argmax(overlaps)))
    step_phase = float(np.asarray(xp.angle(eigenvalues[selected])))
    total_phase = int(n_steps) * step_phase
    identity = float(problem["identity_coefficient"])
    centered_reference = exact_energy - identity
    energy = _unwrap_energy(total_phase, target_time, centered_reference) + identity
    selected_overlap = float(np.asarray(overlaps[selected]))
    _fence(backend_name)
    del unitary, eigenvalues, eigenvectors, overlaps
    _fence(backend_name)
    return {
        "n_steps": int(n_steps),
        "step_time": step_time,
        "selected_exact_ground_overlap": selected_overlap,
        "selected_step_eigenphase_radians": step_phase,
        "selected_total_eigenphase_radians": total_phase,
        "unsynthesized_pf_energy_hartree": float(energy),
        "absolute_trotter_error_hartree": abs(float(energy) - exact_energy),
    }


def _empirical_reference_beta(
    problem: Mapping[str, Any],
    *,
    exact_reference: Mapping[str, Any],
    backend_name: str,
    xp: Any,
    reference_error_hartree: float,
    max_steps: int,
) -> dict[str, Any]:
    """Find the first passing integer on the observed monotone PF2 branch."""

    if exact_reference.get("observable") != "exact_unitary_ground_eigenphase":
        raise ValueError("empirical beta calibration requires an exact-unitary reference")
    exact_energy = float(exact_reference["exact_unitary_ground_energy_hartree"])
    exact_target = xp.asarray(
        [
            complex(float(real), float(imaginary))
            for real, imaginary in exact_reference["exact_ground_state"]
        ],
        dtype=xp.complex128,
    )
    _fence(backend_name)
    evaluations: dict[int, dict[str, Any]] = {}

    def evaluate(n_steps: int) -> dict[str, Any]:
        if n_steps not in evaluations:
            evaluations[n_steps] = _exact_pf_energy(
                problem,
                exact_energy=exact_energy,
                exact_target=exact_target,
                n_steps=n_steps,
                backend_name=backend_name,
                xp=xp,
            )
        return evaluations[n_steps]

    lower = 0
    upper = 1
    while evaluate(upper)["absolute_trotter_error_hartree"] > reference_error_hartree:
        lower = upper
        upper *= 2
        if upper > max_steps:
            raise RuntimeError(f"{problem['name']}: no passing PF2 step count <= {max_steps}")
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if evaluate(middle)["absolute_trotter_error_hartree"] <= reference_error_hartree:
            upper = middle
        else:
            lower = middle
    if upper > 1:
        evaluate(upper - 1)

    ordered = [evaluations[key] for key in sorted(evaluations)]
    monotonicity_violations = 0
    for left, right in zip(ordered, ordered[1:], strict=False):
        if (
            int(right["n_steps"]) > int(left["n_steps"])
            and float(right["absolute_trotter_error_hartree"])
            > float(left["absolute_trotter_error_hartree"]) + 1.0e-12
        ):
            monotonicity_violations += 1
    selected = evaluate(upper)
    _fence(backend_name)
    target_time = float(problem["target_time"])
    return {
        "mode": "empirical_exact",
        "definition": PAPER_TROTTER_SELECTION,
        "reference_error_hartree": float(reference_error_hartree),
        "reference_step_count": int(upper),
        "reference_step_time": target_time / int(upper),
        "reference_trotter_number_beta": int(upper) / target_time,
        "exact_ground_energy_hartree": float(exact_energy),
        "exact_reference_seconds": float(exact_reference["timing_seconds"]["total"]),
        "search_assumption": (
            "binary refinement on the asymptotically monotone PF2 error branch; "
            "the selected r and r-1 are both evaluated"
        ),
        "observed_monotonicity_violations": monotonicity_violations,
        "selected_evaluation": selected,
        "search_evaluations": ordered,
        "fixed_electron_sector_dimension": int(exact_reference["fixed_electron_sector_dimension"]),
        "ground_subspace_dimension": int(exact_reference["exact_ground_subspace_dimension"]),
        "target_anchor_kind": str(exact_reference["target_anchor_kind"]),
        "target_anchor_basis_index": int(exact_reference["target_anchor_basis_index"]),
        "hf_ground_subspace_probability": float(exact_reference["hf_ground_subspace_probability"]),
        "hf_target_probability": float(exact_reference["qpe_success_probability_from_hf"]),
    }


def _pauli_anticommutation_matrix(labels: Sequence[str]) -> np.ndarray:
    """Return exact pairwise anticommutation for Pauli labels."""

    x_masks: list[int] = []
    z_masks: list[int] = []
    for label in labels:
        x_mask = 0
        z_mask = 0
        for index, character in enumerate(label):
            bit = 1 << index
            if character in {"X", "Y"}:
                x_mask |= bit
            if character in {"Z", "Y"}:
                z_mask |= bit
        x_masks.append(x_mask)
        z_masks.append(z_mask)
    count = len(labels)
    result = np.zeros((count, count), dtype=np.bool_)
    for left in range(count):
        for right in range(left + 1, count):
            parity = (
                (x_masks[left] & z_masks[right]).bit_count()
                + (z_masks[left] & x_masks[right]).bit_count()
            ) & 1
            result[left, right] = bool(parity)
            result[right, left] = bool(parity)
    return result


def _double_commutator_bound(
    problem: Mapping[str, Any],
    *,
    reference_error_hartree: float,
    samples: int,
    seed: int,
    confidence: float,
    exhaustive_triples: int,
) -> dict[str, Any]:
    """Estimate the supplement's Eq. 19 using only Pauli-term metadata."""

    terms = sorted(problem["terms"], key=lambda term: str(term["label"]))
    coefficients = np.asarray(
        [abs(float(term["coefficient"])) for term in terms],
        dtype=np.float64,
    )
    count = len(terms)
    anticommuting = _pauli_anticommutation_matrix([str(term["label"]) for term in terms])

    def values(
        alpha: np.ndarray,
        beta: np.ndarray,
        alpha_prime: np.ndarray,
    ) -> np.ndarray:
        ordering = ((alpha > beta) & (alpha_prime > beta)).astype(np.int8) + (
            (beta > alpha) & (alpha_prime > alpha)
        ).astype(np.int8)
        inner_nonzero = anticommuting[beta, alpha_prime]
        outer_nonzero = np.logical_xor(
            anticommuting[alpha, beta],
            anticommuting[alpha, alpha_prime],
        )
        return (
            4.0
            * coefficients[alpha]
            * coefficients[beta]
            * coefficients[alpha_prime]
            * ordering
            * inner_nonzero
            * outer_nonzero
        )

    triple_count = count**3
    if triple_count <= exhaustive_triples:
        total = 0.0
        for alpha_value in range(count):
            alpha = np.full(count * count, alpha_value, dtype=np.int64)
            beta = np.repeat(np.arange(count, dtype=np.int64), count)
            alpha_prime = np.tile(np.arange(count, dtype=np.int64), count)
            total += float(values(alpha, beta, alpha_prime).sum())
        estimate = total
        standard_error = 0.0
        upper = total
        method = "exhaustive"
        sample_count = triple_count
    else:
        if samples < 2:
            raise ValueError("Hamiltonian-bound Monte Carlo requires at least two samples")
        if not 0.5 < confidence < 1.0:
            raise ValueError("bound confidence must lie strictly between 0.5 and 1")
        rng = np.random.default_rng(seed)
        running_sum = 0.0
        running_square_sum = 0.0
        remaining = int(samples)
        while remaining:
            size = min(remaining, 1_000_000)
            alpha = rng.integers(0, count, size=size, dtype=np.int64)
            beta = rng.integers(0, count, size=size, dtype=np.int64)
            alpha_prime = rng.integers(0, count, size=size, dtype=np.int64)
            chunk = values(alpha, beta, alpha_prime)
            running_sum += float(chunk.sum())
            running_square_sum += float(np.square(chunk).sum())
            remaining -= size
        mean = running_sum / samples
        variance = max(
            0.0,
            (running_square_sum - samples * mean * mean) / (samples - 1),
        )
        standard_error_of_mean = math.sqrt(variance / samples)
        z_score = statistics.NormalDist().inv_cdf(confidence)
        estimate = triple_count * mean
        standard_error = triple_count * standard_error_of_mean
        upper = max(0.0, estimate + z_score * standard_error)
        method = "uniform_monte_carlo_upper_confidence_estimate"
        sample_count = int(samples)

    target_time = float(problem["target_time"])
    if upper <= 0.0:
        beta = 1.0 / target_time
    else:
        beta = math.sqrt(upper / reference_error_hartree)
    reference_steps = max(1, math.ceil(target_time * beta))
    return {
        "mode": "hamiltonian_bound",
        "definition": HAMILTONIAN_BOUND_SELECTION,
        "reference_error_hartree": float(reference_error_hartree),
        "reference_step_count": reference_steps,
        "reference_step_time": target_time / reference_steps,
        "reference_trotter_number_beta": float(beta),
        "bound_method": method,
        "bound_samples": sample_count,
        "bound_seed": int(seed),
        "bound_confidence": float(confidence),
        "double_commutator_sum_estimate": float(estimate),
        "double_commutator_sum_standard_error": float(standard_error),
        "double_commutator_sum_upper": float(upper),
        "exact_ground_energy_hartree": None,
    }


def _one_step_t_count(
    problem: Mapping[str, Any],
    *,
    n_steps: int,
    uniform_epsilon: float,
) -> tuple[int, int]:
    """Compile the actual PF2 angles and return one-step T and gate counts."""

    terms = sorted(problem["terms"], key=lambda term: str(term["label"]))
    rotations = _step_rotations(
        terms,
        order=PAPER_FORMULA_ORDER,
        step_time=float(problem["target_time"]) / int(n_steps),
    )
    t_count = 0
    gate_count = 0
    for _term, alpha in rotations:
        theta = float(round(-2.0 * float(alpha), 15))
        compiled = _compile_rz(theta, float(uniform_epsilon))
        if float(compiled["operator_error"]) > float(uniform_epsilon) + 1.0e-15:
            raise RuntimeError("PyGridSynth exceeded the requested synthesis tolerance")
        t_count += int(compiled["t_count"])
        gate_count += int(compiled["gate_count"])
    return t_count, gate_count


def _allocation_candidate(
    problem: Mapping[str, Any],
    *,
    beta: float,
    reference_error_hartree: float,
    total_error_hartree: float,
    trotter_fraction: float,
    rotation_radial_fraction: float,
) -> dict[str, Any]:
    """Instantiate and directly compile one paper-constraint allocation."""

    if not 0.0 < trotter_fraction < 1.0:
        raise ValueError("Trotter fraction must lie strictly between zero and one")
    if not 0.0 < rotation_radial_fraction < 1.0:
        raise ValueError("rotation fraction must lie strictly between zero and one")
    epsilon_ts = total_error_hartree * trotter_fraction
    radial = total_error_hartree - epsilon_ts
    epsilon_rot = radial * rotation_radial_fraction
    epsilon_qpe = math.sqrt(max(0.0, radial * radial - epsilon_rot * epsilon_rot))
    target_time = float(problem["target_time"])
    n_steps = max(
        1,
        math.ceil(target_time * float(beta) * math.sqrt(reference_error_hartree / epsilon_ts)),
    )
    step_time = target_time / n_steps
    terms = len(problem["terms"])
    rotations_per_step = 2 * terms - 1
    uniform_epsilon = epsilon_rot * step_time / rotations_per_step
    one_step_t_count, one_step_gate_count = _one_step_t_count(
        problem,
        n_steps=n_steps,
        uniform_epsilon=uniform_epsilon,
    )
    full_pf_t_count = n_steps * one_step_t_count
    qpe_full_evolution_queries = max(
        1,
        math.ceil(PAPER_PHASE_ESTIMATION_ALPHA / (target_time * epsilon_qpe)),
    )
    qpe_step_queries = qpe_full_evolution_queries * n_steps
    qpe_t_count = qpe_full_evolution_queries * full_pf_t_count
    return {
        "trotter_fraction": float(trotter_fraction),
        "rotation_radial_fraction": float(rotation_radial_fraction),
        "phase_estimation_error_hartree": epsilon_qpe,
        "trotter_error_hartree": epsilon_ts,
        "rotation_synthesis_energy_error_hartree": epsilon_rot,
        "error_constraint_value_hartree": epsilon_ts + math.hypot(epsilon_qpe, epsilon_rot),
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": PAPER_FORMULA_ORDER,
        "n_steps": n_steps,
        "nonidentity_terms": terms,
        "one_step_gate_count": one_step_gate_count,
        "one_step_t_count": one_step_t_count,
        "full_pf_t_count": full_pf_t_count,
        "qpe_full_evolution_queries": qpe_full_evolution_queries,
        "qpe_step_queries": qpe_step_queries,
        "qpe_t_count_proxy": qpe_t_count,
        "reference_error_hartree": float(reference_error_hartree),
        "reference_trotter_number_beta": float(beta),
        "rotation_synthesis": "clifford_plus_t_pygridsynth",
        "scheduled_rotations": n_steps * rotations_per_step,
        "scheduled_rotations_per_step": rotations_per_step,
        "step_repetitions": n_steps,
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "step_time": step_time,
        "target_time": target_time,
        "term_order_strategy": PAPER_TERM_ORDER,
        "total_error_target_hartree": float(total_error_hartree),
        "uniform_epsilon": uniform_epsilon,
        "zero_angle_threshold": 0.0,
    }


def optimize_direct_error_allocation(
    problem: Mapping[str, Any],
    *,
    reference_trotter_number_beta: float,
    reference_error_hartree: float = PAPER_BETA_REFERENCE_ERROR_HARTREE,
    total_error_hartree: float = BENCHMARK_TOTAL_ERROR_HARTREE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Minimize direct PyGridSynth QPE T cost under the paper constraint."""

    if reference_trotter_number_beta <= 0.0:
        raise ValueError("reference Trotter number beta must be positive")
    if reference_error_hartree <= 0.0 or total_error_hartree <= 0.0:
        raise ValueError("error targets must be positive")
    evaluated: dict[tuple[float, float], dict[str, Any]] = {}

    def evaluate(trotter_fraction: float, rotation_fraction: float) -> None:
        key = (round(trotter_fraction, 12), round(rotation_fraction, 12))
        if key not in evaluated:
            evaluated[key] = _allocation_candidate(
                problem,
                beta=reference_trotter_number_beta,
                reference_error_hartree=reference_error_hartree,
                total_error_hartree=total_error_hartree,
                trotter_fraction=key[0],
                rotation_radial_fraction=key[1],
            )

    for trotter_fraction in ALLOCATION_GRID:
        for rotation_fraction in ALLOCATION_GRID:
            evaluate(trotter_fraction, rotation_fraction)

    for _round in range(2):
        best = min(
            evaluated.values(),
            key=lambda row: (
                int(row["qpe_t_count_proxy"]),
                int(row["full_pf_t_count"]),
                float(row["uniform_epsilon"]),
            ),
        )
        for field in ("trotter_fraction", "rotation_radial_fraction"):
            value = float(best[field])
            existing = sorted({key[0 if field == "trotter_fraction" else 1] for key in evaluated})
            position = min(range(len(existing)), key=lambda index: abs(existing[index] - value))
            lower = existing[position - 1] if position > 0 else 0.01
            upper = existing[position + 1] if position + 1 < len(existing) else 0.99
            if field == "trotter_fraction":
                trotter_values = ((lower + value) / 2.0, value, (value + upper) / 2.0)
                rotation_values = (float(best["rotation_radial_fraction"]),)
            else:
                trotter_values = (float(best["trotter_fraction"]),)
                rotation_values = ((lower + value) / 2.0, value, (value + upper) / 2.0)
            for trotter_fraction in trotter_values:
                for rotation_fraction in rotation_values:
                    evaluate(trotter_fraction, rotation_fraction)

    ordered = sorted(
        evaluated.values(),
        key=lambda row: (
            int(row["qpe_t_count_proxy"]),
            int(row["full_pf_t_count"]),
            float(row["trotter_fraction"]),
            float(row["rotation_radial_fraction"]),
        ),
    )
    selected = dict(ordered[0])
    selected["error_allocation"] = (
        "direct_pygridsynth_t_count_optimization_under_paper_variance_constraint"
    )
    selected["allocation_candidates_evaluated"] = len(ordered)
    return selected, ordered


def calibrate_problem(
    problem_path: Path,
    *,
    backend: str,
    mode: str = "empirical_exact",
    target_time: float = 1.0,
    reference_error_hartree: float = PAPER_BETA_REFERENCE_ERROR_HARTREE,
    total_error_hartree: float = BENCHMARK_TOTAL_ERROR_HARTREE,
    max_steps: int = 1 << 20,
    bound_samples: int = 1_000_000,
    bound_seed: int = 1619152114,
    bound_confidence: float = 0.99,
    exhaustive_bound_triples: int = 5_000_000,
) -> dict[str, Any]:
    """Calibrate beta(H), allocate errors, and instantiate one immutable PF2."""

    started = time.perf_counter()
    path = problem_path.resolve()
    problem = load_problem(path)
    if not math.isclose(
        float(problem["target_time"]),
        float(target_time),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(
            f"{path}: source target_time={problem['target_time']} does not match "
            f"requested target_time={target_time}; regenerate the source explicitly"
        )
    if float(problem["target_time"]) <= 0.0:
        raise ValueError("paper baseline target_time must be positive")
    if reference_error_hartree <= 0.0 or total_error_hartree <= 0.0:
        raise ValueError("paper baseline error targets must be positive")
    backend_name, xp = array_backend(backend)

    beta_started = time.perf_counter()
    exact_reference: dict[str, Any] | None = None
    if mode == "empirical_exact":
        exact_reference = evaluate_exact_reference(path, backend=backend_name)
        reference = _empirical_reference_beta(
            problem,
            exact_reference=exact_reference,
            backend_name=backend_name,
            xp=xp,
            reference_error_hartree=reference_error_hartree,
            max_steps=max_steps,
        )
    elif mode == "hamiltonian_bound":
        reference = _double_commutator_bound(
            problem,
            reference_error_hartree=reference_error_hartree,
            samples=bound_samples,
            seed=bound_seed,
            confidence=bound_confidence,
            exhaustive_triples=exhaustive_bound_triples,
        )
    else:
        raise ValueError("mode must be 'empirical_exact' or 'hamiltonian_bound'")
    beta_seconds = time.perf_counter() - beta_started

    allocation_started = time.perf_counter()
    policy, allocation_search = optimize_direct_error_allocation(
        problem,
        reference_trotter_number_beta=float(reference["reference_trotter_number_beta"]),
        reference_error_hartree=reference_error_hartree,
        total_error_hartree=total_error_hartree,
    )
    allocation_seconds = time.perf_counter() - allocation_started
    result = {
        "schema": CALIBRATION_SCHEMA,
        "problem": str(problem["name"]),
        "problem_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "backend": backend_name,
        "calibration_mode": mode,
        "paper": {
            "doi": "10.1073/pnas.1619152114",
            "formula_order": PAPER_FORMULA_ORDER,
            "formula_name": "symmetric_second_order_trotter_suzuki",
            "term_order_strategy": PAPER_TERM_ORDER,
            "beta_reference_error_hartree": float(reference_error_hartree),
            "error_constraint": PAPER_ERROR_CONSTRAINT,
            "phase_estimation_alpha": PAPER_PHASE_ESTIMATION_ALPHA,
            "synthesis_energy_bound": "(2M - 1) * Delta_synth / step_time",
        },
        "benchmark_adaptation": {
            "total_error_target_hartree": float(total_error_hartree),
            "total_error_target_kcal_per_mol": float(
                total_error_hartree / CHEMICAL_ACCURACY_HARTREE
            ),
            "target_time": float(target_time),
            "uses_gamma_delta_synthesis_fit": False,
            "allocation_cost_source": "direct PyGridSynth compilation",
        },
        "reference": reference,
        "exact_reference": exact_reference,
        "instantiated_policy": policy,
        "allocation_search": allocation_search,
        "timing_seconds": {
            "beta_calibration": beta_seconds,
            "direct_allocation_search": allocation_seconds,
            "total": time.perf_counter() - started,
        },
    }
    return seal_calibration(result)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("numpy", "cupynumeric"), default="numpy")
    parser.add_argument(
        "--mode",
        choices=("empirical_exact", "hamiltonian_bound"),
        default="empirical_exact",
    )
    parser.add_argument("--target-time", type=float, default=1.0)
    parser.add_argument(
        "--reference-error-hartree",
        type=float,
        default=PAPER_BETA_REFERENCE_ERROR_HARTREE,
    )
    parser.add_argument(
        "--total-error-hartree",
        type=float,
        default=BENCHMARK_TOTAL_ERROR_HARTREE,
    )
    parser.add_argument("--max-steps", type=int, default=1 << 20)
    parser.add_argument("--bound-samples", type=int, default=1_000_000)
    parser.add_argument("--bound-seed", type=int, default=1619152114)
    parser.add_argument("--bound-confidence", type=float, default=0.99)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = calibrate_problem(
        args.problem,
        backend=args.backend,
        mode=args.mode,
        target_time=args.target_time,
        reference_error_hartree=args.reference_error_hartree,
        total_error_hartree=args.total_error_hartree,
        max_steps=args.max_steps,
        bound_samples=args.bound_samples,
        bound_seed=args.bound_seed,
        bound_confidence=args.bound_confidence,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != encoded:
            raise SystemExit(f"paper baseline calibration is stale: {args.output}")
        return 0
    _write_json(args.output.resolve(), result)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
