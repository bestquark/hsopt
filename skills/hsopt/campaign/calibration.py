"""Calibrate and instantiate the Reiher et al. PF2 paper baseline.

The paper uses two different accuracy quantities that must not be conflated:

* ``0.1 mHa`` is the reference error used to define the molecule-dependent
  Trotter number beta in the supplement's Figure 4.
* This benchmark uses ``1 kcal/mol`` as its total absolute-energy accuracy
  target.

This module owns calibration and baseline parameter preparation;
experiments/baseline/generate_initial_pf.py owns the compact PF object and policy
emission. HSOpt exposes two controlled versions of the same paper PF2/Eq. (34)
baseline.  ``loose`` computes the paper's Eq. (19) double-commutator
coefficient bound exactly for the Pauli decomposition and obtains beta(H)
without using E0.  ``tight`` measures beta(H) numerically against the cached
exact ground energy at the same 0.1 mHa reference error.  Both versions reuse
the same exact reference, error-allocation constants, term order, evaluator,
and total benchmark error target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hsopt.evaluation.compiled_unitary import (
    CHEMICAL_ACCURACY_HARTREE,
    QPE_PHASE_ESTIMATION_ALPHA,
    _apply_compiled_rotation,
    _decode_energy,
    _fence,
    _host_array,
    _step_rotations,
    array_backend,
    evaluate_exact_reference,
    load_problem,
)


PAPER_BETA_REFERENCE_ERROR_HARTREE = 1.0e-4
BENCHMARK_TOTAL_ERROR_HARTREE = CHEMICAL_ACCURACY_HARTREE
PAPER_FORMULA_ORDER = 2
PAPER_TERM_ORDER = "lexicographic"
PAPER_PHASE_ESTIMATION_ALPHA = QPE_PHASE_ESTIMATION_ALPHA
PAPER_SYNTHESIS_GAMMA = 4.0
PAPER_SYNTHESIS_DELTA_0 = 11.0
PAPER_ERROR_CONSTRAINT = "epsilon_QPE + epsilon_TS + epsilon_Rot <= epsilon_total"
PAPER_ALLOCATION_METHOD = "supplement_eq34_rigorous_additive_analytic_optimum"
PAPER_TROTTER_SELECTION = (
    "empirical_exact_ground_branch_at_0.1_mHa_then_beta_sqrt_reference_over_allocated_trotter_error"
)
HAMILTONIAN_BOUND_SELECTION = "paper_eq19_exact_pauli_double_commutator_bound_without_exact_energy"
CALIBRATION_SCHEMA = 5
BASELINE_VARIANT_TO_MODE = {
    "loose": "hamiltonian_bound",
    "tight": "empirical_exact",
}
BASELINE_VARIANTS = tuple(BASELINE_VARIANT_TO_MODE)


def calibration_mode_for_baseline(baseline: str) -> str:
    """Map the public baseline name to its sealed calibration mode."""

    try:
        return BASELINE_VARIANT_TO_MODE[baseline]
    except KeyError as exc:
        raise ValueError(f"baseline must be one of {', '.join(BASELINE_VARIANTS)}") from exc


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
            "double_commutator_sum_float",
            "bound_method",
            "bound_equation",
            "bound_term_count",
            "bound_ordered_triple_space",
            "bound_roundoff_guard_relative",
        )
        if key in reference
    }
    return {
        "schema": payload.get("schema"),
        "problem": payload.get("problem"),
        "problem_sha256": payload.get("problem_sha256"),
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
    if str(policy.get("error_allocation", "")) != PAPER_ALLOCATION_METHOD:
        raise ValueError("paper baseline must use the supplement Eq. (34) allocation")
    if not math.isclose(
        float(policy.get("paper_synthesis_gamma", math.nan)),
        PAPER_SYNTHESIS_GAMMA,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("paper baseline has the wrong rigorous synthesis gamma")
    if not math.isclose(
        float(policy.get("paper_synthesis_delta_0", math.nan)),
        PAPER_SYNTHESIS_DELTA_0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("paper baseline has the wrong rigorous synthesis delta_0")
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
            if not isinstance(exact_reference, Mapping):
                raise ValueError("Hamiltonian-bound exact reference is malformed")
            if exact_reference.get("observable") != "exact_unitary_ground_eigenphase":
                raise ValueError("Hamiltonian-bound calibration has the wrong exact observable")
            if exact_reference.get("problem_sha256") != payload.get("problem_sha256"):
                raise ValueError(
                    "Hamiltonian-bound calibration exact reference has the wrong problem hash"
                )
            if exact_reference.get("phase_encoding") != payload.get("benchmark_adaptation", {}).get(
                "phase_encoding"
            ):
                raise ValueError(
                    "Hamiltonian-bound calibration exact reference has the wrong phase encoding"
                )
        reference = payload.get("reference")
        if not isinstance(reference, Mapping):
            raise ValueError("Hamiltonian-bound calibration is missing its reference data")
        if reference.get("bound_method") != "exact_vectorized_pauli_sum":
            raise ValueError("Hamiltonian-bound calibration is not the exact Eq. (19) sum")
        if reference.get("exact_ground_energy_hartree") is not None:
            raise ValueError("Hamiltonian-bound beta must not depend on an exact energy")
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
    updater = None
    matrices = []
    if backend_name == "cupy-lanczos":
        from hsopt.evaluation.gpu_rotations import RotationUpdater

        updater = RotationUpdater(optimized=True)
    for term, alpha in rotations:
        theta = -2.0 * float(alpha)
        exact_rz = np.asarray(
            [
                [np.exp(-0.5j * theta), 0.0],
                [0.0, np.exp(0.5j * theta)],
            ],
            dtype=np.complex128,
        )
        matrices.append(exact_rz)
        if updater is not None:
            unitary, workspace = updater(
                unitary, workspace, label=str(term["label"]), compiled_rz=exact_rz, xp=xp
            )
        else:
            unitary, workspace = _apply_compiled_rotation(
                unitary, workspace, label=str(term["label"]), compiled_rz=exact_rz, xp=xp
            )
    _fence(backend_name)
    del workspace

    if backend_name == "cupy-lanczos":
        from hsopt.evaluation.gpu_spectrum import eigensolve, phase_enclosure

        eigenvalues, eigenvectors, _diagnostics = eigensolve(
            unitary,
            problem,
            {"n_steps": n_steps, "compiled_phase_enclosure": phase_enclosure(matrices, n_steps)},
            iterative=True,
            target_state=exact_target,
        )
    else:
        eigenvalues, eigenvectors = xp.linalg.eig(unitary)
    _fence(backend_name)
    eigenvectors /= xp.linalg.norm(eigenvectors, axis=0, keepdims=True)
    overlaps = xp.abs(eigenvectors.conj().T @ exact_target) ** 2
    selected = int(_host_array(xp.argmax(overlaps)))
    step_phase = float(_host_array(xp.angle(eigenvalues[selected])))
    total_phase = int(n_steps) * step_phase
    identity = float(problem["identity_coefficient"])
    energy = _decode_energy(
        total_phase,
        target_time=target_time,
        identity_coefficient=identity,
    )
    selected_overlap = float(_host_array(overlaps[selected]))
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
    step_evaluator=None,
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
            evaluations[n_steps] = (step_evaluator or _exact_pf_energy)(
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
    ordered_terms=None,
) -> dict[str, Any]:
    """Compute the supplement's Eq. (19) bound exactly for Pauli terms.

    For Pauli strings, ``[P_beta, P_alpha_prime]`` is nonzero exactly when
    those strings anticommute.  The outer commutator is nonzero exactly when
    ``P_alpha`` anticommutes with one, but not both, of those strings.  This
    lets us sum the paper's ordered triples with matrix-vector products rather
    than materializing ``M**3`` triples or estimating them by Monte Carlo.
    """

    terms = (
        ordered_terms
        if ordered_terms is not None
        else sorted(problem["terms"], key=lambda term: str(term["label"]))
    )
    coefficients = np.asarray(
        [abs(float(term["coefficient"])) for term in terms],
        dtype=np.float64,
    )
    count = len(terms)
    anticommuting = _pauli_anticommutation_matrix([str(term["label"]) for term in terms])

    long_coefficients = coefficients.astype(np.longdouble)
    contributions: list[np.longdouble] = []
    for beta in range(count):
        beta_coefficient = long_coefficients[beta]

        # First Eq. (19) ordering term:
        # delta_{alpha>beta} delta_{alpha_prime>beta}.
        tail_coefficients = long_coefficients[beta + 1 :]
        if tail_coefficients.size:
            inner_weights = tail_coefficients * anticommuting[beta, beta + 1 :]
            outer_nonzero = np.logical_xor(
                anticommuting[beta + 1 :, beta, None],
                anticommuting[beta + 1 :, beta + 1 :],
            )
            contributions.append(
                np.longdouble(4.0)
                * beta_coefficient
                * np.dot(tail_coefficients, outer_nonzero @ inner_weights)
            )

        # Second Eq. (19) ordering term:
        # delta_{beta>alpha} delta_{alpha_prime,alpha}.
        if beta:
            leading_coefficients = long_coefficients[:beta]
            contributions.append(
                np.longdouble(4.0)
                * beta_coefficient
                * np.dot(
                    leading_coefficients * leading_coefficients,
                    anticommuting[:beta, beta],
                )
            )

    coefficient_sum = np.sum(np.asarray(contributions, dtype=np.longdouble))
    # The scientific bound is already highly conservative.  Inflate the
    # nonnegative floating-point sum by one part in 1e12, then round upward
    # once more when converting to JSON's binary64 representation.
    roundoff_guard_relative = np.longdouble("1e-12")
    guarded = max(
        np.longdouble(0.0),
        coefficient_sum * (np.longdouble(1.0) + roundoff_guard_relative),
    )
    upper = math.nextafter(float(guarded), math.inf) if guarded > 0.0 else 0.0

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
        "bound_method": "exact_vectorized_pauli_sum",
        "bound_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_19",
        "bound_term_count": count,
        "bound_ordered_triple_space": count**3,
        "bound_roundoff_guard_relative": float(roundoff_guard_relative),
        "double_commutator_sum_float": float(coefficient_sum),
        "double_commutator_sum_upper": float(upper),
        "exact_ground_energy_hartree": None,
    }


def _paper_eq34_rotation_error(
    *,
    terms: int,
    beta: float,
    total_error_hartree: float,
) -> tuple[float, dict[str, Any]]:
    """Solve the supplement Eq. (34) stationary condition.

    For fixed rotation error ``epsilon_rot``, the additive constraint and
    Eq. (34) give ``epsilon_QPE = 2 * epsilon_TS``.  The remaining scalar
    stationary equation has exactly one root in ``(0, epsilon_total)`` for
    the rigorous-row parameters used here.  Bisection avoids an optional
    Lambert-W dependency while solving that analytic condition directly.
    """

    if terms <= 0:
        raise ValueError("paper allocation requires at least one Hamiltonian term")
    if beta <= 0.0 or total_error_hartree <= 0.0:
        raise ValueError("paper allocation inputs must be positive")
    logarithm_scale = 2.0 * terms * beta
    shifted_log_constant = PAPER_SYNTHESIS_DELTA_0 * math.log(2.0) / PAPER_SYNTHESIS_GAMMA

    def stationarity(epsilon_rot: float) -> float:
        if epsilon_rot == 0.0:
            return -2.0 * total_error_hartree
        return (
            3.0 * epsilon_rot * (math.log(logarithm_scale / epsilon_rot) + shifted_log_constant)
            + 2.0 * epsilon_rot
            - 2.0 * total_error_hartree
        )

    lower = 0.0
    upper = math.nextafter(total_error_hartree, 0.0)
    lower_value = stationarity(lower)
    upper_value = stationarity(upper)
    if not lower_value < 0.0 < upper_value:
        raise RuntimeError("paper Eq. (34) stationary root is not bracketed")
    iterations = 0
    for iterations in range(1, 257):
        middle = 0.5 * (lower + upper)
        value = stationarity(middle)
        if value > 0.0:
            upper = middle
        else:
            lower = middle
        if upper - lower <= max(
            1.0e-15 * total_error_hartree,
            math.ulp(middle),
        ):
            break
    epsilon_rot = 0.5 * (lower + upper)
    return epsilon_rot, {
        "method": PAPER_ALLOCATION_METHOD,
        "paper_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_34",
        "constraint_equation": "Reiher_et_al_PNAS_2017_supplement_Eq_32",
        "stationary_equation": (
            "3*epsilon_Rot*(ln(2*M*beta/epsilon_Rot)"
            "+delta_0*ln(2)/gamma)+2*epsilon_Rot-2*epsilon_total=0"
        ),
        "root_solver": "deterministic_bisection_of_analytic_stationary_condition",
        "root_iterations": iterations,
        "root_residual_hartree": stationarity(epsilon_rot),
    }


def optimize_paper_error_allocation(
    problem: Mapping[str, Any],
    *,
    beta: float,
    reference_error_hartree: float,
    total_error_hartree: float,
    rotations_per_step: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the paper's rigorous Eq. (34) policy without compiling it."""

    if beta <= 0.0:
        raise ValueError("reference Trotter number beta must be positive")
    if reference_error_hartree <= 0.0 or total_error_hartree <= 0.0:
        raise ValueError("error targets must be positive")
    terms = len(problem["terms"])
    epsilon_rot, derivation = _paper_eq34_rotation_error(
        terms=terms,
        beta=beta,
        total_error_hartree=total_error_hartree,
    )
    residual = total_error_hartree - epsilon_rot
    epsilon_ts = residual / 3.0
    epsilon_qpe = 2.0 * residual / 3.0
    target_time = float(problem["target_time"])
    n_steps = max(
        1,
        math.ceil(target_time * float(beta) * math.sqrt(reference_error_hartree / epsilon_ts)),
    )
    step_time = target_time / n_steps
    rotations_per_step = (2 * terms - 1) if rotations_per_step is None else rotations_per_step
    if rotations_per_step < 1:
        raise ValueError("positive rotation count required")
    uniform_epsilon = epsilon_rot * step_time / rotations_per_step
    qpe_full_evolution_queries = max(
        1,
        math.ceil(PAPER_PHASE_ESTIMATION_ALPHA / (target_time * epsilon_qpe)),
    )
    qpe_step_queries = qpe_full_evolution_queries * n_steps
    paper_synthesis_t_per_rotation = (
        PAPER_SYNTHESIS_GAMMA * math.log2((2.0 * terms * float(beta)) / epsilon_rot)
        + PAPER_SYNTHESIS_DELTA_0
    )
    paper_eq34_cost_proxy = (
        2.0
        * terms
        * (PAPER_PHASE_ESTIMATION_ALPHA / epsilon_qpe)
        * (float(beta) * math.sqrt(reference_error_hartree / epsilon_ts))
        * paper_synthesis_t_per_rotation
    )
    policy = {
        "error_allocation": PAPER_ALLOCATION_METHOD,
        "paper_error_constraint": PAPER_ERROR_CONSTRAINT,
        "paper_phase_estimation_alpha": PAPER_PHASE_ESTIMATION_ALPHA,
        "paper_synthesis_gamma": PAPER_SYNTHESIS_GAMMA,
        "paper_synthesis_delta_0": PAPER_SYNTHESIS_DELTA_0,
        "paper_modeled_t_per_rotation": paper_synthesis_t_per_rotation,
        "paper_eq34_cost_proxy": paper_eq34_cost_proxy,
        "qpe_fraction": epsilon_qpe / total_error_hartree,
        "trotter_fraction": epsilon_ts / total_error_hartree,
        "rotation_fraction": epsilon_rot / total_error_hartree,
        "phase_estimation_error_hartree": epsilon_qpe,
        "trotter_error_hartree": epsilon_ts,
        "rotation_synthesis_energy_error_hartree": epsilon_rot,
        "error_constraint_value_hartree": epsilon_qpe + epsilon_ts + epsilon_rot,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "formula_order": PAPER_FORMULA_ORDER,
        "n_steps": n_steps,
        "nonidentity_terms": terms,
        "qpe_full_evolution_queries": qpe_full_evolution_queries,
        "qpe_step_queries": qpe_step_queries,
        "reference_error_hartree": float(reference_error_hartree),
        "reference_trotter_number_beta": float(beta),
        "rotation_synthesis": "clifford_plus_t_cudaq_at_iteration_zero",
        "realized_t_count": None,
        "realized_t_count_stage": "iteration_0_exact_compiled_unitary_evaluation",
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
    derivation.update(
        {
            "epsilon_qpe_hartree": epsilon_qpe,
            "epsilon_ts_hartree": epsilon_ts,
            "epsilon_rot_hartree": epsilon_rot,
            "paper_eq34_cost_proxy": paper_eq34_cost_proxy,
        }
    )
    return policy, derivation


def calibrate_problem(
    problem_path: Path,
    *,
    backend: str,
    mode: str = "hamiltonian_bound",
    include_exact_reference: bool = True,
    target_time: float | None = None,
    reference_error_hartree: float = PAPER_BETA_REFERENCE_ERROR_HARTREE,
    total_error_hartree: float = BENCHMARK_TOTAL_ERROR_HARTREE,
    max_steps: int = 1 << 20,
    precomputed_exact_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate beta(H), allocate errors, and instantiate one immutable PF2."""

    started = time.perf_counter()
    path = problem_path.resolve()
    problem = load_problem(path)
    if target_time is not None and not math.isclose(
        float(problem["target_time"]),
        float(target_time),
        rel_tol=1.0e-12,
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

    exact_reference: dict[str, Any] | None = None
    exact_started = time.perf_counter()
    if include_exact_reference:
        if precomputed_exact_reference is None:
            exact_reference = evaluate_exact_reference(path, backend=backend_name)
        else:
            exact_reference = dict(precomputed_exact_reference)
            if exact_reference.get("observable") != "exact_unitary_ground_eigenphase":
                raise ValueError("precomputed exact reference has the wrong observable")
            if (
                exact_reference.get("problem_sha256")
                != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError("precomputed exact reference has the wrong problem hash")
            if not math.isclose(
                float(exact_reference.get("target_time", math.nan)),
                float(problem["target_time"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("precomputed exact reference has the wrong target time")
    elif precomputed_exact_reference is not None:
        raise ValueError("cannot reuse an exact reference when exact references are disabled")
    exact_reference_seconds = time.perf_counter() - exact_started

    beta_started = time.perf_counter()
    if mode == "empirical_exact":
        if exact_reference is None:
            raise ValueError("empirical_exact mode requires an exact reference")
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
        )
    else:
        raise ValueError("mode must be 'empirical_exact' or 'hamiltonian_bound'")
    beta_seconds = time.perf_counter() - beta_started

    allocation_started = time.perf_counter()
    policy, allocation_derivation = optimize_paper_error_allocation(
        problem,
        beta=float(reference["reference_trotter_number_beta"]),
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
            "synthesis_gamma": PAPER_SYNTHESIS_GAMMA,
            "synthesis_delta_0": PAPER_SYNTHESIS_DELTA_0,
            "allocation_equation": "supplement_Eq_34",
            "synthesis_energy_bound": "(2M - 1) * Delta_synth / step_time",
        },
        "benchmark_adaptation": {
            "total_error_target_hartree": float(total_error_hartree),
            "total_error_target_kcal_per_mol": float(
                total_error_hartree / CHEMICAL_ACCURACY_HARTREE
            ),
            "target_time": float(problem["target_time"]),
            "phase_encoding": dict(problem["phase_encoding"]),
            "uses_gamma_delta_synthesis_fit": True,
            "allocation_cost_source": "paper Eq. (34) rigorous gamma/delta_0 model",
            "realized_t_count_source": "iteration-0 direct CUDA-Q compilation",
        },
        "reference": reference,
        "exact_reference": exact_reference,
        "exact_reference_reused": bool(
            include_exact_reference and precomputed_exact_reference is not None
        ),
        "instantiated_policy": policy,
        "allocation_derivation": allocation_derivation,
        "timing_seconds": {
            "exact_reference": exact_reference_seconds,
            "beta_calibration": beta_seconds,
            "paper_allocation": allocation_seconds,
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


def calibration_main() -> int:
    """CLI for producing one sealed Hamiltonian calibration record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--backend", choices=("numpy", "cupynumeric", "cupy-lanczos"), default="numpy"
    )
    parser.add_argument(
        "--mode",
        choices=("empirical_exact", "hamiltonian_bound"),
        default="hamiltonian_bound",
    )
    parser.add_argument(
        "--without-exact-reference",
        action="store_true",
        help=(
            "derive the initial PF without E0; this output cannot initialize an "
            "autoresearch campaign until an exact evaluator reference is supplied"
        ),
    )
    parser.add_argument(
        "--target-time",
        type=float,
        help=(
            "optional assertion for the per-Hamiltonian normalized evolution time; "
            "by default it is read from phase_encoding"
        ),
    )
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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = calibrate_problem(
        args.problem,
        backend=args.backend,
        mode=args.mode,
        include_exact_reference=not args.without_exact_reference,
        target_time=args.target_time,
        reference_error_hartree=args.reference_error_hartree,
        total_error_hartree=args.total_error_hartree,
        max_steps=args.max_steps,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != encoded:
            raise SystemExit(f"paper baseline calibration is stale: {args.output}")
        return 0
    _write_json(args.output.resolve(), result)
    print(encoded, end="")
    return 0


# Sealed baseline contracts and parameter preparation for the public generator.

COMMON_PAPER_CONTRACT: dict[str, Any] = {
    "schema": 7,
    "citation": {
        "authors": [
            "Markus Reiher",
            "Nathan Wiebe",
            "Krysta M. Svore",
            "Dave Wecker",
            "Matthias Troyer",
        ],
        "title": "Elucidating Reaction Mechanisms on Quantum Computers",
        "journal": "Proceedings of the National Academy of Sciences",
        "year": 2017,
        "doi": "10.1073/pnas.1619152114",
        "arxiv": "1605.03590",
    },
    "paper_method": {
        "formula_order": PAPER_FORMULA_ORDER,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "term_order_strategy": PAPER_TERM_ORDER,
        "rotation_synthesis": "clifford_plus_t_cudaq_at_iteration_zero",
        "beta_reference_error_hartree": PAPER_BETA_REFERENCE_ERROR_HARTREE,
        "phase_estimation_alpha": PAPER_PHASE_ESTIMATION_ALPHA,
        "synthesis_gamma": PAPER_SYNTHESIS_GAMMA,
        "synthesis_delta_0": PAPER_SYNTHESIS_DELTA_0,
        "allocation_equation": "supplement Eq. (34)",
        "error_constraint": PAPER_ERROR_CONSTRAINT,
        "synthesis_energy_bound": "(2M - 1) * Delta_synth / step_time",
    },
    "benchmark_adaptation": {
        "total_error_target_hartree": BENCHMARK_TOTAL_ERROR_HARTREE,
        "total_error_target": "1 kcal/mol",
        "phase_encoding": "centered_nonidentity_pauli_one_norm",
        "dimensionless_qpe_evolution_time": 1.0,
        "physical_target_time": ("inverse of the nonidentity Pauli one-norm, in Hartree^-1"),
        "energy_recovery": "E_hartree = c0 + lambda * principal_phase",
        "allocation_cost": "supplement Eq. (34) with rigorous gamma=4 and delta_0=11",
        "realized_t_count": "direct CUDA-Q compilation during iteration 0",
    },
}
BASELINE_CONTRACTS: dict[str, dict[str, Any]] = {
    "loose": {
        "baseline_id": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
        "calibration_mode": "hamiltonian_bound",
        "display_name": "Reiher et al. Eq. (19)/(34) rigorous PF2",
        "beta_selection": "exact Eq. (19) Pauli double-commutator coefficient bound",
        "scope_note": (
            "The exact energy is cached for evaluation but does not enter beta, "
            "the Trotter step count, or the synthesis tolerance."
        ),
    },
    "tight": {
        "baseline_id": "reiher_wiebe_svore_wecker_troyer_2017_numeric_beta_eq34_pf2",
        "calibration_mode": "empirical_exact",
        "display_name": "Reiher et al. LIQUi|>-style numeric-beta/Eq. (34) PF2 (tight)",
        "beta_selection": (
            "first passing lexicographic PF2 step count against the exact ground energy at 0.1 mHa"
        ),
        "scope_note": (
            "The cached exact ground energy selects the numeric beta calibration; "
            "the paper Eq. (34) allocation and all evaluator semantics remain unchanged."
        ),
    },
}


def paper_contract(baseline: str) -> dict[str, Any]:
    """Return the immutable scientific contract for one public baseline."""

    mode = calibration_mode_for_baseline(baseline)
    variant = BASELINE_CONTRACTS[baseline]
    if variant["calibration_mode"] != mode:
        raise AssertionError(f"inconsistent baseline contract for {baseline}")
    contract = json.loads(json.dumps(COMMON_PAPER_CONTRACT))
    contract.update(variant)
    contract["paper_method"]["beta_selection"] = variant["beta_selection"]
    return contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _problem_paths(
    source: Path,
) -> list[tuple[str | None, float | None, Path, dict[str, Any]]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "n_qubits" in payload and "terms" in payload:
        return [(None, None, source, payload)]
    rows: list[tuple[str | None, float | None, Path, dict[str, Any]]] = []
    for item in payload["terms"]:
        problem_path = (source.parent / str(item["problem"])).resolve()
        problem = json.loads(problem_path.read_text(encoding="utf-8"))
        rows.append(
            (
                str(item["species"]),
                float(item["coefficient"]),
                problem_path,
                problem,
            )
        )
    if not rows:
        raise ValueError(f"{source}: reaction has no Hamiltonians")
    return rows


def refresh_synthesis_provenance(
    payload: Mapping[str, Any], *, problem_path: Path
) -> dict[str, Any]:
    """Reseal a legacy allocation for CUDA-Q without repeating its beta calibration.

    The old reference and every numerical policy parameter must remain equal.
    The returned copy records its parent checksum; the historical file is untouched.
    """
    verify_calibration(payload, problem_path=problem_path)
    problem = json.loads(problem_path.read_text(encoding="utf-8"))
    policy, derivation = optimize_paper_error_allocation(
        problem,
        beta=float(payload["reference"]["reference_trotter_number_beta"]),
        reference_error_hartree=float(payload["reference"]["reference_error_hartree"]),
        total_error_hartree=float(payload["benchmark_adaptation"]["total_error_target_hartree"]),
    )
    old = dict(payload["instantiated_policy"])
    expected = dict(policy)
    old.pop("rotation_synthesis", None)
    expected.pop("rotation_synthesis", None)
    if old != expected:
        raise ValueError(
            "legacy allocation has changed numerical parameters; recalibration required"
        )
    updated = json.loads(json.dumps(payload))
    updated["parent_calibration_sha256"] = payload["calibration_sha256"]
    updated["instantiated_policy"] = policy
    updated["allocation_derivation"] = derivation
    updated["benchmark_adaptation"]["realized_t_count_source"] = (
        "iteration-0 direct CUDA-Q compilation"
    )
    return seal_calibration(updated)


def _read_calibration(path: Path, *, baseline: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_calibration(payload)
    expected_mode = calibration_mode_for_baseline(baseline)
    if payload["calibration_mode"] != expected_mode:
        raise ValueError(
            f"{path}: {baseline} requires {expected_mode}, found {payload['calibration_mode']}"
        )
    return payload


def _calibration_catalog(
    root: Path,
    *,
    baseline: str,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    expected_mode = calibration_mode_for_baseline(baseline)
    catalog: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            verify_calibration(payload)
            if payload["calibration_mode"] != expected_mode:
                continue
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        problem_sha256 = str(payload["problem_sha256"])
        if problem_sha256 in catalog:
            prior_path, prior = catalog[problem_sha256]
            if str(prior["calibration_sha256"]) != str(payload["calibration_sha256"]):
                raise ValueError(
                    f"conflicting calibrations for {problem_sha256}: {prior_path} and {path}"
                )
            continue
        catalog[problem_sha256] = (path, payload)
    if not catalog:
        raise ValueError(f"no sealed {baseline} ({expected_mode}) calibrations found under {root}")
    return catalog


def _policy_row(
    *,
    problem_path: Path,
    problem: Mapping[str, Any],
    calibration_path: Path,
    calibration: Mapping[str, Any],
    species: str | None,
    reaction_coefficient: float | None,
    reaction_species_count: int,
    baseline: str,
) -> tuple[str, dict[str, Any]]:
    verify_calibration(calibration, problem_path=problem_path)
    expected_mode = calibration_mode_for_baseline(baseline)
    if calibration["calibration_mode"] != expected_mode:
        raise ValueError(f"{calibration_path}: not a {baseline} baseline")
    if reaction_coefficient is None:
        species_error_target = BENCHMARK_TOTAL_ERROR_HARTREE
    else:
        if reaction_species_count <= 0 or reaction_coefficient == 0.0:
            raise ValueError("reaction calibration requires nonzero signed terms")
        species_error_target = BENCHMARK_TOTAL_ERROR_HARTREE / (
            reaction_species_count * abs(reaction_coefficient)
        )
    policy, _derivation = optimize_paper_error_allocation(
        problem,
        beta=float(calibration["reference"]["reference_trotter_number_beta"]),
        reference_error_hartree=float(calibration["reference"]["reference_error_hartree"]),
        total_error_hartree=species_error_target,
    )
    if reaction_coefficient is None and policy != calibration["instantiated_policy"]:
        raise ValueError(
            f"{calibration_path}: generator Eq. (34) policy does not match "
            "the sealed calibration policy"
        )
    if reaction_coefficient is not None:
        policy["reaction_coefficient"] = reaction_coefficient
        policy["reaction_species_count"] = reaction_species_count
        policy["reaction_error_allocation"] = (
            "equal_absolute_contribution_under_triangle_inequality"
        )
        policy["reaction_total_error_target_hartree"] = BENCHMARK_TOTAL_ERROR_HARTREE
    if not math_isclose(float(problem["target_time"]), float(policy["target_time"])):
        raise ValueError(
            f"{problem_path}: source target_time={problem['target_time']} differs from "
            f"calibrated target_time={policy['target_time']}"
        )
    name = str(problem["name"])
    contract = paper_contract(baseline)
    policy.update(
        {
            "baseline_variant": baseline,
            "baseline_source": str(contract["baseline_id"]),
            "calibration_mode": str(calibration["calibration_mode"]),
            "calibration_sha256": str(calibration["calibration_sha256"]),
            "problem_sha256": str(calibration["problem_sha256"]),
            "reference_trotter_number_beta": float(
                calibration["reference"]["reference_trotter_number_beta"]
            ),
            "reference_beta_error_hartree": float(
                calibration["reference"]["reference_error_hartree"]
            ),
        }
    )
    reference = calibration["reference"]
    if baseline == "loose":
        policy.update(
            {
                "bound_method": str(reference["bound_method"]),
                "bound_equation": str(reference["bound_equation"]),
                "bound_term_count": int(reference["bound_term_count"]),
                "double_commutator_sum_upper": float(reference["double_commutator_sum_upper"]),
            }
        )
    else:
        policy.update(
            {
                "numeric_reference_step_count": int(reference["reference_step_count"]),
                "numeric_reference_step_time": float(reference["reference_step_time"]),
                "numeric_reference_energy_hartree": float(reference["exact_ground_energy_hartree"]),
                "numeric_reference_error_hartree": float(
                    reference["selected_evaluation"]["absolute_trotter_error_hartree"]
                ),
            }
        )
    if species is not None:
        policy["species"] = species
    return name, policy


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-15 * max(1.0, abs(left), abs(right))


def load_baselines(
    source: Path,
    *,
    baseline: str,
    calibration_path: Path | None,
    calibration_root: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    problems = _problem_paths(source)
    if calibration_path is not None:
        if len(problems) != 1:
            raise ValueError("--calibration can only be used with one Hamiltonian")
        direct = _read_calibration(calibration_path, baseline=baseline)
        catalog = {str(direct["problem_sha256"]): (calibration_path, direct)}
    elif calibration_root is not None:
        catalog = _calibration_catalog(calibration_root, baseline=baseline)
    else:
        raise ValueError("provide --calibration or --calibration-root")

    rows: dict[str, dict[str, Any]] = {}
    used: list[Path] = []
    reaction_species_count = sum(species is not None for species, *_rest in problems)
    for species, reaction_coefficient, problem_path, problem in problems:
        digest = _sha256(problem_path)
        if digest not in catalog:
            raise ValueError(f"no calibration for {problem_path} (SHA-256 {digest})")
        resolved_calibration, calibration = catalog[digest]
        name, row = _policy_row(
            problem_path=problem_path,
            problem=problem,
            calibration_path=resolved_calibration,
            calibration=calibration,
            species=species,
            reaction_coefficient=reaction_coefficient,
            reaction_species_count=reaction_species_count,
            baseline=baseline,
        )
        if name in rows:
            raise ValueError(f"duplicate evaluator-visible problem name {name!r}")
        rows[name] = row
        used.append(resolved_calibration)
    return rows, used
