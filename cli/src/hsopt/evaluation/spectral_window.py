"""One-sided numerical rejection certificates for literal compiled QPE.

For A_m(U) = m^-1 sum_{k=0}^{m-1} exp(-ik theta0) U^k, any spectral
weight inside |theta-theta0| <= delta contributes at least
cos((m-1) delta/2)^2 to ||A_m(U) psi||^2. Thus the latter norm gives an
UPPER bound on accurate-peak weight. This can reject the fixed majority-peak
contract; it never supplies an energy, T95, or an acceptance certificate.
"""

import math
import time

import numpy as np


VERSION = "literal_qpe_window_upper_bound_v1"


class CertifiedGroundInputInfeasible(RuntimeError):
    """No majority-weight peak can satisfy the existing energy contract."""

    def __init__(self, certificate):
        self.certificate = certificate
        super().__init__(
            "Certified absence of an accurate majority-weight QPE peak: "
            f"spectral weight <= {certificate['accurate_window_probability_upper_bound']:.12g}; "
            "the contract requires > 0.5. No eigenenergy was fabricated."
        )


def _gamma(operations):
    product = operations * np.finfo(np.float64).eps
    if not 0 <= product < 0.01:
        raise ValueError("floating-point allowance exceeds the supported range")
    return product / (1 - product)


def certify_infeasible(unitary, problem, metrics, target_state, reference_energy,
                       *, orders=(8, 16, 32, 64, 128, 256), timeout_seconds=30):
    """Return a conservative rejection certificate or None; never guess a peak.

    The literal circuit supplies a per-action roundoff allowance. A missing
    allowance, norm drift inconsistent with it, a wide/aliased window or an
    insufficient bound remains unresolved. Original Lanczos checks still own
    all passing energies and probabilities.
    """
    import cupy as cp
    from . import compiled_unitary as ev

    action_error = getattr(unitary, "action_error_bound", None)
    if action_error is None or not math.isfinite(action_error) or not 0 <= action_error < 0.01:
        return None
    repetitions = int(metrics["n_steps"])
    target_time = float(problem["target_time"])
    if repetitions < 1 or not target_time > 0 or not math.isfinite(reference_energy):
        raise ValueError("spectral window needs a finite reference and positive evolution")
    theta = (reference_energy - problem["identity_coefficient"]) * target_time / repetitions
    reference_allowance = 1e-9
    delta = (ev.CHEMICAL_ACCURACY_HARTREE + reference_allowance) * target_time / repetitions
    if abs(theta) + delta >= math.pi / repetitions:
        return None
    orders = tuple(orders)
    if not orders or any(type(m) is not int or m < 2 for m in orders):
        raise ValueError("filter orders must be integers >= 2")
    v = target_state.copy()
    input_norm = float(cp.linalg.norm(v).get())
    if not math.isfinite(input_norm) or abs(input_norm - 1.0) > 1e-9:
        raise ValueError("spectral window requires the normalized sealed input")
    v /= input_norm
    accumulated = cp.zeros_like(v)
    phase = complex(math.cos(theta), -math.sin(theta))
    started = time.monotonic()
    norm_allowance = _gamma(16 * int(v.size) + 64)
    trials = []
    for order in range(1, max(orders) + 1):
        accumulated += v
        if order in orders:
            angle_radius = (order - 1) * delta / 2
            if angle_radius >= math.pi / 2:
                return None
            norm_v = float(cp.linalg.norm(v).get())
            propagated = math.expm1((order - 1) * math.log1p(action_error))
            if abs(norm_v - 1) > propagated + norm_allowance + 1e-12:
                return None
            amplitude = float(cp.linalg.norm(accumulated).get()) / order
            arithmetic_allowance = propagated + _gamma(64 * order) + norm_allowance
            lower = math.cos(angle_radius) ** 2
            upper = min(1.0, ((amplitude + arithmetic_allowance) ** 2) / lower)
            trial = {"order": order, "filtered_norm_squared": amplitude ** 2,
                     "filter_window_lower_bound": lower,
                     "amplitude_roundoff_allowance": arithmetic_allowance,
                     "accurate_window_probability_upper_bound": upper}
            trials.append(trial)
            if upper < 0.5 - 1e-9:
                return {
                    "method": VERSION, "verified": True,
                    "constraint": "accurate_majority_ground_input_qpe_peak",
                    "accurate_window_probability_upper_bound": upper,
                    "required_peak_probability_strictly_greater_than": 0.5,
                    "chemical_accuracy_hartree": ev.CHEMICAL_ACCURACY_HARTREE,
                    "reference_energy_hartree": reference_energy,
                    "reference_energy_allowance_hartree": reference_allowance,
                    "step_phase_center": theta, "step_phase_halfwidth": delta,
                    "n_steps": repetitions, "unitary_action_roundoff_bound": action_error,
                    "matvec_calls": order - 1, "seconds": time.monotonic() - started,
                    "trials": trials, "acceptance_permitted": False,
                    "full_diagonalization_permitted": False,
                }
        if time.monotonic() - started >= timeout_seconds:
            return None
        if order < max(orders):
            v = phase * (unitary @ v)
    return None
