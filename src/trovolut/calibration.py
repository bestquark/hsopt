"""
Calibration infrastructure for comparing emulator predictions against hardware.

Provides:

- :class:`CalibrationResult` -- paired (emulated, actual) data with linear
  regression coefficients for time and memory correction.
- :class:`CalibrationSuite` -- runs paired benchmarks and fits correction factors.
- :class:`CorrectedCostProfile` -- wraps a ``CostProfile`` and applies
  calibration corrections.
- :func:`standard_benchmarks` -- generates a default set of benchmark workloads.
- :func:`calibrate` -- convenience entry point.

Example::

    from trovolut.calibration import calibrate

    result = calibrate()                       # emulator-only mode
    result = calibrate(save_path="cal.json")   # persist to disk
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================
# CalibrationResult
# ============================================================

@dataclass
class CalibrationResult:
    """Paired emulator-vs-actual data with linear regression coefficients.

    The correction model is::

        actual ≈ slope * emulated + intercept

    for both time and memory.
    """

    # Raw paired data
    emulated_times: List[float] = field(default_factory=list)
    actual_times: List[float] = field(default_factory=list)
    emulated_memory: List[float] = field(default_factory=list)
    actual_memory: List[float] = field(default_factory=list)

    # Linear regression coefficients
    time_slope: float = 1.0
    time_intercept: float = 0.0
    memory_slope: float = 1.0
    memory_intercept: float = 0.0

    # Goodness of fit
    time_r2: float = 1.0
    memory_r2: float = 1.0

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "emulated_times": self.emulated_times,
            "actual_times": self.actual_times,
            "emulated_memory": self.emulated_memory,
            "actual_memory": self.actual_memory,
            "time_slope": self.time_slope,
            "time_intercept": self.time_intercept,
            "memory_slope": self.memory_slope,
            "memory_intercept": self.memory_intercept,
            "time_r2": self.time_r2,
            "memory_r2": self.memory_r2,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CalibrationResult":
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls(
            emulated_times=d["emulated_times"],
            actual_times=d["actual_times"],
            emulated_memory=d["emulated_memory"],
            actual_memory=d["actual_memory"],
            time_slope=d["time_slope"],
            time_intercept=d["time_intercept"],
            memory_slope=d["memory_slope"],
            memory_intercept=d["memory_intercept"],
            time_r2=d["time_r2"],
            memory_r2=d["memory_r2"],
        )


# ============================================================
# CorrectedCostProfile
# ============================================================

class CorrectedCostProfile:
    """Wraps a ``CostProfile`` and applies calibration corrections.

    Delegates all attribute access to the underlying profile except
    for ``corrected_time()`` and ``corrected_memory()``.
    """

    def __init__(self, profile, calibration: CalibrationResult):
        self._profile = profile
        self._calibration = calibration

    def corrected_time(self, hardware) -> float:
        """Apply calibration correction to emulated time."""
        raw = self._profile.emulated_time(hardware)
        cal = self._calibration
        return cal.time_slope * raw + cal.time_intercept

    def corrected_memory(self) -> float:
        """Apply calibration correction to peak memory estimate."""
        raw = float(self._profile.peak_memory_bytes)
        cal = self._calibration
        return cal.memory_slope * raw + cal.memory_intercept

    def __getattr__(self, name):
        return getattr(self._profile, name)


# ============================================================
# CalibrationSuite
# ============================================================

def _fit_linear(x: List[float], y: List[float]):
    """Fit y = slope * x + intercept via numpy.polyfit.

    Returns (slope, intercept, r2).
    """
    if len(x) < 2:
        return 1.0, 0.0, 1.0
    xa, ya = np.array(x), np.array(y)
    coeffs = np.polyfit(xa, ya, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    # R²
    y_pred = slope * xa + intercept
    ss_res = float(np.sum((ya - y_pred) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


class CalibrationSuite:
    """Runs paired benchmarks (emulator + optional real GPU) and fits corrections.

    Parameters
    ----------
    workloads : list[dict]
        Each dict has ``"hamiltonian"``, ``"dynamics"``, and optionally
        ``"kwargs"`` for ``cost_profile()``.
    hardware
        Hardware specification for ``emulated_time()``
        (e.g. ``gpus.A100``).
    hardware_metrics : callable, optional
        ``(dynamics, ham) -> dict`` returning real GPU metrics with keys
        ``"time_seconds"`` and ``"peak_memory_bytes"``.  When ``None``,
        actual values equal emulated values (emulator-only mode).
    """

    def __init__(
        self,
        workloads: List[Dict[str, Any]],
        hardware,
        hardware_metrics=None,
    ):
        self.workloads = workloads
        self.hardware = hardware
        self.hardware_metrics = hardware_metrics

    def run(self) -> CalibrationResult:
        """Iterate workloads, collect costs, fit regression."""
        emulated_times: List[float] = []
        actual_times: List[float] = []
        emulated_memory: List[float] = []
        actual_memory: List[float] = []

        for wl in self.workloads:
            dyn = wl["dynamics"]
            ham = wl["hamiltonian"]
            kwargs = wl.get("kwargs", {})

            try:
                profile = dyn.cost_profile(ham, **kwargs)
            except Exception:
                continue

            et = profile.emulated_time(self.hardware)
            em = float(profile.peak_memory_bytes)

            if self.hardware_metrics is not None:
                try:
                    actual = self.hardware_metrics(dyn, ham)
                    at = float(actual.get("time_seconds", et))
                    am = float(actual.get("peak_memory_bytes", em))
                except Exception:
                    at, am = et, em
            else:
                # Emulator-only mode: actual = emulated
                at, am = et, em

            emulated_times.append(et)
            actual_times.append(at)
            emulated_memory.append(em)
            actual_memory.append(am)

        # Fit linear regression
        ts, ti, tr2 = _fit_linear(emulated_times, actual_times)
        ms, mi, mr2 = _fit_linear(emulated_memory, actual_memory)

        return CalibrationResult(
            emulated_times=emulated_times,
            actual_times=actual_times,
            emulated_memory=emulated_memory,
            actual_memory=actual_memory,
            time_slope=ts,
            time_intercept=ti,
            memory_slope=ms,
            memory_intercept=mi,
            time_r2=tr2,
            memory_r2=mr2,
        )


# ============================================================
# Standard benchmarks factory
# ============================================================

def standard_benchmarks() -> List[Dict[str, Any]]:
    """Generate a default set of benchmark workloads.

    Varies matrix size (4, 8, 16, 32), precision (FP64, FP32, FP16),
    decomposition method (exact, taylor), and step count (1, 10, 50).

    Returns a list of workload dicts suitable for :class:`CalibrationSuite`.
    """
    from trovolut.backends.gpu_mp_emulator.dynamics import (
        MPTrotterEvolution,
    )
    from trovolut.backends.gpu_mp_emulator.utils import Precision

    sizes = [4, 8, 16, 32]
    precisions = [Precision.FP64, Precision.FP32, Precision.FP16]
    decompositions = ["exact", "taylor"]
    step_counts = [1, 10, 50]

    workloads = []
    for n in sizes:
        # Random Hermitian Hamiltonian (fixed seed for reproducibility)
        rng = np.random.default_rng(42 + n)
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2

        for prec in precisions:
            for decomp in decompositions:
                for steps in step_counts:
                    dyn = MPTrotterEvolution(
                        trotter_steps=steps,
                        precision_schedule=prec,
                        decomposition_schedule=decomp,
                    )
                    workloads.append({
                        "hamiltonian": H,
                        "dynamics": dyn,
                        "kwargs": {},
                    })

    return workloads


# ============================================================
# Convenience entry point
# ============================================================

def calibrate(
    hardware=None,
    workloads: Optional[List[Dict[str, Any]]] = None,
    hardware_metrics=None,
    save_path: Optional[str] = None,
) -> CalibrationResult:
    """Run calibration and optionally save results.

    Parameters
    ----------
    hardware
        Hardware spec for emulated_time().  Defaults to ``gpus.A100``.
    workloads : list, optional
        Custom workloads.  Defaults to :func:`standard_benchmarks`.
    hardware_metrics : callable, optional
        Real GPU metrics collector.  When ``None``, runs in emulator-only
        mode (actual = emulated).
    save_path : str, optional
        If provided, save JSON calibration result to this path.

    Returns
    -------
    CalibrationResult
    """
    if hardware is None:
        from trovolut.backends.gpu_mp_emulator.hardware import gpus
        hardware = gpus.A100

    if workloads is None:
        workloads = standard_benchmarks()

    suite = CalibrationSuite(
        workloads=workloads,
        hardware=hardware,
        hardware_metrics=hardware_metrics,
    )
    result = suite.run()

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(result.to_json())

    return result


__all__ = [
    "CalibrationResult",
    "CalibrationSuite",
    "CorrectedCostProfile",
    "standard_benchmarks",
    "calibrate",
]
