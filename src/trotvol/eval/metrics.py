# metrics.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MetricResult:
    # whatever the metric wants to report after aggregation
    summary: Dict[str, float]

    # scalar to feed into global objective (already "the lower the better"
    # or "the higher the better", normalized etc.)
    scalar: float


class Metric:
    name: str

    def accumulate_single(self, payload: Dict) -> float:
        """Extract per-trial scalar from a single trial payload.
        Must NOT throw. Should return a fallback if missing."""
        raise NotImplementedError

    def aggregate(self, vals: List[float]) -> MetricResult:
        """Reduce across trials and return MetricResult.
        You decide what goes in summary + the scalar that goes to combiner."""
        raise NotImplementedError


# ---------------- Specific metrics ----------------


class ErrorMetric(Metric):
    """Numerical error vs exact evolution (||psi_final - psi_exact||). Lower is better."""

    name = "err"

    def accumulate_single(self, payload: Dict) -> float:
        return float(payload.get("error", 1e9))

    def aggregate(self, vals: List[float]) -> MetricResult:
        if not vals:
            vals = [1e9]
        mean_err = float(np.mean(vals))
        min_err = float(np.min(vals))
        # global scalar: mean error (lower is better)
        return MetricResult(
            summary={"err_mean": mean_err, "err_min": min_err},
            scalar=mean_err,
        )


class NormDeviationMetric(Metric):
    """How non-unitary you were: | ||psi|| - 1 | . Lower is better."""

    name = "norm_dev"

    def accumulate_single(self, payload: Dict) -> float:
        # payload["norm"] ~ ||psi_final||
        nrm = float(payload.get("norm", 0.0))
        return abs(nrm - 1.0) if nrm != 0.0 else 1e9

    def aggregate(self, vals: List[float]) -> MetricResult:
        if not vals:
            vals = [1e9]
        mean_dev = float(np.mean(vals))
        return MetricResult(
            summary={"norm_dev": mean_dev},
            scalar=mean_dev,
        )


class MemMetric(Metric):
    """Memory footprint in MB. Lower is better.
    We also allow normalization against a cap."""

    def __init__(self, cap_mb: float):
        self.cap_mb = cap_mb
        self.name = "mem"

    def accumulate_single(self, payload: Dict) -> float:
        # prefer GPU mem
        mem_mb = payload.get("gpu_mem_mb", payload.get("mem_mb", self.cap_mb))
        try:
            return float(mem_mb)
        except Exception:
            return self.cap_mb

    def aggregate(self, vals: List[float]) -> MetricResult:
        if not vals:
            vals = [self.cap_mb]
        mean_mem = float(np.mean(vals))
        mem_norm = min(mean_mem, self.cap_mb) / self.cap_mb  # in [0,1+] clipped
        return MetricResult(
            summary={
                "mem_raw_mean_mb": mean_mem,
                "mem_norm": mem_norm,
            },
            scalar=mem_norm,  # smaller is better (<=1 good, >1 bad but clipped)
        )


class GpuTimeMetric(Metric):
    """Runtime in ms. Lower is better.
    We fold GPU time if present, else wall time."""

    name = "time_ms"

    def accumulate_single(self, payload: Dict) -> float:
        # prefer GPU profiling time if present
        t_ms = payload.get("gpu_time_ms", payload.get("time_ms", 1e9))
        try:
            return float(t_ms)
        except Exception:
            return 1e9

    def aggregate(self, vals: List[float]) -> MetricResult:
        if not vals:
            vals = [1e9]
        mean_ms = float(np.mean(vals))
        # We often want a "soft" penalty like log(1+t)
        soft = math.log1p(mean_ms / 1000.0)
        return MetricResult(
            summary={"time_ms_mean": mean_ms},
            scalar=soft,  # smaller is better
        )
