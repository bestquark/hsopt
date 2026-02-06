"""
GPU-backend constraints for Legate/Legion profiling.

These constraints evaluate metrics collected by the Legate GPU profiler
(legion_prof + DuckDB).  They were originally defined alongside the
mixed-precision emulator constraints but logically belong to the ``gpu``
backend since they require real GPU profiling infrastructure.

Hardware-agnostic constraints (Time, Error, Custom) are in:
    trovolut.core.base_constraints
"""

from __future__ import annotations

from typing import Any, Dict

from trovolut.core.base_constraints import (
    Constraint,
    ConstraintResult,
    Group,
    _score_fn,
    _formula_str,
    Time,
    Error,
    Custom,
)


# ============================================================
# Legate GPU Profiling Constraints
# ============================================================

class LegateGPUSeconds(Constraint):
    """GPU compute time via Legion profiler (sum of kernel durations)."""

    def __init__(
        self,
        max_gpu_seconds: float = 10.0,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("legate_gpu_seconds", weight, hard, hard_penalty)
        self.max_gpu_seconds = max_gpu_seconds
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("gpu_seconds", float("inf"))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_gpu_seconds, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_gpu_seconds

    def to_prompt(self) -> str:
        if self.hard:
            return f"GPU compute time (Legate) MUST stay below {self.max_gpu_seconds} GPU-s (hard)."
        return f"Minimize GPU time (target: <{self.max_gpu_seconds} GPU-s, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "gpu_seconds", self.max_gpu_seconds, self.score_fn,
            self.hard, self.hard_penalty, self.weight,
        )


class LegateFramebuffer(Constraint):
    """GPU VRAM usage via Legion profiler."""

    def __init__(
        self,
        max_mb: float = 1000.0,
        metric: str = "peak",
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__(f"legate_fb_{metric}_mb", weight, hard, hard_penalty)
        self.max_mb = max_mb
        self.metric = metric
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        return m.get(f"{self.metric}_fb_mb", 0.0)

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_mb, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_mb

    def to_prompt(self) -> str:
        if self.hard:
            return f"GPU VRAM ({self.metric}) MUST stay below {self.max_mb} MB (hard)."
        return f"Minimize GPU VRAM ({self.metric}: <{self.max_mb} MB, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            f"{self.metric}_fb_mb", self.max_mb, self.score_fn,
            self.hard, self.hard_penalty, self.weight,
        )


class LegateMakespan(Constraint):
    """Wall-clock makespan via Legion profiler (first to last GPU activity)."""

    def __init__(
        self,
        max_seconds: float = 30.0,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("legate_makespan", weight, hard, hard_penalty)
        self.max_seconds = max_seconds
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("makespan_seconds", float("inf"))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_seconds, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_seconds

    def to_prompt(self) -> str:
        if self.hard:
            return f"Makespan MUST stay below {self.max_seconds}s (hard)."
        return f"Minimize makespan (target: <{self.max_seconds}s, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "makespan", self.max_seconds, self.score_fn,
            self.hard, self.hard_penalty, self.weight,
        )


class LegateConcurrency(Constraint):
    """GPU utilization via Legion profiler (gpu_seconds / makespan)."""

    def __init__(
        self,
        min_concurrency: float = 0.5,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
    ):
        super().__init__("legate_concurrency", weight, hard, hard_penalty)
        self.min_concurrency = min_concurrency

    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("avg_concurrency", 0.0)

    def score(self, v: float) -> float:
        return min(1.0, v / self.min_concurrency)

    def is_satisfied(self, v: float) -> bool:
        return v >= self.min_concurrency

    def to_prompt(self) -> str:
        if self.hard:
            return f"GPU concurrency MUST be at least {self.min_concurrency} (hard)."
        return f"Maximize GPU utilization (target: >={self.min_concurrency}, weight: {self.weight})."

    def formula(self) -> str:
        base = f"min(1, concurrency/{self.min_concurrency})"
        if self.hard:
            return (
                f"score = {base} * 1(concurrency >= {self.min_concurrency}) "
                f"+ ({self.hard_penalty}) * 1(concurrency < {self.min_concurrency})"
            )
        if self.weight != 1.0:
            return f"score = {base}  (x{self.weight} weight)"
        return f"score = {base}"


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Re-exported from core for convenience
    "Constraint",
    "ConstraintResult",
    "Group",
    "Time",
    "Error",
    "Custom",
    # Legate constraints
    "LegateGPUSeconds",
    "LegateFramebuffer",
    "LegateMakespan",
    "LegateConcurrency",
]
