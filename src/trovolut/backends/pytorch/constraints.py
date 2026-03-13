"""PyTorch/CUDA-specific constraints for the autoresearch loop.

Provides constraints for GPU memory, throughput, and fidelity that work
with metrics produced by the PyTorch backend's Metrics collector.
"""

from __future__ import annotations

from typing import Any, Dict

from trovolut.core.base_constraints import (
    Constraint,
    ConstraintResult,
    Group,
    Time,
    Error,
    Custom,
    _score_fn,
)


class GPUMemory(Constraint):
    """Constraint on peak GPU memory usage."""

    def __init__(self, max_mb: float = 1024.0, weight: float = 1.0,
                 hard: bool = False, hard_penalty: float = -100.0,
                 score_fn: str = "linear"):
        super().__init__("gpu_memory", weight, hard, hard_penalty)
        self.max_mb = max_mb
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        mb = m.get("gpu_memory_peak_mb", 0.0)
        if mb == 0:
            # Fallback: convert from bytes
            mb = m.get("gpu_memory_peak_bytes", 0) / (1024 * 1024)
        return float(mb)

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_mb, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_mb

    def to_prompt(self) -> str:
        if self.hard:
            return f"GPU memory MUST stay below {self.max_mb} MB (hard requirement)."
        return f"Minimize GPU memory usage (target: <{self.max_mb} MB, weight: {self.weight})."

    def formula(self) -> str:
        from trovolut.core.base_constraints import _formula_str
        return _formula_str("gpu_mem_mb", self.max_mb, self.score_fn,
                           self.hard, self.hard_penalty, self.weight)


class Throughput(Constraint):
    """Constraint on computational throughput (TFLOPS)."""

    def __init__(self, min_tflops: float = 1.0, weight: float = 1.0,
                 hard: bool = False, hard_penalty: float = -100.0):
        super().__init__("throughput", weight, hard, hard_penalty)
        self.min_tflops = min_tflops

    def extract(self, m: Dict[str, Any]) -> float:
        gflops = m.get("throughput_gflops", 0.0)
        return gflops / 1000.0  # Convert to TFLOPS

    def score(self, v: float) -> float:
        if self.min_tflops <= 0:
            return 1.0
        return min(1.0, v / self.min_tflops)

    def is_satisfied(self, v: float) -> bool:
        return v >= self.min_tflops

    def to_prompt(self) -> str:
        if self.hard:
            return f"Throughput MUST exceed {self.min_tflops} TFLOPS (hard requirement)."
        return f"Maximize throughput (target: >{self.min_tflops} TFLOPS, weight: {self.weight})."

    def formula(self) -> str:
        return f"score = min(1, throughput_tflops / {self.min_tflops})"


class Fidelity(Constraint):
    """Constraint on quantum state fidelity (higher is better)."""

    def __init__(self, min_fidelity: float = 0.99, weight: float = 1.0,
                 hard: bool = False, hard_penalty: float = -100.0):
        super().__init__("fidelity", weight, hard, hard_penalty)
        self.min_fidelity = min_fidelity

    def extract(self, m: Dict[str, Any]) -> float:
        fidelity = m.get("fidelity", 0.0)
        if fidelity == 0.0:
            # Compute from error if available
            error = m.get("error", m.get("mean_error", 1.0))
            fidelity = max(0.0, 1.0 - float(error))
        return float(fidelity)

    def score(self, v: float) -> float:
        return v  # Fidelity is already in [0, 1]

    def is_satisfied(self, v: float) -> bool:
        return v >= self.min_fidelity

    def to_prompt(self) -> str:
        if self.hard:
            return f"Fidelity MUST exceed {self.min_fidelity} (hard requirement)."
        return f"Maximize fidelity (target: >{self.min_fidelity}, weight: {self.weight})."

    def formula(self) -> str:
        return f"score = fidelity (target >= {self.min_fidelity})"


# Re-export base constraints for convenience
__all__ = [
    "GPUMemory",
    "Throughput",
    "Fidelity",
    # Re-exports
    "Constraint",
    "ConstraintResult",
    "Group",
    "Time",
    "Error",
    "Custom",
]
