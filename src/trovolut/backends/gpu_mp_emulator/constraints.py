"""
Constraints for the GPU mixed-precision emulator backend.

Hardware-agnostic constraints (Time, Error, Custom) are in:
    trovolut.core.base_constraints

Legate/Legion profiling constraints are in:
    trovolut.backends.gpu.constraints

This module contains constraints specific to the mixed-precision emulator:

- MemoryCost:     Peak memory (bytes) from CostProfile
- FlopsCost:      Total FLOPs from CostProfile
- SpaceTimeCost:  Space x Time product from CostProfile
- MatMulCount:    Matrix multiplication operation count
"""

from __future__ import annotations

from typing import Any, Dict

# Import hardware-agnostic base classes
from trovolut.core.base_constraints import (
    Constraint,
    ConstraintResult,
    Group,
    _score_fn,
    _formula_str,
    # Re-export common constraints for convenience
    Time,
    Error,
    Custom,
)


# =============================================================================
# Mixed-Precision Cost Constraints
# =============================================================================

class MemoryCost(Constraint):
    """Constraint on peak memory (bytes) from CostProfile.

    Works with the unified Metrics class -- the ``peak_memory_bytes``
    key is injected into the metrics dict by ``CostProfile.to_dict()``.
    """

    def __init__(
        self,
        max_memory_bytes: float = 100 * 1024 * 1024,  # 100 MB default
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("memory_cost", weight, hard, hard_penalty)
        self.max_memory_bytes = max_memory_bytes
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("peak_memory_bytes", m.get("memory_bytes", float("inf"))))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_memory_bytes, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_memory_bytes

    def to_prompt(self) -> str:
        mb = self.max_memory_bytes / (1024 ** 2)
        if self.hard:
            return f"Peak memory MUST stay below {mb:.0f} MB (hard)."
        return f"Minimize peak memory (target: <{mb:.0f} MB, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "peak_memory_bytes", self.max_memory_bytes,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


class FlopsCost(Constraint):
    """Constraint on total FLOPs from CostProfile.

    Scores the ``total_flops`` key returned by ``CostProfile.to_dict()``.
    """

    def __init__(
        self,
        max_flops: float = 1e12,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("flops_cost", weight, hard, hard_penalty)
        self.max_flops = max_flops
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("total_flops", float("inf")))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_flops, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_flops

    def to_prompt(self) -> str:
        if self.hard:
            return f"Total FLOPs MUST stay below {self.max_flops:.2e} (hard)."
        return f"Minimize FLOPs (target: <{self.max_flops:.2e}, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "total_flops", self.max_flops,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


class SpaceTimeCost(Constraint):
    """Constraint on the space-time product (peak_memory_bytes * total_depth).

    Scores the ``space_time_product`` key returned by ``CostProfile.to_dict()``.
    """

    def __init__(
        self,
        max_product: float = 1e15,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("space_time_cost", weight, hard, hard_penalty)
        self.max_product = max_product
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("space_time_product", float("inf")))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_product, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_product

    def to_prompt(self) -> str:
        if self.hard:
            return f"Space x Time product MUST stay below {self.max_product:.2e} (hard)."
        return f"Minimize Space x Time (target: <{self.max_product:.2e}, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "space_time_product", self.max_product,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


# =============================================================================
# Emulator-specific static operation counter
# =============================================================================

class MatMulCount(Constraint):
    """Constraint on number of matrix multiplication operations."""

    def __init__(
        self,
        max_count: int = 100,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("matmul_count", weight, hard, hard_penalty)
        self.max_count = max_count
        self.score_fn = score_fn

    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("matmul_count", 0))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_count, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_count

    def to_prompt(self) -> str:
        if self.hard:
            return f"Matrix multiplications MUST stay below {self.max_count} (hard requirement)."
        return f"Minimize matrix multiplications (target: <{self.max_count}, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "matmul_count", self.max_count,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes (imported from core)
    "Constraint",
    "ConstraintResult",
    "Group",
    # Hardware-agnostic constraints (imported from core)
    "Time",
    "Error",
    "Custom",
    # Mixed-precision cost constraints
    "MemoryCost",
    "FlopsCost",
    "SpaceTimeCost",
    # Operation counter
    "MatMulCount",
]
