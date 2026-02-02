"""
CPU-specific constraints for performance evaluation.

Hardware-agnostic constraints (Time, Error, Custom) are in:
    trovolut.core.base_constraints

This module contains CPU-specific constraints:
    - CPUTime: CPU execution time (user + system)
    - Memory: RAM usage (resident set size)
    - MatMulCount: Matrix multiplication operation count
"""
from __future__ import annotations
from typing import Any, Dict
import math

# Import hardware-agnostic base classes
from trovolut.core.base_constraints import (
    Constraint,
    ConstraintResult,
    ConstraintSet,
    _score_fn,
    _formula_str,
    # Re-export common constraints for convenience
    Time,
    Error,
    Custom,
)


# =============================================================================
# CPU-Specific Constraints
# =============================================================================

class CPUTime(Constraint):
    """Constraint on CPU execution time (user + system)."""
    
    def __init__(self, max_cpu_ms: float = 1000.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "exponential"):
        super().__init__("cpu_time", weight, hard, hard_penalty)
        self.max_cpu_ms = max_cpu_ms
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("cpu_time_ms", float("inf"))
    
    def score(self, v: float) -> float:
        if self.score_fn == "exponential":
            return math.exp(-v / self.max_cpu_ms)
        return _score_fn(v, self.max_cpu_ms, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_cpu_ms
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"CPU time MUST be under {self.max_cpu_ms}ms (hard requirement)."
        return f"Minimize CPU time (target: <{self.max_cpu_ms}ms, weight: {self.weight})."
    
    def formula(self) -> str:
        if self.score_fn == "exponential":
            hard_str = f" (hard, penalty={self.hard_penalty})" if self.hard else ""
            return f"score = exp(-cpu_ms/{self.max_cpu_ms}){hard_str}"
        return _formula_str("cpu_time", self.max_cpu_ms, self.score_fn, self.hard, self.hard_penalty, self.weight)


class Memory(Constraint):
    """Constraint on RAM usage (RSS - resident set size)."""
    
    def __init__(self, max_mb: float = 1000.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("memory", weight, hard, hard_penalty)
        self.max_mb = max_mb
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("memory_mb", 0.0)
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_mb, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_mb
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"Memory usage MUST stay below {self.max_mb} MB (hard requirement)."
        return f"Minimize memory (target: <{self.max_mb} MB, weight: {self.weight})."
    
    def formula(self) -> str:
        return _formula_str("memory", self.max_mb, self.score_fn, self.hard, self.hard_penalty, self.weight)


class MatMulCount(Constraint):
    """Constraint on number of matrix multiplication operations."""
    
    def __init__(self, max_count: int = 100, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
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
        return _formula_str("matmul_count", self.max_count, self.score_fn, self.hard, self.hard_penalty, self.weight)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes (imported from core)
    "Constraint",
    "ConstraintResult",
    "ConstraintSet",
    # Hardware-agnostic constraints (imported from core)
    "Time",
    "Error",
    "Custom",
    # CPU-specific constraints (defined here)
    "CPUTime",
    "Memory",
    "MatMulCount",
]
