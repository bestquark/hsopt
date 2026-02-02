"""
CPU backend for trovolut using NumPy and SciPy.

This is the default backend and is always available.
No additional installation required.
"""

from typing import Optional


def is_available() -> bool:
    """CPU backend is always available."""
    return True


# Lazy imports for dynamics classes
def get_exact_evolution():
    """Get exact evolution dynamics (matrix exponential)."""
    from trovolut.backends.cpu.dynamics import ExactEvolution
    return ExactEvolution


def get_trotter_evolution():
    """Get Trotter evolution dynamics (Suzuki product formulas)."""
    from trovolut.backends.cpu.dynamics import TrotterEvolution
    return TrotterEvolution


def get_qdrift_evolution():
    """Get qDRIFT evolution dynamics (randomized compilation)."""
    from trovolut.backends.cpu.dynamics import QDriftEvolution
    return QDriftEvolution


def get_evaluator():
    """Get CPU evaluator implementation."""
    from trovolut.backends.cpu.evaluator import CPUEvaluator
    return CPUEvaluator


def get_metrics():
    """Get CPU metrics collector."""
    from trovolut.backends.cpu.metrics import CPUMetrics
    return CPUMetrics


def get_constraints():
    """Get CPU-specific constraints (CPUTime, Memory, MatMulCount)."""
    from trovolut.backends.cpu import constraints
    return constraints


__all__ = [
    "is_available",
    "get_exact_evolution",
    "get_trotter_evolution",
    "get_qdrift_evolution",
    "get_evaluator",
    "get_metrics",
    "get_constraints",
]
