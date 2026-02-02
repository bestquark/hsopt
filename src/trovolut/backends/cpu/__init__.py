"""
CPU backend for trovolut using NumPy and SciPy.

This is the default backend and is always available.
No additional installation required.
"""

from typing import Optional


def is_available() -> bool:
    """CPU backend is always available."""
    return True


# Lazy imports
def get_dynamics():
    """Get CPU dynamics implementation."""
    from trovolut.backends.cpu.dynamics import CPUDynamics
    return CPUDynamics


def get_evaluator():
    """Get CPU evaluator implementation."""
    from trovolut.backends.cpu.evaluator import CPUEvaluator
    return CPUEvaluator


def get_metrics():
    """Get CPU metrics collector."""
    from trovolut.backends.cpu.metrics import CPUMetrics
    return CPUMetrics


__all__ = [
    "is_available",
    "get_dynamics",
    "get_evaluator",
    "get_metrics",
]
