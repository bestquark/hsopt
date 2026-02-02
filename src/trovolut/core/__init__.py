"""
Core abstractions for trovolut - backend-agnostic interfaces.

This module provides the base classes that all backends implement:
- BaseDynamics: Time evolution of quantum systems
- BaseEval: Performance evaluation (legacy)
- BaseMetrics: Metrics collection
- Types: Shared data structures
"""

# Import base classes
from trovolut.core.base_dynamics import BaseDynamics
from trovolut.core.base_evaluator import BaseEval
from trovolut.core.base_metrics import BaseMetrics
from trovolut.core.types import (
    Backend,
    EvaluationResult,
    MetricsData,
    DynamicsConfig,
    StateVector,
    HamiltonianMatrix,
)

# Try to import Hamiltonian if it exists
try:
    from trovolut.core.hamiltonian import Hamiltonian
    _has_hamiltonian = True
except ImportError:
    Hamiltonian = None  # type: ignore
    _has_hamiltonian = False

__all__ = [
    "BaseDynamics",
    "BaseEval",
    "BaseMetrics",
    "Backend",
    "EvaluationResult",
    "MetricsData",
    "DynamicsConfig",
    "StateVector",
    "HamiltonianMatrix",
]

if _has_hamiltonian:
    __all__.append("Hamiltonian")
