"""
Core abstractions for trovolut - backend-agnostic interfaces.

This module provides the base classes that all backends implement:
- BaseDynamics: Time evolution of quantum systems
- BaseEval: Performance evaluation (legacy)
- BaseMetrics: Metrics collection
- Constraint system: Composable evaluation constraints
- Types: Shared data structures
"""

# Import base classes
from trovolut.core.base_dynamics import BaseDynamics
from trovolut.core.base_evaluator import (
    BaseEval,
    # Constraint system
    Constraint,
    ConstraintResult,
    ConstraintSet,
    # Hardware-agnostic constraints
    Time,
    Error,
    Custom,
)
from trovolut.core.base_metrics import BaseMetrics
from trovolut.core.types import (
    Backend,
    EvaluationResult,
    MetricsData,
    DynamicsConfig,
    StateVector,
    HamiltonianType,
    HamiltonianMatrix,  # Legacy alias
)

# Try to import Hamiltonian if it exists
try:
    from trovolut.core.hamiltonian import Hamiltonian
    _has_hamiltonian = True
except ImportError:
    Hamiltonian = None  # type: ignore
    _has_hamiltonian = False

__all__ = [
    # Base classes
    "BaseDynamics",
    "BaseEval",
    "BaseMetrics",
    # Constraint system
    "Constraint",
    "ConstraintResult",
    "ConstraintSet",
    # Hardware-agnostic constraints
    "Time",
    "Error",
    "Custom",
    # Types
    "Backend",
    "EvaluationResult",
    "MetricsData",
    "DynamicsConfig",
    "StateVector",
    "HamiltonianType",
    "HamiltonianMatrix",  # Legacy alias
]

if _has_hamiltonian:
    __all__.append("Hamiltonian")
