"""
Core abstractions for trovolut - backend-agnostic interfaces.

This module provides the base classes that all backends implement:
- BaseDynamics: Time evolution of quantum systems
- BaseEvaluator: Backend-specific evaluation interface
- BaseEval: Performance evaluation (legacy)
- WorkflowConstraintSet: Evaluator code generation for OpenEvolve
- Constraint system: Composable evaluation constraints
- Types: Shared data structures
"""

# Import base classes
from trovolut.core.base_dynamics import BaseDynamics
from trovolut.core.base_evaluator import (
    BaseEvaluator,
    BaseEval,
    WorkflowConstraintSet,
)
from trovolut.core.base_constraints import (
    # Constraint system
    Constraint,
    ConstraintResult,
    ConstraintSet,
    # Hardware-agnostic constraints
    Time,
    Error,
    Custom,
)
from trovolut.core.types import (
    Backend,
    EvaluationResult,
    DynamicsConfig,
    StateVector,
    HamiltonianType,
    HamiltonianMatrix,
)


__all__ = [
    # Base classes
    "BaseDynamics",
    "BaseEvaluator",
    "BaseEval",
    "WorkflowConstraintSet",
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
    "DynamicsConfig",
    "StateVector",
    "HamiltonianType",
    "HamiltonianMatrix",  # Legacy alias
]
