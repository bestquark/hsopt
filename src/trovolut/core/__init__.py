"""
Core abstractions for trovolut - backend-agnostic interfaces.

This module provides the base classes that all backends implement:
- BaseDynamics: Time evolution of quantum systems
- BaseMetrics: Backend-specific metrics measurement interface
- Constraint system: Composable evaluation constraints (Group)
- Product formulas: Suzuki-Trotter decompositions (ProductFormula, suzuki)
- Types: Shared data structures
"""

# Import base classes
from trovolut.core.base_dynamics import BaseDynamics
from trovolut.core.base_metrics import BaseMetrics
from trovolut.core.base_constraints import (
    # Constraint system
    Constraint,
    ConstraintResult,
    Group,
    # Hardware-agnostic constraints
    Time,
    Error,
    Custom,
)
from trovolut.core.base_dynamics import (
    ProductFormula,
    LieTrotter,
    SuzukiTrotter,
    SuzukiHighOrder,
    suzuki,
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
    "BaseMetrics",
    # Constraint system
    "Constraint",
    "ConstraintResult",
    "Group",
    # Hardware-agnostic constraints
    "Time",
    "Error",
    "Custom",
    # Product formulas
    "ProductFormula",
    "LieTrotter",
    "SuzukiTrotter",
    "SuzukiHighOrder",
    "suzuki",
    # Types
    "Backend",
    "EvaluationResult",
    "DynamicsConfig",
    "StateVector",
    "HamiltonianType",
    "HamiltonianMatrix",  # Legacy alias
]
