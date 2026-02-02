# src/trovolut/core/types.py
from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray

class Backend(Enum):
    CPU = auto()
    GPU = auto()
    QPU = auto()
    EMULATOR = auto()

@dataclass(frozen=True)
class EvaluationResult:
    """Standardized result from any backend."""
    fidelity: float
    runtime_seconds: float
    memory_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    
    def score(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted score for OpenEvolve optimization."""
        w = weights or {"fidelity": 1.0, "runtime": -0.01, "memory": -1e-9}
        return (
            w.get("fidelity", 0) * self.fidelity +
            w.get("runtime", 0) * self.runtime_seconds +
            w.get("memory", 0) * (self.memory_bytes or 0)
        )

@dataclass
class DynamicsConfig:
    """Configuration for dynamics computation."""
    method: str = "auto"           # e.g., "pade", "taylor", "krylov", "trotter"
    precision: str = "float64"     # "float32", "float64", "mixed"
    tolerance: float = 1e-10
    max_iterations: int = 1000
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricsData:
    """Container for performance metrics from any backend."""
    execution_time_seconds: float
    memory_bytes: int | None = None
    backend_specific: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_time_seconds": self.execution_time_seconds,
            "memory_bytes": self.memory_bytes,
            **self.backend_specific
        }

# Type aliases
StateVector = NDArray[np.complexfloating]
HamiltonianMatrix = NDArray[np.complexfloating]
