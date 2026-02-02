"""
QPU (Quantum Processing Unit) backend for trovolut.

This backend provides access to real quantum hardware and simulators via:
- Qiskit (cross-platform)
- Cirq (cross-platform)
- CUDA Quantum / cudaq (Linux-only, optional)

Works on all platforms with qiskit and cirq.
CUDA Quantum provides additional GPU-accelerated simulation on Linux.

Install with: uv pip install trovolut[qpu]
"""

import sys
from typing import Optional

# Availability check
_available: Optional[bool] = None
_missing_packages: list[str] = []


def is_available() -> bool:
    """Check if QPU backend dependencies are available."""
    global _available, _missing_packages
    
    if _available is not None:
        return _available
    
    # Check required packages
    # Note: cudaq is Linux-only, but qiskit and cirq are cross-platform
    # Backend can work with just qiskit/cirq on any platform
    required = ["qiskit", "cirq"]
    optional = ["cudaq"] if sys.platform == "linux" else []
    
    _missing_packages = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            _missing_packages.append(pkg)
    
    # Check optional packages (cudaq on Linux)
    for pkg in optional:
        try:
            __import__(pkg)
        except ImportError:
            _missing_packages.append(f"{pkg} (optional on Linux)")
    
    # Available if at least the required packages are present
    _available = len([p for p in _missing_packages if "optional" not in p]) == 0
    return _available


def _check_availability():
    """Raise ImportError if backend is not available."""
    if not is_available():
        from trovolut.backends import BackendNotAvailableError
        raise BackendNotAvailableError("qpu", _missing_packages)


# Lazy imports - only import if available
def get_dynamics():
    """Get QPU dynamics implementation (circuit-based)."""
    _check_availability()
    from trovolut.backends.qpu.dynamics import QPUDynamics
    return QPUDynamics


def get_evaluator():
    """Get QPU evaluator implementation."""
    _check_availability()
    from trovolut.backends.qpu.evaluator import QPUEvaluator
    return QPUEvaluator


def get_metrics():
    """Get QPU metrics collector (circuit depth, gate count, etc.)."""
    _check_availability()
    from trovolut.backends.qpu.metrics import QPUMetrics
    return QPUMetrics


__all__ = [
    "is_available",
    "get_dynamics",
    "get_evaluator",
    "get_metrics",
]
