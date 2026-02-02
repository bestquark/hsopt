"""
GPU backend for trovolut using NVIDIA CuPy and cuQuantum.

This backend requires:
- Linux operating system
- NVIDIA GPU with CUDA support
- CuPy and cuQuantum Python packages

Install with: uv pip install trovolut[gpu]
"""

import sys
from typing import Optional

# Availability check
_available: Optional[bool] = None
_missing_packages: list[str] = []


def is_available() -> bool:
    """Check if GPU backend dependencies are available."""
    global _available, _missing_packages
    
    if _available is not None:
        return _available
    
    # Check platform
    if sys.platform != "linux":
        _available = False
        _missing_packages = [f"requires Linux (current: {sys.platform})"]
        return False
    
    # Check required packages
    required = ["cupy", "cuquantum", "pynvml"]
    _missing_packages = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            _missing_packages.append(pkg)
    
    _available = len(_missing_packages) == 0
    return _available


def _check_availability():
    """Raise ImportError if backend is not available."""
    if not is_available():
        from trovolut.backends import BackendNotAvailableError
        raise BackendNotAvailableError("gpu", _missing_packages)


# Lazy imports - only import if available
def get_dynamics():
    """Get GPU dynamics implementation."""
    _check_availability()
    from trovolut.backends.gpu.dynamics import GPUDynamics
    return GPUDynamics


def get_evaluator():
    """Get GPU evaluator implementation."""
    _check_availability()
    from trovolut.backends.gpu.evaluator import GPUEvaluator
    return GPUEvaluator


def get_metrics():
    """Get GPU metrics collector."""
    _check_availability()
    from trovolut.backends.gpu.metrics import GPUMetrics
    return GPUMetrics


__all__ = [
    "is_available",
    "get_dynamics",
    "get_evaluator",
    "get_metrics",
]
