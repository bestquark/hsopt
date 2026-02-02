"""
GPU Mixed Precision backend for trovolut.

This backend provides CPU-based emulation for studying quantized exponentials
and mixed-precision Trotter methods. It simulates GPU low-precision arithmetic
without requiring actual GPU hardware.

Key Features:
- Mixed-precision Trotter decomposition (FP64, FP32, FP16, BF16, FP8, INT8)
- Quantization simulation for studying numerical precision effects
- CPU-based implementation using NumPy (or cupynumeric on GPU systems)
- Evaluations with mp_evals for analyzing precision tradeoffs

Core Modules:
- mp_trotter: Mixed-precision Trotter framework
- simulate_quantization: Quantization utilities
- mp_evals: Mixed-precision evaluation functions
- constraints: GPU-specific constraints
- hamsim: Hamiltonian simulation utilities

Usage:
    >>> from trovolut.backends.gpu_mp_emulator.mp_trotter import (
    ...     Precision, ExpTerm, MixedPrecisionTrotter
    ... )
    >>> from trovolut.backends.gpu_mp_emulator import mp_evals
"""

from typing import Optional

# Availability check
_available: Optional[bool] = None


def is_available() -> bool:
    """
    Check if GPU mixed precision backend is available.
    
    The backend uses NumPy and is always available since NumPy is a core dependency.
    Optional: cupynumeric provides GPU acceleration if available.
    """
    global _available
    
    if _available is not None:
        return _available
    
    # Mixed precision emulator is always available (uses NumPy)
    _available = True
    return _available


def _check_availability():
    """Raise ImportError if backend is not available."""
    if not is_available():
        from trovolut.backends import BackendNotAvailableError
        raise BackendNotAvailableError("gpu-mixed-precision", [])


# Lazy imports
def get_dynamics():
    """
    Get mixed precision dynamics implementation.
    
    Note: This backend focuses on mixed-precision analysis tools (mp_trotter, mp_evals).
    For standard dynamics, use the CPU or GPU backends instead.
    
    Returns:
        Minimal dynamics class (use mp_trotter module directly for mixed-precision simulation)
    """
    _check_availability()
    from trovolut.core.base_dynamics import BaseDynamics
    
    class MixedPrecisionDynamics(BaseDynamics):
        """Minimal dynamics class for gpu_mp_emulator backend."""
        def evolve(self, hamiltonian, initial_state, time, config=None, **kwargs):
            raise NotImplementedError(
                "Standard dynamics not implemented for gpu_mp_emulator backend. "
                "Use mp_trotter.MixedPrecisionTrotter directly for mixed-precision simulation, "
                "or use the CPU/GPU backends for standard evolution."
            )
    
    return MixedPrecisionDynamics


def get_evaluator():
    """
    Get mixed precision evaluator.
    
    Returns:
        Base evaluator (use mp_evals module for mixed-precision analysis)
    """
    _check_availability()
    from trovolut.core.base_evaluator import BaseEval
    return BaseEval


def get_metrics():
    """
    Get mixed precision metrics collector.
    
    Returns:
        Base metrics (use mp_evals module for mixed-precision metrics)
    """
    _check_availability()
    from trovolut.core.base_metrics import BaseMetrics
    return BaseMetrics


def get_mp_trotter():
    """
    Get mixed-precision Trotter module.
    
    Returns:
        Module with Precision, PrecisionConfig, ExpTerm, MixedPrecisionTrotter, etc.
    """
    _check_availability()
    from trovolut.backends.gpu_mp_emulator import mp_trotter
    return mp_trotter


def get_mp_evals():
    """
    Get mixed-precision evaluation functions.
    
    Returns:
        Module with evaluation utilities for analyzing precision tradeoffs
    """
    _check_availability()
    from trovolut.backends.gpu_mp_emulator import mp_evals
    return mp_evals


# Note: Submodules are available for import but not eagerly loaded to avoid circular imports
# Use: from trovolut.backends.gpu_mp_emulator import mp_trotter, mp_evals, etc.

__all__ = [
    "is_available",
    "get_dynamics",
    "get_evaluator",
    "get_metrics",
    "get_mp_trotter",
    "get_mp_evals",
]
