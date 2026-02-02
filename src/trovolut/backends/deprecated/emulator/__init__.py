"""
GPU Emulator backend for trovolut.

This backend provides CPU-based emulation for studying quantized exponentials
and mixed-precision Trotter methods. It simulates GPU low-precision behavior
without requiring actual GPU hardware.

Key Features:
- Mixed-precision Trotter decomposition (FP64, FP32, FP16, BF16, FP8, INT8)
- Quantization simulation for studying numerical precision effects
- CPU-based implementation using NumPy (or cupynumeric on GPU systems)

Install with: uv pip install trovolut[gpu-emulate]
(Note: No additional dependencies required beyond base trovolut)
"""

from typing import Optional

# Availability check
_available: Optional[bool] = None


def is_available() -> bool:
    """
    Check if GPU emulator backend is available.
    
    The emulator uses NumPy and is always available since NumPy is a core dependency.
    Optional: cupynumeric provides GPU acceleration if available.
    """
    global _available
    
    if _available is not None:
        return _available
    
    # Emulator is always available (uses NumPy)
    _available = True
    return _available


def _check_availability():
    """Raise ImportError if backend is not available."""
    if not is_available():
        from trovolut.backends import BackendNotAvailableError
        raise BackendNotAvailableError("gpu-emulate", [])


# Lazy imports - use existing gpu_emulator implementation
def get_dynamics():
    """
    Get GPU emulator dynamics implementation.
    
    Returns module with mixed-precision Trotter methods:
    - suzuki_step(): Suzuki-Trotter decomposition of various orders
    - MixedPrecisionTrotter: Framework for studying quantized exponentials
    """
    _check_availability()
    from trovolut.backends.gpu_emulator import dynamics
    return dynamics


def get_evaluator():
    """
    Get GPU emulator evaluator implementation.
    
    Returns the evaluator builder for OpenEvolve integration.
    """
    _check_availability()
    from trovolut.backends.gpu_emulator import evaluator
    return evaluator


def get_metrics():
    """Get GPU emulator metrics collector."""
    _check_availability()
    from trovolut.backends.emulator.metrics import EmulatorMetrics
    return EmulatorMetrics


# Re-export commonly used classes and functions
def get_mixed_precision_tools():
    """
    Get mixed-precision tools for studying quantized exponentials.
    
    Returns:
        Module with Precision, PrecisionConfig, ExpTerm, MixedPrecisionTrotter, etc.
    """
    _check_availability()
    from trovolut.backends.gpu_emulator.mixed_fps import mp_trotter
    return mp_trotter


# Convenience exports
def suzuki_step(*args, **kwargs):
    """
    Suzuki-Trotter decomposition step.
    
    See trovolut.backends.gpu_emulator.dynamics.suzuki_step for details.
    """
    _check_availability()
    from trovolut.backends.gpu_emulator.dynamics import suzuki_step
    return suzuki_step(*args, **kwargs)


__all__ = [
    "is_available",
    "get_dynamics",
    "get_evaluator",
    "get_metrics",
    "get_mixed_precision_tools",
    "suzuki_step",
]
