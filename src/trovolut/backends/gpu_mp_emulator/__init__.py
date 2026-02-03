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

- `mp_trotter`: Mixed-precision Trotter framework
- `simulate_quantization`: Quantization utilities
- `mp_evals`: Mixed-precision evaluation functions
- `constraints`: GPU-specific constraints
- `hamsim`: Hamiltonian simulation utilities

Example:
    ```python
    from trovolut.backends.gpu_mp_emulator.mp_trotter import (
        Precision, ExpTerm, MixedPrecisionTrotter
    )
    from trovolut.backends.gpu_mp_emulator import mp_evals
    ```
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


__all__ = ["is_available"]
