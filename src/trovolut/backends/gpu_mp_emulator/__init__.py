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

- `dynamics`: Mixed-precision evolution methods (Trotter, QDrift, Taylor, Pade)
  All inherit from MPEvolution and share the same precision infrastructure.
- `constraints`: GPU-specific constraints (SpaceCost, TimeCost, SpaceTimeCost, DynamicsMemory, ...)
- `measure`: Unified Metrics class, CostProfile, CostAnalyzer (low-level)
- `hamsim`: Hamiltonian simulation utilities

Example (unified API, no manual quantization):
    ```python
    from trovolut.backends.gpu_mp_emulator.dynamics import (
        MPTrotterEvolution, MPPadeEvolution, Precision,
    )
    from trovolut.backends.gpu_mp_emulator.measure import Metrics
    from trovolut.backends.gpu_mp_emulator.constraints import SpaceCost, TimeCost

    dyn = MPTrotterEvolution(
        trotter_steps=100,
        precision_schedule=[Precision.FP32, Precision.FP16, Precision.FP8_E4M3],
    )
    metrics = Metrics(
        constraints=[SpaceCost(max_memory_bytes=50e6), TimeCost(max_flops=1e12)],
        test_generator=lambda: (hamiltonian, psi0, 1.0),
    )
    result = metrics(dyn)
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
