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

- `dynamics`: Mixed-precision evolution (unified API: evolve(hamiltonian, state, time))
- `dynamics`: Mixed-precision evolution and low-level Trotter components
- `constraints`: GPU-specific constraints (including DynamicsMemory)
- `measure`: CostAnalyzer, SpacetimeCost for memory/depth
- `hamsim`: Hamiltonian simulation utilities

Example (unified API, no manual quantization):
    ```python
    from trovolut.backends.gpu_mp_emulator.dynamics import MixedPrecisionTrotterEvolution, Precision
    dyn = MixedPrecisionTrotterEvolution(
        trotter_steps=100,
        precision_schedule=[Precision.FP32, Precision.FP16, Precision.FP8_E4M3],  # per term
    )
    state = dyn.evolve(hamiltonian, psi0, time=1.0)
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
