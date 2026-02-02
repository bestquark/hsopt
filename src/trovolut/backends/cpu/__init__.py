"""
CPU backend for trovolut - Dynamics methods using NumPy and SciPy.

This backend provides the core dynamics methods that work with any array backend:
- ExactEvolution: Matrix exponential (scipy.linalg.expm)
- TrotterEvolution: Suzuki product formulas
- QDriftEvolution: Randomized product formula

These methods work with NumPy arrays (CPU) or CuPy arrays (GPU) automatically.

Usage:
    >>> from trovolut.backends.cpu.dynamics import ExactEvolution, TrotterEvolution
    >>> from trovolut.backends.cpu.evaluator import CPUEvaluator
    >>> from trovolut.backends.cpu import constraints
    >>> 
    >>> # Works with NumPy (CPU)
    >>> import numpy as np
    >>> H = np.random.randn(4, 4)
    >>> exact = ExactEvolution()
    >>> result = exact.evolve(H, psi0, time=1.0)
    >>> 
    >>> # Same code works with CuPy (GPU)
    >>> import cupy as cp
    >>> H_gpu = cp.asarray(H)
    >>> result_gpu = exact.evolve(H_gpu, psi0_gpu, time=1.0)
"""


def is_available() -> bool:
    """CPU backend is always available."""
    return True


__all__ = ["is_available"]
