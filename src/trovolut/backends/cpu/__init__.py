"""
CPU backend for trovolut - Dynamics methods using NumPy and SciPy.

This backend provides the core dynamics methods that work with any array backend:

- ExactEvolution: Matrix exponential (scipy.linalg.expm)
- TrotterEvolution: Suzuki product formulas
- QDriftEvolution: Randomized product formula

These methods work with NumPy arrays (CPU) or cupynumeric arrays (GPU) automatically.

Example:
    ```python
    from trovolut.backends.cpu.dynamics import ExactEvolution, TrotterEvolution
    from trovolut.backends.cpu.measure import Metrics
    from trovolut.backends.cpu.constraints import CPUTime, Memory
    
    # Works with NumPy (CPU)
    import numpy as np
    H = np.random.randn(4, 4)
    exact = ExactEvolution()
    result = exact.evolve(H, psi0, time=1.0)
    
    # Measure metrics
    metrics = Metrics(constraints=[CPUTime(1.0)])
    result = metrics(my_dynamics)
    ```
"""


def is_available() -> bool:
    """CPU backend is always available."""
    return True


__all__ = ["is_available"]
