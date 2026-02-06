"""
GPU-accelerated dynamics implementations using cupynumeric and CUDA Quantum.

Class names mirror the CPU backend so that ``trv.set_backend('gpu')`` gives
the same API as ``trv.set_backend('cpu')`` -- the GPU backend simply runs
the operations on NVIDIA GPUs.
"""

from trovolut.core.base_dynamics import BaseDynamics, ProductFormula, SuzukiTrotter


class ExactEvolution(BaseDynamics):
    """
    GPU-accelerated exact evolution via matrix exponential.

    GPU counterpart of ``trovolut.backends.cpu.dynamics.ExactEvolution``.
    Uses cupynumeric for GPU-accelerated matrix operations.
    """

    def __init__(self):
        pass

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        raise NotImplementedError(
            "GPU ExactEvolution is not yet implemented. "
            "Use trv.set_backend('cpu') for CPU-based exact evolution."
        )


class TrotterEvolution(BaseDynamics):
    """
    GPU-accelerated Suzuki-Trotter product formula evolution.

    GPU counterpart of ``trovolut.backends.cpu.dynamics.TrotterEvolution``.
    Accepts the same ``ProductFormula`` objects and parameters.

    Parameters
    ----------
    trotter_steps : int
        Number of Trotter steps (default 10).
    formula : ProductFormula
        Product formula instance (default ``SuzukiTrotter()``).
    """

    def __init__(
        self,
        trotter_steps: int = 10,
        formula: ProductFormula | None = None,
    ):
        self.trotter_steps = trotter_steps
        self.formula = formula if formula is not None else SuzukiTrotter()

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        raise NotImplementedError(
            "GPU TrotterEvolution is not yet implemented. "
            "Use trv.set_backend('cpu') for CPU-based Trotter evolution."
        )


class QDriftEvolution(BaseDynamics):
    """
    GPU-accelerated qDRIFT randomized product formula evolution.

    GPU counterpart of ``trovolut.backends.cpu.dynamics.QDriftEvolution``.

    Parameters
    ----------
    n_samples : int
        Number of random samples (default 100).
    """

    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        raise NotImplementedError(
            "GPU QDriftEvolution is not yet implemented. "
            "Use trv.set_backend('cpu') for CPU-based qDRIFT evolution."
        )


__all__ = ["ExactEvolution", "TrotterEvolution", "QDriftEvolution"]
