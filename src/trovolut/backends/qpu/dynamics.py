"""
QPU dynamics implementations using CUDA Quantum.

Class names mirror the CPU/GPU backends so that ``trv.set_backend('qpu')``
gives a consistent API.  QPU methods compile product formulas into quantum
circuits for execution on real quantum hardware or high-performance simulators.
"""

from trovolut.core.base_dynamics import BaseDynamics, ProductFormula, SuzukiTrotter


class TrotterEvolution(BaseDynamics):
    """
    QPU-based Suzuki-Trotter evolution via circuit compilation.

    QPU counterpart of ``trovolut.backends.cpu.dynamics.TrotterEvolution``.
    Compiles the product formula into a quantum circuit and executes it on
    the specified CUDA Quantum target.

    Parameters
    ----------
    trotter_steps : int
        Number of Trotter steps (default 10).
    formula : ProductFormula
        Product formula instance (default ``SuzukiTrotter()``).
    target : str
        CUDA Quantum target backend (default ``"nvidia-mgpu"``).
    shots : int
        Number of measurement shots (default 1000).
    """

    def __init__(
        self,
        trotter_steps: int = 10,
        formula: ProductFormula | None = None,
        target: str = "nvidia-mgpu",
        shots: int = 1000,
    ):
        self.trotter_steps = trotter_steps
        self.formula = formula if formula is not None else SuzukiTrotter()
        self.target = target
        self.shots = shots

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        raise NotImplementedError(
            "QPU TrotterEvolution is not yet implemented. "
            "Will include Hamiltonian -> circuit compilation."
        )


class QDriftEvolution(BaseDynamics):
    """
    QPU-based qDRIFT evolution via randomized circuit compilation.

    QPU counterpart of ``trovolut.backends.cpu.dynamics.QDriftEvolution``.

    Parameters
    ----------
    n_samples : int
        Number of random samples (default 100).
    target : str
        CUDA Quantum target backend (default ``"nvidia-mgpu"``).
    shots : int
        Number of measurement shots (default 1000).
    """

    def __init__(
        self,
        n_samples: int = 100,
        target: str = "nvidia-mgpu",
        shots: int = 1000,
    ):
        self.n_samples = n_samples
        self.target = target
        self.shots = shots

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        raise NotImplementedError(
            "QPU QDriftEvolution is not yet implemented. "
            "Will include randomized circuit compilation."
        )


__all__ = ["TrotterEvolution", "QDriftEvolution"]
