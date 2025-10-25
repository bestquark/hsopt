from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseHamiltonian(ABC):
    @property
    @abstractmethod
    def hilbert_dim(self) -> int: ...

    @property
    @abstractmethod
    def is_stochastic(self) -> bool: ...

    @abstractmethod
    def to_matrix(self) -> np.ndarray: ...

    @abstractmethod
    def decompose(self) -> Dict[str, Any]: ...

    def sample(self, rng: Optional[np.random.Generator] = None) -> "BaseHamiltonian":
        return self


class MatrixHamiltonian(BaseHamiltonian):
    def __init__(self, H: np.ndarray):
        H = np.array(H, dtype=complex)
        if H.shape[0] != H.shape[1]:
            raise ValueError("H must be square")
        # (optional) Hermiticity check:
        if not np.allclose(H, H.conj().T):
            raise ValueError("H must be Hermitian")
        self._H = H

    @property
    def hilbert_dim(self) -> int:
        return self._H.shape[0]

    @property
    def is_stochastic(self) -> bool:
        return False

    def to_matrix(self) -> np.ndarray:
        return self._H

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "dense_matrix",
            "matrix": self._H,
        }


# ---------------------------------------------------------------------
# Example 3: Electronic structure Hamiltonian (2nd quantized form)
# ---------------------------------------------------------------------


class ElectronicHamiltonian(BaseHamiltonian):
    """
    Standard electronic structure Hamiltonian in 2nd quantization:

        H = sum_{p,q} h_pq a_p^† a_q
          + 0.5 * sum_{p,q,r,s} h_pqrs a_p^† a_q^† a_r a_s
          + const

    Here we just store tensors; `to_matrix` could do Jordan-Wigner,
    BK, etc. (left as placeholder since mapping depends on user codebase).
    """

    def __init__(
        self,
        h1: np.ndarray,  # shape (M,M)
        h2: np.ndarray,  # shape (M,M,M,M)
        constant: float = 0.0,
        mapping: str = "jordan-wigner",
    ):
        self.h1 = np.array(h1, dtype=float)
        self.h2 = np.array(h2, dtype=float)
        self.constant = float(constant)
        self.mapping = mapping  # how to go to qubit space

        M = self.h1.shape[0]
        if self.h1.shape != (M, M):
            raise ValueError("h1 must be square (M,M)")
        if self.h2.shape != (M, M, M, M):
            raise ValueError("h2 must be rank-4 (M,M,M,M)")

        # number of spin orbitals = M
        self.M = M
        # JW gives 2^M Hilbert dim in qubit space
        self._hilbert_dim = 2**M

    @property
    def hilbert_dim(self) -> int:
        return self._hilbert_dim

    @property
    def is_stochastic(self) -> bool:
        return False

    def to_matrix(self) -> np.ndarray:
        """
        Map to qubit matrix via chosen mapping.
        Here we just raise because implementation is system-dependent.
        """
        raise NotImplementedError(
            "ElectronicHamiltonian.to_matrix(): provide JW/BK mapper"
        )

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "second_quantized",
            "one_body": self.h1,
            "two_body": self.h2,
            "constant": self.constant,
            "n_spin_orbs": self.M,
            "mapping": self.mapping,
        }
