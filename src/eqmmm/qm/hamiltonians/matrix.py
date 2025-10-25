from typing import Any, Dict
import cupynumeric as np
from eqmmm.qm.hamiltonians._hamiltonian import BaseHamiltonian


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
