from __future__ import annotations
import cupynumeric as np
import scipy.sparse as sp
from typing import Any, Dict, Tuple, Sequence, Optional, List, Union
from eqmmm.qm.hamiltonians._hamiltonian import Hamiltonian, HamiltonianEnsemble


# ---------------------------------------------------------------------
# Single-qubit Pauli operators
# ---------------------------------------------------------------------

_PAULI_SINGLE_DENSE = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

_PAULI_SINGLE_SPARSE = {k: sp.csr_matrix(v) for k, v in _PAULI_SINGLE_DENSE.items()}


def kron_all(
    ops: Sequence[Union[np.ndarray, sp.spmatrix]], sparse: bool = False
) -> Union[np.ndarray, sp.spmatrix]:
    """Kronecker product over a list of operators."""
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr") if sparse else np.kron(out, op)
    return out


# ---------------------------------------------------------------------
# Deterministic Pauli-sum Hamiltonian
# ---------------------------------------------------------------------


class PauliSumHamiltonian(Hamiltonian):
    """
    Deterministic H = Σ_j c_j P_j, where P_j are tensor products of {I, X, Y, Z}.
    Optionally stored as sparse matrices for efficiency.

    Args:
        n_qubits: number of qubits
        pauli_strings: e.g. ["ZZII", "XXZZ"]
        coeffs: list/array of coefficients
        sparse: if True, store and operate with scipy.sparse matrices
    """

    def __init__(
        self,
        n_qubits: int,
        pauli_strings: Sequence[str],
        coeffs: Sequence[float],
        sparse: bool = False,
    ):
        self.n_qubits = int(n_qubits)
        self.pauli_strings = list(pauli_strings)
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.sparse = sparse

        if len(self.pauli_strings) != len(self.coeffs):
            raise ValueError("len(pauli_strings) must match len(coeffs)")
        for s in self.pauli_strings:
            if len(s) != self.n_qubits:
                raise ValueError("each Pauli string must have length n_qubits")

        self._dim = 2**self.n_qubits

    @property
    def hilbert_dim(self) -> int:
        return self._dim

    def _pauli_to_matrix(self, pstr: str):
        tables = _PAULI_SINGLE_SPARSE if self.sparse else _PAULI_SINGLE_DENSE
        return kron_all([tables[p] for p in pstr], sparse=self.sparse)

    def to_matrix(self) -> Union[np.ndarray, sp.spmatrix]:
        if self.sparse:
            H = sp.csr_matrix((self._dim, self._dim), dtype=complex)
            for c, pstr in zip(self.coeffs, self.pauli_strings):
                H = H + c * self._pauli_to_matrix(pstr)
        else:
            H = np.zeros((self._dim, self._dim), dtype=complex)
            for c, pstr in zip(self.coeffs, self.pauli_strings):
                H += c * self._pauli_to_matrix(pstr)
        return H

    def terms(self) -> Sequence[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c, pstr in zip(self.coeffs, self.pauli_strings):
            P = self._pauli_to_matrix(pstr)
            term = {
                "coef": c,
                "matrix": c * P,
                "pauli": pstr,
            }
            out.append(term)
        return out

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "pauli_sum",
            "n_qubits": self.n_qubits,
            "n_terms": len(self.pauli_strings),
            "sparse": self.sparse,
            "terms": [
                {"coeff": float(c), "pauli": p}
                for c, p in zip(self.coeffs, self.pauli_strings)
            ],
        }


# ---------------------------------------------------------------------
# Stochastic family of Pauli Hamiltonians
# ---------------------------------------------------------------------


class RandomPauliHamiltonian(HamiltonianEnsemble):
    """
    Distribution over Pauli-sum Hamiltonians.

    Args:
        n_qubits: number of qubits
        pauli_strings: list of Pauli terms, e.g. ["ZZII", "XXZZ"]
        coeff_dist: ("normal", {"mean":0.0,"std":1.0}) or ("uniform", {...})
        sparse: if True, samples produce sparse PauliSumHamiltonians
    """

    def __init__(
        self,
        n_qubits: int,
        pauli_strings: Sequence[str],
        coeff_dist: Tuple[str, Dict[str, float]],
        sparse: bool = False,
    ):
        self.n_qubits = int(n_qubits)
        self.pauli_strings = list(pauli_strings)
        self.coeff_dist = coeff_dist
        self.sparse = sparse

    @property
    def hilbert_dim(self) -> int:
        return 2**self.n_qubits

    def _draw_coeffs(self, rng: np.random.Generator) -> np.ndarray:
        dist_name, params = self.coeff_dist
        if dist_name == "normal":
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            return rng.normal(loc=mean, scale=std, size=len(self.pauli_strings))
        elif dist_name == "uniform":
            lo = params.get("low", -1.0)
            hi = params.get("high", 1.0)
            return rng.uniform(low=lo, high=hi, size=len(self.pauli_strings))
        else:
            raise ValueError(f"Unknown coeff_dist {dist_name}")

    def sample(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> PauliSumHamiltonian:
        rng = rng or np.random.default_rng()
        coeffs = self._draw_coeffs(rng)
        return PauliSumHamiltonian(
            n_qubits=self.n_qubits,
            pauli_strings=self.pauli_strings,
            coeffs=coeffs,
            sparse=self.sparse,
        )

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "random_pauli_family",
            "n_qubits": self.n_qubits,
            "pauli_strings": self.pauli_strings,
            "coeff_dist": self.coeff_dist,
            "sparse": self.sparse,
        }
