"""Pauli operator-based Hamiltonian implementations."""

from __future__ import annotations
from typing import Any, Dict, Tuple, Sequence, Optional, List, Union

import cupynumeric as np
import scipy.sparse as sp

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
    """Compute the Kronecker product of all operators in ``ops``.

    Args:
        ops (Sequence[Union[np.ndarray, sp.spmatrix]]): Sequence of square
            matrices to combine.
        sparse (bool): If ``True`` keep the result in sparse CSR form.

    Returns:
        Union[np.ndarray, sp.spmatrix]: Kronecker product of the sequence.
    """
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr") if sparse else np.kron(out, op)
    return out


# ---------------------------------------------------------------------
# Deterministic Pauli-sum Hamiltonian
# ---------------------------------------------------------------------


class PauliSumHamiltonian(Hamiltonian):
    """Deterministic Pauli-sum Hamiltonian ``H = Σ_j c_j P_j``.

    Args:
        n_qubits (int): Number of qubits in the register.
        pauli_strings (Sequence[str]): Pauli strings such as ``"ZZII"``. Each
            string must have length ``n_qubits``.
        coeffs (Sequence[float]): Coefficients associated with each Pauli term.
        sparse (bool): If ``True`` store and operate on sparse matrices.
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
        """int: Hilbert-space dimension ``2**n_qubits``."""
        return self._dim

    def _pauli_to_matrix(self, pstr: str):
        """Convert a Pauli string to a matrix, respecting ``sparse`` mode."""
        tables = _PAULI_SINGLE_SPARSE if self.sparse else _PAULI_SINGLE_DENSE
        return kron_all([tables[p] for p in pstr], sparse=self.sparse)

    def to_matrix(self) -> Union[np.ndarray, sp.spmatrix]:
        """Return the full matrix representation of the Pauli sum.

        Returns:
            Union[np.ndarray, sp.spmatrix]: Dense or sparse matrix depending on ``sparse`` mode.
        """
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
        """Return the list of Pauli terms and their coefficients.

        Returns:
            Sequence[Dict[str, Any]]:
                Entries with ``coef``, ``matrix``, and ``pauli`` metadata.
        """
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
        """Return a JSON-serializable description of the Pauli sum.

        Returns:
            Dict[str, Any]: Metadata including qubit count, sparsity flag, and per-term coefficients.
        """
        return {
            "type": "pauli_sum",
            "n_qubits": self.n_qubits,
            "n_terms": len(self.pauli_strings),
            "sparse": self.sparse,
            "terms": [
                {"coeff": float(c), "pauli": p} for c, p in zip(self.coeffs, self.pauli_strings)
            ],
        }


# ---------------------------------------------------------------------
# Stochastic family of Pauli Hamiltonians
# ---------------------------------------------------------------------


class RandomPauliHamiltonian(HamiltonianEnsemble):
    """Stochastic ensemble whose draws are Pauli-sum Hamiltonians.

    Args:
        n_qubits (int): Number of qubits.
        pauli_strings (Sequence[str]): List of Pauli strings shared by all
            samples.
        coeff_dist (Tuple[str, Dict[str, float]]): Distribution specification,
            e.g. ``("normal", {"mean": 0.0, "std": 1.0})`` or
            ``("uniform", {"low": -1.0, "high": 1.0})``.
        sparse (bool): If ``True`` samples are sparse Pauli sums.
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
        """int: Hilbert-space dimension ``2**n_qubits``."""
        return 2**self.n_qubits

    def _draw_coeffs(self, rng: np.random.Generator) -> np.ndarray:
        """Sample random coefficients following ``coeff_dist``.

        Args:
            rng (np.random.Generator): Source of randomness.

        Returns:
            np.ndarray: Array of coefficients, one per Pauli string.

        Raises:
            ValueError: If the distribution name is not supported.
        """
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
        """Draw a deterministic Pauli-sum Hamiltonian from the ensemble.

        Args:
            rng (np.random.Generator | None): Optional RNG.

        Returns:
            PauliSumHamiltonian: Deterministic Hamiltonian containing the new random coefficients.
        """
        rng = rng or np.random.default_rng()
        coeffs = self._draw_coeffs(rng)
        return PauliSumHamiltonian(
            n_qubits=self.n_qubits,
            pauli_strings=self.pauli_strings,
            coeffs=coeffs,
            sparse=self.sparse,
        )

    def decompose(self) -> Dict[str, Any]:
        """Return a metadata description of the stochastic model.

        Returns:
            Dict[str, Any]: Description with qubit count, Pauli support, coefficient distribution, and sparsity flag.
        """
        return {
            "type": "random_pauli_family",
            "n_qubits": self.n_qubits,
            "pauli_strings": self.pauli_strings,
            "coeff_dist": self.coeff_dist,
            "sparse": self.sparse,
        }
