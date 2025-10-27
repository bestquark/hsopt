"""Pauli operator-based Hamiltonian implementations."""

from __future__ import annotations
from typing import Any, Dict, Tuple, Sequence, Optional, List, Union

import cupynumeric as np
import scipy.sparse as sp

from eqmmm.qm.hamiltonians._hamiltonian import Hamiltonian, HamiltonianEnsemble


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


class PauliSumHamiltonian(Hamiltonian):
    """Deterministic Pauli-sum Hamiltonian H = Σ_j c_j P_j.

    Args:
        n_qubits (int): Number of qubits in the register.
        pauli_strings (Sequence[str]): Sparse notation strings like
            "X0 Z3 Y5", "Z2", or "I" for the all-identity term.
            Identity is implied on all other qubits.
        coeffs (Sequence[complex]): Coefficients for each Pauli term.
        sparse (bool): If True operate/stash matrices in scipy.sparse.
    """

    def __init__(
        self,
        n_qubits: int,
        pauli_strings: Sequence[str],
        coeffs: Sequence[complex],
        sparse: bool = False,
    ):
        self.n_qubits = int(n_qubits)
        self.sparse = sparse

        self.coeffs = np.asarray(coeffs, dtype=complex)
        if len(pauli_strings) != len(self.coeffs):
            raise ValueError("len(pauli_strings) must match len(coeffs)")

        # keep original sparse notation
        self.pauli_sparse = list(pauli_strings)

        parsed_ops = []
        for s in self.pauli_sparse:
            ss = s.replace(" ", "")

            # special case: pure identity term
            if ss == "" or ss == "I":
                parsed_ops.append([])  # no ops, all identity
                continue

            ops_for_term = []
            i = 0
            while i < len(ss):
                op = ss[i]
                i += 1

                if op not in {"X", "Y", "Z"}:
                    raise ValueError(f"Invalid Pauli operator '{op}' in term '{s}'")

                # read the integer qubit index following the op
                digits = []
                while i < len(ss) and ss[i].isdigit():
                    digits.append(ss[i])
                    i += 1
                if not digits:
                    raise ValueError(f"Missing qubit index after '{op}' in term '{s}'")

                idx = int("".join(digits))
                if idx < 0 or idx >= self.n_qubits:
                    raise ValueError(
                        f"Qubit index {idx} out of range for n_qubits={self.n_qubits} in term '{s}'"
                    )

                ops_for_term.append((idx, op))

            parsed_ops.append(ops_for_term)

        # list[ list[(idx, 'X'/'Y'/'Z')] ]
        self._parsed_ops = parsed_ops

        self._dim = 2**self.n_qubits

    @property
    def hilbert_dim(self) -> int:
        return self._dim

    def _pauli_to_matrix(self, sparse_ops: Sequence[Tuple[int, str]]):
        """
        Build the matrix for one Pauli term given its sparse ops.

        sparse_ops: e.g. [(0,'X'), (3,'Z')] means:
            X on qubit 0, Z on qubit 3, I elsewhere.
        """
        tables = _PAULI_SINGLE_SPARSE if self.sparse else _PAULI_SINGLE_DENSE

        # start as all identity
        single_site_ops = [tables["I"] for _ in range(self.n_qubits)]

        # drop in the specified non-identity ops
        for idx, op in sparse_ops:
            single_site_ops[idx] = tables[op]

        return kron_all(single_site_ops, sparse=self.sparse)

    def to_matrix(self) -> Union[np.ndarray, sp.spmatrix]:
        if self.sparse:
            H = sp.csr_matrix((self._dim, self._dim), dtype=complex)
        else:
            H = np.zeros((self._dim, self._dim), dtype=complex)

        cache: Dict[Tuple[Tuple[int, str], ...], Union[np.ndarray, sp.spmatrix]] = {}

        for c, ops in zip(self.coeffs, self._parsed_ops):
            key = tuple(ops)
            if key in cache:
                P = cache[key]
            else:
                P = self._pauli_to_matrix(ops)
                cache[key] = P

            if self.sparse:
                H = H + c * P
            else:
                H += c * P

        return H

    def terms(self) -> Sequence[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        cache: Dict[Tuple[Tuple[int, str], ...], Union[np.ndarray, sp.spmatrix]] = {}

        for c, ops, sparse_str in zip(self.coeffs, self._parsed_ops, self.pauli_sparse):
            key = tuple(ops)
            if key in cache:
                P = cache[key]
            else:
                P = self._pauli_to_matrix(ops)
                cache[key] = P

            out.append(
                {
                    "coef": c,
                    "matrix": c * P,
                    "pauli": sparse_str,
                }
            )

        return out

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "pauli_sum",
            "n_qubits": self.n_qubits,
            "n_terms": len(self.coeffs),
            "sparse": self.sparse,
            "terms": [
                {
                    "coeff": (c.real if abs(c.imag) < 1e-12 else c),
                    "pauli": s,
                }
                for c, s in zip(self.coeffs, self.pauli_sparse)
            ],
        }


class LatticeHamiltonian(PauliSumHamiltonian):
    """Pauli-sum Hamiltonian for a lattice model, with model/lattice metadata.

    Args:
        n_qubits (int): Number of qubits in the register.
        pauli_strings (Sequence[str]): Sparse notation strings like
            "X0 Z3 Y5", "Z2", or "I" for the all-identity term.
            Identity is implied on all other qubits.
        coeffs (Sequence[complex]): Coefficients for each Pauli term.
        sparse (bool): If True operate/stash matrices in scipy.sparse.
        model (str): Name of the lattice model, e.g. "ising", "heisenberg", "hubbard".
        dims (Tuple[int, ...]): Lattice dimensions, e.g. (Lx,) or (Lx, Ly).
        boundary (str): Boundary condition, e.g. "open" or "periodic".
        params (Dict[str, Any]): Model parameters, e.g. {"J": 1.0, "h": 0.5}.
        neighbor_pairs (Sequence[Tuple[int, int]]): List of lattice neighbor
            site index pairs used in the model.
        notes (str | None): Optional free-text notes about the model instance.
    """

    def __init__(
        self,
        *,
        n_qubits: int,
        pauli_strings: Sequence[str],
        coeffs: Sequence[complex],
        sparse: bool,
        model: str,
        dims: Tuple[int, ...],
        boundary: str,
        params: Dict[str, Any],
        neighbor_pairs: Sequence[Tuple[int, int]],
        notes: Optional[str] = None,
    ):
        super().__init__(
            n_qubits=n_qubits,
            pauli_strings=pauli_strings,
            coeffs=coeffs,
            sparse=sparse,
        )
        self.model = model
        self.dims = tuple(dims)
        self.boundary = str(boundary)
        self.params = dict(params)
        self.neighbor_pairs = list(neighbor_pairs)
        self.notes = notes

    def decompose(self) -> Dict[str, Any]:
        """Extend PauliSumHamiltonian.decompose() with lattice metadata."""
        base = super().decompose()
        base["type"] = "lattice_pauli_sum"
        base["model"] = self.model
        base["dims"] = self.dims
        base["boundary"] = self.boundary
        base["params"] = self.params
        base["n_neighbor_pairs"] = len(self.neighbor_pairs)
        if self.notes is not None:
            base["notes"] = self.notes
        return base


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
