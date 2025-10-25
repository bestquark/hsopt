import cupynumeric as np
from typing import Any, Dict, Tuple, Sequence, Optional
from eqmmm.qm.hamiltonians._hamiltonian import BaseHamiltonian

# ---------------------------------------------------------------------
# Example 2: Random Pauli Hamiltonian family
#   H = sum_i J_i P_i
#   where P_i are tensor products of {I,X,Y,Z} on n_qubits
# ---------------------------------------------------------------------

_PAULI_SINGLE = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for m in ops[1:]:
        out = np.kron(out, m)
    return out


class RandomPauliHamiltonian(BaseHamiltonian):
    """
    A *distribution* over Pauli-sum Hamiltonians.
    Example usage:
        fam = RandomPauliHamiltonian(
            n_qubits=4,
            pauli_strings=["ZZII", "XXZZ", "ZIZI"],
            coeff_dist=("normal", {"mean":0.0, "std":1.0})
        )
        H_sample = fam.sample()   # <- deterministic instance
        H_mat = H_sample.to_matrix()
    """

    def __init__(
        self,
        n_qubits: int,
        pauli_strings: Sequence[str],
        coeff_dist: Tuple[str, Dict[str, float]],
    ):
        self.n_qubits = n_qubits
        self.pauli_strings = list(pauli_strings)
        self.coeff_dist = coeff_dist  # e.g. ("normal", {"mean":0.0,"std":1.0})

    @property
    def hilbert_dim(self) -> int:
        return 2**self.n_qubits

    @property
    def is_stochastic(self) -> bool:
        return True

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
            raise ValueError(f"Unknown dist {dist_name}")

    def sample(
        self, rng: Optional[np.random.Generator] = None
    ) -> "PauliSumHamiltonian":
        if rng is None:
            rng = np.random.default_rng()
        coeffs = self._draw_coeffs(rng)
        return PauliSumHamiltonian(self.n_qubits, self.pauli_strings, coeffs)

    def to_matrix(self) -> np.ndarray:
        """
        NOTE: this returns the *expected* Hamiltonian if that makes sense.
        Here: expectation of coeffs. For normal/uniform given above, that's the mean.
        """
        dist_name, params = self.coeff_dist
        if dist_name == "normal":
            mean = params.get("mean", 0.0)
        elif dist_name == "uniform":
            mean = 0.5 * (params.get("low", -1.0) + params.get("high", 1.0))
        else:
            raise ValueError(f"Unknown dist {dist_name}")

        coeffs_mean = np.full(len(self.pauli_strings), mean, dtype=float)
        det = PauliSumHamiltonian(self.n_qubits, self.pauli_strings, coeffs_mean)
        return det.to_matrix()

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "pauli_family",
            "n_qubits": self.n_qubits,
            "pauli_strings": list(self.pauli_strings),
            "coeff_dist": self.coeff_dist,
        }


class PauliSumHamiltonian(BaseHamiltonian):
    """
    Deterministic Σ_j c_j P_j.
    Created by RandomPauliHamiltonian.sample() or by hand.
    """

    def __init__(
        self,
        n_qubits: int,
        pauli_strings: Sequence[str],
        coeffs: Sequence[float],
    ):
        pauli_strings = list(pauli_strings)
        coeffs = np.array(coeffs, dtype=float)

        if len(pauli_strings) != len(coeffs):
            raise ValueError("len(pauli_strings) must match len(coeffs)")
        for s in pauli_strings:
            if len(s) != n_qubits:
                raise ValueError("each pauli string must have length n_qubits")

        self.n_qubits = n_qubits
        self.pauli_strings = pauli_strings
        self.coeffs = coeffs

    @property
    def hilbert_dim(self) -> int:
        return 2**self.n_qubits

    @property
    def is_stochastic(self) -> bool:
        return False

    def _pauli_str_to_matrix(self, pstr: str) -> np.ndarray:
        ops = [_PAULI_SINGLE[p] for p in pstr]
        return kron_all(ops)

    def to_matrix(self) -> np.ndarray:
        dim = self.hilbert_dim
        H = np.zeros((dim, dim), dtype=complex)
        for c, pstr in zip(self.coeffs, self.pauli_strings):
            H += c * self._pauli_str_to_matrix(pstr)
        return H

    def decompose(self) -> Dict[str, Any]:
        terms = []
        for c, p in zip(self.coeffs, self.pauli_strings):
            terms.append({"coeff": float(c), "pauli": p})
        return {
            "type": "pauli_sum",
            "n_qubits": self.n_qubits,
            "terms": terms,
        }
