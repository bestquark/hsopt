from __future__ import annotations
from typing import Any, Dict, Sequence, Optional, List
from eqmmm.qm.hamiltonians._hamiltonian import Hamiltonian, HamiltonianEnsemble
import cupynumeric as np


# -------------------------------------------------
# Deterministic dense-matrix Hamiltonian
# -------------------------------------------------


class MatrixHamiltonian(Hamiltonian):
    """
    Deterministic Hamiltonian explicitly defined as:
        H = Σ_i coef_i * H_i

    Stores:
      - _H : total Hermitian matrix
      - _term_list : [{"coef": scalar, "matrix": Hermitian piece}, ...]
    """

    def __init__(
        self,
        H_total: np.ndarray,
        term_list: Optional[Sequence[Dict[str, np.ndarray]]] = None,
    ):
        H_total = np.array(H_total, dtype=complex)
        if H_total.shape[0] != H_total.shape[1]:
            raise ValueError("H_total must be square")
        if not np.allclose(H_total, H_total.conj().T):
            raise ValueError("H_total must be Hermitian")

        self._H = H_total
        self._dim = H_total.shape[0]

        # Validate / set decomposition
        if term_list is not None:
            recon = np.zeros_like(self._H)
            validated_terms = []
            for t in term_list:
                coef = complex(t["coef"])
                block = np.array(t["matrix"], dtype=complex)
                if block.shape != (self._dim, self._dim):
                    raise ValueError("term matrix has wrong shape")
                if not np.allclose(block, block.conj().T):
                    raise ValueError("term matrix must be Hermitian")
                validated_terms.append({"coef": coef, "matrix": block})
                recon += coef * block
            if not np.allclose(recon, self._H):
                raise ValueError("term_list does not sum back to H_total")
            self._term_list = validated_terms
        else:
            self._term_list = [{"coef": 1.0, "matrix": self._H}]

    @property
    def hilbert_dim(self) -> int:
        return self._dim

    def to_matrix(self) -> np.ndarray:
        return self._H

    def terms(self) -> Sequence[Dict[str, np.ndarray]]:
        return [
            {"coef": t["coef"], "matrix": np.array(t["matrix"], copy=True)}
            for t in self._term_list
        ]

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "matrix",
            "dim": self._dim,
            "n_terms": len(self._term_list),
            "coefficients": [complex(t["coef"]) for t in self._term_list],
        }


# -------------------------------------------------
# Ensemble of random Hermitian Hamiltonians
# -------------------------------------------------


class WeightedRandomHamiltonian(HamiltonianEnsemble):
    """
    Ensemble of random Hermitian Hamiltonians.

    Each draw:
        A ~ complex Gaussian
        H_i = (A + A†)/2
        coef_i ~ user-defined distribution
        H = Σ_i coef_i * H_i

    Parameters
    ----------
    n : int
        Hilbert space dimension.
    m : int
        Number of random Hermitian terms.
    coef_dist : dict or None
        Defines how coefficients are drawn.
        Supported formats:
          {"type": "discrete", "values": [10.0, 0.01], "probs": [0.5, 0.5]}
          {"type": "uniform", "low": -1.0, "high": 1.0}
          {"type": "normal", "mean": 0.0, "std": 1.0}
        Defaults to the discrete two-scale {10.0, 0.01} distribution.
    """

    def __init__(
        self,
        n: int = 100,
        m: int = 5,
        coef_dist: Optional[Dict[str, Any]] = None,
    ):
        self.n = int(n)
        self.m = int(m)
        self.coef_dist = coef_dist or {
            "type": "discrete",
            "values": [10.0, 0.01],
            "probs": [0.5, 0.5],
        }

    @property
    def hilbert_dim(self) -> int:
        return self.n

    def _draw_coefs(self, rng: np.random.Generator) -> np.ndarray:
        """Draw random coefficients according to the distribution."""
        d = self.coef_dist
        t = d["type"]

        if t == "discrete":
            vals = np.array(d["values"], dtype=float)
            probs = np.array(d.get("probs", np.ones_like(vals) / len(vals)))
            return rng.choice(vals, size=self.m, p=probs)

        elif t == "uniform":
            low = d.get("low", -1.0)
            high = d.get("high", 1.0)
            return rng.uniform(low, high, size=self.m)

        elif t == "normal":
            mean = d.get("mean", 0.0)
            std = d.get("std", 1.0)
            return rng.normal(mean, std, size=self.m)

        else:
            raise ValueError(f"Unknown coef_dist type {t}")

    def _draw_terms(self, rng: np.random.Generator) -> List[Dict[str, np.ndarray]]:
        n = self.n
        coefs = self._draw_coefs(rng)
        terms = []
        for coef_i in coefs:
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            H_i = (A + A.conj().T) / 2.0
            terms.append({"coef": coef_i, "matrix": H_i})
        return terms

    def sample(self, rng: Optional[np.random.Generator] = None) -> MatrixHamiltonian:
        rng = rng or np.random.default_rng()
        terms = self._draw_terms(rng)
        H_total = np.sum([t["coef"] * t["matrix"] for t in terms], axis=0)
        return MatrixHamiltonian(H_total, term_list=terms)

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "weighted_random_ensemble",
            "dim": self.n,
            "m_terms": self.m,
            "coef_dist": self.coef_dist,
            "distribution": "complex Gaussian Hermitian symmetrized",
        }
