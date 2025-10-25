from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Optional
import numpy as np


# -------------------------------------------------
# Hamiltonian: concrete, deterministic operator
# -------------------------------------------------


class Hamiltonian(ABC):
    """
    A concrete Hermitian operator H on some Hilbert space.
    """

    is_stochastic: bool = False  # invariant

    @property
    @abstractmethod
    def hilbert_dim(self) -> int:
        """Hilbert space dimension (matrix size)."""
        raise NotImplementedError

    @abstractmethod
    def to_matrix(self) -> np.ndarray:
        """Return the full (dim x dim) dense matrix representation."""
        raise NotImplementedError

    @abstractmethod
    def terms(self) -> Sequence[Dict[str, np.ndarray]]:
        """
        Return explicit additive decomposition:

            H = sum_i coef_i * H_i

        as a list like:
            [{"coef": scalar, "matrix": Hermitian np.ndarray}, ...]

        If there’s no natural decomposition:
            return [{"coef": 1.0, "matrix": self.to_matrix()}]
        """
        raise NotImplementedError

    @abstractmethod
    def decompose(self) -> Dict[str, Any]:
        """Return lightweight structural metadata (no large arrays)."""
        raise NotImplementedError

    def sample(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Hamiltonian:
        """Deterministic Hamiltonians are already concrete."""
        return self


# -------------------------------------------------
# HamiltonianEnsemble: distribution over Hamiltonians
# -------------------------------------------------


class HamiltonianEnsemble(ABC):
    """
    A stochastic family of Hamiltonians.
    Use .sample() to draw a concrete Hamiltonian instance.
    """

    is_stochastic: bool = True  # invariant

    @property
    @abstractmethod
    def hilbert_dim(self) -> int:
        """Hilbert-space dimension of each draw."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Hamiltonian:
        """Draw one concrete Hamiltonian realization."""
        raise NotImplementedError

    @abstractmethod
    def decompose(self) -> Dict[str, Any]:
        """Return metadata describing the ensemble (no actual samples)."""
        raise NotImplementedError
