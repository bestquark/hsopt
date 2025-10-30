"""Common interfaces for deterministic Hamiltonians and Hamiltonian ensembles."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Optional

import numpy as np


class Hamiltonian(ABC):
    """Abstract base class for deterministic Hermitian operators.

    Attributes:
        is_stochastic (bool): Flag used by downstream code to distinguish
            concrete Hamiltonians from stochastic ensembles.
    """

    is_stochastic: bool = False  # invariant

    @property
    @abstractmethod
    def hilbert_dim(self) -> int:
        """int: Hilbert-space dimension (matrix size)."""
        raise NotImplementedError

    @abstractmethod
    def to_matrix(self) -> np.ndarray:
        """Return the dense matrix representation.

        Returns:
            np.ndarray: Full ``(dim, dim)`` complex Hermitian matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def terms(self) -> Sequence[Dict[str, np.ndarray]]:
        """Return an additive decomposition of the Hamiltonian.

        Returns:
            Sequence[Dict[str, np.ndarray]]:
                Terms of the form ``{"coef": scalar, "matrix": Hermitian np.ndarray}``; use ``[{"coef": 1.0, "matrix": self.to_matrix()}]`` when no natural decomposition exists.
        """
        raise NotImplementedError

    @abstractmethod
    def decompose(self) -> Dict[str, Any]:
        """Return metadata describing the Hamiltonian.

        Returns:
            Dict[str, Any]: JSON-serializable metadata without large arrays.
        """
        raise NotImplementedError

    def sample(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Hamiltonian:
        """Return a deterministic Hamiltonian sample.

        Args:
            rng (np.random.Generator | None): Unused RNG placeholder kept for
                interface compatibility.

        Returns:
            Hamiltonian: ``self`` because deterministic Hamiltonians are already concrete.
        """
        return self


# -------------------------------------------------
# HamiltonianEnsemble: distribution over Hamiltonians
# -------------------------------------------------


class HamiltonianEnsemble(ABC):
    """Abstract base class for stochastic Hamiltonian families.

    Use [`~trotvol.qm.hamiltonians._hamiltonian.Hamiltonian.sample`][] to draw a concrete [`~trotvol.qm.hamiltonians._hamiltonian.Hamiltonian`][] instance.
    """

    is_stochastic: bool = True  # invariant

    @property
    @abstractmethod
    def hilbert_dim(self) -> int:
        """int: Hilbert-space dimension shared by all draws."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Hamiltonian:
        """Draw one Hamiltonian from the ensemble.

        Args:
            rng (np.random.Generator | None): Random number generator used to
                parameterize the draw. If ``None`` the implementation must pick
                a sensible default.

        Returns:
            Hamiltonian: Concrete deterministic Hamiltonian sampled from the ensemble.
        """
        raise NotImplementedError

    @abstractmethod
    def decompose(self) -> Dict[str, Any]:
        """Return lightweight metadata about the ensemble.

        Returns:
            Dict[str, Any]: JSON-serializable description of the distribution (no actual samples).
        """
        raise NotImplementedError
