"""
Base dynamics primitives for trovolut.

This module defines:
- ``BaseDynamics``: the abstract interface that all backend dynamics implement.
- ``ProductFormula`` hierarchy: hardware-agnostic Suzuki-Trotter product
  formulas that describe *which* terms to exponentiate and *for how long*,
  without prescribing *how* the exponential is computed.

Product formulas:
    ProductFormula  -- abstract base class
    LieTrotter      -- first-order Lie-Trotter splitting  (order 1)
    SuzukiTrotter   -- symmetric second-order splitting   (order 2)
    SuzukiHighOrder -- recursive Suzuki for order 4, 6, 8, ...
    suzuki(order)   -- convenience factory for any valid order
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from trovolut.core.types import StateVector, HamiltonianType, DynamicsConfig


# ============================================================
# BaseDynamics
# ============================================================

class BaseDynamics(ABC):
    """
    Abstract base class for quantum dynamics implementations.

    All backend-specific dynamics classes (CPU, GPU, QPU, Emulator) must inherit
    from this class and implement the evolve() method.

    The dynamics class is responsible for computing time evolution of quantum states:
        |psi(t)> = exp(-i H t) |psi(0)>

    Different backends may use different methods:
        - CPU: scipy.linalg.expm, Krylov, etc.
        - GPU: cupynumeric/CUDA Quantum matrix exponentials
        - QPU: Trotterization and circuit compilation
        - Emulator: Software simulation of GPU behavior

    Hamiltonian Input Formats:
        The hamiltonian argument can be any of:
        - Hamiltonian object (from trovolut.qm.hamiltonians)
        - HamiltonianEnsemble object (stochastic, use .sample(rng) first)
        - List of numpy arrays [H1, H2, ...] (decomposed terms)
        - Single numpy array (full Hamiltonian matrix)
    """

    @abstractmethod
    def evolve(
        self,
        hamiltonian: HamiltonianType,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        **kwargs,
    ) -> StateVector:
        """
        Evolve a quantum state under a Hamiltonian.

        Args:
            hamiltonian: Hamiltonian operator in any supported format:
                - Hamiltonian object: Use .terms() or .to_matrix()
                - HamiltonianEnsemble: Sample first with .sample(rng)
                - List of arrays: [H1, H2, ...] for Trotterization
                - Single array: Full Hamiltonian matrix
            initial_state: Initial quantum state vector
            time: Evolution time (dimensionless, scaled by hbar=1)
            config: Optional configuration for the dynamics method
            **kwargs: Backend-specific options

        Returns:
            Final quantum state vector after time evolution

        Raises:
            RuntimeError: If evolution fails

        Note:
            Backends should handle different Hamiltonian formats gracefully.
            For Hamiltonian objects, use .terms() for decomposed methods or
            .to_matrix() for exact evolution.
        """
        pass


# ============================================================
# Product formulas
# ============================================================

class ProductFormula(ABC):
    """
    Abstract base for product-formula decompositions.

    A product formula describes *which* terms to exponentiate and *for how
    long*, but says nothing about *how* the exponential is computed.  This
    makes it backend-agnostic: the same formula object can drive a CPU
    ``scipy.linalg.expm`` pipeline, a mixed-precision GPU emulator, or a
    circuit compiler.

    Subclasses must implement:

    - ``order`` (property): the asymptotic error order of the formula.
    - ``get_sequence(n_terms, dt)``: returns a list of ``(term_index, scaled_dt)``
      pairs that define one complete step of the product formula.
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Asymptotic error order of the formula."""
        ...

    @abstractmethod
    def get_sequence(
        self, n_terms: int, dt: float
    ) -> list[tuple[int, float]]:
        """
        Return the term-application sequence for one product-formula step.

        Parameters
        ----------
        n_terms : int
            Number of Hamiltonian terms (indexed 0 .. n_terms-1).
        dt : float
            Total time step for this single Trotter step.

        Returns
        -------
        list of (term_index, scaled_dt)
            Ordered sequence of (j, tau) meaning "apply exp(-i H_j tau)".
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(order={self.order})"


# -- Concrete formulas --------------------------------------------------------

class LieTrotter(ProductFormula):
    """
    First-order Lie-Trotter product formula.

    S_1(dt) = prod_{j=0}^{k-1} exp(-i H_j dt)

    Error per step: O(dt^2).
    """

    @property
    def order(self) -> int:
        return 1

    def get_sequence(self, n_terms: int, dt: float) -> list[tuple[int, float]]:
        return [(j, dt) for j in range(n_terms)]


class SuzukiTrotter(ProductFormula):
    """
    Symmetric second-order Suzuki-Trotter product formula.

    S_2(dt) = prod_{j=0}^{k-2} exp(-i H_j dt/2)
              . exp(-i H_{k-1} dt)
              . prod_{j=k-2}^{0} exp(-i H_j dt/2)

    Error per step: O(dt^3).
    """

    @property
    def order(self) -> int:
        return 2

    def get_sequence(self, n_terms: int, dt: float) -> list[tuple[int, float]]:
        if n_terms == 0:
            return []
        if n_terms == 1:
            return [(0, dt)]
        seq: list[tuple[int, float]] = []
        # Forward half-steps
        for j in range(n_terms - 1):
            seq.append((j, dt / 2))
        # Full step for last term
        seq.append((n_terms - 1, dt))
        # Reverse half-steps
        for j in range(n_terms - 2, -1, -1):
            seq.append((j, dt / 2))
        return seq


class SuzukiHighOrder(ProductFormula):
    """
    Higher even-order Suzuki product formula (order 4, 6, 8, ...).

    Uses the recursive fractal construction:

        S_{2k}(dt) = S_{2k-2}(p dt)^2  .  S_{2k-2}((1-4p) dt)  .  S_{2k-2}(p dt)^2

    where  p_k = 1 / (4 - 4^{1/(2k-1)}).

    Parameters
    ----------
    target_order : int
        Desired formula order (must be even and >= 4).
    """

    def __init__(self, target_order: int = 4):
        if target_order < 4 or target_order % 2 != 0:
            raise ValueError(
                f"SuzukiHighOrder requires even order >= 4, got {target_order}"
            )
        self._order = target_order

    @property
    def order(self) -> int:
        return self._order

    def get_sequence(self, n_terms: int, dt: float) -> list[tuple[int, float]]:
        return self._recursive_sequence(n_terms, dt, self._order)

    @staticmethod
    def _recursive_sequence(
        n_terms: int, dt: float, current_order: int
    ) -> list[tuple[int, float]]:
        if current_order == 2:
            return SuzukiTrotter().get_sequence(n_terms, dt)
        if current_order == 1:
            return LieTrotter().get_sequence(n_terms, dt)

        k = current_order // 2
        p = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k - 1)))

        sub = current_order - 2
        seq: list[tuple[int, float]] = []
        # S_{2k-2}(p dt)  twice
        for _ in range(2):
            seq.extend(
                SuzukiHighOrder._recursive_sequence(n_terms, p * dt, sub)
            )
        # S_{2k-2}((1-4p) dt)  once
        seq.extend(
            SuzukiHighOrder._recursive_sequence(n_terms, (1.0 - 4.0 * p) * dt, sub)
        )
        # S_{2k-2}(p dt)  twice
        for _ in range(2):
            seq.extend(
                SuzukiHighOrder._recursive_sequence(n_terms, p * dt, sub)
            )
        return seq

    def __repr__(self) -> str:
        return f"SuzukiHighOrder(order={self._order})"


# -- Convenience factory ------------------------------------------------------

def suzuki(order: int = 2) -> ProductFormula:
    """
    Return a Suzuki product formula of the requested order.

    Parameters
    ----------
    order : int
        1  -> LieTrotter  (first-order)
        2  -> SuzukiTrotter  (symmetric second-order)
        4, 6, 8, ... -> SuzukiHighOrder (recursive higher-order)

    Raises
    ------
    ValueError
        If order is not 1, 2, or an even integer >= 4.
    """
    if order == 1:
        return LieTrotter()
    if order == 2:
        return SuzukiTrotter()
    if order >= 4 and order % 2 == 0:
        return SuzukiHighOrder(target_order=order)
    raise ValueError(
        f"Suzuki order must be 1, 2, or an even integer >= 4, got {order}"
    )


__all__ = [
    "BaseDynamics",
    "ProductFormula",
    "LieTrotter",
    "SuzukiTrotter",
    "SuzukiHighOrder",
    "suzuki",
]
