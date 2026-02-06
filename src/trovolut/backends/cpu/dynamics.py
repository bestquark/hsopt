"""
CPU dynamics implementations using NumPy and SciPy.

This is the default backend using standard Python scientific libraries.
All product-formula logic delegates to ``trovolut.core.base_dynamics``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from trovolut.core.base_dynamics import BaseDynamics, ProductFormula, suzuki


# ============================================================
# Standalone helpers (backward-compatible public API)
# ============================================================

def suzuki_step(H_terms, dt, order=1):
    """
    Single-step unitary for e^{-i H dt} with Suzuki formula of given order.

    Args:
        H_terms: List of Hamiltonian terms (each a NumPy array)
        dt: Time step
        order: Suzuki order (1 for Lie-Trotter, 2 for symmetric, 4/6/... for higher)

    Returns:
        Unitary matrix U approximating e^{-i H dt}

    Notes:
        - order = 1: first-order Lie-Trotter
        - order = 2: symmetric second-order Suzuki
        - order = 2k (k>=2): recursive Suzuki construction
    """
    formula = suzuki(order)
    sequence = formula.get_sequence(len(H_terms), dt)

    d = H_terms[0].shape[0]
    U = np.eye(d, dtype=complex)
    for j, tau in sequence:
        U_j = scipy.linalg.expm(-1j * H_terms[j] * tau)
        U = U_j @ U
    return U


def product_formula(H_terms, t, r, order=1):
    """
    Product-formula (Trotterization) approximation to e^{-i H t}, where H = sum_j H_j.

    Args:
        H_terms: List of Hamiltonian terms (each a NumPy array)
        t: Total evolution time
        r: Number of Trotter steps
        order: Suzuki order (1, 2, 4, 6, ...)

    Returns:
        Unitary matrix U approximating e^{-i H t}
    """
    dt = t / r
    d = H_terms[0].shape[0]
    U = np.eye(d, dtype=complex)
    for _ in range(r):
        U_step = suzuki_step(H_terms, dt, order=order)
        U = U_step @ U
    return U


def qdrift(H_terms, t, N):
    """
    qDRIFT random product formula approximation to e^{-i H t}, where H = sum_j H_j.

    Args:
        H_terms: List of Hamiltonian terms (each a NumPy array)
        t: Total evolution time
        N: Number of sampled exponentials

    Returns:
        Unitary matrix U approximating e^{-i H t}

    Reference:
        Campbell, Earl T. "Random compiler for fast Hamiltonian simulation."
        Physical Review Letters 123.7 (2019): 070503.
    """
    d = H_terms[0].shape[0]
    norms = np.array([np.linalg.norm(H_j, ord=2) for H_j in H_terms], dtype=float)
    lam = norms.sum()
    if lam == 0:
        return np.eye(d, dtype=complex)

    probs = norms / lam
    dt_eff = t * lam / N

    U = np.eye(d, dtype=complex)
    for _ in range(N):
        j = np.random.choice(len(H_terms), p=probs)
        G_j = H_terms[j] / norms[j]
        U_step = scipy.linalg.expm(-1j * G_j * dt_eff)
        U = U_step @ U
    return U


# ============================================================
# Evolution classes
# ============================================================

class ExactEvolution(BaseDynamics):
    """
    Exact evolution using scipy.linalg.expm.
    Computes U = e^{-i H t} exactly. Best for small systems and benchmarking.
    """

    def __init__(self):
        pass

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        """
        Evolve quantum state using exact matrix exponential.

        Args:
            hamiltonian: Hamiltonian operator (NumPy array)
            initial_state: Initial quantum state (NumPy array)
            time: Evolution time

        Returns:
            Final quantum state |psi(t)> = e^{-i H t} |psi(0)>
        """
        U = scipy.linalg.expm(-1j * hamiltonian * time)
        return U @ initial_state


class TrotterEvolution(BaseDynamics):
    """
    Suzuki-Trotter product formula evolution.
    Approximates e^{-i H t} using U ~ [prod_j e^{-i H_j dt}]^r.
    Supports Suzuki orders: 1 (Lie-Trotter), 2 (symmetric), 4, 6, ...

    Parameters
    ----------
    trotter_steps : int
        Number of Trotter steps (default 10).
    suzuki_order : int
        Suzuki order (default 2). Ignored if ``formula`` is provided.
    formula : ProductFormula | None
        Explicit product formula instance.  Takes precedence over
        ``suzuki_order`` when provided.
    """

    def __init__(
        self,
        trotter_steps: int = 10,
        suzuki_order: int = 2,
        formula: ProductFormula | None = None,
    ):
        self.trotter_steps = trotter_steps
        if formula is not None:
            self.formula = formula
        else:
            self.formula = suzuki(suzuki_order)

    @property
    def suzuki_order(self) -> int:
        """Backward-compatible accessor for the formula order."""
        return self.formula.order

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        """
        Evolve quantum state using Suzuki-Trotter formula.

        Args:
            hamiltonian: List of Hamiltonian terms [H_1, ..., H_k] or single array
            initial_state: Initial quantum state (NumPy array)
            time: Evolution time
            **kwargs: Optional overrides for trotter_steps, suzuki_order, formula

        Returns:
            Final quantum state approximating |psi(t)> = e^{-i H t} |psi(0)>
        """
        if not isinstance(hamiltonian, list):
            hamiltonian = [hamiltonian]

        steps = kwargs.get("trotter_steps", self.trotter_steps)

        # Allow runtime override of formula or suzuki_order
        formula = kwargs.get("formula", None)
        if formula is None:
            override_order = kwargs.get("suzuki_order", None)
            if override_order is not None:
                formula = suzuki(override_order)
            else:
                formula = self.formula

        dt = time / steps
        n_terms = len(hamiltonian)
        d = hamiltonian[0].shape[0]

        U = np.eye(d, dtype=complex)
        for _ in range(steps):
            for j, tau in formula.get_sequence(n_terms, dt):
                U_j = scipy.linalg.expm(-1j * hamiltonian[j] * tau)
                U = U_j @ U

        return U @ initial_state


class QDriftEvolution(BaseDynamics):
    """
    qDRIFT randomized product formula evolution.
    Randomly samples Hamiltonian terms proportional to their norms.
    Reference: Campbell, PRL 123.7 (2019): 070503.
    """

    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples

    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        """
        Evolve quantum state using qDRIFT randomized compilation.

        Args:
            hamiltonian: List of Hamiltonian terms [H_1, ..., H_k] or single array
            initial_state: Initial quantum state (NumPy array)
            time: Evolution time
            **kwargs: Optional override for n_samples

        Returns:
            Final quantum state approximating |psi(t)> = e^{-i H t} |psi(0)>
        """
        if not isinstance(hamiltonian, list):
            hamiltonian = [hamiltonian]

        samples = kwargs.get("n_samples", self.n_samples)

        U = qdrift(hamiltonian, time, samples)
        return U @ initial_state


__all__ = [
    "ExactEvolution",
    "TrotterEvolution",
    "QDriftEvolution",
    "suzuki_step",
    "product_formula",
    "qdrift",
]
