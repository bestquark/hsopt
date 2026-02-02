"""
CPU dynamics implementations using NumPy and SciPy.

This is the default backend using standard Python scientific libraries.
"""

import numpy as np
import scipy.linalg
from trovolut.core.base_dynamics import BaseDynamics


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
    if order == 1:
        U = np.eye(H_terms[0].shape[0], dtype=complex)
        for H_j in H_terms:
            U_j = scipy.linalg.expm(-1j * H_j * dt)
            U = U_j @ U
        return U

    if order % 2 == 1:
        raise ValueError(f"Suzuki formulas are defined for order 1 or even orders >= 2, got {order}.")

    if order == 2:
        # Symmetric 2nd-order: ∏_j e^{-i H_j dt/2} ∏_j^rev e^{-i H_j dt/2}
        half = dt / 2.0
        U = np.eye(H_terms[0].shape[0], dtype=complex)
        for H_j in H_terms:
            U_j = scipy.linalg.expm(-1j * H_j * half)
            U = U_j @ U
        for H_j in reversed(H_terms):
            U_j = scipy.linalg.expm(-1j * H_j * half)
            U = U_j @ U
        return U

    # Recursive Suzuki: S_{2k}(dt) = S_{2k-2}(p_k dt)^2 S_{2k-2}((1-4 p_k) dt) S_{2k-2}(p_k dt)^2
    k = order // 2
    p_k = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k - 1)))

    U_p = suzuki_step(H_terms, p_k * dt, order=order - 2)
    U_m = suzuki_step(H_terms, (1.0 - 4.0 * p_k) * dt, order=order - 2)

    return U_p @ U_p @ U_m @ U_p @ U_p


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
    U = np.eye(H_terms[0].shape[0], dtype=complex)
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
    dt_eff = t * lam / N  # effective time per sampled normalized term

    U = np.eye(d, dtype=complex)
    for _ in range(N):
        j = np.random.choice(len(H_terms), p=probs)
        G_j = H_terms[j] / norms[j]  # normalized term so that H = sum a_j G_j
        U_step = scipy.linalg.expm(-1j * G_j * dt_eff)
        U = U_step @ U
    return U


class ExactEvolution(BaseDynamics):
    """
    Exact evolution using scipy.linalg.expm.
    Computes U = e^{-i H t} exactly. Best for small systems and benchmarking.
    """
    
    def __init__(self):
        """Initialize exact evolution dynamics."""
        pass
    
    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        """
        Evolve quantum state using exact matrix exponential.
        
        Args:
            hamiltonian: Hamiltonian operator (NumPy array)
            initial_state: Initial quantum state (NumPy array)
            time: Evolution time
            **kwargs: Additional options (unused)
            
        Returns:
            Final quantum state |ψ(t)⟩ = e^{-i H t} |ψ(0)⟩
        """
        U = scipy.linalg.expm(-1j * hamiltonian * time)
        return U @ initial_state


class TrotterEvolution(BaseDynamics):
    """
    Suzuki-Trotter product formula evolution.
    Approximates e^{-i H t} using U ≈ [∏_j e^{-i H_j dt}]^r.
    Supports Suzuki orders: 1 (Lie-Trotter), 2 (symmetric), 4, 6, ...
    """
    
    def __init__(self, trotter_steps=10, suzuki_order=2):
        """
        Initialize Trotter evolution.
        
        Args:
            trotter_steps: Number of Trotter steps (default: 10)
            suzuki_order: Suzuki order (default: 2)
        """
        self.trotter_steps = trotter_steps
        self.suzuki_order = suzuki_order
    
    def evolve(self, hamiltonian, initial_state, time, **kwargs):
        """
        Evolve quantum state using Suzuki-Trotter formula.
        
        Args:
            hamiltonian: List of Hamiltonian terms [H_1, ..., H_k] or single array
            initial_state: Initial quantum state (NumPy array)
            time: Evolution time
            **kwargs: Optional overrides for trotter_steps, suzuki_order
            
        Returns:
            Final quantum state approximating |ψ(t)⟩ = e^{-i H t} |ψ(0)⟩
        """
        if not isinstance(hamiltonian, list):
            hamiltonian = [hamiltonian]
        
        steps = kwargs.get("trotter_steps", self.trotter_steps)
        order = kwargs.get("suzuki_order", self.suzuki_order)
        
        U = product_formula(hamiltonian, time, steps, order=order)
        return U @ initial_state


class QDriftEvolution(BaseDynamics):
    """
    qDRIFT randomized product formula evolution.
    Randomly samples Hamiltonian terms proportional to their norms.
    Reference: Campbell, PRL 123.7 (2019): 070503.
    """
    
    def __init__(self, n_samples=100):
        """
        Initialize qDRIFT evolution.
        
        Args:
            n_samples: Number of random samples (default: 100)
        """
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
            Final quantum state approximating |ψ(t)⟩ = e^{-i H t} |ψ(0)⟩
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
    "qdrift"
]
