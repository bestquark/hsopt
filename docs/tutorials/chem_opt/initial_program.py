import scipy
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg
import numpy as np

def initialize_flop_counter():
    """Initialize global flop counter."""
    global flop_counter
    flop_counter = FLOPCounter()

# EVOLVE-BLOCK-START

def run_dynamics(hams, t, psi0):
    """
    Standard first-order Trotter evolution.
    Uses dense fp64 matrices - very expensive in FLOPs!
    """
    global flop_counter

    n_steps = 50
    dt = t / n_steps
    psi = psi0.copy()

    for _ in range(n_steps):
        for H in hams:
            # Convert to dense if sparse (initial naive approach)
            if sp.issparse(H):
                H_dense = H.toarray()
            else:
                H_dense = H

            # Compute exponential (expensive!)
            U = scipy.linalg.expm(-1j * H_dense * dt)
            flop_counter.count_expm(H_dense)

            # Apply to state
            psi = U @ psi
            flop_counter.count_matvec(U, psi)

    return psi

# EVOLVE-BLOCK-END

