import pickle
import sys
import cupynumeric as np

def initial_state_prep(n):
    # define psi here; this function body will be copied to temp.py
    import cupynumeric as np
    psi0 = np.zeros(n, dtype=np.complex64)
    psi0[0] = 1.0
    return psi0

# EVOLVE-BLOCK-START

def trotter_time_evol(H, psi0, t, r):
    # this body will be copied, calling the hamsim function
    psi_final, meta = trv.qm.hamsim.simulate_hamiltonian_no_simulated_precision(H, psi0, t, r)
    return psi_final, meta

# EVOLVE-BLOCK-END

def run_dynamics():
    # whatever main entry point you want in the generated file
    pass

