from __future__ import annotations
import sys
import pickle
import argparse
import cupynumeric as np  
from trovolut.qm.hamsim import simulate_hamiltonian

FMT = {
    "fp8_e5m2": dict(exp_bits=5, man_bits=2, bias=15),
    "fp8_e4m3": dict(exp_bits=4, man_bits=3, bias=7),
    "fp8_e3m4": dict(exp_bits=3, man_bits=4, bias=3),
    "fp8_e2m5": dict(exp_bits=2, man_bits=5, bias=1),
    "fp16": np.float16,
    "fp32": np.float32,
    "fp64": np.float64,
}

# EVOLVE-BLOCK-START
def choose_formats(H_terms):
    """Return per-term precision choices. Keep it simple by default."""
    return ["fp16"] * len(H_terms)

def choose_steps(t: float, H_terms):
    """Return Trotter step count. Simple fixed policy by default."""
    return 1000
# EVOLVE-BLOCK-END

def run(H_terms, psi0, t):
    """
    Evolve psi0 under H(t) = sum_j H_terms[j] for total time t.
    Returns (psi_final, meta).
    """
    # Ensure cupynumeric arrays
    H_terms = [np.array(H, copy=False) for H in H_terms]
    psi0 = np.array(psi0, copy=False)

    fmts = choose_formats(H_terms)
    r = choose_steps(t, H_terms)

    psi_final, meta = simulate_hamiltonian(
        H_terms,
        psi0,
        t,
        r,
        fmt_list=fmts,
        compute_dtype=np.float32,
        work_dtype=np.float32,
        out_dtype=np.complex64,
        verbose=False,
    )
    return psi_final, meta
