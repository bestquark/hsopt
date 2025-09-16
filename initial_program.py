# EVOLVE-BLOCK-START
import cupynumeric as np
from src.simulate_quantization import *
from src.hamsim import *

NATIVE = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}
FMT = {"fp8_e4m3": dict(exp_bits=4, man_bits=3, bias=7)}


def choose_formats(n_terms: int):
    """
    Precision policy for the sequence of terms.
    This is the main target for evolution.
    Return a list of fmt strings for mixed precision: e.g., ["fp16","fp32","fp8_e4m3"].
    """
    base = ["fp8_e4m3", "fp8_e4m3", "fp8_e4m3", "fp8_e4m3"]
    fmts = [base[i % len(base)] for i in range(n_terms)]
    return fmts


def choose_steps(t: float, H_terms):
    """
    Choose Trotter steps r (trade accuracy vs runtime). Target for evolution.
    """
    n = H_terms[0].shape[0]
    return max(4, min(64, int(0.5 * n + 6)))


# EVOLVE-BLOCK-END


def run(H_terms, psi0, t):
    """Entry point used by evaluator. Returns psi_final and metadata."""
    fmts = choose_formats(len(H_terms))
    r = choose_steps(t, H_terms)
    psi_final = simulate_hamiltonian(
        H_terms,
        psi0,
        t,
        r,
        fmts,
        normalize=True,
        compute_dtype=np.complex128,
        work_dtype=np.complex128,
        out_dtype=np.complex128,
        verbose=False,
    )
    return psi_final, {"r": r, "fmts": fmts}
