# EVOLVE-BLOCK-START
import cupynumeric as np
from code_evolution.evaluator import _make_problem
from src.mixed_fps.simulate_quantization import *
from src.qm.hamsim import *

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


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Run the initial program.")
    parser.add_argument(
        "-n",
        "--n",
        "--terms",
        type=int,
        default=4,
        help="Number of Trotter terms (default: 4)",
    )
    parser.add_argument(
        "-t", "--time", type=float, default=0.5, help="Time parameter t (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )
    args = parser.parse_args()

    n = args.n
    t = args.time
    seed = args.seed
    H_terms, psi0 = _make_problem(n=n, seed=seed)

    t0 = time.perf_counter()
    psi_final, meta = run(H_terms, psi0, t)
    elapsed = time.perf_counter() - t0

    print(f"Result norm: {np.linalg.norm(psi_final):.6f}")
    print(f"Elapsed time: {elapsed * 1000:.1f} ms")
    print(f"Meta: {meta}")
