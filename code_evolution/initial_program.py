# EVOLVE-BLOCK-START
import cupynumeric as np
from code_evolution.evaluator import _make_problem

# from src.mixed_fps.simulate_quantization import *
from src.qm.hamsim import simulate_hamiltonian
import random

FMT = {
    "fp4_e2m1": dict(exp_bits=2, man_bits=1, bias=1),
    "fp5_e2m2": dict(exp_bits=2, man_bits=2, bias=1),
    "fp6_e3m2": dict(exp_bits=3, man_bits=2, bias=3),
    "fp7_e3m3": dict(exp_bits=3, man_bits=3, bias=3),
    "fp8_e4m3": dict(exp_bits=4, man_bits=3, bias=7),
    "fp8_e5m2": dict(exp_bits=5, man_bits=2, bias=15),
    "fp16": {"native": np.float16},
    "fp32": {"native": np.float32},
    "fp64": {"native": np.float64},
}
NATIVE = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}


def choose_formats(n_terms: int):
    """
    Precision policy for the sequence of terms.
    This is the main target for evolution.
    Return a list of fmt strings for mixed precision: e.g., ["fp16","fp32","fp8_e4m3"].
    """
    fmts = [random.choice(list(FMT) + list(NATIVE)) for i in range(n_terms)]
    return fmts


def choose_steps(t: float, H_terms):
    """
    Choose Trotter steps r. Target for evolution.
    """
    n = H_terms[0].shape[0]
    r = max(1, int(t * np.linalg.norm(H_terms[0]) * n / 10))
    return r


# EVOLVE-BLOCK-END


def run(H_terms, psi0, t):
    """Entry point used by evaluator. Returns psi_final and metadata."""
    fmts = choose_formats(len(H_terms))
    r = choose_steps(t, H_terms)
    psi_final, meta = simulate_hamiltonian(
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
    return psi_final, meta


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
