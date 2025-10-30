import cupynumeric as np
from code_evolution.evaluator import _make_problem

# from src.mixed_fps.simulate_quantization import *
from src.qm.hamsim import simulate_hamiltonian

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


def choose_formats(n_terms):
    """
    Input: n_terms (List[matrices]): A list of the terms H_j as NDArrays to be time evolved using Trotterization (between 2-5 matrices of size between 10-200).
    Output: a list of fmt strings length n_terms corresponding to the precision to use for each term.

    Can be any of the FMT keys above.
    """
    return ["fp16"] * len(n_terms)  # default branch


def choose_steps(t: float, H_terms):
    """
    Input: t (float): The total time for evolution (between 0.01 and 1.0).
    Input: H_terms (List[matrices]): A list of the terms H_j as NDArrays to be time evolved using Trotterization.
    Output: r (int): The number of Trotter steps to use.
    """
    n = H_terms[0].shape[0]
    r = 100
    return r


# EVOLVE-BLOCK-END


def run(H_terms, psi0, t):
    """
    Performs the Hamiltonian simulation using the chosen formats and steps.

    Higher r => more accurate, but if the quantization is too low-precision, error can increase with r.
    """
    fmts = choose_formats(H_terms)  # now valid: default branch
    r = choose_steps(t, H_terms)
    psi_final, meta = simulate_hamiltonian(
        H_terms,
        psi0,
        t,
        r,
        fmt_list=fmts,  # native 64-bit
        compute_dtype=np.float32,  # feed 64-bit into solver
        work_dtype=np.float32,  # eigensolver in 64-bit
        out_dtype=np.complex64,  # output complex 64-bit
        verbose=False,
    )
    return psi_final, meta


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Run the initial program.")
    parser.add_argument(
        "-n", "--n", type=int, default=100, help="Size of the problem (default: 100)"
    )
    parser.add_argument(
        "-m",
        "--m",
        "--terms",
        type=int,
        default=2,
        help="Number of Trotter terms (default: 8)",
    )
    parser.add_argument(
        "-t", "--time", type=float, default=0.1, help="Time parameter t (default: 0.1)"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")
    args = parser.parse_args()

    n = args.n
    m = args.m
    t = args.time
    seed = args.seed
    H_terms, psi0 = _make_problem(n=n, m=m, seed=seed)
    psi_exact = np.linalg.expm(-1j * t * sum(H_terms)) @ psi0

    t0 = time.perf_counter()
    psi_final, meta = run(H_terms, psi0, t)
    elapsed = time.perf_counter() - t0

    # psi exact

    print(f"Result error: {np.linalg.norm(psi_final - psi_exact):.6f}")
    print(f"Elapsed time: {elapsed * 1000:.1f} ms")
    print(f"Meta: {meta}")
