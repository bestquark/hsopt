import pickle
import sys
import cupynumeric as np
from evaluator import _make_problem

# from src.mixed_fps.simulate_quantization import *
from trovolut.backends.gpu_emulator.hamsim import simulate_hamiltonian


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
    return ["fp16"] * len(n_terms)


def choose_steps(t: float, H_terms):
    """
    Input: t (float): The total time for evolution (between 0.01 and 1.0).
    Input: H_terms (List[matrices]): A list of the terms H_j as NDArrays to be time evolved using Trotterization.
    Output: r (int): The number of Trotter steps to use.
    """
    n = H_terms[0].shape[0]
    r = 1000
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


def main():
    # --- Mode 1: piped pickle over stdin (works for plain python) ---
    if not sys.stdin.isatty() and "--pickle" not in sys.argv:
        try:
            H_terms, psi0, t = pickle.load(sys.stdin.buffer)
            psi_final, meta = run(H_terms, psi0, t)
            pickle.dump((psi_final, meta), sys.stdout.buffer)
            return 0
        except Exception as e:
            print(f"[pickle stdin error] {e}", file=sys.stderr)
            return 1

    # --- Mode 2: file-based pickle (works under Legate) ---
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pickle",
        type=str,
        default=None,
        help="Path to input pickle containing (H_terms, psi0, t).",
    )
    # (optional normal CLI args here if you still want them)
    args, _ = p.parse_known_args()

    if args.pickle:
        try:
            with open(args.pickle, "rb") as f:
                H_terms, psi0, t = pickle.load(f)
            # If you’re using cupynumeric in run(), convert here:
            import cupynumeric as np

            H_terms = [np.array(H, copy=False) for H in H_terms]
            psi0 = np.array(psi0, copy=False)

            psi_final, meta = run(H_terms, psi0, t)
            # Write result as pickle to stdout (no extra prints!)
            pickle.dump((psi_final, meta), sys.stdout.buffer)
            return 0
        except Exception as e:
            print(f"[pickle file error] {e}", ufile=sys.stderr)
            return 1

    # Fallback: your original CLI path or an error
    print("[error] no pickle provided and no stdin data", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import argparse
    import time

    # --- Mode 1: piped pickle input ---
    if not sys.stdin.isatty():
        try:
            sys.exit(main())
        except Exception as e:
            print(f"[main error] {e}", file=sys.stderr)
            sys.exit(1)

    # --- Mode 2: normal CLI (n, m, t, seed) ---
    p = argparse.ArgumentParser(description="Run simulation or accept pickled input.")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--t", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    H_terms, psi0 = _make_problem(args.n, args.m, args.seed)
    psi_exact = np.linalg.expm(-1j * args.t * sum(H_terms)) @ psi0

    t0 = time.perf_counter()
    psi_final, meta = run(H_terms, psi0, args.t)
    elapsed = (time.perf_counter() - t0) * 1000

    err = np.linalg.norm(psi_final - psi_exact)
    print(f"Result error: {err:.6f}")
    print(f"Elapsed time: {elapsed:.1f} ms")
    print(f"Meta: {meta}")
