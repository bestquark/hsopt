# EVOLVE-BLOCK-START
import cupynumeric as np
from code_evolution.evaluator import _make_problem

# from src.mixed_fps.simulate_quantization import *
from src.qm.hamsim import simulate_hamiltonian

NATIVE = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}
FMT = {"fp8_e4m3": dict(exp_bits=4, man_bits=3, bias=7)}

# Explicit precision ranking (higher = more precise)
PREC_RANK = {"fp8_e4m3": 0, "fp16": 1, "fp32": 2, "fp64": 3}


def _ordered_formats():
    names = list(FMT.keys()) + list(NATIVE.keys())
    names.sort(key=lambda s: PREC_RANK.get(s, 0))
    return names


def choose_formats(n_terms: int, precision_weights=None):
    """
    Return a list of fmt strings length n_terms.
    If precision_weights is None, default to a medium format (“fp32”).
    If provided, lower weights -> lower precision (more aggressive compression).
    """
    formats = _ordered_formats()  # e.g. ["fp8_e4m3", "fp16", "fp32", "fp64"]
    if precision_weights is None:
        return ["fp32"] * n_terms

    # Map weights to indices in the ordered list (wrap if fewer formats than terms)
    order = np.argsort(np.asarray(precision_weights))[:n_terms]
    return [formats[i % len(formats)] for i in order]


def choose_steps(t: float, H_terms):
    n = H_terms[0].shape[0]
    # simple heuristic; keep >=1
    r = max(1, int(t * float(np.linalg.norm(H_terms[0])) * n / 10))
    return r


# EVOLVE-BLOCK-END


def run(H_terms, psi0, t):
    fmts = choose_formats(len(H_terms))  # now valid: default branch
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
