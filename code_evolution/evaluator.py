import importlib.util
import time
import os

# import cupynumeric as np
import numpy as np
import traceback

# ============ problem generator & references ============


def _make_problem(n=8, seed=123):
    rng = np.random.default_rng(seed)  # NumPy RNG
    # Hermitian terms H_j (n x n) + random normalized state psi0
    H_terms = []
    for _ in range(3):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2.0
        H_terms.append(H)
    psi0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    psi0 = psi0 / np.linalg.norm(psi0)
    return H_terms, psi0


def _exact_hermitian(H_terms, psi0, t):
    # exact via diagonalization of full H = sum H_j
    H = H_terms[0].copy()
    for j in range(1, len(H_terms)):
        H += H_terms[j]
    w, Q = np.linalg.eigh(H)
    phase = np.exp((-1j * t) * w.astype(np.complex128))
    y = Q.conj().T @ psi0
    return Q @ (y * phase)


def _fid(ref, out):
    ref = ref / np.linalg.norm(ref)
    out = out / np.linalg.norm(out)
    return float(np.abs(np.vdot(ref, out)) ** 2)


# ============ dynamic import ============


def _load(program_path):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # return dict of callables/objs for safety
    return {k: getattr(mod, k) for k in dir(mod)}


# ============ weights & scaling ============

# Cap memory when normalizing so it doesn't dominate (tune to your scale).
# If using true MB, 256 MB is a safe default for small problems.
MEM_CAP_MB = float(os.getenv("OE_MEM_CAP_MB", "256"))  # env override ok
LAMBDA_W = float(os.getenv("OE_LAMBDA", "0.5"))  # 0..1, sweep to trace Pareto

# ============ Stage 1: quick gate ============


def evaluate_stage1(program_path, n=8, t=0.2, seed=123):
    mod = _load(program_path)
    if "run" not in mod:
        return {"stage1_passed": 0.0, "error": "no_run_fn"}

    H_terms, psi0 = _make_problem(n=n, seed=seed)
    try:
        psi_final, meta = mod["run"](H_terms, psi0, t)
    except Exception as e:
        return {"stage1_passed": 0.0, "error": f"runtime_error:{type(e).__name__}"}

    psi_ref = _exact_hermitian(H_terms, psi0, t)
    fid = _fid(psi_ref, psi_final)
    nrm = float(np.linalg.norm(psi_final))

    passed = 1.0 if (fid >= 0.90 and abs(nrm - 1.0) < 5e-3) else 0.0
    return {"stage1_passed": passed, "combined_score": fid, "norm_s1": nrm}


# ============ Full evaluation: accuracy + memory ============


def evaluate(program_path, trials=5, n=8, t=0.5, seeds=None):
    mod = _load(program_path)
    if "run" not in mod:
        return {"combined_score": -1e9, "error": "no_run_fn"}

    seeds = seeds or [1337 + i for i in range(trials)]
    fids, times_ms, norms, mem_mb = [], [], [], []

    for sd in seeds:
        H_terms, psi0 = _make_problem(n=n, seed=sd)
        t0 = time.perf_counter()
        try:
            res = mod["run"](H_terms, psi0, t)
            psi_final, meta = res
        except Exception as e:
            traceback.print_exc()
            return {
                "combined_score": -1e9,
                "error": f"runtime_error:{type(e).__name__}",
            }
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        psi_ref = _exact_hermitian(H_terms, psi0, t)

        fids.append(_fid(psi_ref, psi_final))
        norms.append(float(np.linalg.norm(psi_final)))

        # memory in MB (reported by simulator)
        bu = float(meta.get("bytes_used", 0.0))
        mem_mb.append(bu / (1024.0 * 1024.0))

    fid_mean = float(np.mean(fids))
    fid_min = float(np.min(fids))
    t_mean = float(np.mean(times_ms))
    norm_dev = float(abs(np.mean(norms) - 1.0))
    mem_mb_mean = float(np.mean(mem_mb)) if mem_mb else 0.0

    # scalarization: higher is better
    acc = fid_mean
    # mem_norm = min(mem_mb_mean, MEM_CAP_MB) / MEM_CAP_MB
    mem_norm = mem_mb_mean
    combined = LAMBDA_W * acc - (1.0 - LAMBDA_W) * mem_norm

    result = {
        "combined_score": float(combined),
        "acc": float(acc),
        "mem_norm": float(mem_norm),
        "mem_raw": float(mem_mb_mean),
        "fid_mean": float(fid_mean),
        "fid_min": float(fid_min),
        "time_ms_mean": float(t_mean),
        "norm_dev": float(norm_dev),
        "trials": int(len(seeds)),
    }
    assert "combined_score" in result and isinstance(result["combined_score"], float), (
        f"combined_score missing or not float: {result}"
    )
    return result


# keep evaluate_stage2 for cascade setups (optional)
def evaluate_stage2(program_path, **kw):
    return evaluate(program_path, **kw)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a candidate program.")
    parser.add_argument("program", type=str, help="Path to candidate program .py file")
    parser.add_argument(
        "--stage1", action="store_true", help="Run quick gate (stage 1) evaluation"
    )
    parser.add_argument(
        "--n", type=int, default=8, help="Number of Trotter terms (default: 8)"
    )
    parser.add_argument(
        "--t", type=float, default=0.5, help="Time parameter t (default: 0.5)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials for full eval (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )
    args = parser.parse_args()

    if args.stage1:
        result = evaluate_stage1(args.program, n=args.n, t=0.2, seed=args.seed)
    else:
        result = evaluate(args.program, trials=args.trials, n=args.n, t=args.t)

    print(result)
