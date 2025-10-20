# evaluate.py
# Robust evaluator for OpenEvolve candidates running Hamiltonian sims.
# - Isolates each trial in its own process (survives segfaults / native crashes)
# - Supports both fork (preferred on macOS) and spawn
# - Enforces per-trial timeout and (best-effort) memory limits
# - Returns well-formed metrics even on failure

from __future__ import annotations
import os
import sys
import time
import json
import traceback
import subprocess, tempfile, textwrap
from pathlib import Path
import numpy as np
import scipy
import importlib.util
import multiprocessing as mp
from typing import Dict, Any
from gpu_cost import compute_gpu_metrics

# -------- Global knobs (override via env) ----------
MEM_CAP_MB = float(os.getenv("OE_MEM_CAP_MB", "32"))  # for mem_norm cap
LAMBDA_W = float(os.getenv("OE_LAMBDA", "0.5"))  # scalarization weight
ALPHA_GPU = float(os.getenv("OE_ALPHA_GPU", "0.1"))  # GPU time penalty weight
TRIAL_TIMEOUT_S = float(os.getenv("OE_TRIAL_TIMEOUT_S", "30"))  # per-trial wall timeout
PROC_MEM_LIMIT_MB = float(os.getenv("OE_PROC_MEM_MB", "4096"))  # per-trial RSS cap
NUM_THREADS = os.getenv("OE_NUM_THREADS", "1")  # tame BLAS threads
MP_METHOD_DEFAULT = "fork" if sys.platform == "darwin" else "spawn"  # safer on macOS
MP_METHOD = os.getenv(
    "OE_MP_METHOD", MP_METHOD_DEFAULT
)  # override: fork/spawn/forkserver
PROFILE_GPU = os.getenv("OE_PROFILE_GPU", "0") == "1"      # run Nsight Systems
NSYS_PREFIX = os.getenv("OE_NSYS_PREFIX", "")              # e.g., "/usr/local/cuda/bin/"

# ---------------------------------------------------


def _make_problem(n: int = 100, m: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    H_terms = []
    for _ in range(m):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2.0
        H = H * (10.0 if rng.random() < 0.5 else 0.01)  # vary scale
        H_terms.append(H)
    psi0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    psi0 /= np.linalg.norm(psi0)
    return H_terms, psi0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_module_from_file(path: str):
    # Ensure candidate-relative imports like "src.*" work if repo root is project root
    repo_root = os.path.abspath(os.getcwd())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _bytes_of(*objs) -> int:
    total = 0
    for o in objs:
        if isinstance(o, (list, tuple)):
            total += _bytes_of(*o)
        else:
            try:
                total += int(getattr(o, "nbytes"))
            except Exception:
                pass
    return total


def _limit_child_resources():
    # Best-effort: only on Unix
    try:
        import resource

        bytes_limit = int(PROC_MEM_LIMIT_MB * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Extra CPU time guard (2x timeout)
        cpu_sec = max(5, int(TRIAL_TIMEOUT_S * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
    except Exception:
        pass
    # Tame threads
    os.environ.setdefault("OMP_NUM_THREADS", NUM_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", NUM_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", NUM_THREADS)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", NUM_THREADS)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", NUM_THREADS)


def _trial_worker(program_path: str, n: int, m: int, t: float, seed: int, conn):
    # Child process: runs one trial and reports back.
    try:
        _limit_child_resources()
        # Optional: reset SIGSEGV handler to default; we will detect via exit code.
        import signal

        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
    except Exception:
        pass

    try:
        mod = _load_module_from_file(program_path)
        if not hasattr(mod, "run"):
            conn.send((False, {"error": "no_run_fn"}))
            return

        H_terms, psi0 = _make_problem(n=n, m=m, seed=seed)

        t0 = time.perf_counter()
        res = mod.run(H_terms, psi0, t)  # expected: (psi_final, meta)
        psi_final, meta = res
        wall_ms = (time.perf_counter() - t0) * 1000.0

        # Reference (exact) result
        psi_exact = scipy.linalg.expm(-1j * t * sum(H_terms)) @ psi0

        err = float(np.linalg.norm(psi_final - psi_exact))
        norm_val = float(np.linalg.norm(psi_final))
        mem_bytes = _safe_float((meta or {}).get("bytes_used", 0.0))
        if not mem_bytes:
            mem_bytes = _bytes_of(psi_final, psi0, *H_terms)

        payload = {
            "error": _safe_float(err, 1e9),
            "norm": _safe_float(norm_val, 0.0),
            "mem_mb": _safe_float(mem_bytes, 0.0) / (1024.0 * 1024.0),
            "time_ms": _safe_float(wall_ms, TRIAL_TIMEOUT_S * 1000.0),
        }
        conn.send((True, payload))
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        conn.send((False, {"error": f"runtime_error:{type(e).__name__}", "trace": tb}))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_single_trial(
    program_path: str, n: int, m: int, t: float, seed: int, timeout_s: float, *,
    profile_nsys: bool = False
):
    if profile_nsys:
        # Run the exact same trial logic under Nsight Systems
        return _run_single_trial_nsys(program_path, n, m, t, seed, timeout_s)

    # Original mp-based path
    ctx = mp.get_context(MP_METHOD)
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_trial_worker, args=(program_path, n, m, t, seed, child))
    proc.daemon = True
    proc.start()
    child.close()

    ok, payload = False, {"error": "unknown"}
    if parent.poll(timeout_s):
        try:
            ok, payload = parent.recv()
        except EOFError:
            ok, payload = False, {"error": "eof"}
    else:
        try: proc.terminate()
        except Exception: pass
        ok, payload = False, {"error": "timeout"}

    proc.join(timeout=1.0)
    if proc.is_alive():
        try: proc.kill()
        except Exception: pass
        proc.join()

    if not ok and proc.exitcode is not None and proc.exitcode < 0:
        payload.setdefault("error", f"signal_{-proc.exitcode}")
    return ok, payload


def _run_single_trial_nsys(program_path: str, n: int, m: int, t: float, seed: int, timeout_s: float):
    """
    Runs a single trial under Nsight Systems by emitting a tiny runner script,
    executing it via `uv run legate --profile`, and parsing JSON from stdout.
    Returns (ok, payload) with the same schema as the mp worker.
    """
    runner = textwrap.dedent(f"""\
    import os, sys, time, json, traceback
    import numpy as np, scipy.linalg, import importlib.util

    PROC_MEM_LIMIT_MB = float(os.getenv("OE_PROC_MEM_MB","4096"))
    NUM_THREADS = os.getenv("OE_NUM_THREADS","1")

    def _limit():
        try:
            import resource
            b = int(PROC_MEM_LIMIT_MB*1024*1024)
            resource.setrlimit(resource.RLIMIT_AS,(b,b))
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", NUM_THREADS)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", NUM_THREADS)
        os.environ.setdefault("MKL_NUM_THREADS", NUM_THREADS)
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", NUM_THREADS)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", NUM_THREADS)

    def _load_module(path):
        repo_root = os.path.abspath(os.getcwd())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        spec = importlib.util.spec_from_file_location("candidate", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _bytes_of(*objs):
        tot=0
        for o in objs:
            if isinstance(o,(list,tuple)):
                tot+=_bytes_of(*o)
            else:
                try: tot+=int(getattr(o,"nbytes"))
                except Exception: pass
        return tot

    def _make_problem(n,m,seed):
        rng = np.random.default_rng(seed)
        H_terms=[]
        for _ in range(m):
            A = rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n))
            H = (A + A.conj().T)/2.0
            H = H*(10.0 if rng.random()<0.5 else 0.01)
            H_terms.append(H)
        psi0 = rng.standard_normal(n) + 1j*rng.standard_normal(n)
        psi0 /= np.linalg.norm(psi0)
        return H_terms, psi0

    def _safe_float(x,d=0.0):
        try:
            v=float(x); import numpy as _np
            return v if _np.isfinite(v) else d
        except Exception:
            return d

    def main():
        _limit()
        program = os.environ["OE_PROG"]
        n = int(os.environ["OE_N"]); m = int(os.environ["OE_M"])
        t = float(os.environ["OE_T"]); seed = int(os.environ["OE_SEED"])
        mod = _load_module(program)
        if not hasattr(mod,"run"):
            print(json.dumps((False,{{"error":"no_run_fn"}}))); return
        H_terms, psi0 = _make_problem(n,m,seed)
        t0 = time.perf_counter()
        psi_final, meta = mod.run(H_terms, psi0, t)
        wall_ms = (time.perf_counter() - t0)*1000.0
        psi_exact = scipy.linalg.expm(-1j*t*sum(H_terms)) @ psi0
        err = float(np.linalg.norm(psi_final - psi_exact))
        norm_val = float(np.linalg.norm(psi_final))
        mem_bytes = _safe_float((meta or {{}}).get("bytes_used",0.0))
        if not mem_bytes:
            mem_bytes = _bytes_of(psi_final, psi0, *H_terms)
        payload = {{
            "error": _safe_float(err,1e9),
            "norm": _safe_float(norm_val,0.0),
            "mem_mb": _safe_float(mem_bytes,0.0)/(1024.0*1024.0),
            "time_ms": _safe_float(wall_ms, {timeout_s}*1000.0),
        }}
        print(json.dumps((True,payload)))
    if __name__=="__main__":
        try: main()
        except Exception as e:
            tb = traceback.format_exc(limit=5)
            print(json.dumps((False,{{"error":f"runtime_error:{{type(e).__name__}}","trace":tb}})))
    """)

    env = os.environ.copy()
    env.update({
        "OE_PROG": os.path.abspath(program_path),
        "OE_N": str(n), "OE_M": str(m), "OE_T": str(t), "OE_SEED": str(seed),
    })

    with tempfile.TemporaryDirectory() as td:
        runner_path = Path(td)/"runner_trial.py"
        runner_path.write_text(runner, encoding="utf-8")

        # Prefer --export=sqlite if your nsys supports it; otherwise plain profile
        nsys = os.getenv("OE_NSYS", "nsys")
        base = [nsys, "profile", "-o", f"prof_trial_{seed}",
                "--trace", "cuda,nvtx,osrt", "--sample", "none",
                "--cpuctxsw", "none", "--force-overwrite", "true",
                "--multiprocess", "true",
                "uv", "run", "legate", "--profile", str(runner_path)]
        cmd = base[:]
        try:
            # Fast path
            cmd[3:3] = ["--export", "sqlite"]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s+5)
        except Exception:
            # Fallback without export flag
            p = subprocess.run(base, env=env, capture_output=True, text=True, timeout=timeout_s+5)

    # Parse JSON tuple (ok, payload) from stdout (last line)
    out = (p.stdout.strip().splitlines() or [""])[-1]
    try:
        ok, payload = json.loads(out)
        if not isinstance(payload, dict):
            raise ValueError("bad payload")
    except Exception:
        ok, payload = False, {"error":"nsys_runner_parse_fail", "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}

    return ok, payload


def evaluate(
    program: str,
    trials: int = 10,
    n: int = None,
    m: int = None,
    t: float = None,
    seeds=None,
) -> Dict[str, Any]:
    """
    Entry-point used by OpenEvolve.
    `program` must be a path to a candidate .py file exporting `run(H_terms, psi0, t) -> (psi_final, meta)`.
    Returns the metrics dict expected by your framework.
    """
    if n is None:
        # sample n from 10 to 200
        n = int(10 + (200 - 10) * np.random.beta(2, 5))
    if m is None:
        # sample m from 2 to 5
        m = int(2 + (5 - 2) * np.random.beta(2, 5))
    if t is None:
        # sample t from 0.01 to 1.0
        t = float(0.01 + (1.0 - 0.01) * np.random.beta(2, 5))

    # Sanity: program must exist (let evolution continue on failure)
    if not isinstance(program, str) or not os.path.exists(program):
        return {
            "combined_score": -1e9,
            "err": 1e9,
            "mem_norm": 1.0,
            "mem_raw": MEM_CAP_MB,
            "fid_mean": 1e9,
            "fid_min": 1e9,
            "time_ms_mean": TRIAL_TIMEOUT_S * 1000.0,
            "norm_dev": 1.0,
            "trials": 0,
            "error": "program_path_invalid",
        }

    seeds = list(seeds) if seeds else [1337 + i for i in range(trials)]
    errors, times_ms, norms, mem_mb = [], [], [], []

    db_path = Path("prof.sqlite")

    for sd in seeds:
        ok, res = _run_single_trial(
            program, n, m, t, sd, TRIAL_TIMEOUT_S, profile_nsys=PROFILE_GPU
        )
        if not ok:
            errors.append(1e3)
            times_ms.append(TRIAL_TIMEOUT_S * 1000.0)
            norms.append(0.0)
            mem_mb.append(MEM_CAP_MB)
            continue

        errors.append(_safe_float(res.get("error"), 1e3))
        times_ms.append(_safe_float(res.get("time_ms"), TRIAL_TIMEOUT_S * 1000.0))
        norms.append(_safe_float(res.get("norm"), 0.0))
        mem_mb.append(_safe_float(res.get("mem_mb"), 0.0))

    # ---- Aggregate ----
    err_mean = float(np.mean(errors)) if errors else 1e3
    err_min = float(np.min(errors)) if errors else 1e3
    t_mean = float(np.mean(times_ms)) if times_ms else TRIAL_TIMEOUT_S * 1000.0
    norm_dev = float(abs((np.mean(norms) if norms else 0.0) - 1.0))
    mem_mb_mean = float(np.mean(mem_mb)) if mem_mb else 0.0

    # ---- If GPU profiling is enabled, override with real Nsight metrics ----
    if PROFILE_GPU and db_path.exists():
        try:
            total_time, time_rows, mem_bytes = compute_gpu_metrics(db_path)
            t_mean = total_time * 1000.0  # convert seconds → ms

            # Handle both scalar and dict memory outputs
            if isinstance(mem_bytes, dict):
                mem_bytes = sum(mem_bytes.values())

            mem_mb_mean = float(mem_bytes) / (1024.0 * 1024.0)
            print(f"[GPU] Nsight metrics: {total_time:.3f}s total, {mem_mb_mean:.2f} MB")
        except Exception as e:
            print(f"[WARN] GPU metrics unavailable: {e}")

    # ---- Combine ----
    mem_norm = min(mem_mb_mean, MEM_CAP_MB) / MEM_CAP_MB
    # combined = LAMBDA_W * (-err_mean) + (1.0 - LAMBDA_W) * (-mem_norm)
    combined = (
        LAMBDA_W * (-err_mean)
        + (1.0 - LAMBDA_W) * (-mem_norm)
        - ALPHA_GPU * np.log1p(t_mean / 1000.0)
    )

    out = {
        "combined_score": float(combined),
        "err": float(err_mean),
        "mem_norm": float(mem_norm),
        "mem_raw": float(mem_mb_mean),
        "fid_mean": float(err_mean),
        "fid_min": float(err_min),
        "time_ms_mean": float(t_mean),
        "norm_dev": float(norm_dev),
        "trials": int(len(seeds)),
    }
    # Required invariant for OpenEvolve
    assert isinstance(out["combined_score"], float)
    return out


# Optional CLI for local debugging
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluator (process-isolated)")
    parser.add_argument(
        "program", type=str, help="Path to candidate .py (must export run)"
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--t", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.program, trials=args.trials, n=args.n, m=args.m, t=args.t),
            indent=2,
        )
    )
