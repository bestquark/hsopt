# evaluate.py
from __future__ import annotations
import os
import sys
import time
import json
import traceback
import subprocess
from pathlib import Path
import numpy as np
import scipy
import importlib.util
import multiprocessing as mp
from typing import Dict, Any
from spacetime_gpu import spacetime_gpu_sqlite3
from eqmmm.qm.hamiltonians import WeightedRandomHamiltonian

# -------- Global knobs (override via env) ----------
MEM_CAP_MB = float(os.getenv("OE_MEM_CAP_MB", "32"))
LAMBDA_W = float(os.getenv("OE_LAMBDA", "0.5"))
ALPHA_GPU = float(os.getenv("OE_ALPHA_GPU", "0.1"))
TRIAL_TIMEOUT_S = float(os.getenv("OE_TRIAL_TIMEOUT_S", "30"))
PROC_MEM_LIMIT_MB = float(os.getenv("OE_PROC_MEM_MB", "16384"))
NUM_THREADS = os.getenv("OE_NUM_THREADS", "1")
MP_METHOD_DEFAULT = "fork" if sys.platform == "darwin" else "spawn"
MP_METHOD = os.getenv("OE_MP_METHOD", MP_METHOD_DEFAULT)
PROFILE_GPU = os.getenv("OE_PROFILE_GPU", "0") == "1"
NSYS_PREFIX = os.getenv("OE_NSYS_PREFIX", "")
NSYS_BIN = os.path.join(NSYS_PREFIX, "nsys") if NSYS_PREFIX else "nsys"
# ---------------------------------------------------


def _make_problem(
    n: int = 100,
    m: int = 5,
    seed: int = 0,
    coef_dist: dict | None = None,
):
    """
    Create a random Hermitian problem using WeightedRandomHamiltonian.

    Returns
    -------
    H_terms : list of np.ndarray
        List of individual term matrices (each Hermitian).
    psi0 : np.ndarray
        Random normalized initial state.
    """
    rng = np.random.default_rng(seed)

    ensemble = WeightedRandomHamiltonian(n=n, m=m, coef_dist=coef_dist)
    H = ensemble.sample(rng=rng)

    H_terms = [t["coef"] * t["matrix"] for t in H.terms()]

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
    try:
        import resource

        # Only cap if positive; 0 or negative → no cap
        if PROC_MEM_LIMIT_MB and PROC_MEM_LIMIT_MB > 0:
            bytes_limit = int(PROC_MEM_LIMIT_MB * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        cpu_sec = max(5, int(TRIAL_TIMEOUT_S * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", NUM_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", NUM_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", NUM_THREADS)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", NUM_THREADS)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", NUM_THREADS)


# ---------- Single-trial core used by BOTH paths ----------
def _trial_core(
    program_path: str, n: int, m: int, t: float, seed: int
) -> tuple[bool, dict]:
    try:
        mod = _load_module_from_file(program_path)
        if not hasattr(mod, "run"):
            return False, {"error": "no_run_fn"}

        H_terms, psi0 = _make_problem(n=n, m=m, seed=seed)
        t0 = time.perf_counter()
        psi_final, meta = mod.run(H_terms, psi0, t)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        psi_exact = scipy.linalg.expm(-1j * t * sum(H_terms)) @ psi0
        err = float(np.linalg.norm(psi_final - psi_exact))
        norm_val = float(np.linalg.norm(psi_final))
        mem_bytes = _safe_float((meta or {}).get("bytes_used", 0.0)) or _bytes_of(
            psi_final, psi0, *H_terms
        )
        payload = {
            "error": _safe_float(err, 1e9),
            "norm": _safe_float(norm_val, 0.0),
            "mem_mb": _safe_float(mem_bytes, 0.0) / (1024.0 * 1024.0),
            "time_ms": _safe_float(wall_ms, TRIAL_TIMEOUT_S * 1000.0),
        }
        return True, payload
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return False, {"error": f"runtime_error:{type(e).__name__}", "trace": tb}


# ---------- mp worker (CPU-only) ----------
def _trial_worker(program_path: str, n: int, m: int, t: float, seed: int, conn):
    try:
        _limit_child_resources()
        import signal

        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
    except Exception:
        pass
    try:
        ok, payload = _trial_core(program_path, n, m, t, seed)
        conn.send((ok, payload))
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        conn.send((False, {"error": f"runtime_error:{type(e).__name__}", "trace": tb}))
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------- entry used by Nsight/legate re-exec (same module, no duplication) ----------
def _single_trial_cli():
    _limit_child_resources()
    program = os.environ["OE_PROG"]
    n = int(os.environ["OE_N"])
    m = int(os.environ["OE_M"])
    t = float(os.environ["OE_T"])
    seed = int(os.environ["OE_SEED"])
    ok, payload = _trial_core(program, n, m, t, seed)
    rec = (ok, payload)
    out_path = os.environ.get("OE_JSON_OUT", "")
    if out_path:
        Path(out_path).write_text(json.dumps(rec), encoding="utf-8")
    else:
        print(json.dumps(rec))


# ---------- orchestrators ----------
def _run_single_trial(
    program_path: str,
    n: int,
    m: int,
    t: float,
    seed: int,
    timeout_s: float,
    *,
    profile_nsys: bool = False,
):
    if profile_nsys:
        return _run_single_trial_nsys(program_path, n, m, t, seed, timeout_s)

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
        try:
            proc.terminate()
        except Exception:
            pass
        ok, payload = False, {"error": "timeout"}

    proc.join(timeout=1.0)
    if proc.is_alive():
        try:
            proc.kill()
        except Exception:
            pass
        proc.join()

    if not ok and proc.exitcode is not None and proc.exitcode < 0:
        payload.setdefault("error", f"signal_{-proc.exitcode}")
    return ok, payload


def _run_single_trial_nsys(
    program_path: str, n: int, m: int, t: float, seed: int, timeout_s: float
):
    base = f"prof_{seed}"
    db_path = Path(f"{base}.sqlite")
    rep_path = Path(f"{base}.nsys-rep")
    json_out = Path(f".trial_{seed}.json")
    for p in (db_path, rep_path, json_out):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    # resources
    n_gpus = int(os.getenv("N_GPUS", "1"))
    cpus = int(os.getenv("OE_LEGATE_CPUS", "4"))
    fbmem = str(os.getenv("OE_LEGATE_FBMEM", "256"))

    env = os.environ.copy()
    env.update(
        {
            "OE_PROG": os.path.abspath(program_path),
            "OE_N": str(n),
            "OE_M": str(m),
            "OE_T": str(t),
            "OE_SEED": str(seed),
            "OE_JSON_OUT": str(json_out),
        }
    )
    env.setdefault("LEGATE_AUTO_CONFIG", "0")

    # --- detect child-process capture support ---
    capture_children_flag = []
    try:
        help_txt = subprocess.run(
            [NSYS_BIN, "profile", "--help"], capture_output=True, text=True, check=True
        ).stdout.lower()
        if "--capture-child-processes" in help_txt:
            # prefer "=true"; some builds accept "=all"
            val = "true" if "=true" in help_txt else "all"
            capture_children_flag = ["--capture-child-processes", val]
    except Exception:
        pass

    cmd = [
        NSYS_BIN,
        "profile",
        "-o",
        str(base),
        "--export",
        "sqlite",
        "--trace",
        "cuda,nvtx,osrt",
        "--sample",
        "none",
        "--cpuctxsw",
        "none",
        "--force-overwrite",
        "true",
        "--cuda-memory-usage",
        "true",
        *capture_children_flag,
        "uv",
        "run",
        "legate",
        "--gpus",
        str(n_gpus),
        "--cpus",
        str(cpus),
        "--fbmem",
        str(fbmem),
        # IMPORTANT: do NOT pass Legion's --profile here
        os.path.abspath(__file__),
        "--single-trial",
    ]
    print(f"[NSYS] Running command: {' '.join(cmd)}")

    p = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s + 120
    )

    if not json_out.exists():
        return False, {
            "error": "nsys_runner_parse_fail_or_timeout",
            "stdout": (p.stdout or "")[-4000:],
            "stderr": (p.stderr or "")[-4000:],
        }

    try:
        ok, payload = json.loads(json_out.read_text(encoding="utf-8"))
    except Exception:
        return False, {
            "error": "nsys_runner_bad_json",
            "stdout": (p.stdout or "")[-4000:],
            "stderr": (p.stderr or "")[-4000:],
        }

    # Fold per-trial GPU metrics if present
    if db_path.exists():
        try:
            total_time_s, _rows, mem_bytes = spacetime_gpu_sqlite3(db_path)
            if isinstance(mem_bytes, dict):
                mem_bytes = sum(mem_bytes.values())
            payload["gpu_time_ms"] = float(total_time_s) * 1000.0
            payload["gpu_mem_mb"] = float(mem_bytes) / (1024.0 * 1024.0)
            payload["prof_db"] = str(db_path)
        except Exception as e:
            payload.setdefault("gpu_metrics_error", str(e))
    else:
        payload.setdefault("gpu_metrics_error", "nsys_db_missing")

    return bool(ok), payload


# ---------- public API ----------
def evaluate(
    program: str,
    trials: int = 10,
    n: int = None,
    m: int = None,
    t: float = None,
    seeds=None,
) -> Dict[str, Any]:
    if n is None:
        n = int(10 + (200 - 10) * np.random.beta(2, 5))
    if m is None:
        m = int(2 + (5 - 2) * np.random.beta(2, 5))
    if t is None:
        t = float(0.01 + (1.0 - 0.01) * np.random.beta(2, 5))

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
    errs, norms, mems_mb, times_ms = [], [], [], []

    for sd in seeds:
        ok, res = _run_single_trial(
            program, n, m, t, sd, TRIAL_TIMEOUT_S, profile_nsys=PROFILE_GPU
        )
        print(f"[Trial seed={sd}] ok={ok} res={res}")

        if not ok:
            errs.append(1e3)
            norms.append(0.0)
            mems_mb.append(MEM_CAP_MB)
            times_ms.append(TRIAL_TIMEOUT_S * 1000.0)
            continue

        # take the child’s metrics
        errs.append(_safe_float(res.get("error"), 1e3))
        norms.append(_safe_float(res.get("norm"), 0.0))

        # prefer GPU mem/time if present; fallback to wallclock+estimated mem
        mems_mb.append(
            _safe_float(
                res.get("gpu_mem_mb"), _safe_float(res.get("mem_mb"), MEM_CAP_MB)
            )
        )
        times_ms.append(
            _safe_float(
                res.get("gpu_time_ms"),
                _safe_float(res.get("time_ms"), TRIAL_TIMEOUT_S * 1000.0),
            )
        )

    err_mean = float(np.mean(errs)) if errs else 1e3
    err_min = float(np.min(errs)) if errs else 1e3
    norm_dev = float(abs((np.mean(norms) if norms else 0.0) - 1.0))
    mem_mean = float(np.mean(mems_mb)) if mems_mb else MEM_CAP_MB
    time_mean = float(np.mean(times_ms)) if times_ms else TRIAL_TIMEOUT_S * 1000.0

    mem_norm = min(mem_mean, MEM_CAP_MB) / MEM_CAP_MB
    combined = (
        LAMBDA_W * (-err_mean)
        + (1.0 - LAMBDA_W) * (-mem_norm)
        - ALPHA_GPU * np.log1p(time_mean / 1000.0)
    )

    return {
        "combined_score": float(combined),
        "err": float(err_mean),
        "mem_norm": float(mem_norm),
        "mem_raw": float(mem_mean),
        "fid_mean": float(err_mean),
        "fid_min": float(err_min),
        "time_ms_mean": float(time_mean),
        "norm_dev": float(norm_dev),
        "trials": int(len(seeds)),
    }


# ---------- CLI ----------
if __name__ == "__main__":
    # fast path for Nsight/legate re-entry (no argparse noise)
    if "--single-trial" in sys.argv:
        _single_trial_cli()
        sys.exit(0)

    import argparse

    p = argparse.ArgumentParser(description="Evaluator (process-isolated)")
    p.add_argument("program", type=str, help="Path to candidate .py (must export run)")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--t", type=float, default=0.1)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()
    print(
        json.dumps(
            evaluate(args.program, trials=args.trials, n=args.n, m=args.m, t=args.t),
            indent=2,
        )
    )
