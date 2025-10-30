# trial_runner.py
from __future__ import annotations
import os
import sys
import time
import traceback
import multiprocessing as mp
from typing import Dict, Tuple, Any
import numpy as np
import scipy.linalg

# import your existing helpers
# We'll inline minimal copies here for clarity. You can refactor to import.
# ------------------------------------------------------------------------


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


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


def _limit_child_resources(timeout_s: float, proc_mem_limit_mb: float, num_threads: str):
    try:
        import resource

        if proc_mem_limit_mb and proc_mem_limit_mb > 0:
            bytes_limit = int(proc_mem_limit_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        cpu_sec = max(5, int(timeout_s * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", num_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", num_threads)
    os.environ.setdefault("MKL_NUM_THREADS", num_threads)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", num_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", num_threads)


def _load_module_from_file(path: str):
    repo_root = os.path.abspath(os.getcwd())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib.util

    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _make_problem(n: int, m: int, seed: int, coef_dist=None):
    # you can import your real WeightedRandomHamiltonian here
    from trotvol.qm.hamiltonians import WeightedRandomHamiltonian

    rng = np.random.default_rng(seed)
    ensemble = WeightedRandomHamiltonian(n=n, m=m, coef_dist=coef_dist)
    H = ensemble.sample(rng=rng)
    H_terms = [t["coef"] * t["matrix"] for t in H.terms()]
    psi0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    psi0 /= np.linalg.norm(psi0)
    return H_terms, psi0


def _trial_core(
    program_path: str,
    n: int,
    m: int,
    t: float,
    seed: int,
    timeout_s: float,
    proc_mem_limit_mb: float,
    num_threads: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Runs ONE trial in-process (no GPU profiling). Assumes caller already forked.
    Returns (ok, payload).
    """
    # resource caps INSIDE child
    _limit_child_resources(timeout_s, proc_mem_limit_mb, num_threads)

    # SIGSEGV -> default
    try:
        import signal

        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
    except Exception:
        pass

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
            "time_ms": wall_ms,
        }
        return True, payload

    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return False, {
            "error": f"runtime_error:{type(e).__name__}",
            "trace": tb,
        }


def _mp_worker(program_path, n, m, t, seed, timeout_s, proc_mem_limit_mb, num_threads, conn):
    ok, payload = _trial_core(
        program_path, n, m, t, seed, timeout_s, proc_mem_limit_mb, num_threads
    )
    conn.send((ok, payload))
    conn.close()


class TrialRunner:
    """
    Safe interface to execute a candidate program for a single (n,m,t,seed)
    with timeout, memory cap, and (optionally later) GPU profiling path.
    """

    def __init__(
        self,
        timeout_s: float,
        proc_mem_limit_mb: float,
        num_threads: str,
        mp_method: str | None = None,
    ):
        if mp_method is None:
            mp_method = "fork" if sys.platform == "darwin" else "spawn"
        self.timeout_s = timeout_s
        self.proc_mem_limit_mb = proc_mem_limit_mb
        self.num_threads = num_threads
        self.mp_method = mp_method

    def run_single_trial(
        self,
        program_path: str,
        n: int,
        m: int,
        t: float,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Returns payload = {
            "error": ...,
            "norm": ...,
            "mem_mb": ... or "gpu_mem_mb": ...,
            "time_ms": ... or "gpu_time_ms": ...,
            ...
            "ok": bool,
        }
        Never raises.
        """
        ctx = mp.get_context(self.mp_method)
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_mp_worker,
            args=(
                program_path,
                n,
                m,
                t,
                seed,
                self.timeout_s,
                self.proc_mem_limit_mb,
                self.num_threads,
                child,
            ),
        )
        proc.daemon = True
        proc.start()
        child.close()

        ok = False
        payload: Dict[str, Any] = {"error": "unknown"}

        if parent.poll(self.timeout_s):
            try:
                ok, payload = parent.recv()
            except EOFError:
                ok, payload = False, {"error": "eof"}
        else:
            # timeout kill
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

        payload["ok"] = bool(ok)
        return payload
