from __future__ import annotations
import os, time, json, importlib.util, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la

# ------------- utils -------------
def _load_module_from_file(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _is_cupy_array(x) -> bool:
    try:
        import cupy
        return isinstance(x, cupy.ndarray)
    except Exception:
        return False

def _to_numpy(a):
    if _is_cupy_array(a):
        return np.asarray(a.get())
    return np.asarray(a)

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

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _random_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi0 /= np.linalg.norm(psi0)
    return psi0

def _coerce_terms_from_hamiltonian(ham) -> List[np.ndarray]:
    # ham.terms() -> [{"coef": scalar, "matrix": Hermitian array}]
    out = []
    for t in ham.terms():
        out.append(_safe_float(t["coef"], 1.0) * _to_numpy(t["matrix"]))
    return out

def _dim_from_terms(terms: Sequence[np.ndarray]) -> int:
    return terms[0].shape[0] if terms else 0


# ------------- base evaluator -------------
class BaseEval:
    """
    Base for concrete evaluators.
    You pass either a deterministic Hamiltonian or a HamiltonianEnsemble.
    Optionally pass a fixed psi0; otherwise we draw per-trial via seed.
    """
    def __init__(
        self,
        ham_or_ensemble: object,
        trials: int = 5,
        t: float = 0.05,
        seeds: Optional[Sequence[int]] = None,
        psi0: Optional[np.ndarray] = None,
    ):
        if ham_or_ensemble is None:
            raise ValueError("Must provide a Hamiltonian or HamiltonianEnsemble.")
        self._src = ham_or_ensemble
        self.trials = int(trials)
        self.t = float(t)
        self.seeds = list(seeds) if seeds else [1337 + i for i in range(trials)]
        self._psi0_fixed = _to_numpy(psi0) if psi0 is not None else None
    
    # --- hooks for subclasses ---
    def _score(self, rows: List[Dict[str, float]]) -> float:
        """Compute a scalar score from per-trial rows. Higher is better."""
        raise NotImplementedError

    # --- core execution ---
    def _draw_problem(self, sd: int) -> Tuple[List[np.ndarray], np.ndarray]:
        # If it's an ensemble, sample; else assume deterministic Hamiltonian
        if getattr(self._src, "is_stochastic", False) or hasattr(self._src, "sample"):
            rng = np.random.default_rng(sd)
            ham = self._src.sample(rng=rng)
        else:
            ham = self._src

        terms = _coerce_terms_from_hamiltonian(ham)
        dim = getattr(ham, "hilbert_dim", _dim_from_terms(terms))
        psi0 = self._psi0_fixed if self._psi0_fixed is not None else _random_state(dim, sd + 17)
        return terms, psi0

    def _run_once(self, program_path: str, sd: int) -> Tuple[bool, Dict[str, Any]]:
        try:
            mod = _load_module_from_file(program_path)
            if not hasattr(mod, "run"):
                return False, {"error": "no_run_fn"}

            H_terms, psi0 = self._draw_problem(sd)

            t0 = time.perf_counter()
            psi_final, meta = mod.run(H_terms, psi0, self.t)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            # exact reference on CPU
            H = sum(H_terms)
            psi_ref = la.expm(-1j * self.t * H) @ psi0

            err = float(np.linalg.norm(psi_final - psi_ref))
            norm_val = float(np.linalg.norm(psi_final))
            # prefer candidate-provided GPU metrics if present
            gpu_time_ms = _safe_float((meta or {}).get("gpu_time_ms"), 0.0) or None
            gpu_mem_mb  = _safe_float((meta or {}).get("gpu_mem_mb"), 0.0) or None
            bytes_used  = _safe_float((meta or {}).get("bytes_used"), 0.0)

            # fallback memory estimate
            mem_mb_est = (bytes_used or _bytes_of(psi_final, psi0, H)) / (1024.0 * 1024.0)

            row = {
                "error": err,
                "norm": norm_val,
                "time_ms": gpu_time_ms if gpu_time_ms is not None else wall_ms,
                "mem_mb": gpu_mem_mb if gpu_mem_mb is not None else mem_mb_est,
            }
            return True, row
        except Exception as e:
            return False, {"error": f"runtime_error:{type(e).__name__}", "trace": traceback.format_exc(limit=6)}

    def evaluate(self, program: Optional[str] = None) -> Dict[str, Any]:
        program_path = program or self._program
        if not program_path or not Path(program_path).exists():
            return {"score": -1e9, "trials": 0, "error": "program_path_invalid"}

        rows: List[Dict[str, float]] = []
        for sd in self.seeds:
            ok, res = self._run_once(program_path, sd)
            if ok:
                rows.append(res)
            else:
                # record a failed trial with heavy penalties
                rows.append({"error": 1e3, "norm": 0.0, "time_ms": 1e6, "mem_mb": 1e6})

        score = self._score(rows)
        out = {
            "score": float(score),
            "err_mean": float(np.mean([r["error"] for r in rows])),
            "err_min": float(np.min([r["error"] for r in rows])),
            "norm_dev": float(abs(np.mean([r["norm"] for r in rows]) - 1.0)),
            "time_ms_mean": float(np.mean([r["time_ms"] for r in rows])),
            "mem_mb_mean": float(np.mean([r["mem_mb"] for r in rows])),
            "trials": len(rows),
        }
        return out

