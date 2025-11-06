# evaluate.py
from __future__ import annotations
import os, sys, json, importlib.util, argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
from trovolut.qm.hamiltonians import WeightedRandomHamiltonian

# === import your evaluator classes (must be in PYTHONPATH) ===
from trovolut.eval import (
    GPUMemoryEval,
    SimulatedMemoryEval,
    TrotterErrorEval,
    CombinedEvalMeta,   # provided by the "combined" sugar
    CombinedComponent,
    CombinedSum,
)

# ----------------- Helpers -----------------
def _load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("user_obj", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _resolve_hamiltonian(ham_path: Optional[str], n_fallback: int, m_fallback: int, seed: int):
    """
    Load a Hamiltonian or HamiltonianEnsemble from a python file.
    Conventions supported in the module:
      - a callable `build()` returning Hamiltonian or HamiltonianEnsemble
      - a variable `HAMILTONIAN` or `ENSEMBLE`
    If none provided, fall back to WeightedRandomHamiltonian(n_fallback, m_fallback).sample().
    """
    if ham_path:
        mod = _load_module_from_path(ham_path)
        if hasattr(mod, "build") and callable(mod.build):
            obj = mod.build()
            return obj
        if hasattr(mod, "HAMILTONIAN"):
            return mod.HAMILTONIAN
        if hasattr(mod, "ENSEMBLE"):
            return mod.ENSEMBLE
        raise ValueError(
            f"Could not find `build()`, `HAMILTONIAN`, or `ENSEMBLE` in {ham_path}"
        )

    # Fallback: random ensemble
    rng = np.random.default_rng(seed)
    ens = WeightedRandomHamiltonian(n=n_fallback, m=m_fallback)
    # Let the evaluators sample per-trial if an ensemble is passed; returning the ensemble is better
    return ens

def _parse_combination(expr: str):
    """
    Parse strings like:
      "0.5*GPUMemoryEval + 0.3*TrotterErrorEval + 0.2*SimulatedMemoryEval"
    into a callable Combined metric class.
    """
    # Minimal safe eval: map name->class & allowed ops
    env = {
        "GPUMemoryEval": GPUMemoryEval,
        "SimulatedMemoryEval": SimulatedMemoryEval,
        "TrotterErrorEval": TrotterErrorEval,
    }
    # Enable the meta-ops by making the classes use the metaclass behavior
    # In our implementation these classes already support 0.5*Class + ...
    try:
        CombinedMetric = eval(expr, {"__builtins__": {}}, env)
    except Exception as e:
        raise ValueError(f"Failed to parse --combo '{expr}': {e}")
    return CombinedMetric

# ----------------- Main entry -----------------
def main():
    p = argparse.ArgumentParser(description="Evaluator wrapper for Hamiltonian/Ensemble")
    p.add_argument("program", type=str, help="Path to candidate .py (exports run(H_terms, psi0, t))")

    # How to obtain the Hamiltonian / Ensemble
    p.add_argument("--ham-path", type=str, default=None,
                   help="Python file exporting build()/HAMILTONIAN/ENSEMBLE to provide Hamiltonian or HamiltonianEnsemble")
    p.add_argument("--n-fallback", type=int, default=64, help="dim for fallback WeightedRandomHamiltonian")
    p.add_argument("--m-fallback", type=int, default=3, help="#terms for fallback WeightedRandomHamiltonian")

    # Eval knobs
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--t", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional fixed seeds")

    # Choose metric/evaluator
    p.add_argument("--mode", choices=["gpu_mem", "sim_mem", "trotter", "combined"], default="trotter")
    p.add_argument("--combo", type=str, default="",
                   help="Linear combo, e.g. '0.5*GPUMemoryEval + 0.3*TrotterErrorEval + 0.2*SimulatedMemoryEval' (used when --mode=combined)")

    args = p.parse_args()

    program = Path(args.program)
    if not program.exists():
        print(json.dumps({"error": "program_path_invalid", "program": str(program)}))
        sys.exit(2)

    # Resolve Hamiltonian or Ensemble source
    src = _resolve_hamiltonian(args.ham_path, args.n_fallback, args.m_fallback, seed=1337)

    # Build evaluator
    if args.mode == "gpu_mem":
        evaluator = GPUMemoryEval(src, program=str(program), trials=args.trials, t=args.t, seeds=args.seeds)
    elif args.mode == "sim_mem":
        evaluator = SimulatedMemoryEval(src, program=str(program), trials=args.trials, t=args.t, seeds=args.seeds)
    elif args.mode == "trotter":
        evaluator = TrotterErrorEval(src, program=str(program), trials=args.trials, t=args.t, seeds=args.seeds)
    else:
        if not args.combo.strip():
            print(json.dumps({"error": "combo_missing", "hint": "Provide --combo when --mode=combined"}))
            sys.exit(2)
        CombinedMetric = _parse_combination(args.combo)
        evaluator = CombinedMetric(src, program=str(program), trials=args.trials, t=args.t, seeds=args.seeds)

    # Run & print JSON
    result = evaluator.evaluate()
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()