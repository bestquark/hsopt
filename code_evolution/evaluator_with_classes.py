# evaluate.py
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np

from trovolut.qm.hamiltonians import WeightedRandomHamiltonian
from trovolut.eval import GPUSpaceTimeEval  


def main():
    # --- defaults (override via env if you want) ---
    candidate = Path(os.getenv("OE_CANDIDATE", "candidate.py"))
    if not candidate.exists():
        print("❌ candidate.py not found (set OE_CANDIDATE to override).")
        sys.exit(1)

    # tiny random Hamiltonian (fast)
    n = int(os.getenv("OE_N", "8"))
    m = int(os.getenv("OE_M", "3"))
    t = float(os.getenv("OE_T", "0.05"))
    trials = int(os.getenv("OE_TRIALS", "3"))
    seeds = [1337 + i for i in range(trials)]

    # profiler DB settings
    db_type = os.getenv("OE_DB_TYPE", "duckdb")        # "duckdb" or "sqlite"
    db_path = Path(os.getenv("OE_DB_PATH", "legion_prof"))

    # weights (seconds + FB MiB + error penalty)
    w_time = float(os.getenv("OE_W_TIME", "1.0"))
    w_peak = float(os.getenv("OE_W_PEAK", "0.02"))
    w_avg  = float(os.getenv("OE_W_AVG",  "0.01"))
    w_err  = float(os.getenv("OE_W_ERR",  "100.0"))

    # --- build a concrete Hamiltonian sample ---
    ham = WeightedRandomHamiltonian(n=n, m=m).sample()

    # --- run GPU profiler–driven evaluation ---
    evalr = GPUSpaceTimeEval(
        ham_or_ensemble=ham,
        program=str(candidate),
        trials=trials,
        t=t,
        seeds=seeds,
        db_type=db_type,
        db_path=db_path,
        w_time=w_time,
        w_peak=w_peak,
        w_avg=w_avg,
        w_err=w_err,
    )

    res = evalr.evaluate()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()