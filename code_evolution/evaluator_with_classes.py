# evaluate.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from trovolut.qm.hamiltonians import WeightedRandomHamiltonian
from trovolut.eval import GPUSpaceTimeEval


# =============================
# Configurable parameters
# =============================
CANDIDATE_PATH = Path("code_evolution/initial_program_with_classes.py")   # file exporting run(H_terms, psi0, t)
DB_TYPE = "duckdb"                     
DB_PATH = Path("legion_prof")          

# Hamiltonian and simulation parameters
N = 8
M = 3
T = 0.05
TRIALS = 3
SEEDS = [1337 + i for i in range(TRIALS)]

# Metric weights
W_TIME = 1.0
W_PEAK = 0.02
W_AVG  = 0.01
W_ERR  = 100.0
# =============================


def main():
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"❌ Candidate file not found: {CANDIDATE_PATH}")

    print(f"Running evaluation for {CANDIDATE_PATH} using {DB_TYPE} database...")

    # small random Hamiltonian
    rng = np.random.default_rng(42)
    ham = WeightedRandomHamiltonian(n=N, m=M).sample(rng)

    # initialize GPU evaluation (DuckDB-based)
    evaluator = GPUSpaceTimeEval(
        ham_or_ensemble=ham,
        program=str(CANDIDATE_PATH),
        trials=TRIALS,
        t=T,
        seeds=SEEDS,
        legion_db=DB_PATH,
        w_time=W_TIME,
        w_peak=W_PEAK,
        w_avg=W_AVG,
        w_err=W_ERR,
    )

    result = evaluator.evaluate()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()