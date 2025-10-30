import os
import sys

from .metrics import ErrorMetric, NormDeviationMetric, MemMetric, GpuTimeMetric
from .runner import TrialRunner
from .evaluator import Evaluator, EvalConfig, CombinerWeights


MEM_CAP_MB = float(os.getenv("OE_MEM_CAP_MB", "32"))
TRIAL_TIMEOUT_S = float(os.getenv("OE_TRIAL_TIMEOUT_S", "30"))
PROC_MEM_LIMIT = float(os.getenv("OE_PROC_MEM_MB", "16384"))
NUM_THREADS = os.getenv("OE_NUM_THREADS", "1")
MP_METHOD_DEF = "fork" if sys.platform == "darwin" else "spawn"
MP_METHOD = os.getenv("OE_MP_METHOD", MP_METHOD_DEF)

runner = TrialRunner(
    timeout_s=TRIAL_TIMEOUT_S,
    proc_mem_limit_mb=PROC_MEM_LIMIT,
    num_threads=NUM_THREADS,
    mp_method=MP_METHOD,
)

metrics = [
    ErrorMetric(),
    NormDeviationMetric(),
    MemMetric(cap_mb=MEM_CAP_MB),
    GpuTimeMetric(),
]

combiner = CombinerWeights(
    w_err=float(os.getenv("OE_LAMBDA", "0.5")),  # like LAMBDA_W
    w_mem=1.0 - float(os.getenv("OE_LAMBDA", "0.5")),  # mirror your (1-LAMBDA_W)
    w_time=float(os.getenv("OE_ALPHA_GPU", "0.1")),  # ALPHA_GPU
    w_normdev=0.0,
)

evaluator = Evaluator(
    runner=runner,
    metrics=metrics,
    combiner=combiner,
    mem_cap_mb=MEM_CAP_MB,
)

cfg = EvalConfig(trials=10)  # or set n,m,t,seeds etc.

result = evaluator.evaluate("candidate_algo.py", cfg)
print(result)
