from trovolut.eval.base import BaseEval
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ------------- concrete evaluators -------------
class GPUMemoryEval(BaseEval):
    """
    Minimize GPU memory usage (fallback to estimated total bytes if candidate
    doesn't report gpu_mem_mb). Lightly penalize runtime and error.
    """
    def __init__(self, ham_or_ensemble, program: Optional[str] = None,
                 trials: int = 5, t: float = 0.05, seeds=None,
                 w_time: float = 0.01, w_err: float = 100.0, psi0: Optional[np.ndarray] = None):
        super().__init__(ham_or_ensemble, program, trials, t, seeds, psi0)
        self.w_time = float(w_time)
        self.w_err = float(w_err)

    def _score(self, rows: List[Dict[str, float]]) -> float:
        m = np.array([r["mem_mb"] for r in rows])
        t = np.array([r["time_ms"] for r in rows])
        e = np.array([r["error"] for r in rows])
        # Use p95 of memory to avoid one lucky low sample
        m95 = np.percentile(m, 95)
        return float(-m95 - self.w_time * np.mean(t) - self.w_err * np.mean(e))

