from trovolut.eval.base import BaseEval
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

class TrotterErrorEval(BaseEval):
    """
    Minimize simulation error; soft penalties for time and memory.
    """
    def __init__(self, ham_or_ensemble, program: Optional[str] = None,
                 trials: int = 5, t: float = 0.05, seeds=None,
                 w_time: float = 0.01, w_mem: float = 0.01, psi0: Optional[np.ndarray] = None):
        super().__init__(ham_or_ensemble, program, trials, t, seeds, psi0)
        self.w_time = float(w_time)
        self.w_mem = float(w_mem)

    def _score(self, rows: List[Dict[str, float]]) -> float:
        e = np.array([r["error"] for r in rows])
        t = np.array([r["time_ms"] for r in rows])
        m = np.array([r["mem_mb"] for r in rows])
        # Prefer both low mean error and low best-case error
        return float(-1e3*np.mean(e) - 5e2*np.min(e) - self.w_time*np.mean(t) - self.w_mem*np.mean(m))
