from trovolut.eval.base import BaseEval
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

class SimulatedMemoryEval(BaseEval):
    """
    Minimize estimated host-side memory footprint computed from array sizes
    (uses candidate-provided `bytes_used` if available; otherwise estimates).
    Penalties for egregious runtime/error.
    """
    def __init__(self, ham_or_ensemble, program: Optional[str] = None,
                 trials: int = 5, t: float = 0.05, seeds=None,
                 w_time: float = 0.02, w_err: float = 200.0, psi0: Optional[np.ndarray] = None):
        super().__init__(ham_or_ensemble, program, trials, t, seeds, psi0)
        self.w_time = float(w_time)
        self.w_err = float(w_err)

    def _score(self, rows: List[Dict[str, float]]) -> float:
        m = np.array([r["mem_mb"] for r in rows])
        t = np.array([r["time_ms"] for r in rows])
        e = np.array([r["error"] for r in rows])
        return float(-np.mean(m) - self.w_time*np.mean(t) - self.w_err*np.mean(e))
