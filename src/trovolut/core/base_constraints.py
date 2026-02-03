from __future__ import annotations
import os, time, json, importlib.util, traceback, math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy.linalg as la

if TYPE_CHECKING:
    from trovolut.qm.hamiltonians import Hamiltonian, HamiltonianEnsemble

# ------------- utils -------------
def _load_module_from_file(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _is_cupy_array(x) -> bool:
    """Check if array is cupynumeric or cupy array."""
    try:
        import cupynumeric as cn
        if isinstance(x, cn.ndarray):
            return True
    except Exception:
        pass
    try:
        import cupy
        if isinstance(x, cupy.ndarray):
            return True
    except Exception:
        pass
    return False

def _to_numpy(a):
    if _is_cupy_array(a):
        # Convert GPU array to NumPy
        if hasattr(a, 'get'):
            return np.asarray(a.get())
        else:
            # cupynumeric uses different method
            return np.asarray(a)
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


# =============================================================================
# Constraint System (Hardware-Agnostic)
# =============================================================================

@dataclass
class ConstraintResult:
    """Result from evaluating a single constraint."""
    name: str
    value: float
    score: float
    satisfied: bool
    details: Dict[str, Any] = field(default_factory=dict)


def _score_fn(value: float, max_val: float, fn: str) -> float:
    """Common scoring function."""
    if fn == "linear":
        return max(0, 1.0 - value / max_val)
    elif fn == "inverse":
        return 1.0 / (1.0 + value / max_val)
    elif fn == "threshold":
        return 1.0 if value <= max_val else 0.0
    elif fn == "exponential":
        return math.exp(-value / max_val)
    return 0.0


def _formula_str(var: str, max_val: float, fn: str, hard: bool = False,
                 penalty: float = None, weight: float = None) -> str:
    """Generate formula string for a scoring function."""
    formulas = {
        "linear": f"max(0, 1 - {var}/{max_val})",
        "inverse": f"1 / (1 + {var}/{max_val})",
        "threshold": f"𝟙({var} ≤ {max_val})",
        "exponential": f"exp(-{var}/{max_val})",
    }
    base = formulas.get(fn, f"f({var})")
    if hard and penalty is not None:
        return f"score = {base} · 𝟙({var} ≤ {max_val}) + ({penalty}) · 𝟙({var} > {max_val})"
    if weight is not None and weight != 1.0:
        return f"score = {base}  (×{weight} weight)"
    return f"score = {base}"


class Constraint(ABC):
    """Base class for all constraints (hardware-agnostic and specific)."""
    
    def __init__(self, name: str, weight: float = 1.0, hard: bool = False, hard_penalty: float = -100.0):
        self.name = name
        self.weight = weight
        self.hard = hard
        self.hard_penalty = hard_penalty
    
    @abstractmethod
    def extract(self, metrics: Dict[str, Any]) -> float:
        """Extract the relevant metric value from execution results."""
        pass
    
    @abstractmethod
    def score(self, value: float) -> float:
        """Convert the raw value to a score (higher = better)."""
        pass
    
    @abstractmethod
    def is_satisfied(self, value: float) -> bool:
        """Check if this constraint is satisfied."""
        pass
    
    @abstractmethod
    def to_prompt(self) -> str:
        """Generate description for LLM prompts."""
        pass
    
    def evaluate(self, metrics: Dict[str, Any]) -> ConstraintResult:
        """Evaluate this constraint given execution metrics."""
        value = self.extract(metrics)
        satisfied = bool(self.is_satisfied(value))
        score = self.hard_penalty if (self.hard and not satisfied) else self.score(value) * self.weight
        # Convert numpy types to Python types
        value = float(value) if hasattr(value, 'item') else float(value)
        score = float(score) if hasattr(score, 'item') else float(score)
        return ConstraintResult(self.name, value, score, satisfied, {"hard": self.hard, "weight": self.weight})
    
    def formula(self) -> str:
        """Return a human-readable explanation of the scoring formula."""
        raise NotImplementedError("Subclasses should implement formula()")
    
    def __repr__(self) -> str:
        params = []
        for attr in ["max_seconds", "max_error", "max_mb", "max_cpu_ms", "max_count"]:
            if hasattr(self, attr):
                params.append(f"{getattr(self, attr)}")
                break
        params.append("hard" if self.hard else f"w={self.weight}")
        return f"{self.__class__.__name__}({', '.join(params)})"


class Time(Constraint):
    """Constraint on wall-clock execution time."""
    
    def __init__(self, max_seconds: float = 30.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("time", weight, hard, hard_penalty)
        self.max_seconds = max_seconds
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("time_seconds", float("inf"))
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_seconds, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_seconds
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"Code MUST execute in under {self.max_seconds}s (hard requirement)."
        return f"Prefer faster execution (target: <{self.max_seconds}s, weight: {self.weight})."
    
    def formula(self) -> str:
        return _formula_str("time", self.max_seconds, self.score_fn, self.hard, self.hard_penalty, self.weight)


class Error(Constraint):
    """Constraint on simulation error (lower is better; 0 = perfect)."""
    
    def __init__(self, max_error: float = 1.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "inverse"):
        super().__init__("error", weight, hard, hard_penalty)
        self.max_error = max_error
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("mean_error", float("inf"))
    
    def score(self, v: float) -> float:
        if self.score_fn == "inverse":
            return 1.0 / (1.0 + v)
        elif self.score_fn == "exponential":
            return math.exp(-v)
        return max(0, 1.0 - v / self.max_error)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_error
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"Simulation error MUST be below {self.max_error} (hard requirement)."
        return f"Minimize error (weight: {self.weight})."
    
    def formula(self) -> str:
        formulas = {
            "inverse": "1 / (1 + error)",
            "exponential": "exp(-error)",
            "linear": f"max(0, 1 - error/{self.max_error})",
        }
        base = formulas.get(self.score_fn, "f(error)")
        if self.hard:
            return f"score = {base} · 𝟙(error ≤ {self.max_error}) + ({self.hard_penalty}) · 𝟙(error > {self.max_error})"
        if self.weight != 1.0:
            return f"score = {base}  (×{self.weight} weight)"
        return f"score = {base}"


class Custom(Constraint):
    """User-defined constraint with custom extraction and scoring."""
    
    def __init__(self, name: str, extract_fn: Callable[[Dict[str, Any]], float],
                 score_fn: Callable[[float], float], satisfy_fn: Callable[[float], bool],
                 weight: float = 1.0, hard: bool = False, hard_penalty: float = -100.0):
        super().__init__(name, weight, hard, hard_penalty)
        self._extract = extract_fn
        self._score = score_fn
        self._satisfy = satisfy_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return self._extract(m)
    
    def score(self, v: float) -> float:
        return self._score(v)
    
    def is_satisfied(self, v: float) -> bool:
        return self._satisfy(v)
    
    def to_prompt(self) -> str:
        mode = "hard requirement" if self.hard else f"weight: {self.weight}"
        return f"Optimize for '{self.name}' ({mode})."
    
    def formula(self) -> str:
        base = f"custom_fn({self.name})"
        if self.hard:
            return f"score = {base} · 𝟙(satisfied) + ({self.hard_penalty}) · 𝟙(violated)"
        if self.weight != 1.0:
            return f"score = {base}  (×{self.weight} weight)"
        return f"score = {base}"


class ConstraintSet:
    """A collection of constraints evaluated together."""
    
    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints
    
    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all constraints and compute combined score."""
        results, total_score, total_weight = [], 0.0, 0.0
        
        for c in self.constraints:
            result = c.evaluate(metrics)
            results.append(result)
            
            if c.hard and not result.satisfied:
                return {
                    "combined_score": c.hard_penalty,
                    "all_satisfied": False,
                    "failed_constraint": c.name,
                    "details": {r.name: {"value": r.value, "score": r.score, "satisfied": r.satisfied} for r in results},
                }
            if not c.hard:
                total_score += result.score
                total_weight += c.weight
        
        return {
            "combined_score": total_score / total_weight if total_weight > 0 else total_score,
            "all_satisfied": True,
            "details": {r.name: {"value": r.value, "score": r.score, "satisfied": r.satisfied} for r in results},
        }
    
    def __add__(self, other: Union[Constraint, "ConstraintSet"]) -> "ConstraintSet":
        """Compose constraints using + operator."""
        if isinstance(other, Constraint):
            return ConstraintSet(self.constraints + [other])
        if isinstance(other, ConstraintSet):
            return ConstraintSet(self.constraints + other.constraints)
        raise TypeError(f"Cannot add ConstraintSet with {type(other)}")
    
    def __repr__(self) -> str:
        return f"ConstraintSet({self.constraints})"

