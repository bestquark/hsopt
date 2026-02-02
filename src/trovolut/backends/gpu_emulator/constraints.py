"""
Constraint system for composable evaluation.

Constraints measure specific aspects of program execution and convert them to scores.
They can be hard (must satisfy) or soft (contribute proportionally to final score).

Example:
    time_hard = Time(max_seconds=10, hard=True)
    error = Error(weight=1.0)
    constraints = ConstraintSet([time_hard, error])
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union, TYPE_CHECKING
import math
import inspect
import textwrap

if TYPE_CHECKING:
    from trovolut.filemaker.workflow import Workflow


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
    """Base class for all constraints."""
    
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
        satisfied = bool(self.is_satisfied(value))  # Convert numpy bool to Python bool
        score = self.hard_penalty if (self.hard and not satisfied) else self.score(value) * self.weight
        # Convert numpy types to Python types for JSON serialization
        value = float(value) if hasattr(value, 'item') else float(value)
        score = float(score) if hasattr(score, 'item') else float(score)
        return ConstraintResult(self.name, value, score, satisfied, {"hard": self.hard, "weight": self.weight})
    
    def to_config(self) -> Dict[str, Any]:
        """Export constraint parameters for template serialization."""
        config = {"type": self.__class__.__name__, "name": self.name, 
                  "weight": self.weight, "hard": self.hard, "hard_penalty": self.hard_penalty}
        # Add all public attributes that aren't in base
        for attr in ["max_seconds", "max_error", "max_mb", "max_gpu_seconds", 
                     "n_gpus", "score_fn", "metric", "min_concurrency", "max_count"]:
            if hasattr(self, attr):
                config[attr] = getattr(self, attr)
        return config
    
    def __repr__(self) -> str:
        # Collect key hyperparameters
        params = []
        for attr in ["max_seconds", "max_error", "max_mb", "max_gpu_seconds", "min_concurrency", "max_count"]:
            if hasattr(self, attr):
                params.append(f"{getattr(self, attr)}")
                break
        if hasattr(self, "n_gpus") and self.n_gpus > 1:
            params.append(f"n_gpus={self.n_gpus}")
        params.append("hard" if self.hard else f"w={self.weight}")
        return f"{self.__class__.__name__}({', '.join(params)})"
    
    def formula(self) -> str:
        """Return a human-readable explanation of the scoring formula."""
        raise NotImplementedError("Subclasses should implement formula()")
    
    def explain(self) -> None:
        """Print a detailed explanation of this constraint."""
        print(f"┌─ {self.__class__.__name__} ─")
        print(f"│  {self.formula()}")
        if self.hard:
            print(f"│  Mode: HARD (penalty={self.hard_penalty} if violated)")
        else:
            print(f"│  Mode: soft (weight={self.weight})")
            print(f"│  Contribution: score × {self.weight}")
        print(f"└─")


# =============================================================================
# Core Constraints
# =============================================================================

class Time(Constraint):
    """Constraint on execution time."""
    
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
    """Constraint on simulation error (lower is better; 0 = perfect).
    
    Measures the error/deviation between candidate and reference results.
    - Value of 0 = perfect match (no error)
    - Default metric: fidelity error = 1 - |⟨ref|result⟩|²
    """
    
    def __init__(self, max_error: float = 1.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "inverse"):
        """Initialize error constraint.
        
        Args:
            max_error: Maximum allowed error (default: 1.0). For hard constraints,
                      programs with error > max_error will receive hard_penalty.
            weight: Importance weight for soft constraints (default: 1.0).
            hard: If True, error must be <= max_error (default: False).
            hard_penalty: Score penalty for violating hard constraint (default: -100.0).
            score_fn: Scoring function ('inverse', 'exponential', or 'linear').
                     - 'inverse': 1/(1+error) - smooth, never reaches 0
                     - 'exponential': exp(-error) - steep near 0
                     - 'linear': max(0, 1-error/max_error) - linear decay
        """
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


class CPUTime(Constraint):
    """Constraint on CPU execution time (user + system).
    
    Measures actual CPU time consumed (not wall-clock time).
    Uses process-level CPU accounting via psutil.
    """
    
    def __init__(self, max_cpu_ms: float = 1000.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "exponential"):
        super().__init__("cpu_time", weight, hard, hard_penalty)
        self.max_cpu_ms = max_cpu_ms
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("cpu_time_ms", float("inf"))
    
    def score(self, v: float) -> float:
        if self.score_fn == "exponential":
            import math
            return math.exp(-v / self.max_cpu_ms)
        return _score_fn(v, self.max_cpu_ms, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_cpu_ms
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"CPU time MUST be under {self.max_cpu_ms}ms (hard requirement). Use efficient algorithms, sparse operations, vectorization."
        return f"Minimize CPU time (target: <{self.max_cpu_ms}ms, weight: {self.weight}). Prefer sparse matrices, reduced precision, efficient algorithms."
    
    def formula(self) -> str:
        if self.score_fn == "exponential":
            hard_str = f" (hard, penalty={self.hard_penalty})" if self.hard else ""
            return f"score = exp(-cpu_ms/{self.max_cpu_ms}){hard_str}"
        return _formula_str("cpu_time", self.max_cpu_ms, self.score_fn, self.hard, self.hard_penalty, self.weight)


class Memory(Constraint):
    """Constraint on memory usage (RSS - resident set size)."""
    
    def __init__(self, max_mb: float = 1000.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("memory", weight, hard, hard_penalty)
        self.max_mb = max_mb
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("memory_mb", 0.0)
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_mb, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_mb
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"Memory usage MUST stay below {self.max_mb} MB (hard requirement)."
        return f"Prefer lower memory (target: <{self.max_mb} MB, weight: {self.weight})."
    
    def formula(self) -> str:
        return _formula_str("memory", self.max_mb, self.score_fn, self.hard, self.hard_penalty, self.weight)


class GPUTime(Constraint):
    """Constraint on GPU compute time (space-time product)."""
    
    def __init__(self, max_gpu_seconds: float = 100.0, n_gpus: int = 1, weight: float = 1.0,
                 hard: bool = False, hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("gpu_time", weight, hard, hard_penalty)
        self.max_gpu_seconds = max_gpu_seconds
        self.n_gpus = n_gpus
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        gpu_time = m.get("gpu_time_seconds", m.get("time_seconds", float("inf")))
        return gpu_time * self.n_gpus
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_gpu_seconds, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_gpu_seconds
    
    def to_prompt(self) -> str:
        gpu = f"{self.n_gpus} GPU(s)" if self.n_gpus > 1 else "GPU"
        if self.hard:
            return f"Total {gpu} time MUST stay below {self.max_gpu_seconds} GPU-seconds (hard)."
        return f"Minimize {gpu} time (target: <{self.max_gpu_seconds} GPU-s, weight: {self.weight})."
    
    def formula(self) -> str:
        var = f"gpu_time × {self.n_gpus}" if self.n_gpus > 1 else "gpu_time"
        return _formula_str(var, self.max_gpu_seconds, self.score_fn, self.hard, self.hard_penalty, self.weight)


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


# =============================================================================
# Legate GPU Profiling Constraints
# =============================================================================

class LegateGPUSeconds(Constraint):
    """GPU compute time via Legion profiler (sum of kernel durations)."""
    
    def __init__(self, max_gpu_seconds: float = 10.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("legate_gpu_seconds", weight, hard, hard_penalty)
        self.max_gpu_seconds = max_gpu_seconds
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("gpu_seconds", float("inf"))
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_gpu_seconds, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_gpu_seconds
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"GPU compute time (Legate) MUST stay below {self.max_gpu_seconds} GPU-s (hard)."
        return f"Minimize GPU time (target: <{self.max_gpu_seconds} GPU-s, weight: {self.weight})."
    
    def formula(self) -> str:
        return _formula_str("gpu_seconds", self.max_gpu_seconds, self.score_fn, self.hard, self.hard_penalty, self.weight)


class LegateFramebuffer(Constraint):
    """GPU VRAM usage via Legion profiler."""
    
    def __init__(self, max_mb: float = 1000.0, metric: str = "peak", weight: float = 1.0,
                 hard: bool = False, hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__(f"legate_fb_{metric}_mb", weight, hard, hard_penalty)
        self.max_mb = max_mb
        self.metric = metric
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get(f"{self.metric}_fb_mb", 0.0)
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_mb, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_mb
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"GPU VRAM ({self.metric}) MUST stay below {self.max_mb} MB (hard)."
        return f"Minimize GPU VRAM ({self.metric}: <{self.max_mb} MB, weight: {self.weight})."
    
    def formula(self) -> str:
        return _formula_str(f"{self.metric}_fb_mb", self.max_mb, self.score_fn, self.hard, self.hard_penalty, self.weight)


class LegateMakespan(Constraint):
    """Wall-clock makespan via Legion profiler (first to last GPU activity)."""
    
    def __init__(self, max_seconds: float = 30.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("legate_makespan", weight, hard, hard_penalty)
        self.max_seconds = max_seconds
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("makespan_seconds", float("inf"))
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_seconds, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_seconds
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"Makespan MUST stay below {self.max_seconds}s (hard)."
        return f"Minimize makespan (target: <{self.max_seconds}s, weight: {self.weight})."
    
    def formula(self) -> str:
        return _formula_str("makespan", self.max_seconds, self.score_fn, self.hard, self.hard_penalty, self.weight)


class LegateConcurrency(Constraint):
    """GPU utilization via Legion profiler (gpu_seconds / makespan)."""
    
    def __init__(self, min_concurrency: float = 0.5, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0):
        super().__init__("legate_concurrency", weight, hard, hard_penalty)
        self.min_concurrency = min_concurrency
    
    def extract(self, m: Dict[str, Any]) -> float:
        return m.get("avg_concurrency", 0.0)
    
    def score(self, v: float) -> float:
        return min(1.0, v / self.min_concurrency)
    
    def is_satisfied(self, v: float) -> bool:
        return v >= self.min_concurrency
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"GPU concurrency MUST be at least {self.min_concurrency} (hard)."
        return f"Maximize GPU utilization (target: >={self.min_concurrency}, weight: {self.weight})."
    
    def formula(self) -> str:
        base = f"min(1, concurrency/{self.min_concurrency})"
        if self.hard:
            return f"score = {base} · 𝟙(concurrency ≥ {self.min_concurrency}) + ({self.hard_penalty}) · 𝟙(concurrency < {self.min_concurrency})"
        if self.weight != 1.0:
            return f"score = {base}  (×{self.weight} weight)"
        return f"score = {base}"


class MatMulCount(Constraint):
    """Constraint on number of matrix multiplication operations.
    
    Tracks the count of matrix-matrix or matrix-vector multiplications.
    Useful for optimizing algorithmic complexity in small systems where
    wall-clock time isn't reliable.
    """
    
    def __init__(self, max_count: int = 100, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("matmul_count", weight, hard, hard_penalty)
        self.max_count = max_count
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("matmul_count", 0))
    
    def score(self, v: float) -> float:
        return _score_fn(v, self.max_count, self.score_fn)
    
    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_count
    
    def to_prompt(self) -> str:
        if self.hard:
            return f"Number of matrix multiplications MUST stay below {self.max_count} (hard requirement)."
        return f"Minimize matrix multiplications (target: <{self.max_count}, weight: {self.weight})."
    
    def formula(self) -> str:
        return _formula_str("matmul_count", self.max_count, self.score_fn, self.hard, self.hard_penalty, self.weight)


# =============================================================================
# Constraint Set
# =============================================================================

class ConstraintSet:
    """A collection of constraints evaluated together.
    
    Can be used as a decorator on Workflow to auto-generate evaluators:
    
        constraints = ConstraintSet([Time(max_seconds=30), Error()])
        constraints.config_error(exact_fn, setup_fn).entry_point("run_dynamics")
        
        @constraints
        workflow = Workflow([...], out_path="program.py")
        
        workflow.build()  # generates program.py
        workflow.build_evaluator()  # generates evaluator.py
    """
    
    @classmethod
    def _get_builtin_constraint_names(cls) -> set:
        """Dynamically discover all built-in constraint classes in this module."""
        import sys
        import inspect
        
        # Get the current module
        current_module = sys.modules[cls.__module__]
        
        # Find all classes that:
        # 1. Are subclasses of Constraint
        # 2. Are defined in this module (not imported)
        # 3. Are not the abstract Constraint class itself
        builtin_names = set()
        for name, obj in inspect.getmembers(current_module, inspect.isclass):
            if (issubclass(obj, Constraint) and 
                obj is not Constraint and
                obj.__module__ == current_module.__name__):
                builtin_names.add(name)
        
        return builtin_names
    
    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints
        # Evaluator configuration
        self._setup_fn: Callable | None = None
        self._exact_fn: Callable | None = None
        self._error_fn: Callable | None = None
        self._entry_point_name: str = "run_dynamics"
    
    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all constraints and compute combined score."""
        results, total_score, total_weight = [], 0.0, 0.0
        
        for c in self.constraints:
            result = c.evaluate(metrics)
            results.append(result)
            
            if c.hard and not result.satisfied:
                return {
                    "combined_score": c.hard_penalty, "std_score": 0.0,
                    "all_satisfied": False, "failed_constraint": c.name,
                    "details": {r.name: {"value": r.value, "score": r.score, "satisfied": r.satisfied} for r in results},
                }
            if not c.hard:
                total_score += result.score
                total_weight += c.weight
        
        return {
            "combined_score": total_score / total_weight if total_weight > 0 else total_score,
            "std_score": 0.0, "all_satisfied": True,
            "details": {r.name: {"value": r.value, "score": r.score, "satisfied": r.satisfied} for r in results},
        }
    
    def generate_prompt(self) -> str:
        """Generate description of all constraints for LLM prompts."""
        if not self.constraints:
            return "Optimize the code for best performance."
        
        hard = [c for c in self.constraints if c.hard]
        soft = [c for c in self.constraints if not c.hard]
        
        lines = ["You are optimizing quantum simulation code. Your objectives are:"]
        if hard:
            lines.append("\n**Hard requirements (MUST satisfy):**")
            lines.extend(f"- {c.to_prompt()}" for c in hard)
        if soft:
            lines.append("\n**Optimization goals (weighted):**")
            lines.extend(f"- {c.to_prompt()}" for c in soft)
        return "\n".join(lines)
    
    def __add__(self, other: Union[Constraint, "ConstraintSet"]) -> "ConstraintSet":
        if isinstance(other, Constraint):
            return ConstraintSet(self.constraints + [other])
        if isinstance(other, ConstraintSet):
            return ConstraintSet(self.constraints + other.constraints)
        raise TypeError(f"Cannot add ConstraintSet with {type(other)}")
    
    def formula(self) -> str:
        """Generate formula showing how constraints combine into final score."""
        if not self.constraints:
            return "score = 0 (no constraints)"
        
        hard = [c for c in self.constraints if c.hard]
        soft = [c for c in self.constraints if not c.hard]
        
        lines = []
        
        # Show hard constraint guards
        if hard:
            guards = " ∧ ".join(f"({c.name} satisfied)" for c in hard)
            penalties = ", ".join(f"{c.hard_penalty}" for c in hard)
            lines.append(f"Hard guards: {guards}")
            lines.append(f"  → If any violated: score = penalty ({penalties})")
        
        # Show soft constraint weighted average
        if soft:
            total_weight = sum(c.weight for c in soft)
            terms = []
            for c in soft:
                # Simplified score name
                terms.append(f"{c.weight}·s_{c.name}")
            numerator = " + ".join(terms)
            lines.append(f"Soft score: ({numerator}) / {total_weight}")
        
        if not soft and hard:
            lines.append("Final: score = 1.0 if all hard constraints satisfied")
        
        return "\n".join(lines)
    
    def explain(self) -> None:
        """Print detailed explanation of the constraint set scoring."""
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                    CONSTRAINT SET SCORING                    ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
        hard = [c for c in self.constraints if c.hard]
        soft = [c for c in self.constraints if not c.hard]
        
        if hard:
            print("\n┌─ HARD CONSTRAINTS (must all pass) ─────────────────────────┐")
            for c in hard:
                print(f"│  • {c.name}")
                print(f"│      penalty: {c.hard_penalty}")
            print("└─────────────────────────────────────────────────────────────┘")
        
        if soft:
            total_weight = sum(c.weight for c in soft)
            print("\n┌─ SOFT CONSTRAINTS (weighted average) ──────────────────────┐")
            for c in soft:
                pct = 100 * c.weight / total_weight
                print(f"│  • {c.name} ({pct:.0f}%)")
            print("├─────────────────────────────────────────────────────────────┤")
            terms = " + ".join(f"{c.weight}·s_{c.name}" for c in soft)
            print(f"│  combined = ({terms}) / {total_weight}")
            print("└─────────────────────────────────────────────────────────────┘")
        
        print()
    
    def __repr__(self) -> str:
        return f"ConstraintSet({self.constraints})"
    
    # =========================================================================
    # Evaluator Decorator API
    # =========================================================================
    
    def config_error(
        self,
        exact_fn: Callable,
        setup_fn: Callable,
        error_fn: Callable | None = None,
    ) -> "ConstraintSet":
        """Configure how error is computed.
        
        Args:
            exact_fn: Function computing exact/reference result.
                      Signature: exact_fn(*setup_outputs) -> reference_result
            setup_fn: Function returning inputs for the candidate function.
                      Signature: setup_fn() -> tuple of args to pass to candidate
            error_fn: Optional custom error function.
                      Signature: error_fn(result, reference) -> float
                      Default: 1 - |<ref|result>|^2 (fidelity error)
        
        Returns:
            Self for method chaining.
        
        Example:
            def setup():
                hams = get_hamiltonian()
                psi0 = np.zeros(hams[0].shape[0], dtype=complex)
                psi0[0] = 1.0
                return hams, 1.0, psi0  # args for run_dynamics(hams, t, psi0)
            
            def exact(hams, t, psi0):
                return scipy.linalg.expm(-1j * sum(hams) * t) @ psi0
            
            constraints.config_error(exact, setup)
        """
        self._exact_fn = exact_fn
        self._setup_fn = setup_fn
        self._error_fn = error_fn
        return self
    
    def entry_point(self, name: str) -> "ConstraintSet":
        """Set the function name to call in candidate programs.
        
        Args:
            name: Name of the function to evaluate in candidate modules.
                  Default: "run_dynamics"
        
        Returns:
            Self for method chaining.
        """
        self._entry_point_name = name
        return self
    
    @staticmethod
    def _default_fidelity_error(result, reference) -> float:
        """Default error metric: quantum fidelity error."""
        import numpy as np
        return 1.0 - abs(np.vdot(reference, result)) ** 2
    
    def _has_time_constraint(self) -> bool:
        """Check if any constraint measures time."""
        return any(isinstance(c, Time) for c in self.constraints)
    
    def _has_cpu_time_constraint(self) -> bool:
        """Check if any constraint measures CPU time."""
        return any(isinstance(c, CPUTime) for c in self.constraints)
    
    def _has_memory_constraint(self) -> bool:
        """Check if any constraint measures memory."""
        return any(isinstance(c, Memory) for c in self.constraints)
    
    def _has_error_constraint(self) -> bool:
        """Check if any constraint measures error."""
        return any(isinstance(c, Error) for c in self.constraints)
    
    def __call__(self, workflow: "Workflow") -> "Workflow":
        """Decorator to attach constraints to a Workflow.
        
        Usage:
            @constraints
            workflow = trv.filemaker.Workflow(
                [trv.filemaker.EVO_START, run_dynamics, trv.filemaker.EVO_END],
                out_path="opt/program.py"
            )
            
            workflow.build()  # Build the initial program
            workflow.build_evaluator()  # Auto-generate evaluator.py
        """
        workflow._constraints = self
        return workflow
    
    def wrap(self, workflow: "Workflow") -> "Workflow":
        """Attach constraints to a workflow (alternative to decorator).
        
        Usage:
            workflow = constraints.wrap(trv.filemaker.Workflow(...))
        """
        return self(workflow)
    
    def _get_constraint_init_code(self) -> str:
        """Generate code to reconstruct this ConstraintSet."""
        import inspect
        
        lines = []
        builtin_constraints = self._get_builtin_constraint_names()
        
        for c in self.constraints:
            cls_name = c.__class__.__name__
            is_builtin = cls_name in builtin_constraints
            prefix = "trv.eval." if is_builtin else ""
            
            # Get __init__ signature and extract parameters from the instance
            try:
                sig = inspect.signature(c.__class__.__init__)
                params = []
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    # Try to get the value from the instance
                    if hasattr(c, param_name):
                        val = getattr(c, param_name)
                        if callable(val):
                            continue
                        
                        # Format the value appropriately
                        if isinstance(val, str):
                            params.append(f'{param_name}="{val}"')
                        elif isinstance(val, (int, float, bool, type(None))):
                            params.append(f"{param_name}={val}")
                        elif isinstance(val, (list, tuple)):
                            params.append(f"{param_name}={val}")
                        else:
                            # For other types, use repr
                            params.append(f"{param_name}={repr(val)}")
                
                lines.append(f"    {prefix}{cls_name}({', '.join(params)}),")
            except Exception as e:
                # Fallback: just create with minimal params
                lines.append(f"    # WARNING: Could not reconstruct {cls_name}: {e}")
                lines.append(f"    {prefix}{cls_name}(),")
        
        return "constraints = trv.eval.ConstraintSet([\n" + "\n".join(lines) + "\n])"
    
    def _generate_evaluator_code(self) -> str:
        """Generate the full evaluator.py source code."""
        from trovolut.filemaker.workflow import get_source_with_deps
        
        # Check if we need CPU/memory monitoring
        needs_cpu_monitoring = any(isinstance(c, (CPUTime, Memory)) for c in self.constraints)
        
        lines = [
            "import time",
            "import importlib.util",
            "from typing import List, Tuple, Dict, Any, Optional, Sequence, Union",
            "import numpy as np",
            "import scipy.linalg",
        ]
        
        if needs_cpu_monitoring:
            lines.extend([
                "import psutil",
                "import os",
            ])
        
        lines.extend([
            "import trovolut as trv",
            "from trovolut.qm.hamiltonians import *  # noqa: F401,F403",
            "",
        ])
        
        visited = set()
        
        # Add custom constraint class definitions
        builtin_constraints = self._get_builtin_constraint_names()
        for c in self.constraints:
            cls_name = c.__class__.__name__
            if cls_name not in builtin_constraints:
                # This is a custom constraint - include its source
                try:
                    src = get_source_with_deps(c.__class__, visited=visited, root_name="trv")
                    lines.append("# Custom constraint definition")
                    lines.append(src)
                    lines.append("")
                except Exception as e:
                    # If we can't get source, at least warn
                    lines.append(f"# WARNING: Could not extract source for {cls_name}: {e}")
                    lines.append("")
        
        # Add setup function
        if self._setup_fn is not None:
            src = get_source_with_deps(self._setup_fn, visited=visited, root_name="trv")
            lines.append(src)
            lines.append("")
        
        # Add exact function
        if self._exact_fn is not None:
            src = get_source_with_deps(self._exact_fn, visited=visited, root_name="trv")
            lines.append(src)
            lines.append("")
        
        # Add custom error function if provided
        if self._error_fn is not None:
            src = get_source_with_deps(self._error_fn, visited=visited, root_name="trv")
            lines.append(src)
            lines.append("")
        
        # Add constraint reconstruction
        lines.append(self._get_constraint_init_code())
        lines.append("")
        
        # Generate the evaluate function
        entry_point = self._entry_point_name
        setup_fn_name = self._setup_fn.__name__ if self._setup_fn else None
        exact_fn_name = self._exact_fn.__name__ if self._exact_fn else None
        error_fn_name = self._error_fn.__name__ if self._error_fn else None
        has_time = self._has_time_constraint()
        has_cpu_time = self._has_cpu_time_constraint()
        has_memory = self._has_memory_constraint()
        has_error = self._has_error_constraint()
        
        eval_lines = [
            "",
            'def evaluate(program_path: str) -> dict:',
            '    """Auto-generated evaluator function."""',
            '    # Load candidate module',
            '    spec = importlib.util.spec_from_file_location("candidate", program_path)',
            '    module = importlib.util.module_from_spec(spec)',
            '    spec.loader.exec_module(module)',
            '',
            f'    # Get the function to evaluate',
            f'    candidate_fn = getattr(module, "{entry_point}")',
            '',
            '    metrics = {}',
        ]
        
        # Setup inputs
        if setup_fn_name:
            eval_lines.extend([
                '',
                '    # Setup inputs',
                f'    inputs = {setup_fn_name}()',
                '    if not isinstance(inputs, tuple):',
                '        inputs = (inputs,)',
            ])
        else:
            eval_lines.append('    inputs = ()')
        
        # Run with appropriate monitoring
        if has_cpu_time or has_memory:
            eval_lines.extend([
                '',
                '    # Monitor CPU time and memory',
                '    process = psutil.Process(os.getpid())',
            ])
            if has_cpu_time:
                eval_lines.append('    cpu_times_start = process.cpu_times()')
            if has_memory:
                eval_lines.append('    mem_start = process.memory_info().rss / (1024 * 1024)  # MB')
            eval_lines.extend([
                '    ',
                '    result = candidate_fn(*inputs)',
                '    ',
            ])
            if has_cpu_time:
                eval_lines.extend([
                    '    # Measure CPU time (user + system) in milliseconds',
                    '    cpu_times_end = process.cpu_times()',
                    '    cpu_time_s = (cpu_times_end.user - cpu_times_start.user) + (cpu_times_end.system - cpu_times_start.system)',
                    '    metrics["cpu_time_ms"] = cpu_time_s * 1000',
                ])
            if has_memory:
                eval_lines.extend([
                    '    # Measure memory delta',
                    '    mem_end = process.memory_info().rss / (1024 * 1024)  # MB',
                    '    metrics["memory_mb"] = max(0, mem_end - mem_start)',
                ])
        elif has_time:
            eval_lines.extend([
                '',
                '    # Run and time (wall clock)',
                '    start = time.perf_counter()',
                '    result = candidate_fn(*inputs)',
                '    metrics["time_seconds"] = time.perf_counter() - start',
            ])
        else:
            eval_lines.extend([
                '',
                '    # Run candidate',
                '    result = candidate_fn(*inputs)',
            ])
        
        # Compute error
        if has_error and exact_fn_name:
            eval_lines.extend([
                '',
                '    # Compute error',
                f'    reference = {exact_fn_name}(*inputs)',
            ])
            if error_fn_name:
                eval_lines.append(f'    metrics["mean_error"] = {error_fn_name}(result, reference)')
            else:
                eval_lines.append('    metrics["mean_error"] = 1.0 - abs(np.vdot(reference, result)) ** 2')
        
        eval_lines.extend([
            '',
            '    return constraints.evaluate(metrics)',
        ])
        
        lines.extend(eval_lines)
        
        return "\n".join(lines)


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# Old-style names
TimeConstraint = Time
CPUTimeConstraint = CPUTime
ErrorConstraint = Error
MemoryConstraint = Memory
GPUTimeConstraint = GPUTime
CustomConstraint = Custom
LegateGPUSecondsConstraint = LegateGPUSeconds
LegateFramebufferConstraint = LegateFramebuffer
LegateMakespanConstraint = LegateMakespan
LegateConcurrencyConstraint = LegateConcurrency
MatMulCountConstraint = MatMulCount
