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
        # Use proper parameter names that match __init__ signatures
        if self.hard:
            params.append("hard=True")
        elif self.weight != 1.0:
            params.append(f"weight={self.weight}")
        return f"{self.__class__.__name__}({', '.join(params)})"


class Time(Constraint):
    """Constraint on wall-clock execution time."""
    
    def __init__(self, max_seconds: float = 30.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear"):
        super().__init__("time", weight, hard, hard_penalty)
        self.max_seconds = max_seconds
        self.score_fn = score_fn
    
    def extract(self, m: Dict[str, Any]) -> float:
        # Try multiple possible key names
        return m.get("time", m.get("time_seconds", float("inf")))
    
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
        # Try multiple possible key names
        return m.get("error", m.get("mean_error", float("inf")))
    
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


class Group:
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
    
    def generate_llm_instructions(self) -> str:
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
    
    def __add__(self, other: Union[Constraint, "Group"]) -> "Group":
        """Compose constraints using + operator."""
        if isinstance(other, Constraint):
            return Group(self.constraints + [other])
        if isinstance(other, Group):
            return Group(self.constraints + other.constraints)
        raise TypeError(f"Cannot add Group with {type(other)}")
    
    def __repr__(self) -> str:
        return f"Group({self.constraints})"

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
    