"""
Evaluator builder for generating OpenEvolve-compatible evaluator files.

This module provides builders that configure constraints and generate
standalone evaluator Python files for use with OpenEvolve.

Example:
    builder = EvaluatorBuilder()
    builder.time(max_seconds=10, hard=True)
    builder.accuracy(weight=2.0)
    builder.to_file("evaluator.py")
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

from .constraints import (
    Constraint, ConstraintSet, Time, Accuracy, Memory, GPUTime, Custom,
    LegateGPUSeconds, LegateFramebuffer, LegateMakespan, LegateConcurrency,
)
from .templates import render_evaluator_template, render_legate_evaluator_template


class EvaluatorBuilder:
    """
    Fluent builder for configuring constraints and generating evaluators.
    
    Example:
        builder = (EvaluatorBuilder()
            .time(max_seconds=10, hard=True)
            .accuracy(weight=2.0)
            .memory(max_mb=2000))
        builder.to_file("eval.py")
    """
    
    def __init__(self, use_cupynumeric: bool = False):
        self._constraints: List[Constraint] = []
        self.use_cupynumeric = use_cupynumeric
    
    # -------------------------------------------------------------------------
    # Constraint adders (return self for chaining)
    # -------------------------------------------------------------------------
    
    def time(self, max_seconds: float = 30.0, weight: float = 1.0, hard: bool = False,
             hard_penalty: float = -100.0, score_fn: str = "linear") -> "EvaluatorBuilder":
        self._constraints.append(Time(max_seconds, weight, hard, hard_penalty, score_fn))
        return self
    
    def accuracy(self, max_error: float = 1.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "inverse") -> "EvaluatorBuilder":
        self._constraints.append(Accuracy(max_error, weight, hard, hard_penalty, score_fn))
        return self
    
    def memory(self, max_mb: float = 1000.0, weight: float = 1.0, hard: bool = False,
               hard_penalty: float = -100.0, score_fn: str = "linear") -> "EvaluatorBuilder":
        self._constraints.append(Memory(max_mb, weight, hard, hard_penalty, score_fn))
        return self
    
    def gpu_time(self, max_gpu_seconds: float = 100.0, n_gpus: int = 1, weight: float = 1.0,
                 hard: bool = False, hard_penalty: float = -100.0, score_fn: str = "linear") -> "EvaluatorBuilder":
        self._constraints.append(GPUTime(max_gpu_seconds, n_gpus, weight, hard, hard_penalty, score_fn))
        return self
    
    def custom(self, name: str, extract_fn: Callable[[Dict[str, Any]], float],
               score_fn: Callable[[float], float], satisfy_fn: Callable[[float], bool],
               weight: float = 1.0, hard: bool = False, hard_penalty: float = -100.0) -> "EvaluatorBuilder":
        self._constraints.append(Custom(name, extract_fn, score_fn, satisfy_fn, weight, hard, hard_penalty))
        return self
    
    def add(self, constraint: Constraint) -> "EvaluatorBuilder":
        """Add a pre-built constraint instance."""
        self._constraints.append(constraint)
        return self
    
    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------
    
    @property
    def constraint_set(self) -> ConstraintSet:
        return ConstraintSet(self._constraints)
    
    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate metrics directly using the configured constraints."""
        return self.constraint_set.evaluate(metrics)
    
    def generate_prompt(self) -> str:
        """Generate LLM prompt describing the optimization objectives."""
        return self.constraint_set.generate_prompt()
    
    def _build_configs(self) -> List[Dict[str, Any]]:
        """Build constraint configurations for template serialization."""
        return [c.to_config() for c in self._constraints]
    
    def to_file(self, filepath: str, reference_code: str = "") -> None:
        """Generate a standalone evaluator Python file."""
        content = render_evaluator_template(
            constraint_configs=json.dumps(self._build_configs(), indent=4),
            use_cupynumeric=self.use_cupynumeric,
            reference_code=reference_code,
        )
        Path(filepath).write_text(content)
    
    def __repr__(self) -> str:
        return f"EvaluatorBuilder({self._constraints})"


class LegateEvaluatorBuilder:
    """
    Builder for Legate/Legion-profiled GPU evaluators.
    
    Uses Legion profiler output for accurate GPU timing and memory metrics.
    Generates evaluators that work with `legate` launcher.
    """
    
    def __init__(self, use_cupynumeric: bool = True):
        self._constraints: List[Constraint] = []
        self.use_cupynumeric = use_cupynumeric
    
    # -------------------------------------------------------------------------
    # Legate-specific constraint adders
    # -------------------------------------------------------------------------
    
    def gpu_seconds(self, max_gpu_seconds: float = 10.0, weight: float = 1.0, hard: bool = False,
                    hard_penalty: float = -100.0, score_fn: str = "linear") -> "LegateEvaluatorBuilder":
        self._constraints.append(LegateGPUSeconds(max_gpu_seconds, weight, hard, hard_penalty, score_fn))
        return self
    
    def framebuffer(self, max_mb: float = 1000.0, metric: str = "peak", weight: float = 1.0,
                    hard: bool = False, hard_penalty: float = -100.0, score_fn: str = "linear") -> "LegateEvaluatorBuilder":
        self._constraints.append(LegateFramebuffer(max_mb, metric, weight, hard, hard_penalty, score_fn))
        return self
    
    def makespan(self, max_seconds: float = 30.0, weight: float = 1.0, hard: bool = False,
                 hard_penalty: float = -100.0, score_fn: str = "linear") -> "LegateEvaluatorBuilder":
        self._constraints.append(LegateMakespan(max_seconds, weight, hard, hard_penalty, score_fn))
        return self
    
    def concurrency(self, min_concurrency: float = 0.5, weight: float = 1.0,
                    hard: bool = False, hard_penalty: float = -100.0) -> "LegateEvaluatorBuilder":
        self._constraints.append(LegateConcurrency(min_concurrency, weight, hard, hard_penalty))
        return self
    
    def add(self, constraint: Constraint) -> "LegateEvaluatorBuilder":
        """Add a pre-built constraint instance."""
        self._constraints.append(constraint)
        return self
    
    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------
    
    @property
    def constraint_set(self) -> ConstraintSet:
        return ConstraintSet(self._constraints)
    
    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return self.constraint_set.evaluate(metrics)
    
    def generate_prompt(self) -> str:
        return self.constraint_set.generate_prompt()
    
    def _build_configs(self) -> List[Dict[str, Any]]:
        return [c.to_config() for c in self._constraints]
    
    def to_file(self, filepath: str, reference_code: str = "") -> None:
        """Generate Legate-compatible evaluator with Legion profiling."""
        content = render_legate_evaluator_template(
            constraint_configs=json.dumps(self._build_configs(), indent=4),
            use_cupynumeric=self.use_cupynumeric,
            reference_code=reference_code,
        )
        Path(filepath).write_text(content)
    
    def __repr__(self) -> str:
        return f"LegateEvaluatorBuilder({self._constraints})"
