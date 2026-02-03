"""
Base metrics classes for trovolut - hardware-agnostic metrics measurement infrastructure.

This module provides:
- BaseMetrics: Base class for measuring constraints on dynamics functions
"""

from __future__ import annotations
import os, time, json, importlib.util, traceback, math, inspect, sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy.linalg as la

if TYPE_CHECKING:
    from trovolut.qm.hamiltonians import Hamiltonian, HamiltonianEnsemble
    from trovolut.core.base_constraints import Constraint, Group
    from trovolut.filemaker.workflow import Workflow


# =============================================================================
# Utility Functions
# =============================================================================

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
# Base Evaluator (Simple Interface for Backend-Specific Evaluators)
# =============================================================================

class BaseMetrics:
    """
    Base class for measuring constraints on dynamics functions.
    
    The BaseMetrics class:
    1. Takes constraints at initialization
    2. Is callable: when called with a dynamics function, measures and returns metrics
    
    Hardware-specific metrics (CPUMetrics, GPUMetrics, etc.) inherit from this
    and override measurement methods as needed.
    
    Example:
        ```python
        from trovolut.constraints import Time, Error
        from trovolut.backends.cpu.metrics import CPUMetrics
        
        # Initialize with constraints
        metrics = CPUMetrics(constraints=[Time(max_seconds=1.0), Error(max_error=0.01)])
        
        # Define dynamics function
        def my_dynamics(hams, t, psi0):
            return some_evolution(hams, t, psi0)
        
        # Measure metrics by calling
        result = metrics(my_dynamics, test_cases=[(hams, t, psi0)])
        # Returns: {"score": 0.95, "time": 1.2, "error": 0.01, ...}
        ```
    """
    
    def __init__(self, constraints=None):
        """
        Initialize metrics with constraints.
        
        Args:
            constraints: List of Constraint objects or ConstraintSet
        """
        self.constraints = constraints or []
        if hasattr(constraints, '__iter__') and not isinstance(constraints, str):
            self.constraints = list(constraints)
        else:
            self.constraints = [constraints] if constraints else []
    
    def __call__(
        self,
        dynamics_func,
        test_cases=None,
        program_path=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Measure metrics on a dynamics function.
        
        Args:
            dynamics_func: The dynamics function to measure, or None if program_path provided
            test_cases: List of (hamiltonian, psi0, time) test cases
            program_path: Optional path to load function from file
            **kwargs: Additional backend-specific parameters
        
        Returns:
            Dictionary with constraint measurements and scores
        """
        if program_path:
            # Load function from file
            mod = _load_module_from_file(program_path)
            dynamics_func = getattr(mod, kwargs.get('function_name', 'run_dynamics'))
        
        return self.measure(dynamics_func, test_cases, **kwargs)
    
    def measure(
        self,
        dynamics_func,
        test_cases=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Override this method in backend-specific metrics classes.
        
        Args:
            dynamics_func: The dynamics function to measure
            test_cases: List of test cases
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with measurements
        """
        raise NotImplementedError("Subclasses must implement measure()")


__all__ = [
    "BaseMetrics",
]
