"""
Base evaluator classes for trovolut - hardware-agnostic evaluation infrastructure.

This module provides:
1. BaseEvaluator: Simple base class for backend-specific evaluators
2. BaseEval: Legacy evaluator with Hamiltonian support (from base_constraints.py)
3. WorkflowConstraintSet: Constraint set with evaluator code generation for OpenEvolve
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
    from trovolut.core.base_constraints import Constraint, ConstraintSet
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
    try:
        import cupy
        return isinstance(x, cupy.ndarray)
    except Exception:
        return False

def _to_numpy(a):
    if _is_cupy_array(a):
        return np.asarray(a.get())
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

class BaseEvaluator(ABC):
    """
    Abstract base class for backend-specific evaluators.
    
    Backend evaluators define HOW to execute and measure code on specific hardware.
    For example:
    - CPUEvaluator: Uses tracemalloc, measures CPU time
    - GPUEvaluator: Uses CUDA profiling, measures GPU memory
    - QPUEvaluator: Compiles circuits, measures gate count
    """
    
    @abstractmethod
    def evaluate(self, program_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a program and collect hardware-specific metrics.
        
        Args:
            program_func: Function to evaluate
            *args, **kwargs: Arguments to pass to program_func
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass


# =============================================================================
# Legacy BaseEval (Hamiltonian-Based Evaluator)
# =============================================================================

class BaseEval:
    """
    Base for concrete evaluators with Hamiltonian support (legacy).
    
    Accepts either:
    - Hamiltonian: Deterministic operator (has .terms(), .to_matrix(), .hilbert_dim)
    - HamiltonianEnsemble: Stochastic family (has .sample(rng) -> Hamiltonian)
    
    For ensembles, each trial samples a new Hamiltonian using the provided seeds.
    For deterministic Hamiltonians, all trials use the same operator.
    """
    def __init__(
        self,
        ham_or_ensemble: Union["Hamiltonian", "HamiltonianEnsemble"],
        trials: int = 5,
        t: float = 0.05,
        seeds: Optional[Sequence[int]] = None,
        psi0: Optional[np.ndarray] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            ham_or_ensemble: Hamiltonian or HamiltonianEnsemble object from trovolut.qm
            trials: Number of evaluation trials (for ensembles, each uses different seed)
            t: Evolution time
            seeds: Optional custom seeds for each trial
            psi0: Optional fixed initial state (if None, random states are generated)
        """
        if ham_or_ensemble is None:
            raise ValueError("Must provide a Hamiltonian or HamiltonianEnsemble.")
        self._src = ham_or_ensemble
        self.trials = int(trials)
        self.t = float(t)
        self.seeds = list(seeds) if seeds else [1337 + i for i in range(trials)]
        self._psi0_fixed = _to_numpy(psi0) if psi0 is not None else None
    
    # --- hooks for subclasses ---
    def _score(self, rows: List[Dict[str, float]]) -> float:
        """Compute a scalar score from per-trial rows. Higher is better."""
        raise NotImplementedError

    # --- core execution ---
    def _draw_problem(self, sd: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Sample a problem instance (Hamiltonian terms and initial state).
        
        For HamiltonianEnsemble (is_stochastic=True):
            - Calls .sample(rng) to draw a concrete Hamiltonian
        For deterministic Hamiltonian:
            - Uses the Hamiltonian directly
        
        Returns:
            Tuple of (H_terms, psi0) where H_terms is a list of numpy arrays
            extracted from ham.terms()
        """
        # If it's an ensemble, sample; else use deterministic Hamiltonian
        if getattr(self._src, "is_stochastic", False) or hasattr(self._src, "sample"):
            rng = np.random.default_rng(sd)
            ham = self._src.sample(rng=rng)
        else:
            ham = self._src

        # Extract terms using the standard .terms() method
        terms = _coerce_terms_from_hamiltonian(ham)
        dim = getattr(ham, "hilbert_dim", _dim_from_terms(terms))
        psi0 = self._psi0_fixed if self._psi0_fixed is not None else _random_state(dim, sd + 17)
        return terms, psi0

    def _run_once(self, program_path: str, sd: int) -> Tuple[bool, Dict[str, Any]]:
        try:
            mod = _load_module_from_file(program_path)
            if not hasattr(mod, "run"):
                return False, {"error": "no_run_fn"}

            H_terms, psi0 = self._draw_problem(sd)

            t0 = time.perf_counter()
            psi_final, meta = mod.run(H_terms, psi0, self.t)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            # exact reference on CPU
            H = sum(H_terms)
            psi_ref = la.expm(-1j * self.t * H) @ psi0

            err = float(np.linalg.norm(psi_final - psi_ref))
            norm_val = float(np.linalg.norm(psi_final))
            # prefer candidate-provided GPU metrics if present
            gpu_time_ms = _safe_float((meta or {}).get("gpu_time_ms"), 0.0) or None
            gpu_mem_mb  = _safe_float((meta or {}).get("gpu_mem_mb"), 0.0) or None
            bytes_used  = _safe_float((meta or {}).get("bytes_used"), 0.0)

            # fallback memory estimate
            mem_mb_est = (bytes_used or _bytes_of(psi_final, psi0, H)) / (1024.0 * 1024.0)

            row = {
                "error": err,
                "norm": norm_val,
                "time_ms": gpu_time_ms if gpu_time_ms is not None else wall_ms,
                "mem_mb": gpu_mem_mb if gpu_mem_mb is not None else mem_mb_est,
            }
            return True, row
        except Exception as e:
            return False, {"error": f"runtime_error:{type(e).__name__}", "trace": traceback.format_exc(limit=6)}

    def evaluate(self, program: Optional[str] = None) -> Dict[str, Any]:
        program_path = program or self._program
        if not program_path or not Path(program_path).exists():
            return {"score": -1e9, "trials": 0, "error": "program_path_invalid"}

        rows: List[Dict[str, float]] = []
        for sd in self.seeds:
            ok, res = self._run_once(program_path, sd)
            if ok:
                rows.append(res)
            else:
                # record a failed trial with heavy penalties
                rows.append({"error": 1e3, "norm": 0.0, "time_ms": 1e6, "mem_mb": 1e6})

        score = self._score(rows)
        out = {
            "score": float(score),
            "err_mean": float(np.mean([r["error"] for r in rows])),
            "err_min": float(np.min([r["error"] for r in rows])),
            "norm_dev": float(abs(np.mean([r["norm"] for r in rows]) - 1.0)),
            "time_ms_mean": float(np.mean([r["time_ms"] for r in rows])),
            "mem_mb_mean": float(np.mean([r["mem_mb"] for r in rows])),
            "trials": len(rows),
        }
        return out


# =============================================================================
# WorkflowConstraintSet - Constraint Set with Evaluator Code Generation
# =============================================================================

class WorkflowConstraintSet:
    """
    Extended ConstraintSet with Workflow integration for auto-generating evaluators.
    
    This class combines constraints with evaluator code generation for OpenEvolve.
    It wraps a ConstraintSet and adds the ability to generate evaluator.py files
    that can be used to test candidate programs.
    
    Example:
        from trovolut.core.base_constraints import ConstraintSet, Time, Error
        from trovolut.core.base_evaluator import WorkflowConstraintSet
        
        constraints = WorkflowConstraintSet(ConstraintSet([Time(max_seconds=30), Error()]))
        constraints.config_error(exact_fn, setup_fn).entry_point("run_dynamics")
        
        @constraints
        workflow = Workflow([...], out_path="program.py")
        
        workflow.build()  # generates program.py
        workflow.build_evaluator()  # generates evaluator.py
    """
    
    def __init__(self, constraint_set: "ConstraintSet"):
        """
        Initialize WorkflowConstraintSet.
        
        Args:
            constraint_set: A ConstraintSet instance containing the constraints
        """
        from trovolut.core.base_constraints import ConstraintSet
        if not isinstance(constraint_set, ConstraintSet):
            raise TypeError(f"Expected ConstraintSet, got {type(constraint_set)}")
        
        self._constraint_set = constraint_set
        # Evaluator configuration
        self._setup_fn = None
        self._exact_fn = None
        self._error_fn = None
        self._entry_point_name = "run_dynamics"
    
    @property
    def constraints(self) -> List["Constraint"]:
        """Access the underlying constraints."""
        return self._constraint_set.constraints
    
    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Delegate to the underlying ConstraintSet."""
        return self._constraint_set.evaluate(metrics)
    
    @classmethod
    def _get_builtin_constraint_names(cls) -> set:
        """Dynamically discover all built-in constraint classes."""
        from trovolut.core import base_constraints
        builtin_names = set()
        for name, obj in inspect.getmembers(base_constraints, inspect.isclass):
            # Import Constraint here to avoid circular imports
            from trovolut.core.base_constraints import Constraint
            if (inspect.isclass(obj) and 
                issubclass(obj, Constraint) and 
                obj is not Constraint):
                builtin_names.add(name)
        return builtin_names
    
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
    
    # =========================================================================
    # Evaluator Configuration API
    # =========================================================================
    
    def config_error(
        self,
        exact_fn,
        setup_fn,
        error_fn = None,
    ) -> "WorkflowConstraintSet":
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
    
    def entry_point(self, name: str) -> "WorkflowConstraintSet":
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
        from trovolut.core.base_constraints import Time
        return any(isinstance(c, Time) for c in self.constraints)
    
    def _has_cpu_time_constraint(self) -> bool:
        """Check if any constraint measures CPU time."""
        # Import here to avoid circular dependency
        try:
            from trovolut.backends.cpu.constraints import CPUTime
            return any(isinstance(c, CPUTime) for c in self.constraints)
        except ImportError:
            return False
    
    def _has_memory_constraint(self) -> bool:
        """Check if any constraint measures memory."""
        # Import here to avoid circular dependency
        try:
            from trovolut.backends.cpu.constraints import Memory
            return any(isinstance(c, Memory) for c in self.constraints)
        except ImportError:
            return False
    
    def _has_error_constraint(self) -> bool:
        """Check if any constraint measures error."""
        from trovolut.core.base_constraints import Error
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
        lines = []
        builtin_constraints = self._get_builtin_constraint_names()
        
        for c in self.constraints:
            cls_name = c.__class__.__name__
            is_builtin = cls_name in builtin_constraints
            prefix = "trv.core." if is_builtin else ""
            
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
        
        return "constraints = trv.core.ConstraintSet([\n" + "\n".join(lines) + "\n])"
    
    def _generate_evaluator_code(self) -> str:
        """Generate the full evaluator.py source code."""
        from trovolut.filemaker.workflow import get_source_with_deps
        
        # Check if we need CPU/memory monitoring
        needs_cpu_monitoring = self._has_cpu_time_constraint() or self._has_memory_constraint()
        
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


__all__ = [
    "BaseEvaluator",
    "BaseEval",
    "WorkflowConstraintSet",
]
