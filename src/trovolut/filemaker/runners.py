"""
Backend-specific execution strategies for OpenEvolve evaluators.

A **Runner** encapsulates *how* to execute a candidate program and collect
metrics for a given backend.  The generated evaluator file simply calls
``runner.evaluate(program_path, ...)`` and the runner takes care of
everything: loading the module, running the dynamics, computing error,
scoring constraints, and returning an OpenEvolve-compatible dict.

Runners
-------
BaseRunner      -- abstract interface
CPURunner       -- in-process execution with wall-clock timing
LegateRunner    -- subprocess execution via ``legate --profile`` with GPU
                   metrics extracted from ``legion_prof`` / DuckDB
QPURunner       -- placeholder for future QPU execution (raises NotImplementedError)

Example (generated evaluator file)::

    from trovolut.filemaker.runners import CPURunner

    runner = CPURunner()

    def evaluate(program_path):
        return runner.evaluate(
            program_path,
            test_generator=get_test_cases,
            exact_fn=exact_evolution,
            constraints=constraints,
            n_trials=10,
            timeout=30,
        )
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import time as _time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import scipy.linalg as la


# ============================================================
# Helpers
# ============================================================

def _load_module(path: str):
    """Load a Python module from *path* and return it."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run_with_timeout(func, args=(), kwargs=None, timeout_seconds=30):
    """Run *func* in a thread with a hard timeout."""
    if kwargs is None:
        kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timed out after {timeout_seconds}s")


def _default_exact(H_terms, t, psi0):
    """Fallback exact evolution via ``scipy.linalg.expm``."""
    if isinstance(H_terms, (list, tuple)):
        H = sum(np.asarray(h) for h in H_terms)
    else:
        H = np.asarray(H_terms)
    return la.expm(-1j * H * t) @ np.asarray(psi0)


def _score_constraints(constraints, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate *constraints* (a ``Group``, list, or single ``Constraint``)
    against *metrics* and merge the results back into *metrics*.

    Returns the updated metrics dict with ``combined_score``,
    ``all_satisfied``, and ``details`` keys added.
    """
    from trovolut.core.base_constraints import Group, Constraint

    if constraints is None:
        return metrics

    # Normalise to a Group
    if isinstance(constraints, Group):
        group = constraints
    elif isinstance(constraints, (list, tuple)):
        # List may contain a single Group
        if len(constraints) == 1 and isinstance(constraints[0], Group):
            group = constraints[0]
        else:
            group = Group(list(constraints))
    elif isinstance(constraints, Constraint):
        group = Group([constraints])
    else:
        return metrics

    result = group.evaluate(metrics)
    metrics.update(result)
    return metrics


# ============================================================
# BaseRunner
# ============================================================

class BaseRunner(ABC):
    """Abstract execution strategy for OpenEvolve evaluators.

    Subclasses implement :meth:`_collect_metrics` which does the actual
    backend-specific work (in-process call, Legate subprocess, etc.).
    The public :meth:`evaluate` method adds timeout, error handling,
    and constraint scoring on top.
    """

    @abstractmethod
    def _collect_metrics(
        self,
        program_path: str,
        test_generator: Callable,
        exact_fn: Optional[Callable],
        n_trials: int,
        run_fn_name: str,
    ) -> Dict[str, Any]:
        """Run the candidate and return raw metrics.

        Must return a dict with at least ``time_seconds``, ``mean_error``,
        ``std_error``, and ``error`` keys.
        """
        ...

    def evaluate(
        self,
        program_path: str,
        test_generator: Callable,
        exact_fn: Optional[Callable] = None,
        constraints=None,
        n_trials: int = 10,
        timeout: int = 30,
        run_fn_name: str = "run_dynamics",
    ) -> Dict[str, Any]:
        """Run candidate, collect metrics, score constraints.

        Parameters
        ----------
        program_path : str
            Path to the candidate Python file.
        test_generator : callable
            ``() -> (hams, psi0, t)`` or ``() -> (hams, psi0)``.
        exact_fn : callable, optional
            ``(hams, t, psi0) -> psi_exact`` for error computation.
        constraints : Group, list of Constraint, or Constraint, optional
            Constraints to evaluate.
        n_trials : int
            Number of trials.
        timeout : int
            Hard timeout in seconds.
        run_fn_name : str
            Name of the function to call in the candidate module.

        Returns
        -------
        dict
            OpenEvolve-compatible result with ``combined_score``,
            ``std_score``, ``all_satisfied``, ``details``, etc.
        """
        try:
            metrics = _run_with_timeout(
                self._collect_metrics,
                args=(program_path, test_generator, exact_fn, n_trials, run_fn_name),
                timeout_seconds=timeout,
            )
        except TimeoutError:
            return {
                "combined_score": -100.0,
                "std_score": 0.0,
                "all_satisfied": False,
                "details": {},
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "combined_score": -100.0,
                "std_score": 0.0,
                "all_satisfied": False,
                "details": {},
                "error": str(e),
            }

        # Score constraints
        metrics = _score_constraints(constraints, metrics)

        # Build final OpenEvolve-compatible dict
        combined = float(metrics.get("combined_score", -100.0))
        std = float(metrics.get("std_score", 0.0))
        all_sat = metrics.get("all_satisfied", True)
        details = metrics.get("details", {})

        return {
            "combined_score": combined,
            "std_score": std,
            "all_satisfied": all_sat,
            "details": details,
            # Pass through common metrics
            "time": metrics.get("time_seconds", 0.0),
            "error": metrics.get("error", metrics.get("mean_error", 0.0)),
            "memory_peak_bytes": metrics.get("memory_peak_bytes", 0),
            # GPU-specific (present only for LegateRunner)
            "gpu_seconds": metrics.get("gpu_seconds", None),
            "makespan_seconds": metrics.get("makespan_seconds", None),
            "peak_fb_mb": metrics.get("peak_fb_mb", None),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ============================================================
# CPURunner
# ============================================================

class CPURunner(BaseRunner):
    """Run candidate in-process on CPU.

    Loads the candidate module, calls its ``run_dynamics`` function,
    measures wall-clock time, and computes error versus exact evolution.
    """

    def _collect_metrics(
        self,
        program_path: str,
        test_generator: Callable,
        exact_fn: Optional[Callable],
        n_trials: int,
        run_fn_name: str,
    ) -> Dict[str, Any]:
        module = _load_module(program_path)

        if not hasattr(module, run_fn_name):
            raise AttributeError(
                f"Candidate module missing '{run_fn_name}' function"
            )
        run_fn = getattr(module, run_fn_name)
        if exact_fn is None:
            exact_fn = _default_exact

        times: List[float] = []
        errors: List[float] = []

        for _ in range(n_trials):
            tc = test_generator()
            if len(tc) == 2:
                hams, psi0 = tc
                t = 1.0
            else:
                hams, psi0, t = tc[0], tc[1], tc[2]

            # Run candidate
            start = _time.perf_counter()
            psi_out = run_fn(hams, t, psi0)
            elapsed = _time.perf_counter() - start
            times.append(elapsed)

            # Handle tuple returns (e.g. TrotterSimulator returns (state, stats))
            if isinstance(psi_out, tuple):
                psi_out = psi_out[0]

            # Compute error
            psi_exact = exact_fn(hams, t, psi0)
            err = float(np.linalg.norm(np.asarray(psi_out) - np.asarray(psi_exact)))
            errors.append(err)

        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))

        metrics: Dict[str, Any] = {
            "time_seconds": float(np.mean(times)),
            "time": float(np.mean(times)),
            "error": mean_error,
            "mean_error": mean_error,
            "std_error": std_error,
            "std_score": std_error / (1.0 + mean_error) ** 2,
            "memory_peak_bytes": 0,
        }

        # Spacetime cost (gpu-mp-emulator): find any object with a
        # cost_profile() method and merge the full cost data.
        #
        # Strategy:
        #   1. Check module-level attributes (evolution, simulator, ...)
        #   2. Fallback: check TrotterSimulator._last_instance which is
        #      set automatically when a TrotterSimulator is created
        #      (even inside a local function scope).
        cost_source = None
        for attr_name in ("evolution", "simulator", "evo", "sim"):
            obj = getattr(module, attr_name, None)
            if obj is not None and hasattr(obj, "cost_profile"):
                cost_source = obj
                break

        if cost_source is None:
            try:
                from trovolut.backends.gpu_mp_emulator.dynamics import TrotterSimulator
                if TrotterSimulator._last_instance is not None:
                    cost_source = TrotterSimulator._last_instance
            except ImportError:
                pass

        if cost_source is not None:
            try:
                tc = test_generator()
                hams = tc[0]
                profile = cost_source.cost_profile(hams)
                if profile is not None:
                    # Merge full cost data so all constraints can score
                    if hasattr(profile, "to_dict"):
                        metrics.update(profile.to_dict())
                    else:
                        metrics["peak_memory_bytes"] = getattr(
                            profile, "peak_memory_bytes", 0
                        )
            except Exception:
                pass

        return metrics


# ============================================================
# LegateRunner
# ============================================================

class LegateRunner(BaseRunner):
    """Run candidate under Legate with ``--profile`` and collect GPU metrics.

    Uses :class:`~trovolut.backends.gpu.profiler.LegateProfiler` to launch
    the candidate via ``uv run legate --profile``, parses the resulting
    ``legion_prof`` profile via DuckDB, and returns real GPU hardware
    metrics alongside approximation error.

    Parameters
    ----------
    gpus : int
        Number of GPUs to allocate (default 1).
    cpus : int
        Number of CPU cores (default 4).
    fbmem_mb : int
        Framebuffer memory per GPU in MiB (default 256).
    workdir : Path or str
        Working directory for profile artifacts (default ``"."``).
    """

    def __init__(
        self,
        gpus: int = 1,
        cpus: int = 4,
        fbmem_mb: int = 256,
        workdir: str | Path = ".",
    ) -> None:
        self.gpus = gpus
        self.cpus = cpus
        self.fbmem_mb = fbmem_mb
        self.workdir = Path(workdir)

    def _collect_metrics(
        self,
        program_path: str,
        test_generator: Callable,
        exact_fn: Optional[Callable],
        n_trials: int,
        run_fn_name: str,
    ) -> Dict[str, Any]:
        from trovolut.backends.gpu.profiler import LegateProfiler

        profiler = LegateProfiler(
            gpus=self.gpus,
            cpus=self.cpus,
            fbmem_mb=self.fbmem_mb,
            workdir=self.workdir,
        )

        if exact_fn is None:
            exact_fn = _default_exact

        errors: List[float] = []
        gpu_metrics_last = None

        for _ in range(n_trials):
            tc = test_generator()
            if len(tc) == 2:
                hams, psi0 = tc
                t = 1.0
            else:
                hams, psi0, t = tc[0], tc[1], tc[2]

            # Ensure lists for profiler
            if not isinstance(hams, (list, tuple)):
                hams = [hams]

            # Run under Legate with profiling
            result = profiler.run(program_path, (hams, psi0, t))

            # Compute error
            psi_exact = exact_fn(hams, t, psi0)
            err = float(
                np.linalg.norm(
                    np.asarray(result.psi_final) - np.asarray(psi_exact)
                )
            )
            errors.append(err)

            # Profile the last run
            gpu_metrics_last = profiler.profile(result.prof_dir)

        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))

        # Build metrics dict
        metrics: Dict[str, Any] = {
            "error": mean_error,
            "mean_error": mean_error,
            "std_error": std_error,
            "std_score": std_error / (1.0 + mean_error) ** 2,
        }

        # Merge GPU profiling data from last run
        if gpu_metrics_last is not None:
            metrics.update(gpu_metrics_last.to_dict())
            metrics["time_seconds"] = gpu_metrics_last.gpu_seconds
            metrics["time"] = gpu_metrics_last.gpu_seconds

        return metrics

    def __repr__(self) -> str:
        return f"LegateRunner(gpus={self.gpus}, cpus={self.cpus}, fbmem_mb={self.fbmem_mb})"


# ============================================================
# QPURunner
# ============================================================

# ============================================================
# GenericRunner
# ============================================================

class GenericRunner(BaseRunner):
    """Run ANY Python script as a subprocess and parse metrics from stdout.

    This runner enables the autoresearch loop to optimize arbitrary quantum
    simulation code — not just trovolut's ``run_dynamics`` signature.  The
    target script is executed as a subprocess; metrics are parsed from its
    stdout as JSON (``{"time": 1.5, "error": 0.01}``) or key-value lines
    (``time: 1.5``).

    Parameters
    ----------
    timeout : int
        Hard timeout per execution in seconds (default 300).
    metric_parser : str
        ``"json"`` or ``"keyvalue"`` (default ``"json"``).
    env : dict, optional
        Extra environment variables to pass to the subprocess.
    """

    def __init__(
        self,
        timeout: int = 300,
        metric_parser: str = "json",
        env: Optional[dict] = None,
    ) -> None:
        self._timeout = timeout
        self._metric_parser = metric_parser
        self._env = env or {}

    # -- metric parsing ----------------------------------------------------

    @staticmethod
    def _parse_json(stdout: str) -> Dict[str, Any]:
        """Parse JSON metrics from the last JSON object in stdout."""
        import json as _json

        # Find the last line that looks like JSON
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return _json.loads(line)
                except _json.JSONDecodeError:
                    continue
        return {}

    @staticmethod
    def _parse_keyvalue(stdout: str) -> Dict[str, Any]:
        """Parse key-value metrics from stdout (``key: value`` lines)."""
        metrics: Dict[str, Any] = {}
        for line in stdout.strip().splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                val = val.strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
        return metrics

    def _parse_metrics(self, stdout: str) -> Dict[str, Any]:
        if self._metric_parser == "keyvalue":
            return self._parse_keyvalue(stdout)
        return self._parse_json(stdout)

    # -- execution ---------------------------------------------------------

    def _run_script(self, program_path: str) -> str:
        """Execute the script as a subprocess and return stdout."""
        import subprocess
        import os

        env = {**os.environ, **self._env}
        result = subprocess.run(
            ["python", program_path],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Script failed (exit {result.returncode}):\n{result.stderr[:2000]}"
            )
        return result.stdout

    def _collect_metrics(
        self,
        program_path: str,
        test_generator: Callable,
        exact_fn: Optional[Callable],
        n_trials: int,
        run_fn_name: str,
    ) -> Dict[str, Any]:
        """Run script as subprocess and parse metrics from stdout."""
        stdout = self._run_script(program_path)
        metrics = self._parse_metrics(stdout)

        # Ensure required keys exist
        metrics.setdefault("time_seconds", metrics.get("time", 0.0))
        metrics.setdefault("time", metrics.get("time_seconds", 0.0))
        metrics.setdefault("error", metrics.get("mean_error", 0.0))
        metrics.setdefault("mean_error", metrics.get("error", 0.0))
        metrics.setdefault("std_error", 0.0)
        metrics.setdefault("std_score", 0.0)
        metrics.setdefault("memory_peak_bytes", 0)

        return metrics

    def evaluate(
        self,
        program_path: str,
        test_generator: Optional[Callable] = None,
        exact_fn: Optional[Callable] = None,
        constraints=None,
        n_trials: int = 1,
        timeout: int = 300,
        run_fn_name: str = "run_dynamics",
    ) -> Dict[str, Any]:
        """Run candidate and parse metrics.

        Overrides base to simplify signature — ``test_generator`` and
        ``exact_fn`` are optional since the script handles its own I/O.
        """
        try:
            metrics = _run_with_timeout(
                self._collect_metrics,
                args=(program_path, test_generator, exact_fn, n_trials, run_fn_name),
                timeout_seconds=timeout or self._timeout,
            )
        except TimeoutError:
            return {
                "combined_score": -100.0,
                "std_score": 0.0,
                "all_satisfied": False,
                "details": {},
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "combined_score": -100.0,
                "std_score": 0.0,
                "all_satisfied": False,
                "details": {},
                "error": str(e),
            }

        metrics = _score_constraints(constraints, metrics)

        combined = float(metrics.get("combined_score", -100.0))
        std = float(metrics.get("std_score", 0.0))
        all_sat = metrics.get("all_satisfied", True)
        details = metrics.get("details", {})

        return {
            "combined_score": combined,
            "std_score": std,
            "all_satisfied": all_sat,
            "details": details,
            "time": metrics.get("time", 0.0),
            "error": metrics.get("error", 0.0),
            "memory_peak_bytes": metrics.get("memory_peak_bytes", 0),
            "gpu_memory_mb": metrics.get("gpu_memory_mb", None),
            "cuda_time_seconds": metrics.get("cuda_time_seconds", None),
        }

    def __repr__(self) -> str:
        return f"GenericRunner(metric_parser={self._metric_parser!r})"


# ============================================================
# PyTorchRunner
# ============================================================

class PyTorchRunner(GenericRunner):
    """Run PyTorch scripts with automatic CUDA instrumentation.

    Inherits from GenericRunner but wraps the target script to inject
    ``torch.cuda.reset_peak_memory_stats()`` / timing, adding
    ``gpu_memory_mb`` and ``cuda_time_seconds`` to parsed metrics.

    Parameters
    ----------
    timeout : int
        Hard timeout per execution in seconds.
    metric_parser : str
        ``"json"`` or ``"keyvalue"``.
    device : str
        CUDA device identifier (default ``"cuda:0"``).
    """

    def __init__(
        self,
        timeout: int = 300,
        metric_parser: str = "json",
        device: str = "cuda:0",
        env: Optional[dict] = None,
    ) -> None:
        super().__init__(timeout=timeout, metric_parser=metric_parser, env=env)
        self._device = device

    def _run_script(self, program_path: str) -> str:
        """Execute the script with CUDA instrumentation wrapper."""
        import subprocess
        import os
        import tempfile

        # Generate wrapper script that adds CUDA instrumentation
        wrapper_code = f"""\
import sys, time, json, torch

device = torch.device("{self._device}" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"

if use_cuda:
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

wall_start = time.perf_counter()

# Execute the target script
exec(open("{program_path}").read())

wall_time = time.perf_counter() - wall_start

cuda_metrics = {{}}
if use_cuda:
    end_event.record()
    torch.cuda.synchronize(device)
    cuda_ms = start_event.elapsed_time(end_event)
    mem_stats = torch.cuda.memory_stats(device)
    peak_bytes = mem_stats.get("allocated_bytes.all.peak", 0)
    cuda_metrics["cuda_time_seconds"] = cuda_ms / 1000.0
    cuda_metrics["gpu_memory_mb"] = peak_bytes / (1024 * 1024)
    cuda_metrics["gpu_memory_peak_bytes"] = peak_bytes

cuda_metrics["wall_time_seconds"] = wall_time
print("CUDA_INSTRUMENTATION:" + json.dumps(cuda_metrics))
"""
        env = {**os.environ, **self._env}
        # Write wrapper to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=os.environ.get("TMPDIR", "/tmp")
        ) as f:
            f.write(wrapper_code)
            wrapper_path = f.name

        try:
            result = subprocess.run(
                ["python", wrapper_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
            )
        finally:
            os.unlink(wrapper_path)

        if result.returncode != 0:
            raise RuntimeError(
                f"Script failed (exit {result.returncode}):\n{result.stderr[:2000]}"
            )
        return result.stdout

    def _collect_metrics(
        self,
        program_path: str,
        test_generator: Callable,
        exact_fn: Optional[Callable],
        n_trials: int,
        run_fn_name: str,
    ) -> Dict[str, Any]:
        """Run script and merge CUDA instrumentation metrics."""
        stdout = self._run_script(program_path)

        # Extract CUDA instrumentation line
        import json as _json
        cuda_metrics: Dict[str, Any] = {}
        remaining_lines = []
        for line in stdout.splitlines():
            if line.startswith("CUDA_INSTRUMENTATION:"):
                try:
                    cuda_metrics = _json.loads(line[len("CUDA_INSTRUMENTATION:"):])
                except _json.JSONDecodeError:
                    pass
            else:
                remaining_lines.append(line)

        # Parse user metrics from remaining stdout
        metrics = self._parse_metrics("\n".join(remaining_lines))

        # Merge CUDA metrics (CUDA instrumentation takes precedence)
        metrics.update(cuda_metrics)

        # Ensure required keys
        metrics.setdefault("time_seconds", metrics.get("time", metrics.get("wall_time_seconds", 0.0)))
        metrics.setdefault("time", metrics.get("time_seconds", 0.0))
        metrics.setdefault("error", metrics.get("mean_error", 0.0))
        metrics.setdefault("mean_error", metrics.get("error", 0.0))
        metrics.setdefault("std_error", 0.0)
        metrics.setdefault("std_score", 0.0)
        metrics.setdefault("memory_peak_bytes", metrics.get("gpu_memory_peak_bytes", 0))

        return metrics

    def __repr__(self) -> str:
        return f"PyTorchRunner(device={self._device!r}, metric_parser={self._metric_parser!r})"


# ============================================================
# QPURunner
# ============================================================

class QPURunner(BaseRunner):
    """Placeholder runner for QPU (quantum processing unit) backends.

    This runner is reserved for future integration with quantum hardware
    execution pipelines (e.g. via CUDA Quantum, Qiskit Runtime, or
    Amazon Braket).  All methods raise ``NotImplementedError``.

    Parameters
    ----------
    device : str
        QPU device identifier (default ``"simulator"``).
    shots : int
        Number of measurement shots (default 1024).
    """

    def __init__(self, device: str = "simulator", shots: int = 1024) -> None:
        self.device = device
        self.shots = shots

    def _collect_metrics(
        self,
        program_path: str,
        test_generator: Callable,
        exact_fn: Optional[Callable],
        n_trials: int,
        run_fn_name: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "QPURunner is not yet implemented. "
            "Use CPURunner or LegateRunner for now."
        )

    def __repr__(self) -> str:
        return f"QPURunner(device={self.device!r}, shots={self.shots})"


# ============================================================
# Factory
# ============================================================

def runner_for_backend(backend: str, **kwargs) -> BaseRunner:
    """Return the appropriate runner for *backend*.

    Parameters
    ----------
    backend : str
        ``'cpu'``, ``'gpu'``, ``'gpu-mp-emulator'``, ``'qpu'``, etc.
    **kwargs
        Extra arguments forwarded to the runner constructor
        (e.g. ``gpus=4`` for ``LegateRunner``).
    """
    if backend == "gpu":
        return LegateRunner(**kwargs)
    if backend == "qpu":
        return QPURunner(**kwargs)
    if backend == "pytorch":
        return PyTorchRunner(**kwargs)
    if backend == "generic":
        return GenericRunner(**kwargs)
    # cpu, gpu-mp-emulator all use in-process execution
    return CPURunner()


# ============================================================
# Exports
# ============================================================

__all__ = [
    "BaseRunner",
    "CPURunner",
    "LegateRunner",
    "GenericRunner",
    "PyTorchRunner",
    "QPURunner",
    "runner_for_backend",
]
