"""
GPU-backend metrics using real Legate/Legion profiling.

This module provides a ``Metrics`` class that:

1. Runs a dynamics evolution via the Legate launcher with ``--profile``
2. Parses the resulting profile with ``legion_prof load`` into DuckDB
3. Queries real GPU hardware metrics (kernel time, VRAM, concurrency)
4. Computes approximation error versus an exact reference
5. Returns a merged dict compatible with the Legate constraints

The profiling pipeline is handled by ``trovolut.backends.gpu.profiler``.

Example::

    import trovolut as trv
    trv.set_backend("gpu")

    from trovolut.dynamics import TrotterEvolution
    from trovolut.measure import Metrics
    from trovolut.constraints import LegateGPUSeconds, LegateFramebuffer

    constraints = [LegateGPUSeconds(max_gpu_seconds=5.0),
                   LegateFramebuffer(max_mb=512)]

    metrics = Metrics(
        constraints=constraints,
        test_generator=my_test_gen,
        gpus=4,
    )
    result = metrics(TrotterEvolution(trotter_steps=100))
"""

from __future__ import annotations

import os
import pickle
import tempfile
import textwrap
import time as _time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import scipy.linalg as la

from trovolut.core.base_metrics import BaseMetrics
from trovolut.backends.gpu.profiler import LegateProfiler, GPUMetrics


# ============================================================
# Metrics
# ============================================================

class Metrics(BaseMetrics):
    """Evaluator for the GPU backend with real hardware profiling.

    Collects GPU metrics by running dynamics under the Legate launcher with
    ``--profile``, then analysing the resulting ``legion_prof`` profile via
    DuckDB.

    Metrics returned include ``gpu_seconds``, ``makespan_seconds``,
    ``avg_concurrency``, ``peak_fb_mb``, ``avg_fb_mb``, and ``error``.

    Parameters
    ----------
    constraints : list or Group, optional
        Constraint objects to evaluate.
    test_generator : callable, optional
        ``() -> (H_terms, psi0)`` or ``() -> (H_terms, psi0, t)`` for
        generating test cases.
    exact_fn : callable, optional
        ``(H_terms, t, psi0) -> psi_exact`` for error computation.
        Defaults to ``scipy.linalg.expm`` exact evolution.
    n_trials : int
        Number of profiling trials (default 3).
    timeout : int
        Maximum time per trial in seconds (default 60).
    gpus : int
        Number of GPUs for Legate (default 1).
    cpus : int
        Number of CPU cores for Legate (default 4).
    fbmem_mb : int
        Framebuffer memory per GPU in MiB (default 256).
    workdir : Path or str
        Working directory for temporary scripts and profile artifacts.
    t : float
        Default evolution time if not provided by the test generator.
    """

    def __init__(
        self,
        constraints=None,
        test_generator=None,
        exact_fn=None,
        n_trials: int = 3,
        timeout: int = 60,
        gpus: int = 1,
        cpus: int = 4,
        fbmem_mb: int = 256,
        workdir: Path | str = ".",
        t: float = 0.05,
    ) -> None:
        super().__init__(
            constraints=constraints,
            test_generator=test_generator,
            exact_fn=exact_fn,
            n_trials=n_trials,
            timeout=timeout,
            backend="gpu",
        )
        self.profiler = LegateProfiler(
            gpus=gpus, cpus=cpus, fbmem_mb=fbmem_mb, workdir=workdir,
        )
        self.t = t

    # -- callable API ---------------------------------------------------------

    def measure(self, dynamics_evolution, **kwargs) -> Dict[str, Any]:
        """Run dynamics under Legate with profiling and collect GPU metrics.

        Parameters
        ----------
        dynamics_evolution : BaseDynamics
            A dynamics evolution instance (e.g. ``TrotterEvolution``).
        **kwargs
            Extra parameters forwarded to the dynamics ``evolve()`` call.

        Returns
        -------
        dict
            Merged metrics containing profiling data, error, and constraint
            scores.

        Pipeline
        --------
        1. Generate (or accept) test cases: ``(H_terms, psi0, t)``
        2. Serialize a temporary runner script that calls
           ``dynamics_evolution.evolve()`` under cupynumeric
        3. Execute the script via ``LegateProfiler.run()``
        4. Load the profile and query GPU metrics
        5. Compute approximation error vs exact reference
        6. Return merged dict
        """
        # 1. Generate test case
        H_terms, psi0, t = self._get_test_case(**kwargs)

        # 2. Generate the runner script
        script_path = self._generate_runner_script(dynamics_evolution, H_terms, psi0, t)

        # 3. Run via Legate with profiling
        try:
            result = self.profiler.run(
                str(script_path),
                (H_terms, psi0, t),
            )
        except RuntimeError as e:
            return {
                "error": float("inf"),
                "gpu_seconds": 0.0,
                "makespan_seconds": 0.0,
                "avg_concurrency": 0.0,
                "peak_fb_mb": 0.0,
                "avg_fb_mb": 0.0,
                "legate_error": str(e),
                "score": -1e9,
            }

        # 4. Extract GPU metrics from profile
        gpu_metrics = self.profiler.profile(result.prof_dir)

        # 5. Compute error
        error = self._compute_error(H_terms, psi0, t, result.psi_final)

        # 6. Merge all metrics
        metrics: Dict[str, Any] = {
            **gpu_metrics.to_dict(),
            "error": error,
        }

        # Add metadata from the candidate (if any)
        if result.meta:
            metrics.update({
                k: v for k, v in result.meta.items()
                if k not in metrics
            })

        # Compute constraint scores
        if self.constraints:
            scores = []
            for c in self.constraints:
                v = c.extract(metrics)
                s = c.score(v)
                scores.append(s * c.weight)
                metrics[f"{c.name}_value"] = v
                metrics[f"{c.name}_score"] = s
            metrics["score"] = float(np.mean(scores)) if scores else 0.0
        else:
            metrics["score"] = -error  # default: minimize error

        return metrics

    # -- internals ------------------------------------------------------------

    def _get_test_case(self, **kwargs):
        """Return ``(H_terms, psi0, t)`` from the test generator or kwargs."""
        if "test_case" in kwargs:
            tc = kwargs["test_case"]
            if len(tc) == 3:
                return tc
            return (*tc, self.t)

        if self.test_generator is not None:
            tc = self.test_generator()
            if len(tc) == 3:
                return tc
            return (*tc, self.t)

        raise ValueError(
            "No test case provided. Pass test_generator to __init__ or "
            "test_case=(H_terms, psi0, t) to measure()."
        )

    def _generate_runner_script(self, dynamics_evolution, H_terms, psi0, t) -> Path:
        """Write a temporary Python script that runs the dynamics under Legate.

        The script:
        1. Reads ``(H_terms, psi0, t)`` from an input pickle
        2. Imports cupynumeric and runs the appropriate evolution
        3. Writes ``(psi_final, meta)`` as pickle to stdout
        """
        # Determine which evolution class and parameters to serialize
        cls = type(dynamics_evolution)
        cls_name = cls.__name__
        module = cls.__module__

        # Build init kwargs string
        init_kwargs = self._serialize_init_kwargs(dynamics_evolution)

        script = textwrap.dedent(f"""\
            #!/usr/bin/env python3
            \"\"\"Auto-generated runner for Legate profiling.\"\"\"
            import sys, pickle, time as _time
            import numpy as np

            try:
                import cupynumeric as cn
            except ImportError:
                import numpy as cn

            # Load input
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--pickle", required=True)
            args = parser.parse_args()

            with open(args.pickle, "rb") as f:
                H_terms, psi0, t = pickle.load(f)

            # Convert to cupynumeric arrays
            H_terms = [cn.array(h) for h in H_terms]
            psi0 = cn.array(psi0)

            # Instantiate dynamics
            from {module} import {cls_name}
            dyn = {cls_name}({init_kwargs})

            # Run evolution with timing
            t0 = _time.perf_counter()
            psi_final = dyn.evolve(H_terms, psi0, t)
            elapsed = _time.perf_counter() - t0

            # Convert back to numpy for pickling
            psi_final = np.asarray(psi_final)

            meta = {{"time_ms": elapsed * 1000.0}}

            # Write result to stdout
            sys.stdout.buffer.write(pickle.dumps((psi_final, meta)))
        """)

        # Write to temporary file in workdir
        workdir = self.profiler.workdir
        workdir.mkdir(parents=True, exist_ok=True)
        script_path = workdir / "_legate_runner.py"
        script_path.write_text(script)
        return script_path

    @staticmethod
    def _serialize_init_kwargs(dynamics_evolution) -> str:
        """Extract constructor arguments from a dynamics instance."""
        parts = []

        # ExactEvolution has no params
        if hasattr(dynamics_evolution, "trotter_steps"):
            parts.append(f"trotter_steps={dynamics_evolution.trotter_steps}")
        if hasattr(dynamics_evolution, "formula"):
            formula = dynamics_evolution.formula
            parts.append(f"suzuki_order={formula.order}")
        if hasattr(dynamics_evolution, "n_samples"):
            parts.append(f"n_samples={dynamics_evolution.n_samples}")

        return ", ".join(parts)

    def _compute_error(self, H_terms, psi0, t, psi_final) -> float:
        """Compute approximation error versus exact reference."""
        if self.exact_fn is not None:
            psi_exact = self.exact_fn(H_terms, t, psi0)
        else:
            # Default: scipy expm on CPU
            H_full = sum(np.asarray(h) for h in H_terms)
            psi_exact = la.expm(-1j * H_full * t) @ np.asarray(psi0)

        psi_final = np.asarray(psi_final)
        psi_exact = np.asarray(psi_exact)
        return float(np.linalg.norm(psi_final - psi_exact))


__all__ = ["Metrics"]
