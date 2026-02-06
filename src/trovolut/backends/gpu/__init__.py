"""
GPU backend for trovolut -- real GPU execution via cupynumeric + Legate.

This backend mirrors the CPU backend API but executes all linear-algebra
operations on NVIDIA GPUs via **cupynumeric**.  Programs are launched via
the **Legate** runtime, and profiling is done with **legion_prof**.

Modules
-------
dynamics
    ``ExactEvolution``, ``TrotterEvolution``, ``QDriftEvolution`` --
    same API as the CPU backend, running on GPU arrays.
measure
    ``Metrics`` -- runs dynamics under Legate with ``--profile`` and
    returns real GPU metrics (kernel time, VRAM, concurrency).
profiler
    ``LegateProfiler`` -- low-level profiling infrastructure
    (run -> load profile -> query DuckDB).
constraints
    ``LegateGPUSeconds``, ``LegateFramebuffer``, ``LegateMakespan``,
    ``LegateConcurrency`` -- constraint classes for the Legate profiler
    metrics.
utils
    ``to_gpu``, ``to_cpu``, ``is_gpu_array`` -- array conversion helpers.

Requirements
------------
- cupynumeric (``pip install cupynumeric``)
- legate runtime (for GPU execution and profiling)
- legion_prof CLI (for profile analysis)
- duckdb (for profile queries)

Example::

    import trovolut as trv
    trv.set_backend("gpu")

    from trovolut.dynamics import TrotterEvolution
    from trovolut.measure import Metrics
    from trovolut.constraints import LegateGPUSeconds

    dyn = TrotterEvolution(trotter_steps=100)
    metrics = Metrics(
        constraints=[LegateGPUSeconds(max_gpu_seconds=5.0)],
        test_generator=my_test_gen,
        gpus=4,
    )
    result = metrics(dyn)
"""

import sys
from typing import Optional

# Availability check
_available: Optional[bool] = None
_missing_packages: list[str] = []


def is_available() -> bool:
    """Check if GPU backend dependencies are available.

    The GPU backend requires only ``cupynumeric`` as its core dependency.
    Optional packages (``duckdb``, ``legate``) are needed for profiling
    but are not required for basic dynamics execution.
    """
    global _available, _missing_packages

    if _available is not None:
        return _available

    # Check platform -- Legate currently requires Linux
    if sys.platform != "linux":
        _available = False
        _missing_packages = [f"requires Linux (current: {sys.platform})"]
        return False

    # Only cupynumeric is essential for dynamics
    _missing_packages = []
    try:
        __import__("cupynumeric")
    except ImportError:
        _missing_packages.append("cupynumeric")

    _available = len(_missing_packages) == 0
    return _available


__all__ = ["is_available"]
