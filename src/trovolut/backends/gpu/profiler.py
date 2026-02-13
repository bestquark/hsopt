"""
Legate GPU profiling infrastructure.

Runs programs via ``legate --profile``, loads profiles with ``legion_prof``,
and queries GPU metrics from the resulting DuckDB database.

Pipeline::

    1. Run program:  uv run legate --profile --gpus N program.py
    2. Load profile:  legion_prof load legate_0.prof profile.duckdb
    3. Query metrics: SELECT gpu_seconds, peak_fb_mb, ... FROM DuckDB

Metrics returned:

- ``gpu_seconds``          -- total GPU kernel time (sum across devices)
- ``makespan_seconds``     -- wall-clock time from first to last GPU activity
- ``avg_concurrency``      -- average number of GPUs active concurrently
- ``peak_fb_mb``           -- peak framebuffer (VRAM) usage in MiB
- ``avg_fb_mb``            -- time-weighted average framebuffer usage in MiB

Requirements:

- ``legion_prof`` CLI in PATH (install via ``scripts/config_profiler.sh``)
- ``duckdb`` Python package for profile analysis
- ``legate`` runtime for GPU execution
"""

from __future__ import annotations

import io
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


# ============================================================
# Data classes
# ============================================================

@dataclass
class LegateResult:
    """Result from a single Legate profiled run."""
    psi_final: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    prof_dir: str = ""


@dataclass
class GPUMetrics:
    """GPU metrics extracted from a legion_prof profile.

    Optional NVML fields are populated when ``enable_nvml=True`` is passed
    to :meth:`LegateProfiler.run`.  They default to ``None`` and are
    excluded from :meth:`to_dict` when absent.
    """
    gpu_seconds: float = 0.0
    makespan_seconds: float = 0.0
    avg_concurrency: float = 0.0
    peak_fb_mb: float = 0.0
    avg_fb_mb: float = 0.0

    # Optional NVML hardware counter fields
    nvml_peak_memory_bytes: int | None = None
    nvml_avg_memory_bytes: float | None = None
    nvml_peak_power_watts: float | None = None
    nvml_avg_power_watts: float | None = None
    nvml_avg_gpu_utilization: float | None = None
    nvml_avg_memory_utilization: float | None = None
    nvml_energy_joules: float | None = None
    nvml_max_temperature_c: int | None = None
    nvml_n_samples: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "gpu_seconds": self.gpu_seconds,
            "makespan_seconds": self.makespan_seconds,
            "avg_concurrency": self.avg_concurrency,
            "peak_fb_mb": self.peak_fb_mb,
            "avg_fb_mb": self.avg_fb_mb,
        }
        # Conditionally include NVML fields
        for attr in (
            "nvml_peak_memory_bytes", "nvml_avg_memory_bytes",
            "nvml_peak_power_watts", "nvml_avg_power_watts",
            "nvml_avg_gpu_utilization", "nvml_avg_memory_utilization",
            "nvml_energy_joules", "nvml_max_temperature_c",
            "nvml_n_samples",
        ):
            val = getattr(self, attr)
            if val is not None:
                d[attr] = val
        return d


# ============================================================
# LegateProfiler
# ============================================================

class LegateProfiler:
    """Runs programs via Legate with ``--profile`` and extracts GPU metrics.

    Parameters
    ----------
    gpus : int
        Number of GPUs to allocate.
    cpus : int
        Number of CPU cores to allocate.
    fbmem_mb : int
        Framebuffer memory per GPU in MiB.
    workdir : Path
        Working directory for profile artifacts.

    Example::

        profiler = LegateProfiler(gpus=4)
        result = profiler.run("my_program.py", (H_terms, psi0, t))
        metrics = profiler.profile(result.prof_dir)
    """

    def __init__(
        self,
        gpus: int = 1,
        cpus: int = 4,
        fbmem_mb: int = 256,
        workdir: Path | str = ".",
    ) -> None:
        self.gpus = int(gpus)
        self.cpus = int(cpus)
        self.fbmem_mb = int(fbmem_mb)
        self.workdir = Path(workdir)

    # -- run program via Legate -------------------------------------------

    def run(
        self,
        program_path: str,
        input_data: Tuple[Sequence[np.ndarray], np.ndarray, float],
        *,
        env: Dict[str, str] | None = None,
        enable_nvml: bool = False,
    ) -> LegateResult:
        """Run a program via ``uv run legate --profile`` with pickle I/O.

        Parameters
        ----------
        program_path : str
            Path to the Python script to execute under Legate.
        input_data : tuple
            ``(H_terms, psi0, t)`` -- Hamiltonian terms, initial state, time.
        env : dict, optional
            Extra environment variables.
        enable_nvml : bool
            If True, wrap the subprocess call with an :class:`NVMLMonitor`
            to collect real-time hardware counters (VRAM, power, utilization).

        Returns
        -------
        LegateResult
            Contains ``psi_final``, ``meta``, and ``prof_dir``.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)

        # 1. Write input pickle
        in_pkl = self.workdir / ".legate_in.pkl"
        H_terms, psi0, t = input_data
        with open(in_pkl, "wb") as f:
            pickle.dump((list(H_terms), np.asarray(psi0), float(t)), f)

        # 2. Clean previous profile directories
        for p in self.workdir.glob("legate_*.prof"):
            shutil.rmtree(p, ignore_errors=True)

        # 3. Launch Legate with profiling (optionally with NVML monitoring)
        cmd = [
            "uv", "run", "legate", "--profile",
            "--gpus", str(self.gpus),
            "--cpus", str(self.cpus),
            "--fbmem", str(self.fbmem_mb),
            program_path,
            "--pickle", str(in_pkl),
        ]
        envp = os.environ.copy()
        if env:
            envp.update(env)

        nvml_metrics = None
        if enable_nvml:
            from trovolut.backends.gpu.hardware_counters import NVMLMonitor
            with NVMLMonitor() as mon:
                proc = subprocess.run(cmd, cwd=str(self.workdir), env=envp, capture_output=True)
            nvml_metrics = mon.metrics()
        else:
            proc = subprocess.run(cmd, cwd=str(self.workdir), env=envp, capture_output=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Legate run failed (exit {proc.returncode}).\n"
                f"STDOUT:\n{proc.stdout.decode(errors='ignore')[-2000:]}\n"
                f"STDERR:\n{proc.stderr.decode(errors='ignore')[-4000:]}"
            )

        # 4. Parse output pickle
        psi_final, meta = _parse_pickle_output(proc.stdout)

        # 5. Merge NVML metrics into meta
        if nvml_metrics is not None:
            meta.update(nvml_metrics.to_dict())

        # 6. Find newest profile directory
        prof_dirs = sorted(
            self.workdir.glob("legate_*.prof"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        prof_dir = str(prof_dirs[0]) if prof_dirs else ""

        return LegateResult(psi_final=psi_final, meta=meta or {}, prof_dir=prof_dir)

    # -- profile loading and querying -------------------------------------

    def profile(self, prof_dir: str, db_path: Path | None = None) -> GPUMetrics:
        """Load a Legate profile and extract GPU metrics.

        Parameters
        ----------
        prof_dir : str
            Path to the ``legate_*.prof`` directory.
        db_path : Path, optional
            Path for the DuckDB file.  Defaults to ``workdir/legate_profile.duckdb``.

        Returns
        -------
        GPUMetrics
        """
        if not prof_dir:
            return GPUMetrics()

        db_path = db_path or (self.workdir / "legate_profile.duckdb")
        load_profile(prof_dir, db_path)
        return query_gpu_metrics(db_path)


# ============================================================
# Standalone functions
# ============================================================

def load_profile(prof_dir: str, db_path: Path) -> None:
    """Load a Legate profile into DuckDB via ``legion_prof load``.

    Parameters
    ----------
    prof_dir : str
        Path to the ``legate_*.prof`` directory.
    db_path : Path
        Destination DuckDB database file.

    Raises
    ------
    RuntimeError
        If ``legion_prof load`` fails.
    """
    db_path = Path(db_path)
    db_path.unlink(missing_ok=True)
    try:
        subprocess.run(
            ["legion_prof", "load", str(prof_dir), str(db_path)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "legion_prof not found. Install it via scripts/config_profiler.sh"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"legion_prof load failed: {e.stderr.decode(errors='ignore')[-2000:]}"
        )


def query_gpu_metrics(db_path: Path) -> GPUMetrics:
    """Query GPU metrics from a DuckDB profile database.

    Parameters
    ----------
    db_path : Path
        Path to DuckDB database created by ``legion_prof load``.

    Returns
    -------
    GPUMetrics
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb is required for GPU profiling: pip install duckdb")

    con = duckdb.connect(str(db_path))

    # -- GPU time ---------------------------------------------------------
    time_q = r"""
    WITH gpu_items AS (
      SELECT
        e.long_name AS gpu,
        i.running.start    AS start_us,
        i.running.stop     AS stop_us,
        i.running.duration AS dur_us
      FROM items i
      JOIN entries e USING(entry_slug)
      WHERE i.running IS NOT NULL AND e.long_name ILIKE '%GPU% Device'
    ),
    per_gpu AS (
      SELECT
        gpu,
        MIN(start_us) AS t0_us,
        MAX(stop_us)  AS t1_us,
        SUM(dur_us)   AS run_us
      FROM gpu_items
      GROUP BY gpu
    )
    SELECT
      COALESCE(SUM(run_us), 0)                    AS spacetime_us,
      COALESCE(MAX(t1_us) - MIN(t0_us), 0)        AS makespan_us
    FROM per_gpu;
    """
    row = con.sql(time_q).fetchone()
    spacetime_us = float(row[0]) if row else 0.0
    makespan_us = float(row[1]) if row else 0.0

    gpu_seconds = spacetime_us / 1e6
    makespan_seconds = makespan_us / 1e6
    avg_concurrency = (gpu_seconds / makespan_seconds) if makespan_seconds > 0 else 0.0

    # -- Framebuffer memory -----------------------------------------------
    mem_common = r"""
    WITH fb_instances AS (
      SELECT
        e.entry_slug, e.long_name, i.title, i.size,
        i.lifetime.start AS start_us,
        i.lifetime.stop  AS stop_us,
        i.lifetime.duration AS dur_us
      FROM items i
      JOIN entries e USING(entry_slug)
      WHERE i.lifetime IS NOT NULL
        AND (e.long_name ILIKE '%Framebuffer%' OR e.entry_slug LIKE 'n0_framebuffer%')
        AND i.size IS NOT NULL
    ),
    parsed AS (
      SELECT
        entry_slug, long_name, title, start_us, stop_us, dur_us,
        CAST(REGEXP_EXTRACT(size, '([0-9.]+)') AS DOUBLE) AS v,
        CASE
          WHEN size ILIKE '%KiB%' THEN 1024.0
          WHEN size ILIKE '%MiB%' THEN 1024.0*1024.0
          WHEN size ILIKE '%GiB%' THEN 1024.0*1024.0*1024.0
          WHEN size ILIKE '%KB%'  THEN 1000.0
          WHEN size ILIKE '%MB%'  THEN 1000.0*1000.0
          WHEN size ILIKE '%GB%'  THEN 1000.0*1000.0*1000.0
          WHEN size ILIKE '%B%'   THEN 1.0
          ELSE 1.0
        END AS mult
      FROM fb_instances
    ),
    with_bytes AS (
      SELECT *, v * mult AS bytes FROM parsed WHERE v IS NOT NULL
    )
    """

    try:
        avg_bytes = con.sql(
            mem_common + r"""
            , span AS ( SELECT MIN(start_us) AS t0, MAX(stop_us) AS t1 FROM with_bytes )
            SELECT COALESCE(
              SUM(bytes * dur_us) / NULLIF((SELECT t1 - t0 FROM span), 0), 0
            ) AS avg_bytes
            FROM with_bytes;
            """
        ).fetchone()[0]
    except Exception:
        avg_bytes = 0.0

    try:
        peak_bytes = con.sql(
            mem_common + r"""
            , events AS (
                SELECT start_us AS t,  bytes AS delta FROM with_bytes
                UNION ALL
                SELECT stop_us  AS t, -bytes AS delta FROM with_bytes
              ),
              running AS (
                SELECT t, SUM(delta) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING) AS live_bytes
                FROM events
              )
            SELECT COALESCE(MAX(live_bytes), 0) AS peak_bytes FROM running;
            """
        ).fetchone()[0]
    except Exception:
        peak_bytes = 0.0

    con.close()

    return GPUMetrics(
        gpu_seconds=gpu_seconds,
        makespan_seconds=makespan_seconds,
        avg_concurrency=avg_concurrency,
        peak_fb_mb=float(peak_bytes or 0.0) / (1024.0 ** 2),
        avg_fb_mb=float(avg_bytes or 0.0) / (1024.0 ** 2),
    )


# ============================================================
# Internal helpers
# ============================================================

def _parse_pickle_output(stdout_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Parse ``(psi_final, meta)`` from Legate's stdout pickle stream."""
    try:
        psi_final, meta = pickle.loads(stdout_bytes)
        return np.asarray(psi_final), (meta or {})
    except Exception:
        pass

    # Robust fallback: scan for last valid pickle payload
    obj = None
    bio = io.BytesIO(stdout_bytes)
    up = pickle.Unpickler(bio)
    try:
        while True:
            obj = up.load()
    except Exception:
        pass

    if isinstance(obj, tuple) and len(obj) == 2:
        return np.asarray(obj[0]), (obj[1] or {})

    raise RuntimeError("Could not parse pickled (psi_final, meta) from Legate stdout.")


# ============================================================
# Exports
# ============================================================

__all__ = [
    "LegateProfiler",
    "LegateResult",
    "GPUMetrics",
    "load_profile",
    "query_gpu_metrics",
]
