# trovolut/eval/gpu_evals.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import os, sys, io, pickle, glob, time, shutil, subprocess
import numpy as np

from trovolut.eval.base import BaseEval

# Lazy imports for heavy dependencies
if TYPE_CHECKING:
    import duckdb
    import sqlite3
    from trovolut.qm.hamiltonians._hamiltonian import Hamiltonian, HamiltonianEnsemble

# --------------------------
# DuckDB reader for legate_prof
# --------------------------
def _spacetime_gpu_duckdb(db_path: Path) -> Dict[str, Any]:
    # db_path should be a DuckDB DB *file* produced by `legion_prof load`.
    import duckdb  # lazy import - heavy dependency
    con = duckdb.connect(str(db_path))

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
        SUM(dur_us)   AS run_us,
        (MAX(stop_us) - MIN(start_us)) AS makespan_us,
        COUNT(*) AS n_items
      FROM gpu_items
      GROUP BY gpu
    ),
    tot AS (
      SELECT
        SUM(run_us) AS spacetime_us,
        MIN(t0_us)  AS global_t0_us,
        MAX(t1_us)  AS global_t1_us,
        (MAX(t1_us) - MIN(t0_us)) AS global_makespan_us
      FROM per_gpu
    )
    SELECT * FROM per_gpu
    UNION ALL
    SELECT
      'TOTAL' AS gpu,
      global_t0_us AS t0_us,
      global_t1_us AS t1_us,
      spacetime_us AS run_us,
      global_makespan_us AS makespan_us,
      NULL AS n_items
    FROM tot
    ORDER BY gpu;
    """
    df_time = con.sql(time_q).df()

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

    avg_bytes = con.sql(
        mem_common + r"""
      , span AS ( SELECT MIN(start_us) AS t0, MAX(stop_us) AS t1 FROM with_bytes )
      SELECT COALESCE(SUM(bytes * dur_us) / NULLIF((SELECT t1 - t0 FROM span),0), 0) AS avg_bytes
      FROM with_bytes;
    """
    ).fetchone()[0]

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

    if df_time.empty:
        return {
            "gpu_seconds": 0.0,
            "makespan_seconds": 0.0,
            "avg_concurrency": 0.0,
            "avg_framebuffer_bytes": float(avg_bytes or 0.0),
            "peak_framebuffer_bytes": float(peak_bytes or 0.0),
            "prof_df": df_time,
        }

    total = df_time[df_time["gpu"] == "TOTAL"].iloc[0]
    spacetime_s = float(total["run_us"]) / 1e6
    makespan_s = float(total["makespan_us"] or 0.0) / 1e6
    avg_conc = (spacetime_s / makespan_s) if makespan_s else 0.0
    return {
        "prof_df": df_time,
        "gpu_seconds": spacetime_s,
        "makespan_seconds": makespan_s,
        "avg_concurrency": avg_conc,
        "avg_framebuffer_bytes": float(avg_bytes or 0.0),
        "peak_framebuffer_bytes": float(peak_bytes or 0.0),
    }

# --------------------------
# Helper: run candidate under Legate with --profile and pickle IO
# --------------------------
def _run_candidate_with_legate(
    program: str,
    H_terms: Sequence[np.ndarray],
    psi0: np.ndarray,
    t: float,
    *,
    gpus: int = 1,
    cpus: int = 4,
    fbmem_mb: int = 256,
    workdir: Path | None = None,
    env: Dict[str, str] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    Runs `program` by calling:
      uv run legate --profile --gpus <gpus> --cpus <cpus> --fbmem <fbmem_mb> program --pickle <tmp.pkl>
    Expects program to read the pickle and write (psi_final, meta) as pickle to stdout.

    Returns:
        psi_final, meta, latest_prof_dir
    """
    workdir = Path.cwd() if workdir is None else Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare input pickle
    in_pkl = workdir / ".legate_in.pkl"
    with open(in_pkl, "wb") as f:
        pickle.dump((H_terms, psi0, float(t)), f)

    # 2) Clean previous profile dirs and DBs to avoid mixing runs
    for p in workdir.glob("legate_*.prof"):
        shutil.rmtree(p, ignore_errors=True)

    # 3) Launch legate with profiling
    cmd = [
        "uv", "run", "legate", "--profile",
        "--gpus", str(gpus),
        "--cpus", str(cpus),
        "--fbmem", str(fbmem_mb),
        program,
        "--pickle", str(in_pkl),
    ]
    envp = os.environ.copy()
    if env:
        envp.update(env)

    proc = subprocess.run(
        cmd, cwd=str(workdir), env=envp,
        capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Legate run failed ({proc.returncode}).\nSTDOUT:\n{proc.stdout.decode(errors='ignore')[-2000:]}\n"
            f"STDERR:\n{proc.stderr.decode(errors='ignore')[-4000:]}"
        )

    # 4) Read result from stdout (pickle stream)
    #    Some environments may emit extra logging; try to locate last pickle tuple.
    out_bytes = proc.stdout
    try:
        psi_final, meta = pickle.loads(out_bytes)
    except Exception:
        # Robust parsing: scan for last pickle payload
        obj = None
        bio = io.BytesIO(out_bytes)
        up = pickle.Unpickler(bio)
        try:
            while True:
                obj = up.load()
        except Exception:
            pass
        if not isinstance(obj, tuple) or len(obj) != 2:
            raise RuntimeError("Could not parse pickled (psi_final, meta) from candidate stdout.")
        psi_final, meta = obj

    # 5) Find the newest profile directory
    prof_dirs = sorted(workdir.glob("legate_*.prof"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_prof = str(prof_dirs[0]) if prof_dirs else ""

    return psi_final, (meta or {}), latest_prof

# --------------------------
# Evaluator that runs Legate+profile and scores via DuckDB
# --------------------------
class GPUSpaceTimeEval(BaseEval):
    """
    Runs candidate under Legate with --profile, ingests the profile with `legion_prof load`
    into a DuckDB DB, then scores using GPU space–time and framebuffer residency:

      score = - ( w_time * gpu_seconds
                + w_peak * peak_FB_MiB
                + w_avg  * avg_FB_MiB
                + w_err  * err_mean )

    Requirements:
      - `program` must accept `--pickle <file>` and write `(psi_final, meta)` via pickle to stdout.
      - `legion_prof` CLI must be available in PATH (part of Legate profiling toolchain).
    """
    def __init__(
        self,
        ham_or_ensemble: Hamiltonian | HamiltonianEnsemble,
        program: Optional[str] = None,
        *,
        trials: int = 3,
        t: float = 0.05,
        seeds: Optional[List[int]] = None,
        psi0: Optional[np.ndarray] = None,
        gpus: int = 1,
        cpus: int = 4,
        fbmem_mb: int = 256,
        legion_db: Path = Path("legate_profile.duckdb"),
        w_time: float = 1.0,
        w_peak: float = 0.02,
        w_avg: float  = 0.01,
        w_err: float  = 100.0,
        workdir: Path = Path("."),
    ):
        super().__init__(ham_or_ensemble, trials, t, seeds, psi0)
        self.gpus = int(gpus)
        self.cpus = int(cpus)
        self.fbmem_mb = int(fbmem_mb)
        self.legion_db = Path(legion_db)
        self.w_time = float(w_time)
        self.w_peak = float(w_peak)
        self.w_avg  = float(w_avg)
        self.w_err  = float(w_err)
        self.workdir = Path(workdir)

    # override: run trials using legate --profile
    def _run_one(self, program: str, H_terms, psi0, t, seed: int) -> Dict[str, float]:
        # Let candidate run under Legate with profile
        psi_final, meta, prof_dir = _run_candidate_with_legate(
            program, H_terms, psi0, t,
            gpus=self.gpus, cpus=self.cpus, fbmem_mb=self.fbmem_mb, workdir=self.workdir
        )

        # reference solution on CPU (numpy/scipy), same as BaseEval
        from scipy.linalg import expm
        psi_exact = expm(-1j * t * sum(H_terms)) @ psi0

        err = float(np.linalg.norm(psi_final - psi_exact))
        norm_val = float(np.linalg.norm(psi_final))
        # keep per-trial “local” stats; DB aggregation happens in evaluate()
        return {
            "error": err,
            "norm": norm_val,
            "time_ms": float(meta.get("time_ms", 0.0)),     # optional from candidate
            "mem_mb": float(meta.get("mem_mb", 0.0)),       # optional from candidate
        }

    def evaluate(self, program: Optional[str] = None) -> Dict[str, Any]:
        program = program
        if not program or not os.path.exists(program):
            return {"score": -1e9, "error": "program_path_invalid"}

        # Run trials (each trial will profile and overwrite/create a new legate_*.prof)
        rows: List[Dict[str, float]] = []
        prof_dirs: List[str] = []
        seeds = self._seeds()
        for sd in seeds:
            H_terms, psi0 = self._problem_for_seed(sd)
            rec = self._run_one(program, H_terms, psi0, self.t, sd)
            rows.append(rec)
            # remember the newest prof dir after each trial
            newest = sorted(self.workdir.glob("legate_*.prof"), key=lambda p: p.stat().st_mtime, reverse=True)
            if newest:
                prof_dirs.append(str(newest[0]))

        # Aggregate basic stats
        errs = np.array([r["error"] for r in rows], dtype=float) if rows else np.array([1e9])
        err_mean = float(np.mean(errs))
        err_min  = float(np.min(errs))
        norm_dev = float(abs(np.mean([r["norm"] for r in rows]) - 1.0)) if rows else 1.0

        # Ingest the LAST profile into DuckDB (you can also merge all if you prefer)
        self.legion_db.unlink(missing_ok=True)
        last_prof = sorted(prof_dirs, key=lambda p: Path(p).stat().st_mtime, reverse=True)[0] if prof_dirs else ""
        if last_prof:
            # legion_prof load <prof_dir> <duckdb_file>
            try:
                subprocess.run(
                    ["legion_prof", "load", last_prof, str(self.legion_db)],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError as e:
                # If load fails, we still return results without GPU metrics
                load_err = e.stderr.decode(errors="ignore")[-2000:]
                gpu_seconds = 0.0; peak_mb = 0.0; avg_mb = 0.0
                score = -(self.w_err * err_mean)
                return {
                    "score": float(score),
                    "err_mean": err_mean,
                    "err_min": err_min,
                    "norm_dev": norm_dev,
                    "gpu_seconds": gpu_seconds,
                    "peak_fb_mb": peak_mb,
                    "avg_fb_mb": avg_mb,
                    "prof_load_error": load_err,
                    "prof_dir": last_prof,
                    "prof_db": str(self.legion_db),
                    "trials": len(rows),
                }

            # Query the DB
            prof = _spacetime_gpu_duckdb(self.legion_db)
            gpu_seconds = float(prof.get("gpu_seconds", 0.0))
            peak_mb  = float(prof.get("peak_framebuffer_bytes", 0.0)) / (1024.0**2)
            avg_mb   = float(prof.get("avg_framebuffer_bytes", 0.0))  / (1024.0**2)
        else:
            gpu_seconds = 0.0; peak_mb = 0.0; avg_mb = 0.0

        # Final score
        score = -(
            self.w_time * gpu_seconds
            + self.w_peak * peak_mb
            + self.w_avg  * avg_mb
            + self.w_err  * err_mean
        )

        return {
            "score": float(score),
            "err_mean": err_mean,
            "err_min": err_min,
            "norm_dev": norm_dev,
            "gpu_seconds": float(gpu_seconds),
            "peak_fb_mb": float(peak_mb),
            "avg_fb_mb": float(avg_mb),
            "prof_dir": last_prof,
            "prof_db": str(self.legion_db),
            "trials": int(len(rows)),
        }