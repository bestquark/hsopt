from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import sqlite3
import numpy as np

from trovolut.eval.base import BaseEval


# ---------- Profiling readers ----------
def _spacetime_gpu_duckdb(path: Path = Path("legion_prof"), verbose: bool = False) -> Dict[str, Any]:
    con = duckdb.connect(str(path))

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
        mem_common
        + r"""
      , span AS ( SELECT MIN(start_us) AS t0, MAX(stop_us) AS t1 FROM with_bytes )
      SELECT COALESCE(SUM(bytes * dur_us) / NULLIF((SELECT t1 - t0 FROM span),0), 0) AS avg_bytes
      FROM with_bytes;
    """
    ).fetchone()[0]

    peak_bytes = con.sql(
        mem_common
        + r"""
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
        # No GPU work recorded
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


def _spacetime_gpu_sqlite3(db_path: Path) -> Dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()]

        for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
            if name in tables:
                kernel_table = name
                break
        else:
            # No kernel table -> no GPU activity
            return {
                "gpu_seconds": 0.0,
                "avg_framebuffer_bytes": 0.0,
                "peak_framebuffer_bytes": 0.0,
            }

        cur.execute(f"""
            SELECT deviceId, SUM(end - start)/1e9 AS gpu_seconds
            FROM {kernel_table}
            GROUP BY deviceId
        """)
        time_rows = cur.fetchall()
        total_time = float(sum(t for _, t in time_rows))

        mem_bytes = {}
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            cur.execute("""
                SELECT deviceId, SUM(bytes) AS total_bytes
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                GROUP BY deviceId
            """)
            mem_bytes = dict(cur.fetchall())
        elif "CUPTI_ACTIVITY_KIND_MEMORY" in tables:
            cur.execute("""
                SELECT deviceId, SUM(bytes) AS total_bytes
                FROM CUPTI_ACTIVITY_KIND_MEMORY
                GROUP BY deviceId
            """)
            mem_bytes = dict(cur.fetchall())

    # SQLite path has traffic, not FB residency. Use traffic as a loose proxy.
    total_traffic = float(sum(mem_bytes.values())) if mem_bytes else 0.0
    return {
        "gpu_seconds": total_time,
        "avg_framebuffer_bytes": 0.0,
        "peak_framebuffer_bytes": 0.0,
        "total_memory_bytes": total_traffic,
    }


# ---------- Evaluator using profiler DB ----------
class GPUSpaceTimeEval(BaseEval):
    """
    Score from profiler DB (DuckDB or SQLite):
      score = - ( w_time * gpu_seconds
                + w_peak * peak_FB_MiB
                + w_avg  * avg_FB_MiB
                + w_err  * err_mean_from_BaseEval )
    It runs trials via BaseEval, then replaces/augments the final score using the DB.
    """
    def __init__(
        self,
        ham_or_ensemble,
        program: Optional[str] = None,
        trials: int = 5,
        t: float = 0.05,
        seeds: Optional[List[int]] = None,
        psi0: Optional[np.ndarray] = None,
        *,
        db_type: str = "duckdb",                 # "duckdb" or "sqlite"
        db_path: Path = Path("legion_prof"),     # DuckDB DB file or SQLite .sqlite
        w_time: float = 1.0,
        w_peak: float = 0.02,
        w_avg: float  = 0.01,
        w_err: float  = 100.0,
    ):
        super().__init__(ham_or_ensemble, program, trials, t, seeds, psi0)
        self.db_type = db_type
        self.db_path = Path(db_path)
        self.w_time = float(w_time)
        self.w_peak = float(w_peak)
        self.w_avg  = float(w_avg)
        self.w_err  = float(w_err)

    def _score(self, rows: List[Dict[str, float]]) -> float:
        """
        BaseEval will call this, but we only need rows to get the mean error.
        The profiler metrics are read in `evaluate` to build the final score.
        """
        e = np.array([r["error"] for r in rows], dtype=float)
        return float(-self.w_err * float(np.mean(e)))

    def evaluate(self, program: Optional[str] = None) -> Dict[str, Any]:
        # 1) Let BaseEval run trials (time/error/mem from candidate meta)
        res = super().evaluate(program)

        # 2) Read profiler DB for GPU seconds + framebuffer residency
        if self.db_type.lower() == "duckdb":
            prof = _spacetime_gpu_duckdb(self.db_path, verbose=False)
            gpu_seconds = float(prof.get("gpu_seconds", 0.0))
            peak_fb_mb  = float(prof.get("peak_framebuffer_bytes", 0.0)) / (1024.0**2)
            avg_fb_mb   = float(prof.get("avg_framebuffer_bytes", 0.0))  / (1024.0**2)
        else:
            prof = _spacetime_gpu_sqlite3(self.db_path)
            gpu_seconds = float(prof.get("gpu_seconds", 0.0))
            # SQLite path lacks residency; leave zeros, or you can map traffic → proxy
            peak_fb_mb  = float(prof.get("peak_framebuffer_bytes", 0.0)) / (1024.0**2)
            avg_fb_mb   = float(prof.get("avg_framebuffer_bytes", 0.0))  / (1024.0**2)

        err_mean = float(res.get("err_mean", 0.0))

        # 3) Recompute score from profiler metrics (+ error penalty)
        score = -(
            self.w_time * gpu_seconds
            + self.w_peak * peak_fb_mb
            + self.w_avg  * avg_fb_mb
            + self.w_err  * err_mean
        )

        # 4) Return enriched result
        out = dict(res)
        out.update({
            "score": float(score),
            "gpu_seconds": float(gpu_seconds),
            "peak_fb_mb": float(peak_fb_mb),
            "avg_fb_mb": float(avg_fb_mb),
            "prof_db_type": self.db_type,
            "prof_db_path": str(self.db_path),
        })
        return out