from pathlib import Path

import duckdb
import sqlite3


def spacetime_gpu_duckdb(path: Path = Path("legion_prof"), verbose: bool = True):
    con = duckdb.connect(str(path))

    # --- Time / space–time (GPU device only) ---
    time_q = r"""
    WITH gpu_items AS (
      SELECT
        e.long_name AS gpu,
        i.running.start   AS start_us,
        i.running.stop    AS stop_us,
        i.running.duration AS dur_us
      FROM items i
      JOIN entries e USING(entry_slug)
      WHERE i.running IS NOT NULL
        AND e.long_name ILIKE '%GPU% Device'
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

    # --- Memory (Framebuffer instances → bytes over time) ---
    mem_common_cte = r"""
      WITH fb_instances AS (
        SELECT
          e.entry_slug,
          e.long_name,
          i.title,
          i.size,
          i.lifetime.start   AS start_us,
          i.lifetime.stop    AS stop_us,
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
        mem_common_cte
        + r"""
      , span AS ( SELECT MIN(start_us) AS t0, MAX(stop_us) AS t1 FROM with_bytes )
      SELECT COALESCE(SUM(bytes * dur_us) / NULLIF((SELECT t1 - t0 FROM span),0), 0) AS avg_bytes
      FROM with_bytes;
    """
    ).fetchone()[0]

    peak_bytes = con.sql(
        mem_common_cte
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

    if verbose and not df_time.empty:
        total = df_time[df_time["gpu"] == "TOTAL"].iloc[0]
        spacetime_s = total["run_us"] / 1e6
        makespan_s = (total["makespan_us"] or 0) / 1e6
        avg_conc = (spacetime_s / makespan_s) if makespan_s else 0.0
        print(df_time)
        print("\n== Space–time & Memory ==")
        print(f"GPU space–time:   {spacetime_s:.3f} GPU·s")
        print(f"Makespan:         {makespan_s:.3f} s")
        print(f"Avg concurrency:  {avg_conc:.3f} GPUs")
        print(f"Avg FB footprint: {avg_bytes / 1024**2:.3f} MiB")
        print(f"Peak FB footprint:{peak_bytes / 1024**2:.3f} MiB")

    return {
        "prof_df": df_time,
        "gpu_seconds": spacetime_s,
        "makespan_seconds": makespan_s,
        "avg_concurrency": avg_conc,
        "avg_framebuffer_bytes": avg_bytes,
        "peak_framebuffer_bytes": peak_bytes,
    }


def spacetime_gpu_sqlite3(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # --- Find which kernel table exists ---
        tables = [
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        ]

        for name in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"]:
            if name in tables:
                kernel_table = name
                break
        else:
            raise RuntimeError("No CUDA kernel table found in prof.sqlite")

        # --- Compute total GPU seconds ---
        cur.execute(f"""
            SELECT deviceId, SUM(end - start)/1e9 AS gpu_seconds
            FROM {kernel_table}
            GROUP BY deviceId
        """)
        time_rows = cur.fetchall()
        total_time = sum(t for _, t in time_rows)

        # --- Compute memory traffic (bytes moved) if available ---
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

    return total_time, time_rows, mem_bytes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute GPU time and memory metrics from prof.sqlite"
    )
    parser.add_argument(
        "--db-type",
        type=str,
        choices=["sqlite", "duckdb"],
        default="duckdb",
        help="Type of database to use for analysis (default: duckdb)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("legion_prof"),
        help="Path to the profiling database file (default: legion_prof)",
    )

    args = parser.parse_args()
    if args.db_type == "duckdb":
        spacetime_gpu_duckdb(args.path, verbose=True)
    else:
        total_time, time_rows, mem_bytes = spacetime_gpu_sqlite3(args.path)

        print(f"Total GPU time: {total_time:.6f} seconds")

        total_bytes = sum(mem_bytes.values())
        print(f"Total memory moved: {total_bytes / 1e9:.6f} GB\n")

        print("Per-GPU breakdown:")
        for dev, gpu_seconds in time_rows:
            mem_gb = mem_bytes.get(dev, 0) / 1e9
            print(f"  > GPU {dev}: {gpu_seconds:.6f} s, {mem_gb:.6f} GB moved")
