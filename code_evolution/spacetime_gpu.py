# spacetime_gpu.py
# DuckDB -> compute GPU “space–time cost” (= ∫ active_GPUs dt).
# Uses your existing DB file named `legion_prof` (no .duckdb).

import duckdb

con = duckdb.connect("legion_prof")

# Per-GPU metrics + TOTAL space–time (microseconds). “running” is the on-GPU time.
q = r"""
WITH gpu_items AS (
  SELECT
    e.long_name AS gpu,                          -- e.g. "Node 0 GPU 3 Device"
    i.running.start AS start_us,
    i.running.stop  AS stop_us,
    i.running.duration AS dur_us
  FROM items i
  JOIN entries e USING(entry_slug)
  WHERE i.running IS NOT NULL
    AND e.long_name ILIKE '%GPU% Device'         -- DEVICE, not HOST
),
per_gpu AS (
  SELECT
    gpu,
    MIN(start_us) AS t0_us,
    MAX(stop_us)  AS t1_us,
    SUM(dur_us)   AS run_us,                     -- time GPU was busy
    (MAX(stop_us) - MIN(start_us)) AS makespan_us,
    COUNT(*) AS n_items
  FROM gpu_items
  GROUP BY gpu
),
tot AS (
  SELECT
    SUM(run_us)                           AS spacetime_us,     -- Σ over GPUs (GPU·μs)
    MIN(t0_us)                            AS global_t0_us,
    MAX(t1_us)                            AS global_t1_us,
    (MAX(t1_us) - MIN(t0_us))             AS global_makespan_us
  FROM per_gpu
)
SELECT * FROM per_gpu
UNION ALL
SELECT
  'TOTAL'                                  AS gpu,
  global_t0_us                             AS t0_us,
  global_t1_us                             AS t1_us,
  spacetime_us                             AS run_us,           -- == space–time cost
  global_makespan_us                       AS makespan_us,
  NULL                                     AS n_items
FROM tot
ORDER BY gpu;
"""

df = con.sql(q).df()
print(df)

# Optional: human-friendly summary.
if not df.empty:
  total = df[df["gpu"] == "TOTAL"].iloc[0]
  spacetime_s = total["run_us"] / 1e6
  makespan_s  = total["makespan_us"] / 1e6
  avg_conc    = (total["run_us"] / total["makespan_us"]) if total["makespan_us"] else 0.0
  print("\n== Space–time summary ==")
  print(f"GPU space–time cost: {spacetime_s:.3f} GPU·s")
  print(f"Global makespan:     {makespan_s:.3f} s")
  print(f"Avg GPU concurrency: {avg_conc:.3f} GPUs")