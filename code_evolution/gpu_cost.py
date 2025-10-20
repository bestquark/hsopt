import sqlite3
from pathlib import Path

# Path to your Nsight Systems export
db_path = Path("prof.sqlite")

with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()

    # detect which table exists
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()]

    for name in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"]:
        if name in tables:
            table = name
            break
    else:
        raise RuntimeError("No CUDA kernel table found in prof.sqlite")

    # query per-GPU and total time
    cur.execute(f"""
        SELECT deviceId, SUM(end - start)/1e9 AS gpu_seconds
        FROM {table}
        GROUP BY deviceId
    """)
    rows = cur.fetchall()

    total = sum(t for _, t in rows)
    for dev, t in rows:
        print(f"GPU {dev}: {t:.6f} s")
    print(f"TOTAL GPU-seconds: {total:.6f}")