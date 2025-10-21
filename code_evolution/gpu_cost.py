import sqlite3
from pathlib import Path

def compute_gpu_metrics(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # --- Find which kernel table exists ---
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()]

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
    db_path = Path("prof.sqlite")
    total_time, time_rows, mem_bytes = compute_gpu_metrics(db_path)

    print(f"Total GPU time: {total_time:.6f} seconds")
   
    total_bytes = sum(mem_bytes.values())
    print(f"Total memory moved: {total_bytes/1e9:.6f} GB\n")
    
    print("Per-GPU breakdown:")
    for dev, gpu_seconds in time_rows:
        mem_gb = mem_bytes.get(dev, 0) / 1e9
        print(f"  > GPU {dev}: {gpu_seconds:.6f} s, {mem_gb:.6f} GB moved")