import sqlite3
from pathlib import Path

def compute_gpu_seconds(db_path: Path) -> float:
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

    return total, rows

if __name__ == "__main__":

    db_path = Path("prof.sqlite")
    total, rows = compute_gpu_seconds(db_path)

    print(f"Total GPU time: {total:.6f} seconds")

    for device_id, gpu_seconds in rows:
        print(f"  Device {device_id}: {gpu_seconds:.6f} seconds")
