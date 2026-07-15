#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-hamlib121-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX


PAPER_RUNS = 5
PAPER_LANES = 121
PAPER_ITERATIONS = 50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def campaign_progress(
    replicate_root: Path,
    *,
    runs: int,
    expected_lanes: int,
    expected_iterations: int,
) -> dict:
    expected_proposals = expected_lanes * expected_iterations
    run_rows = []
    for index in range(1, runs + 1):
        run_name = f"run_{index:02d}"
        database = replicate_root / run_name / "campaign.sqlite"
        if not database.is_file():
            run_rows.append(
                {
                    "run": run_name,
                    "active_lanes": 0,
                    "proposals": 0,
                    "expected_proposals": expected_proposals,
                    "min_iteration": None,
                    "max_iteration": None,
                    "complete": False,
                    "status": "missing_campaign_database",
                }
            )
            continue
        try:
            with sqlite3.connect(database) as conn:
                active_lanes = int(
                    conn.execute("SELECT COUNT(*) FROM lanes WHERE active=1").fetchone()[0]
                )
                proposals = int(
                    conn.execute("SELECT COUNT(*) FROM evaluations WHERE iteration>0").fetchone()[0]
                )
                min_iteration, max_iteration = conn.execute(
                    "SELECT MIN(iteration), MAX(iteration) FROM lanes WHERE active=1"
                ).fetchone()
                open_batches = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM eval_batches WHERE state!='collected'"
                    ).fetchone()[0]
                )
        except sqlite3.Error as exc:
            run_rows.append(
                {
                    "run": run_name,
                    "active_lanes": 0,
                    "proposals": 0,
                    "expected_proposals": expected_proposals,
                    "min_iteration": None,
                    "max_iteration": None,
                    "complete": False,
                    "status": "invalid_campaign_database",
                    "error": str(exc),
                }
            )
            continue
        complete = (
            active_lanes == expected_lanes
            and proposals == expected_proposals
            and min_iteration is not None
            and max_iteration is not None
            and int(min_iteration) == expected_iterations
            and int(max_iteration) == expected_iterations
            and open_batches == 0
        )
        run_rows.append(
            {
                "run": run_name,
                "active_lanes": active_lanes,
                "proposals": proposals,
                "expected_proposals": expected_proposals,
                "min_iteration": min_iteration,
                "max_iteration": max_iteration,
                "open_batches": open_batches,
                "complete": complete,
                "status": "complete" if complete else "incomplete",
            }
        )
    return {
        "checked_at": now_iso(),
        "complete": all(row["complete"] for row in run_rows),
        "total_proposals": sum(row["proposals"] for row in run_rows),
        "expected_total_proposals": runs * expected_proposals,
        "runs": run_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for five replicate campaigns, then run the strict final exporter."
    )
    parser.add_argument("--replicate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=PAPER_RUNS)
    parser.add_argument("--expected-lanes", type=int, default=PAPER_LANES)
    parser.add_argument("--expected-iterations", type=int, default=PAPER_ITERATIONS)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.runs,
        args.expected_lanes,
        args.expected_iterations,
    ) != (PAPER_RUNS, PAPER_LANES, PAPER_ITERATIONS):
        parser.error("the final watcher is fixed to 5 runs, 121 lanes, and 50 proposal iterations")

    root = args.replicate_root.resolve()
    output = args.output_dir.resolve()
    status_path = output.parent / f"{output.name}_watch_status.json"
    exporter = root / "run_01" / "scripts" / "export_replicate_results.py"

    while True:
        progress = campaign_progress(
            root,
            runs=args.runs,
            expected_lanes=args.expected_lanes,
            expected_iterations=args.expected_iterations,
        )
        atomic_write_json(status_path, progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
        if progress["complete"]:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(exporter),
                    "--replicate-root",
                    str(root),
                    "--output-dir",
                    str(output),
                    "--runs",
                    str(args.runs),
                    "--expected-lanes",
                    str(args.expected_lanes),
                    "--expected-iterations",
                    str(args.expected_iterations),
                ],
                cwd=root / "run_01",
                text=True,
                capture_output=True,
                check=False,
            )
            export_record = {
                **progress,
                "exported_at": now_iso(),
                "export_returncode": completed.returncode,
                "export_stdout": completed.stdout,
                "export_stderr": completed.stderr,
            }
            atomic_write_json(status_path, export_record)
            if completed.returncode == 0:
                return
            print(completed.stderr, file=sys.stderr, flush=True)
        if args.once:
            raise SystemExit(1)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
