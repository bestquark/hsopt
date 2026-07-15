#!/usr/bin/env python3
"""Internal durable launcher for one isolated Codex mutation process.

The launcher is placed in a new session by mutate_iteration.py. It writes its
PID/process-group receipt itself before exec, so a supervisor crash cannot lose
the identity of a still-running Codex process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal durable launcher used by mutate_iteration.py."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("launcher received no command")
    receipt = {
        "schema_version": 1,
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "work_dir": str(args.work_dir.resolve()),
        "command_sha256": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
        "status": "launched",
        "started_at_epoch": time.time(),
    }
    write_json(args.receipt, receipt)
    try:
        os.execvp(command[0], command)
    except Exception as exc:
        receipt["status"] = "exec_error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.receipt, receipt)
        raise


if __name__ == "__main__":
    main()
