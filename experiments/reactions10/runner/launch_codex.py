#!/usr/bin/env python3
"""Durably distinguish wrapper exec from an actual Codex JSONL startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
from pathlib import Path

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-runner-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX


EXEC_ACK_PROTOCOL = "codex_jsonl_lifecycle_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CODEX_START_EVENT = "thread.started"


def canonical_event_sha256(event: dict[str, object]) -> str:
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_all(fd: int, payload: bytes) -> None:
    while payload:
        written = os.write(fd, payload)
        payload = payload[written:]


def codex_start_event(raw_line: bytes) -> tuple[str, str] | None:
    try:
        event = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(event, dict) or event.get("type") != CODEX_START_EVENT:
        return None
    thread_id = event.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id or len(thread_id) > 256:
        return None
    return thread_id, canonical_event_sha256(event)


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
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("launcher received no command")
    if SHA256_RE.fullmatch(args.launch_nonce) is None:
        raise SystemExit("launcher received an invalid launch nonce")
    started_at = time.time()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "exec_pid": None,
        "work_dir": str(args.work_dir.resolve()),
        "command_sha256": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
        "status": "exec_error",
        "exec_ack_protocol": EXEC_ACK_PROTOCOL,
        "launch_nonce": args.launch_nonce,
        "low_level_exec_status": "not_confirmed",
        "started_at_epoch": started_at,
    }
    if receipt["pid"] != receipt["pgid"]:
        receipt["error"] = "launcher must be the leader of its isolated process group"
        write_json(args.receipt, receipt)
        raise SystemExit(127)
    try:
        exec_read_fd, exec_write_fd = os.pipe()
        stdout_read_fd, stdout_write_fd = os.pipe()
        os.set_inheritable(exec_write_fd, False)
        child_pid = os.fork()
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.receipt, receipt)
        raise SystemExit(127) from exc

    if child_pid == 0:
        os.close(exec_read_fd)
        os.close(stdout_read_fd)
        try:
            os.dup2(stdout_write_fd, 1)
            os.close(stdout_write_fd)
            os.execvp(command[0], command)
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")[:4096]
            try:
                os.write(exec_write_fd, message)
            except OSError:
                pass
            os._exit(127)

    receipt["exec_pid"] = child_pid
    os.close(exec_write_fd)
    os.close(stdout_write_fd)
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(exec_read_fd, 4096)
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.append(chunk)
    os.close(exec_read_fd)

    if chunks:
        _waited_pid, status = os.waitpid(child_pid, 0)
        os.close(stdout_read_fd)
        receipt["error"] = b"".join(chunks).decode("utf-8", errors="replace")
        receipt["child_wait_status"] = status
        write_json(args.receipt, receipt)
        raise SystemExit(127)

    receipt["status"] = "exec_confirmed_pending_codex"
    receipt["low_level_exec_status"] = "exec_confirmed"
    receipt["exec_confirmed_at_epoch"] = time.time()
    write_json(args.receipt, receipt)
    line_buffer = b""
    codex_started = False
    while True:
        try:
            chunk = os.read(stdout_read_fd, 65536)
        except InterruptedError:
            continue
        if not chunk:
            break
        write_all(1, chunk)
        if codex_started:
            continue
        line_buffer += chunk
        if len(line_buffer) > 8 * 1024 * 1024 and b"\n" not in line_buffer:
            # An actual Codex JSONL lifecycle never emits an unbounded line.
            line_buffer = b""
            continue
        while b"\n" in line_buffer:
            raw_line, line_buffer = line_buffer.split(b"\n", 1)
            start = codex_start_event(raw_line)
            if start is None:
                continue
            thread_id, event_hash = start
            receipt["status"] = "codex_started"
            receipt["codex_thread_id"] = thread_id
            receipt["codex_start_event_sha256"] = event_hash
            receipt["codex_started_at_epoch"] = time.time()
            write_json(args.receipt, receipt)
            codex_started = True
            break
    os.close(stdout_read_fd)
    _waited_pid, status = os.waitpid(child_pid, 0)
    if not codex_started:
        receipt["status"] = "codex_start_error"
        receipt["error"] = "Codex emitted no valid thread.started JSONL event"
        receipt["child_wait_status"] = status
        write_json(args.receipt, receipt)
    if os.WIFEXITED(status):
        raise SystemExit(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        raise SystemExit(128 + os.WTERMSIG(status))
    raise SystemExit(125)


if __name__ == "__main__":
    main()
