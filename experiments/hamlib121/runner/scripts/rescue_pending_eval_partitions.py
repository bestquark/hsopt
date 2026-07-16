#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
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

from campaign_common import (  # noqa: E402
    db_connect,
    ensure_dirs,
    init_db,
    log_event,
    ssh_command,
    validate_ssh_host,
)
from submit_eval_batch import validate_slurm_name  # noqa: E402


STUCK_REASON_MARKERS = (
    "DOWN",
    "DRAINED",
    "Dependency",
    "Priority",
    "reserved for jobs in higher priority partitions",
)
EVAL_JOB_NAMES = {"hamlib_pf_eval", "hamlib_pf_batch", "hamlib_pf_autofill"}
REMOTE_TIMEOUT_SECONDS = 30


def run_remote(
    command: str,
    *,
    query_host: str,
    timeout_seconds: int = REMOTE_TIMEOUT_SECONDS,
    proxy_jump_host: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ssh_command(query_host, command, proxy_jump_host=proxy_jump_host),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=124,
            stdout=stdout,
            stderr=(stderr + f"\nlocal timeout after {timeout_seconds}s").strip(),
        )


def normalize_job_id(job_id: str) -> str:
    return job_id.strip().split("_", 1)[0]


def is_eval_job_name(name: str) -> bool:
    return name in EVAL_JOB_NAMES or re.fullmatch(r"hpf_[0-9a-f]{12}", name) is not None


def pending_eval_tasks(
    primary_partition: str,
    *,
    query_host: str,
    job_ids: list[str],
    proxy_jump_host: str | None = None,
) -> list[tuple[str, str]]:
    normalized_job_ids: list[str] = []
    for supplied_job_id in job_ids:
        normalized_job_id = normalize_job_id(supplied_job_id)
        if re.fullmatch(r"\d+", normalized_job_id) is None:
            raise ValueError(f"invalid Slurm job ID: {supplied_job_id!r}")
        if normalized_job_id not in normalized_job_ids:
            normalized_job_ids.append(normalized_job_id)
    if not normalized_job_ids:
        return []

    job_selector = ",".join(normalized_job_ids)
    query_host = validate_ssh_host(query_host, "query_host")
    primary_partition = validate_slurm_name(primary_partition, "primary_partition")
    completed = run_remote(
        shlex.join(
            ["timeout", "20s", "squeue", "-h", "-j", job_selector, "-t", "PD", "-o", "%i|%P|%j|%R"]
        ),
        query_host=query_host,
        proxy_jump_host=proxy_jump_host,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    tasks: list[tuple[str, str]] = []
    seen_job_ids: set[str] = set()
    for line in completed.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        job_id, partition, name, reason = parts
        if not is_eval_job_name(name):
            continue
        if partition != primary_partition:
            continue
        if any(marker in reason for marker in STUCK_REASON_MARKERS):
            normalized_job_id = normalize_job_id(job_id)
            if normalized_job_id in seen_job_ids:
                continue
            seen_job_ids.add(normalized_job_id)
            tasks.append((normalized_job_id, reason.strip()))
    return tasks


def rescue_once(
    *,
    fallback_partition: str,
    job_ids: list[str],
    primary_partition: str,
    query_host: str,
    proxy_jump_host: str | None = None,
) -> list[dict]:
    fallback_partition = validate_slurm_name(fallback_partition, "fallback_partition")
    primary_partition = validate_slurm_name(primary_partition, "primary_partition")
    query_host = validate_ssh_host(query_host, "query_host")
    rescued: list[dict] = []
    for job_id, reason in pending_eval_tasks(
        primary_partition,
        query_host=query_host,
        job_ids=job_ids,
        proxy_jump_host=proxy_jump_host,
    ):
        completed = run_remote(
            shlex.join(
                [
                    "timeout",
                    "20s",
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    f"Partition={fallback_partition}",
                ]
            ),
            query_host=query_host,
            proxy_jump_host=proxy_jump_host,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"failed to move job {job_id} to {fallback_partition}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        rescued.append(
            {
                "job_id": job_id,
                "reason": reason,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    return rescued


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move stuck pending evaluation batches to a fallback partition."
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--query-host", default=os.environ.get("HSOPT_QUERY_HOST"))
    parser.add_argument(
        "--proxy-jump-host",
        help="Optional SSH config alias used as ProxyJump for the query host.",
    )
    parser.add_argument("--primary-partition", default=os.environ.get("HSOPT_CPU_PARTITION"))
    parser.add_argument(
        "--fallback-partition",
        default=os.environ.get("HSOPT_FALLBACK_PARTITION"),
    )
    parser.add_argument(
        "--enable-fallback",
        action="store_true",
        help="Actually move pending jobs to the fallback partition. Default is no-op.",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="Slurm job ID owned by this campaign (repeat for each active batch).",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if not args.enable_fallback:
        print(
            json.dumps(
                {
                    "rescued": [],
                    "disabled": True,
                    "message": "fallback partition rescue disabled; leaving pending jobs on primary partition",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    missing = [
        label
        for label, value in (
            ("--query-host or HSOPT_QUERY_HOST", args.query_host),
            ("--primary-partition or HSOPT_CPU_PARTITION", args.primary_partition),
            (
                "--fallback-partition or HSOPT_FALLBACK_PARTITION",
                args.fallback_partition,
            ),
        )
        if not value
    ]
    if missing:
        parser.error("missing cluster configuration: " + ", ".join(missing))
    if not args.job_id:
        parser.error("--enable-fallback requires at least one campaign-scoped --job-id")
    try:
        args.query_host = validate_ssh_host(str(args.query_host), "query_host")
        args.proxy_jump_host = (
            validate_ssh_host(str(args.proxy_jump_host), "proxy_jump_host")
            if args.proxy_jump_host is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    root = args.campaign_root.resolve()
    ensure_dirs(root)
    conn = db_connect(root)
    init_db(conn)

    while True:
        rescued = rescue_once(
            primary_partition=args.primary_partition,
            fallback_partition=args.fallback_partition,
            query_host=args.query_host,
            job_ids=args.job_id,
            proxy_jump_host=args.proxy_jump_host,
        )
        if rescued:
            print(json.dumps({"rescued": rescued}, sort_keys=True), flush=True)
            log_event(
                conn,
                kind="slurm-rescue",
                message=f"moved {len(rescued)} pending eval task(s) to {args.fallback_partition}",
                payload={"rescued": rescued},
            )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
