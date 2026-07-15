#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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

from campaign_common import (  # noqa: E402
    campaign_lock,
    ensure_campaign_runner_runtime,
    open_collect_transactions,
    read_json,
    require_campaign,
)


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def run_child(command: list[str], cwd: Path, *, allowed_codes: set[int] = {0}) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode not in allowed_codes:
        raise RuntimeError(f"campaign child exited {completed.returncode}: {' '.join(command)}")
    return completed.returncode


def latest_submitted_jobs(state: dict[str, Any], iteration: int) -> dict[str, str]:
    jobs: dict[str, str] = {}
    for batch in state["batches"].values():
        if int(batch.get("iteration", -1)) != iteration or batch.get("state") != "submitted":
            continue
        attempts = batch.get("attempts", [])
        if not attempts or attempts[-1].get("state") != "submitted":
            raise RuntimeError(f"submitted batch {batch.get('batch_id')} lacks a submitted attempt")
        job_id = str(attempts[-1].get("job_id", ""))
        if not job_id.isdigit():
            raise RuntimeError(f"invalid persisted Slurm job ID {job_id!r}")
        jobs[job_id] = str(batch["batch_id"])
    return jobs


def query_slurm_states(host: str, job_ids: list[str], timeout_seconds: int) -> dict[str, str]:
    joined = ",".join(job_ids)
    squeue = subprocess.run(
        ["ssh", host, "squeue", "-h", "-j", joined, "-o", "%A|%T"],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if squeue.returncode != 0:
        raise RuntimeError(f"Slurm squeue query failed: {squeue.stderr.strip()}")
    states: dict[str, str] = {}
    for line in squeue.stdout.splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) == 2 and fields[0] in job_ids:
            states[fields[0]] = fields[1].split("+", 1)[0].strip().upper()
    absent = [job_id for job_id in job_ids if job_id not in states]
    if absent:
        sacct = subprocess.run(
            [
                "ssh",
                host,
                "sacct",
                "-n",
                "-X",
                "-j",
                ",".join(absent),
                "--format=JobIDRaw,State",
                "-P",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if sacct.returncode != 0:
            raise RuntimeError(f"Slurm sacct query failed: {sacct.stderr.strip()}")
        for line in sacct.stdout.splitlines():
            fields = line.strip().split("|", 1)
            if len(fields) != 2:
                continue
            job_id = fields[0].strip()
            if job_id in absent:
                states[job_id] = fields[1].split("+", 1)[0].strip().upper()
    missing = [job_id for job_id in job_ids if not states.get(job_id)]
    if missing:
        raise RuntimeError(
            f"Slurm returned no state for jobs {missing}; refusing to assume completion"
        )
    return states


def wait_for_terminal_jobs(
    *, host: str, jobs: dict[str, str], poll_seconds: int, timeout_seconds: int
) -> dict[str, str]:
    while True:
        states = query_slurm_states(host, sorted(jobs), timeout_seconds)
        print(json.dumps({jobs[job_id]: state for job_id, state in states.items()}, sort_keys=True))
        if all(state in TERMINAL_STATES for state in states.values()):
            return states
        unknown = sorted(
            set(states.values())
            - TERMINAL_STATES
            - {
                "CONFIGURING",
                "COMPLETING",
                "PENDING",
                "RUNNING",
                "RESIZING",
                "SUSPENDED",
            }
        )
        if unknown:
            raise RuntimeError(f"unrecognized Slurm states: {unknown}")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume and supervise the fixed reaction campaign through iteration 50."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--sync-host", required=True)
    parser.add_argument("--submit-host", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--max-concurrent-mutators", type=int, default=8)
    parser.add_argument("--mutation-timeout-seconds", type=int, default=1200)
    parser.add_argument("--evaluator-workers", type=int, default=2)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--ssh-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-evaluation-retries", type=int, default=3)
    parser.add_argument("--unbounded-evaluation-retries", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.max_concurrent_mutators,
            args.mutation_timeout_seconds,
            args.evaluator_workers,
            args.cpus_per_task,
            args.poll_seconds,
            args.ssh_timeout_seconds,
        )
        <= 0
    ):
        parser.error("concurrency, timeout, worker, CPU, and polling values must be positive")
    if args.max_evaluation_retries < 0:
        parser.error("--max-evaluation-retries must be nonnegative")

    root = args.campaign_root.resolve()
    runner = Path(__file__).resolve().parent
    with campaign_lock(root, "supervisor"):
        while True:
            manifest, state = require_campaign(root, allow_open_collect_transaction=True)
            ensure_campaign_runner_runtime(root, Path(__file__), manifest)
            open_transactions = open_collect_transactions(root)
            if open_transactions:
                if len(open_transactions) != 1:
                    raise RuntimeError(
                        f"multiple open collection transactions: {open_transactions}"
                    )
                transaction = read_json(open_transactions[0])
                run_child(
                    [
                        sys.executable,
                        str(runner / "collect_iteration.py"),
                        "--campaign-root",
                        str(root),
                        "--iteration",
                        str(transaction["iteration"]),
                        "--skip-rsync",
                    ],
                    root,
                )
                continue
            manifest, state = require_campaign(root)
            next_iteration = state.get("next_iteration")
            if next_iteration is None:
                if int(state["current_collected_iteration"]) != 50:
                    raise RuntimeError("campaign ended before collecting iteration 50")
                print("reaction campaign complete: baseline + 50 proposals collected for 20 lanes")
                return
            iteration = int(next_iteration)
            if iteration > 0:
                run_child(
                    [
                        sys.executable,
                        str(runner / "mutate_iteration.py"),
                        "--campaign-root",
                        str(root),
                        "--iteration",
                        str(iteration),
                        "--max-concurrent",
                        str(args.max_concurrent_mutators),
                        "--timeout-seconds",
                        str(args.mutation_timeout_seconds),
                    ],
                    root,
                )

            _manifest, state = require_campaign(root)
            batches = [
                batch
                for batch in state["batches"].values()
                if int(batch.get("iteration", -1)) == iteration
            ]
            if not batches or any(
                batch.get("state") in {"prepared", "retry_ready"} for batch in batches
            ):
                retry_counts = [
                    int(batch.get("retry_count", 0))
                    for batch in batches
                    if batch.get("state") == "retry_ready"
                ]
                if (
                    retry_counts
                    and not args.unbounded_evaluation_retries
                    and max(retry_counts) > args.max_evaluation_retries
                ):
                    raise RuntimeError(
                        f"iteration {iteration} exceeded {args.max_evaluation_retries} "
                        "evaluation retries; inspect errors or pass --unbounded-evaluation-retries"
                    )
                run_child(
                    [
                        sys.executable,
                        str(runner / "submit_iteration.py"),
                        "--campaign-root",
                        str(root),
                        "--iteration",
                        str(iteration),
                        "--sync-host",
                        args.sync_host,
                        "--submit-host",
                        args.submit_host,
                        "--account",
                        args.account,
                        "--partition",
                        args.partition,
                        "--workers",
                        str(args.evaluator_workers),
                        "--cpus-per-task",
                        str(args.cpus_per_task),
                        "--memory",
                        args.memory,
                        "--time-limit",
                        args.time_limit,
                    ],
                    root,
                )

            _manifest, state = require_campaign(root)
            jobs = latest_submitted_jobs(state, iteration)
            if not jobs:
                raise RuntimeError(f"iteration {iteration} has no submitted evaluation jobs")
            wait_for_terminal_jobs(
                host=args.submit_host,
                jobs=jobs,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.ssh_timeout_seconds,
            )
            collect_code = run_child(
                [
                    sys.executable,
                    str(runner / "collect_iteration.py"),
                    "--campaign-root",
                    str(root),
                    "--iteration",
                    str(iteration),
                    "--sync-host",
                    args.sync_host,
                ],
                root,
                allowed_codes={0, 2},
            )
            if collect_code == 2:
                print(
                    f"iteration {iteration}: retryable evaluator output; resubmitting", flush=True
                )


if __name__ == "__main__":
    main()
