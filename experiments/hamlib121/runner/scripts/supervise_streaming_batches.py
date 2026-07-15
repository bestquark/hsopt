#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from campaign_common import (
    DEFAULT_CODEX_BIN,
    PAPER_MODEL,
    PAPER_REASONING_EFFORT,
    db_connect,
    ensure_autoresearch_provenance,
    ensure_dirs,
    eval_submission_identity,
    init_db,
    load_json,
    load_open_eval_batches,
    log_event,
    mark_eval_batch_collected,
    mark_eval_batch_submitted,
    now_iso,
    register_eval_batch,
)
from mutate_iteration import load_lanes, prepare_codex_tools, run_one


@dataclass
class EvalJob:
    job_id: str | None
    iteration: int
    batch_id: str
    lane_ids: list[str]


def run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    return completed.stdout.strip()


def result_path(root: Path, iteration: int, lane_id: str) -> Path:
    return root / "iterations" / f"iter_{iteration:04d}" / "results" / f"{lane_id}.json"


def job_is_done(job_id: str, slurm_query_host: str) -> bool:
    try:
        queued = subprocess.run(
            ["ssh", slurm_query_host, f"squeue -h -j {job_id} -o '%T'"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"squeue timed out for job {job_id}", flush=True)
        return False
    if queued.returncode == 0:
        return not bool(queued.stdout.strip())

    try:
        accounted = subprocess.run(
            [
                "ssh",
                slurm_query_host,
                f"sacct -n -X -j {job_id} --format=State --parsable2",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"sacct timed out for job {job_id}", flush=True)
        return False
    states = [
        line.split("|", 1)[0].strip().split()[0]
        for line in accounted.stdout.splitlines()
        if line.strip()
    ]
    if not states:
        message = accounted.stderr.strip() or queued.stderr.strip()
        if message:
            print(message, flush=True)
        return False
    terminal_states = {
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
    return all(state.rstrip("+") in terminal_states for state in states)


def rescue_pending(
    root: Path,
    *,
    job_ids: list[str],
    query_host: str,
    primary_partition: str,
    fallback_partition: str,
) -> None:
    command = [
        sys.executable,
        "scripts/rescue_pending_eval_partitions.py",
        "--campaign-root",
        str(root),
        "--once",
        "--enable-fallback",
        "--query-host",
        query_host,
        "--primary-partition",
        primary_partition,
        "--fallback-partition",
        fallback_partition,
    ]
    for job_id in job_ids:
        command.extend(["--job-id", job_id])
    subprocess.run(command, cwd=root, check=False)


def submit_batch(
    *,
    root: Path,
    iteration: int,
    batch_id: str,
    lane_ids: list[str],
    account: str,
    partition: str,
    cpu_partition: str,
    sync_host: str,
    submit_host: str,
    legate_fbmem: str | None,
    uv_base_dir: str | None,
    uv_cache_dir: str | None,
    uv_python_install_dir: str | None,
) -> EvalJob:
    command = [
        sys.executable,
        "scripts/submit_eval_batch.py",
        "--campaign-root",
        str(root),
        "--iteration",
        str(iteration),
        "--batch-id",
        batch_id,
        "--partition",
        partition,
        "--cpu-partition",
        cpu_partition,
        "--account",
        account,
        "--sync-host",
        sync_host,
        "--submit-host",
        submit_host,
        "--compute-baseline",
    ]
    if legate_fbmem:
        command.extend(["--legate-fbmem", legate_fbmem])
    if uv_base_dir:
        command.extend(["--uv-base-dir", uv_base_dir])
    if uv_cache_dir:
        command.extend(["--uv-cache-dir", uv_cache_dir])
    if uv_python_install_dir:
        command.extend(["--uv-python-install-dir", uv_python_install_dir])
    for lane_id in lane_ids:
        command.extend(["--lane-id", lane_id])
    job_id = run(command, cwd=root).splitlines()[-1].strip()
    return EvalJob(job_id=job_id, iteration=iteration, batch_id=batch_id, lane_ids=lane_ids)


def collect_iteration(root: Path, iteration: int, sync_host: str, lane_ids: list[str]) -> None:
    command = [
        sys.executable,
        "scripts/collect_iteration.py",
        "--campaign-root",
        str(root),
        "--iteration",
        str(iteration),
        "--sync-host",
        sync_host,
    ]
    for lane_id in lane_ids:
        command.extend(["--lane-id", lane_id])
    run(command, cwd=root)


def initialize_lane_progress(
    lanes: list[dict], *, start_iteration: int, final_iteration: int
) -> tuple[dict[str, str], dict[str, int]]:
    lane_state: dict[str, str] = {}
    next_iteration: dict[str, int] = {}
    for lane in lanes:
        lane_id = lane["lane_id"]
        last_collected = int(lane.get("iteration") or 0)
        iteration = max(start_iteration, last_collected + 1)
        next_iteration[lane_id] = iteration
        lane_state[lane_id] = "done" if iteration > final_iteration else "idle"
    return lane_state, next_iteration


def least_advanced_first(lane_order: list[str], next_iteration: dict[str, int]) -> list[str]:
    lane_rank = {lane_id: rank for rank, lane_id in enumerate(lane_order)}
    return sorted(
        lane_order,
        key=lambda lane_id: (next_iteration[lane_id], lane_rank[lane_id]),
    )


def highest_existing_batch_index(root: Path, conn=None) -> int:
    highest = 0
    for path in root.glob("iterations/iter_*/batches/iter*_batch*"):
        match = re.fullmatch(r"iter\d+_batch(\d+)", path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    if conn is not None:
        for row in conn.execute("SELECT batch_id FROM eval_batches").fetchall():
            match = re.fullmatch(r"iter\d+_batch(\d+)", str(row["batch_id"]))
            if match:
                highest = max(highest, int(match.group(1)))
    return highest


def acquire_supervisor_lock(root: Path):
    lock_path = root / ".supervisor.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another streaming supervisor owns {lock_path}") from exc
    return handle


def archive_failed_mutation_attempt(
    root: Path,
    *,
    lane_id: str,
    iteration: int,
    attempt: int,
    returncode: int,
    error: str | None,
) -> Path:
    mutation_dir = root / "iterations" / f"iter_{iteration:04d}" / "mutations" / lane_id
    archive_dir = mutation_dir / "failed_attempts" / f"attempt_{attempt:03d}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "prompt.md",
        "codex.stdout",
        "codex.stderr",
        "last_message.txt",
        "before_simple_pf.py",
        "after_simple_pf.py",
    ):
        source = mutation_dir / name
        if source.is_file():
            shutil.copy2(source, archive_dir / name)
    (archive_dir / "failure.json").write_text(
        json.dumps(
            {
                "attempt": attempt,
                "error": error,
                "iteration": iteration,
                "lane_id": lane_id,
                "returncode": returncode,
                "recorded_at": now_iso(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive_dir


def next_failed_mutation_attempt(root: Path, *, lane_id: str, iteration: int) -> int:
    attempts_dir = (
        root / "iterations" / f"iter_{iteration:04d}" / "mutations" / lane_id / "failed_attempts"
    )
    existing = [
        int(match.group(1))
        for path in attempts_dir.glob("attempt_*")
        if (match := re.fullmatch(r"attempt_(\d+)", path.name))
    ]
    return max(existing, default=0) + 1


async def supervise(args: argparse.Namespace) -> int:
    root = args.campaign_root.resolve()
    ensure_dirs(root)
    supervisor_lock = acquire_supervisor_lock(root)
    conn = db_connect(root)
    init_db(conn)
    lanes = load_lanes(conn, limit=args.limit)
    accepted_baselines = conn.execute(
        "SELECT COUNT(*) FROM evaluations WHERE iteration=0 AND accepted=1"
    ).fetchone()[0]
    if int(accepted_baselines) < len(lanes):
        raise RuntimeError(
            f"campaign has {accepted_baselines}/{len(lanes)} accepted iteration-0 baselines; "
            "evaluate and collect the initial policy snapshot before starting mutation supervision"
        )
    ensure_autoresearch_provenance(
        campaign_root=root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        codex_bin=args.codex_bin,
    )
    lane_by_id = {lane["lane_id"]: lane for lane in lanes}
    lane_order = [lane["lane_id"] for lane in lanes]
    lane_state, next_iteration = initialize_lane_progress(
        lanes,
        start_iteration=args.start_iteration,
        final_iteration=args.iterations,
    )

    active_mutations: dict[asyncio.Task, tuple[str, int]] = {}
    ready: dict[int, deque[str]] = defaultdict(deque)
    first_ready_at: dict[int, float] = {}
    running_jobs: dict[str, EvalJob] = {}
    mutation_retries: dict[tuple[str, int], int] = defaultdict(int)
    mutation_retry_after: dict[str, float] = {}
    mutation_pause_until = 0.0
    consecutive_mutation_failures = 0
    batch_index = highest_existing_batch_index(root, conn)
    last_rescue = 0.0

    manifest = load_json(root / "manifests" / "campaign_manifest.json")
    cluster_root = str(manifest["cluster_root"])
    open_batch_lane_ids: set[str] = set()
    open_records = load_open_eval_batches(conn)
    for record in open_records:
        expected_key, expected_name = eval_submission_identity(
            cluster_root=cluster_root,
            iteration=record.iteration,
            batch_id=record.batch_id,
            policy_snapshot="current",
            lane_ids=record.lane_ids,
        )
        if (record.submission_key, record.job_name) != (expected_key, expected_name):
            raise RuntimeError(f"batch {record.batch_id}: persisted submission identity changed")
        if record.iteration < args.start_iteration or record.iteration > args.iterations:
            raise RuntimeError(
                f"batch {record.batch_id}: iteration {record.iteration} is outside requested "
                f"range {args.start_iteration}..{args.iterations}"
            )
        unknown = [lane_id for lane_id in record.lane_ids if lane_id not in lane_by_id]
        if unknown:
            raise RuntimeError(f"batch {record.batch_id}: unknown or inactive lanes {unknown}")
        for lane_id in record.lane_ids:
            pending = lane_by_id[lane_id].get("pending_iteration")
            if pending is None or int(pending) != record.iteration:
                raise RuntimeError(
                    f"batch {record.batch_id}: lane {lane_id} does not retain pending iteration "
                    f"{record.iteration}"
                )
            open_batch_lane_ids.add(lane_id)
            next_iteration[lane_id] = record.iteration
            lane_state[lane_id] = "evaluating"

        job = EvalJob(
            job_id=record.job_id,
            iteration=record.iteration,
            batch_id=record.batch_id,
            lane_ids=record.lane_ids,
        )
        if job.job_id is None:
            job = await asyncio.to_thread(
                submit_batch,
                root=root,
                iteration=record.iteration,
                batch_id=record.batch_id,
                lane_ids=record.lane_ids,
                account=args.account,
                partition=args.partition,
                cpu_partition=args.cpu_partition,
                sync_host=args.sync_host,
                submit_host=args.submit_host,
                legate_fbmem=args.legate_fbmem,
                uv_base_dir=args.uv_base_dir,
                uv_cache_dir=args.uv_cache_dir,
                uv_python_install_dir=args.uv_python_install_dir,
            )
            assert job.job_id is not None
            mark_eval_batch_submitted(conn, batch_id=record.batch_id, job_id=job.job_id)
        running_jobs[record.batch_id] = job

    timestamp = time.monotonic()
    for lane in lanes:
        lane_id = str(lane["lane_id"])
        if lane_id in open_batch_lane_ids:
            continue
        status = str(lane.get("status") or "")
        pending_raw = lane.get("pending_iteration")
        pending_iteration = None if pending_raw is None else int(pending_raw)
        if pending_iteration is None and status == "eval_retry":
            pending_iteration = int(lane.get("iteration") or 0)
        if pending_iteration is None and status in {"mutated", "queued_eval"}:
            pending_iteration = next_iteration[lane_id]

        if pending_iteration is None and lane_state[lane_id] == "done":
            continue

        if pending_iteration is not None:
            retry_count = int(lane.get("eval_retry_count") or 0)
            if retry_count > args.max_eval_retries:
                raise RuntimeError(
                    f"lane {lane_id}: evaluation retry limit already exceeded "
                    f"({retry_count}>{args.max_eval_retries})"
                )
            if not args.start_iteration <= pending_iteration <= args.iterations:
                raise RuntimeError(
                    f"lane {lane_id}: pending iteration {pending_iteration} is outside requested range"
                )
            candidate_path = (
                root
                / "iterations"
                / f"iter_{pending_iteration:04d}"
                / "mutations"
                / lane_id
                / "after_simple_pf.py"
            )
            if not candidate_path.is_file():
                raise RuntimeError(
                    f"lane {lane_id}: pending iteration {pending_iteration} has no candidate policy"
                )
            live_policy = root / "lanes" / lane_id / "simple_pf.py"
            shutil.copy2(candidate_path, live_policy)
            next_iteration[lane_id] = pending_iteration
            lane_state[lane_id] = "queued_eval"
            ready[pending_iteration].append(lane_id)
            first_ready_at.setdefault(pending_iteration, timestamp)
            conn.execute(
                """
                UPDATE lanes
                SET pending_iteration=?, status='queued_eval', updated_at=?
                WHERE lane_id=?
                """,
                (pending_iteration, now_iso(), lane_id),
            )
            continue

        if status in {"submitting", "evaluating"}:
            raise RuntimeError(
                f"lane {lane_id}: status {status!r} has no durable open batch record"
            )
        if status == "mutating":
            iteration = next_iteration[lane_id]
            before_path = (
                root
                / "iterations"
                / f"iter_{iteration:04d}"
                / "mutations"
                / lane_id
                / "before_simple_pf.py"
            )
            if before_path.is_file():
                shutil.copy2(before_path, root / "lanes" / lane_id / "simple_pf.py")
            conn.execute(
                "UPDATE lanes SET status='mutation_retry', updated_at=? WHERE lane_id=?",
                (now_iso(), lane_id),
            )
    conn.commit()

    tools_dir = prepare_codex_tools(root)

    log_event(
        conn,
        kind="stream-supervisor",
        message=(
            f"starting streaming supervisor: iterations={args.iterations}, "
            f"batch_size={args.batch_size}, max_mutators={args.max_concurrent_mutators}, "
            f"resume_range={min(next_iteration.values())}-{max(next_iteration.values())}"
        ),
        payload={
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "max_concurrent_mutators": args.max_concurrent_mutators,
            "max_running_eval_jobs": args.max_running_eval_jobs,
            "resumed_batch_index": batch_index,
            "partition": args.partition,
            "cpu_partition": args.cpu_partition,
            "account": args.account,
            "sync_host": args.sync_host,
            "submit_host": args.submit_host,
            "slurm_query_host": args.slurm_query_host,
            "legate_fbmem": args.legate_fbmem,
            "uv_base_dir": args.uv_base_dir,
            "uv_cache_dir": args.uv_cache_dir,
            "uv_python_install_dir": args.uv_python_install_dir,
            "recovered_eval_batches": len(open_records),
        },
    )

    def previous_result_ready(lane_id: str, iteration: int) -> bool:
        return (
            iteration <= args.start_iteration or result_path(root, iteration - 1, lane_id).exists()
        )

    def has_more_for_iteration(iteration: int) -> bool:
        return any(
            next_iteration[lane_id] == iteration and lane_state[lane_id] in {"idle", "mutating"}
            for lane_id in lane_order
        )

    def schedule_mutations() -> None:
        if time.monotonic() < mutation_pause_until:
            return
        for lane_id in least_advanced_first(lane_order, next_iteration):
            if len(active_mutations) >= args.max_concurrent_mutators:
                return
            if lane_state[lane_id] != "idle":
                continue
            if time.monotonic() < mutation_retry_after.get(lane_id, 0.0):
                continue
            iteration = next_iteration[lane_id]
            if iteration > args.iterations:
                lane_state[lane_id] = "done"
                continue
            if not previous_result_ready(lane_id, iteration):
                continue
            lane_state[lane_id] = "mutating"
            conn.execute(
                "UPDATE lanes SET status='mutating', updated_at=? WHERE lane_id=?",
                (now_iso(), lane_id),
            )
            conn.commit()
            task = asyncio.create_task(
                run_one(
                    root=root,
                    lane=lane_by_id[lane_id],
                    iteration=iteration,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    codex_bin=args.codex_bin,
                    timeout_seconds=args.timeout_seconds,
                    tools_dir=tools_dir,
                    semaphore=asyncio.Semaphore(1),
                )
            )
            active_mutations[task] = (lane_id, iteration)

    async def drain_finished_mutations() -> None:
        nonlocal mutation_pause_until, consecutive_mutation_failures
        if not active_mutations:
            return
        done = [task for task in active_mutations if task.done()]
        for task in done:
            lane_id, iteration = active_mutations.pop(task)
            try:
                result_lane_id, code, accepted, validation_error = task.result()
                lane_id = result_lane_id
            except BaseException as exc:
                code = 1
                accepted = False
                validation_error = f"{type(exc).__name__}: {exc}"
            status = "mutated" if accepted else "mutation_retry"
            print(
                json.dumps(
                    {
                        "event": "mutation_done",
                        "lane_id": lane_id,
                        "iteration": iteration,
                        "status": status,
                        "returncode": code,
                        "validation_error": validation_error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if accepted:
                consecutive_mutation_failures = 0
                conn.execute(
                    """
                    UPDATE lanes
                    SET status='queued_eval', pending_iteration=?, updated_at=?
                    WHERE lane_id=?
                    """,
                    (iteration, now_iso(), lane_id),
                )
                conn.commit()
                mutation_retry_after.pop(lane_id, None)
                mutation_retries.pop((lane_id, iteration), None)
                lane_state[lane_id] = "queued_eval"
                ready[iteration].append(lane_id)
                first_ready_at.setdefault(iteration, time.monotonic())
                continue

            key = (lane_id, iteration)
            attempt = max(
                mutation_retries[key] + 1,
                next_failed_mutation_attempt(root, lane_id=lane_id, iteration=iteration),
            )
            mutation_retries[key] = attempt
            consecutive_mutation_failures += 1
            archive_dir = archive_failed_mutation_attempt(
                root,
                lane_id=lane_id,
                iteration=iteration,
                attempt=attempt,
                returncode=code,
                error=validation_error,
            )
            if attempt > args.max_mutation_retries:
                conn.execute(
                    "UPDATE lanes SET status='mutation_error', updated_at=? WHERE lane_id=?",
                    (now_iso(), lane_id),
                )
                conn.commit()
                raise RuntimeError(
                    f"mutation failed for {lane_id} iteration {iteration} after "
                    f"{args.max_mutation_retries} retries; latest attempt archived at {archive_dir}"
                )
            retry_delay = min(
                args.mutation_retry_max_seconds,
                args.mutation_retry_base_seconds
                * (2 ** (max(attempt, consecutive_mutation_failures) - 1)),
            )
            mutation_retry_after[lane_id] = time.monotonic() + retry_delay
            mutation_pause_until = max(mutation_pause_until, time.monotonic() + retry_delay)
            lane_state[lane_id] = "idle"
            conn.execute(
                "UPDATE lanes SET status='mutation_retry', updated_at=? WHERE lane_id=?",
                (now_iso(), lane_id),
            )
            conn.commit()
            log_event(
                conn,
                kind="mutation-retry",
                lane_id=lane_id,
                iteration=iteration,
                message=(
                    f"mutation attempt {attempt} failed; retrying in {retry_delay}s: "
                    f"{validation_error or f'return code {code}'}"
                ),
                payload={
                    "attempt": attempt,
                    "archive_dir": str(archive_dir),
                    "error": validation_error,
                    "returncode": code,
                    "retry_delay_seconds": retry_delay,
                    "consecutive_failures": consecutive_mutation_failures,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "mutation_retry_scheduled",
                        "lane_id": lane_id,
                        "iteration": iteration,
                        "attempt": attempt,
                        "retry_delay_seconds": retry_delay,
                        "archive_dir": str(archive_dir),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    async def dispatch_ready_batches() -> None:
        nonlocal batch_index
        for iteration in sorted(list(ready)):
            queue = ready[iteration]
            while queue and len(running_jobs) < args.max_running_eval_jobs:
                timed_out = (
                    time.monotonic() - first_ready_at.get(iteration, time.monotonic())
                    >= args.max_batch_wait_seconds
                )
                final_partial = not has_more_for_iteration(iteration)
                if len(queue) < args.batch_size and not timed_out and not final_partial:
                    break
                take = min(args.batch_size, len(queue))
                lane_ids = [queue.popleft() for _ in range(take)]
                if not queue:
                    first_ready_at.pop(iteration, None)
                batch_index += 1
                batch_id = f"iter{iteration:04d}_batch{batch_index:04d}"
                submission_key, job_name = eval_submission_identity(
                    cluster_root=cluster_root,
                    iteration=iteration,
                    batch_id=batch_id,
                    policy_snapshot="current",
                    lane_ids=lane_ids,
                )
                register_eval_batch(
                    conn,
                    batch_id=batch_id,
                    iteration=iteration,
                    lane_ids=lane_ids,
                    submission_key=submission_key,
                    job_name=job_name,
                )
                for lane_id in lane_ids:
                    lane_state[lane_id] = "submitting"
                try:
                    job = await asyncio.to_thread(
                        submit_batch,
                        root=root,
                        iteration=iteration,
                        batch_id=batch_id,
                        lane_ids=lane_ids,
                        account=args.account,
                        partition=args.partition,
                        cpu_partition=args.cpu_partition,
                        sync_host=args.sync_host,
                        submit_host=args.submit_host,
                        legate_fbmem=args.legate_fbmem,
                        uv_base_dir=args.uv_base_dir,
                        uv_cache_dir=args.uv_cache_dir,
                        uv_python_install_dir=args.uv_python_install_dir,
                    )
                except BaseException as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    log_event(
                        conn,
                        kind="batch-submit-error",
                        iteration=iteration,
                        message=f"submission failed for {len(lane_ids)} lanes: {message}",
                        payload={"lane_ids": lane_ids, "error": message},
                    )
                    print(
                        json.dumps(
                            {
                                "event": "batch_submit_failed",
                                "iteration": iteration,
                                "lane_count": len(lane_ids),
                                "error": message,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    raise RuntimeError(
                        f"submission of durable batch {batch_id} is ambiguous; restart the "
                        "supervisor to reconcile the same batch ID"
                    ) from exc
                assert job.job_id is not None
                mark_eval_batch_submitted(conn, batch_id=batch_id, job_id=job.job_id)
                running_jobs[batch_id] = job
                for lane_id in lane_ids:
                    lane_state[lane_id] = "evaluating"
                print(
                    json.dumps(
                        {
                            "event": "batch_submitted",
                            "job_id": job.job_id,
                            "batch_id": job.batch_id,
                            "iteration": iteration,
                            "lane_count": len(lane_ids),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not queue:
                ready.pop(iteration, None)

    async def collect_finished_jobs() -> None:
        done_batch_ids = [
            batch_id
            for batch_id, job in running_jobs.items()
            if job.job_id is not None and job_is_done(job.job_id, args.slurm_query_host)
        ]
        for batch_id in done_batch_ids:
            job = running_jobs[batch_id]
            assert job.job_id is not None
            await asyncio.to_thread(
                collect_iteration, root, job.iteration, args.sync_host, job.lane_ids
            )
            successful_lane_ids: list[str] = []
            retry_lane_ids: list[str] = []
            exceeded_retries: list[tuple[str, int, str]] = []
            for lane_id in job.lane_ids:
                lane_result_path = result_path(root, job.iteration, lane_id)
                result = (
                    json.loads(lane_result_path.read_text(encoding="utf-8"))
                    if lane_result_path.exists()
                    else None
                )
                if result is not None and result.get("status") == "ok":
                    successful_lane_ids.append(lane_id)
                    if job.iteration >= args.iterations:
                        lane_state[lane_id] = "done"
                    else:
                        next_iteration[lane_id] = job.iteration + 1
                        lane_state[lane_id] = "idle"
                else:
                    retry_lane_ids.append(lane_id)
                    retry_count = (
                        int(
                            conn.execute(
                                "SELECT eval_retry_count FROM lanes WHERE lane_id=?", (lane_id,)
                            ).fetchone()[0]
                        )
                        + 1
                    )
                    if retry_count > args.max_eval_retries:
                        error_detail = str(
                            "missing result"
                            if result is None
                            else result.get("error", "evaluation error")
                        )
                        exceeded_retries.append((lane_id, retry_count, error_detail))
                    candidate_path = (
                        root
                        / "iterations"
                        / f"iter_{job.iteration:04d}"
                        / "mutations"
                        / lane_id
                        / "after_simple_pf.py"
                    )
                    live_policy = root / "lanes" / lane_id / "simple_pf.py"
                    if candidate_path.is_file():
                        shutil.copy2(candidate_path, live_policy)
                    lane_result_path.unlink(missing_ok=True)
                    lane_state[lane_id] = "queued_eval"
                    ready[job.iteration].append(lane_id)
                    first_ready_at.setdefault(job.iteration, time.monotonic())
            mark_eval_batch_collected(
                conn,
                batch_id=batch_id,
                successful_lane_ids=successful_lane_ids,
                retry_lane_ids=retry_lane_ids,
            )
            running_jobs.pop(batch_id)
            if exceeded_retries:
                details = "; ".join(
                    f"{lane_id}: attempt {attempt} ({error})"
                    for lane_id, attempt, error in exceeded_retries
                )
                raise RuntimeError(
                    f"evaluation retry limit exceeded at iteration {job.iteration}: {details}"
                )
            print(
                json.dumps(
                    {
                        "event": "batch_collected",
                        "job_id": job.job_id,
                        "batch_id": job.batch_id,
                        "iteration": job.iteration,
                        "lane_count": len(job.lane_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    while True:
        schedule_mutations()
        await drain_finished_mutations()
        await dispatch_ready_batches()
        await collect_finished_jobs()
        now = time.monotonic()
        if (
            args.enable_partition_fallback
            and running_jobs
            and now - last_rescue >= args.rescue_poll_seconds
        ):
            await asyncio.to_thread(
                rescue_pending,
                root,
                job_ids=[job.job_id for job in running_jobs.values() if job.job_id is not None],
                query_host=args.slurm_query_host,
                primary_partition=args.cpu_partition,
                fallback_partition=args.fallback_partition,
            )
            last_rescue = now

        remaining = [state for state in lane_state.values() if state != "done"]
        if not remaining and not active_mutations and not ready and not running_jobs:
            break
        await asyncio.sleep(args.poll_seconds)

    log_event(
        conn,
        kind="stream-supervisor",
        message="streaming supervisor completed all requested iterations",
    )
    supervisor_lock.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-lane streaming mutation/evaluation batches."
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--start-iteration", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-batch-wait-seconds", type=int, default=900)
    parser.add_argument("--max-running-eval-jobs", type=int, default=8)
    parser.add_argument("--max-eval-retries", type=int, default=3)
    parser.add_argument("--max-mutation-retries", type=int, default=20)
    parser.add_argument("--mutation-retry-base-seconds", type=int, default=60)
    parser.add_argument("--mutation-retry-max-seconds", type=int, default=900)
    parser.add_argument("--max-concurrent-mutators", type=int, default=16)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--rescue-poll-seconds", type=int, default=60)
    parser.add_argument("--account", default=os.environ.get("HSOPT_SLURM_ACCOUNT"))
    parser.add_argument("--partition", default=os.environ.get("HSOPT_GPU_PARTITION"))
    parser.add_argument("--cpu-partition", default=os.environ.get("HSOPT_CPU_PARTITION"))
    parser.add_argument("--fallback-partition", default=os.environ.get("HSOPT_FALLBACK_PARTITION"))
    parser.add_argument(
        "--enable-partition-fallback",
        action="store_true",
        help="Opt in to moving stuck pending jobs to --fallback-partition.",
    )
    parser.add_argument("--sync-host", default=os.environ.get("HSOPT_SYNC_HOST"))
    parser.add_argument("--submit-host", default=os.environ.get("HSOPT_SUBMIT_HOST"))
    parser.add_argument("--slurm-query-host", default=os.environ.get("HSOPT_QUERY_HOST"))
    parser.add_argument("--legate-fbmem")
    parser.add_argument("--uv-base-dir")
    parser.add_argument("--uv-cache-dir")
    parser.add_argument("--uv-python-install-dir")
    parser.add_argument("--mutation-limit", dest="limit", type=int)
    parser.add_argument("--model", default=os.environ.get("CODEX_MODEL", PAPER_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("CODEX_REASONING_EFFORT", PAPER_REASONING_EFFORT),
    )
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", DEFAULT_CODEX_BIN))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("CODEX_MUTATION_TIMEOUT_SECONDS", "1200")),
    )
    args = parser.parse_args()
    required = {
        "--account or HSOPT_SLURM_ACCOUNT": args.account,
        "--partition or HSOPT_GPU_PARTITION": args.partition,
        "--cpu-partition or HSOPT_CPU_PARTITION": args.cpu_partition,
        "--sync-host or HSOPT_SYNC_HOST": args.sync_host,
        "--submit-host or HSOPT_SUBMIT_HOST": args.submit_host,
        "--slurm-query-host or HSOPT_QUERY_HOST": args.slurm_query_host,
    }
    missing = [label for label, value in required.items() if not value]
    if args.enable_partition_fallback and not args.fallback_partition:
        missing.append("--fallback-partition or HSOPT_FALLBACK_PARTITION")
    if missing:
        parser.error("missing cluster configuration: " + ", ".join(missing))
    raise SystemExit(asyncio.run(supervise(args)))


if __name__ == "__main__":
    main()
