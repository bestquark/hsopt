"""Durable Omni supervisor for exact compiled-unitary autoresearch campaigns."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .exact_export import export_family
from .exact_loop import propose_lane, propose_replay_lane, reconcile_lane
from .exact_queue import queue_status, requeue_missing
from .nvpark_queue import (
    DEFAULT_ACCELERATOR,
    DEFAULT_ACCOUNT,
    DEFAULT_BACKEND,
    DEFAULT_CPUS,
    DEFAULT_HOST,
    DEFAULT_MEMORY,
    DEFAULT_PARTITION,
    DEFAULT_QOS,
    DEFAULT_REMOTE_ROOT,
    DEFAULT_WALLTIME,
    collect_batch,
    remote_status,
    submit_batch,
    submit_ready,
)
from .tgate_opt.reporting import generate_reports, verify_strict_export


TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:1000]


def _run_mutation_proposals(
    lane_dirs: list[Path],
    *,
    workers: int,
    timeout_seconds: int,
) -> list[tuple[Path, dict[str, Any]]]:
    """Mutate distinct sequential lanes concurrently in isolated sandboxes."""

    selected = lane_dirs[:workers]

    def propose(lane_dir: Path) -> tuple[Path, dict[str, Any]]:
        try:
            result = propose_lane(lane_dir, timeout_seconds=timeout_seconds)
        except Exception as exc:
            result = {"status": "error", "error": _safe_error(exc)}
        return lane_dir, result

    if workers == 1:
        return [propose(selected[0])] if selected else []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hsopt-mutation") as pool:
        return list(pool.map(propose, selected))


def _append_event(campaign_root: Path, event: str, **payload: Any) -> None:
    event_path = campaign_root / "supervisor" / "events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": _now(), "event": event, **payload}
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _campaign_lock(campaign_root: Path) -> Iterator[None]:
    lock_path = campaign_root / "supervisor" / "supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another supervisor holds {lock_path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _campaign_paths(campaign_root: Path) -> tuple[dict[str, Any], Path, list[Path]]:
    campaign_root = campaign_root.resolve()
    campaign = _read_json(campaign_root / "campaign.json")
    if int(campaign.get("schema", 0)) != 1:
        raise ValueError("unsupported campaign schema")
    queue_dir = Path(campaign["queue_dir"]).resolve()
    if not queue_dir.is_relative_to(campaign_root):
        raise ValueError("campaign queue must live under the campaign root")
    lane_dirs: list[Path] = []
    for row in campaign["lanes"]:
        lane_dir = (campaign_root / str(row["lane_dir"])).resolve()
        if not lane_dir.is_relative_to(campaign_root):
            raise ValueError(f"unsafe lane path: {row['lane_dir']}")
        for required in ("manifest.json", "state.json", "initial_pf.py", "incumbent.py"):
            if not (lane_dir / required).is_file():
                raise FileNotFoundError(lane_dir / required)
        lane_dirs.append(lane_dir)
    if len(lane_dirs) != len(set(lane_dirs)):
        raise ValueError("campaign contains duplicate lane directories")
    return campaign, queue_dir, lane_dirs


def slurm_state(status: dict[str, Any]) -> str:
    """Extract the root job state from ``sacct --parsable2`` output."""

    job_id = str(status.get("job_id") or "")
    fallback: str | None = None
    for raw in status.get("rows", []):
        columns = str(raw).split("|")
        if len(columns) < 2:
            continue
        row_job_id = columns[0].strip()
        state = columns[1].strip().split(maxsplit=1)[0].rstrip("+").upper()
        if not state:
            continue
        if row_job_id == job_id:
            return state
        if fallback is None and row_job_id.split(".", maxsplit=1)[0] == job_id:
            fallback = state
    return fallback or "UNKNOWN"


def _batch_task_count(batch_dir: Path) -> int:
    return len(_read_json(batch_dir / "batch.json").get("task_ids", []))


def _collect_terminal_batch(
    *,
    campaign_root: Path,
    queue_dir: Path,
    batch_dir: Path,
    state: str,
    max_batch_attempts: int,
) -> dict[str, Any]:
    collected = collect_batch(queue_dir, batch_dir)
    recovery: dict[str, Any] | None = None
    if collected.get("missing"):
        recovery = requeue_missing(
            queue_dir,
            batch_dir,
            reason=f"Slurm state {state}",
            max_attempts=max_batch_attempts,
        )
    marker = {
        "schema": 1,
        "collected_at": _now(),
        "slurm_state": state,
        "collection": collected,
        "recovery": recovery,
    }
    _write_json(batch_dir / "collected.json", marker)
    _append_event(
        campaign_root,
        "batch_collected",
        batch_id=batch_dir.name,
        job_id=collected.get("job_id"),
        slurm_state=state,
        merged=collected.get("merged", 0),
        missing=len(collected.get("missing", [])),
        requeued=len((recovery or {}).get("requeued", [])),
        exhausted=len((recovery or {}).get("exhausted", [])),
    )
    return marker


def _lane_summary(lane_dirs: list[Path]) -> tuple[dict[str, Any], list[Path]]:
    states = []
    ready = []
    by_family: dict[str, Counter[str]] = {}
    for lane_dir in lane_dirs:
        manifest = _read_json(lane_dir / "manifest.json")
        state = _read_json(lane_dir / "state.json")
        target = int(manifest["iterations_target"])
        completed = int(state["iterations_completed"])
        if completed >= target:
            status = "complete"
        elif state.get("pending_kind") == "mutation":
            status = "mutating"
        elif state.get("pending_kind") == "replay_validation_failed":
            status = "needs_attention"
        elif state.get("pending_task_id") is not None:
            status = "waiting_evaluation"
        else:
            status = "ready"
            ready.append(lane_dir)
        states.append((manifest, state, status))
        by_family.setdefault(str(manifest["family"]), Counter())[status] += 1

    ready.sort(
        key=lambda lane_dir: (
            int(_read_json(lane_dir / "state.json")["iterations_completed"]),
            str(_read_json(lane_dir / "state.json").get("updated_at", "")),
            str(lane_dir),
        )
    )
    status_counts = Counter(status for _, _, status in states)
    return (
        {
            "total": len(states),
            "iterations_completed": sum(
                int(state["iterations_completed"]) for _, state, _ in states
            ),
            "iterations_target": sum(
                int(manifest["iterations_target"]) for manifest, _, _ in states
            ),
            "accepted_mutations": sum(
                int(state.get("accepted_mutations", 0)) for _, state, _ in states
            ),
            "by_status": dict(sorted(status_counts.items())),
            "by_family": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(by_family.items())
            },
        },
        ready,
    )


def _finalize_campaign(
    campaign_root: Path,
    *,
    hamlib_export_dir: Path,
    hamlib_report_dir: Path,
    reactions_export_dir: Path,
    reactions_report_dir: Path,
) -> dict[str, Any]:
    """Create both checksummed exports and both report trees exactly once."""

    pairs = (
        ("hamlib121", hamlib_export_dir, hamlib_report_dir),
        ("reactions10", reactions_export_dir, reactions_report_dir),
    )
    records: dict[str, Any] = {}
    for family, export_dir, report_dir in pairs:
        export_dir = export_dir.expanduser().absolute()
        report_dir = report_dir.expanduser().absolute()
        if export_dir.exists():
            verify_strict_export(export_dir)
            export_status = "existing"
        else:
            export_family(campaign_root, family, export_dir)
            export_status = "created"
        if report_dir.exists():
            if not (report_dir / "report_manifest.json").is_file():
                raise RuntimeError(f"existing report directory is incomplete: {report_dir}")
            report_status = "existing"
        else:
            generate_reports(family, export_dir, report_dir)
            report_status = "created"
        records[family] = {
            "export_dir": str(export_dir),
            "export_status": export_status,
            "report_dir": str(report_dir),
            "report_status": report_status,
        }
    result = {
        "schema": 1,
        "status": "complete",
        "completed_at": _now(),
        "families": records,
    }
    _write_json(campaign_root / "supervisor" / "finalization.json", result)
    return result


def supervise_once(
    campaign_root: Path,
    *,
    repo_dir: Path,
    threshold: int = 40,
    max_active_jobs: int = 4,
    mutation_workers: int = 1,
    mutation_timeout_seconds: int = 900,
    max_batch_attempts: int = 3,
    mutations_enabled: bool = True,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    account: str = DEFAULT_ACCOUNT,
    partition: str = DEFAULT_PARTITION,
    qos: str = DEFAULT_QOS,
    cpus: int = DEFAULT_CPUS,
    memory: str = DEFAULT_MEMORY,
    walltime: str = DEFAULT_WALLTIME,
    backend: str = DEFAULT_BACKEND,
    accelerator: str = DEFAULT_ACCELERATOR,
    runtime_python: str | None = None,
    replay_campaign_root: Path | None = None,
    hamlib_export_dir: Path | None = None,
    hamlib_report_dir: Path | None = None,
    reactions_export_dir: Path | None = None,
    reactions_report_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one idempotent collect/reconcile/submit/mutate cycle."""

    campaign_root = campaign_root.resolve()
    repo_dir = repo_dir.resolve()
    _, queue_dir, lane_dirs = _campaign_paths(campaign_root)
    submit_options = {
        "repo_dir": repo_dir,
        "host": host,
        "remote_root": remote_root,
        "account": account,
        "partition": partition,
        "qos": qos,
        "cpus": cpus,
        "memory": memory,
        "walltime": walltime,
        "backend": backend,
        "accelerator": accelerator,
        "runtime_python": runtime_python,
    }

    batch_rows: list[dict[str, Any]] = []
    active_jobs = 0
    stranded_batches: list[Path] = []
    for batch_dir in sorted((queue_dir / "batches").glob("*")):
        if not batch_dir.is_dir():
            continue
        if (batch_dir / "collected.json").is_file():
            marker = _read_json(batch_dir / "collected.json")
            batch_rows.append(
                {
                    "batch_id": batch_dir.name,
                    "state": "COLLECTED",
                    "slurm_state": marker.get("slurm_state"),
                    "tasks": _batch_task_count(batch_dir),
                }
            )
            continue
        submission_path = batch_dir / "submission.json"
        if not submission_path.is_file() or not _read_json(submission_path).get("job_id"):
            stranded_batches.append(batch_dir)
            batch_rows.append(
                {
                    "batch_id": batch_dir.name,
                    "state": "PREPARED",
                    "tasks": _batch_task_count(batch_dir),
                }
            )
            continue
        try:
            remote = remote_status(batch_dir)
            state = slurm_state(remote)
            job_id = remote["job_id"]
        except Exception as exc:
            state = "QUERY_ERROR"
            job_id = _read_json(submission_path).get("job_id")
            _append_event(
                campaign_root,
                "batch_status_error",
                batch_id=batch_dir.name,
                job_id=job_id,
                error=_safe_error(exc),
            )
        if state in TERMINAL_SLURM_STATES:
            try:
                _collect_terminal_batch(
                    campaign_root=campaign_root,
                    queue_dir=queue_dir,
                    batch_dir=batch_dir,
                    state=state,
                    max_batch_attempts=max_batch_attempts,
                )
            except Exception as exc:
                _append_event(
                    campaign_root,
                    "batch_collection_error",
                    batch_id=batch_dir.name,
                    job_id=job_id,
                    slurm_state=state,
                    error=_safe_error(exc),
                )
            batch_rows.append(
                {
                    "batch_id": batch_dir.name,
                    "job_id": job_id,
                    "state": state,
                    "tasks": _batch_task_count(batch_dir),
                }
            )
        else:
            active_jobs += 1
            batch_rows.append(
                {
                    "batch_id": batch_dir.name,
                    "job_id": job_id,
                    "state": state,
                    "tasks": _batch_task_count(batch_dir),
                }
            )

    reconciliation_errors = 0
    for lane_dir in lane_dirs:
        try:
            result = reconcile_lane(lane_dir)
            if result["status"] in {
                "reference_ready",
                "baseline_ready",
                "accepted",
                "rejected",
            }:
                _append_event(
                    campaign_root,
                    "lane_reconciled",
                    lane=str(lane_dir.relative_to(campaign_root)),
                    status=result["status"],
                    iteration=result.get("iteration"),
                    score=result.get("candidate_score", result.get("score")),
                    accepted=result.get("accepted"),
                )
        except Exception as exc:
            reconciliation_errors += 1
            state_path = lane_dir / "state.json"
            state = _read_json(state_path)
            error = _safe_error(exc)
            if state.get("last_error") != error:
                state["last_error"] = error
                state["updated_at"] = _now()
                _write_json(state_path, state)
                _append_event(
                    campaign_root,
                    "lane_reconcile_error",
                    lane=str(lane_dir.relative_to(campaign_root)),
                    error=error,
                )

    lane_metrics, ready_lanes = _lane_summary(lane_dirs)

    for batch_dir in stranded_batches:
        if active_jobs >= max_active_jobs:
            break
        try:
            submitted = submit_batch(batch_dir, **submit_options)
            active_jobs += 1
            _append_event(
                campaign_root,
                "batch_submitted",
                batch_id=batch_dir.name,
                job_id=submitted["job_id"],
                tasks=_batch_task_count(batch_dir),
                recovered=bool(submitted.get("recovered_submission")),
                resources=submitted["resources"],
            )
        except Exception as exc:
            _append_event(
                campaign_root,
                "batch_submission_error",
                batch_id=batch_dir.name,
                error=_safe_error(exc),
            )

    while active_jobs < max_active_jobs:
        current_queue = queue_status(queue_dir)
        pending = int(current_queue["pending"])
        if pending == 0:
            break
        force_partial = pending < threshold and (
            not ready_lanes
            or (replay_campaign_root is not None and active_jobs == 0)
        )
        if pending < threshold and not force_partial:
            break
        try:
            submitted = submit_ready(
                queue_dir,
                threshold=threshold,
                force=force_partial,
                **submit_options,
            )
        except Exception as exc:
            _append_event(
                campaign_root,
                "batch_submission_error",
                batch_id=None,
                error=_safe_error(exc),
            )
            break
        if submitted.get("status") != "submitted":
            break
        active_jobs += 1
        _append_event(
            campaign_root,
            "batch_submitted",
            batch_id=submitted["batch_id"],
            job_id=submitted["job_id"],
            tasks=submitted["task_count"],
            forced_partial=force_partial,
            resources=submitted["resources"],
        )

    mutation: dict[str, Any] | None = None
    mutations: list[dict[str, Any]] = []
    if replay_campaign_root is not None and ready_lanes:
        replay_campaign_root = replay_campaign_root.expanduser().resolve()
        for lane_dir in ready_lanes:
            canonical_lane = replay_campaign_root / lane_dir.relative_to(campaign_root)
            try:
                mutation = propose_replay_lane(lane_dir, canonical_lane)
            except Exception as exc:
                mutation = {
                    "status": "error",
                    "lane": str(lane_dir.relative_to(campaign_root)),
                    "error": _safe_error(exc),
                }
                _append_event(
                    campaign_root,
                    "replay_error",
                    lane=str(lane_dir.relative_to(campaign_root)),
                    error=mutation["error"],
                )
                break
            if mutation["status"] == "replay_proposed":
                _append_event(
                    campaign_root,
                    "candidate_replayed",
                    lane=str(lane_dir.relative_to(campaign_root)),
                    iteration=mutation["iteration"],
                    task_id=mutation["task_id"],
                    canonical_candidate_sha256=mutation[
                        "canonical_candidate_sha256"
                    ],
                )
                break
        if mutation is None:
            mutation = {"status": "waiting_canonical"}
    elif mutations_enabled and ready_lanes:
        for lane_dir, result in _run_mutation_proposals(
            ready_lanes,
            workers=mutation_workers,
            timeout_seconds=mutation_timeout_seconds,
        ):
            mutation = {
                **result,
                "lane": str(lane_dir.relative_to(campaign_root)),
            }
            mutations.append(mutation)
            if mutation["status"] == "error":
                _append_event(
                    campaign_root,
                    "mutation_error",
                    lane=mutation["lane"],
                    error=mutation["error"],
                )
            else:
                _append_event(
                    campaign_root,
                    "mutation_proposed",
                    lane=mutation["lane"],
                    iteration=mutation.get("iteration"),
                    task_id=mutation.get("task_id"),
                    queue_status=mutation.get("queue_status"),
                )

    lane_metrics, _ = _lane_summary(lane_dirs)
    status = {
        "schema": 1,
        "updated_at": _now(),
        "campaign_root": str(campaign_root),
        "repo_dir": str(repo_dir),
        "queue": queue_status(queue_dir),
        "lanes": lane_metrics,
        "batches": batch_rows,
        "active_jobs": active_jobs,
        "reconciliation_errors": reconciliation_errors,
        "last_mutation": mutation,
        "last_mutations": mutations,
        "configuration": {
            "threshold": threshold,
            "max_active_jobs": max_active_jobs,
            "mutation_workers": mutation_workers,
            "mutation_timeout_seconds": mutation_timeout_seconds,
            "max_batch_attempts": max_batch_attempts,
            "mutations_enabled": mutations_enabled,
            "host": host,
            "remote_root": remote_root,
            "account": account,
            "partition": partition,
            "qos": qos,
            "cpus": cpus,
            "memory": memory,
            "walltime": walltime,
            "backend": backend,
            "accelerator": accelerator,
            "runtime_python": runtime_python,
            "replay_campaign_root": (
                str(replay_campaign_root) if replay_campaign_root else None
            ),
            "hamlib_export_dir": str(hamlib_export_dir) if hamlib_export_dir else None,
            "hamlib_report_dir": str(hamlib_report_dir) if hamlib_report_dir else None,
            "reactions_export_dir": (
                str(reactions_export_dir) if reactions_export_dir else None
            ),
            "reactions_report_dir": (
                str(reactions_report_dir) if reactions_report_dir else None
            ),
        },
    }
    _write_json(campaign_root / "supervisor" / "status.json", status)
    if lane_metrics["by_status"].get("complete", 0) == lane_metrics["total"]:
        complete_path = campaign_root / "campaign_complete.json"
        if not complete_path.is_file():
            _write_json(
                complete_path,
                {
                    "schema": 1,
                    "completed_at": _now(),
                    "lanes": lane_metrics,
                },
            )
            _append_event(campaign_root, "campaign_complete", lanes=lane_metrics["total"])
        final_paths = (
            hamlib_export_dir,
            hamlib_report_dir,
            reactions_export_dir,
            reactions_report_dir,
        )
        if any(path is not None for path in final_paths):
            if not all(path is not None for path in final_paths):
                raise ValueError("all four final export/report directories must be configured")
            finalization_path = campaign_root / "supervisor" / "finalization.json"
            if not finalization_path.is_file() and active_jobs == 0:
                try:
                    finalization = _finalize_campaign(
                        campaign_root,
                        hamlib_export_dir=hamlib_export_dir,
                        hamlib_report_dir=hamlib_report_dir,
                        reactions_export_dir=reactions_export_dir,
                        reactions_report_dir=reactions_report_dir,
                    )
                    status["finalization"] = finalization
                    _append_event(campaign_root, "campaign_finalized")
                except Exception as exc:
                    status["finalization"] = {
                        "status": "error",
                        "error": _safe_error(exc),
                    }
                    _append_event(
                        campaign_root,
                        "campaign_finalization_error",
                        error=_safe_error(exc),
                    )
            elif finalization_path.is_file():
                status["finalization"] = _read_json(finalization_path)
            _write_json(campaign_root / "supervisor" / "status.json", status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--threshold", type=int, default=40)
    parser.add_argument("--max-active-jobs", type=int, default=4)
    parser.add_argument("--mutation-workers", type=int, default=1)
    parser.add_argument("--mutation-timeout-seconds", type=int, default=900)
    parser.add_argument("--max-batch-attempts", type=int, default=3)
    parser.add_argument("--no-mutations", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--qos", default=DEFAULT_QOS)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument(
        "--backend",
        choices=("cupynumeric", "numpy"),
        default=DEFAULT_BACKEND,
    )
    parser.add_argument(
        "--accelerator",
        choices=("gpu", "cpu"),
        default=DEFAULT_ACCELERATOR,
    )
    parser.add_argument("--runtime-python")
    parser.add_argument("--replay-campaign-root", type=Path)
    parser.add_argument("--final-hamlib-export-dir", type=Path)
    parser.add_argument("--final-hamlib-report-dir", type=Path)
    parser.add_argument("--final-reactions-export-dir", type=Path)
    parser.add_argument("--final-reactions-report-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")
    if (
        args.threshold <= 0
        or args.max_active_jobs <= 0
        or args.mutation_workers <= 0
        or args.max_batch_attempts <= 0
    ):
        raise ValueError(
            "threshold, max-active-jobs, mutation-workers, and max-batch-attempts "
            "must be positive"
        )
    final_paths = (
        args.final_hamlib_export_dir,
        args.final_hamlib_report_dir,
        args.final_reactions_export_dir,
        args.final_reactions_report_dir,
    )
    if any(path is not None for path in final_paths) and not all(
        path is not None for path in final_paths
    ):
        raise ValueError("all four final export/report directories must be configured")
    options = {
        "repo_dir": args.repo_dir,
        "threshold": args.threshold,
        "max_active_jobs": args.max_active_jobs,
        "mutation_workers": args.mutation_workers,
        "mutation_timeout_seconds": args.mutation_timeout_seconds,
        "max_batch_attempts": args.max_batch_attempts,
        "mutations_enabled": not args.no_mutations,
        "host": args.host,
        "remote_root": args.remote_root,
        "account": args.account,
        "partition": args.partition,
        "qos": args.qos,
        "cpus": args.cpus,
        "memory": args.memory,
        "walltime": args.walltime,
        "backend": args.backend,
        "accelerator": args.accelerator,
        "runtime_python": args.runtime_python,
        "replay_campaign_root": args.replay_campaign_root,
        "hamlib_export_dir": args.final_hamlib_export_dir,
        "hamlib_report_dir": args.final_hamlib_report_dir,
        "reactions_export_dir": args.final_reactions_export_dir,
        "reactions_report_dir": args.final_reactions_report_dir,
    }
    with _campaign_lock(args.campaign_root.resolve()):
        while True:
            try:
                result = supervise_once(args.campaign_root, **options)
                print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as exc:
                _append_event(
                    args.campaign_root.resolve(),
                    "supervisor_cycle_error",
                    error=_safe_error(exc),
                )
                if args.once:
                    raise
            if args.once:
                return 0
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
