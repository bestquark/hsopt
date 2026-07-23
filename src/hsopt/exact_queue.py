"""Content-addressed queue for exact compiled-unitary evaluations.

Each queued task is self-contained: it snapshots the candidate policy and all
Hamiltonian JSON needed by the evaluator.  A batch evaluates tasks sequentially
so only one dense unitary is resident at a time.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compiled_unitary import (
    evaluate_difference,
    evaluate_exact_difference_reference,
    evaluate_exact_reference,
    evaluate_molecule,
)


QUEUE_PHASES = ("pending", "completed", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def _enqueue_lock(queue_dir: Path):
    """Serialize content-addressed insertion for concurrent mutation workers."""

    with (queue_dir / ".enqueue.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _source_files(kind: str, source: Path) -> list[Path]:
    source = source.resolve()
    if kind == "absolute":
        return [source]
    if kind != "difference":
        raise ValueError("kind must be 'absolute' or 'difference'")
    reaction = json.loads(source.read_text(encoding="utf-8"))
    return [source, *[(source.parent / item["problem"]).resolve() for item in reaction["terms"]]]


def _task_id(
    kind: str,
    source: Path,
    *,
    stage: str,
    policy: Path | None = None,
    reference: Path | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"hsopt-exact-task-v2\0")
    digest.update(stage.encode())
    digest.update(kind.encode())
    for path in _source_files(kind, source):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if policy is not None:
        digest.update(policy.resolve().read_bytes())
    if reference is not None:
        digest.update(reference.resolve().read_bytes())
    return digest.hexdigest()[:24]


def _find_task(queue_dir: Path, task_id: str) -> tuple[str, Path] | None:
    for phase in QUEUE_PHASES:
        path = queue_dir / phase / task_id
        if path.is_dir():
            return phase, path
    for batch in sorted((queue_dir / "batches").glob("*")):
        path = batch / "tasks" / task_id
        if path.is_dir():
            return "batched", path
    return None


def _snapshot_source(kind: str, source: Path, task_dir: Path) -> str:
    source = source.resolve()
    if kind == "absolute":
        shutil.copyfile(source, task_dir / "problem.json")
        return "problem.json"

    reaction = json.loads(source.read_text(encoding="utf-8"))
    problems_dir = task_dir / "problems"
    problems_dir.mkdir()
    for index, item in enumerate(reaction["terms"]):
        original = (source.parent / str(item["problem"])).resolve()
        snapshot_name = f"{index:03d}-{original.name}"
        shutil.copyfile(original, problems_dir / snapshot_name)
        item["problem"] = f"problems/{snapshot_name}"
    _write_json(task_dir / "reaction.json", reaction)
    return "reaction.json"


def enqueue(
    queue_dir: Path,
    *,
    kind: str,
    source: Path,
    policy: Path,
    reference: Path | None = None,
) -> dict[str, Any]:
    """Snapshot one candidate into the pending queue, deduplicated by content."""

    queue_dir = queue_dir.resolve()
    policy = policy.resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    if reference is not None:
        reference = reference.resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
    for phase in (*QUEUE_PHASES, "batches"):
        (queue_dir / phase).mkdir(parents=True, exist_ok=True)

    task_id = _task_id(
        kind,
        source,
        stage="candidate",
        policy=policy,
        reference=reference,
    )
    with _enqueue_lock(queue_dir):
        existing = _find_task(queue_dir, task_id)
        if existing is not None:
            phase, path = existing
            return {
                "status": "exists",
                "phase": phase,
                "task_id": task_id,
                "path": str(path),
            }

        temporary = queue_dir / f".enqueue-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            source_name = _snapshot_source(kind, source, temporary)
            shutil.copyfile(policy, temporary / "policy.py")
            if reference is not None:
                shutil.copyfile(reference, temporary / "reference.json")
            metadata = {
                "schema": 1,
                "task_id": task_id,
                "stage": "candidate",
                "kind": kind,
                "source": source_name,
                "source_original": str(source.resolve()),
                "policy_original": str(policy),
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "reference": "reference.json" if reference is not None else None,
                "reference_original": str(reference) if reference is not None else None,
                "reference_sha256": (
                    hashlib.sha256(reference.read_bytes()).hexdigest()
                    if reference is not None
                    else None
                ),
                "queued_at": _now(),
                "batch_attempts": 0,
            }
            _write_json(temporary / "task.json", metadata)
            destination = queue_dir / "pending" / task_id
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {"status": "queued", "phase": "pending", "task_id": task_id, "path": str(destination)}


def enqueue_reference(
    queue_dir: Path,
    *,
    kind: str,
    source: Path,
) -> dict[str, Any]:
    """Snapshot one immutable Hamiltonian reference task, deduplicated by content."""

    queue_dir = queue_dir.resolve()
    source = source.resolve()
    for phase in (*QUEUE_PHASES, "batches"):
        (queue_dir / phase).mkdir(parents=True, exist_ok=True)
    task_id = _task_id(kind, source, stage="reference")
    with _enqueue_lock(queue_dir):
        existing = _find_task(queue_dir, task_id)
        if existing is not None:
            phase, path = existing
            return {
                "status": "exists",
                "phase": phase,
                "task_id": task_id,
                "path": str(path),
            }

        temporary = queue_dir / f".enqueue-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            source_name = _snapshot_source(kind, source, temporary)
            metadata = {
                "schema": 1,
                "task_id": task_id,
                "stage": "reference",
                "kind": kind,
                "source": source_name,
                "source_original": str(source),
                "queued_at": _now(),
                "batch_attempts": 0,
            }
            _write_json(temporary / "task.json", metadata)
            destination = queue_dir / "pending" / task_id
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {"status": "queued", "phase": "pending", "task_id": task_id, "path": str(destination)}


def queue_status(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.resolve()
    counts = {
        phase: len([path for path in (queue_dir / phase).glob("*") if path.is_dir()])
        for phase in QUEUE_PHASES
    }
    batches = [path for path in (queue_dir / "batches").glob("*") if path.is_dir()]
    counts["batches"] = len(batches)
    counts["batched_tasks"] = sum(
        len([path for path in (batch / "tasks").glob("*") if path.is_dir()]) for batch in batches
    )
    return {"queue_dir": str(queue_dir), **counts}


def prepare_batch(queue_dir: Path, *, limit: int = 40) -> dict[str, Any]:
    """Move up to ``limit`` pending tasks into an immutable batch directory."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    queue_dir = queue_dir.resolve()
    pending = sorted(path for path in (queue_dir / "pending").glob("*") if path.is_dir())
    selected = pending[:limit]
    if not selected:
        raise RuntimeError("the queue has no pending tasks")
    batch_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    batch_dir = queue_dir / "batches" / batch_id
    tasks_dir = batch_dir / "tasks"
    results_dir = batch_dir / "results"
    tasks_dir.mkdir(parents=True)
    results_dir.mkdir()
    for task in selected:
        destination = tasks_dir / task.name
        task.rename(destination)
        metadata_path = destination / "task.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["batch_attempts"] = int(metadata.get("batch_attempts", 0)) + 1
        _write_json(metadata_path, metadata)
    manifest = {
        "schema": 1,
        "batch_id": batch_id,
        "created_at": _now(),
        "task_ids": [task.name for task in selected],
        "result_backend": None,
    }
    _write_json(batch_dir / "batch.json", manifest)
    return {
        "status": "prepared",
        "batch_id": batch_id,
        "batch_dir": str(batch_dir),
        "task_count": len(selected),
    }


def run_batch(
    batch_dir: Path,
    *,
    backend: str = "cupynumeric",
    accelerator: str = "unknown",
) -> dict[str, Any]:
    """Evaluate every task in one batch, sequentially."""

    batch_dir = batch_dir.resolve()
    manifest_path = batch_dir / "batch.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_dir = batch_dir / "results"
    results_dir.mkdir(exist_ok=True)
    succeeded = 0
    failed = 0
    for task_id in manifest["task_ids"]:
        task_dir = batch_dir / "tasks" / task_id
        result_path = results_dir / f"{task_id}.json"
        if result_path.is_file():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            succeeded += previous.get("status") == "ok"
            failed += previous.get("status") != "ok"
            continue
        metadata = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        task_started_at = _now()
        task_started = time.perf_counter()
        try:
            source = task_dir / metadata["source"]
            if metadata.get("stage", "candidate") == "reference":
                if metadata["kind"] == "absolute":
                    result = evaluate_exact_reference(source, backend=backend)
                else:
                    result = evaluate_exact_difference_reference(source, backend=backend)
            else:
                policy = task_dir / "policy.py"
                reference = (
                    task_dir / str(metadata["reference"]) if metadata.get("reference") else None
                )
                if metadata["kind"] == "absolute":
                    result = evaluate_molecule(
                        source,
                        policy,
                        backend=backend,
                        reference_path=reference,
                    )
                else:
                    result = evaluate_difference(
                        source,
                        policy,
                        backend=backend,
                        reference_path=reference,
                    )
            result["task_id"] = task_id
            succeeded += 1
        except Exception as exc:  # keep the rest of the batch runnable
            result = {
                "status": "error",
                "task_id": task_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failed += 1
        result.update(
            {
                "task_started_at": task_started_at,
                "task_finished_at": _now(),
                "task_wall_seconds": time.perf_counter() - task_started,
                "execution_backend": backend,
                "execution_accelerator": accelerator,
            }
        )
        _write_json(result_path, result)

    manifest["finished_at"] = _now()
    manifest["result_backend"] = backend
    manifest["result_accelerator"] = accelerator
    manifest["succeeded"] = succeeded
    manifest["failed"] = failed
    _write_json(manifest_path, manifest)
    return {
        "status": "finished",
        "batch_id": manifest["batch_id"],
        "task_count": len(manifest["task_ids"]),
        "succeeded": succeeded,
        "failed": failed,
        "backend": backend,
        "accelerator": accelerator,
    }


def merge_batch(queue_dir: Path, batch_dir: Path) -> dict[str, Any]:
    """Move evaluated tasks into the queue's completed or failed history."""

    queue_dir = queue_dir.resolve()
    batch_dir = batch_dir.resolve()
    manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    merged = 0
    missing = []
    for task_id in manifest["task_ids"]:
        result_path = batch_dir / "results" / f"{task_id}.json"
        if not result_path.is_file():
            missing.append(task_id)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        phase = "completed" if result.get("status") == "ok" else "failed"
        destination = queue_dir / phase / task_id
        if not destination.exists():
            (batch_dir / "tasks" / task_id).rename(destination)
        shutil.copyfile(result_path, destination / "result.json")
        merged += 1
    return {
        "status": "merged" if not missing else "partial",
        "batch_id": manifest["batch_id"],
        "merged": merged,
        "missing": missing,
    }


def requeue_missing(
    queue_dir: Path,
    batch_dir: Path,
    *,
    reason: str,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Return tasks without result files to pending after a terminal batch failure."""

    queue_dir = queue_dir.resolve()
    batch_dir = batch_dir.resolve()
    manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    pending_dir = queue_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    requeued: list[str] = []
    exhausted: list[str] = []
    already_resolved: list[str] = []
    unavailable: list[str] = []
    for task_id in manifest["task_ids"]:
        if (batch_dir / "results" / f"{task_id}.json").is_file():
            already_resolved.append(task_id)
            continue
        task_dir = batch_dir / "tasks" / task_id
        destination = pending_dir / task_id
        if destination.is_dir():
            already_resolved.append(task_id)
        elif task_dir.is_dir():
            metadata = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
            attempts = int(metadata.get("batch_attempts", 1))
            if attempts >= max_attempts:
                failed_dir = queue_dir / "failed" / task_id
                task_dir.rename(failed_dir)
                _write_json(
                    failed_dir / "result.json",
                    {
                        "status": "error",
                        "task_id": task_id,
                        "error_type": "BatchAttemptsExhausted",
                        "error": (
                            f"no evaluator result after {attempts} terminal batch attempts; "
                            f"last reason: {reason}"
                        ),
                    },
                )
                exhausted.append(task_id)
            else:
                task_dir.rename(destination)
                requeued.append(task_id)
        elif any((queue_dir / phase / task_id).is_dir() for phase in QUEUE_PHASES):
            already_resolved.append(task_id)
        else:
            unavailable.append(task_id)

    recovery_path = batch_dir / "recovery.json"
    history = (
        json.loads(recovery_path.read_text(encoding="utf-8"))
        if recovery_path.is_file()
        else {"schema": 1, "attempts": []}
    )
    history["attempts"].append(
        {
            "at": _now(),
            "reason": reason,
            "requeued": requeued,
            "exhausted": exhausted,
            "already_resolved": already_resolved,
            "unavailable": unavailable,
        }
    )
    _write_json(recovery_path, history)
    return {
        "status": "requeued" if not unavailable else "incomplete",
        "batch_id": manifest["batch_id"],
        "requeued": requeued,
        "exhausted": exhausted,
        "already_resolved": already_resolved,
        "unavailable": unavailable,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue_parser = subparsers.add_parser("enqueue")
    enqueue_parser.add_argument("--queue-dir", type=Path, required=True)
    enqueue_parser.add_argument("--kind", choices=("absolute", "difference"), required=True)
    enqueue_parser.add_argument("--source", type=Path, required=True)
    enqueue_parser.add_argument("--policy", type=Path, required=True)
    enqueue_parser.add_argument("--reference", type=Path)

    reference_parser = subparsers.add_parser("enqueue-reference")
    reference_parser.add_argument("--queue-dir", type=Path, required=True)
    reference_parser.add_argument("--kind", choices=("absolute", "difference"), required=True)
    reference_parser.add_argument("--source", type=Path, required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--queue-dir", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare-batch")
    prepare_parser.add_argument("--queue-dir", type=Path, required=True)
    prepare_parser.add_argument("--limit", type=int, default=40)

    run_parser = subparsers.add_parser("run-batch")
    run_parser.add_argument("--batch-dir", type=Path, required=True)
    run_parser.add_argument("--backend", choices=("cupynumeric", "numpy"), default="cupynumeric")
    run_parser.add_argument(
        "--accelerator",
        choices=("gpu", "cpu", "unknown"),
        default="unknown",
    )

    merge_parser = subparsers.add_parser("merge-batch")
    merge_parser.add_argument("--queue-dir", type=Path, required=True)
    merge_parser.add_argument("--batch-dir", type=Path, required=True)

    recover_parser = subparsers.add_parser("recover-batch")
    recover_parser.add_argument("--queue-dir", type=Path, required=True)
    recover_parser.add_argument("--batch-dir", type=Path, required=True)
    recover_parser.add_argument("--reason", required=True)
    recover_parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "enqueue":
        result = enqueue(
            args.queue_dir,
            kind=args.kind,
            source=args.source,
            policy=args.policy,
            reference=args.reference,
        )
    elif args.command == "enqueue-reference":
        result = enqueue_reference(
            args.queue_dir,
            kind=args.kind,
            source=args.source,
        )
    elif args.command == "status":
        result = queue_status(args.queue_dir)
    elif args.command == "prepare-batch":
        result = prepare_batch(args.queue_dir, limit=args.limit)
    elif args.command == "run-batch":
        result = run_batch(
            args.batch_dir,
            backend=args.backend,
            accelerator=args.accelerator,
        )
    elif args.command == "merge-batch":
        result = merge_batch(args.queue_dir, args.batch_dir)
    else:
        result = requeue_missing(
            args.queue_dir,
            args.batch_dir,
            reason=args.reason,
            max_attempts=args.max_attempts,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
