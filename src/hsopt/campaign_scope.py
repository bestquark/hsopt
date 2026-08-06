"""Narrow an existing exact campaign without deleting retired lane histories."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lock_nonblocking(stack: ExitStack, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = stack.enter_context(path.open("a+", encoding="utf-8"))
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"campaign is active through {path}; stop its supervisor and local workers first"
        ) from exc


def _task_lane(task_dir: Path, campaign_root: Path) -> str | None:
    task_path = task_dir / "task.json"
    if not task_path.is_file():
        return None
    original = _read_json(task_path).get("policy_original")
    if not isinstance(original, str):
        return None
    try:
        relative = Path(original).expanduser().resolve().relative_to(campaign_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 4 or parts[0] != "lanes":
        return None
    return Path(*parts[:4]).as_posix()


def _archive(source: Path, destination_root: Path) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(source, destination)
    return destination


def _archive_retired_queue_work(
    campaign_root: Path,
    *,
    retired_lanes: set[str],
    archive_root: Path,
) -> list[dict[str, str]]:
    queue_root = campaign_root / "queue"
    planned: list[tuple[Path, Path, str]] = []
    for phase in ("pending", "running"):
        phase_root = queue_root / phase
        for task_dir in sorted(phase_root.glob("*")) if phase_root.is_dir() else ():
            if task_dir.is_dir() and _task_lane(task_dir, campaign_root) in retired_lanes:
                planned.append((task_dir, archive_root / phase, phase))

    batches_root = queue_root / "batches"
    for batch_dir in sorted(batches_root.glob("*")) if batches_root.is_dir() else ():
        if not batch_dir.is_dir():
            continue
        marker_path = batch_dir / "collected.json"
        marker = _read_json(marker_path) if marker_path.is_file() else {}
        if marker and marker.get("slurm_state") != "LOCAL_RUNNING":
            continue
        submission_path = batch_dir / "submission.json"
        submission = _read_json(submission_path) if submission_path.is_file() else {}
        if submission.get("job_id"):
            raise RuntimeError(
                f"uncollected Slurm batch {batch_dir} must be cancelled or collected first"
            )
        task_dirs = [path for path in (batch_dir / "tasks").glob("*") if path.is_dir()]
        task_lanes = {_task_lane(path, campaign_root) for path in task_dirs}
        retired = {lane for lane in task_lanes if lane in retired_lanes}
        retained = {lane for lane in task_lanes if lane is not None and lane not in retired_lanes}
        if retired and retained:
            raise RuntimeError(f"batch mixes retained and retired lanes: {batch_dir}")
        if retired:
            planned.append((batch_dir, archive_root / "batches", "batch"))

    archived = []
    for source, destination_root, phase in planned:
        destination = _archive(source, destination_root)
        archived.append(
            {
                "phase": phase,
                "source": str(source.relative_to(campaign_root)),
                "archive": str(destination.relative_to(campaign_root)),
            }
        )
    return archived


def reduce_hamlib_replicates(
    campaign_root: Path,
    *,
    replicates: int = 3,
    reason: str = "reduce production runtime while retaining predetermined replicates",
) -> dict[str, Any]:
    """Keep ``run_01`` through ``run_N`` and retire higher HamLib replicates.

    Lane directories and completed history are never deleted. Pending or
    uncollected local work for retired lanes is moved below the dated scope
    record so it cannot be claimed after the active manifest is narrowed.
    """

    if replicates < 1:
        raise ValueError("at least one HamLib replicate is required")
    campaign_root = campaign_root.expanduser().resolve()
    campaign_path = campaign_root / "campaign.json"
    if not campaign_path.is_file():
        raise FileNotFoundError(campaign_path)

    with ExitStack() as locks:
        _lock_nonblocking(locks, campaign_root / "supervisor" / "supervisor.lock")
        worker_locks = set((campaign_root / "supervisor").glob("local_worker-*.lock"))
        worker_locks.update((campaign_root / "supervisor").glob("local_worker.lock"))
        for worker_lock in sorted(worker_locks):
            _lock_nonblocking(locks, worker_lock)

        campaign = _read_json(campaign_path)
        if int(campaign.get("schema", 0)) != 1:
            raise ValueError("unsupported campaign schema")
        hamlib = campaign.get("hamlib")
        lanes = campaign.get("lanes")
        if not isinstance(hamlib, dict) or not isinstance(lanes, list):
            raise ValueError("campaign manifest lacks HamLib metadata or lanes")
        previous_replicates = int(hamlib.get("replicates", 0))
        if previous_replicates < replicates:
            raise ValueError(
                f"cannot increase an existing campaign from {previous_replicates} to {replicates}"
            )

        retained_names = {f"run_{index:02d}" for index in range(1, replicates + 1)}
        retained_lanes: list[dict[str, Any]] = []
        retired_rows: list[dict[str, Any]] = []
        for row in lanes:
            if not isinstance(row, dict):
                raise ValueError("campaign lane entries must be JSON objects")
            if row.get("family") == "abs_energy" and row.get("replicate") not in retained_names:
                retired_rows.append(row)
            else:
                retained_lanes.append(row)

        if previous_replicates == replicates and not retired_rows:
            return {
                "status": "already_scoped",
                "campaign_root": str(campaign_root),
                "hamlib_replicates": replicates,
                "active_lanes": len(retained_lanes),
            }
        if not retired_rows:
            raise ValueError("campaign has no higher-numbered HamLib replicates to retire")

        retired_lane_dirs = {str(row["lane_dir"]) for row in retired_rows}
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        change_id = f"{stamp}-hamlib-replicates-{replicates}"
        change_root = campaign_root / "scope_changes" / change_id
        change_root.mkdir(parents=True, exist_ok=False)
        before_sha256 = _sha256(campaign_path)
        before_path = change_root / "campaign.before.json"
        _write_json(before_path, campaign)

        archived = _archive_retired_queue_work(
            campaign_root,
            retired_lanes=retired_lane_dirs,
            archive_root=change_root / "queue_archive",
        )
        changed_at = _now()
        previous_changes = campaign.get("scope_changes")
        scope_changes = list(previous_changes) if isinstance(previous_changes, list) else []
        scope_changes.append(
            {
                "change_id": change_id,
                "changed_at": changed_at,
                "reason": reason,
                "hamlib_replicates_before": previous_replicates,
                "hamlib_replicates_after": replicates,
                "retired_lane_count": len(retired_rows),
                "record": str((change_root / "scope_change.json").relative_to(campaign_root)),
            }
        )
        campaign["hamlib"] = {**hamlib, "replicates": replicates}
        campaign["lanes"] = retained_lanes
        campaign["scope_revision"] = int(campaign.get("scope_revision", 0)) + 1
        campaign["scope_updated_at"] = changed_at
        campaign["scope_changes"] = scope_changes
        _write_json(campaign_path, campaign)

        record = {
            "schema": 1,
            "status": "applied",
            "change_id": change_id,
            "changed_at": changed_at,
            "reason": reason,
            "selection_rule": f"retain run_01 through run_{replicates:02d} before analysis",
            "campaign_root": str(campaign_root),
            "campaign_before": str(before_path.relative_to(campaign_root)),
            "campaign_before_sha256": before_sha256,
            "campaign_after_sha256": _sha256(campaign_path),
            "hamlib_replicates_before": previous_replicates,
            "hamlib_replicates_after": replicates,
            "active_lane_count": len(retained_lanes),
            "retired_lane_count": len(retired_rows),
            "retired_lanes": sorted(retired_lane_dirs),
            "archived_queue_work": archived,
            "retired_histories_deleted": False,
        }
        _write_json(change_root / "scope_change.json", record)
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--hamlib-replicates", type=int, default=3)
    parser.add_argument(
        "--reason",
        default="reduce production runtime while retaining predetermined replicates",
    )
    args = parser.parse_args()
    result = reduce_hamlib_replicates(
        args.campaign_root,
        replicates=args.hamlib_replicates,
        reason=args.reason,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
