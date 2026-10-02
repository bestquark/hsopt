"""Collect finished tasks without declaring a still-running Slurm batch terminal."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..campaign.queue import merge_batch


def collect_running_results(queue_dir: Path, batch_dir: Path) -> dict[str, Any]:
    """Stage, validate and merge complete result JSONs; leave live work untouched.

    The campaign supervisor's exclusive lock must be held by the caller. Never
    download batch.json, mutable telemetry, or stdout here. In particular, this
    function does not requeue missing tasks or write a collected/terminal marker.
    """
    queue_dir, batch_dir = queue_dir.resolve(), batch_dir.resolve()
    if batch_dir.parent != queue_dir / "batches":
        raise ValueError("live batch must belong to this queue")
    manifest = json.loads((batch_dir / "batch.json").read_text())
    submission = json.loads((batch_dir / "submission.json").read_text())
    task_ids = manifest["task_ids"]
    if len(task_ids) != len(set(task_ids)) or any(
        re.fullmatch(r"[0-9a-f]{24}", task_id) is None for task_id in task_ids
    ):
        raise ValueError("invalid batch task IDs")
    results_dir = batch_dir / "results"
    results_dir.mkdir(exist_ok=True)
    deferred, newly_merged = [], []
    with tempfile.TemporaryDirectory(prefix=".live-results-", dir=batch_dir) as temporary:
        staging = Path(temporary)
        subprocess.run(
            [
                "rsync",
                "-az",
                "--protect-args",
                "--timeout=15",
                "-e",
                "ssh -S none -o BatchMode=yes -o ConnectTimeout=8",
                "--include=*.json",
                "--exclude=*",
                f"{submission['host']}:{submission['remote_batch']}/results/",
                str(staging) + "/",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        admitted = []
        for task_id in task_ids:
            source = staging / f"{task_id}.json"
            if source.is_symlink():
                raise ValueError(f"refusing a symlink result for {task_id}")
            if not source.is_file():
                continue
            payload = source.read_bytes()
            try:
                result = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                deferred.append(task_id)  # The remote writer may still be writing.
                continue
            if not isinstance(result, dict) or not result.get("task_finished_at"):
                deferred.append(task_id)
                continue
            task = batch_dir / "tasks" / task_id
            resolved = [queue_dir / phase / task_id for phase in ("completed", "failed")]
            metadata_dir = (
                task if task.is_dir() else next((path for path in resolved if path.is_dir()), None)
            )
            if metadata_dir is None:
                raise ValueError(f"missing task metadata for {task_id}")
            metadata = json.loads((metadata_dir / "task.json").read_text())
            expected = {
                "task_id": task_id,
                "evaluation_contract_version": metadata["evaluation_contract_version"],
                "execution_backend": submission["backend"],
                "execution_accelerator": submission["accelerator"],
            }
            if metadata["kind"] == "difference":
                expected["difference_error_contract"] = metadata["difference_error_contract"]
            for key, value in expected.items():
                if result.get(key) != value:
                    raise ValueError(f"live result {task_id} disagrees on {key}")
            elapsed = float(result.get("task_wall_seconds", -1))
            if (
                result.get("status") not in {"ok", "error"}
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError(f"invalid completed result for {task_id}")
            destination = results_dir / source.name
            for archived in [destination, *(path / "result.json" for path in resolved)]:
                if archived.exists() and archived.read_bytes() != payload:
                    raise ValueError(f"refusing to overwrite an archived result for {task_id}")
            admitted.append((source, destination))
            if task.is_dir():
                newly_merged.append(task_id)
        # Validate the entire incoming set before changing authoritative history.
        for source, destination in admitted:
            if not destination.exists():
                source.replace(destination)
        merged = merge_batch(queue_dir, batch_dir)
    return {
        **merged,
        "job_id": submission["job_id"],
        "newly_merged": newly_merged,
        "deferred_incomplete": deferred,
    }
