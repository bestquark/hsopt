"""Portable evaluation requests and durable result-to-mutation handoff.

This module never selects machines, submits jobs, or waits on a scheduler.
The existing file ledger remains the authoritative trajectory store.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..evaluation import compiled_unitary
from .loop import propose_lane, reconcile_lane
from .queue import _enqueue_lock, _find_task, evaluate_task


SCHEMA = "hsopt-evaluation-v1"
TASK_FIELDS = (
    "task_id",
    "stage",
    "kind",
    "source",
    "reference",
    "evaluation_contract_version",
    "difference_error_contract",
    "difference_error_contract_version",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json(value) + b"\n")
    temporary.replace(path)


@contextmanager
def _lane_lock(lane: Path):
    with (lane / ".advance.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _snapshot_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"invalid snapshot file: {relative}")
    return path


def evaluation_request(task_dir: Path) -> dict[str, Any]:
    """Describe and fingerprint a task independently of its execution location."""
    task_dir = task_dir.resolve()
    metadata = _read(task_dir / "task.json")
    source_name = str(metadata["source"])
    names = {source_name}
    if metadata["kind"] == "difference":
        source = _read(_snapshot_path(task_dir, source_name))
        names.update(str(item["problem"]) for item in source["terms"])
    if metadata.get("stage", "candidate") == "candidate":
        names.add("policy.py")
    if metadata.get("reference"):
        names.add(str(metadata["reference"]))
    request = {
        "schema": SCHEMA,
        "task_id": metadata["task_id"],
        "task": {key: metadata.get(key) for key in TASK_FIELDS},
        "files": {
            name: hashlib.sha256(_snapshot_path(task_dir, name).read_bytes()).hexdigest()
            for name in sorted(names)
        },
        "evaluator_sha256": hashlib.sha256(
            b"".join(
                name.encode() + b"\0" + Path(compiled_unitary.__file__).with_name(name).read_bytes()
                for name in ("compiled_unitary.py", "gpu_rotations.py", "gpu_spectrum.py")
            )
        ).hexdigest(),
    }
    request["request_sha256"] = hashlib.sha256(_json(request)).hexdigest()
    return {**request, "task_dir": str(task_dir)}


def run_evaluation(
    request: dict[str, Any],
    *,
    task_dir: Path | None = None,
    backend: str = "numpy",
    accelerator: str = "unknown",
) -> dict[str, Any]:
    """Worker API: verify staged inputs, evaluate, and return a portable receipt."""
    root = task_dir if task_dir is not None else Path(request["task_dir"])
    actual = evaluation_request(root)
    if any(
        actual[key] != request.get(key)
        for key in ("schema", "task_id", "task", "files", "evaluator_sha256", "request_sha256")
    ):
        raise ValueError("staged inputs, task metadata, or evaluator differ from the request")
    result = evaluate_task(root, backend=backend, accelerator=accelerator)
    result["evaluation_request_sha256"] = actual["request_sha256"]
    result["evaluator_sha256"] = actual["evaluator_sha256"]
    return {
        "schema": SCHEMA,
        "task_id": actual["task_id"],
        "request_sha256": actual["request_sha256"],
        "evaluator_sha256": actual["evaluator_sha256"],
        "execution": {"host": socket.gethostname(), "python": sys.version.split()[0]},
        "result": result,
    }


def _pending_request(lane: Path) -> dict[str, Any] | None:
    state = _read(lane / "state.json")
    if state.get("pending_task_id") is None:
        return None
    queue = Path(_read(lane / "manifest.json")["queue_dir"])
    with _enqueue_lock(queue):
        found = _find_task(queue, state["pending_task_id"])
        if found is None:
            raise FileNotFoundError(f"missing task {state['pending_task_id']}")
        path = found[1] / "evaluation_request.json"
        actual = evaluation_request(found[1])
        if path.exists():
            request = _read(path)
            if request["task"] != actual["task"] or request["files"] != actual["files"]:
                raise ValueError("a previously requested task snapshot has changed")
            request["task_dir"] = actual["task_dir"]
        else:
            request = actual
            _write(path, request)
    associations_path = lane / "evaluation_requests.json"
    associations = _read(associations_path) if associations_path.exists() else {}
    associations[request["task_id"]] = request["request_sha256"]
    _write(associations_path, associations)
    return request


def _advance(lane: Path, *, mutate: bool, timeout_seconds: int) -> dict[str, Any]:
    reconciled = reconcile_lane(lane)
    state = _read(lane / "state.json")
    next_state = (
        propose_lane(lane, timeout_seconds=timeout_seconds)
        if mutate and state.get("pending_kind") != "replay_validation_failed"
        else reconciled
    )
    request = _pending_request(lane)
    return {"lane_dir": str(lane), "state": next_state, "request": request}


def advance_lane(
    lane_dir: Path,
    *,
    mutate: bool = True,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Reconcile feedback and propose at most one next mutation under a lane lock."""
    lane = lane_dir.resolve()
    with _lane_lock(lane):
        return _advance(lane, mutate=mutate, timeout_seconds=timeout_seconds)


def deliver_result(
    lane_dir: Path,
    receipt: dict[str, Any],
    *,
    mutate: bool = True,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Record once, apply acceptance, and wake the lane for its next proposal.

    Retried identical deliveries are harmless; conflicting or foreign results
    cannot overwrite a recorded trajectory. The result is durable before the
    agent is invoked, so an interrupted wake-up can be retried with advance.
    """
    lane = lane_dir.resolve()
    with _lane_lock(lane):
        manifest = _read(lane / "manifest.json")
        queue = Path(manifest["queue_dir"]).resolve()
        task_id = receipt.get("task_id")
        if receipt.get("schema") != SCHEMA or not isinstance(task_id, str):
            raise ValueError("unsupported evaluation receipt")
        if len(task_id) != 24 or any(c not in "0123456789abcdef" for c in task_id):
            raise ValueError("invalid task id")
        associations_path = lane / "evaluation_requests.json"
        associations = _read(associations_path) if associations_path.exists() else {}
        if associations.get(task_id) != receipt.get("request_sha256"):
            raise ValueError("receipt does not belong to a request issued by this lane")
        with _enqueue_lock(queue):
            found = _find_task(queue, task_id)
            if found is None:
                raise ValueError("unknown evaluation task")
            phase, task = found
            expected = _read(task / "evaluation_request.json")
            actual = evaluation_request(task)
            if expected["task"] != actual["task"] or expected["files"] != actual["files"]:
                raise ValueError("the requested task snapshot has changed")
            for key in ("request_sha256", "evaluator_sha256"):
                if receipt.get(key) != expected[key]:
                    raise ValueError(f"evaluation receipt has a different {key}")
            result = receipt.get("result")
            if not isinstance(result, dict) or result.get("task_id") != task_id:
                raise ValueError("result task id does not match the receipt")
            if result.get("status") not in {"ok", "error"}:
                raise ValueError("result must be a completed evaluation or an explicit error")
            for key, expected_value in (
                ("evaluation_request_sha256", expected["request_sha256"]),
                ("evaluator_sha256", expected["evaluator_sha256"]),
            ):
                if result.get(key) != expected_value:
                    raise ValueError(f"result has a different {key}")
            for key in (
                "evaluation_contract_version",
                "difference_error_contract",
                "difference_error_contract_version",
            ):
                if result.get(key) != expected["task"][key]:
                    raise ValueError(f"result has a different {key}")
            _json(result)  # reject non-finite or non-JSON metrics before writing
            receipt_path = task / "evaluation_receipt.json"
            if phase in {"completed", "failed"}:
                if not receipt_path.exists() or _json(_read(receipt_path)) != _json(receipt):
                    raise ValueError("conflicting result for an already completed task")
                recorded = False
            else:
                if _read(lane / "state.json")["pending_task_id"] != task_id:
                    raise ValueError("result does not belong to this lane's pending evaluation")
                _write(task / "result.json", result)
                _write(receipt_path, receipt)
                destination = queue / ("completed" if result["status"] == "ok" else "failed")
                destination.mkdir(exist_ok=True)
                task.rename(destination / task_id)
                recorded = True
        return {
            "recorded": recorded,
            "task_id": task_id,
            **_advance(lane, mutate=mutate, timeout_seconds=timeout_seconds),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("advance", "deliver"):
        p = commands.add_parser(name)
        p.add_argument("--lane", type=Path, required=True)
        p.add_argument("--no-mutate", action="store_true")
        p.add_argument("--timeout-seconds", type=int, default=900)
        if name == "deliver":
            p.add_argument("--receipt", type=Path, required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--task-dir", type=Path)
    p.add_argument(
        "--backend", choices=("numpy", "cupynumeric", "cupy-lanczos"), default="cupy-lanczos"
    )
    p.add_argument("--accelerator", default="unknown")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = run_evaluation(
            _read(args.request),
            task_dir=args.task_dir,
            backend=args.backend,
            accelerator=args.accelerator,
        )
        _write(args.output, result)
    else:
        kwargs = {"mutate": not args.no_mutate, "timeout_seconds": args.timeout_seconds}
        result = (
            deliver_result(args.lane, _read(args.receipt), **kwargs)
            if args.command == "deliver"
            else advance_lane(args.lane, **kwargs)
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
