"""Sequential exact-evaluation autoresearch lanes.

The benchmark initializer is copied once and never edited.  Each OpenCode
proposal and evaluator result is retained in an external lane directory, while
the shared exact-evaluation queue batches dense GPU work independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exact_queue import enqueue
from .openshell_agent import mutate


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_result(queue_dir: Path, task_id: str) -> tuple[str, dict[str, Any] | None]:
    for phase in ("completed", "failed"):
        result_path = queue_dir / phase / task_id / "result.json"
        if result_path.is_file():
            return phase, _read_json(result_path)
    return "pending", None


def initialize_lane(
    lane_dir: Path,
    *,
    queue_dir: Path,
    kind: str,
    source: Path,
    initial_policy: Path,
    iterations: int = 50,
    family: str = "custom",
    replicate: str | None = None,
    instance: str | None = None,
    reference_task_id: str | None = None,
) -> dict[str, Any]:
    """Create one reproducible lane, waiting for its shared exact reference first."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    lane_dir = lane_dir.resolve()
    queue_dir = queue_dir.resolve()
    source = source.resolve()
    initial_policy = initial_policy.resolve()
    if lane_dir.exists() and any(lane_dir.iterdir()):
        raise RuntimeError(f"lane directory must be empty: {lane_dir}")
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / "iterations").mkdir()
    shutil.copyfile(initial_policy, lane_dir / "initial_pf.py")
    shutil.copyfile(initial_policy, lane_dir / "incumbent.py")
    queued = (
        None
        if reference_task_id is not None
        else enqueue(
            queue_dir,
            kind=kind,
            source=source,
            policy=lane_dir / "initial_pf.py",
        )
    )
    manifest = {
        "schema": 1,
        "created_at": _now(),
        "kind": kind,
        "source": str(source),
        "source_sha256": _sha256(source),
        "initial_policy_source": str(initial_policy),
        "initial_policy_sha256": _sha256(initial_policy),
        "iterations_target": iterations,
        "queue_dir": str(queue_dir),
        "family": family,
        "replicate": replicate,
        "instance": instance,
        "exact_reference_task_id": reference_task_id,
    }
    state = {
        "schema": 1,
        "iterations_completed": 0,
        "pending_kind": "reference" if reference_task_id is not None else "baseline",
        "pending_iteration": None,
        "pending_task_id": reference_task_id if reference_task_id is not None else queued["task_id"],
        "incumbent_score": None,
        "incumbent_task_id": None,
        "accepted_mutations": 0,
        "updated_at": _now(),
    }
    _write_json(lane_dir / "manifest.json", manifest)
    _write_json(lane_dir / "state.json", state)
    return {
        "status": "initialized",
        "lane_dir": str(lane_dir),
        "reference_task_id": reference_task_id,
        "baseline_task_id": queued["task_id"] if queued is not None else None,
        "queue_status": queued["status"] if queued is not None else "waiting_reference",
    }


def reconcile_lane(lane_dir: Path) -> dict[str, Any]:
    """Apply a completed exact result to one lane's incumbent state."""

    lane_dir = lane_dir.resolve()
    manifest = _read_json(lane_dir / "manifest.json")
    state = _read_json(lane_dir / "state.json")
    task_id = state["pending_task_id"]
    if task_id is None:
        status = "mutating" if state["pending_kind"] == "mutation" else "ready"
        return {"status": status, **state}

    phase, result = _task_result(Path(manifest["queue_dir"]), task_id)
    if result is None:
        return {"status": "waiting", "phase": phase, "task_id": task_id}

    if state["pending_kind"] == "reference":
        if result.get("status") != "ok" or not str(result.get("observable", "")).startswith(
            "exact_unitary_"
        ):
            raise RuntimeError("the immutable exact-unitary reference failed")
        reference_path = lane_dir / "exact_reference.json"
        shutil.copyfile(
            Path(manifest["queue_dir"]) / phase / task_id / "result.json",
            reference_path,
        )
        queued = enqueue(
            Path(manifest["queue_dir"]),
            kind=manifest["kind"],
            source=Path(manifest["source"]),
            policy=lane_dir / "initial_pf.py",
            reference=reference_path,
        )
        state.update(
            {
                "pending_kind": "baseline",
                "pending_task_id": queued["task_id"],
                "exact_reference_task_id": task_id,
                "updated_at": _now(),
            }
        )
        _write_json(lane_dir / "state.json", state)
        return {
            "status": "reference_ready",
            "reference_task_id": task_id,
            "baseline_task_id": queued["task_id"],
            "queue_status": queued["status"],
        }

    if state["pending_kind"] == "baseline":
        if result.get("status") != "ok" or not result.get("passes_chemical_accuracy"):
            raise RuntimeError("the generated baseline did not pass the exact accuracy contract")
        shutil.copyfile(
            Path(manifest["queue_dir"]) / phase / task_id / "result.json",
            lane_dir / "incumbent_result.json",
        )
        shutil.copyfile(
            Path(manifest["queue_dir"]) / phase / task_id / "result.json",
            lane_dir / "baseline_result.json",
        )
        state.update(
            {
                "pending_kind": None,
                "pending_task_id": None,
                "incumbent_score": result["score"],
                "incumbent_task_id": task_id,
                "updated_at": _now(),
            }
        )
        _write_json(lane_dir / "state.json", state)
        return {"status": "baseline_ready", "score": result["score"], "task_id": task_id}

    iteration = int(state["pending_iteration"])
    iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
    _write_json(iteration_dir / "result.json", result)
    accepted = (
        result.get("status") == "ok"
        and bool(result.get("passes_chemical_accuracy"))
        and int(result["score"]) < int(state["incumbent_score"])
    )
    previous_score = state["incumbent_score"]
    if accepted:
        shutil.copyfile(iteration_dir / "candidate.py", lane_dir / "incumbent.py")
        shutil.copyfile(iteration_dir / "result.json", lane_dir / "incumbent_result.json")
        state["incumbent_score"] = result["score"]
        state["incumbent_task_id"] = task_id
        state["accepted_mutations"] += 1
    decision = {
        "iteration": iteration,
        "accepted": accepted,
        "candidate_score": result.get("score"),
        "previous_incumbent_score": previous_score,
        "result_status": result.get("status"),
        "passes_chemical_accuracy": result.get("passes_chemical_accuracy", False),
        "task_id": task_id,
        "decided_at": _now(),
    }
    _write_json(iteration_dir / "decision.json", decision)
    state.update(
        {
            "iterations_completed": iteration,
            "pending_kind": None,
            "pending_iteration": None,
            "pending_task_id": None,
            "updated_at": _now(),
        }
    )
    _write_json(lane_dir / "state.json", state)
    return {"status": "accepted" if accepted else "rejected", **decision}


def propose_lane(lane_dir: Path, *, timeout_seconds: int = 900) -> dict[str, Any]:
    """Ask OpenCode/Nemotron for the next mutation and enqueue it."""

    lane_dir = lane_dir.resolve()
    reconciled = reconcile_lane(lane_dir)
    state = _read_json(lane_dir / "state.json")
    manifest = _read_json(lane_dir / "manifest.json")
    if state["pending_task_id"] is not None:
        return reconciled
    if state["pending_kind"] == "mutation":
        return reconciled
    if state["iterations_completed"] >= manifest["iterations_target"]:
        return {"status": "complete", **state}

    iteration = state["iterations_completed"] + 1
    iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
    state.update(
        {
            "pending_kind": "mutation",
            "pending_iteration": iteration,
            "last_error": None,
            "updated_at": _now(),
        }
    )
    _write_json(lane_dir / "state.json", state)
    try:
        mutation = mutate(
            lane_dir / "incumbent.py",
            lane_dir / "incumbent_result.json",
            iteration_dir,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        state.update(
            {
                "pending_kind": None,
                "pending_iteration": None,
                "last_error": f"{type(exc).__name__}: {exc}",
                "updated_at": _now(),
            }
        )
        _write_json(lane_dir / "state.json", state)
        raise
    queued = enqueue(
        Path(manifest["queue_dir"]),
        kind=manifest["kind"],
        source=Path(manifest["source"]),
        policy=Path(mutation["candidate"]),
        reference=lane_dir / "exact_reference.json",
    )
    state.update(
        {
            "pending_kind": "proposal",
            "pending_iteration": iteration,
            "pending_task_id": queued["task_id"],
            "last_error": None,
            "updated_at": _now(),
        }
    )
    _write_json(lane_dir / "state.json", state)
    return {
        "status": "proposed",
        "iteration": iteration,
        "task_id": queued["task_id"],
        "queue_status": queued["status"],
        "candidate": mutation["candidate"],
    }


def lane_status(lane_dir: Path) -> dict[str, Any]:
    lane_dir = lane_dir.resolve()
    manifest = _read_json(lane_dir / "manifest.json")
    state = _read_json(lane_dir / "state.json")
    return {
        "lane_dir": str(lane_dir),
        "iterations_target": manifest["iterations_target"],
        **state,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--lane-dir", type=Path, required=True)
    init.add_argument("--queue-dir", type=Path, required=True)
    init.add_argument("--kind", choices=("absolute", "difference"), required=True)
    init.add_argument("--source", type=Path, required=True)
    init.add_argument("--initial-policy", type=Path, required=True)
    init.add_argument("--iterations", type=int, default=50)
    init.add_argument("--family", default="custom")
    init.add_argument("--replicate")
    init.add_argument("--instance")
    init.add_argument("--reference-task-id")

    for command in ("reconcile", "status"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--lane-dir", type=Path, required=True)

    propose = subparsers.add_parser("propose")
    propose.add_argument("--lane-dir", type=Path, required=True)
    propose.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "init":
        result = initialize_lane(
            args.lane_dir,
            queue_dir=args.queue_dir,
            kind=args.kind,
            source=args.source,
            initial_policy=args.initial_policy,
            iterations=args.iterations,
            family=args.family,
            replicate=args.replicate,
            instance=args.instance,
            reference_task_id=args.reference_task_id,
        )
    elif args.command == "reconcile":
        result = reconcile_lane(args.lane_dir)
    elif args.command == "propose":
        result = propose_lane(args.lane_dir, timeout_seconds=args.timeout_seconds)
    else:
        result = lane_status(args.lane_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
