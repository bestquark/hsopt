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

from .compiled_unitary import EVALUATION_CONTRACT_VERSION
from .exact_queue import enqueue
from .openshell_agent import mutate


REPLAY_ENERGY_ATOL_HARTREE = 1.0e-10
LEGACY_INACCURATE_BASELINE_ERROR = (
    "RuntimeError: the generated baseline did not pass the exact accuracy contract"
)


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


def _exact_error(result: dict[str, Any]) -> float:
    """Return the exact constrained error for either benchmark family."""

    for key in (
        "absolute_energy_error_hartree",
        "absolute_difference_error_hartree",
    ):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise RuntimeError("the exact evaluator result has no absolute energy error")


def _objective_cost(result: dict[str, Any]) -> int:
    """Return the confidence-adjusted objective, with legacy fallback."""

    for key in ("confidence_adjusted_t_count", "score", "t_count"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    raise RuntimeError("the exact evaluator result has no T-count objective")


def _accept_candidate(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
) -> tuple[bool, str]:
    """Apply a deterministic feasibility-first constrained objective.

    Once a lane is chemically accurate, only an accurate strict reduction in
    the confidence-adjusted QPE T cost can replace its incumbent. Before
    feasibility, exact-error progress is retained so an initially inaccurate
    paper baseline can be improved over multiple mutations.
    """

    if candidate.get("status") != "ok":
        return False, "evaluator_error"
    candidate_passes = bool(candidate.get("passes_chemical_accuracy"))
    incumbent_passes = bool(incumbent.get("passes_chemical_accuracy"))
    if incumbent_passes:
        if not candidate_passes:
            return False, "chemical_accuracy_failed"
        accepted = _objective_cost(candidate) < _objective_cost(incumbent)
        return accepted, (
            "strict_confidence_adjusted_t_gate_improvement"
            if accepted
            else "no_strict_confidence_adjusted_t_gate_improvement"
        )
    if candidate_passes:
        return True, "chemical_accuracy_reached"

    candidate_error = _exact_error(candidate)
    incumbent_error = _exact_error(incumbent)
    if candidate_error < incumbent_error - REPLAY_ENERGY_ATOL_HARTREE:
        return True, "exact_error_improvement"
    if abs(candidate_error - incumbent_error) <= REPLAY_ENERGY_ATOL_HARTREE and _objective_cost(
        candidate
    ) < _objective_cost(incumbent):
        return True, "infeasible_tie_break_confidence_adjusted_t_gate_improvement"
    return False, "no_exact_error_improvement"


def _archive_mutation_attempt(lane_dir: Path, iteration_dir: Path) -> Path | None:
    """Preserve a failed or interrupted agent attempt before retrying it."""

    if not iteration_dir.is_dir() or not any(iteration_dir.iterdir()):
        return None
    archive_root = lane_dir / "mutation_failures"
    archive_root.mkdir(exist_ok=True)
    attempt = 1
    while (archive_root / f"{iteration_dir.name}-attempt-{attempt:03d}").exists():
        attempt += 1
    destination = archive_root / f"{iteration_dir.name}-attempt-{attempt:03d}"
    iteration_dir.rename(destination)
    return destination


def _validate_replay(
    iteration_dir: Path,
    *,
    result: dict[str, Any],
    accepted: bool,
    acceptance_reason: str,
) -> dict[str, Any] | None:
    canonical_result_path = iteration_dir / "canonical_cpu_result.json"
    canonical_decision_path = iteration_dir / "canonical_cpu_decision.json"
    replay_path = iteration_dir / "replay.json"
    if not all(
        path.is_file() for path in (canonical_result_path, canonical_decision_path, replay_path)
    ):
        return None
    canonical_result = _read_json(canonical_result_path)
    canonical_decision = _read_json(canonical_decision_path)
    replay = _read_json(replay_path)
    candidate_sha256 = _sha256(iteration_dir / "candidate.py")
    checks: dict[str, bool] = {
        "candidate_sha256": candidate_sha256 == replay["canonical_candidate_sha256"],
        "status": result.get("status") == canonical_result.get("status"),
        "t_count": result.get("t_count") == canonical_result.get("t_count"),
        "qpe_repetitions": result.get("qpe_repetitions") == canonical_result.get("qpe_repetitions"),
        "confidence_adjusted_t_count": result.get("confidence_adjusted_t_count")
        == canonical_result.get("confidence_adjusted_t_count"),
        "score": result.get("score") == canonical_result.get("score"),
        "policy": result.get("policy") == canonical_result.get("policy"),
        "passes_chemical_accuracy": result.get("passes_chemical_accuracy")
        == canonical_result.get("passes_chemical_accuracy"),
        "accepted": accepted == bool(canonical_decision.get("accepted")),
    }
    if "acceptance_reason" in canonical_decision:
        checks["acceptance_reason"] = acceptance_reason == canonical_decision["acceptance_reason"]
    energy_differences: dict[str, float] = {}
    for key in (
        "exact_unitary_ground_energy_hartree",
        "compiled_pf_energy_hartree",
        "absolute_energy_error_hartree",
        "exact_difference_hartree",
        "compiled_pf_difference_hartree",
        "signed_difference_error_hartree",
        "absolute_difference_error_hartree",
    ):
        cpu_value = canonical_result.get(key)
        gpu_value = result.get(key)
        if isinstance(cpu_value, (int, float)) and isinstance(gpu_value, (int, float)):
            difference = abs(float(cpu_value) - float(gpu_value))
            energy_differences[key] = difference
            checks[f"energy:{key}"] = difference <= REPLAY_ENERGY_ATOL_HARTREE
    validation = {
        "schema": 1,
        "validated_at": _now(),
        "status": "ok" if all(checks.values()) else "mismatch",
        "energy_atol_hartree": REPLAY_ENERGY_ATOL_HARTREE,
        "checks": checks,
        "energy_absolute_differences_hartree": energy_differences,
        "canonical_candidate_sha256": replay["canonical_candidate_sha256"],
        "candidate_sha256": candidate_sha256,
        "cpu_task_id": canonical_result.get("task_id"),
        "replay_task_id": result.get("task_id"),
    }
    _write_json(iteration_dir / "replay_validation.json", validation)
    return validation


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
    exact_reference_path: Path | None = None,
) -> dict[str, Any]:
    """Create one reproducible lane, waiting for its shared exact reference first."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    lane_dir = lane_dir.resolve()
    queue_dir = queue_dir.resolve()
    source = source.resolve()
    initial_policy = initial_policy.resolve()
    if reference_task_id is not None and exact_reference_path is not None:
        raise ValueError("provide reference_task_id or exact_reference_path, not both")
    if exact_reference_path is not None:
        exact_reference_path = exact_reference_path.resolve()
        reference = _read_json(exact_reference_path)
        if reference.get("evaluation_contract_version") != EVALUATION_CONTRACT_VERSION:
            raise ValueError("precomputed exact reference uses a different evaluator contract")
        if kind == "absolute":
            if reference.get("observable") != "exact_unitary_ground_eigenphase":
                raise ValueError("absolute lane requires a ground-eigenphase reference")
            if reference.get("problem_sha256") != _sha256(source):
                raise ValueError("precomputed exact reference does not match lane source")
        elif kind == "difference":
            if reference.get("observable") != "exact_unitary_signed_energy_difference":
                raise ValueError("difference lane requires a signed-difference reference")
            if reference.get("reaction_sha256") != _sha256(source):
                raise ValueError("precomputed exact reference does not match lane source")
        else:
            raise ValueError(f"unknown evaluation kind {kind!r}")
    if lane_dir.exists() and any(lane_dir.iterdir()):
        raise RuntimeError(f"lane directory must be empty: {lane_dir}")
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / "iterations").mkdir()
    shutil.copyfile(initial_policy, lane_dir / "initial_pf.py")
    shutil.copyfile(initial_policy, lane_dir / "incumbent.py")
    lane_reference = lane_dir / "exact_reference.json"
    if exact_reference_path is not None:
        shutil.copyfile(exact_reference_path, lane_reference)
    queued = (
        None
        if reference_task_id is not None
        else enqueue(
            queue_dir,
            kind=kind,
            source=source,
            policy=lane_dir / "initial_pf.py",
            reference=lane_reference if exact_reference_path is not None else None,
        )
    )
    manifest = {
        "schema": 2,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
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
        "exact_reference_source": (
            str(exact_reference_path) if exact_reference_path is not None else None
        ),
        "exact_reference_sha256": (
            _sha256(exact_reference_path) if exact_reference_path is not None else None
        ),
    }
    state = {
        "schema": 1,
        "iterations_completed": 0,
        "pending_kind": "reference" if reference_task_id is not None else "baseline",
        "pending_iteration": None,
        "pending_task_id": reference_task_id
        if reference_task_id is not None
        else queued["task_id"],
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
        "reference_reused": exact_reference_path is not None,
    }


def reconcile_lane(lane_dir: Path) -> dict[str, Any]:
    """Apply a completed exact result to one lane's incumbent state."""

    lane_dir = lane_dir.resolve()
    manifest = _read_json(lane_dir / "manifest.json")
    state = _read_json(lane_dir / "state.json")
    task_id = state["pending_task_id"]
    if task_id is None:
        if (
            state.get("last_error") == LEGACY_INACCURATE_BASELINE_ERROR
            and (lane_dir / "baseline_result.json").is_file()
        ):
            state["last_error"] = None
            state["updated_at"] = _now()
            _write_json(lane_dir / "state.json", state)
        if state["pending_kind"] == "mutation":
            iteration = int(state["pending_iteration"])
            archived = _archive_mutation_attempt(
                lane_dir,
                lane_dir / "iterations" / f"{iteration:04d}",
            )
            state.update(
                {
                    "pending_kind": None,
                    "pending_iteration": None,
                    "last_error": "interrupted mutation attempt recovered",
                    "updated_at": _now(),
                }
            )
            _write_json(lane_dir / "state.json", state)
            return {
                "status": "mutation_recovered",
                "iteration": iteration,
                "archived_attempt": str(archived) if archived is not None else None,
                **state,
            }
        return {"status": "ready", **state}

    phase, result = _task_result(Path(manifest["queue_dir"]), task_id)
    if result is None:
        return {"status": "waiting", "phase": phase, "task_id": task_id}

    if state["pending_kind"] == "reference":
        if result.get("status") != "ok" or not str(result.get("observable", "")).startswith(
            "exact_unitary_"
        ):
            raise RuntimeError("the immutable exact-unitary reference failed")
        if result.get("evaluation_contract_version") != manifest.get("evaluation_contract_version"):
            raise RuntimeError("the immutable exact reference uses a different evaluator contract")
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
        if result.get("status") != "ok":
            raise RuntimeError("the generated baseline exact evaluation failed")
        if result.get("evaluation_contract_version") != manifest.get("evaluation_contract_version"):
            raise RuntimeError("the baseline result uses a different evaluator contract")
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
                "incumbent_score": result.get("score"),
                "incumbent_task_id": task_id,
                "last_error": None,
                "updated_at": _now(),
            }
        )
        _write_json(lane_dir / "state.json", state)
        return {
            "status": "baseline_ready",
            "score": result.get("score"),
            "t_count": result["t_count"],
            "passes_chemical_accuracy": bool(result.get("passes_chemical_accuracy")),
            "task_id": task_id,
        }

    iteration = int(state["pending_iteration"])
    iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
    if result.get("evaluation_contract_version") != manifest.get("evaluation_contract_version"):
        raise RuntimeError("the candidate result uses a different evaluator contract")
    _write_json(iteration_dir / "result.json", result)
    incumbent_result = _read_json(lane_dir / "incumbent_result.json")
    accepted, acceptance_reason = _accept_candidate(result, incumbent_result)
    previous_score = state["incumbent_score"]
    replay_validation = _validate_replay(
        iteration_dir,
        result=result,
        accepted=accepted,
        acceptance_reason=acceptance_reason,
    )
    if replay_validation is not None and replay_validation["status"] != "ok":
        decision = {
            "iteration": iteration,
            "accepted": False,
            "candidate_score": result.get("score"),
            "previous_incumbent_score": previous_score,
            "result_status": result.get("status"),
            "passes_chemical_accuracy": result.get("passes_chemical_accuracy", False),
            "acceptance_reason": acceptance_reason,
            "task_id": task_id,
            "decided_at": _now(),
            "replay_validation": "mismatch",
        }
        _write_json(iteration_dir / "decision.json", decision)
        state.update(
            {
                "pending_kind": "replay_validation_failed",
                "pending_task_id": None,
                "last_error": "CPU/A100 replay validation mismatch",
                "updated_at": _now(),
            }
        )
        _write_json(lane_dir / "state.json", state)
        return {
            "status": "replay_validation_failed",
            "replay_validation": replay_validation,
            **decision,
        }
    if accepted:
        shutil.copyfile(iteration_dir / "candidate.py", lane_dir / "incumbent.py")
        shutil.copyfile(iteration_dir / "result.json", lane_dir / "incumbent_result.json")
        state["incumbent_score"] = result.get("score")
        state["incumbent_task_id"] = task_id
        state["accepted_mutations"] += 1
    decision = {
        "iteration": iteration,
        "accepted": accepted,
        "candidate_score": result.get("score"),
        "previous_incumbent_score": previous_score,
        "result_status": result.get("status"),
        "passes_chemical_accuracy": result.get("passes_chemical_accuracy", False),
        "acceptance_reason": acceptance_reason,
        "task_id": task_id,
        "decided_at": _now(),
        "replay_validation": (
            replay_validation["status"] if replay_validation is not None else None
        ),
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
    archived = _archive_mutation_attempt(lane_dir, iteration_dir)
    state.update(
        {
            "pending_kind": "mutation",
            "pending_iteration": iteration,
            "last_error": (
                f"previous mutation attempt archived at {archived}"
                if archived is not None
                else None
            ),
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


def propose_replay_lane(lane_dir: Path, canonical_lane_dir: Path) -> dict[str, Any]:
    """Enqueue the next archived canonical candidate without running another agent."""

    lane_dir = lane_dir.resolve()
    canonical_lane_dir = canonical_lane_dir.resolve()
    reconciled = reconcile_lane(lane_dir)
    state = _read_json(lane_dir / "state.json")
    manifest = _read_json(lane_dir / "manifest.json")
    if state["pending_task_id"] is not None or state["pending_kind"] == "mutation":
        return reconciled
    if state["iterations_completed"] >= manifest["iterations_target"]:
        return {"status": "complete", **state}

    iteration = int(state["iterations_completed"]) + 1
    canonical_iteration = canonical_lane_dir / "iterations" / f"{iteration:04d}"
    required = ("candidate.py", "result.json", "decision.json", "mutation_result.json")
    if any(not (canonical_iteration / name).is_file() for name in required):
        return {
            "status": "waiting_canonical",
            "iteration": iteration,
            "canonical_lane_dir": str(canonical_lane_dir),
        }

    iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
    archived = _archive_mutation_attempt(lane_dir, iteration_dir)
    iteration_dir.mkdir(parents=True)
    for source_name, destination_name in (
        ("candidate.py", "candidate.py"),
        ("mutation_result.json", "mutation_result.json"),
        ("agent_audit.json", "agent_audit.json"),
        ("agent_actions.jsonl", "agent_actions.jsonl"),
        ("decision.json", "canonical_cpu_decision.json"),
        ("result.json", "canonical_cpu_result.json"),
    ):
        source = canonical_iteration / source_name
        if source.is_file():
            shutil.copyfile(source, iteration_dir / destination_name)
    provenance = {
        "schema": 1,
        "replayed_at": _now(),
        "canonical_lane_dir": str(canonical_lane_dir),
        "canonical_iteration": iteration,
        "canonical_candidate_sha256": _sha256(canonical_iteration / "candidate.py"),
        "candidate_sha256": _sha256(iteration_dir / "candidate.py"),
        "canonical_decision_sha256": _sha256(canonical_iteration / "decision.json"),
        "archived_previous_attempt": str(archived) if archived is not None else None,
    }
    _write_json(iteration_dir / "replay.json", provenance)
    queued = enqueue(
        Path(manifest["queue_dir"]),
        kind=manifest["kind"],
        source=Path(manifest["source"]),
        policy=iteration_dir / "candidate.py",
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
        "status": "replay_proposed",
        "iteration": iteration,
        "task_id": queued["task_id"],
        "queue_status": queued["status"],
        "candidate": str(iteration_dir / "candidate.py"),
        "canonical_candidate_sha256": provenance["canonical_candidate_sha256"],
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
    init.add_argument("--exact-reference", type=Path)

    for command in ("reconcile", "status"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--lane-dir", type=Path, required=True)

    propose = subparsers.add_parser("propose")
    propose.add_argument("--lane-dir", type=Path, required=True)
    propose.add_argument("--timeout-seconds", type=int, default=900)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--lane-dir", type=Path, required=True)
    replay.add_argument("--canonical-lane-dir", type=Path, required=True)
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
            exact_reference_path=args.exact_reference,
        )
    elif args.command == "reconcile":
        result = reconcile_lane(args.lane_dir)
    elif args.command == "propose":
        result = propose_lane(args.lane_dir, timeout_seconds=args.timeout_seconds)
    elif args.command == "replay":
        result = propose_replay_lane(args.lane_dir, args.canonical_lane_dir)
    else:
        result = lane_status(args.lane_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
