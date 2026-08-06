from __future__ import annotations

import json
from pathlib import Path

from hsopt.exact_loop import (
    LEGACY_INACCURATE_BASELINE_ERROR,
    _accept_candidate,
    initialize_lane,
    propose_replay_lane,
    reconcile_lane,
)
from hsopt.compiled_unitary import EVALUATION_CONTRACT_VERSION


REPO = Path(__file__).resolve().parents[2]
ABSOLUTE = REPO / "cli" / "tests" / "fixtures" / "abs_energy_h2"


def _result(
    *,
    error: float,
    t_count: int,
    passes: bool,
    confidence_adjusted_t_count: int | None = None,
) -> dict[str, object]:
    cost = confidence_adjusted_t_count or t_count
    return {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "passes_chemical_accuracy": passes,
        "absolute_energy_error_hartree": error,
        "per_attempt_t_count": t_count,
        "qpe_repetitions": cost // t_count,
        "confidence_adjusted_t_count": cost,
        "t_count": t_count,
        "score": cost if passes else None,
    }


def test_infeasible_incumbent_uses_exact_error_until_accuracy_is_reached() -> None:
    baseline = _result(error=0.004, t_count=100, passes=False)

    assert _accept_candidate(_result(error=0.003, t_count=120, passes=False), baseline) == (
        True,
        "exact_error_improvement",
    )
    assert _accept_candidate(_result(error=0.005, t_count=80, passes=False), baseline) == (
        False,
        "no_exact_error_improvement",
    )
    assert _accept_candidate(_result(error=0.001, t_count=140, passes=True), baseline) == (
        True,
        "chemical_accuracy_reached",
    )

    feasible = _result(error=0.001, t_count=140, passes=True)
    assert _accept_candidate(_result(error=0.0011, t_count=130, passes=True), feasible) == (
        True,
        "strict_confidence_adjusted_t_gate_improvement",
    )
    assert _accept_candidate(
        _result(
            error=0.0011,
            t_count=100,
            passes=True,
            confidence_adjusted_t_count=200,
        ),
        feasible,
    ) == (False, "no_strict_confidence_adjusted_t_gate_improvement")
    assert _accept_candidate(_result(error=0.004, t_count=80, passes=False), feasible) == (
        False,
        "chemical_accuracy_failed",
    )


def test_reconcile_archives_an_interrupted_mutation_attempt(tmp_path: Path) -> None:
    lane_dir = tmp_path / "lane"
    initialize_lane(
        lane_dir,
        queue_dir=tmp_path / "queue",
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
    )
    state_path = lane_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "pending_kind": "mutation",
            "pending_iteration": 1,
            "pending_task_id": None,
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    interrupted = lane_dir / "iterations" / "0001"
    interrupted.mkdir()
    (interrupted / "candidate.py").write_text("interrupted\n", encoding="utf-8")

    recovered = reconcile_lane(lane_dir)

    updated = json.loads(state_path.read_text(encoding="utf-8"))
    archived = lane_dir / "mutation_failures" / "0001-attempt-001"
    assert recovered["status"] == "mutation_recovered"
    assert updated["pending_kind"] is None
    assert updated["pending_iteration"] is None
    assert updated["last_error"] == "interrupted mutation attempt recovered"
    assert not interrupted.exists()
    assert (archived / "candidate.py").read_text(encoding="utf-8") == "interrupted\n"


def test_reconcile_clears_the_retired_inaccurate_baseline_error(tmp_path: Path) -> None:
    lane_dir = tmp_path / "lane"
    initialize_lane(
        lane_dir,
        queue_dir=tmp_path / "queue",
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
    )
    state_path = lane_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "pending_kind": None,
            "pending_task_id": None,
            "last_error": LEGACY_INACCURATE_BASELINE_ERROR,
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (lane_dir / "baseline_result.json").write_text(
        json.dumps(_result(error=0.004, t_count=100, passes=False)) + "\n",
        encoding="utf-8",
    )

    reconciled = reconcile_lane(lane_dir)

    assert reconciled["status"] == "ready"
    assert json.loads(state_path.read_text(encoding="utf-8"))["last_error"] is None


def test_candidate_evaluator_error_is_retried_without_consuming_iteration(tmp_path: Path) -> None:
    lane_dir = tmp_path / "campaign" / "lanes" / "abs_energy" / "fixture" / "run_01"
    queue_dir = tmp_path / "campaign" / "queue"
    initialize_lane(
        lane_dir,
        queue_dir=queue_dir,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
        iterations=2,
        family="abs_energy",
        instance="fixture",
        replicate="run_01",
    )
    state_path = lane_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "pending_kind": "proposal",
            "pending_iteration": 1,
            "pending_task_id": "failed-task",
            "incumbent_score": 100,
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (lane_dir / "incumbent_result.json").write_text(
        json.dumps(_result(error=0.001, t_count=100, passes=True)) + "\n",
        encoding="utf-8",
    )
    iteration_dir = lane_dir / "iterations" / "0001"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "candidate.py").write_text("# invalid candidate\n", encoding="utf-8")
    failed_task = queue_dir / "failed" / "failed-task"
    failed_task.mkdir(parents=True)
    failed_result = {
        "status": "error",
        "task_id": "failed-task",
        "error_type": "NameError",
        "error": "name 'term' is not defined",
    }
    (failed_task / "result.json").write_text(
        json.dumps(failed_result) + "\n",
        encoding="utf-8",
    )

    reconciled = reconcile_lane(lane_dir)

    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert reconciled == {
        "status": "candidate_evaluation_failed",
        "iteration": 1,
        "task_id": "failed-task",
        "error_type": "NameError",
        "error": "name 'term' is not defined",
    }
    assert updated["iterations_completed"] == 0
    assert updated["pending_kind"] is None
    assert updated["pending_iteration"] is None
    assert updated["pending_task_id"] is None
    assert updated["last_error"] == "NameError: name 'term' is not defined"
    assert json.loads((iteration_dir / "result.json").read_text()) == failed_result


def test_gpu_lane_replays_the_exact_archived_cpu_candidate(tmp_path: Path) -> None:
    lane_dir = tmp_path / "gpu-campaign" / "lanes" / "abs_energy" / "fixture" / "run_01"
    initialize_lane(
        lane_dir,
        queue_dir=tmp_path / "gpu-campaign" / "queue",
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
        iterations=2,
        family="abs_energy",
        instance="fixture",
        replicate="run_01",
    )
    state_path = lane_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "pending_kind": None,
            "pending_iteration": None,
            "pending_task_id": None,
            "incumbent_score": 100,
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (lane_dir / "exact_reference.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    (lane_dir / "incumbent_result.json").write_text(
        json.dumps(_result(error=0.001, t_count=100, passes=True)) + "\n",
        encoding="utf-8",
    )

    canonical = tmp_path / "cpu-campaign" / "lanes" / "abs_energy" / "fixture" / "run_01"
    iteration = canonical / "iterations" / "0001"
    iteration.mkdir(parents=True)
    candidate = (ABSOLUTE / "initial_pf.py").read_text(encoding="utf-8") + "\n# CPU trace\n"
    (iteration / "candidate.py").write_text(candidate, encoding="utf-8")
    canonical_result = {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "score": 90,
        "per_attempt_t_count": 90,
        "qpe_repetitions": 1,
        "confidence_adjusted_t_count": 90,
        "t_count": 90,
        "passes_chemical_accuracy": True,
        "policy": {},
    }
    (iteration / "result.json").write_text(json.dumps(canonical_result) + "\n", encoding="utf-8")
    (iteration / "decision.json").write_text('{"iteration":1,"accepted":true}\n', encoding="utf-8")
    (iteration / "mutation_result.json").write_text('{"status":"mutated"}\n', encoding="utf-8")

    replayed = propose_replay_lane(lane_dir, canonical)

    replay_dir = lane_dir / "iterations" / "0001"
    assert replayed["status"] == "replay_proposed"
    assert (replay_dir / "candidate.py").read_text(encoding="utf-8") == candidate
    assert (
        json.loads((replay_dir / "replay.json").read_text())["canonical_candidate_sha256"]
        == replayed["canonical_candidate_sha256"]
    )
    assert (replay_dir / "canonical_cpu_decision.json").is_file()
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["pending_kind"] == "proposal"
    assert updated["pending_iteration"] == 1

    queue = tmp_path / "gpu-campaign" / "queue"
    task_id = replayed["task_id"]
    (queue / "pending" / task_id).rename(queue / "completed" / task_id)
    (queue / "completed" / task_id / "result.json").write_text(
        json.dumps({**canonical_result, "task_id": task_id}) + "\n",
        encoding="utf-8",
    )
    reconciled = reconcile_lane(lane_dir)

    assert reconciled["status"] == "accepted"
    assert json.loads((replay_dir / "replay_validation.json").read_text())["status"] == "ok"
    assert json.loads((replay_dir / "decision.json").read_text())["replay_validation"] == "ok"
