from __future__ import annotations

import json
from pathlib import Path

from hsopt.exact_loop import initialize_lane, propose_replay_lane, reconcile_lane


REPO = Path(__file__).resolve().parents[1]
ABSOLUTE = REPO / "examples" / "abs_energy" / "hamlib_h2_r0_7"


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


def test_gpu_lane_replays_the_exact_archived_cpu_candidate(tmp_path: Path) -> None:
    lane_dir = tmp_path / "gpu-campaign" / "lanes" / "hamlib121" / "fixture" / "run_01"
    initialize_lane(
        lane_dir,
        queue_dir=tmp_path / "gpu-campaign" / "queue",
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
        iterations=2,
        family="hamlib121",
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

    canonical = (
        tmp_path / "cpu-campaign" / "lanes" / "hamlib121" / "fixture" / "run_01"
    )
    iteration = canonical / "iterations" / "0001"
    iteration.mkdir(parents=True)
    candidate = (ABSOLUTE / "initial_pf.py").read_text(encoding="utf-8") + "\n# CPU trace\n"
    (iteration / "candidate.py").write_text(candidate, encoding="utf-8")
    canonical_result = {
        "status": "ok",
        "score": 90,
        "t_count": 90,
        "passes_chemical_accuracy": True,
        "policy": {},
    }
    (iteration / "result.json").write_text(
        json.dumps(canonical_result) + "\n", encoding="utf-8"
    )
    (iteration / "decision.json").write_text(
        '{"iteration":1,"accepted":true}\n', encoding="utf-8"
    )
    (iteration / "mutation_result.json").write_text(
        '{"status":"mutated"}\n', encoding="utf-8"
    )

    replayed = propose_replay_lane(lane_dir, canonical)

    replay_dir = lane_dir / "iterations" / "0001"
    assert replayed["status"] == "replay_proposed"
    assert (replay_dir / "candidate.py").read_text(encoding="utf-8") == candidate
    assert json.loads((replay_dir / "replay.json").read_text())[
        "canonical_candidate_sha256"
    ] == replayed["canonical_candidate_sha256"]
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
    assert json.loads((replay_dir / "replay_validation.json").read_text())[
        "status"
    ] == "ok"
    assert json.loads((replay_dir / "decision.json").read_text())[
        "replay_validation"
    ] == "ok"
