from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsopt.campaign.supervisor import _pending_runtime_estimate, supervise_once


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _timed_pending(tmp_path: Path) -> tuple[Path, list[Path]]:
    queue = tmp_path / "queue"
    lanes = [tmp_path / f"lane-{i}" for i in range(2)]
    for i, lane in enumerate(lanes):
        (queue / "pending" / f"task-{i}").mkdir(parents=True)
        _write(
            lane / "state.json",
            {
                "pending_task_id": f"task-{i}",
                "pending_kind": "proposal",
                "iterations_completed": 1,
            },
        )
        result = {
            "status": "ok",
            "execution_backend": "cupynumeric",
            "execution_accelerator": "gpu",
            "evaluation_wall_seconds": 900,
        }
        _write(lane / "incumbent_result.json", result)
        _write(
            lane / "iterations/0001/result.json",
            {
                **result,
                "evaluation_wall_seconds": 600 if i == 0 else 1200,
            },
        )
    return queue, lanes


def _estimate(queue: Path, lanes: list[Path], count: int = 2):
    return _pending_runtime_estimate(
        queue,
        lanes,
        expected_count=count,
        backend="cupynumeric",
        accelerator="gpu",
    )


def test_estimate_uses_shorter_matching_timings_without_executing_policies(tmp_path):
    queue, lanes = _timed_pending(tmp_path)
    for lane in lanes:
        (lane / "candidate.py").write_text('raise RuntimeError("must never execute")')
    estimate = _estimate(queue, lanes)
    assert estimate["seconds"] == 1500
    assert estimate["task_seconds"] == {"task-0": 600, "task-1": 900}


def test_first_mutations_cannot_use_cold_baseline_timings(tmp_path):
    queue, lanes = _timed_pending(tmp_path)
    for lane in lanes:
        state = json.loads((lane / "state.json").read_text())
        _write(lane / "state.json", {**state, "iterations_completed": 0})
        result = json.loads((lane / "incumbent_result.json").read_text())
        _write(lane / "incumbent_result.json", {**result, "evaluation_wall_seconds": 1800})
    assert _estimate(queue, lanes) is None


@pytest.mark.parametrize("damage", ["missing", "backend", "nan"])
def test_latest_mutation_must_have_a_usable_timing(tmp_path, damage):
    queue, lanes = _timed_pending(tmp_path)
    path = lanes[0] / "iterations/0001/result.json"
    if damage == "missing":
        path.unlink()
    else:
        result = json.loads(path.read_text())
        if damage == "backend":
            result["execution_backend"] = "numpy"
        else:
            result["evaluation_wall_seconds"] = float("nan")
        _write(path, result)
    assert _estimate(queue, lanes) is None


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_task",
        "baseline",
        "backend",
        "accelerator",
        "missing_timing",
        "nan",
        "infinite",
        "negative",
        "malformed",
        "duplicate_lane",
        "changed_count",
        "single",
    ],
)
def test_estimate_cannot_authorize_unverified_work(tmp_path, damage):
    queue, lanes = _timed_pending(tmp_path)
    count = 2
    if damage == "unknown_task":
        (queue / "pending/task-0").rename(queue / "pending/unknown")
    elif damage == "baseline":
        state = json.loads((lanes[0] / "state.json").read_text())
        _write(lanes[0] / "state.json", {**state, "pending_kind": "baseline"})
    elif damage == "duplicate_lane":
        lanes = [lanes[0], lanes[0], lanes[1]]
    elif damage == "changed_count":
        count = 3
    elif damage == "single":
        (queue / "pending/task-1").rmdir()
        lanes, count = lanes[:1], 1
    else:
        for path in [lanes[0] / "incumbent_result.json", lanes[0] / "iterations/0001/result.json"]:
            result = json.loads(path.read_text())
            if damage == "backend":
                result["execution_backend"] = "numpy"
            elif damage == "accelerator":
                result["execution_accelerator"] = "cpu"
            elif damage == "missing_timing":
                del result["evaluation_wall_seconds"]
            elif damage == "malformed":
                path.write_text("{")
                continue
            else:
                result["evaluation_wall_seconds"] = {
                    "nan": float("nan"),
                    "infinite": float("inf"),
                    "negative": -1,
                }[damage]
            _write(path, result)
    assert _estimate(queue, lanes, count) is None


@pytest.mark.parametrize("minimum", [-1, float("nan"), float("inf")])
def test_idle_estimate_option_rejects_invalid_thresholds(tmp_path, minimum):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        supervise_once(tmp_path, repo_dir=tmp_path, idle_batch_min_estimated_seconds=minimum)
