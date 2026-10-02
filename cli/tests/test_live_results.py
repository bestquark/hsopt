from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hsopt.cluster import live_results


TASKS = [f"{index:024x}" for index in (1, 2)]


def setup_batch(tmp_path, monkeypatch):
    queue = tmp_path / "queue"
    batch = queue / "batches" / "batch"
    batch.mkdir(parents=True)
    for phase in ("pending", "completed", "failed"):
        (queue / phase).mkdir()
    for task_id in TASKS:
        task = batch / "tasks" / task_id
        task.mkdir(parents=True)
        (task / "task.json").write_text(
            json.dumps(
                {
                    "kind": "difference",
                    "evaluation_contract_version": "test-v1",
                    "difference_error_contract": "error_cancellation",
                }
            )
        )
    (batch / "batch.json").write_text(json.dumps({"task_ids": TASKS, "batch_id": "batch"}))
    (batch / "submission.json").write_text(
        json.dumps(
            {
                "host": "nvlogin",
                "remote_batch": "/remote/batch",
                "job_id": "123",
                "backend": "cupynumeric",
                "accelerator": "gpu",
            }
        )
    )
    incoming = {}

    def download(command, **kwargs):
        assert command[-2] == "nvlogin:/remote/batch/results/"
        assert "--timeout=15" in command
        assert "--include=*.json" in command
        assert kwargs["timeout"] == 30
        assert not any("gpu-utilization" in arg or "batch.json" in arg for arg in command)
        for task_id, data in incoming.items():
            payload = json.dumps(data) if isinstance(data, dict) else data
            (Path(command[-1]) / f"{task_id}.json").write_text(payload)

    monkeypatch.setattr(live_results.subprocess, "run", download)
    return queue, batch, incoming


def result(task, **overrides):
    return {
        "task_id": task,
        "status": "ok",
        "evaluation_contract_version": "test-v1",
        "difference_error_contract": "error_cancellation",
        "execution_backend": "cupynumeric",
        "execution_accelerator": "gpu",
        "task_finished_at": "2026-09-11T05:00:00+00:00",
        "task_wall_seconds": 15.0,
        **overrides,
    }


def test_partial_results_merge_without_requeuing_or_finishing_job(tmp_path, monkeypatch):
    queue, batch, incoming = setup_batch(tmp_path, monkeypatch)
    original_manifest = (batch / "batch.json").read_bytes()
    incoming[TASKS[0]] = result(TASKS[0])
    incoming[TASKS[1]] = '{"status": "ok"'  # Not yet a complete JSON document.
    first = live_results.collect_running_results(queue, batch)
    assert first["status"] == "partial"
    assert first["newly_merged"] == [TASKS[0]]
    assert first["missing"] == first["deferred_incomplete"] == [TASKS[1]]
    assert (batch / "tasks" / TASKS[1]).is_dir()
    assert not list((queue / "pending").iterdir())
    assert not (batch / "collected.json").exists()
    assert (batch / "batch.json").read_bytes() == original_manifest
    archived = queue / "completed" / TASKS[0] / "result.json"
    original_result = archived.read_bytes()
    assert live_results.collect_running_results(queue, batch)["newly_merged"] == []
    assert archived.read_bytes() == original_result
    incoming[TASKS[1]] = result(TASKS[1], status="error")
    last = live_results.collect_running_results(queue, batch)
    assert last["status"] == "merged"
    assert last["newly_merged"] == [TASKS[1]]
    assert (queue / "failed" / TASKS[1] / "result.json").is_file()
    assert not (batch / "collected.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", TASKS[1]),
        ("difference_error_contract", "individual_errors"),
        ("execution_accelerator", "cpu"),
        ("evaluation_contract_version", "wrong-version"),
    ],
)
def test_wrong_provenance_is_rejected(tmp_path, monkeypatch, field, value):
    queue, batch, incoming = setup_batch(tmp_path, monkeypatch)
    incoming[TASKS[0]] = result(TASKS[0], **{field: value})
    with pytest.raises(ValueError, match=field):
        live_results.collect_running_results(queue, batch)
    assert not list((queue / "completed").iterdir())
    assert (batch / "tasks" / TASKS[0]).is_dir()


def test_existing_result_is_never_overwritten(tmp_path, monkeypatch):
    queue, batch, incoming = setup_batch(tmp_path, monkeypatch)
    incoming[TASKS[0]] = result(TASKS[0])
    live_results.collect_running_results(queue, batch)
    archived = queue / "completed" / TASKS[0] / "result.json"
    original = archived.read_bytes()
    incoming[TASKS[0]] = result(TASKS[0], task_wall_seconds=999)
    incoming[TASKS[1]] = result(TASKS[1])
    with pytest.raises(ValueError, match="overwrite"):
        live_results.collect_running_results(queue, batch)
    assert archived.read_bytes() == original
    assert (batch / "tasks" / TASKS[1]).is_dir()


def test_transfer_timeout_leaves_tasks_untouched(tmp_path, monkeypatch):
    queue, batch, _ = setup_batch(tmp_path, monkeypatch)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("rsync", 30)

    monkeypatch.setattr(live_results.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        live_results.collect_running_results(queue, batch)
    assert len(list((batch / "tasks").iterdir())) == 2
    assert not list(batch.glob(".live-results-*"))


def test_missing_finish_timestamp_is_deferred(tmp_path, monkeypatch):
    queue, batch, incoming = setup_batch(tmp_path, monkeypatch)
    incoming[TASKS[0]] = result(TASKS[0], task_finished_at=None)
    response = live_results.collect_running_results(queue, batch)
    assert response["deferred_incomplete"] == [TASKS[0]]
    assert not list((queue / "completed").iterdir())
