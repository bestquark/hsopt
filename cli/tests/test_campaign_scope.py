from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from hsopt.campaign.scope import reduce_hamlib_replicates
from hsopt.evaluation.compiled_unitary import DIFFERENCE_ERROR_CONTRACTS


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def task(root: Path, path: Path, lane_dir: str) -> None:
    write_json(
        path / "task.json",
        {
            "schema": 2,
            "task_id": path.name,
            "policy_original": str(root / lane_dir / "iterations/0001/candidate.py"),
        },
    )


def test_reduce_hamlib_replicates_preserves_histories_and_archives_queue_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    abs_lanes = [
        {
            "family": "abs_energy",
            "instance": "h2",
            "replicate": f"run_{index:02d}",
            "lane_dir": f"lanes/abs_energy/h2/run_{index:02d}",
        }
        for index in range(1, 6)
    ]
    reaction_lanes = [
        {
            "family": "diff_energy",
            "instance": "reaction",
            "replicate": contract,
            "lane_dir": f"lanes/diff_energy/reaction/{contract}",
        }
        for contract in DIFFERENCE_ERROR_CONTRACTS
    ]
    write_json(
        root / "campaign.json",
        {
            "schema": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "iterations": 50,
            "hamlib": {"instances": 1, "replicates": 5},
            "reactions": {
                "instances": 1,
                "contracts": list(DIFFERENCE_ERROR_CONTRACTS),
            },
            "lanes": abs_lanes + reaction_lanes,
        },
    )
    for row in abs_lanes + reaction_lanes:
        write_json(root / row["lane_dir"] / "state.json", {"history": "preserved"})

    task(root, root / "queue/pending/active-task", abs_lanes[0]["lane_dir"])
    task(root, root / "queue/pending/retired-task", abs_lanes[3]["lane_dir"])
    task(root, root / "queue/running/retired-running", abs_lanes[4]["lane_dir"])

    active_batch = root / "queue/batches/active-batch"
    task(root, active_batch / "tasks/active-batch-task", abs_lanes[1]["lane_dir"])
    retired_batch = root / "queue/batches/retired-batch"
    task(root, retired_batch / "tasks/retired-batch-task", abs_lanes[4]["lane_dir"])
    write_json(retired_batch / "collected.json", {"slurm_state": "LOCAL_RUNNING"})
    completed_batch = root / "queue/batches/completed-retired-batch"
    task(root, completed_batch / "tasks/completed-task", abs_lanes[3]["lane_dir"])
    write_json(completed_batch / "collected.json", {"slurm_state": "LOCAL_COMPLETED"})

    result = reduce_hamlib_replicates(
        root,
        replicates=3,
        reason="test predetermined three-replicate scope",
    )

    campaign = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    assert result["status"] == "applied"
    assert result["active_lane_count"] == 5
    assert result["retired_lane_count"] == 2
    assert result["retired_histories_deleted"] is False
    assert campaign["hamlib"]["replicates"] == 3
    assert [row["replicate"] for row in campaign["lanes"] if row["family"] == "abs_energy"] == [
        "run_01",
        "run_02",
        "run_03",
    ]
    assert [row["replicate"] for row in campaign["lanes"] if row["family"] == "diff_energy"] == [
        *DIFFERENCE_ERROR_CONTRACTS,
    ]

    change_root = root / "scope_changes" / result["change_id"]
    before = json.loads((change_root / "campaign.before.json").read_text(encoding="utf-8"))
    assert before["hamlib"]["replicates"] == 5
    assert len(before["lanes"]) == 7
    assert not (root / "queue/pending/retired-task").exists()
    assert not (root / "queue/running/retired-running").exists()
    assert not retired_batch.exists()
    assert (root / "queue/pending/active-task").is_dir()
    assert active_batch.is_dir()
    assert completed_batch.is_dir()
    assert (root / abs_lanes[3]["lane_dir"] / "state.json").is_file()
    assert (root / abs_lanes[4]["lane_dir"] / "state.json").is_file()
    assert len(result["archived_queue_work"]) == 3

    repeated = reduce_hamlib_replicates(root, replicates=3)
    assert repeated["status"] == "already_scoped"


def test_reduce_hamlib_replicates_refuses_generic_active_worker_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    write_json(
        root / "campaign.json",
        {
            "schema": 1,
            "hamlib": {"instances": 1, "replicates": 5},
            "lanes": [
                {
                    "family": "abs_energy",
                    "instance": "h2",
                    "replicate": f"run_{index:02d}",
                    "lane_dir": f"lanes/abs_energy/h2/run_{index:02d}",
                }
                for index in range(1, 6)
            ],
        },
    )
    lock_path = root / "supervisor/local_worker.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="stop its supervisor and local workers"):
            reduce_hamlib_replicates(root, replicates=3)

    campaign = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["hamlib"]["replicates"] == 5
    assert not (root / "scope_changes").exists()
