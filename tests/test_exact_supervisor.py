from __future__ import annotations

import json
import shutil
import time
from threading import Lock
from pathlib import Path

from hsopt.campaign_setup import initialize_campaign
from hsopt.exact_loop import initialize_lane, reconcile_lane
from hsopt.exact_queue import enqueue_reference, merge_batch, prepare_batch, run_batch
from hsopt.exact_supervisor import (
    _finalize_campaign,
    _run_mutation_proposals,
    slurm_state,
    supervise_once,
)


REPO = Path(__file__).resolve().parents[1]
ABSOLUTE = REPO / "examples" / "abs_energy" / "hamlib_h2_r0_7"


def _campaign(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "campaign"
    queue = root / "queue"
    lane = root / "lanes" / "h2" / "run_01"
    initialized = initialize_lane(
        lane,
        queue_dir=queue,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
        iterations=2,
        family="hamlib121",
        instance="h2",
        replicate="run_01",
    )
    (root / "campaign.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "queue_dir": str(queue.resolve()),
                "lanes": [
                    {
                        "family": "hamlib121",
                        "instance": "h2",
                        "replicate": "run_01",
                        "lane_dir": str(lane.relative_to(root)),
                        "baseline_task_id": initialized["baseline_task_id"],
                    }
                ],
            }
        )
        + "\n"
    )
    return root, queue, lane


def test_slurm_state_prefers_the_root_job_row() -> None:
    status = {
        "job_id": "12345",
        "rows": [
            "12345.batch|FAILED|00:00:02||",
            "12345|COMPLETED|00:00:02||",
        ],
    }

    assert slurm_state(status) == "COMPLETED"


def test_mutation_workers_run_distinct_lanes_concurrently(monkeypatch) -> None:
    lock = Lock()
    active = 0
    peak = 0

    def fake_propose(lane_dir: Path, *, timeout_seconds: int) -> dict[str, object]:
        nonlocal active, peak
        assert timeout_seconds == 123
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"status": "proposed", "candidate": str(lane_dir / "candidate.py")}

    monkeypatch.setattr("hsopt.exact_supervisor.propose_lane", fake_propose)
    lanes = [Path(f"/lane-{index}") for index in range(4)]

    results = _run_mutation_proposals(lanes, workers=4, timeout_seconds=123)

    assert [lane for lane, _ in results] == lanes
    assert peak == 4


def test_supervisor_collects_and_reconciles_a_completed_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, queue, lane = _campaign(tmp_path)
    batch_dir = Path(prepare_batch(queue)["batch_dir"])
    run_batch(batch_dir, backend="numpy")
    (batch_dir / "submission.json").write_text(
        json.dumps({"job_id": "12345", "host": "nvlogin", "remote_batch": "/unused"})
        + "\n"
    )

    monkeypatch.setattr(
        "hsopt.exact_supervisor.remote_status",
        lambda path: {"job_id": "12345", "rows": ["12345|COMPLETED|00:00:02||"]},
    )
    monkeypatch.setattr(
        "hsopt.exact_supervisor.collect_batch",
        lambda queue_dir, path: {
            **merge_batch(queue_dir, path),
            "job_id": "12345",
            "slurm_log_downloaded": True,
            "gpu_utilization_downloaded": True,
        },
    )

    status = supervise_once(
        root,
        repo_dir=REPO,
        mutations_enabled=False,
    )

    assert (batch_dir / "collected.json").is_file()
    assert (lane / "baseline_result.json").is_file()
    assert status["lanes"]["by_status"] == {"ready": 1}
    assert status["queue"]["completed"] == 1


def test_supervisor_forces_a_partial_batch_when_every_lane_is_waiting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, queue, _ = _campaign(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_submit_ready(queue_dir: Path, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        prepared = prepare_batch(queue_dir)
        batch_dir = Path(str(prepared["batch_dir"]))
        submission = {
            "status": "submitted",
            "job_id": "999",
            "resources": {
                "nodes": 1,
                "gpus": 1,
                "cpus": 2,
                "memory": "16G",
                "walltime": "01:00:00",
            },
        }
        (batch_dir / "submission.json").write_text(json.dumps(submission) + "\n")
        return {**prepared, **submission}

    monkeypatch.setattr("hsopt.exact_supervisor.submit_ready", fake_submit_ready)

    status = supervise_once(
        root,
        repo_dir=REPO,
        mutations_enabled=False,
        max_active_jobs=1,
        threshold=40,
    )

    assert calls[0]["force"] is True
    assert status["active_jobs"] == 1
    assert status["queue"]["pending"] == 0
    assert status["queue"]["batched_tasks"] == 1


def test_idle_replay_supervisor_forces_its_partial_gpu_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, queue, lane = _campaign(tmp_path)
    state_path = lane / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "pending_kind": None,
            "pending_task_id": None,
            "incumbent_score": 100,
        }
    )
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_submit_ready(queue_dir: Path, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        prepared = prepare_batch(queue_dir)
        batch_dir = Path(str(prepared["batch_dir"]))
        submission = {
            "status": "submitted",
            "job_id": "999",
            "resources": {
                "nodes": 1,
                "gpus": 1,
                "cpus": 2,
                "memory": "16G",
                "walltime": "03:00:00",
            },
        }
        (batch_dir / "submission.json").write_text(json.dumps(submission) + "\n")
        return {**prepared, **submission}

    monkeypatch.setattr("hsopt.exact_supervisor.submit_ready", fake_submit_ready)

    status = supervise_once(
        root,
        repo_dir=REPO,
        mutations_enabled=False,
        replay_campaign_root=tmp_path / "canonical",
        max_active_jobs=2,
        threshold=40,
    )

    assert calls[0]["force"] is True
    assert status["active_jobs"] == 1
    assert status["queue"]["pending"] == 0
    assert status["queue"]["batched_tasks"] == 1


def test_lane_computes_reference_then_iteration_zero_before_any_mutation(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue"
    lane = tmp_path / "lane"
    reference = enqueue_reference(
        queue,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
    )
    initialized = initialize_lane(
        lane,
        queue_dir=queue,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
        iterations=2,
        reference_task_id=reference["task_id"],
    )

    assert initialized["baseline_task_id"] is None
    assert json.loads((lane / "state.json").read_text())["pending_kind"] == "reference"
    assert not (lane / "baseline_result.json").exists()

    reference_batch = Path(prepare_batch(queue)["batch_dir"])
    run_batch(reference_batch, backend="numpy")
    merge_batch(queue, reference_batch)
    reference_event = reconcile_lane(lane)

    assert reference_event["status"] == "reference_ready"
    assert (lane / "exact_reference.json").is_file()
    state = json.loads((lane / "state.json").read_text())
    assert state["pending_kind"] == "baseline"
    assert state["iterations_completed"] == 0

    baseline_batch = Path(prepare_batch(queue)["batch_dir"])
    run_batch(baseline_batch, backend="numpy")
    merge_batch(queue, baseline_batch)
    baseline_event = reconcile_lane(lane)

    assert baseline_event["status"] == "baseline_ready"
    assert (lane / "baseline_result.json").is_file()
    state = json.loads((lane / "state.json").read_text())
    assert state["pending_kind"] is None
    assert state["iterations_completed"] == 0
    assert not any((lane / "iterations").iterdir())


def test_iteration_zero_retains_an_inaccurate_paper_baseline(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue"
    lane = tmp_path / "lane"
    initialized = initialize_lane(
        lane,
        queue_dir=queue,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        initial_policy=ABSOLUTE / "initial_pf.py",
        iterations=2,
    )
    baseline_batch = Path(prepare_batch(queue)["batch_dir"])
    run_batch(baseline_batch, backend="numpy")
    merge_batch(queue, baseline_batch)
    task_id = initialized["baseline_task_id"]
    result_path = queue / "completed" / task_id / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "passes_chemical_accuracy": False,
            "score": None,
            "absolute_energy_error_hartree": 0.004,
        }
    )
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    state_path = lane / "state.json"
    pre_reconcile = json.loads(state_path.read_text(encoding="utf-8"))
    pre_reconcile["last_error"] = "an earlier transient reconciliation error"
    state_path.write_text(json.dumps(pre_reconcile) + "\n", encoding="utf-8")

    baseline_event = reconcile_lane(lane)

    assert baseline_event["status"] == "baseline_ready"
    assert baseline_event["passes_chemical_accuracy"] is False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["pending_kind"] is None
    assert state["incumbent_score"] is None
    assert state["last_error"] is None
    assert json.loads((lane / "baseline_result.json").read_text(encoding="utf-8"))[
        "passes_chemical_accuracy"
    ] is False


def test_campaign_shares_one_reference_before_all_replicate_baselines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hamlib_inputs = tmp_path / "hamlib"
    reaction_inputs = tmp_path / "reactions"
    hamlib_inputs.mkdir()
    reaction_inputs.mkdir()
    (hamlib_inputs / "manifest.json").write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "instance": "h2",
                        "problem": str((ABSOLUTE / "hamiltonian.json").resolve()),
                    }
                ]
            }
        )
        + "\n"
    )
    reaction_source = REPO / "examples" / "diff_energy" / "h2_bond_stretch" / "reaction.json"
    (reaction_inputs / "manifest.json").write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "instance": "stretch",
                        "reaction": str(reaction_source.resolve()),
                    }
                ]
            }
        )
        + "\n"
    )
    seed_validation = tmp_path / "seed-validation.json"
    seed_validation.write_text("[]\n")

    def fake_generate_policy(*, output: Path, **kwargs: object) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ABSOLUTE / "initial_pf.py", output)

    monkeypatch.setattr("hsopt.campaign_setup._generate_policy", fake_generate_policy)
    campaign = initialize_campaign(
        hamlib_inputs=hamlib_inputs,
        reaction_inputs=reaction_inputs,
        seed_validation=seed_validation,
        generator=REPO / "examples" / "generate_initial_pf.py",
        campaign_root=tmp_path / "campaign",
        hamlib_replicates=5,
        iterations=50,
    )

    assert len(campaign["lanes"]) == 7
    assert len({row["reference_task_id"] for row in campaign["lanes"]}) == 2
    assert all(row["baseline_task_id"] is None for row in campaign["lanes"])
    queue = Path(campaign["queue_dir"])
    assert len(list((queue / "pending").iterdir())) == 2
    for row in campaign["lanes"]:
        state = json.loads(
            (tmp_path / "campaign" / row["lane_dir"] / "state.json").read_text()
        )
        assert state["pending_kind"] == "reference"
        assert state["iterations_completed"] == 0


def test_finalizer_runs_both_requested_export_and_report_pairs_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    exports: list[tuple[str, Path]] = []
    reports: list[tuple[str, Path, Path]] = []

    def fake_export(root: Path, family: str, output: Path) -> None:
        assert root == campaign
        exports.append((family, output))
        output.mkdir(parents=True)

    def fake_reports(family: str, export: Path, output: Path) -> None:
        reports.append((family, export, output))
        output.mkdir(parents=True)
        (output / "report_manifest.json").write_text("{}\n")

    monkeypatch.setattr("hsopt.exact_supervisor.export_family", fake_export)
    monkeypatch.setattr("hsopt.exact_supervisor.generate_reports", fake_reports)
    result = _finalize_campaign(
        campaign,
        hamlib_export_dir=tmp_path / "exports/hamlib",
        hamlib_report_dir=tmp_path / "reports/hamlib",
        reactions_export_dir=tmp_path / "exports/reactions",
        reactions_report_dir=tmp_path / "reports/reactions",
    )

    assert [family for family, _ in exports] == ["hamlib121", "reactions10"]
    assert [family for family, _, _ in reports] == ["hamlib121", "reactions10"]
    assert result["status"] == "complete"
    assert (campaign / "supervisor/finalization.json").is_file()
