from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hsopt.campaign.queue import (
    enqueue,
    enqueue_reference,
    merge_batch,
    prepare_batch,
    queue_status,
    requeue_missing,
    run_batch,
)
from hsopt.evaluation.compiled_unitary import (
    DIFFERENCE_ERROR_CONTRACT_CANCELLATION,
    DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL,
    evaluate_exact_difference_reference,
)
from hsopt.cluster.nvpark import _system_memory_mib, submit_batch, submit_ready


REPO = Path(__file__).resolve().parents[2]
ABSOLUTE = REPO / "cli" / "tests" / "fixtures" / "abs_energy_h2"
DIFFERENCE = REPO / "cli" / "tests" / "fixtures" / "diff_energy_h2_bond_stretch"


def test_queue_deduplicates_snapshots_and_merges_results(tmp_path: Path) -> None:
    first = enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
    )
    duplicate = enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
    )

    assert first["status"] == "queued"
    assert duplicate == {**first, "status": "exists"}
    prepared = prepare_batch(tmp_path, limit=40)
    finished = run_batch(Path(prepared["batch_dir"]), backend="numpy")
    merged = merge_batch(tmp_path, Path(prepared["batch_dir"]))

    assert finished["succeeded"] == 1
    assert finished["failed"] == 0
    assert merged["status"] == "merged"
    assert queue_status(tmp_path)["completed"] == 1
    result = json.loads((tmp_path / "completed" / first["task_id"] / "result.json").read_text())
    assert result["observable"] == "compiled_unitary_lowest_hf_accessible_eigenphase"
    assert result["task_wall_seconds"] > 0.0
    assert result["execution_backend"] == "numpy"
    assert result["execution_accelerator"] == "unknown"
    assert result["task_started_at"] < result["task_finished_at"]


def test_legate_system_memory_tracks_the_slurm_request() -> None:
    assert _system_memory_mib("16G") == 12_000
    assert _system_memory_mib("64G") == 61_000


def test_concurrent_queue_insertion_remains_content_deduplicated(
    tmp_path: Path,
) -> None:
    def insert(_: int) -> dict[str, object]:
        return enqueue(
            tmp_path,
            kind="absolute",
            source=ABSOLUTE / "hamiltonian.json",
            policy=ABSOLUTE / "initial_pf.py",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(insert, range(8)))

    assert len({str(result["task_id"]) for result in results}) == 1
    assert sum(result["status"] == "queued" for result in results) == 1
    assert sum(result["status"] == "exists" for result in results) == 7
    assert queue_status(tmp_path)["pending"] == 1


def test_reference_is_a_distinct_cached_stage_before_candidate_evaluation(
    tmp_path: Path,
) -> None:
    reference = enqueue_reference(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
    )
    reference_batch = Path(prepare_batch(tmp_path)["batch_dir"])
    run_batch(reference_batch, backend="numpy")
    merge_batch(tmp_path, reference_batch)
    reference_path = tmp_path / "completed" / reference["task_id"] / "result.json"

    candidate = enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
        reference=reference_path,
    )
    candidate_task = tmp_path / "pending" / candidate["task_id"]
    metadata = json.loads((candidate_task / "task.json").read_text())

    assert metadata["stage"] == "candidate"
    assert metadata["reference"] == "reference.json"
    assert (candidate_task / "reference.json").read_bytes() == reference_path.read_bytes()
    assert queue_status(tmp_path)["completed"] == 1
    assert queue_status(tmp_path)["pending"] == 1


def test_difference_reference_is_rebound_to_the_staged_reaction(
    tmp_path: Path,
) -> None:
    reference = evaluate_exact_difference_reference(
        DIFFERENCE / "reaction.json",
        backend="numpy",
    )
    reference_path = tmp_path / "exact_reference.json"
    reference_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    candidate = enqueue(
        tmp_path / "queue",
        kind="difference",
        source=DIFFERENCE / "reaction.json",
        policy=DIFFERENCE / "initial_pf.py",
        reference=reference_path,
        difference_error_contract=DIFFERENCE_ERROR_CONTRACT_CANCELLATION,
    )
    task = tmp_path / "queue" / "pending" / candidate["task_id"]
    staged_reference = json.loads((task / "reference.json").read_text())
    staged_reaction_sha256 = hashlib.sha256((task / "reaction.json").read_bytes()).hexdigest()

    assert staged_reference["source_reaction_sha256"] == reference["reaction_sha256"]
    assert staged_reference["reaction_sha256"] == staged_reaction_sha256
    batch = Path(prepare_batch(tmp_path / "queue")["batch_dir"])
    finished = run_batch(batch, backend="numpy")
    assert finished["succeeded"] == 1
    assert finished["failed"] == 0


def test_difference_contracts_have_distinct_queue_identities(tmp_path: Path) -> None:
    queued = {
        contract: enqueue(
            tmp_path,
            kind="difference",
            source=DIFFERENCE / "reaction.json",
            policy=DIFFERENCE / "initial_pf.py",
            difference_error_contract=contract,
        )
        for contract in (
            DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL,
            DIFFERENCE_ERROR_CONTRACT_CANCELLATION,
        )
    }

    assert (
        queued[DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL]["task_id"]
        != queued[DIFFERENCE_ERROR_CONTRACT_CANCELLATION]["task_id"]
    )
    for contract, record in queued.items():
        task = tmp_path / "pending" / record["task_id"] / "task.json"
        assert json.loads(task.read_text())["difference_error_contract"] == contract


def test_nvpark_submission_requests_only_one_small_gpu_allocation(
    tmp_path: Path, monkeypatch
) -> None:
    enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
    )
    prepared = prepare_batch(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, *, cwd=None):
        calls.append(list(command))
        stdout = "12345\n" if command[:2] == ["ssh", "nvlogin"] and "sbatch" in command[-1] else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("hsopt.cluster.nvpark._run", fake_run)
    submission = submit_batch(
        Path(prepared["batch_dir"]),
        repo_dir=REPO,
        remote_root="/lustre/test/hsopt-exact",
    )

    assert submission["job_id"] == "12345"
    assert submission["resources"]["gpus"] == 1
    assert submission["resources"]["cpus"] == 2
    assert submission["resources"]["memory"] == "16G"
    sbatch = next(command[-1] for command in calls if "sbatch" in command[-1])
    assert "--gpus-per-node=1" in sbatch
    assert "--cpus-per-task=2" in sbatch
    assert "--mem=16G" in sbatch
    assert "--gpus=1" in sbatch
    assert "--fbmem=60000" in sbatch
    assert "uv run --no-sync" in sbatch
    assert "gpu-utilization.csv" in sbatch
    assert "--format=csv -l 5" in sbatch
    assert "uv sync --extra cupynumeric --frozen" not in sbatch


def test_submit_ready_uses_threshold_as_the_batch_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pending = tmp_path / "pending"
    pending.mkdir()
    for index in range(20):
        task = pending / f"task-{index:02d}"
        task.mkdir()
        (task / "task.json").write_text(
            json.dumps({"batch_attempts": 0}) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "hsopt.cluster.nvpark.submit_batch",
        lambda batch_dir, **kwargs: {
            "status": "submitted",
            "batch_id": batch_dir.name,
            "job_id": "12345",
        },
    )

    submitted = submit_ready(
        tmp_path,
        repo_dir=REPO,
        threshold=15,
    )

    assert submitted["task_count"] == 15
    assert queue_status(tmp_path)["pending"] == 5
    assert queue_status(tmp_path)["batched_tasks"] == 15


def test_submit_ready_balances_difference_tasks_when_absolute_work_is_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pending = tmp_path / "pending"
    pending.mkdir()
    for prefix, kind, count in (
        ("000-difference", "difference", 12),
        ("999-absolute", "absolute", 20),
    ):
        for index in range(count):
            task = pending / f"{prefix}-{index:02d}"
            task.mkdir()
            (task / "task.json").write_text(
                json.dumps({"batch_attempts": 0, "kind": kind}) + "\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "hsopt.cluster.nvpark.submit_batch",
        lambda batch_dir, **kwargs: {"status": "submitted", "job_id": "12345"},
    )
    submitted = submit_ready(
        tmp_path,
        repo_dir=REPO,
        threshold=15,
        max_difference_tasks=3,
    )
    manifest = json.loads((Path(submitted["batch_dir"]) / "batch.json").read_text(encoding="utf-8"))

    assert submitted["task_count"] == 15
    assert manifest["task_kinds"] == {"absolute": 12, "difference": 3}


def test_submit_ready_fills_from_differences_when_no_absolute_work_remains(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pending = tmp_path / "pending"
    pending.mkdir()
    for index in range(20):
        task = pending / f"difference-{index:02d}"
        task.mkdir()
        (task / "task.json").write_text(
            json.dumps({"batch_attempts": 0, "kind": "difference"}) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "hsopt.cluster.nvpark.submit_batch",
        lambda batch_dir, **kwargs: {"status": "submitted", "job_id": "12345"},
    )
    submitted = submit_ready(
        tmp_path,
        repo_dir=REPO,
        threshold=15,
        max_difference_tasks=3,
    )
    manifest = json.loads((Path(submitted["batch_dir"]) / "batch.json").read_text(encoding="utf-8"))

    assert submitted["task_count"] == 15
    assert manifest["task_kinds"] == {"difference": 15}


def test_nvpark_cupynumeric_submission_uses_cpu_partition_without_a_gpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
    )
    prepared = prepare_batch(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, *, cwd=None):
        calls.append(list(command))
        stdout = "54321\n" if command[:2] == ["ssh", "nvlogin"] and "sbatch" in command[-1] else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("hsopt.cluster.nvpark._run", fake_run)
    submission = submit_batch(
        Path(prepared["batch_dir"]),
        repo_dir=REPO,
        remote_root="/lustre/test/hsopt-exact",
        partition="cpu",
        qos="normal",
        backend="cupynumeric",
        accelerator="cpu",
    )

    assert submission["backend"] == "cupynumeric"
    assert submission["accelerator"] == "cpu"
    assert submission["resources"]["gpus"] == 0
    assert submission["runtime_python"] is None
    assert "--cpus=1 --gpus=0" in submission["resources"]["legate_config"]
    assert "--fbmem=0" in submission["resources"]["legate_config"]
    assert "--sysmem=12000" in submission["resources"]["legate_config"]
    sbatch = next(command[-1] for command in calls if "sbatch" in command[-1])
    assert "--partition=cpu" in sbatch
    assert "--qos=normal" in sbatch
    assert "--gpus-per-node" not in sbatch
    assert "nvidia-smi" not in sbatch
    assert "LEGATE_AUTO_CONFIG=0" in sbatch
    assert "LEGATE_CONFIG" in sbatch
    assert "--backend cupynumeric" in sbatch
    assert "--accelerator cpu" in sbatch
    assert sum("sbatch" in command[-1] for command in calls) == 1


def test_nvpark_cpu_submission_can_use_pinned_cupynumeric_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
    )
    prepared = prepare_batch(tmp_path)
    calls: list[list[str]] = []
    runtime_python = "/lustre/test/envs/cupynumeric-cpu/bin/python"

    def fake_run(command, *, cwd=None):
        calls.append(list(command))
        stdout = "54321\n" if command[:2] == ["ssh", "nvlogin"] and "sbatch" in command[-1] else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("hsopt.cluster.nvpark._run", fake_run)
    submission = submit_batch(
        Path(prepared["batch_dir"]),
        repo_dir=REPO,
        remote_root="/lustre/test/hsopt-exact",
        partition="cpu",
        qos="normal",
        backend="cupynumeric",
        accelerator="cpu",
        runtime_python=runtime_python,
    )

    assert submission["runtime_python"] == runtime_python
    assert len(submission["evaluator_source_id"]) == 20
    assert submission["evaluator_source_id"] != submission["source_id"]
    environment_sync = next(command[-1] for command in calls if "test -x" in command[-1])
    assert runtime_python in environment_sync
    assert "uv sync" not in environment_sync
    source_upload = next(command for command in calls if "./cli/src/hsopt" in command)
    assert source_upload[-2] == "./cli/src/hsopt"
    assert "--exclude=__pycache__" in source_upload
    assert "--exclude=*.py[co]" in source_upload
    sbatch = next(command[-1] for command in calls if "sbatch" in command[-1])
    runtime_root = f"/lustre/test/hsopt-exact/sources/{submission['source_id']}/cli/src"
    assert f"PYTHONPATH={runtime_root}" in sbatch
    assert f"{runtime_python} -m hsopt.campaign.queue" in sbatch
    assert "uv run --no-sync python -m hsopt.campaign.queue" not in sbatch


def test_terminal_batch_requeues_only_tasks_without_results(tmp_path: Path) -> None:
    queued = enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
    )
    prepared = prepare_batch(tmp_path)
    batch_dir = Path(prepared["batch_dir"])

    recovered = requeue_missing(tmp_path, batch_dir, reason="Slurm state TIMEOUT")

    assert recovered["requeued"] == [queued["task_id"]]
    assert recovered["exhausted"] == []
    assert (tmp_path / "pending" / queued["task_id"]).is_dir()
    assert not (batch_dir / "tasks" / queued["task_id"]).exists()


def test_terminal_batch_exhaustion_becomes_a_durable_failure(tmp_path: Path) -> None:
    queued = enqueue(
        tmp_path,
        kind="absolute",
        source=ABSOLUTE / "hamiltonian.json",
        policy=ABSOLUTE / "initial_pf.py",
    )
    batch_dir = Path(prepare_batch(tmp_path)["batch_dir"])

    recovered = requeue_missing(
        tmp_path,
        batch_dir,
        reason="Slurm state OUT_OF_MEMORY",
        max_attempts=1,
    )

    result_path = tmp_path / "failed" / queued["task_id"] / "result.json"
    assert recovered["requeued"] == []
    assert recovered["exhausted"] == [queued["task_id"]]
    assert json.loads(result_path.read_text())["error_type"] == "BatchAttemptsExhausted"
