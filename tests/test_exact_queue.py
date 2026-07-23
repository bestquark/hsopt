from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hsopt.exact_queue import (
    enqueue,
    enqueue_reference,
    merge_batch,
    prepare_batch,
    queue_status,
    requeue_missing,
    run_batch,
)
from hsopt.nvpark_queue import submit_batch


REPO = Path(__file__).resolve().parents[1]
ABSOLUTE = REPO / "examples" / "abs_energy" / "hamlib_h2_r0_7"


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
    assert result["observable"] == "compiled_unitary_ground_eigenphase"


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

    monkeypatch.setattr("hsopt.nvpark_queue._run", fake_run)
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
    assert "uv run --no-sync" in sbatch
    assert "gpu-utilization.csv" in sbatch
    assert "uv sync --extra cupynumeric --frozen" not in sbatch


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

    monkeypatch.setattr("hsopt.nvpark_queue._run", fake_run)
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
    sbatch = next(command[-1] for command in calls if "sbatch" in command[-1])
    assert "--partition=cpu" in sbatch
    assert "--qos=normal" in sbatch
    assert "--gpus-per-node" not in sbatch
    assert "nvidia-smi" not in sbatch
    assert "LEGATE_AUTO_CONFIG=0" in sbatch
    assert "LEGATE_CONFIG" in sbatch
    assert "--backend cupynumeric" in sbatch
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

    monkeypatch.setattr("hsopt.nvpark_queue._run", fake_run)
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
    sbatch = next(command[-1] for command in calls if "sbatch" in command[-1])
    assert f"PYTHONPATH=/lustre/test/hsopt-exact/sources/{submission['source_id']}/src" in sbatch
    assert f"{runtime_python} -m hsopt.exact_queue" in sbatch
    assert "uv run --no-sync python -m hsopt.exact_queue" not in sbatch


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
