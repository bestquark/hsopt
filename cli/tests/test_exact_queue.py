from __future__ import annotations

import hashlib
import json
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
