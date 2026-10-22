"""Scientific task snapshots and evaluation records.

Tasks freeze the policy, Hamiltonian inputs, reference, and scoring contract.
Execution resources, dispatch, and retries belong to the external runtime.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..evaluation.compiled_unitary import (
    DIFFERENCE_ERROR_CONTRACTS,
    DIFFERENCE_ERROR_CONTRACT_VERSION,
    EVALUATION_CONTRACT_VERSION,
    evaluate_difference,
    evaluate_exact_difference_reference,
    evaluate_exact_reference,
    evaluate_molecule,
)
from ..evaluation import ground_branch, gpu_circuit


QUEUE_PHASES = ("pending", "completed", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def _enqueue_lock(queue_dir: Path):
    """Serialize content-addressed insertion for concurrent mutation workers."""

    with (queue_dir / ".enqueue.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _source_files(kind: str, source: Path) -> list[Path]:
    source = source.resolve()
    if kind == "absolute":
        return [source]
    if kind != "difference":
        raise ValueError("kind must be 'absolute' or 'difference'")
    reaction = json.loads(source.read_text(encoding="utf-8"))
    return [source, *[(source.parent / item["problem"]).resolve() for item in reaction["terms"]]]


def _task_id(
    kind: str,
    source: Path,
    *,
    stage: str,
    policy: Path | None = None,
    reference: Path | None = None,
    difference_error_contract: str | None = None,
    energy_target: str = "hf_accessible",
) -> str:
    digest = hashlib.sha256()
    digest.update(b"hsopt-exact-task-v4\0")
    digest.update(EVALUATION_CONTRACT_VERSION.encode())
    if energy_target != "hf_accessible":
        digest.update(b"\0" + ground_branch.CONTRACT_VERSION.encode())
        # A verifier upgrade must not reuse an older inconclusive task receipt.
        # The physical scoring contract remains the same; numerical provenance
        # gets its own content-addressed task identity.
        digest.update(b"\0" + gpu_circuit.CERTIFIER_VERSION.encode())
    digest.update(b"\0")
    digest.update(stage.encode())
    digest.update(kind.encode())
    digest.update(b"\0")
    digest.update((difference_error_contract or "").encode())
    if difference_error_contract is not None:
        digest.update(DIFFERENCE_ERROR_CONTRACT_VERSION.encode())
    for path in _source_files(kind, source):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if policy is not None:
        digest.update(policy.resolve().read_bytes())
    if reference is not None:
        digest.update(reference.resolve().read_bytes())
    return digest.hexdigest()[:24]


def _find_task(queue_dir: Path, task_id: str) -> tuple[str, Path] | None:
    for phase in QUEUE_PHASES:
        path = queue_dir / phase / task_id
        if path.is_dir():
            return phase, path
    for batch in sorted((queue_dir / "batches").glob("*")):
        path = batch / "tasks" / task_id
        if path.is_dir():
            return "batched", path
    return None


def _snapshot_source(kind: str, source: Path, task_dir: Path) -> str:
    source = source.resolve()
    if kind == "absolute":
        shutil.copyfile(source, task_dir / "problem.json")
        return "problem.json"

    reaction = json.loads(source.read_text(encoding="utf-8"))
    problems_dir = task_dir / "problems"
    problems_dir.mkdir()
    for index, item in enumerate(reaction["terms"]):
        original = (source.parent / str(item["problem"])).resolve()
        snapshot_name = f"{index:03d}-{original.name}"
        shutil.copyfile(original, problems_dir / snapshot_name)
        item["problem"] = f"problems/{snapshot_name}"
    _write_json(task_dir / "reaction.json", reaction)
    return "reaction.json"


def _snapshot_reference(
    kind: str,
    source: Path,
    reference: Path,
    task_dir: Path,
) -> None:
    """Copy a reference and bind reaction metadata to the staged path layout."""

    destination = task_dir / "reference.json"
    shutil.copyfile(reference, destination)
    if kind != "difference":
        return

    payload = json.loads(destination.read_text(encoding="utf-8"))
    original_sha256 = hashlib.sha256(source.resolve().read_bytes()).hexdigest()
    if payload.get("reaction_sha256") != original_sha256:
        raise ValueError(f"exact reference does not match {source.resolve()}")
    staged_reaction = task_dir / "reaction.json"
    payload["source_reaction_sha256"] = original_sha256
    payload["reaction_sha256"] = hashlib.sha256(staged_reaction.read_bytes()).hexdigest()
    payload["reaction_path"] = str(staged_reaction)
    _write_json(destination, payload)


def enqueue(
    queue_dir: Path,
    *,
    kind: str,
    source: Path,
    policy: Path,
    reference: Path | None = None,
    difference_error_contract: str | None = None,
    energy_target: str = "hf_accessible",
) -> dict[str, Any]:
    """Snapshot one candidate into the pending queue, deduplicated by content."""

    queue_dir = queue_dir.resolve()
    policy = policy.resolve()
    if energy_target not in {"hf_accessible", "ground_branch"} or (
        energy_target == "ground_branch" and (kind != "absolute" or reference is None)
    ):
        raise ValueError("ground-input tasks require an absolute-energy case and cached reference")
    if not policy.is_file():
        raise FileNotFoundError(policy)
    if reference is not None:
        reference = reference.resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
    if kind == "difference":
        if difference_error_contract not in DIFFERENCE_ERROR_CONTRACTS:
            raise ValueError(
                "difference candidates require difference_error_contract in "
                f"{DIFFERENCE_ERROR_CONTRACTS}"
            )
    elif difference_error_contract is not None:
        raise ValueError("absolute candidates cannot set difference_error_contract")
    for phase in QUEUE_PHASES:
        (queue_dir / phase).mkdir(parents=True, exist_ok=True)

    task_id = _task_id(
        kind,
        source,
        stage="candidate",
        policy=policy,
        reference=reference,
        difference_error_contract=difference_error_contract,
        energy_target=energy_target,
    )
    with _enqueue_lock(queue_dir):
        existing = _find_task(queue_dir, task_id)
        if existing is not None:
            phase, path = existing
            return {
                "status": "exists",
                "phase": phase,
                "task_id": task_id,
                "path": str(path),
            }

        temporary = queue_dir / f".enqueue-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            source_name = _snapshot_source(kind, source, temporary)
            shutil.copyfile(policy, temporary / "policy.py")
            if reference is not None:
                _snapshot_reference(kind, source, reference, temporary)
            metadata = {
                "schema": 3,
                "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
                "task_id": task_id,
                "stage": "candidate",
                "kind": kind,
                "difference_error_contract": difference_error_contract,
                "difference_error_contract_version": (
                    DIFFERENCE_ERROR_CONTRACT_VERSION
                    if difference_error_contract is not None
                    else None
                ),
                "source": source_name,
                "source_original": str(source.resolve()),
                "policy_original": str(policy),
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "reference": "reference.json" if reference is not None else None,
                "reference_original": str(reference) if reference is not None else None,
                "reference_sha256": (
                    hashlib.sha256(reference.read_bytes()).hexdigest()
                    if reference is not None
                    else None
                ),
                "reference_snapshot_sha256": (
                    hashlib.sha256((temporary / "reference.json").read_bytes()).hexdigest()
                    if reference is not None
                    else None
                ),
                "queued_at": _now(),
            }
            if energy_target != "hf_accessible":
                metadata.update(energy_target=energy_target,
                                evaluation_contract_version=ground_branch.CONTRACT_VERSION)
            _write_json(temporary / "task.json", metadata)
            destination = queue_dir / "pending" / task_id
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {"status": "queued", "phase": "pending", "task_id": task_id, "path": str(destination)}


def enqueue_reference(
    queue_dir: Path,
    *,
    kind: str,
    source: Path,
) -> dict[str, Any]:
    """Snapshot one immutable Hamiltonian reference task, deduplicated by content."""

    queue_dir = queue_dir.resolve()
    source = source.resolve()
    for phase in QUEUE_PHASES:
        (queue_dir / phase).mkdir(parents=True, exist_ok=True)
    task_id = _task_id(kind, source, stage="reference")
    with _enqueue_lock(queue_dir):
        existing = _find_task(queue_dir, task_id)
        if existing is not None:
            phase, path = existing
            return {
                "status": "exists",
                "phase": phase,
                "task_id": task_id,
                "path": str(path),
            }

        temporary = queue_dir / f".enqueue-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            source_name = _snapshot_source(kind, source, temporary)
            metadata = {
                "schema": 2,
                "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
                "task_id": task_id,
                "stage": "reference",
                "kind": kind,
                "source": source_name,
                "source_original": str(source),
                "queued_at": _now(),
            }
            _write_json(temporary / "task.json", metadata)
            destination = queue_dir / "pending" / task_id
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {"status": "queued", "phase": "pending", "task_id": task_id, "path": str(destination)}


def evaluate_task(
    task_dir: Path, *, backend: str = "numpy", accelerator: str = "unknown"
) -> dict[str, Any]:
    """Run the fixed scientific evaluator on one immutable task snapshot."""
    task_dir = task_dir.resolve()
    metadata = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    task_id = metadata["task_id"]
    task_started_at = _now()
    task_started = time.perf_counter()
    try:
        energy_target = metadata.get("energy_target", "hf_accessible")
        expected_contract = (
            ground_branch.CONTRACT_VERSION if energy_target == "ground_branch"
            else EVALUATION_CONTRACT_VERSION
        )
        if metadata.get("evaluation_contract_version") != expected_contract:
            raise ValueError(
                "queued task uses a different evaluator contract; keep old results separate"
            )
        source = task_dir / metadata["source"]
        if metadata.get("stage", "candidate") == "reference":
            if metadata["kind"] == "absolute":
                result = evaluate_exact_reference(source, backend=backend)
            else:
                result = evaluate_exact_difference_reference(source, backend=backend)
        else:
            policy = task_dir / "policy.py"
            reference = task_dir / str(metadata["reference"]) if metadata.get("reference") else None
            if metadata["kind"] == "absolute":
                result = evaluate_molecule(
                    source,
                    policy,
                    backend=backend,
                    reference_path=reference,
                    **({"energy_target": energy_target} if energy_target != "hf_accessible" else {}),
                )
            else:
                result = evaluate_difference(
                    source,
                    policy,
                    error_contract=metadata["difference_error_contract"],
                    backend=backend,
                    reference_path=reference,
                )
        result["task_id"] = task_id
    except Exception as exc:  # record the scientific evaluation failure
        from ..evaluation.spectral_window import CertifiedGroundInputInfeasible
        certified = isinstance(exc, CertifiedGroundInputInfeasible)
        result = {
            "status": "infeasible" if certified else "error",
            "task_id": task_id,
            "evaluation_contract_version": metadata.get("evaluation_contract_version"),
            "difference_error_contract": metadata.get("difference_error_contract"),
            "difference_error_contract_version": metadata.get("difference_error_contract_version"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if metadata.get("energy_target"):
            result["energy_target"] = metadata["energy_target"]
        if certified:
            result.update(
                rejection_certificate=exc.certificate,
                passes_chemical_accuracy=False,
                constraint_violation="accurate_majority_ground_input_qpe_peak",
                energy_estimate_available=False,
            )
    result.update(
        {
            "task_started_at": task_started_at,
            "task_finished_at": _now(),
            "task_wall_seconds": time.perf_counter() - task_started,
            "execution_backend": backend,
            "execution_accelerator": accelerator,
        }
    )
    return result


def queue_status(queue_dir: Path) -> dict[str, Any]:
    """Count scientific tasks without selecting an execution strategy."""
    queue_dir = queue_dir.resolve()
    return {
        "queue_dir": str(queue_dir),
        **{
            phase: sum(path.is_dir() for path in (queue_dir / phase).glob("*"))
            for phase in QUEUE_PHASES
        },
    }
