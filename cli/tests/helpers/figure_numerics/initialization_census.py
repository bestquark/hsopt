"""Numerical validation extracted from archived compute_iteration0_t95.py; no plotting frontend."""

from __future__ import annotations


import csv

import hashlib

import json

from collections import Counter, defaultdict

from datetime import UTC, datetime

from pathlib import Path

from typing import Any

from hsopt.evaluation.compiled_unitary import (
    DIFFERENCE_ERROR_CONTRACTS,
    DIFFERENCE_ERROR_CONTRACT_VERSION,
    EVALUATION_CONTRACT_VERSION,
)

from hsopt.campaign.queue import enqueue, queue_status

EXPECTED_ABSOLUTE = 82

EXPECTED_REACTIONS = 10

CSV_FIELDS = (
    "family",
    "instance",
    "kind",
    "difference_error_contract",
    "task_id",
    "n_qubits",
    "max_n_qubits",
    "total_species_qubits",
    "species_count",
    "per_attempt_t_count",
    "t95",
    "passes_chemical_accuracy",
    "absolute_error_mha",
    "absolute_signed_sum_error_mha",
    "sum_absolute_species_error_mha",
    "error_cancellation_fraction",
    "qpe_repetitions",
    "species_qpe_repetitions",
    "evaluation_wall_seconds",
    "compiled_unitary_eigensolve_seconds",
    "execution_backend",
    "execution_accelerator",
    "policy_sha256",
    "reference_sha256",
    "result_sha256",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_scientific_sha256(path: Path) -> str:
    """Hash exact-reference content while excluding nondeterministic timings."""

    def strip_timings(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_timings(item)
                for key, item in value.items()
                if key not in {"evaluation_wall_seconds", "timing_seconds"}
            }
        if isinstance(value, list):
            return [strip_timings(item) for item in value]
        return value

    canonical = json.dumps(
        strip_timings(_read_json(path)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _lane_inputs(campaign_root: Path, lane: dict[str, Any]) -> dict[str, Any]:
    lane_dir = campaign_root / str(lane["lane_dir"])
    manifest = _read_json(lane_dir / "manifest.json")
    policy = lane_dir / "initial_pf.py"
    reference = lane_dir / "exact_reference.json"
    source = Path(str(manifest["source"]))
    expected_kind = "absolute" if lane["family"] == "abs_energy" else "difference"
    if manifest.get("kind") != expected_kind:
        raise ValueError(f"{lane_dir}: expected kind {expected_kind!r}")
    if manifest.get("instance") != lane["instance"]:
        raise ValueError(f"{lane_dir}: lane and manifest instance disagree")
    if manifest.get("evaluation_contract_version") != EVALUATION_CONTRACT_VERSION:
        raise ValueError(f"{lane_dir}: evaluator contract does not match the current evaluator")
    difference_error_contract = manifest.get("difference_error_contract")
    if expected_kind == "difference":
        if difference_error_contract != lane.get("replicate"):
            raise ValueError(f"{lane_dir}: lane and manifest reaction-error contracts disagree")
        if difference_error_contract not in DIFFERENCE_ERROR_CONTRACTS:
            raise ValueError(f"{lane_dir}: unknown reaction-error contract")
        if manifest.get("difference_error_contract_version") != (DIFFERENCE_ERROR_CONTRACT_VERSION):
            raise ValueError(f"{lane_dir}: unknown reaction-error contract version")
    elif difference_error_contract is not None:
        raise ValueError(f"{lane_dir}: absolute lane declares a reaction-error contract")
    for path in (source, policy, reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {
        "source_sha256": _sha256(source),
        "policy_sha256": _sha256(policy),
        "reference_sha256": _sha256(reference),
    }
    manifest_keys = {
        "source_sha256": "source_sha256",
        "policy_sha256": "initial_policy_sha256",
        "reference_sha256": "exact_reference_sha256",
    }
    for key, value in hashes.items():
        manifest_key = manifest_keys[key]
        if manifest.get(manifest_key) != value:
            raise ValueError(f"{lane_dir}: {manifest_key} does not match its file")
    return {
        "lane_dir": lane_dir,
        "manifest": manifest,
        "source": source,
        "policy": policy,
        "reference": reference,
        "reference_scientific_sha256": _reference_scientific_sha256(reference),
        "difference_error_contract": difference_error_contract,
        **hashes,
    }


def _problem_qubits(path: Path) -> int:
    value = int(_read_json(path)["n_qubits"])
    if value <= 0:
        raise ValueError(f"{path}: n_qubits must be positive")
    return value


def _source_shape(kind: str, source: Path) -> dict[str, Any]:
    if kind == "absolute":
        n_qubits = _problem_qubits(source)
        species_qubits = [n_qubits]
    else:
        reaction = _read_json(source)
        seen: set[str] = set()
        species_qubits = []
        for item in reaction["terms"]:
            species = str(item["species"])
            if species in seen:
                continue
            seen.add(species)
            species_qubits.append(_problem_qubits((source.parent / item["problem"]).resolve()))
        if not species_qubits:
            raise ValueError(f"{source}: reaction has no unique species")
        n_qubits = max(species_qubits)
    # Historical census scheduling hint, retained for manifest compatibility.
    # It is not a measured runtime model and does not enter the reported T95.
    weight = sum(8**value for value in species_qubits)
    return {
        "n_qubits": n_qubits,
        "max_n_qubits": max(species_qubits),
        "total_species_qubits": sum(species_qubits),
        "species_count": len(species_qubits),
        "species_qubits": species_qubits,
        "balance_weight": weight,
    }


def _same_across_lanes(
    instance: str,
    records: list[dict[str, Any]],
    field: str,
) -> Any:
    values = {str(record[field]) for record in records}
    if len(values) != 1:
        raise ValueError(f"{instance}: {field} differs across iteration-0 lanes")
    return records[0][field]


def prepare_census(
    *,
    campaign_root: Path,
    census_root: Path,
    expected_absolute: int = EXPECTED_ABSOLUTE,
    expected_reactions: int = EXPECTED_REACTIONS,
) -> dict[str, Any]:
    """Snapshot the unique iteration-0 tasks from an initialized campaign."""

    campaign_root = campaign_root.resolve()
    census_root = census_root.resolve()
    campaign_path = campaign_root / "campaign.json"
    campaign = _read_json(campaign_path)
    lanes = campaign.get("lanes", [])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for lane in lanes:
        grouped[(str(lane["family"]), str(lane["instance"]))].append(lane)

    counts = Counter(family for family, _ in grouped)
    expected = {"abs_energy": expected_absolute, "diff_energy": expected_reactions}
    if dict(counts) != {key: value for key, value in expected.items() if value}:
        raise ValueError(f"campaign contains {dict(counts)}, expected {expected}")

    queue_dir = census_root / "queue"
    records = []
    for (family, instance), instance_lanes in sorted(grouped.items()):
        inputs = [_lane_inputs(campaign_root, lane) for lane in instance_lanes]
        for field in (
            "source_sha256",
            "policy_sha256",
            "reference_sha256",
            "reference_scientific_sha256",
        ):
            _same_across_lanes(instance, inputs, field)
        source_paths = {str(record["source"].resolve()) for record in inputs}
        if len(source_paths) != 1:
            raise ValueError(f"{instance}: source path differs across iteration-0 lanes")
        lane_by_replicate = {
            str(lane["replicate"]): inputs[index] for index, lane in enumerate(instance_lanes)
        }
        kind = "absolute" if family == "abs_energy" else "difference"
        if family == "abs_energy":
            selections = [(None, lane_by_replicate.get("run_01", inputs[0]))]
        else:
            if set(lane_by_replicate) != set(DIFFERENCE_ERROR_CONTRACTS):
                raise ValueError(
                    f"{instance}: expected reaction contracts {DIFFERENCE_ERROR_CONTRACTS}, "
                    f"found {tuple(sorted(lane_by_replicate))}"
                )
            selections = [
                (contract, lane_by_replicate[contract]) for contract in DIFFERENCE_ERROR_CONTRACTS
            ]
        for difference_error_contract, selected in selections:
            queued = enqueue(
                queue_dir,
                kind=kind,
                source=selected["source"],
                policy=selected["policy"],
                reference=selected["reference"],
                difference_error_contract=difference_error_contract,
            )
            shape = _source_shape(kind, selected["source"])
            records.append(
                {
                    "family": family,
                    "instance": instance,
                    "kind": kind,
                    "difference_error_contract": difference_error_contract,
                    "task_id": queued["task_id"],
                    "source_original": str(selected["source"]),
                    "source_sha256": selected["source_sha256"],
                    "policy_original": str(selected["policy"]),
                    "policy_sha256": selected["policy_sha256"],
                    "reference_original": str(selected["reference"]),
                    "reference_sha256": selected["reference_sha256"],
                    "reference_scientific_sha256": selected["reference_scientific_sha256"],
                    "source_lane_count": len(instance_lanes) if family == "abs_energy" else 1,
                    **shape,
                }
            )

    task_ids = [record["task_id"] for record in records]
    if len(set(task_ids)) != len(records):
        duplicates = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
        raise ValueError(f"distinct instances collapsed to duplicate tasks: {duplicates}")
    payload = {
        "schema": 2,
        "created_at": _now(),
        "purpose": "immutable_iteration0_t95_census",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "baseline_variant": campaign.get("baseline_variant"),
        "campaign_root": str(campaign_root),
        "campaign_manifest_sha256": _sha256(campaign_path),
        "campaign_generator_sha256": campaign.get("generator_sha256"),
        "counts": {
            "absolute": counts.get("abs_energy", 0),
            "reactions": counts.get("diff_energy", 0),
            "reaction_contracts": sum(record["family"] == "diff_energy" for record in records),
            "total": len(records),
        },
        "deduplication": (
            "absolute replicates are reduced only after source, initial policy, and "
            "exact-reference hashes agree; reaction-error contracts are never deduplicated"
        ),
        "tasks": records,
    }
    manifest_path = census_root / "manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        stable_keys = (
            "evaluation_contract_version",
            "baseline_variant",
            "campaign_manifest_sha256",
            "campaign_generator_sha256",
            "counts",
            "tasks",
        )
        if any(previous.get(key) != payload.get(key) for key in stable_keys):
            raise ValueError(f"{manifest_path} describes a different census")
        return previous
    _write_json(manifest_path, payload)
    return payload


def _batch_dirs(census_root: Path) -> list[Path]:
    return sorted(
        path for path in (census_root.resolve() / "queue" / "batches").glob("*") if path.is_dir()
    )


def census_status(census_root: Path) -> dict[str, Any]:
    census_root = census_root.resolve()
    manifest = _read_json(census_root / "manifest.json")
    phases: dict[str, str] = {}
    for phase in ("pending", "completed", "failed"):
        for path in (census_root / "queue" / phase).glob("*"):
            if path.is_dir():
                phases[path.name] = phase
    for batch in _batch_dirs(census_root):
        for path in (batch / "tasks").glob("*"):
            if path.is_dir() and path.name not in phases:
                phases[path.name] = "batched"
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for record in manifest["tasks"]:
        by_family[str(record["family"])][phases.get(str(record["task_id"]), "missing")] += 1
    batches = []
    for batch in _batch_dirs(census_root):
        submission = batch / "submission.json"
        row: dict[str, Any] = {
            "batch_id": batch.name,
            "task_count": len(_read_json(batch / "batch.json")["task_ids"]),
            "submitted": submission.is_file(),
        }
        if submission.is_file():
            row["submission"] = _read_json(submission)
        batches.append(row)
    return {
        "census_root": str(census_root),
        "expected": manifest["counts"],
        "queue": queue_status(census_root / "queue"),
        "by_family": {
            family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())
        },
        "batches": batches,
    }


def _result_row(
    record: dict[str, Any], result: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    species = result.get("species", {})
    if species:
        eigensolve = sum(
            float(item["timing_seconds"]["compiled_unitary_eigensolve"])
            for item in species.values()
        )
        qpe_repetitions: int | str = ""
        species_qpe = json.dumps(
            {name: int(item["qpe_repetitions"]) for name, item in sorted(species.items())},
            sort_keys=True,
        )
        error_mha = float(result["contract_error_mha"])
        signed_sum_error_mha: float | str = float(result["absolute_difference_error_mha"])
        sum_absolute_error_mha: float | str = float(result["sum_absolute_species_error_mha"])
        cancellation_fraction: float | str = float(result["error_cancellation_fraction"])
    else:
        eigensolve = float(result["timing_seconds"]["compiled_unitary_eigensolve"])
        qpe_repetitions = int(result["qpe_repetitions"])
        species_qpe = ""
        error_mha = float(result["absolute_energy_error_mha"])
        signed_sum_error_mha = ""
        sum_absolute_error_mha = ""
        cancellation_fraction = ""
    return {
        "family": record["family"],
        "instance": record["instance"],
        "kind": record["kind"],
        "difference_error_contract": record.get("difference_error_contract") or "",
        "task_id": record["task_id"],
        "n_qubits": record["n_qubits"],
        "max_n_qubits": record["max_n_qubits"],
        "total_species_qubits": record["total_species_qubits"],
        "species_count": record["species_count"],
        "per_attempt_t_count": int(result["per_attempt_t_count"]),
        "t95": int(result["confidence_adjusted_t_count"]),
        "passes_chemical_accuracy": bool(result["passes_chemical_accuracy"]),
        "absolute_error_mha": error_mha,
        "absolute_signed_sum_error_mha": signed_sum_error_mha,
        "sum_absolute_species_error_mha": sum_absolute_error_mha,
        "error_cancellation_fraction": cancellation_fraction,
        "qpe_repetitions": qpe_repetitions,
        "species_qpe_repetitions": species_qpe,
        "evaluation_wall_seconds": float(result["evaluation_wall_seconds"]),
        "compiled_unitary_eigensolve_seconds": eigensolve,
        "execution_backend": result["execution_backend"],
        "execution_accelerator": result["execution_accelerator"],
        "policy_sha256": record["policy_sha256"],
        "reference_sha256": record["reference_sha256"],
        "result_sha256": _sha256(result_path),
    }


def export_census(census_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Export measured T95 only after all planned tasks completed successfully."""

    census_root = census_root.resolve()
    output_dir = (output_dir or census_root / "results").resolve()
    manifest = _read_json(census_root / "manifest.json")
    rows = []
    missing = []
    for record in manifest["tasks"]:
        result_path = census_root / "queue" / "completed" / record["task_id"] / "result.json"
        if not result_path.is_file():
            missing.append(str(record["task_id"]))
            continue
        result = _read_json(result_path)
        if result.get("status") != "ok":
            raise ValueError(f"{result_path}: result is not successful")
        if result.get("evaluation_contract_version") != EVALUATION_CONTRACT_VERSION:
            raise ValueError(f"{result_path}: evaluator contract changed")
        if result.get("difference_error_contract") != record.get("difference_error_contract"):
            raise ValueError(f"{result_path}: reaction-error contract changed")
        if (
            record.get("difference_error_contract") is not None
            and result.get("difference_error_contract_version") != DIFFERENCE_ERROR_CONTRACT_VERSION
        ):
            raise ValueError(f"{result_path}: reaction-error contract version changed")
        rows.append(_result_row(record, result, result_path))
    if missing:
        raise RuntimeError(f"census is incomplete: {len(missing)} tasks have no completed result")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "iteration0_t95.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    family_counts = Counter(str(row["family"]) for row in rows)
    summary = {
        "schema": 1,
        "completed_at": _now(),
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "census_manifest_sha256": _sha256(census_root / "manifest.json"),
        "row_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "all_pass_chemical_accuracy": all(row["passes_chemical_accuracy"] for row in rows),
        "csv": csv_path.name,
        "csv_sha256": _sha256(csv_path),
    }
    _write_json(output_dir / "iteration0_t95_manifest.json", summary)
    return summary
