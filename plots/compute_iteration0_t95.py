#!/usr/bin/env python3
"""Run an independent, immutable iteration-0 T95 census.

The census snapshots one initial policy and exact reference per physical
benchmark, evaluates it through ``hsopt.exact_queue``, and never reads or
changes an autoresearch incumbent.  Absolute-energy replicates and reaction
contracts are validated for byte-identical iteration-0 inputs before they are
deduplicated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hsopt.compiled_unitary import EVALUATION_CONTRACT_VERSION
from hsopt.exact_queue import enqueue, queue_status
from hsopt.nvpark_queue import (
    DEFAULT_ACCOUNT,
    DEFAULT_HOST,
    DEFAULT_PARTITION,
    DEFAULT_QOS,
    collect_batch,
    remote_status,
    submit_batch,
)


EXPECTED_ABSOLUTE = 91
EXPECTED_REACTIONS = 10
DEFAULT_JOBS = 4
DEFAULT_REMOTE_ROOT = "/lustre/fs12/portfolios/nvr/users/lmantilla/hsopt-iteration0-t95-census-v1"
CSV_FIELDS = (
    "family",
    "instance",
    "kind",
    "task_id",
    "n_qubits",
    "max_n_qubits",
    "total_species_qubits",
    "species_count",
    "per_attempt_t_count",
    "t95",
    "passes_chemical_accuracy",
    "absolute_error_mha",
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
    # Dense nonsymmetric eigensolves dominate.  The 8**n proxy is used only
    # to spread work across jobs; it does not enter the reported T95.
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
        for field in ("source_sha256", "policy_sha256", "reference_sha256"):
            _same_across_lanes(instance, inputs, field)
        source_paths = {str(record["source"].resolve()) for record in inputs}
        if len(source_paths) != 1:
            raise ValueError(f"{instance}: source path differs across iteration-0 lanes")
        preferred = "run_01" if family == "abs_energy" else "equal_share"
        lane_by_replicate = {
            str(lane["replicate"]): inputs[index] for index, lane in enumerate(instance_lanes)
        }
        selected = lane_by_replicate.get(preferred, inputs[0])
        kind = "absolute" if family == "abs_energy" else "difference"
        queued = enqueue(
            queue_dir,
            kind=kind,
            source=selected["source"],
            policy=selected["policy"],
            reference=selected["reference"],
        )
        shape = _source_shape(kind, selected["source"])
        records.append(
            {
                "family": family,
                "instance": instance,
                "kind": kind,
                "task_id": queued["task_id"],
                "source_original": str(selected["source"]),
                "source_sha256": selected["source_sha256"],
                "policy_original": str(selected["policy"]),
                "policy_sha256": selected["policy_sha256"],
                "reference_original": str(selected["reference"]),
                "reference_sha256": selected["reference_sha256"],
                "source_lane_count": len(instance_lanes),
                **shape,
            }
        )

    task_ids = [record["task_id"] for record in records]
    if len(set(task_ids)) != len(records):
        duplicates = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
        raise ValueError(f"distinct instances collapsed to duplicate tasks: {duplicates}")
    payload = {
        "schema": 1,
        "created_at": _now(),
        "purpose": "immutable_iteration0_t95_census",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "campaign_root": str(campaign_root),
        "campaign_manifest_sha256": _sha256(campaign_path),
        "campaign_generator_sha256": campaign.get("generator_sha256"),
        "counts": {
            "absolute": counts.get("abs_energy", 0),
            "reactions": counts.get("diff_energy", 0),
            "total": len(records),
        },
        "deduplication": (
            "five absolute replicates and two reaction contracts are reduced only "
            "after source, initial policy, and exact-reference hashes agree"
        ),
        "tasks": records,
    }
    manifest_path = census_root / "manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        stable_keys = (
            "evaluation_contract_version",
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


def _create_batch(queue_dir: Path, records: list[dict[str, Any]], batch_index: int) -> Path:
    batch_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-census{batch_index:02d}-{uuid.uuid4().hex[:6]}"
    batch_dir = queue_dir / "batches" / batch_id
    tasks_dir = batch_dir / "tasks"
    results_dir = batch_dir / "results"
    tasks_dir.mkdir(parents=True)
    results_dir.mkdir()
    for record in records:
        task_id = str(record["task_id"])
        source = queue_dir / "pending" / task_id
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = tasks_dir / task_id
        source.rename(destination)
        metadata_path = destination / "task.json"
        metadata = _read_json(metadata_path)
        metadata["batch_attempts"] = int(metadata.get("batch_attempts", 0)) + 1
        _write_json(metadata_path, metadata)
    _write_json(
        batch_dir / "batch.json",
        {
            "schema": 1,
            "batch_id": batch_id,
            "created_at": _now(),
            "task_ids": [record["task_id"] for record in records],
            "task_kinds": dict(sorted(Counter(record["kind"] for record in records).items())),
            "census_batch_index": batch_index,
            "balance_weight": sum(int(record["balance_weight"]) for record in records),
            "result_backend": None,
        },
    )
    return batch_dir


def prepare_balanced_batches(census_root: Path, jobs: int = DEFAULT_JOBS) -> list[Path]:
    """Greedily spread dense-eigensolve work over a fixed number of jobs."""

    census_root = census_root.resolve()
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    queue_dir = census_root / "queue"
    existing = sorted(path for path in (queue_dir / "batches").glob("*") if path.is_dir())
    if existing:
        return existing
    manifest = _read_json(census_root / "manifest.json")
    pending = {path.name for path in (queue_dir / "pending").glob("*") if path.is_dir()}
    records = [record for record in manifest["tasks"] if record["task_id"] in pending]
    if len(records) != int(manifest["counts"]["total"]):
        raise RuntimeError("balanced batches may only be created before the census starts")
    if jobs > len(records):
        raise ValueError("jobs cannot exceed the number of census tasks")

    bins: list[list[dict[str, Any]]] = [[] for _ in range(jobs)]
    loads = [0] * jobs
    for record in sorted(
        records,
        key=lambda item: (
            -int(item["balance_weight"]),
            -int(item["max_n_qubits"]),
            str(item["instance"]),
        ),
    ):
        index = min(range(jobs), key=lambda value: (loads[value], len(bins[value]), value))
        bins[index].append(record)
        loads[index] += int(record["balance_weight"])
    return [_create_batch(queue_dir, records, index) for index, records in enumerate(bins, start=1)]


def submit_census(
    *,
    census_root: Path,
    repo_dir: Path,
    jobs: int,
    host: str,
    remote_root: str,
    account: str,
    partition: str,
    qos: str,
    cpus: int,
    memory: str,
    walltime: str,
    backend: str,
    accelerator: str,
    runtime_python: str | None,
) -> list[dict[str, Any]]:
    batches = prepare_balanced_batches(census_root, jobs=jobs)
    return [
        submit_batch(
            batch,
            repo_dir=repo_dir,
            host=host,
            remote_root=remote_root,
            account=account,
            partition=partition,
            qos=qos,
            cpus=cpus,
            memory=memory,
            walltime=walltime,
            backend=backend,
            accelerator=accelerator,
            runtime_python=runtime_python,
        )
        for batch in batches
    ]


def _batch_dirs(census_root: Path) -> list[Path]:
    return sorted(
        path for path in (census_root.resolve() / "queue" / "batches").glob("*") if path.is_dir()
    )


def collect_census(census_root: Path) -> list[dict[str, Any]]:
    queue_dir = census_root.resolve() / "queue"
    return [
        collect_batch(queue_dir, batch)
        for batch in _batch_dirs(census_root)
        if (batch / "submission.json").is_file()
    ]


def census_status(census_root: Path, *, query_remote: bool = False) -> dict[str, Any]:
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
            if query_remote:
                row["remote"] = remote_status(batch)
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
        error_mha = float(result["absolute_difference_error_mha"])
    else:
        eigensolve = float(result["timing_seconds"]["compiled_unitary_eigensolve"])
        qpe_repetitions = int(result["qpe_repetitions"])
        species_qpe = ""
        error_mha = float(result["absolute_energy_error_mha"])
    return {
        "family": record["family"],
        "instance": record["instance"],
        "kind": record["kind"],
        "task_id": record["task_id"],
        "n_qubits": record["n_qubits"],
        "max_n_qubits": record["max_n_qubits"],
        "total_species_qubits": record["total_species_qubits"],
        "species_count": record["species_count"],
        "per_attempt_t_count": int(result["per_attempt_t_count"]),
        "t95": int(result["confidence_adjusted_t_count"]),
        "passes_chemical_accuracy": bool(result["passes_chemical_accuracy"]),
        "absolute_error_mha": error_mha,
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


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _add_counts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-absolute", type=int, default=EXPECTED_ABSOLUTE)
    parser.add_argument("--expected-reactions", type=int, default=EXPECTED_REACTIONS)


def _add_submit_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--qos", default=DEFAULT_QOS)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--memory", default="16G")
    parser.add_argument("--walltime", default="04:00:00")
    parser.add_argument("--backend", choices=("cupynumeric", "numpy"), default="cupynumeric")
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--runtime-python")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="snapshot the 101 immutable baselines")
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--census-root", type=Path, required=True)
    _add_counts(prepare)

    submit = commands.add_parser("submit", help="balance and submit the census batches")
    submit.add_argument("--census-root", type=Path, required=True)
    _add_submit_options(submit)

    launch = commands.add_parser("launch", help="prepare and submit in one command")
    launch.add_argument("--campaign-root", type=Path, required=True)
    launch.add_argument("--census-root", type=Path, required=True)
    _add_counts(launch)
    _add_submit_options(launch)

    status = commands.add_parser("status", help="show authoritative census progress")
    status.add_argument("--census-root", type=Path, required=True)
    status.add_argument("--remote", action="store_true")

    collect = commands.add_parser("collect", help="download and merge available results")
    collect.add_argument("--census-root", type=Path, required=True)

    export = commands.add_parser("export", help="write the complete paper-facing CSV")
    export.add_argument("--census-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path)
    return parser


def _submission_options(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "census_root",
        "repo_dir",
        "jobs",
        "host",
        "remote_root",
        "account",
        "partition",
        "qos",
        "cpus",
        "memory",
        "walltime",
        "backend",
        "accelerator",
        "runtime_python",
    )
    return {name: getattr(args, name) for name in names}


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_census(
            campaign_root=args.campaign_root,
            census_root=args.census_root,
            expected_absolute=args.expected_absolute,
            expected_reactions=args.expected_reactions,
        )
    elif args.command == "submit":
        result = submit_census(**_submission_options(args))
    elif args.command == "launch":
        prepared = prepare_census(
            campaign_root=args.campaign_root,
            census_root=args.census_root,
            expected_absolute=args.expected_absolute,
            expected_reactions=args.expected_reactions,
        )
        result = {
            "prepared": prepared["counts"],
            "submissions": submit_census(**_submission_options(args)),
        }
    elif args.command == "status":
        result = census_status(args.census_root, query_remote=args.remote)
    elif args.command == "collect":
        result = collect_census(args.census_root)
    else:
        result = export_census(args.census_root, args.output_dir)
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
