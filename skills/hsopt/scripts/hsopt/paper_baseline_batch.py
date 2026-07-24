"""Build, run, and audit resumable paper-baseline calibration batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

PAPER_BETA_REFERENCE_ERROR_HARTREE = 1.0e-4
BENCHMARK_TOTAL_ERROR_HARTREE = 1.0 / 627.5094740631


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _safe_key(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe calibration key {value!r}")
    return value


def build_manifest(
    abs_inputs: Path,
    reaction_inputs: Path,
    output: Path,
) -> dict[str, Any]:
    """List each unique Hamiltonian exactly once for CPU calibration."""

    abs_inputs = abs_inputs.resolve()
    reaction_inputs = reaction_inputs.resolve()
    absolute_manifest = json.loads((abs_inputs / "manifest.json").read_text(encoding="utf-8"))
    reaction_manifest = json.loads((reaction_inputs / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    seen_hashes: dict[str, dict[str, Any]] = {}
    for item in absolute_manifest["instances"]:
        key = _safe_key(str(item["instance"]))
        problem = (abs_inputs / str(item["problem"])).resolve()
        digest = _sha256(problem)
        record = {
            "family": "abs_energy",
            "key": key,
            "problem": str(problem),
            "problem_sha256": digest,
            "n_qubits": int(item["n_qubits"]),
            "consumers": [f"abs_energy/{key}"],
        }
        if digest in seen_hashes:
            raise ValueError(
                f"absolute Hamiltonians unexpectedly share SHA-256 {digest}: "
                f"{seen_hashes[digest]['key']} and {key}"
            )
        seen_hashes[digest] = record
        records.append(record)

    reaction_species: dict[str, str] = {}
    for item in reaction_manifest["instances"]:
        reaction_key = _safe_key(str(item["instance"]))
        reaction_path = (reaction_inputs / str(item["reaction"])).resolve()
        reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
        for term in reaction["terms"]:
            species = _safe_key(str(term["species"]))
            problem = (reaction_path.parent / str(term["problem"])).resolve()
            digest = _sha256(problem)
            if species in reaction_species and reaction_species[species] != digest:
                raise ValueError(f"reaction species {species!r} has multiple Hamiltonian hashes")
            reaction_species[species] = digest
            consumer = f"diff_energy/{reaction_key}/{species}"
            if digest in seen_hashes:
                record = seen_hashes[digest]
                if record["family"] != "diff_energy":
                    raise ValueError(f"absolute and reaction Hamiltonians share SHA-256 {digest}")
                record["consumers"].append(consumer)
                continue
            problem_payload = json.loads(problem.read_text(encoding="utf-8"))
            record = {
                "family": "diff_energy",
                "key": species,
                "problem": str(problem),
                "problem_sha256": digest,
                "n_qubits": int(problem_payload["n_qubits"]),
                "consumers": [consumer],
            }
            seen_hashes[digest] = record
            records.append(record)

    records.sort(
        key=lambda row: (
            -int(row["n_qubits"]),
            str(row["family"]),
            str(row["key"]),
        )
    )
    payload = {
        "schema": 2,
        "abs_inputs": str(abs_inputs),
        "abs_manifest_sha256": _sha256(abs_inputs / "manifest.json"),
        "reaction_inputs": str(reaction_inputs),
        "reaction_manifest_sha256": _sha256(reaction_inputs / "manifest.json"),
        "absolute_hamiltonians": sum(record["family"] == "abs_energy" for record in records),
        "reaction_hamiltonians": sum(record["family"] == "diff_energy" for record in records),
        "problems": records,
    }
    _write_json(output.resolve(), payload)
    return payload


def _output_path(output_dir: Path, record: Mapping[str, Any]) -> Path:
    return output_dir / str(record["family"]) / f"{_safe_key(str(record['key']))}.json"


def run_batch(
    manifest_path: Path,
    output_dir: Path,
    *,
    backend: str,
    mode: str,
    target_time: float,
    reference_error_hartree: float,
    total_error_hartree: float,
    shard_index: int,
    shard_count: int,
    max_steps: int,
    bound_samples: int,
    bound_seed: int,
    bound_confidence: float,
) -> dict[str, Any]:
    """Run one deterministic shard, safely skipping sealed outputs."""

    from hsopt.paper_baseline import calibrate_problem, seal_calibration, verify_calibration

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must satisfy 0 <= index < shard count")
    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_records = list(manifest["problems"])
    records = [
        record for index, record in enumerate(all_records) if index % shard_count == shard_index
    ]
    completed = 0
    skipped = 0
    failed = 0
    for index, record in enumerate(records, start=1):
        key = str(record["key"])
        problem = Path(str(record["problem"])).resolve()
        expected_sha256 = str(record["problem_sha256"])
        output = _output_path(output_dir, record)
        failure = output.with_suffix(".failure.json")
        if _sha256(problem) != expected_sha256:
            raise ValueError(f"{key}: input checksum mismatch for {problem}")
        if output.is_file():
            try:
                prior = json.loads(output.read_text(encoding="utf-8"))
                verify_calibration(prior, problem_path=problem)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                if (
                    str(prior["calibration_mode"]) == mode
                    and float(prior["instantiated_policy"]["target_time"]) == target_time
                    and float(prior["reference"]["reference_error_hartree"])
                    == reference_error_hartree
                    and float(prior["instantiated_policy"]["total_error_target_hartree"])
                    == total_error_hartree
                ):
                    skipped += 1
                    print(
                        f"[{index}/{len(records)}] skip verified {record['family']}/{key}",
                        flush=True,
                    )
                    continue
        print(
            f"[{index}/{len(records)}] calibrating {record['family']}/{key} "
            f"({record['n_qubits']} qubits)",
            flush=True,
        )
        try:
            result = calibrate_problem(
                problem,
                backend=backend,
                mode=mode,
                target_time=target_time,
                reference_error_hartree=reference_error_hartree,
                total_error_hartree=total_error_hartree,
                max_steps=max_steps,
                bound_samples=bound_samples,
                bound_seed=bound_seed,
                bound_confidence=bound_confidence,
            )
            result["dataset_key"] = key
            result["dataset_family"] = str(record["family"])
            result["n_qubits"] = int(record["n_qubits"])
            result["consumers"] = list(record["consumers"])
            # Dataset annotations are included in the checksum.
            result = seal_calibration(result)
            _write_json(output, result)
            failure.unlink(missing_ok=True)
            completed += 1
            print(
                f"[{index}/{len(records)}] completed {record['family']}/{key}: "
                f"beta={result['reference']['reference_trotter_number_beta']}, "
                f"r={result['instantiated_policy']['n_steps']}, "
                f"dt={result['instantiated_policy']['step_time']}, "
                f"epsilon={result['instantiated_policy']['uniform_epsilon']}",
                flush=True,
            )
        except Exception as exc:  # independent Hamiltonians remain runnable
            failed += 1
            _write_json(
                failure,
                {
                    "dataset_key": key,
                    "dataset_family": str(record["family"]),
                    "problem_sha256": expected_sha256,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"[{index}/{len(records)}] failed {record['family']}/{key}: {exc}",
                flush=True,
            )
    summary = {
        "schema": 2,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "backend": backend,
        "mode": mode,
        "target_time": target_time,
        "reference_error_hartree": reference_error_hartree,
        "total_error_hartree": total_error_hartree,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_problems": len(records),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "batch_wall_seconds": time.perf_counter() - started,
    }
    _write_json(
        output_dir / f"summary-shard-{shard_index:03d}-of-{shard_count:03d}.json",
        summary,
    )
    return summary


def audit_batch(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Require one sealed source-matching result for every manifest row."""

    from hsopt.paper_baseline import verify_calibration

    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    invalid: list[str] = []
    results: list[dict[str, Any]] = []
    for record in manifest["problems"]:
        output = _output_path(output_dir, record)
        label = f"{record['family']}/{record['key']}"
        if not output.is_file():
            missing.append(label)
            continue
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
            verify_calibration(
                payload,
                problem_path=Path(str(record["problem"])).resolve(),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append(f"{label}: {exc}")
            continue
        results.append(payload)
    summary = {
        "schema": 2,
        "manifest": str(manifest_path),
        "expected": len(manifest["problems"]),
        "verified": len(results),
        "missing": missing,
        "invalid": invalid,
        "complete": not missing and not invalid,
        "by_mode": {
            mode: sum(result["calibration_mode"] == mode for result in results)
            for mode in ("empirical_exact", "hamiltonian_bound")
        },
        "by_qubits": {
            str(n_qubits): sum(int(result["n_qubits"]) == n_qubits for result in results)
            for n_qubits in sorted({int(result["n_qubits"]) for result in results})
        },
    }
    _write_json(output_dir / "audit.json", summary)
    return summary


def generate_policies(
    manifest_path: Path,
    calibration_dir: Path,
    output_dir: Path,
    *,
    generator: Path,
) -> dict[str, Any]:
    """Render one immutable initializer per dataset element after a complete audit."""

    manifest_path = manifest_path.resolve()
    calibration_dir = calibration_dir.resolve()
    output_dir = output_dir.resolve()
    generator = generator.resolve()
    audit = audit_batch(manifest_path, calibration_dir)
    if not audit["complete"]:
        raise RuntimeError("cannot generate initial policies from an incomplete calibration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    abs_inputs = Path(str(manifest["abs_inputs"])).resolve()
    reaction_inputs = Path(str(manifest["reaction_inputs"])).resolve()
    absolute_manifest = json.loads((abs_inputs / "manifest.json").read_text(encoding="utf-8"))
    reaction_manifest = json.loads((reaction_inputs / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []

    def render(family: str, key: str, source: Path) -> None:
        output = output_dir / family / _safe_key(key) / "initial_pf.py"
        command = [
            sys.executable,
            str(generator),
            str(source.resolve()),
            "--output",
            str(output),
            "--calibration-root",
            str(calibration_dir),
        ]
        if output.is_file():
            subprocess.run([*command, "--check"], check=True)
            status = "verified"
        else:
            subprocess.run(command, check=True)
            status = "generated"
        records.append(
            {
                "family": family,
                "key": key,
                "source": str(source.resolve()),
                "source_sha256": _sha256(source),
                "policy": str(output.relative_to(output_dir)),
                "policy_sha256": _sha256(output),
                "status": status,
            }
        )

    for item in absolute_manifest["instances"]:
        render(
            "abs_energy",
            str(item["instance"]),
            abs_inputs / str(item["problem"]),
        )
    for item in reaction_manifest["instances"]:
        render(
            "diff_energy",
            str(item["instance"]),
            reaction_inputs / str(item["reaction"]),
        )
    index = {
        "schema": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "calibration_dir": str(calibration_dir),
        "calibration_audit_sha256": _sha256(calibration_dir / "audit.json"),
        "generator": str(generator),
        "generator_sha256": _sha256(generator),
        "absolute_initial_policies": sum(row["family"] == "abs_energy" for row in records),
        "reaction_initial_policies": sum(row["family"] == "diff_energy" for row in records),
        "policies": records,
    }
    _write_json(output_dir / "index.json", index)
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-manifest")
    build.add_argument("--abs-inputs", required=True, type=Path)
    build.add_argument("--reaction-inputs", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)

    run = commands.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--backend", choices=("numpy", "cupynumeric"), default="numpy")
    run.add_argument(
        "--mode",
        choices=("empirical_exact", "hamiltonian_bound"),
        default="empirical_exact",
    )
    run.add_argument("--target-time", type=float, default=1.0)
    run.add_argument(
        "--reference-error-hartree",
        type=float,
        default=PAPER_BETA_REFERENCE_ERROR_HARTREE,
    )
    run.add_argument(
        "--total-error-hartree",
        type=float,
        default=BENCHMARK_TOTAL_ERROR_HARTREE,
    )
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=1 << 20)
    run.add_argument("--bound-samples", type=int, default=1_000_000)
    run.add_argument("--bound-seed", type=int, default=1619152114)
    run.add_argument("--bound-confidence", type=float, default=0.99)

    audit = commands.add_parser("audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--output-dir", required=True, type=Path)

    generate = commands.add_parser("generate")
    generate.add_argument("manifest", type=Path)
    generate.add_argument("--calibration-dir", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--generator", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build-manifest":
        result = build_manifest(
            args.abs_inputs,
            args.reaction_inputs,
            args.output,
        )
    elif args.command == "run":
        result = run_batch(
            args.manifest,
            args.output_dir,
            backend=args.backend,
            mode=args.mode,
            target_time=args.target_time,
            reference_error_hartree=args.reference_error_hartree,
            total_error_hartree=args.total_error_hartree,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_steps=args.max_steps,
            bound_samples=args.bound_samples,
            bound_seed=args.bound_seed,
            bound_confidence=args.bound_confidence,
        )
    elif args.command == "audit":
        result = audit_batch(args.manifest, args.output_dir)
    else:
        result = generate_policies(
            args.manifest,
            args.calibration_dir,
            args.output_dir,
            generator=args.generator,
        )
    print(_json_text(result), end="")
    if args.command == "run":
        return int(result["failed"] > 0)
    if args.command == "audit":
        return int(not result["complete"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
