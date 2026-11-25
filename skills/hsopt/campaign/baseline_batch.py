"""Build, run, and audit resumable paper-baseline calibration batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from .baseline import baseline_module


_BASELINE = baseline_module()
BASELINE_VARIANTS = _BASELINE.BASELINE_VARIANTS
calibration_mode_for_baseline = _BASELINE.calibration_mode_for_baseline
calibrate_problem = _BASELINE.calibrate_problem
seal_calibration = _BASELINE.seal_calibration
verify_calibration = _BASELINE.verify_calibration

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
    target_time: float | None,
    reference_error_hartree: float,
    total_error_hartree: float,
    shard_index: int,
    shard_count: int,
    max_steps: int,
    include_exact_reference: bool,
    exact_reference_root: Path | None = None,
    min_qubits: int | None = None,
    max_qubits: int | None = None,
) -> dict[str, Any]:
    """Run one deterministic shard, safely skipping sealed outputs."""

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must satisfy 0 <= index < shard count")
    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    if exact_reference_root is not None:
        exact_reference_root = exact_reference_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if min_qubits is not None and min_qubits <= 0:
        raise ValueError("min_qubits must be positive")
    if max_qubits is not None and max_qubits <= 0:
        raise ValueError("max_qubits must be positive")
    if min_qubits is not None and max_qubits is not None and min_qubits > max_qubits:
        raise ValueError("min_qubits cannot exceed max_qubits")
    all_records = [
        record
        for record in manifest["problems"]
        if (min_qubits is None or int(record["n_qubits"]) >= min_qubits)
        and (max_qubits is None or int(record["n_qubits"]) <= max_qubits)
    ]
    records = [
        record for index, record in enumerate(all_records) if index % shard_count == shard_index
    ]
    completed = 0
    skipped = 0
    failed = 0
    exact_references_reused = 0
    for index, record in enumerate(records, start=1):
        key = str(record["key"])
        problem = Path(str(record["problem"])).resolve()
        problem_target_time = float(json.loads(problem.read_text(encoding="utf-8"))["target_time"])
        expected_sha256 = str(record["problem_sha256"])
        output = _output_path(output_dir, record)
        failure = output.with_suffix(".failure.json")
        if _sha256(problem) != expected_sha256:
            raise ValueError(f"{key}: input checksum mismatch for {problem}")
        reusable_exact_reference: Mapping[str, Any] | None = None
        if output.is_file():
            try:
                prior = json.loads(output.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = None
            if isinstance(prior, Mapping):
                candidate_reference = prior.get("exact_reference")
                if (
                    include_exact_reference
                    and isinstance(candidate_reference, Mapping)
                    and candidate_reference.get("observable") == "exact_unitary_ground_eigenphase"
                    and candidate_reference.get("problem_sha256") == expected_sha256
                    and float(candidate_reference.get("target_time", float("nan")))
                    == problem_target_time
                ):
                    reusable_exact_reference = candidate_reference
            try:
                if not isinstance(prior, Mapping):
                    raise ValueError("calibration output is not a JSON object")
                verify_calibration(prior, problem_path=problem)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if (
                    str(prior["calibration_mode"]) == mode
                    and float(prior["instantiated_policy"]["target_time"]) == problem_target_time
                    and float(prior["reference"]["reference_error_hartree"])
                    == reference_error_hartree
                    and float(prior["instantiated_policy"]["total_error_target_hartree"])
                    == total_error_hartree
                    and bool(prior.get("exact_reference")) == include_exact_reference
                ):
                    skipped += 1
                    print(
                        f"[{index}/{len(records)}] skip verified {record['family']}/{key}",
                        flush=True,
                    )
                    continue
        if (
            include_exact_reference
            and reusable_exact_reference is None
            and exact_reference_root is not None
        ):
            reference_source = _output_path(exact_reference_root, record)
            if reference_source.is_file():
                try:
                    reference_calibration = json.loads(reference_source.read_text(encoding="utf-8"))
                    verify_calibration(reference_calibration, problem_path=problem)
                    candidate_reference = reference_calibration.get("exact_reference")
                    if not isinstance(candidate_reference, Mapping):
                        raise ValueError("calibration has no reusable exact reference")
                    if (
                        candidate_reference.get("problem_sha256") != expected_sha256
                        or float(candidate_reference.get("target_time", float("nan")))
                        != problem_target_time
                    ):
                        raise ValueError("exact reference does not match the source problem")
                    reusable_exact_reference = candidate_reference
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{record['family']}/{key}: invalid reusable exact reference "
                        f"at {reference_source}: {exc}"
                    ) from exc
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
                include_exact_reference=include_exact_reference,
                precomputed_exact_reference=reusable_exact_reference,
            )
            if reusable_exact_reference is not None:
                exact_references_reused += 1
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
        "target_time_mode": (
            "asserted_constant" if target_time is not None else "per_problem_phase_encoding"
        ),
        "reference_error_hartree": reference_error_hartree,
        "total_error_hartree": total_error_hartree,
        "includes_exact_reference": include_exact_reference,
        "min_qubits": min_qubits,
        "max_qubits": max_qubits,
        "exact_reference_root": (
            str(exact_reference_root) if exact_reference_root is not None else None
        ),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_problems": len(records),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "exact_references_reused": exact_references_reused,
        "batch_wall_seconds": time.perf_counter() - started,
    }
    _write_json(
        output_dir / f"summary-shard-{shard_index:03d}-of-{shard_count:03d}.json",
        summary,
    )
    return summary


def audit_batch(
    manifest_path: Path,
    output_dir: Path,
    *,
    expected_mode: str = "hamiltonian_bound",
    require_exact_reference: bool = True,
) -> dict[str, Any]:
    """Require one sealed source-matching result for every manifest row."""

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
            if str(payload["calibration_mode"]) != expected_mode:
                raise ValueError(f"expected {expected_mode}, found {payload['calibration_mode']}")
            if require_exact_reference and not isinstance(payload.get("exact_reference"), dict):
                raise ValueError("production calibration is missing its exact reference")
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
        "expected_mode": expected_mode,
        "requires_exact_reference": require_exact_reference,
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
    repository_root: Path | None = None,
    require_exact_reference: bool = True,
    baseline: str = "loose",
) -> dict[str, Any]:
    """Render external and optionally checked-in immutable initializers."""

    manifest_path = manifest_path.resolve()
    calibration_dir = calibration_dir.resolve()
    output_dir = output_dir.resolve()
    generator = generator.resolve()
    if repository_root is not None:
        repository_root = repository_root.resolve()
        if (repository_root / "experiments/baseline/molecules.json").is_file():
            raise ValueError("main keeps generated policies external; omit --repository-root")
        if not (repository_root / "experiments/baseline/generate_initial_pf.py").is_file():
            raise ValueError(f"not an hsopt repository root: {repository_root}")
    audit = audit_batch(
        manifest_path,
        calibration_dir,
        expected_mode=calibration_mode_for_baseline(baseline),
        require_exact_reference=require_exact_reference,
    )
    if not audit["complete"]:
        raise RuntimeError("cannot generate initial policies from an incomplete calibration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    abs_inputs = Path(str(manifest["abs_inputs"])).resolve()
    reaction_inputs = Path(str(manifest["reaction_inputs"])).resolve()
    absolute_manifest = json.loads((abs_inputs / "manifest.json").read_text(encoding="utf-8"))
    reaction_manifest = json.loads((reaction_inputs / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []

    def render(family: str, key: str, source: Path) -> None:
        policy_name = f"initial_pf_{baseline}.py"
        output = output_dir / family / _safe_key(key) / policy_name
        command = [
            sys.executable,
            str(generator),
            str(source.resolve()),
            "--output",
            str(output),
            "--calibration-root",
            str(calibration_dir),
            "--baseline",
            baseline,
        ]
        if output.is_file():
            subprocess.run([*command, "--check"], check=True)
            status = "verified"
        else:
            subprocess.run(command, check=True)
            status = "generated"
        checked_in_policy = None
        if repository_root is not None:
            checked_in_policy = (
                repository_root / "experiments" / family / _safe_key(key) / policy_name
            )
            checked_in_policy.parent.mkdir(parents=True, exist_ok=True)
            if (
                checked_in_policy.is_file()
                and checked_in_policy.read_bytes() == output.read_bytes()
            ):
                checked_in_status = "verified"
            else:
                temporary = checked_in_policy.with_suffix(".py.tmp")
                shutil.copyfile(output, temporary)
                temporary.replace(checked_in_policy)
                checked_in_status = "generated"
        records.append(
            {
                "family": family,
                "key": key,
                "source": str(source.resolve()),
                "source_sha256": _sha256(source),
                "policy": str(output.relative_to(output_dir)),
                "policy_sha256": _sha256(output),
                "status": status,
                **(
                    {
                        "checked_in_policy": str(checked_in_policy.relative_to(repository_root)),
                        "checked_in_policy_sha256": _sha256(checked_in_policy),
                        "checked_in_status": checked_in_status,
                    }
                    if checked_in_policy is not None and repository_root is not None
                    else {}
                ),
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
        "baseline_variant": baseline,
        "calibration_mode": calibration_mode_for_baseline(baseline),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "calibration_dir": str(calibration_dir),
        "calibration_audit_sha256": _sha256(calibration_dir / "audit.json"),
        "generator": str(generator),
        "generator_sha256": _sha256(generator),
        "repository_root": (str(repository_root) if repository_root is not None else None),
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
        default="hamiltonian_bound",
    )
    run.add_argument(
        "--target-time",
        type=float,
        help=(
            "optional assertion for every problem's normalized target time; "
            "normally omitted because the Pauli one-norm makes it problem-specific"
        ),
    )
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
    run.add_argument("--min-qubits", type=int)
    run.add_argument("--max-qubits", type=int)
    run.add_argument(
        "--exact-reference-root",
        type=Path,
        help=(
            "reuse exact references from a separate verified calibration tree; "
            "use the loose tree when preparing the tight baseline"
        ),
    )
    run.add_argument(
        "--without-exact-reference",
        action="store_true",
        help="prepare policies only; outputs cannot initialize a production campaign",
    )

    audit = commands.add_parser("audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--output-dir", required=True, type=Path)
    audit.add_argument("--baseline", choices=BASELINE_VARIANTS, default="loose")
    audit.add_argument(
        "--without-exact-reference",
        action="store_true",
        help="audit policy-only calibrations that intentionally omit U_exact",
    )

    generate = commands.add_parser("generate")
    generate.add_argument("manifest", type=Path)
    generate.add_argument("--calibration-dir", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--generator", required=True, type=Path)
    generate.add_argument("--baseline", choices=BASELINE_VARIANTS, default="loose")
    generate.add_argument(
        "--without-exact-reference",
        action="store_true",
        help="allow policy-only calibrations that intentionally omit U_exact",
    )
    generate.add_argument(
        "--repository-root",
        type=Path,
        help=(
            "legacy-only: mirror policies into an explicitly supplied single-instance tree; "
            "main keeps generated policies external and rejects this option"
        ),
    )
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
            include_exact_reference=not args.without_exact_reference,
            exact_reference_root=args.exact_reference_root,
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
        )
    elif args.command == "audit":
        result = audit_batch(
            args.manifest,
            args.output_dir,
            expected_mode=calibration_mode_for_baseline(args.baseline),
            require_exact_reference=not args.without_exact_reference,
        )
    else:
        result = generate_policies(
            args.manifest,
            args.calibration_dir,
            args.output_dir,
            generator=args.generator,
            repository_root=args.repository_root,
            require_exact_reference=not args.without_exact_reference,
            baseline=args.baseline,
        )
    print(_json_text(result), end="")
    if args.command == "run":
        return int(result["failed"] > 0)
    if args.command == "audit":
        return int(not result["complete"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
