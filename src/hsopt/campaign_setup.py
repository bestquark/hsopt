"""Prepare reproducible exact-unitary inputs and initialize autoresearch lanes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .compiled_unitary import assemble_exact_difference_reference, qpe_phase_encoding
from .exact_loop import initialize_lane
from .paper_baseline import verify_calibration


HAMLIB_BASE_URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/electronic"
PAULI_TOKEN = re.compile(r"([IXYZ])(\d+)")
ABS_ENERGY_FAMILY = "abs_energy"
DIFF_ENERGY_FAMILY = "diff_energy"
HAMLIB91_INSTANCES = 91
REACTION_INSTANCES = 10


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _attach_phase_encoding(problem: dict[str, Any]) -> dict[str, Any]:
    """Attach the explicit norm-based QPE encoding to a physical Hamiltonian."""

    encoding = qpe_phase_encoding(
        float(problem["identity_coefficient"]),
        problem["terms"],
    )
    problem["phase_encoding"] = encoding
    problem["target_time"] = float(
        encoding["physical_evolution_time_hartree_inverse"]
    )
    return problem


def _require_empty(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _calibration_catalog(root: Path) -> dict[str, dict[str, Any]]:
    """Index sealed calibrations by the exact source-Hamiltonian checksum."""

    root = root.resolve()
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            verify_calibration(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        problem_sha256 = str(payload["problem_sha256"])
        if problem_sha256 in catalog:
            if str(catalog[problem_sha256]["calibration_sha256"]) != str(
                payload["calibration_sha256"]
            ):
                raise ValueError(f"conflicting sealed calibrations for {problem_sha256}")
            continue
        catalog[problem_sha256] = payload
    if not catalog:
        raise ValueError(f"no sealed calibrations found under {root}")
    return catalog


def _calibrated_exact_reference(
    source: Path,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the exact reference stored beside the bound-derived baseline."""

    digest = _sha256(source)
    if digest not in catalog:
        raise ValueError(f"no calibration for {source} (SHA-256 {digest})")
    calibration = catalog[digest]
    verify_calibration(calibration, problem_path=source)
    if calibration.get("calibration_mode") != "hamiltonian_bound":
        raise ValueError(
            f"{source}: campaign initialization requires the canonical "
            "hamiltonian_bound baseline calibration"
        )
    reference = calibration.get("exact_reference")
    if not isinstance(reference, dict):
        raise ValueError(f"{source}: calibration has no reusable exact reference")
    return reference


def _write_absolute_reference(
    *,
    source: Path,
    output: Path,
    catalog: dict[str, dict[str, Any]],
) -> Path:
    _write_json(output, _calibrated_exact_reference(source, catalog))
    return output


def _write_reaction_reference(
    *,
    source: Path,
    output: Path,
    catalog: dict[str, dict[str, Any]],
) -> Path:
    reaction = json.loads(source.read_text(encoding="utf-8"))
    species_references: dict[str, dict[str, Any]] = {}
    backends: set[str] = set()
    for item in reaction["terms"]:
        species = str(item["species"])
        problem = (source.parent / str(item["problem"])).resolve()
        reference = _calibrated_exact_reference(problem, catalog)
        species_references[species] = reference
        backends.add(str(reference["backend"]))
    backend = next(iter(backends)) if len(backends) == 1 else "mixed_precomputed"
    _write_json(
        output,
        assemble_exact_difference_reference(
            source,
            species_references,
            backend=backend,
        ),
    )
    return output


def _checksum_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0\r\n]+)", raw)
        if match is None:
            raise ValueError(f"{path.name}:{number}: malformed checksum row")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"{path.name}:{number}: unsafe relative path")
        if name in records:
            raise ValueError(f"{path.name}:{number}: duplicate path {name}")
        records[name] = digest
    return records


def _clean_hdf5_attribute(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _parse_hamlib_archive(
    archive_path: Path,
    *,
    member_name: str,
    dataset: str,
    n_qubits: int,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    import h5py

    with zipfile.ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or PurePosixPath(members[0].filename).name != member_name:
            raise ValueError(f"{archive_path.name}: expected only {member_name}")
        payload = archive.read(members[0])
    with h5py.File(io.BytesIO(payload), "r") as handle:
        data = handle[dataset]
        text = data[()].decode("utf-8")
        attributes = {str(key): _clean_hdf5_attribute(value) for key, value in data.attrs.items()}

    identity = 0.0
    coefficients: dict[str, float] = {}
    maximum_imaginary = 0.0
    for raw in text.splitlines():
        line = raw.strip().removesuffix("+").strip()
        if not line:
            continue
        coefficient_text, pauli_text = line.split(" [", 1)
        coefficient = complex(coefficient_text.strip().strip("()"))
        maximum_imaginary = max(maximum_imaginary, abs(coefficient.imag))
        if abs(coefficient.imag) > 1.0e-6:
            raise ValueError(f"{archive_path.name}: non-real coefficient {coefficient}")
        body = pauli_text.rstrip("]").strip()
        if not body:
            identity += float(coefficient.real)
            continue
        label = ["I"] * n_qubits
        for pauli, index_text in PAULI_TOKEN.findall(body):
            index = int(index_text)
            if not 0 <= index < n_qubits:
                raise ValueError(f"{archive_path.name}: Pauli index {index} out of range")
            label[index] = pauli
        label_text = "".join(label)
        coefficients[label_text] = coefficients.get(label_text, 0.0) + float(coefficient.real)
    terms = [
        {"label": label, "coefficient": coefficient}
        for label, coefficient in sorted(coefficients.items())
        if abs(coefficient) > 1.0e-15
    ]
    return (
        identity,
        terms,
        {
            "hdf5_attributes": attributes,
            "maximum_abs_imaginary_coefficient": maximum_imaginary,
        },
    )


def prepare_hamlib(
    *,
    active_manifest: Path,
    checksum_manifest: Path,
    data_root: Path,
    output_root: Path,
    max_qubits: int = 14,
) -> dict[str, Any]:
    """Convert the exact-capable HamLib subset into compact immutable JSON."""

    output_root = _require_empty(output_root)
    rows = list(csv.DictReader(active_manifest.read_text(encoding="utf-8").splitlines()))
    selected = [row for row in rows if int(row["nqubits"]) <= max_qubits]
    if max_qubits == 14 and len(selected) != HAMLIB91_INSTANCES:
        raise ValueError(
            f"canonical HamLib <=14-qubit selection must contain "
            f"{HAMLIB91_INSTANCES} instances, found {len(selected)}"
        )
    checksums = _checksum_manifest(checksum_manifest)
    verified_archives: dict[str, str] = {}
    records = []
    for row in selected:
        archive_name = row["file"].replace(".hdf5", ".zip")
        matches = [
            relative
            for relative in checksums
            if PurePosixPath(relative).parts[0] == row["collection"]
            and PurePosixPath(relative).name == archive_name
        ]
        if len(matches) != 1:
            raise ValueError(f"{row['lane_id']}: expected one archive checksum, found {matches}")
        relative = matches[0]
        archive_path = (data_root / relative).resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        expected_digest = checksums[relative]
        actual_digest = verified_archives.get(relative)
        if actual_digest is None:
            actual_digest = _sha256(archive_path)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"{relative}: expected SHA-256 {expected_digest}, found {actual_digest}"
                )
            verified_archives[relative] = actual_digest
        identity, terms, parse_info = _parse_hamlib_archive(
            archive_path,
            member_name=row["file"],
            dataset=row["dataset"],
            n_qubits=int(row["nqubits"]),
        )
        if len(terms) + 1 != int(row["terms"]):
            raise ValueError(
                f"{row['lane_id']}: parsed {len(terms) + 1} terms, expected {row['terms']}"
            )
        one_norm = float(row["one_norm"])
        problem = _attach_phase_encoding({
            "name": f"HamLib {row['lane_id']}",
            "source": {
                "library": "HamLib",
                "base_url": HAMLIB_BASE_URL,
                "archive": relative,
                "archive_sha256": expected_digest,
                "dataset": row["dataset"],
                "active_manifest_lane_id": row["lane_id"],
                "hamlib_one_norm": one_norm,
                **parse_info,
            },
            "n_qubits": int(row["nqubits"]),
            "electrons": int(row["electrons"]),
            "identity_coefficient": identity,
            "terms": terms,
        })
        problem_path = output_root / row["lane_id"] / "hamiltonian.json"
        _write_json(problem_path, problem)
        records.append(
            {
                "instance": row["lane_id"],
                "problem": str(problem_path.relative_to(output_root)),
                "problem_sha256": _sha256(problem_path),
                "n_qubits": int(row["nqubits"]),
                "electrons": int(row["electrons"]),
                "terms": len(terms),
                "archive": relative,
                "archive_sha256": expected_digest,
            }
        )
    manifest = {
        "schema": 1,
        "family": ABS_ENERGY_FAMILY,
        "selection": f"canonical active manifest rows with nqubits <= {max_qubits}",
        "source_rows": len(rows),
        "selected_instances": len(selected),
        "max_qubits": max_qubits,
        "active_manifest": str(active_manifest.resolve()),
        "active_manifest_sha256": _sha256(active_manifest),
        "checksum_manifest": str(checksum_manifest.resolve()),
        "checksum_manifest_sha256": _sha256(checksum_manifest),
        "data_root": str(data_root.resolve()),
        "created_at": _now(),
        "instances": records,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _verify_reaction_cache(cache_root: Path, checksums: dict[str, str]) -> None:
    expected = {name.removeprefix("problem_cache/"): digest for name, digest in checksums.items()}
    actual_names = {path.name for path in cache_root.iterdir() if path.is_file()}
    if actual_names != set(expected):
        raise ValueError(
            f"reaction cache files differ: missing={sorted(set(expected) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected))}"
        )
    for name, digest in expected.items():
        actual = _sha256(cache_root / name)
        if actual != digest:
            raise ValueError(f"{name}: expected SHA-256 {digest}, found {actual}")


def prepare_reactions(
    *,
    reaction_manifest: Path,
    cache_checksums: Path,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Convert the checksummed ten-reaction cache into compact immutable JSON."""

    output_root = _require_empty(output_root)
    checksums = _checksum_manifest(cache_checksums)
    _verify_reaction_cache(cache_root, checksums)
    registry = json.loads(reaction_manifest.read_text(encoding="utf-8"))
    records = []
    for reaction_key in registry["reaction_keys"]:
        definition = registry["reactions"][reaction_key]
        reaction_dir = output_root / reaction_key
        reaction_terms = []
        for item in definition["terms"]:
            species = str(item["species"])
            metadata_path = cache_root / f"{species}.json"
            npz_path = cache_root / f"{species}.npz"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as arrays:
                labels = [str(value) for value in arrays["pauli_labels"].tolist()]
                coefficients = [float(value) for value in arrays["pauli_coefficients"].tolist()]
            if len(labels) != len(coefficients) or len(labels) != int(
                metadata["total_pauli_rotation_terms"]
            ):
                raise ValueError(f"{species}: inconsistent cached Pauli decomposition")
            problem = _attach_phase_encoding({
                "name": f"Reaction seed {species}",
                "source": {
                    "library": "HSOpt paper reaction cache",
                    "metadata_sha256": _sha256(metadata_path),
                    "npz_sha256": _sha256(npz_path),
                    "basis": metadata["basis"],
                    "geometry": metadata["geometry"],
                    "localization": metadata["localization"],
                    "active_orbitals": metadata["active_orbitals"],
                },
                "n_qubits": int(metadata["n_qubits"]),
                "electrons": int(metadata["active_electrons"]),
                "identity_coefficient": float(metadata["identity_coefficient"]),
                "terms": [
                    {"label": label, "coefficient": coefficient}
                    for label, coefficient in zip(labels, coefficients, strict=True)
                ],
            })
            relative_problem = Path("hamiltonians") / f"{species}.json"
            _write_json(reaction_dir / relative_problem, problem)
            reaction_terms.append(
                {
                    "species": species,
                    "coefficient": float(item["coefficient"]),
                    "problem": relative_problem.as_posix(),
                }
            )
        reaction = {
            "name": str(definition["label"]),
            "reaction_key": reaction_key,
            "terms": reaction_terms,
        }
        reaction_path = reaction_dir / "reaction.json"
        _write_json(reaction_path, reaction)
        records.append(
            {
                "instance": reaction_key,
                "reaction": str(reaction_path.relative_to(output_root)),
                "reaction_sha256": _sha256(reaction_path),
            }
        )
    manifest = {
        "schema": 1,
        "family": DIFF_ENERGY_FAMILY,
        "selected_instances": len(records),
        "contracts": ["equal_share", "reaction_aware"],
        "reaction_manifest": str(reaction_manifest.resolve()),
        "reaction_manifest_sha256": _sha256(reaction_manifest),
        "cache_checksums": str(cache_checksums.resolve()),
        "cache_checksums_sha256": _sha256(cache_checksums),
        "created_at": _now(),
        "instances": records,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _generate_policy(
    *,
    generator: Path,
    source: Path,
    output: Path,
    calibration_root: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(generator),
        str(source),
        "--output",
        str(output),
        "--calibration-root",
        str(calibration_root),
    ]
    subprocess.run(command, check=True)


def initialize_campaign(
    *,
    hamlib_inputs: Path,
    reaction_inputs: Path,
    calibration_root: Path,
    generator: Path,
    campaign_root: Path,
    hamlib_replicates: int = 5,
    iterations: int = 50,
    expected_hamlib_instances: int = HAMLIB91_INSTANCES,
    expected_reaction_instances: int = REACTION_INSTANCES,
) -> dict[str, Any]:
    """Create lanes from calibration-cached immutable exact-unitary references."""

    campaign_root = _require_empty(campaign_root)
    queue_dir = campaign_root / "queue"
    lanes_root = campaign_root / "lanes"
    policies_root = campaign_root / "policies"
    references_root = campaign_root / "references"
    calibration_catalog = _calibration_catalog(calibration_root)
    hamlib_manifest = json.loads((hamlib_inputs / "manifest.json").read_text(encoding="utf-8"))
    reaction_manifest = json.loads((reaction_inputs / "manifest.json").read_text(encoding="utf-8"))
    if hamlib_manifest.get("family") != ABS_ENERGY_FAMILY:
        raise ValueError("HamLib input manifest must use family 'abs_energy'")
    if len(hamlib_manifest.get("instances", [])) != expected_hamlib_instances:
        raise ValueError(
            f"abs_energy requires exactly {expected_hamlib_instances} HamLib instances"
        )
    if reaction_manifest.get("family") != DIFF_ENERGY_FAMILY:
        raise ValueError("reaction input manifest must use family 'diff_energy'")
    if len(reaction_manifest.get("instances", [])) != expected_reaction_instances:
        raise ValueError(
            f"diff_energy requires exactly {expected_reaction_instances} reaction instances"
        )
    lanes = []
    for record in hamlib_manifest["instances"]:
        instance = str(record["instance"])
        source = hamlib_inputs / str(record["problem"])
        reference_path = _write_absolute_reference(
            source=source,
            output=references_root / ABS_ENERGY_FAMILY / f"{instance}.json",
            catalog=calibration_catalog,
        )
        policy = policies_root / ABS_ENERGY_FAMILY / instance / "initial_pf.py"
        _generate_policy(
            generator=generator,
            source=source,
            output=policy,
            calibration_root=calibration_root,
        )
        for replicate_index in range(1, hamlib_replicates + 1):
            replicate = f"run_{replicate_index:02d}"
            lane_dir = lanes_root / ABS_ENERGY_FAMILY / instance / replicate
            initialized = initialize_lane(
                lane_dir,
                queue_dir=queue_dir,
                kind="absolute",
                source=source,
                initial_policy=policy,
                iterations=iterations,
                family=ABS_ENERGY_FAMILY,
                replicate=replicate,
                instance=instance,
                exact_reference_path=reference_path,
            )
            lanes.append(
                {
                    "family": ABS_ENERGY_FAMILY,
                    "instance": instance,
                    "replicate": replicate,
                    "lane_dir": str(lane_dir.relative_to(campaign_root)),
                    "reference_task_id": None,
                    "reference_sha256": _sha256(reference_path),
                    "baseline_task_id": initialized["baseline_task_id"],
                }
            )
    for record in reaction_manifest["instances"]:
        instance = str(record["instance"])
        source = reaction_inputs / str(record["reaction"])
        reference_path = _write_reaction_reference(
            source=source,
            output=references_root / DIFF_ENERGY_FAMILY / f"{instance}.json",
            catalog=calibration_catalog,
        )
        for contract in ("equal_share", "reaction_aware"):
            policy = policies_root / DIFF_ENERGY_FAMILY / instance / contract / "initial_pf.py"
            _generate_policy(
                generator=generator,
                source=source,
                output=policy,
                calibration_root=calibration_root,
            )
            lane_dir = lanes_root / DIFF_ENERGY_FAMILY / instance / contract
            initialized = initialize_lane(
                lane_dir,
                queue_dir=queue_dir,
                kind="difference",
                source=source,
                initial_policy=policy,
                iterations=iterations,
                family=DIFF_ENERGY_FAMILY,
                replicate=contract,
                instance=instance,
                exact_reference_path=reference_path,
            )
            lanes.append(
                {
                    "family": DIFF_ENERGY_FAMILY,
                    "instance": instance,
                    "replicate": contract,
                    "lane_dir": str(lane_dir.relative_to(campaign_root)),
                    "reference_task_id": None,
                    "reference_sha256": _sha256(reference_path),
                    "baseline_task_id": initialized["baseline_task_id"],
                }
            )
    manifest = {
        "schema": 1,
        "created_at": _now(),
        "iterations": iterations,
        "queue_dir": str(queue_dir),
        "hamlib": {
            "instances": len(hamlib_manifest["instances"]),
            "replicates": hamlib_replicates,
            "input_manifest_sha256": _sha256(hamlib_inputs / "manifest.json"),
        },
        "reactions": {
            "instances": len(reaction_manifest["instances"]),
            "contracts": ["equal_share", "reaction_aware"],
            "input_manifest_sha256": _sha256(reaction_inputs / "manifest.json"),
        },
        "calibration_root": str(calibration_root.resolve()),
        "generator": str(generator.resolve()),
        "generator_sha256": _sha256(generator),
        "lanes": lanes,
    }
    _write_json(campaign_root / "campaign.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    hamlib = commands.add_parser("prepare-hamlib")
    hamlib.add_argument("--active-manifest", type=Path, required=True)
    hamlib.add_argument("--checksum-manifest", type=Path, required=True)
    hamlib.add_argument("--data-root", type=Path, required=True)
    hamlib.add_argument("--output-root", type=Path, required=True)
    hamlib.add_argument("--max-qubits", type=int, default=14)

    reactions = commands.add_parser("prepare-reactions")
    reactions.add_argument("--reaction-manifest", type=Path, required=True)
    reactions.add_argument("--cache-checksums", type=Path, required=True)
    reactions.add_argument("--cache-root", type=Path, required=True)
    reactions.add_argument("--output-root", type=Path, required=True)

    initialize = commands.add_parser("initialize")
    initialize.add_argument("--hamlib-inputs", type=Path, required=True)
    initialize.add_argument("--reaction-inputs", type=Path, required=True)
    initialize.add_argument("--calibration-root", type=Path, required=True)
    initialize.add_argument("--generator", type=Path, required=True)
    initialize.add_argument("--campaign-root", type=Path, required=True)
    initialize.add_argument("--hamlib-replicates", type=int, default=5)
    initialize.add_argument("--iterations", type=int, default=50)
    return parser


def main() -> int:
    args = _parser().parse_args()
    values = vars(args)
    command = values.pop("command")
    if command == "prepare-hamlib":
        result = prepare_hamlib(**values)
    elif command == "prepare-reactions":
        result = prepare_reactions(**values)
    else:
        result = initialize_campaign(**values)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
