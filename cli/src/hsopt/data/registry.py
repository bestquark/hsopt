"""Compact input registries for shared molecular and reaction protocols.

Legacy CSV/JSON/checksum inputs remain readable for archived requests. New
checkouts keep source metadata in baseline/{molecules,reactions}.json and
materialize preparation views only in an external workspace.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath


def molecule_rows(path: Path, selection: str | None = None) -> list[dict]:
    if path.suffix != ".json":
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1:
        raise ValueError("unsupported molecular registry schema")
    name = selection or payload["default_selection"]
    if name not in payload["selections"]:
        raise ValueError(f"unknown molecular selection: {name}")
    rows = payload["selections"][name]["rows"]
    ids = [row["lane_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("molecular selection must have unique, nonempty IDs")
    return rows


def checksum_records(path: Path) -> dict[str, str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        records = payload.get("archive_checksums", payload.get("historical_cache_checksums"))
        if not isinstance(records, dict):
            raise ValueError("registry has no checksum table")
    else:
        records = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            name = name.strip().removeprefix("*")
            if name in records:
                raise ValueError(f"duplicate checksum entry: {name}")
            records[name] = digest
    for name, digest in records.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid checksum path: {name}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA-256 for {name}")
    return records


def reaction_definitions(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("reaction_definitions", payload)


def species_definitions(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("species_definitions", payload)


def validate_registries(repo: Path) -> dict:
    molecules = repo / "baseline/molecules.json"
    reactions = repo / "baseline/reactions.json"
    payload = json.loads(molecules.read_text())
    checksums = checksum_records(molecules)
    for name in payload["selections"]:
        for row in molecule_rows(molecules, name):
            archive_name = row["file"].removesuffix(".hdf5") + ".zip"
            matches = [
                archive
                for archive in checksums
                if PurePosixPath(archive).parts[0] == row["collection"]
                and PurePosixPath(archive).name == archive_name
            ]
            if len(matches) != 1:
                raise ValueError(f"{row['lane_id']}: expected one archive checksum")
            if row.get("archive_sha256", checksums[matches[0]]) != checksums[matches[0]]:
                raise ValueError(f"{row['lane_id']}: archive checksum mismatch")
    from .reactions import load_definitions

    species, definitions = load_definitions(reactions, reactions)
    historical = json.loads(reactions.read_text())["historical_metadata"]["species"]
    historical_checksums = checksum_records(reactions)
    for name in species["species"]:
        if (
            historical[name]["metadata_sha256"]
            != historical_checksums[f"problem_cache/{name}.json"]
        ):
            raise ValueError(f"{name}: historical metadata checksum mismatch")
        if f"problem_cache/{name}.npz" not in historical_checksums:
            raise ValueError(f"{name}: missing historical coefficients checksum")
    return {
        "molecular_hamiltonians": len(molecule_rows(molecules)),
        "molecular_selections": {
            name: len(molecule_rows(molecules, name)) for name in payload["selections"]
        },
        "reaction_species": len(species["species"]),
        "reactions": len(definitions["reaction_keys"]),
        "archive_checksums": len(checksums),
    }


def export_registries(repo: Path, output: Path, *, selection: str = "elote115") -> dict:
    """Export immutable preparation views, without generating Hamiltonians."""
    validate_registries(repo)
    molecules = repo / "baseline/molecules.json"
    reactions = repo / "baseline/reactions.json"
    molecular = json.loads(molecules.read_text())
    reaction = json.loads(reactions.read_text())
    rows = molecule_rows(molecules, selection)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"use a fresh metadata output directory: {output}")
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError("exported metadata belongs outside the Git checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        with (staging / "molecules.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=molecular["selections"][selection]["columns"],
            )
            writer.writeheader()
            writer.writerows(rows)
        for name, table in [
            ("hamlib_zip_sha256.txt", checksum_records(molecules)),
            ("historical_cache.sha256", checksum_records(reactions)),
        ]:
            (staging / name).write_text(
                "".join(f"{digest}  {key}\n" for key, digest in sorted(table.items()))
            )
        for name, value in [
            ("species.json", reaction["species_definitions"]),
            ("reactions.json", reaction["reaction_definitions"]),
            ("historical_metadata.json", reaction["historical_metadata"]),
        ]:
            (staging / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        provenance = {
            "schema": 1,
            "selection": selection,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (molecules, reactions)
            },
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()
            },
            "historical_cache_scope": "Audit of archived inputs; not checksums for newly generated physical Hamiltonians",
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staging.rename(output)
    return {"output_root": str(output), "selection": selection, "molecular_hamiltonians": len(rows)}
