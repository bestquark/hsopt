"""Registry migration must preserve prepared Hamiltonians and source definitions."""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from hsopt.campaign.setup import prepare_hamlib
from hsopt.data.acquire import selected_archives
from hsopt.data.registry import export_registries, molecule_rows, validate_registries

REPO = Path(__file__).resolve().parents[1]


def test_external_exports_preserve_every_selection_and_reaction_definition(tmp_path):
    molecules = json.loads((REPO / "experiments/baseline/molecules.json").read_text())
    reactions = json.loads((REPO / "experiments/baseline/reactions.json").read_text())
    for selection in molecules["selections"]:
        output = tmp_path / selection
        export_registries(REPO, output, selection=selection)
        assert molecule_rows(output / "molecules.csv") == molecule_rows(
            REPO / "experiments/baseline/molecules.json",
            selection,
        )
        assert json.loads((output / "species.json").read_text()) == reactions["species_definitions"]
        assert (
            json.loads((output / "reactions.json").read_text()) == reactions["reaction_definitions"]
        )
        assert (
            json.loads((output / "historical_metadata.json").read_text())
            == reactions["historical_metadata"]
        )
        provenance = json.loads((output / "provenance.json").read_text())
        for name, digest in provenance["files"].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    legacy = tmp_path / "single_molecule82"
    assert len(selected_archives(legacy / "molecules.csv", legacy / "hamlib_zip_sha256.txt")) == 60
    result = validate_registries(REPO)
    assert result["molecular_selections"] == {
        "elote115": 115,
        "generalization100": 100,
        "single_molecule82": 82,
    }


def test_export_never_overwrites_an_existing_workspace(tmp_path):
    output = tmp_path / "sealed"
    output.mkdir()
    sentinel = output / "candidate.py"
    sentinel.write_text("archived policy")
    with pytest.raises(FileExistsError, match="fresh metadata"):
        export_registries(REPO, output)
    assert sentinel.read_text() == "archived policy"


def test_unknown_selection_is_rejected():
    with pytest.raises(ValueError, match="unknown molecular selection"):
        molecule_rows(REPO / "experiments/baseline/molecules.json", "missing")


def test_json_and_csv_preparation_produce_identical_hamiltonian_bytes(tmp_path):
    payload = io.BytesIO()
    with h5py.File(payload, "w") as handle:
        data = handle.create_dataset(
            "ham_JW", data=np.bytes_("(0.2+0j) [] +\n(0.7+0j) [Z0] +\n(0.5+0j) [X0 X1]")
        )
        data.attrs.update(nqubits=2, terms=3, one_norm=1.4)
    archive = tmp_path / "archives/standard/h2.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("h2.hdf5", payload.getvalue())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    row = {
        "lane_id": "h2",
        "collection": "standard",
        "file": "h2.hdf5",
        "dataset": "/ham_JW",
        "nqubits": "2",
        "electrons": "2",
        "one_norm": "1.4",
        "terms": "3",
    }
    registry = tmp_path / "molecules.json"
    registry.write_text(
        json.dumps(
            {
                "schema": 1,
                "default_selection": "test",
                "selections": {"test": {"columns": list(row), "rows": [row]}},
                "archive_checksums": {"standard/h2.zip": digest},
            }
        )
    )
    selection, checksums = tmp_path / "selection.csv", tmp_path / "checksums.txt"
    with selection.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    checksums.write_text(f"{digest}  standard/h2.zip\n")
    for label, source, checksum in [("json", registry, registry), ("csv", selection, checksums)]:
        prepare_hamlib(
            active_manifest=source,
            checksum_manifest=checksum,
            data_root=tmp_path / "archives",
            output_root=tmp_path / label,
            max_qubits=16,
        )
    assert (tmp_path / "json/h2/hamiltonian.json").read_bytes() == (
        tmp_path / "csv/h2/hamiltonian.json"
    ).read_bytes()
