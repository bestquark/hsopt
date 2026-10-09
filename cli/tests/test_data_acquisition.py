"""HamLib downloads must preserve the selected, checksummed payloads."""

import hashlib
from pathlib import Path

import pytest

from hsopt.data.acquire import download_hamlib, selected_archives

REPO_ROOT = Path(__file__).resolve().parents[2]


def _download_inputs(tmp_path, payload=b"original archive"):
    upstream = tmp_path / "upstream"
    archive = upstream / "standard/h2/h2.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(payload)
    selection = tmp_path / "selection.csv"
    selection.write_text("lane_id,collection,file\nh2,standard,h2.hdf5\n")
    checksum = tmp_path / "checksums.txt"
    checksum.write_text(hashlib.sha256(payload).hexdigest() + "  standard/h2/h2.zip\n")
    return upstream, selection, checksum


def test_download_resumes_without_accessing_upstream(tmp_path):
    upstream, selection, checksum = _download_inputs(tmp_path)
    output = tmp_path / "download"
    first = download_hamlib(selection, checksum, output, base_url=upstream.as_uri())
    assert first["archives_downloaded"] == 1
    (upstream / "standard/h2/h2.zip").unlink()
    second = download_hamlib(selection, checksum, output, base_url=upstream.as_uri())
    assert second["archives_verified"] == 1
    assert second["archives_downloaded"] == 0


def test_bad_download_is_not_installed(tmp_path):
    upstream, selection, checksum = _download_inputs(tmp_path)
    (upstream / "standard/h2/h2.zip").write_bytes(b"changed upstream data")
    output = tmp_path / "download"
    with pytest.raises(ValueError, match="download fails SHA-256"):
        download_hamlib(selection, checksum, output, base_url=upstream.as_uri())
    assert not (output / "standard/h2/h2.zip").exists()
    assert list((output / "standard/h2").iterdir()) == []


def test_existing_corrupt_archive_is_not_overwritten(tmp_path):
    upstream, selection, checksum = _download_inputs(tmp_path)
    output = tmp_path / "download"
    path = output / "standard/h2/h2.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt local copy")
    with pytest.raises(ValueError, match="existing archive fails"):
        download_hamlib(selection, checksum, output, base_url=upstream.as_uri())
    assert path.read_bytes() == b"corrupt local copy"


def test_selected_archives_cover_every_checked_in_instance():
    inputs = REPO_ROOT / "experiments/abs_energy/inputs"
    selected = selected_archives(inputs / "instances.csv", inputs / "hamlib_zip_sha256.txt")
    assert len(selected) == 60


def test_checksum_paths_cannot_escape_the_data_root(tmp_path):
    upstream, selection, checksum = _download_inputs(tmp_path)
    checksum.write_text(hashlib.sha256(b"archive").hexdigest() + "  ../h2.zip\n")
    with pytest.raises(ValueError, match="invalid checksum path"):
        download_hamlib(selection, checksum, tmp_path / "output", base_url=upstream.as_uri())
