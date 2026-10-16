"""Count-only HamLib reads must match the canonical preparation convention."""

import hashlib
import importlib.util
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import pytest

from hsopt.campaign.setup import _parse_hamlib_archive


spec = importlib.util.spec_from_file_location(
    "hamlib_scaling", Path(__file__).parents[2] / "plots/_hamlib_scaling.py"
)
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


def _fixture(tmp_path, text="1.25 [] +\n-0.5 [X0 Y2] +\n(0.125+1e-10j) [Z1]\n1e-16 [X1]"):
    payload = io.BytesIO()
    with h5py.File(payload, "w") as handle:
        data = handle.create_dataset("ham_JW", data=text.encode())
        data.attrs["nqubits"] = 4
        handle.create_dataset("other_mapping", data=b"not the selected dataset")
    root = tmp_path / "archives"
    archive = root / "standard" / "example.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("folder/example.hdf5", payload.getvalue())
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "csv_file,file,dataset,mapping,nqubits,terms\n"
        "standard/chemistry.csv,example.hdf5,/ham_JW,JW,4,4\n"
    )
    return archive, catalog, root


def test_streaming_zip_hdf5_and_term_fields_match_preparation(tmp_path, monkeypatch):
    archive, catalog, root = _fixture(tmp_path)
    identity, expected, metadata = _parse_hamlib_archive(
        archive, member_name="example.hdf5", dataset="/ham_JW", n_qubits=4
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Do not materialize the whole HDF5 archive member")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden)
    row = scaling.resolve_archives(catalog, root)[0]
    text, digest = scaling.load_text(row)
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    terms = list(scaling.iter_terms(text, 4))
    assert (
        sorted(
            ({k: v for k, v in t.items() if k in {"label", "coefficient"}} for t in terms),
            key=lambda t: t["label"],
        )
        == expected
    )
    assert terms[0] == dict(
        label="XIYI", coefficient=-0.5, support_size=2, x_count=1, y_count=1, z_count=0
    )
    stats = scaling.inspect_text(text, 4, expected_terms=4)
    assert stats["identity_coefficient"] == identity
    assert stats["one_norm_nonidentity_hartree"] == 0.625
    assert stats["nonidentity_terms"] == 2
    assert stats["max_coeff"] == 0.5
    assert stats["center_label"] == "XIYI"
    assert stats["center_coefficient"] == -0.5
    assert (
        stats["maximum_abs_imaginary_coefficient"] == metadata["maximum_abs_imaginary_coefficient"]
    )


@pytest.mark.parametrize(
    "text,match",
    [
        ("(1+0.01j) [X0]", "non-real"),
        ("nan [X0]", "nonfinite"),
        ("inf [X0]", "nonfinite"),
        ("1 [X4]", "out of range"),
        ("1 [X0 Z0]", "repeated Pauli"),
        ("1 [W0]", "invalid Pauli"),
        ("1 [X0", "closing bracket"),
    ],
)
def test_invalid_coefficients_or_paulis_are_rejected(text, match):
    with pytest.raises(ValueError, match=match):
        list(scaling.iter_terms(text, 4))


def test_catalog_counts_are_audited_and_lines_need_not_end_in_newline():
    text = "\n1 [] +\n\n0.2 [X0] +\n-0.3 [Z1]"
    stats = scaling.inspect_text(text, 2, expected_terms=3)
    assert stats["parsed_terms"] == 3
    assert stats["center_label"] == "XI"
    with pytest.raises(ValueError, match="Catalog has 4 terms; parsed 3"):
        scaling.inspect_text(text, 2, expected_terms=4)


def test_wrong_hdf5_qubit_count_is_rejected(tmp_path):
    _, catalog, root = _fixture(tmp_path)
    row = scaling.resolve_archives(catalog, root)[0]
    row["nqubits"] = "8"
    with pytest.raises(ValueError, match="qubit counts differ"):
        scaling.load_text(row)


def test_unsafe_catalog_paths_are_rejected(tmp_path):
    _, catalog, root = _fixture(tmp_path)
    catalog.write_text(
        "csv_file,file,dataset,mapping,nqubits\n../outside.csv,example.hdf5,/ham_JW,JW,4\n"
    )
    with pytest.raises(ValueError, match="Unsafe catalog path"):
        scaling.resolve_archives(catalog, root)


def test_archive_cache_extracts_once_and_reuses_read_only_payload(tmp_path, monkeypatch):
    _, catalog, root = _fixture(tmp_path)
    row = scaling.resolve_archives(catalog, root)[0]
    scratch = tmp_path / "scratch"
    first, digest = scaling.load_text(row, cache_dir=scratch)
    payload = scratch / f"{digest}.hdf5"
    assert payload.is_file()
    assert payload.stat().st_mode & 0o222 == 0
    assert not list(scratch.glob("*.tmp"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached ZIP must not be decompressed twice")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    second, same_digest = scaling.load_text(row, cache_dir=scratch)
    assert (second, same_digest) == (first, digest)


def test_archive_cache_refuses_wrong_size_or_symlink(tmp_path):
    _, catalog, root = _fixture(tmp_path)
    row = scaling.resolve_archives(catalog, root)[0]
    scratch = tmp_path / "scratch"
    _, digest = scaling.load_text(row, cache_dir=scratch)
    payload = scratch / f"{digest}.hdf5"
    payload.chmod(0o600)
    payload.write_bytes(b"truncated cache")
    with pytest.raises(ValueError, match="unexpected size"):
        scaling.load_text(row, cache_dir=scratch)
    payload.unlink()
    outside = tmp_path / "outside"
    outside.write_text("must not be replaced")
    payload.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        scaling.load_text(row, cache_dir=scratch)
    assert outside.read_text() == "must not be replaced"


def test_archive_cache_never_extracts_traversal_member(tmp_path):
    archive, catalog, root = _fixture(tmp_path)
    with zipfile.ZipFile(archive) as handle:
        payload = handle.read(handle.infolist()[0])
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../example.hdf5", payload)
    row = scaling.resolve_archives(catalog, root)[0]
    with pytest.raises(ValueError, match="Unsafe ZIP member path"):
        scaling.load_text(row, cache_dir=tmp_path / "scratch")
    assert not (tmp_path / "example.hdf5").exists()


def test_concurrent_archive_readers_share_one_extraction(tmp_path, monkeypatch):
    _, catalog, root = _fixture(tmp_path)
    row = scaling.resolve_archives(catalog, root)[0]
    original = zipfile.ZipFile.open
    opens = []

    def counted(archive, *args, **kwargs):
        opens.append(1)
        return original(archive, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", counted)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: scaling.load_text(row, tmp_path / "scratch"), range(4)))
    assert results == [results[0]] * 4
    assert len(opens) == 1


def test_failed_extraction_leaves_no_partial_payload(tmp_path, monkeypatch):
    _, catalog, root = _fixture(tmp_path)
    row = scaling.resolve_archives(catalog, root)[0]

    def interrupted(source, output, **kwargs):
        output.write(b"incomplete")
        raise OSError("interrupted extraction")

    monkeypatch.setattr(scaling.shutil, "copyfileobj", interrupted)
    scratch = tmp_path / "scratch"
    with pytest.raises(OSError, match="interrupted extraction"):
        scaling.load_text(row, cache_dir=scratch)
    assert not list(scratch.glob("*.hdf5"))
    assert not list(scratch.glob("*.tmp"))
