"""Read HamLib Pauli strings for count-only scaling, without Hilbert-space arrays.

Only one dataset's text is retained at once. The JW catalog stores canonical,
unique Pauli labels; this streaming reader deliberately does not combine duplicate
labels as the preparation parser does. Catalog term-count checks and small
preparation-parser parity checks audit that assumption before scaling.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import math
import os
import re
import shutil
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe catalog path: {value!r}")
    return path


def resolve_archives(catalog_path: Path, archive_root: Path) -> list[dict]:
    """Return JW catalog rows annotated with existing archive/member paths."""
    root = Path(archive_root).resolve()
    with Path(catalog_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    resolved = []
    seen = set()
    for row in rows:
        if row.get("mapping", "JW") != "JW":
            continue
        file = _relative(row["file"])
        csv_file = _relative(row["csv_file"])
        archive = root / csv_file.parent / file.with_suffix(".zip")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        key = (str(archive), row["dataset"])
        if key in seen:
            raise ValueError(f"Duplicate catalog dataset: {key}")
        seen.add(key)
        resolved.append(dict(row, archive_path=str(archive), member_name=file.name))
    if not resolved:
        raise ValueError("Catalog contains no Jordan-Wigner Hamiltonians")
    return resolved


@lru_cache(maxsize=512)
def _archive_sha256(path: str, size: int, mtime_ns: int) -> str:
    # The stat values invalidate the cache if an archive changes between calls.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_member(archive, member, digest: str, cache_dir: Path) -> Path:
    """Extract once under a digest key, sharing an advisory extraction lock."""
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{digest}.hdf5"
    lock_path = root / f"{digest}.lock"
    if destination.is_symlink() or lock_path.is_symlink():
        raise ValueError("Archive cache entries must not be symbolic links")
    # Every written basename comes from our hexadecimal digest, never a member
    # path. O_NOFOLLOW also closes the symlink race for the advisory lock.
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_symlink():
            raise ValueError("Archive cache entries must not be symbolic links")
        if destination.exists():
            if not destination.is_file() or destination.stat().st_size != member.file_size:
                raise ValueError("Cached HDF5 member has an unexpected size or file type")
            return destination
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=root, prefix=f".{digest}.", suffix=".tmp", delete=False
            ) as output:
                temporary = Path(output.name)
                with archive.open(member) as source:
                    # Reading to EOF also checks the ZIP member CRC.
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != member.file_size:
                raise ValueError("Extracted HDF5 member size differs from ZIP metadata")
            temporary.chmod(0o444)
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination


def load_text(row: dict, cache_dir: Path | None = None) -> tuple[str, str]:
    """Read only the selected scalar dataset through a seekable zip file object.

    This avoids ``archive.read(member)`` and an additional whole-HDF5 byte copy.
    With ``cache_dir``, one verified whole HDF5 member is extracted atomically
    under its archive digest and opened read-only. This avoids repeated ZIP
    decompression when many selected JW datasets share the same HDF5 file.
    The caller owns this external scratch cache and its retention policy.
    """
    import h5py

    path = Path(row["archive_path"])
    before = path.stat()
    digest = _archive_sha256(str(path), before.st_size, before.st_mtime_ns)
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or PurePosixPath(members[0].filename).name != row["member_name"]:
            raise ValueError(f"{path.name}: expected only {row['member_name']}")
        member_path = PurePosixPath(members[0].filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("Unsafe ZIP member path")
        if cache_dir is not None:
            source = _cached_member(archive, members[0], digest, cache_dir)
        else:
            source = archive.open(members[0])
        try:
            with h5py.File(source, "r") as handle:
                dataset = handle[row["dataset"]]
                if dataset.shape != ():
                    raise ValueError(f"{row['dataset']}: expected scalar Pauli text")
                if "nqubits" in dataset.attrs and int(dataset.attrs["nqubits"]) != int(
                    row["nqubits"]
                ):
                    raise ValueError("HDF5 and catalog qubit counts differ")
                value = dataset[()]
                text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        finally:
            if cache_dir is None:
                source.close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path.name}: archive changed while reading")
    return text, digest


_PAULI = re.compile(r"([XYZ])(\d+)")
_BODY = re.compile(r"(?:[XYZ]\d+)(?:\s+[XYZ]\d+)*")


def _parsed_terms(text: str, n_qubits: int):
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    start = 0
    line_number = 0
    while start < len(text):
        end = text.find("\n", start)
        if end < 0:
            end = len(text)
        line_number += 1
        line = text[start:end].strip().removesuffix("+").strip()
        start = end + 1
        if not line:
            continue
        try:
            coefficient_text, body = line.split(" [", 1)
            if not body.endswith("]"):
                raise ValueError("missing closing bracket")
            coefficient = complex(coefficient_text.strip().strip("()"))
            if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
                raise ValueError("nonfinite coefficient")
            if abs(coefficient.imag) > 1.0e-6:
                raise ValueError(f"non-real coefficient {coefficient}")
            body = body[:-1].strip()
            label = ["I"] * n_qubits
            if body:
                if _BODY.fullmatch(body) is None:
                    raise ValueError("invalid Pauli expression")
                for match in _PAULI.finditer(body):
                    pauli, index_text = match.groups()
                    index = int(index_text)
                    if not 0 <= index < n_qubits:
                        raise ValueError(f"Pauli index {index} out of range")
                    if label[index] != "I":
                        raise ValueError(f"repeated Pauli index {index}")
                    label[index] = pauli
            yield "".join(label), float(coefficient.real), abs(coefficient.imag), not body
        except (ValueError, TypeError) as error:
            raise ValueError(f"Pauli text line {line_number}: {error}") from error


def iter_terms(text: str, n_qubits: int):
    """Yield nonidentity terms in source order with the scored policy's fields.

    Labels follow canonical preparation: label[q] is the Pauli on qubit q.
    Coefficients with magnitude <=1e-15 are omitted, as in preparation.
    """
    for label, coefficient, _, identity in _parsed_terms(text, n_qubits):
        if identity or abs(coefficient) <= 1.0e-15:
            continue
        x, y, z = label.count("X"), label.count("Y"), label.count("Z")
        yield {
            "label": label,
            "coefficient": coefficient,
            "support_size": x + y + z,
            "x_count": x,
            "y_count": y,
            "z_count": z,
        }


def inspect_text(text: str, n_qubits: int, expected_terms: int | None = None) -> dict:
    """Reduce Pauli text to scalar statistics and the lexicographic PF2 center.

    ``expected_terms`` audits raw catalog lines, including identity and any terms
    subsequently dropped at the canonical 1e-15 threshold. Duplicate labels are
    assumed absent in the canonical HamLib catalog; no all-label set is retained.
    """
    identity = norm = max_coeff = max_imag = 0.0
    parsed_count = count = 0
    center_label = None
    center_coefficient = None
    for label, coefficient, imaginary, is_identity in _parsed_terms(text, n_qubits):
        parsed_count += 1
        max_imag = max(max_imag, imaginary)
        if is_identity:
            identity += coefficient
        elif abs(coefficient) > 1.0e-15:
            count += 1
            norm += abs(coefficient)
            max_coeff = max(max_coeff, abs(coefficient))
            if center_label is None or label > center_label:
                center_label, center_coefficient = label, coefficient
    if expected_terms is not None and parsed_count != int(expected_terms):
        raise ValueError(f"Catalog has {expected_terms} terms; parsed {parsed_count}")
    if count == 0:
        raise ValueError("Hamiltonian contains no nonidentity terms")
    if not math.isfinite(identity) or not math.isfinite(norm):
        raise ValueError("Hamiltonian aggregate is nonfinite")
    return {
        "identity_coefficient": identity,
        "nonidentity_terms": count,
        "parsed_terms": parsed_count,
        "max_coeff": max_coeff,
        "one_norm_nonidentity_hartree": norm,
        "center_label": center_label,
        "center_coefficient": center_coefficient,
        "maximum_abs_imaginary_coefficient": max_imag,
    }
