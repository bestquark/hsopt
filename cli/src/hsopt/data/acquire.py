"""Download HamLib archives and materialize the exact paper reaction cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

HAMLIB_URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/electronic"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _checksums(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().removeprefix("*")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid checksum path: {name}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA-256 for {name}")
        if name in result:
            raise ValueError(f"duplicate checksum entry: {name}")
        result[name] = digest
    return result


def selected_archives(selection: Path, checksum_manifest: Path) -> dict[str, str]:
    """Resolve the checked-in instance selection to unique, pinned archives."""
    checksums = _checksums(checksum_manifest)
    result = {}
    with selection.open(newline="") as handle:
        for row in csv.DictReader(handle):
            filename = row["file"].removesuffix(".hdf5") + ".zip"
            matches = [
                name
                for name in checksums
                if PurePosixPath(name).parts[0] == row["collection"]
                and PurePosixPath(name).name == filename
            ]
            if len(matches) != 1:
                raise ValueError(f"{row['lane_id']}: expected one archive checksum")
            result[matches[0]] = checksums[matches[0]]
    if not result:
        raise ValueError("instance selection is empty")
    return result


def download_hamlib(
    selection: Path,
    checksum_manifest: Path,
    output_root: Path,
    *,
    base_url: str = HAMLIB_URL,
) -> dict:
    """Resume verified downloads; a failed download never replaces an archive."""
    selected = selected_archives(selection, checksum_manifest)
    output_root = output_root.expanduser().resolve()
    downloaded = 0
    for name, expected in sorted(selected.items()):
        destination = output_root / name
        if destination.exists():
            if not destination.is_file() or _sha256(destination) != expected:
                raise ValueError(f"existing archive fails SHA-256 verification: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = base_url.rstrip("/") + "/" + urllib.parse.quote(name, safe="/")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, handle)
            if _sha256(temporary) != expected:
                raise ValueError(f"download fails SHA-256 verification: {name}")
            temporary.replace(destination)
            downloaded += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {
        "output_root": str(output_root),
        "archives_verified": len(selected),
        "archives_downloaded": downloaded,
        "selection_sha256": _sha256(selection),
        "checksum_manifest_sha256": _sha256(checksum_manifest),
    }


def materialize_reactions(bundle: Path, checksum_manifest: Path, output_root: Path) -> dict:
    """Install the original reaction coefficients without regenerating chemistry."""
    checksums = _checksums(checksum_manifest)
    expected = {}
    for relative, digest in checksums.items():
        path = PurePosixPath(relative)
        if len(path.parts) != 2 or path.parts[0] != "problem_cache":
            raise ValueError(f"invalid reaction-cache path: {relative}")
        expected[path.name] = digest
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        actual = {p.name for p in output_root.iterdir()}
        if actual != set(expected):
            raise ValueError("existing reaction cache has missing or additional entries")
        for name, digest in expected.items():
            if not (output_root / name).is_file() or _sha256(output_root / name) != digest:
                raise ValueError(f"existing reaction-cache checksum mismatch: {name}")
        return {"output_root": str(output_root), "files_verified": len(expected)}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary:
        staging = Path(temporary) / "problem_cache"
        staging.mkdir()
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise ValueError("reaction bundle differs from the checked-in checksum manifest")
            for member in members:
                if not member.isfile():
                    raise ValueError(f"reaction bundle contains a non-file: {member.name}")
                payload = archive.extractfile(member)
                assert payload is not None
                with payload, (staging / member.name).open("wb") as handle:
                    shutil.copyfileobj(payload, handle)
                if _sha256(staging / member.name) != expected[member.name]:
                    raise ValueError(f"reaction bundle checksum mismatch: {member.name}")
        staging.rename(output_root)
    return {"output_root": str(output_root), "files_verified": len(expected)}


def _repo_root() -> Path:
    for parent in (Path.cwd(), *Path(__file__).resolve().parents):
        if (parent / "experiments/abs_energy/inputs/instances.csv").is_file():
            return parent
    raise FileNotFoundError("run from the HSOpt checkout or pass --repo-root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, help="HSOpt source checkout")
    commands = parser.add_subparsers(dest="command", required=True)
    hamlib = commands.add_parser("hamlib", help="download the selected, checksummed archives")
    hamlib.add_argument("--output-root", type=Path, required=True)
    hamlib.add_argument("--base-url", default=HAMLIB_URL, help="upstream or an exact-data mirror")
    reactions = commands.add_parser("reactions", help="extract the checksummed paper species")
    reactions.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve() if args.repo_root else _repo_root()
    if args.command == "hamlib":
        inputs = repo / "experiments/abs_energy/inputs"
        result = download_hamlib(
            inputs / "instances.csv",
            inputs / "hamlib_zip_sha256.txt",
            args.output_root,
            base_url=args.base_url,
        )
    else:
        inputs = repo / "experiments/diff_energy/inputs"
        result = materialize_reactions(
            inputs / "reaction_cache.tar.gz", inputs / "problem_cache.sha256", args.output_root
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
