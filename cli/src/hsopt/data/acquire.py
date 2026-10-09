"""Download HamLib archives or generate the fixed reaction species from source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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
    reactions = commands.add_parser(
        "reactions", help="generate reaction species from fixed geometries"
    )
    reactions.add_argument("--output-root", type=Path, required=True)
    reactions.add_argument(
        "--convention", choices=("physical", "historical-replay"), default="physical"
    )
    reactions.add_argument(
        "--species", nargs="+", help="generate a subset for a bounded smoke check"
    )
    reactions.add_argument("--threads", type=int, default=1, help="PySCF CPU threads (default: 1)")
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
        from hsopt.data.reactions import generate_reactions

        if args.output_root.expanduser().resolve().is_relative_to(repo):
            parser.error("generated reaction data must live outside the Git checkout")
        inputs = repo / "experiments/diff_energy/inputs"
        result = generate_reactions(
            inputs / "species.json",
            inputs / "reactions.json",
            args.output_root,
            convention=args.convention,
            species=args.species,
            threads=args.threads,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
