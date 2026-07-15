#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import stat
import sys
import tempfile
from pathlib import Path

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-hamlib121-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

from campaign_common import (  # noqa: E402
    EXPECTED_HAMLIB_ARCHIVES,
    hamlib_archive_key,
    locate_zip,
    parse_hamlib_zip_checksums,
    sha256_file,
    validate_canonical_hamlib_zip_checksum_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash the exact 83 HamLib ZIP archives used by the canonical 121 lanes."
    )
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--hamlib-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.hamlib_data_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite checksum manifest: {output}")
    with args.active_manifest.open(encoding="utf-8") as handle:
        lanes = list(csv.DictReader(handle))
    if len(lanes) != 121:
        raise RuntimeError(f"active manifest must contain 121 lanes, found {len(lanes)}")
    archive_by_key: dict[str, Path] = {}
    for lane in lanes:
        key = hamlib_archive_key(lane)
        archive = locate_zip(data_root, lane)
        if archive.is_symlink() or not archive.is_file():
            raise RuntimeError(f"{key}: archive is missing, a symlink, or not a regular file")
        previous = archive_by_key.setdefault(key, archive)
        if previous.resolve() != archive.resolve():
            raise RuntimeError(f"{key}: lanes resolve to different archive paths")
    if len(archive_by_key) != EXPECTED_HAMLIB_ARCHIVES:
        raise RuntimeError(
            f"canonical lanes must resolve to {EXPECTED_HAMLIB_ARCHIVES} archives, "
            f"found {len(archive_by_key)}"
        )

    lines: list[str] = []
    for key in sorted(archive_by_key):
        archive = archive_by_key[key]
        try:
            relative = archive.resolve().relative_to(data_root)
        except ValueError as exc:
            raise RuntimeError(f"{key}: archive resolves outside HamLib data root") from exc
        lines.append(f"{sha256_file(archive)}  {relative.as_posix()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parse_hamlib_zip_checksums(temporary, lanes)
    validate_canonical_hamlib_zip_checksum_manifest(temporary)
    os.replace(temporary, output)
    print(f"wrote {len(lines)} immutable HamLib ZIP checksums to {output}")


if __name__ == "__main__":
    main()
