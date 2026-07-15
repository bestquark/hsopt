#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    sha256_file,
    validate_campaign_identity,
    validate_hamlib_archive,
    validate_runner_provenance,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only verification of all HamLib121 archive and member inputs."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--hamlib-data-root", type=Path, default=None)
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    manifest, identity = validate_campaign_identity(root)
    runner = validate_runner_provenance(root)
    data_root = Path(args.hamlib_data_root or manifest["hamlib_data_root"]).resolve()
    lanes = [lane for lane in manifest["lanes"] if lane.get("exact_fit")]
    if len(lanes) != 121:
        raise RuntimeError(f"paper preflight requires 121 active lanes, found {len(lanes)}")
    records = list(manifest.get("hamlib_archives", []))
    if len(records) != EXPECTED_HAMLIB_ARCHIVES:
        raise RuntimeError(
            f"paper preflight requires {EXPECTED_HAMLIB_ARCHIVES} archives, found {len(records)}"
        )
    verified_keys: set[str] = set()
    for lane in lanes:
        _path, provenance = validate_hamlib_archive(
            data_root=data_root,
            lane=lane,
            archive_records=records,
        )
        verified_keys.add(str(provenance["archive_key"]))
    if len(verified_keys) != EXPECTED_HAMLIB_ARCHIVES:
        raise RuntimeError(
            f"preflight verified {len(verified_keys)} distinct archives, "
            f"expected {EXPECTED_HAMLIB_ARCHIVES}"
        )

    receipt = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": identity["campaign_manifest_sha256"],
        "runner_provenance": runner,
        "hamlib_zip_checksums_sha256": manifest["hamlib_zip_checksums_sha256"],
        "active_lanes_verified": len(lanes),
        "archives_verified": len(verified_keys),
        "archive_keys": sorted(verified_keys),
    }
    receipt_path = root / "manifests" / "hamlib_archive_preflight.json"
    if receipt_path.exists():
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise RuntimeError(f"archive preflight receipt is unsafe: {receipt_path}")
        import json

        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("archive preflight receipt differs from the verified campaign")
    else:
        write_json(receipt_path, receipt)
    print(
        f"verified lanes={len(lanes)} archives={len(verified_keys)} "
        f"receipt_sha256={sha256_file(receipt_path)}"
    )


if __name__ == "__main__":
    main()
