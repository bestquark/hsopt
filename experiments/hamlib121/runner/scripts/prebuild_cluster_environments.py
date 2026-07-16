#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import stat
import subprocess
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
    sha256_file,
    validate_campaign_identity,
    validate_runner_provenance,
)


def write_receipt(environment_root: Path, name: str, value: str) -> None:
    receipt = environment_root / name
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(value + "\n", encoding="utf-8")
    os.replace(temporary, receipt)


def build(root: Path, environment: str, extras: list[str]) -> None:
    env = os.environ.copy()
    # Cluster home directories commonly have small quotas.  Keep uv's mutable
    # build cache with the external campaign instead of relying on ~/.cache.
    env.setdefault("UV_CACHE_DIR", str(root / ".cache" / "uv"))
    env["UV_PROJECT_ENVIRONMENT"] = environment
    subprocess.run(["uv", "sync", "--locked", *extras], cwd=root, env=env, check=True)
    environment_root = root / environment
    python = environment_root / "bin" / "python"
    if environment_root.is_symlink() or not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"{environment}: prebuilt Python environment is missing or unsafe")
    if environment == ".venv":
        legate = environment_root / "bin" / "legate"
        if not legate.is_file() or not os.access(legate, os.X_OK):
            raise RuntimeError("GPU environment did not install an executable Legate launcher")
    write_receipt(environment_root, ".hsopt-uv-lock-sha256", sha256_file(root / "uv.lock"))
    write_receipt(
        environment_root,
        ".hsopt-runner-provenance-sha256",
        sha256_file(root / "manifests" / "runner_provenance.json"),
    )
    write_receipt(
        environment_root,
        ".hsopt-campaign-identity-sha256",
        sha256_file(root / "manifests" / "campaign_identity.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild the locked GPU HamLib evaluator environment once."
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    validate_runner_provenance(root)
    validate_campaign_identity(root)
    build(root, ".venv", ["--extra", "gpu"])
    print(f"prebuilt locked GPU environment under {root}")


if __name__ == "__main__":
    main()
