#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-reactions10-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

from campaign_common import require_campaign  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(environment: Path, name: str, value: str) -> None:
    path = environment / name
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild the locked reaction evaluator environment on the cluster."
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    require_campaign(root)

    project = root / "runner" / "environment"
    environment = project / ".venv"
    runtime = root / ".runtime"
    env = os.environ.copy()
    # Never place uv's mutable, symlink-heavy cache inside the immutable runner
    # snapshot or the small cluster home directory.
    env.setdefault("UV_CACHE_DIR", str(runtime / "uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(runtime / "uv-python"))
    subprocess.run(
        ["uv", "sync", "--locked", "--project", str(project)],
        cwd=root,
        env=env,
        check=True,
    )

    python = environment / "bin" / "python"
    legate = environment / "bin" / "legate"
    if (
        environment.is_symlink()
        or not python.is_file()
        or not os.access(python, os.X_OK)
        or not legate.is_file()
        or not os.access(legate, os.X_OK)
    ):
        raise RuntimeError("locked reaction GPU environment is missing or unsafe")
    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata, numpy; "
                "assert numpy.__version__ == '1.26.4'; "
                "assert importlib.metadata.version('nvidia-cupynumeric') == '25.11.0'; "
                "assert importlib.metadata.version('scipy') == '1.17.1'"
            ),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("reaction GPU environment version preflight failed")
    write_receipt(
        environment,
        ".hsopt-uv-lock-sha256",
        sha256_file(project / "uv.lock"),
    )
    write_receipt(
        environment,
        ".hsopt-manifest-sha256",
        sha256_file(root / "manifest.json"),
    )
    print(f"prebuilt locked reaction GPU environment under {project}")


if __name__ == "__main__":
    main()
