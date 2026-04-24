from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path


BACKEND_ENV = "TROVOLUT_PFS_ARRAY_BACKEND"
LEGATE_REEXEC_ENV = "TROVOLUT_PFS_LEGATE_REEXEC"
UNDER_LEGATE_ENV = "TROVOLUT_PFS_UNDER_LEGATE"
LEGATE_BIN_ENV = "TROVOLUT_PFS_LEGATE_BIN"
LEGATE_ARGS_ENV = "TROVOLUT_PFS_LEGATE_ARGS"
LEGATE_GPUS_ENV = "TROVOLUT_PFS_LEGATE_GPUS"
LEGATE_FBMEM_ENV = "TROVOLUT_PFS_LEGATE_FBMEM"


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _uses_cupynumeric_backend() -> bool:
    requested = os.environ.get(BACKEND_ENV, "").strip().lower()
    return requested in {"cupynumeric", "cunumeric", "gpu"}


def _legate_args() -> list[str]:
    explicit_args = os.environ.get(LEGATE_ARGS_ENV)
    if explicit_args:
        return shlex.split(explicit_args)

    args = ["--gpus", os.environ.get(LEGATE_GPUS_ENV, "1").strip() or "1"]
    fbmem = os.environ.get(LEGATE_FBMEM_ENV)
    if fbmem:
        args.extend(["--fbmem", fbmem.strip()])
    return args


def maybe_reexec_under_legate() -> None:
    if not _uses_cupynumeric_backend():
        return
    if os.environ.get(UNDER_LEGATE_ENV) == "1":
        return
    if not _truthy(os.environ.get(LEGATE_REEXEC_ENV), default=True):
        return

    legate_bin = os.environ.get(LEGATE_BIN_ENV, "legate")
    if shutil.which(legate_bin) is None:
        raise RuntimeError(
            f"{BACKEND_ENV}=cupynumeric requested GPU execution, but {legate_bin!r} "
            f"was not found. Install Legate or set {LEGATE_BIN_ENV}."
        )

    legate_args = _legate_args()
    script = str(Path(sys.argv[0]).resolve())
    env = os.environ.copy()
    env[UNDER_LEGATE_ENV] = "1"
    os.execvpe(legate_bin, [legate_bin, *legate_args, script, *sys.argv[1:]], env)
