from __future__ import annotations

import os

import numpy as _numpy


BACKEND_ENV = "TROVOLUT_PFS_ARRAY_BACKEND"
LEGACY_USE_CUPYNUMERIC_ENV = "TROVOLUT_PFS_USE_CUPYNUMERIC"


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _requested_backend() -> str:
    requested = os.environ.get(BACKEND_ENV, "").strip().lower()
    if not requested and _truthy(os.environ.get(LEGACY_USE_CUPYNUMERIC_ENV)):
        requested = "cupynumeric"
    return requested or "numpy"


def _load_backend():
    requested = _requested_backend()
    if requested in {"numpy", "np", "cpu"}:
        return "numpy", _numpy

    if requested in {"cupynumeric", "cunumeric", "gpu"}:
        try:
            import cupynumeric as _cupynumeric
        except Exception as exc:
            raise ImportError(
                f"{BACKEND_ENV}={requested!r} could not import NVIDIA cuPyNumeric. "
                "Install a cupynumeric build compatible with the active Python/NumPy "
                f"environment, or unset {BACKEND_ENV} to use NumPy. Original error: {exc}"
            ) from exc
        return "cupynumeric", _cupynumeric

    if requested == "auto":
        try:
            import cupynumeric as _cupynumeric
        except Exception:
            return "numpy", _numpy
        return "cupynumeric", _cupynumeric

    raise ValueError(
        f"unsupported {BACKEND_ENV}={requested!r}; expected 'numpy', 'cupynumeric', or 'auto'"
    )


ARRAY_BACKEND_NAME, np = _load_backend()


def array_backend_name() -> str:
    return ARRAY_BACKEND_NAME


def as_numpy(value, dtype=None):
    return _numpy.asarray(value, dtype=dtype)
