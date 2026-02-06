from .generate_system import generate_sdf
from .op_counter import (
    OperationCounter,
    get_counter,
    count_matmul,
    wrap_numpy_ops,
    wrap_legate_ops,
    count_matmul_in_source,
    count_matmul_in_file,
)
from .code_diff_gif import create_diff_gif

_backend_utils = None


def _set_backend_utils(module) -> None:
    """Register backend-specific utils to expose via this package."""
    global _backend_utils
    _backend_utils = module


def __getattr__(name: str):
    if _backend_utils is not None and hasattr(_backend_utils, name):
        return getattr(_backend_utils, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
