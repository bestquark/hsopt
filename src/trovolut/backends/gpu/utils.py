"""
GPU backend utilities.

Provides helper functions for converting arrays between CPU (NumPy) and GPU
(cupynumeric) representations.

When ``cupynumeric`` is available, ``to_gpu`` wraps arrays as GPU-resident
cupynumeric arrays.  When it is not available, both functions are identity
operations so that code using them is portable.
"""

from __future__ import annotations

import numpy as _np


def to_gpu(array):
    """Convert a NumPy array to a cupynumeric GPU array.

    Parameters
    ----------
    array : array-like
        Input array (NumPy or any array-like).

    Returns
    -------
    cupynumeric.ndarray or numpy.ndarray
        GPU array if cupynumeric is available, otherwise the original array.
    """
    try:
        import cupynumeric as cn  # type: ignore[import-untyped]
        return cn.array(array)
    except ImportError:
        return _np.asarray(array)


def to_cpu(array):
    """Convert a cupynumeric GPU array back to NumPy.

    Parameters
    ----------
    array : array-like
        Input array (cupynumeric, cupy, or NumPy).

    Returns
    -------
    numpy.ndarray
        CPU-side NumPy array.
    """
    # cupynumeric arrays support np.asarray directly
    if hasattr(array, "get"):
        # cupy arrays use .get() to transfer to host
        return _np.asarray(array.get())
    return _np.asarray(array)


def is_gpu_array(array) -> bool:
    """Check whether *array* lives on a GPU.

    Returns ``True`` for cupynumeric and cupy arrays, ``False`` otherwise.
    """
    try:
        import cupynumeric as cn  # type: ignore[import-untyped]
        if isinstance(array, cn.ndarray):
            return True
    except ImportError:
        pass
    try:
        import cupy  # type: ignore[import-untyped]
        if isinstance(array, cupy.ndarray):
            return True
    except ImportError:
        pass
    return False


__all__ = ["to_gpu", "to_cpu", "is_gpu_array"]
