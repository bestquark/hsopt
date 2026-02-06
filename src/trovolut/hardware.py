"""
Proxy module for backend-specific hardware definitions.

At **type-checking time** (Pylance / mypy), wildcard imports from the
concrete backend modules make every public name visible, so autocomplete
and Ctrl+Click (Go to Definition) navigate straight to the real
implementations -- no manual name lists to maintain.

At **runtime**, ``__getattr__`` dispatches every lookup to whichever
backend was activated by :func:`trovolut.set_backend`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trovolut.backends.gpu_mp_emulator.hardware import *  # noqa: F403

# ── Runtime backend reference (set by trovolut.set_backend()) ───
_backend_module = None

_BACKEND_SUBMODULES = [
    "trovolut.backends.gpu_mp_emulator.hardware",
]


def __getattr__(name: str):
    """Forward attribute lookups to the active backend's hardware module."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    if _backend_module is not None:
        try:
            return getattr(_backend_module, name)
        except AttributeError:
            pass

    import trovolut as _trv

    if _trv._active_backend is None:
        raise ImportError(
            f"Cannot import '{name}' from 'trovolut.hardware' -- "
            f"call trv.set_backend(...) first."
        )
    raise ImportError(
        f"Backend '{_trv._active_backend}' has no '{name}' in its "
        f"hardware module."
    )


def _public_names(mod):
    """Return public names from a module (prefer __all__, fall back to dir)."""
    return getattr(mod, "__all__", [n for n in dir(mod) if not n.startswith("_")])


def __dir__():
    """List all available names for autocomplete."""
    from importlib import import_module

    names: set[str] = set()
    for path in _BACKEND_SUBMODULES:
        try:
            names.update(_public_names(import_module(path)))
        except (ImportError, Exception):
            pass
    if _backend_module is not None:
        names.update(_public_names(_backend_module))
    return sorted(names)
