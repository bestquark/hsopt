"""
Trovolut: Evolutionary Trotterization for Quantum Simulation

A framework for optimizing quantum simulation algorithms across different backends:
- CPU: NumPy/SciPy (always available)
- GPU: NVIDIA cupynumeric/CUDA Quantum (trovolut[gpu])
- QPU: Quantum hardware via CUDA Quantum (trovolut[qpu])
- Emulator: Software GPU emulation (trovolut[gpu-emulate])

Quick Start:
    ```python
    import trovolut as trv
    
    # Set active backend
    trv.set_backend('cpu')
    
    # Now access modules directly via short paths
    from trovolut.dynamics import TrotterEvolution
    trotter = TrotterEvolution(order=2)
    ```
"""

from typing import Optional
from importlib import import_module

# Core abstractions (always available)
from trovolut import core
from trovolut import backends

# Quantum mechanics modules (backend-agnostic)
from trovolut import qm
from trovolut import utils

# Workflow and high-level interfaces
from trovolut import filemaker

# Autoresearch module
from trovolut import autoresearch

# Calibration
from trovolut.calibration import calibrate

# Version info
__version__ = "0.1.0"

# Active backend tracking
_active_backend: Optional[str] = None
_backend_modules: dict = {}


# ── Backend proxy modules ────────────────────────────────────
# Real Python files (dynamics.py, measure.py, constraints.py,
# hardware.py) live next to this __init__.py.  They provide
# TYPE_CHECKING imports for Pylance (Ctrl+Click / autocomplete)
# and __getattr__ for runtime dispatch.  set_backend() wires them
# up by setting each module's _backend_module attribute.

from trovolut import dynamics  # noqa: E402
from trovolut import measure  # noqa: E402
from trovolut import hardware  # noqa: E402
from trovolut import constraints  # noqa: E402

_proxy_modules = {
    "dynamics": dynamics,
    "measure": measure,
    "hardware": hardware,
    "constraints": constraints,
}


def set_backend(backend_name: str) -> None:
    """
    Set the active backend and make its modules accessible via trv.dynamics, trv.measure, etc.
    
    Args:
        backend_name: Name of backend ('cpu', 'gpu', 'qpu', 'gpu-mp-emulator')
    
    Example:
        ```python
        import trovolut as trv
        
        # Set CPU backend
        trv.set_backend('cpu')
        
        # Access dynamics directly
        from trovolut.dynamics import TrotterEvolution
        trotter = TrotterEvolution(order=2)
        ```
    
    Raises:
        ValueError: If backend is not available
    """
    global _active_backend, _backend_modules
    
    # Check if backend is available
    if not backends.is_available(backend_name):
        available = backends.list_available()
        raise ValueError(
            f"Backend '{backend_name}' is not available. "
            f"Available backends: {available}"
        )
    
    # Map backend names to module paths
    backend_map = {
        'cpu': 'trovolut.backends.cpu',
        'gpu': 'trovolut.backends.gpu',
        'qpu': 'trovolut.backends.qpu',
        'gpu-mp-emulator': 'trovolut.backends.gpu_mp_emulator',
        'pytorch': 'trovolut.backends.pytorch',
    }
    
    if backend_name not in backend_map:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    backend_module_path = backend_map[backend_name]
    
    # Import backend modules
    _backend_modules = {}
    
    # Try to import common submodules
    for submodule in ['dynamics', 'measure', 'constraints', 'hardware', 'utils']:
        try:
            mod = import_module(f"{backend_module_path}.{submodule}")
            _backend_modules[submodule] = mod
        except (ImportError, AttributeError):
            # Not all backends have all submodules
            pass
    
    _active_backend = backend_name

    # Wire each proxy module to the real backend submodule
    for name, proxy in _proxy_modules.items():
        proxy._backend_module = _backend_modules.get(name)

    # Wire backend-specific utils into the base utils namespace
    try:
        from trovolut import utils as _core_utils
        _core_utils._set_backend_utils(_backend_modules.get("utils"))
    except Exception:
        pass


def get_active_backend() -> Optional[str]:
    """
    Get the name of the currently active backend.
    
    Returns:
        Name of active backend, or None if no backend is set
    
    Example:
        ```python
        import trovolut as trv
        trv.set_backend('cpu')
        print(trv.get_active_backend())  # 'cpu'
        ```
    """
    return _active_backend


__all__ = [
    # Core modules
    "core",
    "backends",
    "qm",
    "utils",
    "filemaker",
    "autoresearch",
    # Backend proxy modules (Ctrl+Click navigable)
    "dynamics",
    "measure",
    "constraints",
    "hardware",
    # Backend management
    "set_backend",
    "get_active_backend",
    # Calibration
    "calibrate",
    # Version
    "__version__",
]
