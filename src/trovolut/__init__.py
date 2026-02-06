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
    
    # Now access modules directly
    from trovolut.backends.cpu.dynamics import TrotterEvolution
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

# Version info
__version__ = "0.1.0"

# Lazy import for constraints to avoid circular import
def __getattr__(name):
    """Lazy loading for submodules to avoid circular imports."""
    if name == "constraints":
        from trovolut.core import base_constraints
        return base_constraints
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Active backend tracking
_active_backend: Optional[str] = None
_backend_modules: dict = {}


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
    
    # Inject modules into trovolut namespace
    import sys
    current_module = sys.modules[__name__]
    for name, mod in _backend_modules.items():
        if name != "utils":
            setattr(current_module, name, mod)

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
    # Backend management
    "set_backend",
    "get_active_backend",
    "constraints",
    # Version
    "__version__",
]
