"""
Trovolut: Evolutionary Trotterization for Quantum Simulation

A framework for optimizing quantum simulation algorithms across different backends:
- CPU: NumPy/SciPy (always available)
- GPU: NVIDIA cupynumeric/CUDA Quantum (trovolut[gpu])
- QPU: Quantum hardware via CUDA Quantum (trovolut[qpu])
- Emulator: Software GPU emulation (trovolut[gpu-emulate])

Quick Start:
    >>> import trovolut as trv
    >>> # Check available backends
    >>> trv.backends.list_available()
    ['cpu', 'gpu']
    
    >>> # Get GPU dynamics implementation
    >>> gpu_dynamics = trv.backends.get_dynamics('gpu')
"""

# Core abstractions (always available)
from trovolut import core
from trovolut import backends

# Quantum mechanics modules (backend-agnostic)
from trovolut import qm
from trovolut import utils

# Workflow and high-level interfaces
try:
    from trovolut import filemaker
except ImportError:
    filemaker = None  # Optional module

# Version info
__version__ = "0.1.0"

__all__ = [
    # Core modules
    "core",
    "backends",
    "qm",
    "utils",
    "filemaker",
    # Version
    "__version__",
]
