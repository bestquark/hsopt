"""
QPU (Quantum Processing Unit) backend for trovolut.

This backend provides QPU-specific tools for quantum hardware:

- QPUEvaluator: Circuit compilation and hardware execution

For Hamiltonian time evolution on QPUs, this backend handles:

- Trotter circuit compilation
- Hardware-specific optimizations
- Execution on real quantum devices

This backend provides access to quantum hardware via:

- Qiskit (cross-platform)
- Cirq (cross-platform)
- CUDA Quantum / cudaq (Linux-only, optional)

Works on all platforms with qiskit and cirq.
CUDA Quantum provides additional GPU-accelerated simulation on Linux.

Install with: `uv pip install trovolut[qpu]`

Example:
    ```python
    from trovolut.backends.qpu.evaluator import QPUEvaluator
    
    # QPU tools handle circuit compilation and execution
    evaluator = QPUEvaluator()
    ```
"""

import sys
from typing import Optional

# Availability check
_available: Optional[bool] = None
_missing_packages: list[str] = []


def is_available() -> bool:
    """Check if QPU backend dependencies are available."""
    global _available, _missing_packages
    
    if _available is not None:
        return _available
    
    # Check required packages
    # Note: cudaq is Linux-only, but qiskit and cirq are cross-platform
    # Backend can work with just qiskit/cirq on any platform
    required = ["qiskit", "cirq"]
    optional = ["cudaq"] if sys.platform == "linux" else []
    
    _missing_packages = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            _missing_packages.append(pkg)
    
    # Check optional packages (cudaq on Linux)
    for pkg in optional:
        try:
            __import__(pkg)
        except ImportError:
            _missing_packages.append(f"{pkg} (optional on Linux)")
    
    # Available if at least the required packages are present
    _available = len([p for p in _missing_packages if "optional" not in p]) == 0
    return _available


__all__ = ["is_available"]
