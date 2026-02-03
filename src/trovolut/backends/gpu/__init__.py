"""
GPU backend for trovolut - Hardware-specific tools for NVIDIA GPUs.

This backend provides GPU-specific tools for profiling and monitoring:

- GPUEvaluator: CUDA profiling and benchmarking

For dynamics, use the CPU backend methods with cupynumeric arrays - they run on GPU automatically.

Requirements:

- Linux operating system
- NVIDIA GPU with CUDA support
- cupynumeric and CUDA Quantum Python packages

Install with: `uv pip install trovolut[gpu]`

Example:
    ```python
    # For dynamics on GPU, use CPU methods with cupynumeric arrays
    import cupynumeric as cn
    from trovolut.backends.cpu.dynamics import TrotterEvolution
    
    trotter = TrotterEvolution(order=2)
    H_gpu = cn.asarray(H)  # Move to GPU
    result = trotter.evolve(H_gpu, psi0_gpu, time=1.0)  # Runs on GPU!
    
    # Use GPU-specific tools for profiling
    from trovolut.backends.gpu.evaluator import GPUEvaluator
    ```
"""

import sys
from typing import Optional

# Availability check
_available: Optional[bool] = None
_missing_packages: list[str] = []


def is_available() -> bool:
    """Check if GPU backend dependencies are available."""
    global _available, _missing_packages
    
    if _available is not None:
        return _available
    
    # Check platform
    if sys.platform != "linux":
        _available = False
        _missing_packages = [f"requires Linux (current: {sys.platform})"]
        return False
    
    # Check required packages
    required = ["cupynumeric", "cudaq", "pynvml"]
    _missing_packages = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            _missing_packages.append(pkg)
    
    _available = len(_missing_packages) == 0
    return _available


__all__ = ["is_available"]
