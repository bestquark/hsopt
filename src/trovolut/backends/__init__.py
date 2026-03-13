"""
Backend registry and availability checking for trovolut.

This module provides a plugin-style system for different computational backends:

- **cpu**: Dynamics methods (ExactEvolution, TrotterEvolution, QDriftEvolution) that work with any array backend
- **gpu**: GPU-accelerated dynamics (ExactEvolution, TrotterEvolution, QDriftEvolution) for NVIDIA GPUs
- **qpu**: QPU dynamics (TrotterEvolution, QDriftEvolution) via circuit compilation on quantum hardware
- **gpu-mp-emulator**: Mixed-precision emulator for studying quantization effects

Dynamics methods are hardware-agnostic - just pass NumPy arrays for CPU or cupynumeric arrays for GPU.

Example:
    ```python
    import trovolut as trv
    
    # List available backends
    trv.backends.list_available()
    # ['cpu', 'gpu-mp-emulator']
    
    # Import dynamics methods (work with NumPy or cupynumeric)
    from trovolut.backends.cpu.dynamics import ExactEvolution, TrotterEvolution
    
    # Import hardware-specific tools
    from trovolut.backends.gpu.evaluator import GPUEvaluator
    ```
"""

import sys
from typing import Any, Dict, List, Optional
from importlib import import_module


class BackendNotAvailableError(ImportError):
    """Raised when attempting to use a backend whose dependencies aren't installed."""
    
    def __init__(self, backend: str, missing_packages: List[str]):
        self.backend = backend
        self.missing_packages = missing_packages
        
        install_cmd = f"uv pip install trovolut[{backend}]"
        
        # GPU backend specifically requires Linux
        if sys.platform != "linux" and backend == "gpu":
            msg = (
                f"Backend '{backend}' requires Linux and NVIDIA hardware.\n"
                f"Current platform: {sys.platform}\n"
                f"Install command (Linux only): {install_cmd}"
            )
        else:
            msg = (
                f"Backend '{backend}' is not available.\n"
                f"Missing packages: {', '.join(missing_packages)}\n"
                f"Install with: {install_cmd}"
            )
        super().__init__(msg)


class BackendRegistry:
    """Registry for dynamically discovering and loading backends."""
    
    _backends: Dict[str, Dict[str, Any]] = {
        "cpu": {
            "module": "trovolut.backends.cpu",
            "required_packages": [],  # CPU is always available
            "description": "Dynamics methods (ExactEvolution, TrotterEvolution, QDriftEvolution)",
            "platform": "all",
        },
        "gpu": {
            "module": "trovolut.backends.gpu",
            "required_packages": ["cupynumeric", "cudaq", "pynvml"],
            "description": "GPU profiling tools (evaluator, metrics) for NVIDIA GPUs",
            "platform": "linux",
        },
        "qpu": {
            "module": "trovolut.backends.qpu",
            "required_packages": ["qiskit", "cirq"],  # cudaq is optional (Linux-only)
            "description": "QPU tools (circuit compilation, hardware execution)",
            "platform": "all",
        },
        "gpu-mp-emulator": {
            "module": "trovolut.backends.gpu_mp_emulator",
            "required_packages": [],  # No additional deps, uses NumPy
            "description": "Mixed-precision emulator (dynamics, mp_evals)",
            "platform": "all",
        },
        "pytorch": {
            "module": "trovolut.backends.pytorch",
            "required_packages": ["torch"],
            "description": "PyTorch TEBD tensor network engine (MPS, mixed-precision tensor cores)",
            "platform": "all",
        },
    }
    
    def __init__(self):
        self._cache: Dict[str, Optional[Any]] = {}
    
    def is_available(self, backend: str) -> bool:
        """Check if a backend is available with all dependencies installed."""
        if backend not in self._backends:
            return False
        
        info = self._backends[backend]
        
        # Check platform compatibility
        if info["platform"] != "all" and sys.platform != info["platform"]:
            return False
        
        # Check if required packages are importable
        for pkg in info["required_packages"]:
            try:
                import_module(pkg)
            except ImportError:
                return False
        
        return True
    
    def get_missing_packages(self, backend: str) -> List[str]:
        """Get list of missing packages for a backend."""
        if backend not in self._backends:
            raise ValueError(f"Unknown backend: {backend}")
        
        info = self._backends[backend]
        missing = []
        
        for pkg in info["required_packages"]:
            try:
                import_module(pkg)
            except ImportError:
                missing.append(pkg)
        
        return missing
    
    def list_available(self) -> List[str]:
        """List all available backends on this system."""
        return [name for name in self._backends if self.is_available(name)]
    
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all registered backends with their availability status."""
        return {
            name: {
                **info,
                "available": self.is_available(name),
            }
            for name, info in self._backends.items()
        }
    
    def require(self, backend: str) -> Any:
        """
        Import and return the backend module, raising BackendNotAvailableError if unavailable.
        
        Note: This function is primarily for checking backend availability.
        For actual use, import dynamics/evaluator/metrics directly:
            from trovolut.backends.cpu.dynamics import ExactEvolution
        
        Args:
            backend: Backend name (e.g., 'gpu', 'qpu')
            
        Returns:
            The backend module
            
        Raises:
            BackendNotAvailableError: If backend dependencies are not installed
            ValueError: If backend name is unknown
        """
        if backend not in self._backends:
            available = ", ".join(self._backends.keys())
            raise ValueError(f"Unknown backend: {backend}. Available: {available}")
        
        # Check cache first
        if backend in self._cache:
            return self._cache[backend]
        
        # Check platform compatibility
        info = self._backends[backend]
        if info["platform"] != "all" and sys.platform != info["platform"]:
            raise BackendNotAvailableError(
                backend, 
                [f"requires {info['platform']}, current platform: {sys.platform}"]
            )
        
        # Check availability
        missing = self.get_missing_packages(backend)
        if missing:
            raise BackendNotAvailableError(backend, missing)
        
        # Import and cache
        module = import_module(info["module"])
        self._cache[backend] = module
        return module


# Global registry instance
_registry = BackendRegistry()

# Public API
is_available = _registry.is_available
list_available = _registry.list_available
list_all = _registry.list_all
require = _registry.require


__all__ = [
    "BackendNotAvailableError",
    "BackendRegistry",
    "is_available",
    "list_available",
    "list_all",
    "require",
]
