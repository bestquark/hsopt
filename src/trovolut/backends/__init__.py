"""
Backend registry and availability checking for trovolut.

This module provides a plugin-style system for different computational backends:
- cpu: Standard CPU implementation (always available)
- gpu: NVIDIA GPU with CuPy/cuQuantum (requires trovolut[gpu])
- qpu: Quantum processing unit via CUDA Quantum (requires trovolut[qpu])
- gpu-mp-emulator: Mixed-precision emulator for studying quantization effects (always available)

Usage:
    >>> import trovolut as trv
    >>> trv.backends.list_available()
    ['cpu', 'gpu-mp-emulator']
    
    >>> gpu_dynamics = trv.backends.get_dynamics('gpu')
    >>> mp_emulator = trv.backends.get_dynamics('gpu-mp-emulator')
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
            "description": "Standard CPU implementation (NumPy/SciPy)",
            "platform": "all",
        },
        "gpu": {
            "module": "trovolut.backends.gpu",
            "required_packages": ["cupy", "cuquantum", "pynvml"],
            "description": "NVIDIA GPU with CuPy and cuQuantum",
            "platform": "linux",
        },
        "qpu": {
            "module": "trovolut.backends.qpu",
            "required_packages": ["qiskit", "cirq"],  # cudaq is optional (Linux-only)
            "description": "Quantum processing unit (Qiskit, Cirq, CUDA-Q)",
            "platform": "all",
        },
        "gpu-mp-emulator": {
            "module": "trovolut.backends.gpu_mp_emulator",
            "required_packages": [],  # No additional deps, uses NumPy
            "description": "GPU mixed-precision emulator for studying quantized exponentials",
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
    
    def get_dynamics(self, backend: str):
        """Get the dynamics implementation for a backend."""
        module = self.require(backend)
        if not hasattr(module, "get_dynamics"):
            raise AttributeError(f"Backend '{backend}' does not provide get_dynamics()")
        return module.get_dynamics()
    
    def get_evaluator(self, backend: str):
        """Get the evaluator implementation for a backend."""
        module = self.require(backend)
        if not hasattr(module, "get_evaluator"):
            raise AttributeError(f"Backend '{backend}' does not provide get_evaluator()")
        return module.get_evaluator()
    
    def get_metrics(self, backend: str):
        """Get the metrics collector for a backend."""
        module = self.require(backend)
        if not hasattr(module, "get_metrics"):
            raise AttributeError(f"Backend '{backend}' does not provide get_metrics()")
        return module.get_metrics()


# Global registry instance
_registry = BackendRegistry()

# Public API
is_available = _registry.is_available
list_available = _registry.list_available
list_all = _registry.list_all
require = _registry.require
get_dynamics = _registry.get_dynamics
get_evaluator = _registry.get_evaluator
get_metrics = _registry.get_metrics


__all__ = [
    "BackendNotAvailableError",
    "BackendRegistry",
    "is_available",
    "list_available",
    "list_all",
    "require",
    "get_dynamics",
    "get_evaluator",
    "get_metrics",
]
