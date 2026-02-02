"""
Base class for metrics collection across all backends.

This module defines the abstract interface for collecting performance metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMetrics(ABC):
    """
    Abstract base class for metrics collection.
    
    All backend-specific metrics collectors (CPU, GPU, QPU, Emulator) must inherit
    from this class and implement the collect() method.
    
    Metrics can include:
        - Execution time
        - Memory usage
        - Hardware utilization (CPU%, GPU%, etc.)
        - Backend-specific metrics (gate count, circuit depth, etc.)
    """
    
    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """
        Collect current metrics.
        
        Returns:
            Dictionary with metric name -> value mappings
            
        Example:
            {
                "time_seconds": 1.234,
                "memory_bytes": 1073741824,
                "utilization_percent": 85.2,
            }
        """
        pass
    
    def start_recording(self):
        """
        Start recording metrics (optional method for continuous monitoring).
        
        Backends can override this to enable continuous metric collection.
        """
        pass
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and return accumulated metrics.
        
        Returns:
            Dictionary with accumulated metrics
        """
        return self.collect()


__all__ = ["BaseMetrics"]
