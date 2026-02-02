"""
GPU metrics collection using pynvml (NVIDIA Management Library).

Provides real-time monitoring of:
- GPU utilization
- Memory usage (used/free/total)
- Temperature
- Power consumption
- SM (streaming multiprocessor) activity
"""

from typing import Dict, Any, Optional
from trovolut.core.base_metrics import BaseMetrics


class GPUMetrics(BaseMetrics):
    """
    GPU hardware metrics collector using pynvml.
    
    Provides:
        - GPU utilization percentage
        - Memory usage (bytes)
        - Temperature (Celsius)
        - Power draw (Watts)
        - Clock speeds
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU metrics collector.
        
        Args:
            device_id: CUDA device ID to monitor
        """
        self.device_id = device_id
        self._handle: Optional[Any] = None
        self._init_nvml()
    
    def _init_nvml(self):
        """Initialize NVIDIA Management Library."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pynvml: {e}")
    
    def collect(self) -> Dict[str, Any]:
        """
        Collect current GPU metrics.
        
        Returns:
            Dictionary with GPU metrics:
                - gpu_utilization: GPU usage percentage (0-100)
                - memory_used_bytes: GPU memory in use
                - memory_total_bytes: Total GPU memory
                - temperature_celsius: GPU temperature
                - power_watts: Current power draw
        """
        import pynvml
        
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W
            
            return {
                "gpu_utilization": utilization.gpu,
                "memory_used_bytes": memory_info.used,
                "memory_total_bytes": memory_info.total,
                "memory_free_bytes": memory_info.free,
                "temperature_celsius": temperature,
                "power_watts": power,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to collect GPU metrics: {e}")
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except:
            pass


__all__ = ["GPUMetrics"]
