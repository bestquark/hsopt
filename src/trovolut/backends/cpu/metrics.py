"""
CPU metrics collection using psutil and tracemalloc.

Provides monitoring of:
- CPU utilization
- Memory usage
- Process-level statistics
"""

from typing import Dict, Any
import tracemalloc
from trovolut.core.base_metrics import BaseMetrics


class CPUMetrics(BaseMetrics):
    """
    CPU metrics collector using psutil.
    
    Provides:
        - CPU utilization percentage
        - Memory usage (RSS, VMS)
        - CPU time (user, system)
    """
    
    def __init__(self):
        """Initialize CPU metrics collector."""
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None
    
    def collect(self) -> Dict[str, Any]:
        """
        Collect current CPU metrics.
        
        Returns:
            Dictionary with CPU metrics:
                - cpu_percent: CPU usage percentage
                - memory_rss_bytes: Resident set size
                - memory_vms_bytes: Virtual memory size
        """
        metrics = {}
        
        if self.process:
            import psutil
            metrics["cpu_percent"] = self.process.cpu_percent()
            mem_info = self.process.memory_info()
            metrics["memory_rss_bytes"] = mem_info.rss
            metrics["memory_vms_bytes"] = mem_info.vms
        
        # Add tracemalloc info if enabled
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            metrics["tracemalloc_current_bytes"] = current
            metrics["tracemalloc_peak_bytes"] = peak
        
        return metrics


__all__ = ["CPUMetrics"]
