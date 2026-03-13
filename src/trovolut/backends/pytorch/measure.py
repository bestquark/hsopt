"""Metrics collection for the PyTorch backend.

Provides CUDA event timing, memory tracking, and throughput measurement.
"""

from __future__ import annotations

import time as _time
from typing import Any, Dict, Optional

import torch

from trovolut.core.base_metrics import BaseMetrics


class Metrics(BaseMetrics):
    """PyTorch/CUDA metrics collector.

    Measures wall-clock time, CUDA kernel time (via events),
    peak GPU memory, and tensor core throughput.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_cuda = self.device.type == "cuda"
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._wall_start: float = 0.0

    def start(self) -> None:
        """Begin timing."""
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        self._wall_start = _time.perf_counter()

    def stop(self) -> Dict[str, Any]:
        """Stop timing and return metrics dict."""
        wall_time = _time.perf_counter() - self._wall_start

        metrics: Dict[str, Any] = {
            "time_seconds": wall_time,
            "time": wall_time,
        }

        if self._use_cuda and self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            cuda_ms = self._start_event.elapsed_time(self._end_event)
            metrics["cuda_time_seconds"] = cuda_ms / 1000.0

            mem_stats = torch.cuda.memory_stats(self.device)
            metrics["gpu_memory_peak_bytes"] = mem_stats.get(
                "allocated_bytes.all.peak", 0
            )
            metrics["gpu_memory_peak_mb"] = metrics["gpu_memory_peak_bytes"] / (1024 * 1024)
            metrics["memory_peak_bytes"] = metrics["gpu_memory_peak_bytes"]
        else:
            metrics["cuda_time_seconds"] = wall_time
            metrics["gpu_memory_peak_bytes"] = 0
            metrics["gpu_memory_peak_mb"] = 0.0
            metrics["memory_peak_bytes"] = 0

        return metrics


def measure_cuda_time(func, *args, device=None, **kwargs):
    """Convenience: run func with CUDA timing and return (result, metrics)."""
    m = Metrics(device=device)
    m.start()
    result = func(*args, **kwargs)
    metrics = m.stop()
    return result, metrics


__all__ = [
    "Metrics",
    "measure_cuda_time",
]
