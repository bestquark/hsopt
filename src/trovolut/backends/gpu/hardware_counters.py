"""
Real-time GPU hardware counters via NVIDIA Management Library (pynvml).

Provides:

- :func:`nvml_available` -- checks if pynvml is importable and NVML initializes
- :class:`HardwareCounterMetrics` -- dataclass of sampled metrics
- :class:`NVMLMonitor` -- context manager with background sampling thread

These complement the Legate profiler (scheduling-level metrics) with
real-time hardware counters: actual VRAM usage, power draw, GPU/memory
utilization, and temperature.

Example::

    from trovolut.backends.gpu.hardware_counters import NVMLMonitor

    with NVMLMonitor(gpu_index=0, sample_interval=0.1) as mon:
        # ... run GPU workload ...
        pass
    metrics = mon.metrics()
    print(metrics.peak_memory_bytes, metrics.avg_power_watts)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ============================================================
# Availability check
# ============================================================

def nvml_available() -> bool:
    """Check if pynvml is importable and NVML can be initialized."""
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except Exception:
        return False


# ============================================================
# Metrics dataclass
# ============================================================

@dataclass
class HardwareCounterMetrics:
    """Aggregated hardware counter metrics from an NVML sampling session.

    All values default to 0 when NVML is unavailable (graceful no-op).
    """

    peak_memory_bytes: int = 0
    avg_memory_bytes: float = 0.0
    peak_power_watts: float = 0.0
    avg_power_watts: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    energy_joules: float = 0.0
    duration_seconds: float = 0.0
    max_temperature_c: int = 0
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return dict with all keys prefixed with ``nvml_``."""
        return {
            "nvml_peak_memory_bytes": self.peak_memory_bytes,
            "nvml_avg_memory_bytes": self.avg_memory_bytes,
            "nvml_peak_power_watts": self.peak_power_watts,
            "nvml_avg_power_watts": self.avg_power_watts,
            "nvml_avg_gpu_utilization": self.avg_gpu_utilization,
            "nvml_avg_memory_utilization": self.avg_memory_utilization,
            "nvml_energy_joules": self.energy_joules,
            "nvml_duration_seconds": self.duration_seconds,
            "nvml_max_temperature_c": self.max_temperature_c,
            "nvml_n_samples": self.n_samples,
        }


# ============================================================
# NVMLMonitor context manager
# ============================================================

@dataclass
class _Sample:
    """A single NVML sample."""
    memory_bytes: int = 0
    power_watts: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    temperature_c: int = 0


class NVMLMonitor:
    """Context manager that samples GPU hardware counters in a background thread.

    Silently no-ops when pynvml is unavailable (returns zero-filled metrics).

    Parameters
    ----------
    gpu_index : int
        GPU device index (default 0).
    sample_interval : float
        Seconds between samples (default 0.1).
    """

    def __init__(self, gpu_index: int = 0, sample_interval: float = 0.1):
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self._samples: List[_Sample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_ok = False
        self._start_time = 0.0
        self._end_time = 0.0

    def __enter__(self) -> "NVMLMonitor":
        self._start_time = time.monotonic()
        if not nvml_available():
            return self

        self._nvml_ok = True
        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.monotonic()
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

            while not self._stop_event.is_set():
                s = _Sample()
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    s.memory_bytes = mem_info.used
                except Exception:
                    pass
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    s.power_watts = power_mw / 1000.0
                except Exception:
                    pass
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    s.gpu_utilization = float(util.gpu)
                    s.memory_utilization = float(util.memory)
                except Exception:
                    pass
                try:
                    s.temperature_c = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU,
                    )
                except Exception:
                    pass

                self._samples.append(s)
                self._stop_event.wait(self.sample_interval)

            pynvml.nvmlShutdown()
        except Exception:
            # Silently degrade — metrics() will return zeros
            pass

    def metrics(self) -> HardwareCounterMetrics:
        """Aggregate collected samples into :class:`HardwareCounterMetrics`."""
        duration = self._end_time - self._start_time if self._end_time > self._start_time else 0.0

        if not self._samples:
            return HardwareCounterMetrics(duration_seconds=duration)

        n = len(self._samples)
        peak_mem = max(s.memory_bytes for s in self._samples)
        avg_mem = sum(s.memory_bytes for s in self._samples) / n
        peak_power = max(s.power_watts for s in self._samples)
        avg_power = sum(s.power_watts for s in self._samples) / n
        avg_gpu_util = sum(s.gpu_utilization for s in self._samples) / n
        avg_mem_util = sum(s.memory_utilization for s in self._samples) / n
        max_temp = max(s.temperature_c for s in self._samples)
        energy = avg_power * duration

        return HardwareCounterMetrics(
            peak_memory_bytes=peak_mem,
            avg_memory_bytes=avg_mem,
            peak_power_watts=peak_power,
            avg_power_watts=avg_power,
            avg_gpu_utilization=avg_gpu_util,
            avg_memory_utilization=avg_mem_util,
            energy_joules=energy,
            duration_seconds=duration,
            max_temperature_c=max_temp,
            n_samples=n,
        )


__all__ = [
    "nvml_available",
    "HardwareCounterMetrics",
    "NVMLMonitor",
]
