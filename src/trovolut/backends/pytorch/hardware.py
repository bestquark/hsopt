"""Hardware detection and specification for PyTorch/CUDA backends.

Wraps torch.cuda.get_device_properties() into trovolut's hardware abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from trovolut.core.base_hardware import BaseHardware


def get_cuda_device_info(device_index: int = 0) -> Dict[str, Any]:
    """Query CUDA device properties and return as a dict."""
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)

    return {
        "available": True,
        "name": props.name,
        "compute_capability": f"{major}.{minor}",
        "sm_version": major * 10 + minor,
        "total_memory_mb": props.total_mem / (1024 * 1024),
        "total_memory_bytes": props.total_mem,
        "multi_processor_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "device_index": device_index,
    }


def get_all_cuda_devices() -> List[Dict[str, Any]]:
    """Query all available CUDA devices."""
    if not torch.cuda.is_available():
        return []
    return [get_cuda_device_info(i) for i in range(torch.cuda.device_count())]


class CUDADevice(BaseHardware):
    """Hardware spec for a single CUDA GPU."""

    @classmethod
    def from_device(cls, device_index: int = 0) -> "CUDADevice":
        """Create from actual hardware detection."""
        info = get_cuda_device_info(device_index)
        return cls(name=info.get("name", "unknown"), specs=info)

    @classmethod
    def from_spec(cls, name: str, memory_mb: float, sm_version: int,
                  sm_count: int = 0) -> "CUDADevice":
        """Create from manual specification (for emulation/planning)."""
        return cls(name=name, specs={
            "available": True,
            "name": name,
            "sm_version": sm_version,
            "total_memory_mb": memory_mb,
            "total_memory_bytes": int(memory_mb * 1024 * 1024),
            "multi_processor_count": sm_count,
        })

    @property
    def memory_mb(self) -> float:
        return self.specs.get("total_memory_mb", 0.0)

    @property
    def sm_version(self) -> int:
        return self.specs.get("sm_version", 0)

    @property
    def supports_fp8(self) -> bool:
        return self.sm_version >= 90

    @property
    def supports_fp4(self) -> bool:
        return self.sm_version >= 100


# Common GPU specs for planning
A100_40GB = CUDADevice.from_spec("A100-40GB", 40960, 80, 108)
A100_80GB = CUDADevice.from_spec("A100-80GB", 81920, 80, 108)
H100_80GB = CUDADevice.from_spec("H100-80GB", 81920, 90, 132)
GB200 = CUDADevice.from_spec("GB200", 196608, 100, 0)


__all__ = [
    "get_cuda_device_info",
    "get_all_cuda_devices",
    "CUDADevice",
    "A100_40GB",
    "A100_80GB",
    "H100_80GB",
    "GB200",
]
