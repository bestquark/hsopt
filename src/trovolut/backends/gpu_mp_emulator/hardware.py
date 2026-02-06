"""
GPU hardware specifications for the mixed-precision emulator backend.

Provides:

- :class:`GPUSpec` -- :class:`~trovolut.core.base_hardware.BaseHardware`
  subclass with typed accessors for per-precision FLOP/s and memory bandwidth.
- :class:`Interconnect` -- describes a link type (NVLink, PCIe, ...).
- :class:`GPUCluster` -- :class:`~trovolut.core.base_hardware.HardwareCluster`
  subclass that tracks intra-node and inter-node interconnects.
- :class:`GPU` -- namespace of predefined NVIDIA GPU specifications.

Predefined interconnect constants:
    ``NVLINK_2``, ``NVLINK_3``, ``NVLINK_4``, ``NVLINK_5``,
    ``PCIE_GEN4``, ``PCIE_GEN5``

Example::

    from trovolut.backends.gpu_mp_emulator.hardware import GPU

    cluster = 4 * GPU.A100
    cluster.total_memory_bytes          # 320 GB
    cluster.total_flops("fp16")         # 4 * 312 TFLOPS
    cluster.intra_node_bandwidth        # 4 * 600 GB/s  (NVLink 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trovolut.core.base_hardware import BaseHardware, HardwareCluster


# -- unit helpers ---------------------------------------------------------

GB = 1024 ** 3
TB = 1024 ** 4
TFLOPS = 1e12


# ============================================================
# Interconnect
# ============================================================

@dataclass(frozen=True)
class Interconnect:
    """Describes a link type between hardware units.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"NVLink 4 (Hopper)"``).
    bandwidth_per_device : float
        Bidirectional bandwidth in **bytes/s** per device.
    """

    name: str
    bandwidth_per_device: float  # bytes/s

    def __repr__(self) -> str:
        bw = self.bandwidth_per_device
        if bw >= 1e12:
            return f"Interconnect({self.name!r}, {bw / 1e12:.1f} TB/s)"
        return f"Interconnect({self.name!r}, {bw / 1e9:.0f} GB/s)"


# -- Predefined interconnects --------------------------------------------

NVLINK_2 = Interconnect("NVLink 2 (Volta)", bandwidth_per_device=300e9)
NVLINK_3 = Interconnect("NVLink 3 (Ampere)", bandwidth_per_device=600e9)
NVLINK_4 = Interconnect("NVLink 4 (Hopper)", bandwidth_per_device=900e9)
NVLINK_5 = Interconnect("NVLink 5 (Blackwell)", bandwidth_per_device=1.8e12)
PCIE_GEN4 = Interconnect("PCIe Gen4", bandwidth_per_device=64e9)
PCIE_GEN5 = Interconnect("PCIe Gen5", bandwidth_per_device=128e9)

_INTERCONNECT_LOOKUP: dict[str, Interconnect] = {
    "nvlink_2": NVLINK_2,
    "nvlink_3": NVLINK_3,
    "nvlink_4": NVLINK_4,
    "nvlink_5": NVLINK_5,
    "pcie_gen4": PCIE_GEN4,
    "pcie_gen5": PCIE_GEN5,
}


# ============================================================
# GPUSpec  (single GPU card)
# ============================================================

@dataclass(frozen=True)
class GPUSpec(BaseHardware):
    """GPU with typed precision-throughput accessors.

    All per-precision fields are stored in the base ``specs`` dict and
    exposed via ``@property`` for convenience.  The
    ``specs["default_interconnect"]`` key (a string like ``"nvlink_3"``)
    controls which :class:`Interconnect` is auto-selected when doing
    ``4 * GPU.A100``.
    """

    # -- typed accessors (delegate to specs dict) -------------------------

    @property
    def memory_bytes(self) -> int:
        """VRAM in bytes."""
        return int(self.spec("memory_bytes"))

    @property
    def fp64_flops(self) -> float:
        return self.spec("fp64_flops")

    @property
    def fp32_flops(self) -> float:
        return self.spec("fp32_flops")

    @property
    def fp16_flops(self) -> float:
        return self.spec("fp16_flops")

    @property
    def bf16_flops(self) -> float:
        return self.spec("bf16_flops")

    @property
    def fp8_flops(self) -> float:
        return self.spec("fp8_flops")

    @property
    def fp4_flops(self) -> float:
        return self.spec("fp4_flops")

    @property
    def int8_tops(self) -> float:
        return self.spec("int8_tops")

    @property
    def memory_bandwidth(self) -> float:
        """Memory bandwidth in bytes/s."""
        return self.spec("memory_bandwidth")

    def max_flops(self, precision: str = "fp32") -> float:
        """Peak FLOP/s for the given precision string (e.g. ``'fp16'``)."""
        return self.spec(f"{precision}_flops")

    # -- composability: return GPUCluster instead of HardwareCluster ------

    def __mul__(self, n: int) -> "GPUCluster":
        if not isinstance(n, int):
            return NotImplemented
        ic_key = self.specs.get("default_interconnect", "")
        intra = _INTERCONNECT_LOOKUP.get(str(ic_key))
        return GPUCluster(unit=self, count=n, intra_node=intra)

    def __rmul__(self, n: int) -> "GPUCluster":
        return self.__mul__(n)


# ============================================================
# GPUCluster  (multi-GPU with interconnect)
# ============================================================

class GPUCluster(HardwareCluster):
    """GPU cluster with interconnect information.

    Parameters
    ----------
    unit : GPUSpec | BaseHardware
        The GPU model.
    count : int
        Number of GPUs.
    intra_node : Interconnect | None
        GPU-to-GPU interconnect within a node (e.g. NVLink).
    inter_node : Interconnect | None
        Node-to-node interconnect (e.g. InfiniBand).
    """

    def __init__(
        self,
        unit: BaseHardware,
        count: int = 1,
        *,
        intra_node: Interconnect | None = None,
        inter_node: Interconnect | None = None,
    ) -> None:
        super().__init__(unit, count)
        self.intra_node = intra_node
        self.inter_node = inter_node

    # -- interconnect bandwidth -------------------------------------------

    @property
    def intra_node_bandwidth(self) -> float:
        """Total intra-node bandwidth in bytes/s (0 if no interconnect)."""
        if self.intra_node is None:
            return 0.0
        return self.intra_node.bandwidth_per_device * self.total_units

    @property
    def inter_node_bandwidth(self) -> float:
        """Total inter-node bandwidth in bytes/s (0 if single-node / unknown)."""
        if self.inter_node is None:
            return 0.0
        return self.inter_node.bandwidth_per_device * self.total_units

    # -- convenience ------------------------------------------------------

    def total_flops(self, precision: str = "fp32") -> float:
        """Sum of per-GPU ``max_flops(precision)`` across all units."""
        return self.total_spec(f"{precision}_flops")

    @property
    def total_memory_bandwidth(self) -> float:
        """Aggregate memory bandwidth in bytes/s."""
        return self.total_spec("memory_bandwidth")

    # -- arithmetic (preserve GPUCluster type) ----------------------------

    def __mul__(self, n: int) -> "GPUCluster":
        if not isinstance(n, int):
            return NotImplemented
        new = GPUCluster.__new__(GPUCluster)
        new.units = [(u, c * n) for u, c in self.units]
        new.intra_node = self.intra_node
        new.inter_node = self.inter_node
        return new

    def __rmul__(self, n: int) -> "GPUCluster":
        return self.__mul__(n)

    def __add__(self, other: "HardwareCluster | BaseHardware") -> "GPUCluster":
        if isinstance(other, BaseHardware):
            other = HardwareCluster(other, 1)
        if not isinstance(other, HardwareCluster):
            return NotImplemented
        new = GPUCluster.__new__(GPUCluster)
        new.units = list(self.units) + list(other.units)
        # Preserve intra/inter from self; other's are silently dropped
        new.intra_node = self.intra_node
        new.inter_node = getattr(other, "inter_node", None) or self.inter_node
        return new

    # -- display ----------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for u, c in self.units:
            parts.append(f"{c}x {u.name}")
        ic = f", {self.intra_node.name}" if self.intra_node else ""
        return f"GPUCluster({', '.join(parts)}{ic})"


# ============================================================
# Predefined NVIDIA GPU specifications
# ============================================================

class GPU:
    """Namespace of predefined NVIDIA GPU specifications.

    Usage::

        from trovolut.backends.gpu_mp_emulator.hardware import GPU

        cluster = 4 * GPU.A100
        cluster.total_memory_bytes   # 320 GB
        cluster.total_flops("fp16")  # 4 * 312 TFLOPS
    """

    # -- Volta ------------------------------------------------------------

    V100 = GPUSpec(
        "V100-32GB",
        specs={
            "memory_bytes": 32 * GB,
            "fp64_flops": 7.8 * TFLOPS,
            "fp32_flops": 15.7 * TFLOPS,
            "fp16_flops": 31.4 * TFLOPS,
            "bf16_flops": 0.0,
            "fp8_flops": 0.0,
            "fp4_flops": 0.0,
            "int8_tops": 0.0,
            "memory_bandwidth": 900e9,
            "default_interconnect": "nvlink_2",
        },
    )

    # -- Ampere -----------------------------------------------------------

    A100_40 = GPUSpec(
        "A100-40GB",
        specs={
            "memory_bytes": 40 * GB,
            "fp64_flops": 9.7 * TFLOPS,
            "fp32_flops": 19.5 * TFLOPS,
            "fp16_flops": 312 * TFLOPS,
            "bf16_flops": 312 * TFLOPS,
            "fp8_flops": 0.0,
            "fp4_flops": 0.0,
            "int8_tops": 624 * TFLOPS,
            "memory_bandwidth": 2.0e12,
            "default_interconnect": "nvlink_3",
        },
    )

    A100 = GPUSpec(
        "A100-80GB",
        specs={
            "memory_bytes": 80 * GB,
            "fp64_flops": 9.7 * TFLOPS,
            "fp32_flops": 19.5 * TFLOPS,
            "fp16_flops": 312 * TFLOPS,
            "bf16_flops": 312 * TFLOPS,
            "fp8_flops": 0.0,
            "fp4_flops": 0.0,
            "int8_tops": 624 * TFLOPS,
            "memory_bandwidth": 2.0e12,
            "default_interconnect": "nvlink_3",
        },
    )

    # -- Hopper -----------------------------------------------------------

    H100 = GPUSpec(
        "H100-80GB",
        specs={
            "memory_bytes": 80 * GB,
            "fp64_flops": 33.5 * TFLOPS,
            "fp32_flops": 66.9 * TFLOPS,
            "fp16_flops": 1979 * TFLOPS,
            "bf16_flops": 1979 * TFLOPS,
            "fp8_flops": 3958 * TFLOPS,
            "fp4_flops": 0.0,
            "int8_tops": 3958 * TFLOPS,
            "memory_bandwidth": 3.35e12,
            "default_interconnect": "nvlink_4",
        },
    )

    H200 = GPUSpec(
        "H200-141GB",
        specs={
            "memory_bytes": 141 * GB,
            # Same Hopper compute die as H100
            "fp64_flops": 33.5 * TFLOPS,
            "fp32_flops": 66.9 * TFLOPS,
            "fp16_flops": 1979 * TFLOPS,
            "bf16_flops": 1979 * TFLOPS,
            "fp8_flops": 3958 * TFLOPS,
            "fp4_flops": 0.0,
            "int8_tops": 3958 * TFLOPS,
            "memory_bandwidth": 4.8e12,
            "default_interconnect": "nvlink_4",
        },
    )

    # -- Blackwell --------------------------------------------------------

    B100 = GPUSpec(
        "B100-192GB",
        specs={
            "memory_bytes": 192 * GB,
            "fp64_flops": 30 * TFLOPS,
            "fp32_flops": 60 * TFLOPS,
            "fp16_flops": 3500 * TFLOPS,
            "bf16_flops": 3500 * TFLOPS,
            "fp8_flops": 7000 * TFLOPS,
            "fp4_flops": 14_000 * TFLOPS,
            "int8_tops": 7000 * TFLOPS,
            "memory_bandwidth": 8.0e12,
            "default_interconnect": "nvlink_5",
        },
    )

    B200 = GPUSpec(
        "B200-192GB",
        specs={
            "memory_bytes": 192 * GB,
            "fp64_flops": 45 * TFLOPS,
            "fp32_flops": 90 * TFLOPS,
            "fp16_flops": 4500 * TFLOPS,
            "bf16_flops": 4500 * TFLOPS,
            "fp8_flops": 9000 * TFLOPS,
            "fp4_flops": 18_000 * TFLOPS,
            "int8_tops": 9000 * TFLOPS,
            "memory_bandwidth": 8.0e12,
            "default_interconnect": "nvlink_5",
        },
    )

    # -- Blackwell Ultra --------------------------------------------------

    B300 = GPUSpec(
        "B300-288GB",
        specs={
            "memory_bytes": 288 * GB,
            "fp64_flops": 1.25 * TFLOPS,
            "fp32_flops": 37 * TFLOPS,
            "fp16_flops": 2250 * TFLOPS,
            "bf16_flops": 2250 * TFLOPS,
            "fp8_flops": 4500 * TFLOPS,
            "fp4_flops": 14_000 * TFLOPS,
            "int8_tops": 4500 * TFLOPS,
            "memory_bandwidth": 8.0e12,
            "default_interconnect": "nvlink_5",
        },
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Classes
    "GPUSpec",
    "GPUCluster",
    "Interconnect",
    "GPU",
    # Predefined interconnects
    "NVLINK_2",
    "NVLINK_3",
    "NVLINK_4",
    "NVLINK_5",
    "PCIE_GEN4",
    "PCIE_GEN5",
]
