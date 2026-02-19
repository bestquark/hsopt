"""
GPU hardware specifications for the mixed-precision emulator backend.

Provides:

- :class:`GPU` -- :class:`~trovolut.core.base_hardware.BaseHardware`
  subclass with typed accessors for per-precision FLOP/s and memory bandwidth.
- :class:`Interconnect` -- describes a link type (NVLink, PCIe, ...).
- :class:`GPUCluster` -- :class:`~trovolut.core.base_hardware.HardwareCluster`
  subclass that tracks intra-node and inter-node interconnects.
- :data:`gpus` -- namespace of predefined NVIDIA GPU specifications.

Predefined interconnect constants:
    ``NVLINK_2``, ``NVLINK_3``, ``NVLINK_4``, ``NVLINK_5``,
    ``PCIE_GEN4``, ``PCIE_GEN5``

All Tensor Core throughput values are stored as **dense** (non-sparsity)
numbers.  Structured-sparsity (2:4) variants are available via the
``_sparse`` suffix keys (e.g. ``fp16_flops_sparse``) and through
``max_flops(precision, sparse=True)``.

Example::

    import trovolut as trv
    trv.set_backend('gpu-mp-emulator')

    from trovolut.hardware import gpus

    cluster = 5 * gpus.A100
    cluster.total_memory_bytes          # 400 GB
    cluster.total_flops("fp16")         # 5 * 312 TFLOPS
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from trovolut.core.base_hardware import BaseHardware, HardwareCluster


# -- unit helpers ---------------------------------------------------------

KB = 1024
MB = 1024 ** 2
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
# Sources:
#   NVLink overview  -- https://www.nvidia.com/en-us/data-center/nvlink/
#   NVLink Wikipedia -- https://en.wikipedia.org/wiki/NVLink
#   PCIe specs       -- https://pcisig.com/specifications

NVLINK_2 = Interconnect("NVLink 2 (Volta)", bandwidth_per_device=300e9)       # 6 links x 50 GB/s
NVLINK_3 = Interconnect("NVLink 3 (Ampere)", bandwidth_per_device=600e9)      # 12 links x 50 GB/s
NVLINK_4 = Interconnect("NVLink 4 (Hopper)", bandwidth_per_device=900e9)      # 18 links x 50 GB/s
NVLINK_5 = Interconnect("NVLink 5 (Blackwell)", bandwidth_per_device=1.8e12)  # 18 links x 100 GB/s
PCIE_GEN4 = Interconnect("PCIe Gen4", bandwidth_per_device=64e9)              # x16 bidirectional
PCIE_GEN5 = Interconnect("PCIe Gen5", bandwidth_per_device=128e9)             # x16 bidirectional

_INTERCONNECT_LOOKUP: dict[str, Interconnect] = {
    "nvlink_2": NVLINK_2,
    "nvlink_3": NVLINK_3,
    "nvlink_4": NVLINK_4,
    "nvlink_5": NVLINK_5,
    "pcie_gen4": PCIE_GEN4,
    "pcie_gen5": PCIE_GEN5,
}


# ============================================================
# GPU  (single GPU card)
# ============================================================

@dataclass(frozen=True)
class GPU(BaseHardware):
    """A single GPU with typed precision-throughput accessors.

    All per-precision fields are stored in the base ``specs`` dict and
    exposed via ``@property`` for convenience.  The
    ``specs["default_interconnect"]`` key (a string like ``"nvlink_3"``)
    controls which :class:`Interconnect` is auto-selected when doing
    ``5 * gpus.A100``.

    Example::

        from trovolut.hardware import gpus

        gpus.A100.fp16_flops   # 312 TFLOPS
        gpus.A100.memory_bytes # 80 GB
        cluster = 5 * gpus.A100
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

    # -- architectural spec accessors -------------------------------------

    @property
    def cuda_cores(self) -> int:
        return int(self.spec("cuda_cores"))

    @property
    def tensor_cores(self) -> int:
        return int(self.spec("tensor_cores"))

    @property
    def sm_count(self) -> int:
        return int(self.spec("sm_count"))

    @property
    def shared_mem_per_sm(self) -> int:
        return int(self.spec("shared_mem_per_sm"))

    @property
    def max_shared_mem_per_block(self) -> int:
        return int(self.spec("max_shared_mem_per_block"))

    @property
    def l2_cache_bytes(self) -> int:
        return int(self.spec("l2_cache_bytes"))

    @property
    def tdp_watts(self) -> int:
        return int(self.spec("tdp_watts"))

    def max_flops(self, precision: str = "fp32", sparse: bool = False) -> float:
        """Peak FLOP/s for the given precision string (e.g. ``'fp16'``).

        Parameters
        ----------
        sparse : bool
            If *True*, return the structured-sparsity (2:4) throughput
            when available; falls back to the dense value otherwise.
        """
        if sparse:
            val = self.spec(f"{precision}_flops_sparse")
            if val > 0:
                return val
        return self.spec(f"{precision}_flops")

    def __repr__(self) -> str:
        # Try to return a reference to a predefined GPU (eval-able code)
        try:
            from trovolut.backends.gpu_mp_emulator.hardware import gpus as _gpus
            for attr in dir(_gpus):
                if attr.startswith("_"):
                    continue
                candidate = getattr(_gpus, attr, None)
                if isinstance(candidate, GPU) and candidate.name == self.name:
                    return f"gpus.{attr}"
        except Exception:
            pass
        return f"GPU({self.name!r})"

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
    unit : GPU | BaseHardware
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

    def total_flops(self, precision: str = "fp32", sparse: bool = False) -> float:
        """Sum of per-GPU ``max_flops(precision)`` across all units.

        Parameters
        ----------
        sparse : bool
            If *True*, use the structured-sparsity (2:4) throughput.
        """
        if sparse:
            key = f"{precision}_flops_sparse"
            val = self.total_spec(key)
            if val > 0:
                return val
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
        # Produce eval-able code: ``5 * gpus.A100`` or ``(2 * gpus.A100 + 3 * gpus.H100)``
        parts = []
        for u, c in self.units:
            parts.append(f"{c} * {u!r}")
        if len(parts) == 1:
            return parts[0]
        return f"({' + '.join(parts)})"


# ============================================================
# Predefined NVIDIA GPU specifications
# ============================================================
# All Tensor Core FLOP/s values are **dense** (non-sparsity).
# Sparse (2:4 structured sparsity) variants are stored in
# ``*_flops_sparse`` keys and equal 2x the dense value for
# GPUs that support it (Ampere and later).

gpus = SimpleNamespace(

    # -- Volta -----------------------------------------------------------
    # Datasheet: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    # SXM2 variant. FP16 Tensor Core perf = 125 TFLOPS (dense).
    # V100 pre-dates BF16/FP8/FP4 Tensor Core support and structured sparsity.

    V100=GPU(
        "V100-32GB",
        specs={
            "memory_bytes": 32 * GB,           # 32 GB HBM2
            "fp64_flops": 7.8 * TFLOPS,        # FP64
            "fp32_flops": 15.7 * TFLOPS,       # FP32
            "fp16_flops": 31.4 * TFLOPS,       # FP16 (CUDA cores, not Tensor)
            "bf16_flops": 0.0,                 # not supported
            "fp8_flops": 0.0,                  # not supported
            "fp4_flops": 0.0,                  # not supported
            "int8_tops": 0.0,                  # not supported
            # No sparse keys — V100 does not support structured sparsity
            "fp16_flops_sparse": 0.0,
            "bf16_flops_sparse": 0.0,
            "fp8_flops_sparse": 0.0,
            "fp4_flops_sparse": 0.0,
            "int8_tops_sparse": 0.0,
            "memory_bandwidth": 900e9,          # 900 GB/s HBM2
            "default_interconnect": "nvlink_2",
            # Architectural specs
            "cuda_cores": 5120,
            "tensor_cores": 640,
            "sm_count": 80,
            "shared_mem_per_sm": 96 * KB,
            "max_shared_mem_per_block": 96 * KB,
            "l2_cache_bytes": 6 * MB,
            "tdp_watts": 300,
        },
    ),

    # -- Ampere ----------------------------------------------------------
    # Datasheet (40 GB): https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
    # Datasheet (80 GB): https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf
    # SXM variants. All Tensor Core values are dense.

    A100_40=GPU(
        "A100-40GB",
        specs={
            "memory_bytes": 40 * GB,           # 40 GB HBM2e
            "fp64_flops": 9.7 * TFLOPS,        # FP64
            "fp32_flops": 19.5 * TFLOPS,       # FP32
            "fp16_flops": 312 * TFLOPS,        # FP16 Tensor Core (dense)
            "bf16_flops": 312 * TFLOPS,        # BF16 Tensor Core (dense)
            "fp8_flops": 0.0,                  # not supported
            "fp4_flops": 0.0,                  # not supported
            "int8_tops": 624 * TFLOPS,         # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 624 * TFLOPS,
            "bf16_flops_sparse": 624 * TFLOPS,
            "fp8_flops_sparse": 0.0,
            "fp4_flops_sparse": 0.0,
            "int8_tops_sparse": 1248 * TFLOPS,
            "memory_bandwidth": 1.555e12,       # 1.555 TB/s HBM2e (40GB SXM)
            "default_interconnect": "nvlink_3",
            # Architectural specs
            "cuda_cores": 6912,
            "tensor_cores": 432,
            "sm_count": 108,
            "shared_mem_per_sm": 164 * KB,
            "max_shared_mem_per_block": 163 * KB,
            "l2_cache_bytes": 40 * MB,
            "tdp_watts": 400,
        },
    ),

    A100=GPU(
        "A100-80GB",
        specs={
            "memory_bytes": 80 * GB,           # 80 GB HBM2e
            "fp64_flops": 9.7 * TFLOPS,        # FP64
            "fp32_flops": 19.5 * TFLOPS,       # FP32
            "fp16_flops": 312 * TFLOPS,        # FP16 Tensor Core (dense)
            "bf16_flops": 312 * TFLOPS,        # BF16 Tensor Core (dense)
            "fp8_flops": 0.0,                  # not supported
            "fp4_flops": 0.0,                  # not supported
            "int8_tops": 624 * TFLOPS,         # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 624 * TFLOPS,
            "bf16_flops_sparse": 624 * TFLOPS,
            "fp8_flops_sparse": 0.0,
            "fp4_flops_sparse": 0.0,
            "int8_tops_sparse": 1248 * TFLOPS,
            "memory_bandwidth": 2.039e12,       # 2.039 TB/s HBM2e (80GB SXM)
            "default_interconnect": "nvlink_3",
            # Architectural specs
            "cuda_cores": 6912,
            "tensor_cores": 432,
            "sm_count": 108,
            "shared_mem_per_sm": 164 * KB,
            "max_shared_mem_per_block": 163 * KB,
            "l2_cache_bytes": 40 * MB,
            "tdp_watts": 400,
        },
    ),

    # -- Hopper ----------------------------------------------------------
    # Datasheet: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # Product page: https://www.nvidia.com/en-us/data-center/h100/
    # H200 datasheet: https://nvidia.com/en-us/data-center/h200/
    # SXM variants. All Tensor Core values are dense (halved from datasheet
    # sparsity numbers).
    # H200 shares the Hopper compute die with H100 but has HBM3e memory.

    H100=GPU(
        "H100-80GB",
        specs={
            "memory_bytes": 80 * GB,           # 80 GB HBM3
            "fp64_flops": 33.5 * TFLOPS,       # FP64 Tensor Core
            "fp32_flops": 66.9 * TFLOPS,       # FP32
            "fp16_flops": 990 * TFLOPS,        # FP16 Tensor Core (dense)
            "bf16_flops": 990 * TFLOPS,        # BF16 Tensor Core (dense)
            "fp8_flops": 1979 * TFLOPS,        # FP8 Tensor Core (dense)
            "fp4_flops": 0.0,                  # not supported
            "int8_tops": 1979 * TFLOPS,        # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 1979 * TFLOPS,
            "bf16_flops_sparse": 1979 * TFLOPS,
            "fp8_flops_sparse": 3958 * TFLOPS,
            "fp4_flops_sparse": 0.0,
            "int8_tops_sparse": 3958 * TFLOPS,
            "memory_bandwidth": 3.35e12,        # 3.35 TB/s HBM3 (SXM)
            "default_interconnect": "nvlink_4",
            # Architectural specs
            "cuda_cores": 16896,
            "tensor_cores": 528,
            "sm_count": 132,
            "shared_mem_per_sm": 228 * KB,
            "max_shared_mem_per_block": 227 * KB,
            "l2_cache_bytes": 50 * MB,
            "tdp_watts": 700,
        },
    ),

    H200=GPU(
        "H200-141GB",
        specs={
            "memory_bytes": 141 * GB,          # 141 GB HBM3e
            "fp64_flops": 33.5 * TFLOPS,       # same Hopper compute die as H100
            "fp32_flops": 66.9 * TFLOPS,
            "fp16_flops": 990 * TFLOPS,        # FP16 Tensor Core (dense)
            "bf16_flops": 990 * TFLOPS,        # BF16 Tensor Core (dense)
            "fp8_flops": 1979 * TFLOPS,        # FP8 Tensor Core (dense)
            "fp4_flops": 0.0,                  # not supported
            "int8_tops": 1979 * TFLOPS,        # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 1979 * TFLOPS,
            "bf16_flops_sparse": 1979 * TFLOPS,
            "fp8_flops_sparse": 3958 * TFLOPS,
            "fp4_flops_sparse": 0.0,
            "int8_tops_sparse": 3958 * TFLOPS,
            "memory_bandwidth": 4.8e12,         # 4.8 TB/s HBM3e
            "default_interconnect": "nvlink_4",
            # Architectural specs
            "cuda_cores": 16896,
            "tensor_cores": 528,
            "sm_count": 132,
            "shared_mem_per_sm": 228 * KB,
            "max_shared_mem_per_block": 227 * KB,
            "l2_cache_bytes": 50 * MB,
            "tdp_watts": 700,
        },
    ),

    # -- Blackwell -------------------------------------------------------
    # B200 datasheet: https://www.nvidia.com/en-us/data-center/dgx-b200/
    # B100 specs: https://flopper.io/gpu/nvidia-b100-sxm-192gb
    # Architecture overview: https://cudocompute.com/blog/nvidias-blackwell-architecture-breaking-down-the-b100-b200-and-gb200
    # SXM variants. 2nd-gen Transformer Engine with FP4. 208B transistors.
    # All Tensor Core values are dense (halved from datasheet sparsity numbers).

    B100=GPU(
        "B100-192GB",
        specs={
            "memory_bytes": 192 * GB,          # 192 GB HBM3e
            "fp64_flops": 30 * TFLOPS,         # FP64
            "fp32_flops": 60 * TFLOPS,         # FP32
            "fp16_flops": 1750 * TFLOPS,       # FP16 Tensor Core (dense)
            "bf16_flops": 1750 * TFLOPS,       # BF16 Tensor Core (dense)
            "fp8_flops": 3500 * TFLOPS,        # FP8 Tensor Core (dense)
            "fp4_flops": 7000 * TFLOPS,        # FP4 Tensor Core (dense, NVFP4)
            "int8_tops": 3500 * TFLOPS,        # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 3500 * TFLOPS,
            "bf16_flops_sparse": 3500 * TFLOPS,
            "fp8_flops_sparse": 7000 * TFLOPS,
            "fp4_flops_sparse": 14_000 * TFLOPS,
            "int8_tops_sparse": 7000 * TFLOPS,
            "memory_bandwidth": 8.0e12,         # 8 TB/s HBM3e
            "default_interconnect": "nvlink_5",
            # Architectural specs
            "cuda_cores": 18432,
            "tensor_cores": 528,
            "sm_count": 144,
            "shared_mem_per_sm": 228 * KB,
            "max_shared_mem_per_block": 227 * KB,
            "l2_cache_bytes": 64 * MB,
            "tdp_watts": 700,
        },
    ),

    B200=GPU(
        "B200-192GB",
        specs={
            "memory_bytes": 192 * GB,          # 192 GB HBM3e
            "fp64_flops": 45 * TFLOPS,         # FP64
            "fp32_flops": 90 * TFLOPS,         # FP32
            "fp16_flops": 2250 * TFLOPS,       # FP16 Tensor Core (dense)
            "bf16_flops": 2250 * TFLOPS,       # BF16 Tensor Core (dense)
            "fp8_flops": 4500 * TFLOPS,        # FP8 Tensor Core (dense)
            "fp4_flops": 9000 * TFLOPS,        # FP4 Tensor Core (dense, NVFP4)
            "int8_tops": 4500 * TFLOPS,        # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 4500 * TFLOPS,
            "bf16_flops_sparse": 4500 * TFLOPS,
            "fp8_flops_sparse": 9000 * TFLOPS,
            "fp4_flops_sparse": 18_000 * TFLOPS,
            "int8_tops_sparse": 9000 * TFLOPS,
            "memory_bandwidth": 8.0e12,         # 8 TB/s HBM3e
            "default_interconnect": "nvlink_5",
            # Architectural specs
            "cuda_cores": 18432,
            "tensor_cores": 528,
            "sm_count": 144,
            "shared_mem_per_sm": 228 * KB,
            "max_shared_mem_per_block": 227 * KB,
            "l2_cache_bytes": 64 * MB,
            "tdp_watts": 1000,
        },
    ),

    # -- Blackwell Ultra -------------------------------------------------
    # Datasheet: https://www.tomorrow-net.co.jp/cms_k6TztgC2/wp-content/uploads/2025/03/blackwell-datasheet-ultra-blackwell-4169750.pdf
    # Tom's Hardware: https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4
    # 288 GB HBM3e, 1400W TDP. Ships H2 2025.

    B300=GPU(
        "B300-288GB",
        specs={
            "memory_bytes": 288 * GB,          # 288 GB HBM3e (12-high stacks)
            "fp64_flops": 1.25 * TFLOPS,       # FP64 (estimated from DGX specs)
            "fp32_flops": 37 * TFLOPS,         # FP32 (estimated)
            "fp16_flops": 2250 * TFLOPS,       # FP16 Tensor Core (dense)
            "bf16_flops": 2250 * TFLOPS,       # BF16 Tensor Core (dense)
            "fp8_flops": 4500 * TFLOPS,        # FP8 Tensor Core (dense)
            "fp4_flops": 14_000 * TFLOPS,      # FP4 Tensor Core (dense)
            "int8_tops": 4500 * TFLOPS,        # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 4500 * TFLOPS,
            "bf16_flops_sparse": 4500 * TFLOPS,
            "fp8_flops_sparse": 9000 * TFLOPS,
            "fp4_flops_sparse": 28_000 * TFLOPS,
            "int8_tops_sparse": 9000 * TFLOPS,
            "memory_bandwidth": 8.0e12,         # 8 TB/s HBM3e
            "default_interconnect": "nvlink_5",
            # Architectural specs
            "cuda_cores": 18432,
            "tensor_cores": 528,
            "sm_count": 144,
            "shared_mem_per_sm": 228 * KB,
            "max_shared_mem_per_block": 227 * KB,
            "l2_cache_bytes": 64 * MB,
            "tdp_watts": 1400,
        },
    ),

    # -- GeForce RTX 5090 ------------------------------------------------
    # Blackwell consumer flagship. GDDR7, PCIe Gen5.
    # Sources:
    #   https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
    #   https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216

    RTX_5090=GPU(
        "RTX 5090-32GB",
        specs={
            "memory_bytes": 32 * GB,           # 32 GB GDDR7
            "fp64_flops": 3.3 * TFLOPS,        # FP64 (1:32 rate)
            "fp32_flops": 105 * TFLOPS,        # FP32
            "fp16_flops": 210 * TFLOPS,        # FP16 Tensor Core (dense)
            "bf16_flops": 210 * TFLOPS,        # BF16 Tensor Core (dense)
            "fp8_flops": 420 * TFLOPS,         # FP8 Tensor Core (dense)
            "fp4_flops": 840 * TFLOPS,         # FP4 Tensor Core (dense)
            "int8_tops": 420 * TFLOPS,         # INT8 Tensor Core (dense)
            # Sparse (2:4) = 2x dense
            "fp16_flops_sparse": 420 * TFLOPS,
            "bf16_flops_sparse": 420 * TFLOPS,
            "fp8_flops_sparse": 840 * TFLOPS,
            "fp4_flops_sparse": 1680 * TFLOPS,
            "int8_tops_sparse": 840 * TFLOPS,
            "memory_bandwidth": 1.792e12,       # 1.792 TB/s GDDR7
            "default_interconnect": "pcie_gen5",
            # Architectural specs
            "cuda_cores": 21760,
            "tensor_cores": 680,
            "sm_count": 170,
            "shared_mem_per_sm": 228 * KB,
            "max_shared_mem_per_block": 227 * KB,
            "l2_cache_bytes": 98 * MB,
            "tdp_watts": 575,
        },
    ),
)
"""Predefined NVIDIA GPU specifications.

All values are for the SXM / data-center variants where applicable.
Tensor Core throughput is consistently listed as **dense** (non-sparsity).
Structured-sparsity (2:4) values are in ``*_flops_sparse`` keys.

Usage::

    from trovolut.hardware import gpus

    gpus.A100          # GPU('A100-80GB')
    gpus.H100          # GPU('H100-80GB')
    gpus.RTX_5090      # GPU('RTX 5090-32GB')
    cluster = 5 * gpus.A100
"""


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Classes
    "GPU",
    "GPUCluster",
    "Interconnect",
    "gpus",
    # Unit helpers
    "KB",
    "MB",
    "GB",
    "TB",
    "TFLOPS",
    # Predefined interconnects
    "NVLINK_2",
    "NVLINK_3",
    "NVLINK_4",
    "NVLINK_5",
    "PCIE_GEN4",
    "PCIE_GEN5",
]
