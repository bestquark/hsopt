"""Unified tensor core GEMM interface with FP4/FP8/FP16/FP32 support.

Auto-detects GPU capability and dispatches to the optimal precision path.
Architecture-specific subclasses (A100GEMM, H100GEMM, GB200GEMM) provide
hardware tensor core paths; the base class handles FP32 and FP16 only.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# GPU capability detection thresholds
_SM_80 = 80  # A100: FP16 tensor cores
_SM_90 = 90  # H100: FP8 E4M3 tensor cores
_SM_100 = 100  # GB200: FP4 E2M1 tensor cores

_SINGLETON: Optional["TensorCoreGEMM"] = None

def _get_sm_version(device: Optional[torch.device] = None) -> int:
    """Return SM version (e.g. 90 for H100) or 0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0
    idx = 0 if device is None else device.index or 0
    major, minor = torch.cuda.get_device_capability(idx)
    return major * 10 + minor


def _resolve_torch_dtype(name: str) -> torch.dtype:
    """Convert string dtype name to torch.dtype."""
    mapping = {
        "fp64": torch.float64,
        "float64": torch.float64,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name in mapping:
        return mapping[name]
    if "float8" in name or name == "fp8":
        return torch.float8_e4m3fn
    return torch.float32


class TensorCoreGEMM:
    """Base GEMM interface — FP32 and FP16 only.

    Subclasses (A100GEMM, H100GEMM, GB200GEMM) add higher-precision
    tensor core paths.  The base class falls back to FP16 for any
    precision request it cannot handle (fp8, fp4).
    """

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.sm_version = _get_sm_version(device)

        # Throughput tracking
        self.total_flops = 0
        self.total_time = 0.0
        self._call_count = 0

    @property
    def max_precision(self) -> str:
        if self.sm_version >= _SM_80:
            return "fp16"
        return "fp32"

    def gemm(self, a: torch.Tensor, b: torch.Tensor,
             precision_override: Optional[str] = None) -> torch.Tensor:
        """Matrix multiply with automatic precision selection.

        Args:
            a: Left matrix (..., M, K)
            b: Right matrix (..., K, N)
            precision_override: Force specific precision ("fp4", "fp8", "fp16", "fp32")

        Returns:
            Result tensor in FP32.
        """
        precision = precision_override or self.max_precision
        m, k = a.shape[-2], a.shape[-1]
        n = b.shape[-1]
        flops = 2 * m * k * n

        t0 = time.perf_counter()
        result = self._dispatch(a, b, precision)
        elapsed = time.perf_counter() - t0

        self.total_flops += flops
        self.total_time += elapsed
        self._call_count += 1
        return result

    def _dispatch(self, a: torch.Tensor, b: torch.Tensor, precision: str) -> torch.Tensor:
        is_complex = a.is_complex() or b.is_complex()

        if is_complex and precision in ("fp8", "fp4", "fp16", "bf16"):
            if precision == "fp8" and hasattr(self, "_complex_fp8_gemm"):
                return self._complex_fp8_gemm(a, b)
            if precision == "fp4" and hasattr(self, "_complex_fp4_gemm"):
                return self._complex_fp4_gemm(a, b)
            if precision in ("fp16", "bf16"):
                return self._complex_fp16_gemm(a, b)
            return torch.mm(a.to(torch.complex64), b.to(torch.complex64))

        if precision == "fp64":
            return self._gemm_fp64(a, b)
        elif precision == "fp32":
            return self._gemm_fp32(a, b)
        elif precision in ("fp16", "bf16"):
            return self._gemm_fp16(a, b)
        elif precision == "fp8":
            return self._gemm_fp8(a, b)
        elif precision == "fp4":
            return self._gemm_fp4(a, b)
        return self._gemm_fp32(a, b)

    def _gemm_fp64(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.is_complex() or b.is_complex():
            return torch.mm(a.to(torch.complex128), b.to(torch.complex128))
        return torch.mm(a.double(), b.double())

    def _gemm_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.is_complex() or b.is_complex():
            return torch.mm(a.to(torch.complex64), b.to(torch.complex64))
        return torch.mm(a.float(), b.float())

    def _complex_fp16_gemm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex GEMM via 4 real FP16 tensor-core GEMMs."""
        a = a.to(torch.complex64)
        b = b.to(torch.complex64)
        ar, ai = a.real.half(), a.imag.half()
        br, bi = b.real.half(), b.imag.half()
        rr = torch.mm(ar, br).float()
        ri = torch.mm(ai, bi).float()
        ir = torch.mm(ar, bi).float()
        ii = torch.mm(ai, br).float()
        return torch.complex(rr - ri, ir + ii)

    def _gemm_fp16(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.is_complex() or b.is_complex():
            return self._complex_fp16_gemm(a, b)
        return torch.mm(a.half(), b.half()).float()

    def _gemm_fp8(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Base class fallback — no FP8 tensor cores, use FP16."""
        return self._gemm_fp16(a, b)

    def _gemm_fp4(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Base class fallback — no FP4 tensor cores, use FP16."""
        return self._gemm_fp16(a, b)

    @property
    def throughput_gflops(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_flops / self.total_time / 1e9

    def reset_stats(self) -> None:
        self.total_flops = 0
        self.total_time = 0.0
        self._call_count = 0


def get_tc_gemm(device: Optional[torch.device] = None) -> TensorCoreGEMM:
    """Get or create the singleton GEMM instance for the detected architecture."""
    global _SINGLETON
    if _SINGLETON is None or (device is not None and _SINGLETON.device != device):
        sm = _get_sm_version(device)
        # Architecture-specific subclasses are optional imports
        if sm >= _SM_100:
            try:
                from gb200.gemm_gb200 import GB200GEMM
                _SINGLETON = GB200GEMM(device)
            except ImportError:
                _SINGLETON = TensorCoreGEMM(device)
        elif sm >= _SM_90:
            try:
                from h100.gemm_h100 import H100GEMM
                _SINGLETON = H100GEMM(device)
            except ImportError:
                _SINGLETON = TensorCoreGEMM(device)
        elif sm >= _SM_80:
            try:
                from a100.gemm_a100 import A100GEMM
                _SINGLETON = A100GEMM(device)
            except ImportError:
                _SINGLETON = TensorCoreGEMM(device)
        else:
            _SINGLETON = TensorCoreGEMM(device)
    return _SINGLETON


def tc_gemm(a: torch.Tensor, b: torch.Tensor,
            precision: Optional[str] = None) -> torch.Tensor:
    """Convenience function for singleton GEMM."""
    return get_tc_gemm(a.device).gemm(a, b, precision_override=precision)


# --- Quantization utilities ---

def quantize_to_fp8(tensor: torch.Tensor,
                    return_scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 tensor to FP8 E4M3 with per-tensor scale."""
    amax = tensor.abs().max().clamp(min=1e-12)
    fp8_max = 448.0
    scale = amax / fp8_max
    scaled = tensor / scale
    fp8_tensor = scaled.to(torch.float8_e4m3fn)
    if return_scale:
        return fp8_tensor, scale
    return fp8_tensor, torch.ones(1, device=tensor.device)


def dequantize_from_fp8(fp8_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor back to FP32."""
    return fp8_tensor.float() * scale


def quantize_to_fp4(tensor: torch.Tensor,
                    block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to simulated FP4 E2M1 using block-wise scaling."""
    flat = tensor.flatten().float()
    n = flat.numel()
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)

    amax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    fp4_max = 6.0
    scales = amax / fp4_max
    normalized = blocks / scales

    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=tensor.device)
    signs = normalized.sign()
    abs_vals = normalized.abs()
    diffs = (abs_vals.unsqueeze(-1) - fp4_values.unsqueeze(0).unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)
    sign_bits = (signs < 0).to(torch.int8) << 3
    packed = (indices.to(torch.int8) | sign_bits).view(-1)
    if pad > 0:
        packed = packed[:n]
    return packed.view(tensor.shape), scales.squeeze(1)


def dequantize_from_fp4(packed: torch.Tensor, scales: torch.Tensor,
                        block_size: int = 32) -> torch.Tensor:
    """Dequantize FP4 block-quantized tensor back to FP32."""
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    original_shape = packed.shape
    flat = packed.flatten().to(torch.int8)
    n = flat.numel()

    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat = F.pad(flat.float(), (0, pad)).to(torch.int8)

    blocks = flat.view(-1, block_size)
    indices = blocks & 0x07
    sign_bits = (blocks >> 3) & 0x01
    signs = 1.0 - 2.0 * sign_bits.float()
    values = fp4_values[indices.long()] * signs * scales.unsqueeze(1)

    result = values.reshape(-1)
    if pad > 0:
        result = result[:n]
    return result.view(original_shape)


__all__ = [
    "TensorCoreGEMM",
    "get_tc_gemm",
    "tc_gemm",
    "quantize_to_fp8",
    "dequantize_from_fp8",
    "quantize_to_fp4",
    "dequantize_from_fp4",
    "_resolve_torch_dtype",
    "_get_sm_version",
]
