"""
Mixed-precision utilities for the GPU emulator backend.

Precision definitions, casting infrastructure, and format enumeration.

Public API:
    Precision, PrecisionConfig, PrecisionCaster,
    MPFormat, representable_values
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from enum import Enum
import numpy as np


# ============================================================
# 1. Precision definitions and casting infrastructure
# ============================================================

class Precision(Enum):
    """Supported precision levels for mixed-precision emulation."""

    FP64 = "float64"
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8_E4M3 = "float8_e4m3"   # NVIDIA FP8 (range)
    FP8_E5M2 = "float8_e5m2"   # NVIDIA FP8 (precision)
    FP4_E2M1 = "float4_e2m1"   # NVIDIA NVFP4 (1 sign, 2 exp, 1 mantissa)
    FP4_E4M0 = "float4_e4m0"   # NVIDIA NVFP4 (1 sign, 3 exp, 0 mantissa)
    INT8 = "int8"               # quantized

    @property
    def bits(self) -> int:
        return {
            "float64": 64, "float32": 32, "float16": 16,
            "bfloat16": 16, "float8_e4m3": 8, "float8_e5m2": 8,
            "float4_e2m1": 4, "float4_e4m0": 4, "int8": 8,
        }[self.value]


@dataclass
class PrecisionConfig:
    """Fine-grained control over precision at different stages."""

    compute: Precision = Precision.FP32       # Precision for matrix ops
    storage: Precision = Precision.FP32       # Precision for storing generators
    accumulate: Precision = Precision.FP64    # Precision for state accumulation

    # Quantization parameters (for INT8/FP8)
    scale: float | None = None
    zero_point: float = 0.0
    dynamic_scaling: bool = True


class PrecisionCaster:
    """Handles casting between precisions, including emulated formats."""

    NATIVE_DTYPES = {
        Precision.FP64: (np.float64, np.complex128),
        Precision.FP32: (np.float32, np.complex64),
        Precision.FP16: (np.float16, np.complex64),
    }

    def __init__(self, use_torch: bool = False, use_cuda: bool = False):
        self.use_torch = use_torch
        self.use_cuda = use_cuda
        self._torch = None
        if use_torch:
            try:
                import torch
                self._torch = torch
            except ImportError:
                self.use_torch = False

    def cast(
        self,
        arr: np.ndarray,
        precision: Precision,
        config: PrecisionConfig | None = None,
    ) -> np.ndarray:
        is_complex = np.iscomplexobj(arr)
        if precision in self.NATIVE_DTYPES:
            real_dtype, complex_dtype = self.NATIVE_DTYPES[precision]
            target_dtype = complex_dtype if is_complex else real_dtype
            return arr.astype(target_dtype)
        if is_complex:
            real_part = self._cast_real(arr.real, precision, config)
            imag_part = self._cast_real(arr.imag, precision, config)
            return real_part.astype(np.float32) + 1j * imag_part.astype(np.float32)
        return self._cast_real(arr, precision, config)

    def _cast_real(self, arr, precision, config):
        arr = np.asarray(arr, dtype=np.float64)
        if precision == Precision.BF16:
            return self._to_bfloat16(arr)
        elif precision == Precision.FP8_E4M3:
            return self._to_fp8_e4m3(arr, config)
        elif precision == Precision.FP8_E5M2:
            return self._to_fp8_e5m2(arr, config)
        elif precision == Precision.FP4_E2M1:
            return self._to_fp4_e2m1(arr, config)
        elif precision == Precision.FP4_E4M0:
            return self._to_fp4_e4m0(arr, config)
        elif precision == Precision.INT8:
            return self._to_int8(arr, config)
        raise ValueError(f"Unsupported precision: {precision}")

    def _to_bfloat16(self, arr):
        if self.use_torch and self._torch:
            t = self._torch.from_numpy(arr.astype(np.float32))
            return t.to(self._torch.bfloat16).float().numpy()
        fp32 = arr.astype(np.float32).copy()
        int_view = fp32.view(np.uint32)
        int_view &= 0xFFFF0000
        return fp32

    def _to_fp8_e4m3(self, arr, config):
        return self._quantize_to_fp8(arr, e_bits=4, m_bits=3, config=config)

    def _to_fp8_e5m2(self, arr, config):
        return self._quantize_to_fp8(arr, e_bits=5, m_bits=2, config=config)

    def _to_fp4_e2m1(self, arr, config):
        return self._quantize_to_fp4(arr, e_bits=2, m_bits=1, config=config)

    def _to_fp4_e4m0(self, arr, config):
        return self._quantize_to_fp4(arr, e_bits=3, m_bits=0, config=config)

    def _quantize_to_fp4(self, arr, e_bits, m_bits, config):
        bias = 2 ** (e_bits - 1) - 1
        max_exp = 2 ** e_bits - 1 - bias
        min_exp = 1 - bias
        max_val = 2 ** max_exp * (2 - 2 ** (-m_bits)) if m_bits > 0 else 2 ** max_exp
        min_normal = 2 ** min_exp
        fp32 = arr.astype(np.float32)
        sign = np.sign(fp32)
        abs_val = np.abs(fp32)
        result = np.zeros_like(fp32)
        nonzero_mask = abs_val > 0
        if np.any(nonzero_mask):
            abs_nz = np.clip(
                abs_val[nonzero_mask],
                2 ** (min_exp - m_bits) if m_bits > 0 else min_normal,
                max_val,
            )
            if m_bits > 0:
                log2_val = np.floor(np.log2(abs_nz + 1e-45))
                mantissa_lsb = 2 ** (log2_val - m_bits)
                quantized = np.round(abs_nz / mantissa_lsb) * mantissa_lsb
            else:
                log2_val = np.round(np.log2(abs_nz + 1e-45))
                log2_val = np.clip(log2_val, min_exp, max_exp)
                quantized = 2.0 ** log2_val
            result[nonzero_mask] = sign[nonzero_mask] * quantized
        return result

    def _quantize_to_fp8(self, arr, e_bits, m_bits, config):
        if self.use_torch and self._torch and hasattr(self._torch, "float8_e4m3fn"):
            t = self._torch.from_numpy(arr.astype(np.float32))
            if self.use_cuda and self._torch.cuda.is_available():
                t = t.cuda()
            dtype = self._torch.float8_e4m3fn if e_bits == 4 else self._torch.float8_e5m2
            result = t.to(dtype).float()
            if self.use_cuda:
                result = result.cpu()
            return result.numpy()
        bias = 2 ** (e_bits - 1) - 1
        max_exp = 2 ** e_bits - 1 - bias
        min_exp = 1 - bias
        max_val = 2 ** max_exp * (2 - 2 ** (-m_bits))
        fp32 = arr.astype(np.float32)
        sign = np.sign(fp32)
        abs_val = np.abs(fp32)
        nonzero_mask = abs_val > 0
        result = np.zeros_like(fp32)
        if np.any(nonzero_mask):
            abs_nz = abs_val[nonzero_mask]
            abs_nz = np.clip(abs_nz, 2 ** (min_exp - m_bits), max_val)
            log2_val = np.floor(np.log2(abs_nz + 1e-45))
            mantissa_lsb = 2 ** (log2_val - m_bits)
            quantized = np.round(abs_nz / mantissa_lsb) * mantissa_lsb
            result[nonzero_mask] = sign[nonzero_mask] * quantized
        return result

    def _to_int8(self, arr, config):
        config = config or PrecisionConfig()
        if config.dynamic_scaling or config.scale is None:
            max_val = np.max(np.abs(arr))
            scale = max_val / 127.0 if max_val > 0 else 1.0
        else:
            scale = config.scale
        quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
        return quantized.astype(np.float32) * scale


# ============================================================
# 2. Format enumeration utilities
# ============================================================

@dataclass(frozen=True)
class MPFormat:
    """Describe a floating-point format as exponent/mantissa bits."""

    name: str
    e_bits: int
    m_bits: int


_KNOWN_FORMATS = {
    "FP4_E2M1": MPFormat("FP4_E2M1", e_bits=2, m_bits=1),
    "FP4_E4M0": MPFormat("FP4_E4M0", e_bits=3, m_bits=0),
    "FP8_E4M3": MPFormat("FP8_E4M3", e_bits=4, m_bits=3),
    "FP8_E5M2": MPFormat("FP8_E5M2", e_bits=5, m_bits=2),
}


def _parse_format(fmt) -> MPFormat:
    """Parse Precision/str/tuple into an MPFormat."""
    if isinstance(fmt, MPFormat):
        return fmt
    if isinstance(fmt, Precision):
        key = fmt.name
        if key in _KNOWN_FORMATS:
            return _KNOWN_FORMATS[key]
        raise ValueError(f"Unsupported precision: {fmt}")
    if isinstance(fmt, tuple) and len(fmt) == 2:
        e_bits, m_bits = fmt
        return MPFormat(f"E{e_bits}M{m_bits}", e_bits=int(e_bits), m_bits=int(m_bits))
    if isinstance(fmt, str):
        key = fmt.strip().upper()
        if key in _KNOWN_FORMATS:
            return _KNOWN_FORMATS[key]
        if "E" in key and "M" in key:
            try:
                e_part = key.split("E", 1)[1]
                e_bits = int(e_part.split("M", 1)[0])
                m_bits = int(e_part.split("M", 1)[1])
                return MPFormat(f"E{e_bits}M{m_bits}", e_bits=e_bits, m_bits=m_bits)
            except Exception as exc:
                raise ValueError(f"Could not parse precision format '{fmt}'") from exc
    raise TypeError(
        "fmt must be Precision, MPFormat, (e_bits, m_bits), or str like 'E2M1'"
    )


def _enumerate_positive_values(
    e_bits: int, m_bits: int, include_zero: bool, include_subnormals: bool
) -> List[float]:
    bias = 2 ** (e_bits - 1) - 1
    exp_min = 1 - bias
    exp_max = 2 ** e_bits - 1 - bias
    values = set()
    if include_zero:
        values.add(0.0)
    if m_bits > 0 and include_subnormals:
        for mant in range(1, 2 ** m_bits):
            values.add((mant / 2 ** m_bits) * 2 ** exp_min)
    for exp in range(exp_min, exp_max + 1):
        if m_bits == 0:
            values.add(2.0 ** exp)
        else:
            for mant in range(0, 2 ** m_bits):
                values.add((1.0 + mant / 2 ** m_bits) * 2 ** exp)
    return sorted(values)


def representable_values(
    fmt,
    *,
    include_zero: bool = True,
    include_negatives: bool = True,
    include_subnormals: bool = True,
    max_values: int = 4096,
):
    """
    Return all representable values for a small MP format, or a summary for large ones.

    Args:
        fmt: Precision, MPFormat, (e_bits, m_bits), or string like "E2M1" / "FP8_E4M3".
        include_zero: include +0.0 in the set.
        include_negatives: include negative values.
        include_subnormals: include subnormal values when m_bits > 0.
        max_values: if total count exceeds this, return summary instead of full list.
    """
    mp = _parse_format(fmt)
    e_bits, m_bits = mp.e_bits, mp.m_bits

    bias = 2 ** (e_bits - 1) - 1
    exp_min = 1 - bias
    exp_max = 2 ** e_bits - 1 - bias
    min_normal = 2.0 ** exp_min
    max_normal = (2.0 ** exp_max) * (
        (2.0 - (2.0 ** (-m_bits))) if m_bits > 0 else 1.0
    )
    min_subnormal = (2.0 ** exp_min) * (2.0 ** (-m_bits)) if m_bits > 0 else min_normal

    sub_count = (2 ** m_bits - 1) if (include_subnormals and m_bits > 0) else 0
    norm_count = (exp_max - exp_min + 1) * (2 ** m_bits if m_bits > 0 else 1)
    pos_count = (1 if include_zero else 0) + sub_count + norm_count
    total_count = pos_count * (2 if include_negatives else 1) - (
        1 if include_zero and include_negatives else 0
    )

    if max_values is not None and total_count > max_values:
        return {
            "format": mp.name,
            "e_bits": e_bits,
            "m_bits": m_bits,
            "bias": bias,
            "exp_min": exp_min,
            "exp_max": exp_max,
            "min_positive": min_subnormal,
            "min_normal": min_normal,
            "max_normal": max_normal,
            "count": total_count,
            "values": None,
            "note": f"Total values ({total_count}) exceeds max_values={max_values}.",
        }

    positives = _enumerate_positive_values(
        e_bits, m_bits, include_zero, include_subnormals
    )
    if include_negatives:
        negatives = [-v for v in positives if v != 0.0]
        values = sorted(negatives + positives)
    else:
        values = positives

    return {
        "format": mp.name,
        "e_bits": e_bits,
        "m_bits": m_bits,
        "bias": bias,
        "exp_min": exp_min,
        "exp_max": exp_max,
        "min_positive": min_subnormal,
        "min_normal": min_normal,
        "max_normal": max_normal,
        "count": total_count,
        "values": values,
    }


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Precision infrastructure
    "Precision",
    "PrecisionConfig",
    "PrecisionCaster",
    # Format enumeration
    "MPFormat",
    "representable_values",
]
