from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math

from trovolut.backends.gpu_mp_emulator._mp_utils import Precision


@dataclass(frozen=True)
class MPFormat:
    """Describe a floating-point format as exponent/mantissa bits."""
    name: str
    e_bits: int
    m_bits: int


_KNOWN_FORMATS = {
    "FP4_E2M1": MPFormat("FP4_E2M1", e_bits=2, m_bits=1),
    "FP4_E4M0": MPFormat("FP4_E4M0", e_bits=3, m_bits=0),  # NVFP4 in this codebase
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
        # Accept E2M1-like strings
        if "E" in key and "M" in key:
            try:
                e_part = key.split("E", 1)[1]
                e_bits = int(e_part.split("M", 1)[0])
                m_bits = int(e_part.split("M", 1)[1])
                return MPFormat(f"E{e_bits}M{m_bits}", e_bits=e_bits, m_bits=m_bits)
            except Exception as exc:
                raise ValueError(f"Could not parse precision format '{fmt}'") from exc
    raise TypeError("fmt must be Precision, MPFormat, (e_bits, m_bits), or str like 'E2M1'")


def _enumerate_positive_values(e_bits: int, m_bits: int, include_zero: bool,
                               include_subnormals: bool) -> List[float]:
    bias = 2**(e_bits - 1) - 1
    exp_min = 1 - bias
    exp_max = 2**e_bits - 1 - bias
    values = set()
    if include_zero:
        values.add(0.0)
    if m_bits > 0 and include_subnormals:
        # Subnormals: exponent fixed at exp_min, no implicit 1
        for mant in range(1, 2**m_bits):
            values.add((mant / 2**m_bits) * 2**exp_min)
    # Normals
    for exp in range(exp_min, exp_max + 1):
        if m_bits == 0:
            values.add(2.0**exp)
        else:
            for mant in range(0, 2**m_bits):
                values.add((1.0 + mant / 2**m_bits) * 2**exp)
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

    bias = 2**(e_bits - 1) - 1
    exp_min = 1 - bias
    exp_max = 2**e_bits - 1 - bias
    min_normal = 2.0**exp_min
    max_normal = (2.0**exp_max) * (2.0 - (2.0**(-m_bits)) if m_bits > 0 else 1.0)
    min_subnormal = (2.0**exp_min) * (2.0**(-m_bits)) if m_bits > 0 else min_normal

    # Count of positive values including zero
    sub_count = (2**m_bits - 1) if (include_subnormals and m_bits > 0) else 0
    norm_count = (exp_max - exp_min + 1) * (2**m_bits if m_bits > 0 else 1)
    pos_count = (1 if include_zero else 0) + sub_count + norm_count
    total_count = pos_count * (2 if include_negatives else 1) - (1 if include_zero and include_negatives else 0)

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

    positives = _enumerate_positive_values(e_bits, m_bits, include_zero, include_subnormals)
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
