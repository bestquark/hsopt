"""Tests for Precision enum and PrecisionConfig.

Verify that every Precision variant reports the correct byte size,
and that PrecisionConfig defaults are sensible.
"""

import pytest

from trovolut.backends.gpu_mp_emulator.utils import Precision, PrecisionConfig


# ── Precision.bits ──────────────────────────────────────────────

EXPECTED_BITS = {
    Precision.FP64:     64,
    Precision.FP32:     32,
    Precision.FP16:     16,
    Precision.BF16:     16,
    Precision.FP8_E4M3:  8,
    Precision.FP8_E5M2:  8,
    Precision.FP4_E2M1:  4,
    Precision.FP4_E4M0:  4,
    Precision.INT8:      8,
}


@pytest.mark.parametrize("prec, expected_bits", EXPECTED_BITS.items(), ids=lambda p: p.value if hasattr(p, "value") else str(p))
def test_precision_bits(prec, expected_bits):
    """Each Precision variant must report the correct number of bits."""
    assert prec.bits == expected_bits


@pytest.mark.parametrize("prec, expected_bits", EXPECTED_BITS.items(), ids=lambda p: p.value if hasattr(p, "value") else str(p))
def test_precision_bytes(prec, expected_bits):
    """bytes = max(1, bits // 8) — this is the formula used by cost models."""
    expected_bytes = max(1, expected_bits // 8)
    actual_bytes = max(1, prec.bits // 8)
    assert actual_bytes == expected_bytes


# ── PrecisionConfig defaults ────────────────────────────────────

def test_precision_config_defaults():
    pc = PrecisionConfig()
    assert pc.compute == Precision.FP32
    assert pc.storage == Precision.FP32
    assert pc.accumulate == Precision.FP64


def test_precision_config_custom():
    pc = PrecisionConfig(
        compute=Precision.FP16,
        storage=Precision.FP16,
        accumulate=Precision.FP32,
    )
    assert pc.compute == Precision.FP16
    assert pc.storage == Precision.FP16
    assert pc.accumulate == Precision.FP32
