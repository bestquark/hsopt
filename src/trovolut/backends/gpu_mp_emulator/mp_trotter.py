"""
Mixed-precision Trotter framework (public API).

Re-exports from _mp_utils so existing code continues to work.
For dynamics with unified backend API and quantization schedules, use:
    from trovolut.backends.gpu_mp_emulator.dynamics import MixedPrecisionTrotterEvolution, Precision
"""

from trovolut.backends.gpu_mp_emulator._mp_utils import (
    Precision,
    PrecisionConfig,
    PrecisionCaster,
    ExpTerm,
    Exponentiator,
    ExactExponentiator,
    TaylorExponentiator,
    KrylovExponentiator,
    TrotterFormula,
    SecondOrderTrotter,
    TrotterSimulator,
    PrecisionStats,
)

__all__ = [
    "Precision",
    "PrecisionConfig",
    "PrecisionCaster",
    "ExpTerm",
    "Exponentiator",
    "ExactExponentiator",
    "TaylorExponentiator",
    "KrylovExponentiator",
    "TrotterFormula",
    "SecondOrderTrotter",
    "TrotterSimulator",
    "PrecisionStats",
]
