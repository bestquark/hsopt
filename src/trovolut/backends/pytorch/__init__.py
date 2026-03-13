"""PyTorch TEBD backend for trovolut.

Provides tensor network (MPS/TEBD) simulation with mixed-precision
tensor core acceleration. Ported from the TEBD standalone package.

Requires: torch (PyTorch)

Modules:
    dynamics     -- TEBDEvolution (BaseDynamics-conforming)
    measure      -- CUDA event timing, memory tracking
    constraints  -- GPUMemory, Throughput, Fidelity
    hardware     -- CUDA device detection
    mps          -- MPS, FP8Tensor, DistributedMPS
    tc_ops       -- TensorCoreGEMM, FP4/FP8 quantize/dequantize
    precision_policy -- PrecisionPolicy, evolvable parameters
    truncation   -- SVD, randomized SVD, variational truncation
    gates        -- TrotterGates from Hamiltonians
    gate_applicator  -- Precision-policy-aware gate application
    sweep        -- TEBD sweep scheduler
"""

from __future__ import annotations


def _check_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = [
    "_check_available",
]
