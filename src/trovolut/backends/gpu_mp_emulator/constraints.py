"""
Constraints for the GPU mixed-precision emulator backend.

Hardware-agnostic constraints (Time, Error, Custom) are in:
    trovolut.core.base_constraints

Legate/Legion profiling constraints are in:
    trovolut.backends.gpu.constraints

This module contains constraints specific to the mixed-precision emulator:

- MemoryCost:     Peak memory (bytes) from CostProfile
- FlopsCost:      Total FLOPs from CostProfile
- SpaceTimeCost:  Space x Time product from CostProfile
- MatMulCount:    Matrix multiplication operation count
"""

from __future__ import annotations

from typing import Any, Dict

# Import hardware-agnostic base classes
from trovolut.core.base_constraints import (
    Constraint,
    ConstraintResult,
    Group,
    _score_fn,
    _formula_str,
    # Re-export common constraints for convenience
    Time,
    Error,
    Custom,
)


# =============================================================================
# Mixed-Precision Cost Constraints
# =============================================================================

class MemoryCost(Constraint):
    """Constraint on peak memory (bytes) from CostProfile.

    Works with the unified Metrics class -- the ``peak_memory_bytes``
    key is injected into the metrics dict by ``CostProfile.to_dict()``.
    """

    def __init__(
        self,
        max_memory_bytes: float = 100 * 1024 * 1024,  # 100 MB default
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("memory_cost", weight, hard, hard_penalty)
        self.max_memory_bytes = max_memory_bytes
        self.score_fn = score_fn

    def __repr__(self) -> str:
        parts = [f"{self.max_memory_bytes!r}"]
        if self.weight != 1.0:
            parts.append(f"weight={self.weight!r}")
        if self.hard:
            parts.append("hard=True")
        if self.hard_penalty != -100.0:
            parts.append(f"hard_penalty={self.hard_penalty!r}")
        if self.score_fn != "linear":
            parts.append(f"score_fn={self.score_fn!r}")
        return f"MemoryCost({', '.join(parts)})"

    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("peak_memory_bytes", m.get("memory_bytes", float("inf"))))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_memory_bytes, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_memory_bytes

    def to_prompt(self) -> str:
        mb = self.max_memory_bytes / (1024 ** 2)
        if self.hard:
            return f"Peak memory MUST stay below {mb:.0f} MB (hard)."
        return f"Minimize peak memory (target: <{mb:.0f} MB, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "peak_memory_bytes", self.max_memory_bytes,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


class FlopsCost(Constraint):
    """Constraint on total FLOPs from CostProfile.

    Scores the ``total_flops`` key returned by ``CostProfile.to_dict()``.
    """

    def __init__(
        self,
        max_flops: float = 1e12,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("flops_cost", weight, hard, hard_penalty)
        self.max_flops = max_flops
        self.score_fn = score_fn

    def __repr__(self) -> str:
        parts = [f"{self.max_flops!r}"]
        if self.weight != 1.0:
            parts.append(f"weight={self.weight!r}")
        if self.hard:
            parts.append("hard=True")
        if self.hard_penalty != -100.0:
            parts.append(f"hard_penalty={self.hard_penalty!r}")
        if self.score_fn != "linear":
            parts.append(f"score_fn={self.score_fn!r}")
        return f"FlopsCost({', '.join(parts)})"

    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("total_flops", float("inf")))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_flops, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_flops

    def to_prompt(self) -> str:
        if self.hard:
            return f"Total FLOPs MUST stay below {self.max_flops:.2e} (hard)."
        return f"Minimize FLOPs (target: <{self.max_flops:.2e}, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "total_flops", self.max_flops,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


class SpaceTimeCost(Constraint):
    """Constraint on emulated spacetime cost.

    **Hardware-aware mode** (recommended for GPU emulation):
        Pass a ``hardware`` cluster (e.g. ``5 * gpus.A100``).  The cost is
        computed as ``peak_memory_bytes * emulated_time`` where emulated
        time divides FLOPs at each precision by the hardware's throughput
        at that precision and sums (byte-seconds).

    **Abstract mode** (backward-compatible, for QPU analysis):
        When ``hardware`` is *None*, falls back to the abstract metric
        ``peak_memory_bytes * total_depth`` (bytes * circuit-depth).

    Scores the result against ``max_product``.
    """

    def __init__(
        self,
        max_product: float = 1e15,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
        hardware=None,
    ):
        super().__init__("space_time_cost", weight, hard, hard_penalty)
        self.max_product = max_product
        self.score_fn = score_fn
        self.hardware = hardware

    def __repr__(self) -> str:
        parts = []
        if self.max_product != 1e15:
            parts.append(f"max_product={self.max_product!r}")
        if self.weight != 1.0:
            parts.append(f"weight={self.weight!r}")
        if self.hard:
            parts.append("hard=True")
        if self.hard_penalty != -100.0:
            parts.append(f"hard_penalty={self.hard_penalty!r}")
        if self.score_fn != "linear":
            parts.append(f"score_fn={self.score_fn!r}")
        if self.hardware is not None:
            parts.append(f"hardware={self.hardware!r}")
        return f"SpaceTimeCost({', '.join(parts)})"

    def extract(self, m: Dict[str, Any]) -> float:
        if self.hardware is not None:
            return self._extract_emulated(m)
        return float(m.get("space_time_product", float("inf")))

    def _extract_emulated(self, m: Dict[str, Any]) -> float:
        """Compute precision-aware emulated spacetime (byte-seconds)."""
        peak_mem = float(m.get("peak_memory_bytes", 0))
        flops_by_prec = m.get("flops_by_precision", {})
        if not flops_by_prec:
            # Fallback: treat all FLOPs as FP32
            total_flops = float(m.get("total_flops", 0))
            _spec = getattr(self.hardware, "total_spec", None) or self.hardware.spec
            throughput = _spec("fp32_flops")
            return peak_mem * (total_flops / max(throughput, 1.0))

        _PREC_TO_HW = {
            "float64": "fp64_flops", "float32": "fp32_flops",
            "float16": "fp16_flops", "bfloat16": "bf16_flops",
            "float8_e4m3": "fp8_flops", "float8_e5m2": "fp8_flops",
            "int8": "fp8_flops", "float4": "fp4_flops",
        }
        _spec = getattr(self.hardware, "total_spec", None) or self.hardware.spec
        emulated_time = 0.0
        for prec_key, flops in flops_by_prec.items():
            hw_key = _PREC_TO_HW.get(prec_key, "fp32_flops")
            throughput = _spec(hw_key)
            if throughput > 0:
                emulated_time += flops / throughput
            else:
                emulated_time += flops / max(_spec("fp32_flops"), 1.0)
        return peak_mem * emulated_time

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_product, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_product

    def to_prompt(self) -> str:
        mode = "byte-seconds" if self.hardware else "bytes*depth"
        if self.hard:
            return f"Spacetime cost ({mode}) MUST stay below {self.max_product:.2e} (hard)."
        return (
            f"Minimize spacetime cost ({mode})"
            f" (target: <{self.max_product:.2e}, weight: {self.weight})."
        )

    def formula(self) -> str:
        metric = "emulated_spacetime" if self.hardware else "space_time_product"
        return _formula_str(
            metric, self.max_product,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


# =============================================================================
# Emulator-specific static operation counter
# =============================================================================

class MatMulCount(Constraint):
    """Constraint on number of matrix multiplication operations."""

    def __init__(
        self,
        max_count: int = 100,
        weight: float = 1.0,
        hard: bool = False,
        hard_penalty: float = -100.0,
        score_fn: str = "linear",
    ):
        super().__init__("matmul_count", weight, hard, hard_penalty)
        self.max_count = max_count
        self.score_fn = score_fn

    def __repr__(self) -> str:
        parts = [f"{self.max_count!r}"]
        if self.weight != 1.0:
            parts.append(f"weight={self.weight!r}")
        if self.hard:
            parts.append("hard=True")
        if self.hard_penalty != -100.0:
            parts.append(f"hard_penalty={self.hard_penalty!r}")
        if self.score_fn != "linear":
            parts.append(f"score_fn={self.score_fn!r}")
        return f"MatMulCount({', '.join(parts)})"

    def extract(self, m: Dict[str, Any]) -> float:
        return float(m.get("matmul_count", 0))

    def score(self, v: float) -> float:
        return _score_fn(v, self.max_count, self.score_fn)

    def is_satisfied(self, v: float) -> bool:
        return v <= self.max_count

    def to_prompt(self) -> str:
        if self.hard:
            return f"Matrix multiplications MUST stay below {self.max_count} (hard requirement)."
        return f"Minimize matrix multiplications (target: <{self.max_count}, weight: {self.weight})."

    def formula(self) -> str:
        return _formula_str(
            "matmul_count", self.max_count,
            self.score_fn, self.hard, self.hard_penalty, self.weight,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes (imported from core)
    "Constraint",
    "ConstraintResult",
    "Group",
    # Hardware-agnostic constraints (imported from core)
    "Time",
    "Error",
    "Custom",
    # Mixed-precision cost constraints
    "MemoryCost",
    "FlopsCost",
    "SpaceTimeCost",
    # Operation counter
    "MatMulCount",
]
