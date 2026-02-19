"""Tests for MemoryCost, FlopsCost, SpaceTimeCost, MatMulCount constraints.

Each test validates extract(), score(), and is_satisfied() with hand-computed
values from the _score_fn formula:
    linear:      max(0, 1 - v / max_val)
    inverse:     1 / (1 + v / max_val)
    exponential: exp(-v / max_val)
"""

import math

import pytest

from trovolut.backends.gpu_mp_emulator.constraints import (
    MemoryCost,
    FlopsCost,
    SpaceTimeCost,
    MatMulCount,
)
from trovolut.backends.gpu_mp_emulator.hardware import gpus, TFLOPS


# ============================================================
# 1. MemoryCost
# ============================================================

class TestMemoryCost:

    def test_extract(self):
        c = MemoryCost(max_memory_bytes=1000)
        assert c.extract({"peak_memory_bytes": 500}) == 500.0

    def test_extract_fallback(self):
        """Falls back to memory_bytes when peak_memory_bytes is absent."""
        c = MemoryCost(max_memory_bytes=1000)
        assert c.extract({"memory_bytes": 300}) == 300.0

    def test_score_linear_at_zero(self):
        c = MemoryCost(max_memory_bytes=1000, score_fn="linear")
        assert c.score(0) == pytest.approx(1.0)

    def test_score_linear_at_max(self):
        c = MemoryCost(max_memory_bytes=1000, score_fn="linear")
        assert c.score(1000) == pytest.approx(0.0)

    def test_score_linear_halfway(self):
        c = MemoryCost(max_memory_bytes=1000, score_fn="linear")
        assert c.score(500) == pytest.approx(0.5)

    def test_score_inverse(self):
        """inverse: 1/(1 + v/max) at v=max => 1/(1+1) = 0.5."""
        c = MemoryCost(max_memory_bytes=1000, score_fn="inverse")
        assert c.score(1000) == pytest.approx(0.5)

    def test_score_exponential(self):
        """exp(-v/max) at v=max => exp(-1)."""
        c = MemoryCost(max_memory_bytes=1000, score_fn="exponential")
        assert c.score(1000) == pytest.approx(math.exp(-1))

    def test_is_satisfied(self):
        c = MemoryCost(max_memory_bytes=1000)
        assert c.is_satisfied(999) is True
        assert c.is_satisfied(1001) is False


# ============================================================
# 2. FlopsCost
# ============================================================

class TestFlopsCost:

    def test_extract(self):
        c = FlopsCost(max_flops=1e12)
        assert c.extract({"total_flops": 5e11}) == 5e11

    def test_extract_missing(self):
        c = FlopsCost(max_flops=1e12)
        assert c.extract({}) == float("inf")

    def test_score_linear(self):
        c = FlopsCost(max_flops=1e12, score_fn="linear")
        # half of max => score = 0.5
        assert c.score(5e11) == pytest.approx(0.5)

    def test_score_over_budget(self):
        """Linear score clamps to 0 when over budget."""
        c = FlopsCost(max_flops=1e12, score_fn="linear")
        assert c.score(2e12) == pytest.approx(0.0)


# ============================================================
# 3. SpaceTimeCost
# ============================================================

class TestSpaceTimeCost:

    def test_abstract_mode_extract(self):
        """Without hardware, extract returns space_time_product from metrics."""
        c = SpaceTimeCost(max_product=1e6)
        v = c.extract({"space_time_product": 5e5})
        assert v == 5e5

    def test_abstract_mode_inf_when_missing(self):
        c = SpaceTimeCost(max_product=1e6)
        v = c.extract({})
        assert v == float("inf")

    def test_hardware_mode_extract(self):
        """With hardware, extract = peak_mem * emulated_time.

        If all flops are fp32 at exactly the A100 rate:
            emulated_time = flops / fp32_throughput = 1.0 s
            extract = peak_mem * 1.0 = peak_mem
        """
        hw = gpus.A100
        fp32_rate = int(hw.fp32_flops)
        c = SpaceTimeCost(max_product=1e15, hardware=hw)
        v = c.extract({
            "peak_memory_bytes": 1000,
            "flops_by_precision": {"float32": fp32_rate},
            "total_flops": fp32_rate,
        })
        assert v == pytest.approx(1000.0, rel=1e-6)

    def test_hardware_mode_fallback_no_flops_by_prec(self):
        """When flops_by_precision is empty, falls back to fp32."""
        hw = gpus.A100
        fp32_rate = int(hw.fp32_flops)
        c = SpaceTimeCost(max_product=1e15, hardware=hw)
        v = c.extract({
            "peak_memory_bytes": 1000,
            "total_flops": fp32_rate,
        })
        assert v == pytest.approx(1000.0, rel=1e-6)

    def test_score_linear(self):
        c = SpaceTimeCost(max_product=1e6, score_fn="linear")
        assert c.score(5e5) == pytest.approx(0.5)

    def test_is_satisfied(self):
        c = SpaceTimeCost(max_product=1e6)
        assert c.is_satisfied(1e5) is True
        assert c.is_satisfied(1e7) is False

    def test_repr_roundtrip(self):
        """repr should be eval-able (given the right imports)."""
        c = SpaceTimeCost(max_product=1e6, hardware=5 * gpus.A100, score_fn="exponential")
        r = repr(c)
        assert "SpaceTimeCost" in r
        assert "max_product" in r
        assert "gpus.A100" in r
        assert "exponential" in r

    def test_hardware_mode_sparse(self):
        """Sparse mode should yield lower emulated spacetime for fp16 workloads.

        Dense FP16 on A100: 312 TFLOPS → time = flops / 312e12
        Sparse FP16 on A100: 624 TFLOPS → time = flops / 624e12  (faster)
        """
        hw = gpus.A100
        fp16_flops = int(312e12)
        metrics = {
            "peak_memory_bytes": 1000,
            "flops_by_precision": {"float16": fp16_flops},
            "total_flops": fp16_flops,
        }
        c_dense = SpaceTimeCost(max_product=1e15, hardware=hw, sparse=False)
        c_sparse = SpaceTimeCost(max_product=1e15, hardware=hw, sparse=True)

        v_dense = c_dense.extract(metrics)
        v_sparse = c_sparse.extract(metrics)
        assert v_sparse < v_dense
        # Dense: 1000 * (312e12 / 312e12) = 1000
        # Sparse: 1000 * (312e12 / 624e12) = 500
        assert v_dense == pytest.approx(1000.0, rel=1e-6)
        assert v_sparse == pytest.approx(500.0, rel=1e-6)

    def test_repr_with_sparse(self):
        c = SpaceTimeCost(hardware=gpus.A100, sparse=True)
        r = repr(c)
        assert "sparse=True" in r


# ============================================================
# 4. MatMulCount
# ============================================================

class TestMatMulCount:

    def test_extract(self):
        c = MatMulCount(max_count=100)
        assert c.extract({"matmul_count": 42}) == 42

    def test_extract_default(self):
        c = MatMulCount(max_count=100)
        assert c.extract({}) == 0.0

    def test_score_linear(self):
        c = MatMulCount(max_count=100, score_fn="linear")
        assert c.score(50) == pytest.approx(0.5)

    def test_is_satisfied(self):
        c = MatMulCount(max_count=100)
        assert c.is_satisfied(100) is True
        assert c.is_satisfied(101) is False

    def test_repr(self):
        c = MatMulCount(50, weight=2.0, hard=True)
        r = repr(c)
        assert "MatMulCount" in r
        assert "50" in r
        assert "weight=2.0" in r
        assert "hard=True" in r
