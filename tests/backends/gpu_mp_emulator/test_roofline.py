"""Tests for the roofline-aware cost model.

Verifies bytes_accessed formulas, roofline bottleneck selection,
backward compatibility, arithmetic_intensity, and to_dict() keys.
"""

import math
import pytest

from trovolut.backends.gpu_mp_emulator.measure import (
    CostProfile,
    DecompositionCostModel,
    SpacetimeCost,
    TermCost,
)
from trovolut.backends.gpu_mp_emulator.hardware import gpus


# ============================================================
# 1. bytes_accessed formulas (hand-computed n=2, fp32)
# ============================================================

N = 2
BITS = 32
BPE = 4            # max(1, 32 // 8)
COMPLEX_ELEM = 8   # BPE * 2


class TestBytesAccessedFormulas:
    """Verify bytes_accessed values from DecompositionCostModel."""

    def test_exact_bytes_accessed(self):
        """exact: bytes_accessed = 26 * n * n * bpe * 2."""
        r = DecompositionCostModel.exact_expm_cost(N, BITS)
        expected = 26 * N * N * COMPLEX_ELEM  # 26*4*8 = 832
        assert r["bytes_accessed"] == expected

    def test_taylor_bytes_accessed(self):
        """taylor: bytes_accessed = (n*n + order*n) * bpe * 2 * 2."""
        order = 5
        r = DecompositionCostModel.taylor_cost(N, order, BITS)
        expected = (N * N + order * N) * COMPLEX_ELEM * 2  # (4+10)*8*2 = 224
        assert r["bytes_accessed"] == expected

    def test_krylov_bytes_accessed(self):
        """krylov: bytes_accessed = (m*n*n + n*m) * bpe * 2 * 2."""
        krylov_dim = 10
        m = min(krylov_dim, N)  # m=2
        r = DecompositionCostModel.krylov_cost(N, krylov_dim, BITS)
        expected = (m * N * N + N * m) * COMPLEX_ELEM * 2  # (8+4)*8*2 = 192
        assert r["bytes_accessed"] == expected

    def test_sparse_bytes_accessed(self):
        """sparse: bytes_accessed = order * nnz * bpe * 2 * 2."""
        nnz, order = 3, 5
        r = DecompositionCostModel.sparse_cost(N, nnz, order, BITS)
        expected = order * nnz * COMPLEX_ELEM * 2  # 5*3*8*2 = 240
        assert r["bytes_accessed"] == expected


# ============================================================
# 2. TermCost bytes_accessed field
# ============================================================

class TestTermCostBytesAccessed:
    def test_default_zero(self):
        tc = TermCost(name="t0")
        assert tc.bytes_accessed == 0

    def test_explicit_value(self):
        tc = TermCost(name="t0", bytes_accessed=1234)
        assert tc.bytes_accessed == 1234


# ============================================================
# 3. Roofline bottleneck selection
# ============================================================

class TestRooflineBottleneck:
    """Test that emulated_time picks the correct bottleneck."""

    def test_compute_bound_when_bytes_tiny(self, a100_gpu):
        """Tiny bytes_accessed → compute-bound (compute_s > memory_s)."""
        flops = int(19.5e12)  # exactly 1 second of A100 fp32
        cp = CostProfile(
            total_flops=flops,
            flops_by_precision={"float32": flops},
            total_bytes_accessed=1,  # negligible
        )
        t = cp.emulated_time(a100_gpu)
        # compute_s ≈ 1.0, memory_s ≈ 1 / 2.039e12 ≈ tiny → max is compute
        assert t == pytest.approx(1.0, rel=1e-6)

    def test_memory_bound_when_flops_tiny(self, a100_gpu):
        """Tiny FLOPs → memory-bound (memory_s > compute_s)."""
        bw = 2.039e12  # A100-80GB memory bandwidth
        bytes_accessed = int(bw)  # exactly 1 second of bandwidth
        cp = CostProfile(
            total_flops=1,  # negligible
            flops_by_precision={"float32": 1},
            total_bytes_accessed=bytes_accessed,
        )
        t = cp.emulated_time(a100_gpu)
        # memory_s ≈ 1.0, compute_s ≈ tiny → max is memory
        assert t == pytest.approx(1.0, rel=1e-6)

    def test_backward_compatible_zero_bytes(self, a100_gpu):
        """Default total_bytes_accessed=0 returns pure compute time."""
        flops = int(19.5e12)
        cp = CostProfile(
            total_flops=flops,
            flops_by_precision={"float32": flops},
            # total_bytes_accessed defaults to 0
        )
        t = cp.emulated_time(a100_gpu)
        assert t == pytest.approx(1.0, rel=1e-6)


# ============================================================
# 4. SpacetimeCost roofline
# ============================================================

class TestSpacetimeCostRoofline:

    def test_spacetimecost_backward_compat(self, a100_gpu):
        """SpacetimeCost with total_bytes_accessed=0 returns pure compute."""
        sc = SpacetimeCost(n_steps=1)
        tc = TermCost(name="t0", flops=int(19.5e12), compute_bits=32)
        sc.cost_by_term["t0"] = tc
        t = sc.emulated_time(a100_gpu)
        assert t == pytest.approx(1.0, rel=1e-6)

    def test_spacetimecost_memory_bound(self, a100_gpu):
        """SpacetimeCost picks memory-bound when bytes >> flops."""
        bw = 2.039e12  # A100-80GB memory bandwidth
        sc = SpacetimeCost(
            n_steps=1,
            total_bytes_accessed=int(bw),  # 1 sec of bandwidth
        )
        tc = TermCost(name="t0", flops=1, compute_bits=32)  # tiny
        sc.cost_by_term["t0"] = tc
        t = sc.emulated_time(a100_gpu)
        assert t == pytest.approx(1.0, rel=1e-6)


# ============================================================
# 5. Arithmetic intensity
# ============================================================

class TestArithmeticIntensity:

    def test_finite_intensity(self):
        cp = CostProfile(total_flops=1000, total_bytes_accessed=100)
        assert cp.arithmetic_intensity == pytest.approx(10.0)

    def test_infinite_when_no_bytes(self):
        cp = CostProfile(total_flops=1000, total_bytes_accessed=0)
        assert math.isinf(cp.arithmetic_intensity)

    def test_spacetimecost_intensity(self):
        sc = SpacetimeCost(total_flops=500, total_bytes_accessed=50)
        assert sc.arithmetic_intensity == pytest.approx(10.0)


# ============================================================
# 6. to_dict() includes new keys
# ============================================================

class TestToDictNewKeys:

    def test_costprofile_to_dict_has_bandwidth_keys(self):
        cp = CostProfile(
            total_flops=100,
            total_bytes_accessed=50,
            bytes_by_precision={"float32": 50},
        )
        d = cp.to_dict()
        assert "total_bytes_accessed" in d
        assert "bytes_by_precision" in d
        assert "arithmetic_intensity" in d
        assert d["total_bytes_accessed"] == 50
        assert d["bytes_by_precision"] == {"float32": 50}
        assert d["arithmetic_intensity"] == pytest.approx(2.0)


# ============================================================
# 7. Sparse emulated time
# ============================================================

class TestSparseEmulatedTime:
    """Verify that sparse=True yields faster emulated time for tensor-core precisions."""

    def test_costprofile_sparse_faster_fp16(self):
        """Sparse FP16 throughput (2x) halves emulated time for FP16 workloads."""
        flops = int(624e12)  # A100 sparse FP16 rate
        cp = CostProfile(
            total_flops=flops,
            flops_by_precision={"float16": flops},
        )
        t_dense = cp.emulated_time(gpus.A100, sparse=False)
        t_sparse = cp.emulated_time(gpus.A100, sparse=True)
        assert t_sparse < t_dense
        # Dense: 624e12 / 312e12 = 2.0s; Sparse: 624e12 / 624e12 = 1.0s
        assert t_dense == pytest.approx(2.0, rel=1e-6)
        assert t_sparse == pytest.approx(1.0, rel=1e-6)

    def test_costprofile_sparse_no_effect_fp32(self):
        """FP32 is a CUDA-core precision — sparse flag has no effect."""
        flops = int(19.5e12)
        cp = CostProfile(
            total_flops=flops,
            flops_by_precision={"float32": flops},
        )
        t_dense = cp.emulated_time(gpus.A100, sparse=False)
        t_sparse = cp.emulated_time(gpus.A100, sparse=True)
        assert t_dense == pytest.approx(t_sparse, rel=1e-9)

    def test_spacetimecost_sparse_faster_16bit(self):
        """SpacetimeCost with 16-bit terms should be faster with sparse=True."""
        sc = SpacetimeCost(n_steps=1)
        tc = TermCost(name="t0", flops=int(624e12), compute_bits=16)
        sc.cost_by_term["t0"] = tc
        t_dense = sc.emulated_time(gpus.A100, sparse=False)
        t_sparse = sc.emulated_time(gpus.A100, sparse=True)
        assert t_sparse < t_dense

    def test_spacetimecost_sparse_no_effect_32bit(self):
        """SpacetimeCost with 32-bit terms — sparse has no effect."""
        sc = SpacetimeCost(n_steps=1)
        tc = TermCost(name="t0", flops=int(19.5e12), compute_bits=32)
        sc.cost_by_term["t0"] = tc
        t_dense = sc.emulated_time(gpus.A100, sparse=False)
        t_sparse = sc.emulated_time(gpus.A100, sparse=True)
        assert t_dense == pytest.approx(t_sparse, rel=1e-9)
