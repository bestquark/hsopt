"""Tests for CostProfile and DecompositionCostModel.

All expected values are hand-computed from the formulas in measure.py.
We use tiny matrices (n=2, n=4) so results are trivially verifiable.
"""

import pytest

from trovolut.backends.gpu_mp_emulator.measure import (
    CostProfile,
    DecompositionCostModel,
)
from trovolut.backends.gpu_mp_emulator.utils import Precision


# ============================================================
# 1. CostProfile dataclass
# ============================================================

class TestCostProfile:
    """Verify CostProfile derived properties and to_dict()."""

    def test_space_time_product(self):
        """space_time_product = peak_memory_bytes * total_depth."""
        cp = CostProfile(peak_memory_bytes=1000, total_depth=50)
        assert cp.space_time_product == 1000 * 50

    def test_memory_mb(self):
        """memory_mb = peak_memory_bytes / 1024^2."""
        cp = CostProfile(peak_memory_bytes=1024 * 1024)
        assert cp.memory_mb == pytest.approx(1.0)

    def test_to_dict_keys(self):
        """to_dict() must contain all keys that constraints look up."""
        cp = CostProfile(
            method="test",
            peak_memory_bytes=100,
            total_flops=200,
            total_depth=10,
            flops_by_precision={"float32": 200},
        )
        d = cp.to_dict()
        required = {
            "method", "peak_memory_bytes", "memory_bytes",
            "generator_memory_bytes", "state_memory_bytes",
            "auxiliary_memory_bytes", "memory_by_precision",
            "total_flops", "total_depth", "flops_by_precision",
            "space_time_product", "memory_mb", "n_terms", "n_steps",
        }
        assert required.issubset(d.keys())

    def test_to_dict_values(self):
        cp = CostProfile(
            peak_memory_bytes=256,
            total_flops=1000,
            total_depth=5,
        )
        d = cp.to_dict()
        assert d["peak_memory_bytes"] == 256
        assert d["total_flops"] == 1000
        assert d["space_time_product"] == 256 * 5

    def test_emulated_time_single_precision(self, a100_gpu):
        """With only fp32 flops, emulated time = flops / fp32_throughput."""
        flops = int(19.5e12)  # exactly the A100's fp32 rate
        cp = CostProfile(
            total_flops=flops,
            flops_by_precision={"float32": flops},
        )
        t = cp.emulated_time(a100_gpu)
        assert t == pytest.approx(1.0, rel=1e-6)

    def test_emulated_spacetime(self, a100_gpu):
        """emulated_spacetime = peak_memory_bytes * emulated_time."""
        flops = int(19.5e12)
        cp = CostProfile(
            peak_memory_bytes=1000,
            total_flops=flops,
            flops_by_precision={"float32": flops},
        )
        st = cp.emulated_spacetime(a100_gpu)
        assert st == pytest.approx(1000.0, rel=1e-6)


# ============================================================
# 2. DecompositionCostModel — hand-computed examples
# ============================================================

class TestDecompositionCostModel:
    """
    Test each decomposition's cost formulas with tiny dimensions.

    For n=2, fp32 (32 bits => bpe=4):
        - complex element = bpe * 2 = 8 bytes
    """

    N = 2
    BITS = 32
    BPE = 4                # max(1, 32 // 8)
    COMPLEX_ELEM = 8       # BPE * 2

    # -- exact_expm_cost --------------------------------------------------

    def test_exact_flops(self):
        """exact flops = 13 * (2 * n^3)."""
        r = DecompositionCostModel.exact_expm_cost(self.N, self.BITS)
        assert r["flops"] == 13 * (2 * self.N ** 3)  # 13 * 16 = 208

    def test_exact_auxiliary_memory(self):
        """auxiliary = 6 * n * n * bpe * 2."""
        r = DecompositionCostModel.exact_expm_cost(self.N, self.BITS)
        assert r["auxiliary_memory"] == 6 * self.N * self.N * self.COMPLEX_ELEM  # 6*4*8 = 192

    def test_exact_matmat_count(self):
        r = DecompositionCostModel.exact_expm_cost(self.N, self.BITS)
        assert r["matmat_count"] == 13
        assert r["matvec_count"] == 0

    # -- taylor_cost ------------------------------------------------------

    def test_taylor_flops(self):
        """taylor flops = order * (2 * n^2)."""
        order = 5
        r = DecompositionCostModel.taylor_cost(self.N, order, self.BITS)
        assert r["flops"] == order * (2 * self.N ** 2)  # 5 * 8 = 40

    def test_taylor_auxiliary_memory(self):
        """auxiliary = 2 * n * bpe * 2."""
        order = 5
        r = DecompositionCostModel.taylor_cost(self.N, order, self.BITS)
        assert r["auxiliary_memory"] == 2 * self.N * self.COMPLEX_ELEM  # 2*2*8 = 32

    def test_taylor_counts(self):
        order = 5
        r = DecompositionCostModel.taylor_cost(self.N, order, self.BITS)
        assert r["matvec_count"] == order
        assert r["matmat_count"] == 0

    # -- krylov_cost ------------------------------------------------------

    def test_krylov_flops(self):
        """flops = m*(2*n^2) + 13*(2*m^3)  where m = min(krylov_dim, n)."""
        krylov_dim = 10
        m = min(krylov_dim, self.N)  # m = 2
        r = DecompositionCostModel.krylov_cost(self.N, krylov_dim, self.BITS)
        expected = m * (2 * self.N ** 2) + 13 * (2 * m ** 3)  # 2*8 + 13*16 = 16+208 = 224
        assert r["flops"] == expected

    def test_krylov_auxiliary_memory(self):
        """auxiliary = n*(m+1)*bpe*2 + (m+1)*m*bpe*2."""
        krylov_dim = 10
        m = min(krylov_dim, self.N)  # m = 2
        r = DecompositionCostModel.krylov_cost(self.N, krylov_dim, self.BITS)
        expected = (
            self.N * (m + 1) * self.COMPLEX_ELEM
            + (m + 1) * m * self.COMPLEX_ELEM
        )  # 2*3*8 + 3*2*8 = 48 + 48 = 96
        assert r["auxiliary_memory"] == expected

    def test_krylov_counts(self):
        krylov_dim = 10
        m = min(krylov_dim, self.N)  # m = 2
        r = DecompositionCostModel.krylov_cost(self.N, krylov_dim, self.BITS)
        assert r["matvec_count"] == m
        assert r["matmat_count"] == 1

    # -- sparse_cost ------------------------------------------------------

    def test_sparse_flops(self):
        """sparse flops = order * (2 * nnz)."""
        nnz, order = 3, 5
        r = DecompositionCostModel.sparse_cost(self.N, nnz, order, self.BITS)
        assert r["flops"] == order * (2 * nnz)  # 5*6 = 30

    def test_sparse_auxiliary_memory(self):
        """auxiliary = 2 * n * bpe * 2  (same as taylor)."""
        nnz, order = 3, 5
        r = DecompositionCostModel.sparse_cost(self.N, nnz, order, self.BITS)
        assert r["auxiliary_memory"] == 2 * self.N * self.COMPLEX_ELEM

    def test_sparse_counts(self):
        nnz, order = 3, 5
        r = DecompositionCostModel.sparse_cost(self.N, nnz, order, self.BITS)
        assert r["matvec_count"] == order
        assert r["matmat_count"] == 0

    # -- cross-check: lower precision means less memory -------------------

    def test_fp16_uses_less_memory_than_fp32(self):
        """FP16 (2 bytes) should use less auxiliary memory than FP32 (4 bytes)."""
        r32 = DecompositionCostModel.exact_expm_cost(4, 32)
        r16 = DecompositionCostModel.exact_expm_cost(4, 16)
        assert r16["auxiliary_memory"] < r32["auxiliary_memory"]

    def test_fp16_same_flops_as_fp32(self):
        """FLOPs formula doesn't depend on precision (same operation count)."""
        r32 = DecompositionCostModel.exact_expm_cost(4, 32)
        r16 = DecompositionCostModel.exact_expm_cost(4, 16)
        assert r32["flops"] == r16["flops"]
