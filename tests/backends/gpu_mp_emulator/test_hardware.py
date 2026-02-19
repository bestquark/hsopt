"""Tests for GPU, GPUCluster, and predefined hardware specs.

Ensures that GPU specs are sane, cluster arithmetic works, and
emulated time calculations are correct.
"""

import pytest

from trovolut.backends.gpu_mp_emulator.hardware import (
    GPU,
    GPUCluster,
    gpus,
    NVLINK_3,
    KB,
    MB,
    GB,
    TFLOPS,
)


# ============================================================
# 1. Individual GPU specs
# ============================================================

class TestGPUSpecs:
    """Verify key specs for predefined GPUs."""

    def test_a100_memory(self):
        assert gpus.A100.memory_bytes == 80 * GB

    def test_a100_fp32_flops(self):
        assert gpus.A100.fp32_flops == 19.5 * TFLOPS

    def test_a100_fp16_flops(self):
        assert gpus.A100.fp16_flops == 312 * TFLOPS

    def test_h100_memory(self):
        assert gpus.H100.memory_bytes == 80 * GB

    def test_h100_fp64_flops(self):
        assert gpus.H100.fp64_flops == 33.5 * TFLOPS

    def test_v100_no_bf16(self):
        """V100 should have 0 BF16 throughput (pre-dates BF16 support)."""
        assert gpus.V100.bf16_flops == 0.0

    def test_b200_fp4_support(self):
        """Blackwell should have non-zero FP4 FLOPS."""
        assert gpus.B200.fp4_flops > 0


# ============================================================
# 2. GPU.__repr__ — eval-able reference
# ============================================================

class TestGPURepr:

    def test_predefined_gpu_repr(self):
        assert repr(gpus.A100) == "gpus.A100"

    def test_custom_gpu_repr(self):
        g = GPU("MyGPU", specs={"memory_bytes": 1 * GB})
        assert repr(g) == "GPU('MyGPU')"


# ============================================================
# 3. GPUCluster arithmetic
# ============================================================

class TestGPUCluster:

    def test_scalar_multiply(self):
        cluster = 5 * gpus.A100
        assert isinstance(cluster, GPUCluster)
        assert cluster.total_units == 5

    def test_rmul_equals_mul(self):
        c1 = gpus.A100 * 3
        c2 = 3 * gpus.A100
        assert c1.total_units == c2.total_units == 3

    def test_total_memory(self):
        cluster = 4 * gpus.A100
        assert cluster.total_memory_bytes == 4 * 80 * GB

    def test_total_flops(self):
        cluster = 2 * gpus.A100
        assert cluster.total_flops("fp32") == 2 * 19.5 * TFLOPS

    def test_cluster_addition(self):
        c = 2 * gpus.A100 + 3 * gpus.H100
        assert c.total_units == 5

    def test_intra_node_interconnect(self):
        """5 * gpus.A100 should auto-select NVLink 3."""
        cluster = 5 * gpus.A100
        assert cluster.intra_node is not None
        assert cluster.intra_node.name == "NVLink 3 (Ampere)"

    def test_cluster_repr_single(self):
        cluster = 5 * gpus.A100
        assert repr(cluster) == "5 * gpus.A100"

    def test_cluster_repr_composite(self):
        cluster = 2 * gpus.A100 + 3 * gpus.H100
        assert repr(cluster) == "(2 * gpus.A100 + 3 * gpus.H100)"


# ============================================================
# 4. Emulated time on hardware
# ============================================================

class TestEmulatedTime:

    def test_known_throughput(self):
        """If flops == hardware throughput, emulated time should be ~1 second."""
        from trovolut.backends.gpu_mp_emulator.measure import CostProfile

        fp32_rate = gpus.A100.fp32_flops
        cp = CostProfile(
            total_flops=int(fp32_rate),
            flops_by_precision={"float32": int(fp32_rate)},
        )
        t = cp.emulated_time(gpus.A100)
        assert t == pytest.approx(1.0, rel=1e-6)

    def test_cluster_faster_than_single(self):
        """More GPUs → lower emulated time (for same workload)."""
        from trovolut.backends.gpu_mp_emulator.measure import CostProfile

        flops = int(100e12)
        cp = CostProfile(
            total_flops=flops,
            flops_by_precision={"float32": flops},
        )
        t_single = cp.emulated_time(gpus.A100)
        t_cluster = cp.emulated_time(5 * gpus.A100)
        assert t_cluster < t_single
        assert t_cluster == pytest.approx(t_single / 5, rel=1e-6)

    def test_fp16_faster_than_fp32(self):
        """FP16 has higher throughput on A100, so same FLOP count runs faster."""
        from trovolut.backends.gpu_mp_emulator.measure import CostProfile

        flops = int(100e12)
        cp_fp32 = CostProfile(total_flops=flops, flops_by_precision={"float32": flops})
        cp_fp16 = CostProfile(total_flops=flops, flops_by_precision={"float16": flops})
        t_fp32 = cp_fp32.emulated_time(gpus.A100)
        t_fp16 = cp_fp16.emulated_time(gpus.A100)
        assert t_fp16 < t_fp32


# ============================================================
# 5. Normalized dense FLOPS
# ============================================================

class TestGPUSpecsNormalized:
    """Verify that H100/H200/B100/B200 use dense (non-sparsity) values."""

    def test_h100_fp16_dense(self):
        assert gpus.H100.fp16_flops == 990 * TFLOPS

    def test_h100_fp8_dense(self):
        assert gpus.H100.fp8_flops == 1979 * TFLOPS

    def test_b200_fp16_dense(self):
        assert gpus.B200.fp16_flops == 2250 * TFLOPS

    def test_b200_fp8_dense(self):
        assert gpus.B200.fp8_flops == 4500 * TFLOPS

    def test_b200_fp4_dense(self):
        assert gpus.B200.fp4_flops == 9000 * TFLOPS

    def test_a100_80gb_bandwidth(self):
        assert gpus.A100.memory_bandwidth == 2.039e12

    def test_a100_40gb_bandwidth(self):
        assert gpus.A100_40.memory_bandwidth == 1.555e12


# ============================================================
# 6. Sparse FLOPS via max_flops()
# ============================================================

class TestSparseFlops:
    """Verify max_flops(sparse=True) returns 2x dense for supported GPUs."""

    def test_h100_fp16_sparse(self):
        assert gpus.H100.max_flops("fp16", sparse=True) == 1979 * TFLOPS

    def test_a100_fp16_sparse(self):
        assert gpus.A100.max_flops("fp16", sparse=True) == 624 * TFLOPS

    def test_v100_fp16_sparse_fallback(self):
        """V100 has no sparse support — should fall back to dense."""
        assert gpus.V100.max_flops("fp16", sparse=True) == 31.4 * TFLOPS

    def test_b200_fp4_sparse(self):
        assert gpus.B200.max_flops("fp4", sparse=True) == 18_000 * TFLOPS

    def test_cluster_sparse_flops(self):
        cluster = 2 * gpus.H100
        assert cluster.total_flops("fp16", sparse=True) == 2 * 1979 * TFLOPS

    def test_dense_equals_no_sparse_flag(self):
        """max_flops without sparse flag returns dense value."""
        assert gpus.H100.max_flops("fp16") == 990 * TFLOPS
        assert gpus.H100.max_flops("fp16", sparse=False) == 990 * TFLOPS


# ============================================================
# 7. Architectural specs
# ============================================================

class TestArchitecturalSpecs:

    def test_a100_cuda_cores(self):
        assert gpus.A100.cuda_cores == 6912

    def test_a100_tensor_cores(self):
        assert gpus.A100.tensor_cores == 432

    def test_a100_sm_count(self):
        assert gpus.A100.sm_count == 108

    def test_h100_cuda_cores(self):
        assert gpus.H100.cuda_cores == 16896

    def test_v100_tdp(self):
        assert gpus.V100.tdp_watts == 300

    def test_b200_tdp(self):
        assert gpus.B200.tdp_watts == 1000

    def test_h100_l2_cache(self):
        assert gpus.H100.l2_cache_bytes == 50 * MB

    def test_a100_shared_mem_per_sm(self):
        assert gpus.A100.shared_mem_per_sm == 164 * KB


# ============================================================
# 8. RTX 5090
# ============================================================

class TestRTX5090:

    def test_memory(self):
        assert gpus.RTX_5090.memory_bytes == 32 * GB

    def test_fp32_flops(self):
        assert gpus.RTX_5090.fp32_flops == 105 * TFLOPS

    def test_fp16_dense(self):
        assert gpus.RTX_5090.fp16_flops == 210 * TFLOPS

    def test_fp16_sparse(self):
        assert gpus.RTX_5090.max_flops("fp16", sparse=True) == 420 * TFLOPS

    def test_interconnect(self):
        cluster = 2 * gpus.RTX_5090
        assert cluster.intra_node is not None
        assert cluster.intra_node.name == "PCIe Gen5"

    def test_cuda_cores(self):
        assert gpus.RTX_5090.cuda_cores == 21760

    def test_tensor_cores(self):
        assert gpus.RTX_5090.tensor_cores == 680

    def test_tdp(self):
        assert gpus.RTX_5090.tdp_watts == 575
