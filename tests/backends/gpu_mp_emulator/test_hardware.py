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
