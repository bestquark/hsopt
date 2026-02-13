"""End-to-end tests for MPTrotterEvolution.cost_profile().

Uses tiny 2x2 Hamiltonians so expected values can be hand-computed:

Setup:
    - H = sigma_z (2x2 diagonal)
    - Single term, FP32, decomposition="exact"
    - n_steps = 1, LieTrotter formula

Expected cost for one "exact" term on n=2, FP32:
    bpe = 4, complex_elem = 8 bytes
    generator_memory = n*n * bpe * 2  = 2*2*4*2 = 32
    auxiliary_memory  = 6*n*n * bpe*2  = 6*4*8 = 192
    state_memory      = n * 16         = 32
    peak_memory       = 32 + 32 + 192  = 256

    flops = 13 * (2 * n^3) = 13 * 16 = 208   (per step)
    total_flops = 208 * 1 = 208

    depth of exact = matvec(0) + matmat(13)*10 = 130
    total_depth = 130 * 1 = 130
"""

import numpy as np
import pytest

from trovolut.backends.gpu_mp_emulator.dynamics import (
    MPTrotterEvolution,
    MPPadeEvolution,
    MPTaylorEvolution,
)
from trovolut.backends.gpu_mp_emulator.utils import Precision, PrecisionConfig
from trovolut.core.base_dynamics import LieTrotter, SuzukiTrotter

# ── Shared arrays (module-level, zero cost) ─────────────────────

SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)


# ============================================================
# 1. Single-term, single-step Trotter
# ============================================================

class TestTrotterSingleTerm:

    N = 2
    BPE = 4                    # FP32 = 32 bits => max(1, 32//8)
    COMPLEX_ELEM = 8           # bpe * 2

    GENERATOR_MEM = N * N * COMPLEX_ELEM       # 32
    AUX_MEM = 6 * N * N * COMPLEX_ELEM         # 192
    STATE_MEM = N * 16                          # 32
    PEAK_MEM = GENERATOR_MEM + STATE_MEM + AUX_MEM  # 256

    FLOPS_PER_STEP = 13 * (2 * N ** 3)         # 208
    DEPTH_PER_STEP = 0 + 13 * 10               # matvec=0 + matmat=13*10 = 130

    @pytest.fixture(scope="class")
    def cost(self):
        """Compute cost profile once for all tests in this class."""
        evo = MPTrotterEvolution(
            trotter_steps=1,
            precision_schedule=Precision.FP32,
            formula=LieTrotter(),
        )
        return evo.cost_profile(SIGMA_Z)

    def test_peak_memory(self, cost):
        assert cost.peak_memory_bytes == self.PEAK_MEM

    def test_generator_memory(self, cost):
        assert cost.generator_memory_bytes == self.GENERATOR_MEM

    def test_auxiliary_memory(self, cost):
        assert cost.auxiliary_memory_bytes == self.AUX_MEM

    def test_state_memory(self, cost):
        assert cost.state_memory_bytes == self.STATE_MEM

    def test_total_flops(self, cost):
        assert cost.total_flops == self.FLOPS_PER_STEP

    def test_total_depth(self, cost):
        assert cost.total_depth == self.DEPTH_PER_STEP

    def test_method_label(self, cost):
        assert cost.method == "trotter"


# ============================================================
# 2. Multi-step Trotter (same term, 10 steps)
# ============================================================

class TestTrotterMultiStep:

    N = 2
    FLOPS_PER_STEP = 13 * (2 * 2 ** 3)   # 208
    DEPTH_PER_STEP = 130
    N_STEPS = 10

    @pytest.fixture(scope="class")
    def costs(self):
        """Return (cost_1step, cost_10step) computed once."""
        evo1 = MPTrotterEvolution(trotter_steps=1, precision_schedule=Precision.FP32, formula=LieTrotter())
        evo10 = MPTrotterEvolution(trotter_steps=self.N_STEPS, precision_schedule=Precision.FP32, formula=LieTrotter())
        return evo1.cost_profile(SIGMA_Z), evo10.cost_profile(SIGMA_Z)

    def test_flops_scale_with_steps(self, costs):
        _, cp10 = costs
        assert cp10.total_flops == self.FLOPS_PER_STEP * self.N_STEPS

    def test_depth_scales_with_steps(self, costs):
        _, cp10 = costs
        assert cp10.total_depth == self.DEPTH_PER_STEP * self.N_STEPS

    def test_memory_independent_of_steps(self, costs):
        """Peak memory should NOT grow with step count."""
        cp1, cp10 = costs
        assert cp1.peak_memory_bytes == cp10.peak_memory_bytes


# ============================================================
# 3. Two-term Suzuki-Trotter: H = sigma_x + sigma_z
# ============================================================

class TestTrotterTwoTerms:
    """
    Two terms (sigma_x, sigma_z), SuzukiTrotter, n_steps=1, FP32.

    SuzukiTrotter sequence for 2 terms:
        [(0, dt/2), (1, dt), (0, dt/2)]
    => 3 exponentials per step, each is "exact" (13 matmats).

    generator_memory = 2 * (2*2*4*2) = 64
    auxiliary_memory  = max over terms = 192  (same for both)
    state_memory      = 32
    peak_memory       = 64 + 32 + 192 = 288

    step_flops   = 3 * 208 = 624
    total_flops  = 624 * 1 = 624

    step_depth   = 3 * 130 = 390
    total_depth  = 390 * 1 = 390
    """

    @pytest.fixture(scope="class")
    def cost(self):
        evo = MPTrotterEvolution(
            trotter_steps=1,
            precision_schedule=Precision.FP32,
            formula=SuzukiTrotter(),
        )
        return evo.cost_profile([SIGMA_X, SIGMA_Z])

    def test_generator_memory(self, cost):
        assert cost.generator_memory_bytes == 2 * 32  # 64

    def test_peak_memory(self, cost):
        assert cost.peak_memory_bytes == 64 + 32 + 192  # 288

    def test_total_flops(self, cost):
        assert cost.total_flops == 3 * 208  # 624

    def test_total_depth(self, cost):
        assert cost.total_depth == 3 * 130  # 390


# ============================================================
# 4. Mixed-precision: FP32 + FP16 terms
# ============================================================

class TestMixedPrecision:
    """Two terms with different precisions should report separate memory."""

    @pytest.fixture(scope="class")
    def cost(self):
        schedule = {
            "term_0": PrecisionConfig(compute=Precision.FP32),
            "term_1": PrecisionConfig(compute=Precision.FP16),
        }
        evo = MPTrotterEvolution(
            trotter_steps=1,
            precision_schedule=schedule,
            formula=LieTrotter(),
        )
        return evo.cost_profile([SIGMA_X, SIGMA_Z])

    def test_flops_by_precision(self, cost):
        assert "float32" in cost.flops_by_precision
        assert "float16" in cost.flops_by_precision
        assert cost.flops_by_precision["float32"] == 208
        assert cost.flops_by_precision["float16"] == 208

    def test_memory_by_precision(self, cost):
        # FP32: 2*2*4*2 = 32 bytes;  FP16: 2*2*2*2 = 16 bytes
        assert cost.memory_by_precision.get("float32", 0) == 32
        assert cost.memory_by_precision.get("float16", 0) == 16

    def test_fp16_less_generator_memory(self):
        """FP16 term should use less generator memory than FP32."""
        evo32 = MPTrotterEvolution(trotter_steps=1, precision_schedule=Precision.FP32, formula=LieTrotter())
        evo16 = MPTrotterEvolution(trotter_steps=1, precision_schedule=Precision.FP16, formula=LieTrotter())
        cp32 = evo32.cost_profile(SIGMA_Z)
        cp16 = evo16.cost_profile(SIGMA_Z)
        assert cp16.generator_memory_bytes < cp32.generator_memory_bytes


# ============================================================
# 5. Pade and Taylor cost_profile smoke tests
# ============================================================

class TestOtherMethods:
    """Smoke tests for non-Trotter methods' cost_profile."""

    @pytest.fixture(scope="class")
    def pade_cost(self):
        evo = MPPadeEvolution(precision_schedule=Precision.FP32)
        return evo.cost_profile(SIGMA_Z)

    @pytest.fixture(scope="class")
    def taylor_cost(self):
        evo = MPTaylorEvolution(order=5, precision_schedule=Precision.FP32)
        return evo.cost_profile(SIGMA_Z)

    def test_pade_method_label(self, pade_cost):
        assert pade_cost.method == "pade"

    def test_pade_flops(self, pade_cost):
        """Pade flops = 13 * (2 * n^3) for n=2."""
        assert pade_cost.total_flops == 13 * (2 * 2 ** 3)  # 208

    def test_taylor_method_label(self, taylor_cost):
        assert taylor_cost.method == "taylor"

    def test_taylor_flops(self, taylor_cost):
        """Taylor flops = order * (2 * n^2) for n=2."""
        assert taylor_cost.total_flops == 5 * (2 * 2 ** 2)  # 40
