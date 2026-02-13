"""Tests for calibration infrastructure.

All tests run in emulator-only mode (no GPU required).
"""

import json
import pytest
import numpy as np

from trovolut.calibration import (
    CalibrationResult,
    CalibrationSuite,
    CorrectedCostProfile,
    standard_benchmarks,
    calibrate,
)
from trovolut.backends.gpu_mp_emulator.measure import CostProfile
from trovolut.backends.gpu_mp_emulator.hardware import gpus


# ============================================================
# 1. CalibrationResult JSON roundtrip
# ============================================================

class TestCalibrationResult:

    def test_json_roundtrip(self):
        cr = CalibrationResult(
            emulated_times=[1.0, 2.0, 3.0],
            actual_times=[1.1, 2.1, 3.1],
            emulated_memory=[100.0, 200.0],
            actual_memory=[110.0, 210.0],
            time_slope=1.05,
            time_intercept=0.05,
            memory_slope=1.1,
            memory_intercept=10.0,
            time_r2=0.99,
            memory_r2=0.98,
        )
        json_str = cr.to_json()
        cr2 = CalibrationResult.from_json(json_str)

        assert cr2.emulated_times == cr.emulated_times
        assert cr2.actual_times == cr.actual_times
        assert cr2.time_slope == pytest.approx(cr.time_slope)
        assert cr2.time_intercept == pytest.approx(cr.time_intercept)
        assert cr2.memory_slope == pytest.approx(cr.memory_slope)
        assert cr2.memory_intercept == pytest.approx(cr.memory_intercept)
        assert cr2.time_r2 == pytest.approx(cr.time_r2)
        assert cr2.memory_r2 == pytest.approx(cr.memory_r2)

    def test_defaults(self):
        cr = CalibrationResult()
        assert cr.time_slope == 1.0
        assert cr.time_intercept == 0.0
        assert cr.time_r2 == 1.0


# ============================================================
# 2. CorrectedCostProfile
# ============================================================

class TestCorrectedCostProfile:

    def test_corrected_time_applies_slope_intercept(self):
        profile = CostProfile(
            total_flops=int(19.5e12),
            flops_by_precision={"float32": int(19.5e12)},
            peak_memory_bytes=1000,
        )
        cal = CalibrationResult(time_slope=2.0, time_intercept=0.5)
        corrected = CorrectedCostProfile(profile, cal)

        hw = gpus.A100
        raw_time = profile.emulated_time(hw)
        expected = 2.0 * raw_time + 0.5
        assert corrected.corrected_time(hw) == pytest.approx(expected)

    def test_corrected_memory(self):
        profile = CostProfile(peak_memory_bytes=1000)
        cal = CalibrationResult(memory_slope=1.5, memory_intercept=100.0)
        corrected = CorrectedCostProfile(profile, cal)

        expected = 1.5 * 1000 + 100.0
        assert corrected.corrected_memory() == pytest.approx(expected)

    def test_delegates_attributes(self):
        profile = CostProfile(method="trotter", total_flops=500)
        cal = CalibrationResult()
        corrected = CorrectedCostProfile(profile, cal)

        assert corrected.method == "trotter"
        assert corrected.total_flops == 500


# ============================================================
# 3. CalibrationSuite (emulator-only mode)
# ============================================================

class TestCalibrationSuite:

    def test_emulator_only_mode(self):
        """In emulator-only mode, slope ≈ 1 and intercept ≈ 0."""
        from trovolut.backends.gpu_mp_emulator.dynamics import MPTrotterEvolution

        # Use different matrix sizes to get varying memory values
        workloads = []
        for n in [2, 4, 8, 16]:
            rng = np.random.default_rng(42 + n)
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            H = (A + A.conj().T) / 2
            dyn = MPTrotterEvolution(trotter_steps=10)
            workloads.append({"hamiltonian": H, "dynamics": dyn})

        suite = CalibrationSuite(
            workloads=workloads,
            hardware=gpus.A100,
            hardware_metrics=None,  # emulator-only
        )
        result = suite.run()

        # In emulator-only mode, actual == emulated, so slope ≈ 1, intercept ≈ 0
        assert result.time_slope == pytest.approx(1.0, abs=1e-6)
        assert result.time_intercept == pytest.approx(0.0, abs=1e-6)
        assert result.memory_slope == pytest.approx(1.0, abs=1e-6)
        assert result.memory_intercept == pytest.approx(0.0, abs=1e-6)
        assert len(result.emulated_times) == 4

    def test_empty_workloads(self):
        suite = CalibrationSuite(workloads=[], hardware=gpus.A100)
        result = suite.run()
        assert result.time_slope == 1.0  # default when no data


# ============================================================
# 4. standard_benchmarks()
# ============================================================

class TestStandardBenchmarks:

    def test_generates_expected_count(self):
        wl = standard_benchmarks()
        # 4 sizes * 3 precisions * 2 decompositions * 3 step_counts = 72
        assert len(wl) == 72

    def test_workload_structure(self):
        wl = standard_benchmarks()
        for w in wl:
            assert "hamiltonian" in w
            assert "dynamics" in w
            assert hasattr(w["dynamics"], "cost_profile")


# ============================================================
# 5. calibrate() convenience function
# ============================================================

class TestCalibrateFunction:

    def test_calibrate_emulator_only(self):
        """calibrate() runs without errors in default emulator-only mode."""
        result = calibrate()
        assert isinstance(result, CalibrationResult)
        assert len(result.emulated_times) > 0
        assert result.time_r2 == pytest.approx(1.0, abs=1e-6)

    def test_calibrate_save_path(self, tmp_path):
        path = str(tmp_path / "cal.json")
        result = calibrate(save_path=path)

        with open(path) as f:
            loaded = CalibrationResult.from_json(f.read())
        assert loaded.time_slope == pytest.approx(result.time_slope)
