"""Tests for NVML hardware counters (works without a GPU).

Verifies HardwareCounterMetrics defaults and to_dict() key prefixing,
NVMLMonitor graceful no-op, and NVML constraint extract/score.
"""

import pytest

from trovolut.backends.gpu.hardware_counters import (
    HardwareCounterMetrics,
    NVMLMonitor,
)
from trovolut.backends.gpu.constraints import (
    NVMLPeakMemory,
    NVMLPowerBudget,
    NVMLUtilization,
)


# ============================================================
# 1. HardwareCounterMetrics
# ============================================================

class TestHardwareCounterMetrics:

    def test_defaults_are_zero(self):
        m = HardwareCounterMetrics()
        assert m.peak_memory_bytes == 0
        assert m.avg_power_watts == 0.0
        assert m.n_samples == 0

    def test_to_dict_prefixes(self):
        m = HardwareCounterMetrics(
            peak_memory_bytes=1024,
            avg_power_watts=250.0,
            n_samples=10,
        )
        d = m.to_dict()
        assert "nvml_peak_memory_bytes" in d
        assert "nvml_avg_power_watts" in d
        assert "nvml_n_samples" in d
        assert d["nvml_peak_memory_bytes"] == 1024
        assert d["nvml_avg_power_watts"] == 250.0

    def test_to_dict_has_all_keys(self):
        m = HardwareCounterMetrics()
        d = m.to_dict()
        expected_keys = {
            "nvml_peak_memory_bytes", "nvml_avg_memory_bytes",
            "nvml_peak_power_watts", "nvml_avg_power_watts",
            "nvml_avg_gpu_utilization", "nvml_avg_memory_utilization",
            "nvml_energy_joules", "nvml_duration_seconds",
            "nvml_max_temperature_c", "nvml_n_samples",
        }
        assert expected_keys == set(d.keys())


# ============================================================
# 2. NVMLMonitor (graceful no-op without GPU)
# ============================================================

class TestNVMLMonitor:

    def test_context_manager_no_gpu(self):
        """NVMLMonitor works as context manager even without pynvml/GPU."""
        with NVMLMonitor(gpu_index=0, sample_interval=0.01) as mon:
            pass
        m = mon.metrics()
        assert isinstance(m, HardwareCounterMetrics)
        # Without a GPU, n_samples is 0 (or a small number if nvml is available)
        assert m.n_samples >= 0

    def test_metrics_returns_duration(self):
        """Duration should be non-negative even without samples."""
        import time
        with NVMLMonitor(sample_interval=0.01) as mon:
            time.sleep(0.05)
        m = mon.metrics()
        assert m.duration_seconds >= 0.0


# ============================================================
# 3. NVML Constraint classes
# ============================================================

class TestNVMLPeakMemory:

    def test_extract(self):
        c = NVMLPeakMemory(max_bytes=1000)
        assert c.extract({"nvml_peak_memory_bytes": 500}) == 500.0
        assert c.extract({}) == 0.0  # missing key → 0

    def test_score_under_limit(self):
        c = NVMLPeakMemory(max_bytes=1000, score_fn="linear")
        assert c.score(0) == pytest.approx(1.0)
        assert c.score(500) == pytest.approx(0.5)
        assert c.score(1000) == pytest.approx(0.0)

    def test_is_satisfied(self):
        c = NVMLPeakMemory(max_bytes=1000)
        assert c.is_satisfied(999)
        assert not c.is_satisfied(1001)


class TestNVMLPowerBudget:

    def test_extract(self):
        c = NVMLPowerBudget(max_watts=300)
        assert c.extract({"nvml_avg_power_watts": 200.0}) == 200.0
        assert c.extract({}) == 0.0

    def test_score(self):
        c = NVMLPowerBudget(max_watts=300, score_fn="linear")
        assert c.score(0) == pytest.approx(1.0)
        assert c.score(150) == pytest.approx(0.5)

    def test_is_satisfied(self):
        c = NVMLPowerBudget(max_watts=300)
        assert c.is_satisfied(299)
        assert not c.is_satisfied(301)


class TestNVMLUtilization:

    def test_extract(self):
        c = NVMLUtilization(target_utilization=80.0)
        assert c.extract({"nvml_avg_gpu_utilization": 90.0}) == 90.0
        assert c.extract({}) == 0.0

    def test_score_higher_is_better(self):
        c = NVMLUtilization(target_utilization=80.0)
        assert c.score(80.0) == pytest.approx(1.0)
        assert c.score(40.0) == pytest.approx(0.5)
        assert c.score(160.0) == pytest.approx(1.0)  # capped at 1

    def test_is_satisfied(self):
        c = NVMLUtilization(target_utilization=80.0)
        assert c.is_satisfied(80.0)
        assert not c.is_satisfied(79.0)
