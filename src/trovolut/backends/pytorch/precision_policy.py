"""Adaptive precision policy evolved by OpenEvolve.

Maps (operation, bond_index, time_step, hamiltonian_info) to precision settings.
This is the key file connecting the TEBD engine to OpenEvolve's evolutionary search.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class OperationPrecision:
    """Precision configuration for a single GEMM operation."""
    gemm_input_dtype: str = "fp8"   # "fp4", "fp8", "fp16", "fp32"
    gemm_accum_dtype: str = "fp32"  # "fp16", "fp32"
    storage_dtype: str = "fp8"      # "fp4", "fp8", "fp16", "fp32"


@dataclass
class TruncationConfig:
    """Configuration for truncation at a specific bond."""
    method: str = "svd"              # "svd", "randomized_svd", "variational", "qr"
    compute_dtype: str = "fp32"      # "fp16", "fp32"
    n_iter: int = 8                  # iterations for variational (1-20)
    oversampling: int = 10           # for randomized SVD (5-50)
    n_power_iter: int = 2            # for randomized SVD (0-4)


class PrecisionPolicy:
    """Maps operations to precision settings. Evolved by OpenEvolve.

    The policy has several "genes" that OpenEvolve mutates:
    1. Default precision for each operation type
    2. Per-bond overrides (some bonds need higher precision)
    3. Time-dependent rules (early steps might tolerate lower precision)
    4. Hamiltonian-aware rules (near critical points, increase precision)
    5. Truncation method selection per bond
    6. Canonicalization frequency
    7. Site-dependent precision (boundary vs bulk vs critical bonds)
    8. Late-stage precision upgrade (entanglement grows over time)
    9. Periodic renormalization to prevent norm drift
    """

    def __init__(self):
        # === EVOLVABLE PARAMETERS ===
        self.contraction_precision = OperationPrecision("fp8", "fp32", "fp8")
        self.gate_apply_precision = OperationPrecision("fp8", "fp32", "fp8")
        self.truncation_config = TruncationConfig("variational", "fp32", 8, 10, 2)
        self.storage_precision = "fp8"
        self.canonicalize_every: int = 100
        self.chi_max: int = 256
        self.bond_overrides: Dict[int, OperationPrecision] = {}
        self.bond_truncation_overrides: Dict[int, TruncationConfig] = {}
        self.high_precision_warmup_steps: int = 50
        self.precision_schedule: List[Tuple[int, str]] = []
        self.upgrade_precision_if_truncation_error_above: float = 1e-4
        self.downgrade_precision_if_truncation_error_below: float = 1e-8
        self.boundary_bond_count: int = 2
        self.boundary_contraction_dtype: str = "fp32"
        self.boundary_truncation_method: str = "svd"
        self.critical_bond_contraction_dtype: str = "fp16"
        self.critical_bond_truncation_method: str = "svd"
        self.late_stage_step_fraction: float = 0.5
        self.late_stage_contraction_dtype: str = "fp16"
        self.late_stage_storage_dtype: str = "fp16"
        self.renormalize_every: int = 0

    def _is_boundary_bond(self, bond_idx: int, hamiltonian_info: dict) -> bool:
        if self.boundary_bond_count <= 0:
            return False
        n_sites = hamiltonian_info.get("n_sites", 0)
        n_bonds = n_sites - 1 if n_sites > 0 else 0
        if n_bonds == 0:
            return False
        return (bond_idx < self.boundary_bond_count
                or bond_idx >= n_bonds - self.boundary_bond_count)

    def _is_critical_bond(self, bond_idx: int, hamiltonian_info: dict) -> bool:
        if not hamiltonian_info:
            return False
        strong_bonds = hamiltonian_info.get("strong_bonds", [])
        if bond_idx in strong_bonds:
            return True
        if hamiltonian_info.get("is_critical", False):
            return True
        return False

    def _is_late_stage(self, step: int, hamiltonian_info: dict) -> bool:
        if self.late_stage_step_fraction <= 0 or self.late_stage_step_fraction >= 1:
            return False
        n_steps = hamiltonian_info.get("n_steps", 0)
        if n_steps <= 0:
            return False
        return step > n_steps * self.late_stage_step_fraction

    def get_contraction_precision(self, bond_idx: int, step: int,
                                   hamiltonian_info: dict) -> OperationPrecision:
        if step < self.high_precision_warmup_steps:
            return OperationPrecision("fp32", "fp32", "fp32")
        if bond_idx in self.bond_overrides:
            return self.bond_overrides[bond_idx]
        if self._is_boundary_bond(bond_idx, hamiltonian_info):
            return OperationPrecision(
                self.boundary_contraction_dtype, "fp32",
                self.boundary_contraction_dtype)
        if self._is_critical_bond(bond_idx, hamiltonian_info):
            return OperationPrecision(
                self.critical_bond_contraction_dtype, "fp32",
                self.critical_bond_contraction_dtype)
        if self._is_late_stage(step, hamiltonian_info):
            return OperationPrecision(
                self.late_stage_contraction_dtype, "fp32",
                self.late_stage_contraction_dtype)
        current_precision = None
        for sched_step, dtype in self.precision_schedule:
            if step >= sched_step:
                current_precision = dtype
        if current_precision is not None:
            return OperationPrecision(current_precision, "fp32", current_precision)
        return self.contraction_precision

    def get_gate_apply_precision(self, bond_idx: int, step: int,
                                  hamiltonian_info: dict) -> OperationPrecision:
        if step < self.high_precision_warmup_steps:
            return OperationPrecision("fp32", "fp32", "fp32")
        if bond_idx in self.bond_overrides:
            return self.bond_overrides[bond_idx]
        if self._is_boundary_bond(bond_idx, hamiltonian_info):
            return OperationPrecision(
                self.boundary_contraction_dtype, "fp32",
                self.boundary_contraction_dtype)
        if self._is_critical_bond(bond_idx, hamiltonian_info):
            return OperationPrecision(
                self.critical_bond_contraction_dtype, "fp32",
                self.critical_bond_contraction_dtype)
        return self.gate_apply_precision

    def get_truncation_config(self, bond_idx: int, step: int,
                               hamiltonian_info: dict) -> TruncationConfig:
        if step < self.high_precision_warmup_steps:
            return TruncationConfig("svd", "fp32", 0, 10, 2)
        if bond_idx in self.bond_truncation_overrides:
            return self.bond_truncation_overrides[bond_idx]
        if self._is_boundary_bond(bond_idx, hamiltonian_info):
            return TruncationConfig(
                self.boundary_truncation_method,
                self.truncation_config.compute_dtype,
                self.truncation_config.n_iter,
                self.truncation_config.oversampling,
                self.truncation_config.n_power_iter)
        if self._is_critical_bond(bond_idx, hamiltonian_info):
            return TruncationConfig(
                self.critical_bond_truncation_method,
                self.truncation_config.compute_dtype,
                self.truncation_config.n_iter,
                self.truncation_config.oversampling,
                self.truncation_config.n_power_iter)
        return self.truncation_config

    def get_storage_dtype(self, bond_idx: int, step: int) -> str:
        if step < self.high_precision_warmup_steps:
            return "fp32"
        return self.storage_precision

    def should_canonicalize(self, step: int) -> bool:
        if self.canonicalize_every <= 0:
            return False
        return step > 0 and step % self.canonicalize_every == 0

    def should_renormalize(self, step: int) -> bool:
        if self.renormalize_every <= 0:
            return False
        return step > 0 and step % self.renormalize_every == 0

    def should_upgrade_precision(self, truncation_error: float) -> bool:
        return truncation_error > self.upgrade_precision_if_truncation_error_above

    def should_downgrade_precision(self, truncation_error: float) -> bool:
        return truncation_error < self.downgrade_precision_if_truncation_error_below

    def to_max_precision(self) -> "PrecisionPolicy":
        """Clone this policy but override ALL precisions to FP64."""
        p = copy.deepcopy(self)
        p.storage_precision = "fp64"
        p.contraction_precision = OperationPrecision("fp64", "fp64", "fp64")
        p.gate_apply_precision = OperationPrecision("fp64", "fp64", "fp64")
        p.truncation_config.compute_dtype = "fp64"
        p.boundary_contraction_dtype = "fp64"
        p.critical_bond_contraction_dtype = "fp64"
        p.late_stage_contraction_dtype = "fp64"
        p.late_stage_storage_dtype = "fp64"
        p.high_precision_warmup_steps = 0
        p.bond_overrides = {k: OperationPrecision("fp64", "fp64", "fp64")
                            for k in self.bond_overrides}
        p.bond_truncation_overrides = {
            k: TruncationConfig(v.method, "fp64", v.n_iter,
                                v.oversampling, v.n_power_iter)
            for k, v in self.bond_truncation_overrides.items()
        }
        return p

    to_fp64 = to_max_precision

    @classmethod
    def make_max_precision_baseline(cls) -> "PrecisionPolicy":
        """Create a minimal max-precision baseline."""
        p = cls()
        p.storage_precision = "fp64"
        p.contraction_precision = OperationPrecision("fp64", "fp64", "fp64")
        p.gate_apply_precision = OperationPrecision("fp64", "fp64", "fp64")
        p.truncation_config = TruncationConfig("svd", "fp64")
        p.high_precision_warmup_steps = 0
        p.boundary_bond_count = 0
        p.boundary_contraction_dtype = "fp64"
        p.boundary_truncation_method = "svd"
        p.critical_bond_contraction_dtype = "fp64"
        p.critical_bond_truncation_method = "svd"
        p.late_stage_step_fraction = 0.0
        p.late_stage_contraction_dtype = "fp64"
        p.late_stage_storage_dtype = "fp64"
        p.canonicalize_every = 0
        p.renormalize_every = 0
        p.chi_max = 256
        return p

    make_fp64_baseline = make_max_precision_baseline

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "contraction_precision": asdict(self.contraction_precision),
            "gate_apply_precision": asdict(self.gate_apply_precision),
            "truncation_config": asdict(self.truncation_config),
            "storage_precision": self.storage_precision,
            "canonicalize_every": self.canonicalize_every,
            "chi_max": self.chi_max,
            "bond_overrides": {
                str(k): asdict(v) for k, v in self.bond_overrides.items()
            },
            "bond_truncation_overrides": {
                str(k): asdict(v) for k, v in self.bond_truncation_overrides.items()
            },
            "high_precision_warmup_steps": self.high_precision_warmup_steps,
            "precision_schedule": self.precision_schedule,
            "upgrade_precision_if_truncation_error_above":
                self.upgrade_precision_if_truncation_error_above,
            "downgrade_precision_if_truncation_error_below":
                self.downgrade_precision_if_truncation_error_below,
            "boundary_bond_count": self.boundary_bond_count,
            "boundary_contraction_dtype": self.boundary_contraction_dtype,
            "boundary_truncation_method": self.boundary_truncation_method,
            "critical_bond_contraction_dtype": self.critical_bond_contraction_dtype,
            "critical_bond_truncation_method": self.critical_bond_truncation_method,
            "late_stage_step_fraction": self.late_stage_step_fraction,
            "late_stage_contraction_dtype": self.late_stage_contraction_dtype,
            "late_stage_storage_dtype": self.late_stage_storage_dtype,
            "renormalize_every": self.renormalize_every,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PrecisionPolicy":
        """Deserialize from checkpoint."""
        policy = cls()
        policy.contraction_precision = OperationPrecision(**d["contraction_precision"])
        policy.gate_apply_precision = OperationPrecision(**d["gate_apply_precision"])
        policy.truncation_config = TruncationConfig(**d["truncation_config"])
        policy.storage_precision = d["storage_precision"]
        policy.canonicalize_every = d["canonicalize_every"]
        policy.chi_max = d.get("chi_max", 256)
        policy.bond_overrides = {
            int(k): OperationPrecision(**v)
            for k, v in d.get("bond_overrides", {}).items()
        }
        policy.bond_truncation_overrides = {
            int(k): TruncationConfig(**v)
            for k, v in d.get("bond_truncation_overrides", {}).items()
        }
        policy.high_precision_warmup_steps = d.get("high_precision_warmup_steps", 50)
        policy.precision_schedule = d.get("precision_schedule", [])
        policy.upgrade_precision_if_truncation_error_above = d.get(
            "upgrade_precision_if_truncation_error_above", 1e-4)
        policy.downgrade_precision_if_truncation_error_below = d.get(
            "downgrade_precision_if_truncation_error_below", 1e-8)
        policy.boundary_bond_count = d.get("boundary_bond_count", 2)
        policy.boundary_contraction_dtype = d.get("boundary_contraction_dtype", "fp32")
        policy.boundary_truncation_method = d.get("boundary_truncation_method", "svd")
        policy.critical_bond_contraction_dtype = d.get("critical_bond_contraction_dtype", "fp16")
        policy.critical_bond_truncation_method = d.get("critical_bond_truncation_method", "svd")
        policy.late_stage_step_fraction = d.get("late_stage_step_fraction", 0.5)
        policy.late_stage_contraction_dtype = d.get("late_stage_contraction_dtype", "fp16")
        policy.late_stage_storage_dtype = d.get("late_stage_storage_dtype", "fp16")
        policy.renormalize_every = d.get("renormalize_every", 0)
        return policy

    def to_code_string(self) -> str:
        """Serialize as a Python-like code string for OpenEvolve's LLM."""
        bond_overrides_str = {k: v.gemm_input_dtype for k, v in self.bond_overrides.items()}
        trunc_overrides_str = {k: v.method for k, v in self.bond_truncation_overrides.items()}
        return f"""\
# === TEBD Precision Policy ===
CONTRACTION_GEMM_INPUT = "{self.contraction_precision.gemm_input_dtype}"
CONTRACTION_GEMM_ACCUM = "{self.contraction_precision.gemm_accum_dtype}"
GATE_APPLY_GEMM_INPUT = "{self.gate_apply_precision.gemm_input_dtype}"
GATE_APPLY_GEMM_ACCUM = "{self.gate_apply_precision.gemm_accum_dtype}"
STORAGE_DTYPE = "{self.storage_precision}"
TRUNCATION_METHOD = "{self.truncation_config.method}"
TRUNCATION_COMPUTE_DTYPE = "{self.truncation_config.compute_dtype}"
VARIATIONAL_ITERATIONS = {self.truncation_config.n_iter}
RSVD_OVERSAMPLING = {self.truncation_config.oversampling}
RSVD_POWER_ITERATIONS = {self.truncation_config.n_power_iter}
CANONICALIZE_EVERY = {self.canonicalize_every}
CHI_MAX = {self.chi_max}
HIGH_PRECISION_WARMUP_STEPS = {self.high_precision_warmup_steps}
UPGRADE_PRECISION_THRESHOLD = {self.upgrade_precision_if_truncation_error_above}
DOWNGRADE_PRECISION_THRESHOLD = {self.downgrade_precision_if_truncation_error_below}
BOUNDARY_BOND_COUNT = {self.boundary_bond_count}
BOUNDARY_CONTRACTION_DTYPE = "{self.boundary_contraction_dtype}"
BOUNDARY_TRUNCATION_METHOD = "{self.boundary_truncation_method}"
CRITICAL_BOND_CONTRACTION_DTYPE = "{self.critical_bond_contraction_dtype}"
CRITICAL_BOND_TRUNCATION_METHOD = "{self.critical_bond_truncation_method}"
LATE_STAGE_STEP_FRACTION = {self.late_stage_step_fraction}
LATE_STAGE_CONTRACTION_DTYPE = "{self.late_stage_contraction_dtype}"
LATE_STAGE_STORAGE_DTYPE = "{self.late_stage_storage_dtype}"
RENORMALIZE_EVERY = {self.renormalize_every}
BOND_OVERRIDES = {bond_overrides_str}
BOND_TRUNCATION_OVERRIDES = {trunc_overrides_str}
"""

    @classmethod
    def from_code_string(cls, code: str) -> "PrecisionPolicy":
        """Parse a code string back into a PrecisionPolicy object."""
        policy = cls()

        def _extract(pattern: str, default: str = "") -> str:
            m = re.search(pattern, code)
            return m.group(1) if m else default

        def _extract_num(pattern: str, default: float = 0) -> float:
            m = re.search(pattern, code)
            return float(m.group(1)) if m else default

        policy.contraction_precision.gemm_input_dtype = _extract(
            r'CONTRACTION_GEMM_INPUT\s*=\s*"(\w+)"', "fp8")
        policy.contraction_precision.gemm_accum_dtype = _extract(
            r'CONTRACTION_GEMM_ACCUM\s*=\s*"(\w+)"', "fp32")
        policy.gate_apply_precision.gemm_input_dtype = _extract(
            r'GATE_APPLY_GEMM_INPUT\s*=\s*"(\w+)"', "fp8")
        policy.gate_apply_precision.gemm_accum_dtype = _extract(
            r'GATE_APPLY_GEMM_ACCUM\s*=\s*"(\w+)"', "fp32")
        policy.storage_precision = _extract(
            r'STORAGE_DTYPE\s*=\s*"(\w+)"', "fp8")
        policy.truncation_config.method = _extract(
            r'TRUNCATION_METHOD\s*=\s*"(\w+)"', "svd")
        policy.truncation_config.compute_dtype = _extract(
            r'TRUNCATION_COMPUTE_DTYPE\s*=\s*"(\w+)"', "fp32")
        policy.truncation_config.n_iter = int(_extract_num(
            r'VARIATIONAL_ITERATIONS\s*=\s*(\d+)', 8))
        policy.truncation_config.oversampling = int(_extract_num(
            r'RSVD_OVERSAMPLING\s*=\s*(\d+)', 10))
        policy.truncation_config.n_power_iter = int(_extract_num(
            r'RSVD_POWER_ITERATIONS\s*=\s*(\d+)', 2))
        policy.canonicalize_every = int(_extract_num(
            r'CANONICALIZE_EVERY\s*=\s*(\d+)', 100))
        policy.chi_max = int(_extract_num(
            r'CHI_MAX\s*=\s*(\d+)', 256))
        policy.high_precision_warmup_steps = int(_extract_num(
            r'HIGH_PRECISION_WARMUP_STEPS\s*=\s*(\d+)', 50))
        policy.upgrade_precision_if_truncation_error_above = _extract_num(
            r'UPGRADE_PRECISION_THRESHOLD\s*=\s*([0-9eE.\-]+)', 1e-4)
        policy.downgrade_precision_if_truncation_error_below = _extract_num(
            r'DOWNGRADE_PRECISION_THRESHOLD\s*=\s*([0-9eE.\-]+)', 1e-8)
        policy.boundary_bond_count = int(_extract_num(
            r'BOUNDARY_BOND_COUNT\s*=\s*(\d+)', 2))
        policy.boundary_contraction_dtype = _extract(
            r'BOUNDARY_CONTRACTION_DTYPE\s*=\s*"(\w+)"', "fp32")
        policy.boundary_truncation_method = _extract(
            r'BOUNDARY_TRUNCATION_METHOD\s*=\s*"(\w+)"', "svd")
        policy.critical_bond_contraction_dtype = _extract(
            r'CRITICAL_BOND_CONTRACTION_DTYPE\s*=\s*"(\w+)"', "fp16")
        policy.critical_bond_truncation_method = _extract(
            r'CRITICAL_BOND_TRUNCATION_METHOD\s*=\s*"(\w+)"', "svd")
        policy.late_stage_step_fraction = _extract_num(
            r'LATE_STAGE_STEP_FRACTION\s*=\s*([0-9.]+)', 0.5)
        policy.late_stage_contraction_dtype = _extract(
            r'LATE_STAGE_CONTRACTION_DTYPE\s*=\s*"(\w+)"', "fp16")
        policy.late_stage_storage_dtype = _extract(
            r'LATE_STAGE_STORAGE_DTYPE\s*=\s*"(\w+)"', "fp16")
        policy.renormalize_every = int(_extract_num(
            r'RENORMALIZE_EVERY\s*=\s*(\d+)', 0))

        bo_match = re.search(r'BOND_OVERRIDES\s*=\s*(\{[^}]*\})', code)
        if bo_match:
            try:
                bo = eval(bo_match.group(1))  # noqa: S307
                policy.bond_overrides = {
                    int(k): OperationPrecision(v, "fp32", v)
                    for k, v in bo.items()
                }
            except Exception:
                pass

        bto_match = re.search(r'BOND_TRUNCATION_OVERRIDES\s*=\s*(\{[^}]*\})', code)
        if bto_match:
            try:
                bto = eval(bto_match.group(1))  # noqa: S307
                policy.bond_truncation_overrides = {
                    int(k): TruncationConfig(method=v)
                    for k, v in bto.items()
                }
            except Exception:
                pass

        return policy


__all__ = [
    "OperationPrecision",
    "TruncationConfig",
    "PrecisionPolicy",
]
