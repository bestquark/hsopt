"""Gate application with precision-policy-aware dispatch.

Provides both local (single-GPU) and distributed (multi-GPU per bond)
gate applicators with a common interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch

from trovolut.backends.pytorch.tc_ops import TensorCoreGEMM, _resolve_torch_dtype
from trovolut.backends.pytorch.precision_policy import PrecisionPolicy, TruncationConfig
from trovolut.backends.pytorch.truncation import (
    qr_truncation,
    randomized_svd_truncation,
    svd_truncation,
    variational_truncation,
)


class GateApplicator(ABC):
    """Abstract interface for gate application + truncation."""

    @abstractmethod
    def apply(self, A: torch.Tensor, B: torch.Tensor, gate: torch.Tensor,
              chi_max: int, bond_idx: int, step: int,
              hamiltonian_info: dict) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply a two-site gate and truncate."""


def _dispatch_truncation(theta: torch.Tensor, chi_max: int,
                          config: TruncationConfig,
                          tc_gemm: TensorCoreGEMM
                          ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Dispatch to the appropriate truncation method based on config."""
    compute_dtype = _resolve_torch_dtype(config.compute_dtype)

    if config.method == "svd":
        return svd_truncation(theta, chi_max, compute_dtype)
    elif config.method == "randomized_svd":
        return randomized_svd_truncation(
            theta, chi_max, tc_gemm, compute_dtype,
            oversampling=config.oversampling,
            n_power_iter=config.n_power_iter,
        )
    elif config.method == "variational":
        return variational_truncation(
            theta, chi_max, tc_gemm,
            n_iter=config.n_iter,
            compute_dtype=compute_dtype,
        )
    elif config.method == "qr":
        A, B, _ = qr_truncation(theta, chi_max)
        return A, B, 0.0
    else:
        raise ValueError(f"Unknown truncation method: {config.method}")


class LocalGateApplicator(GateApplicator):
    """Single-GPU gate application with precision-policy-aware dispatch."""

    def __init__(self, tc_gemm: TensorCoreGEMM, precision_policy: PrecisionPolicy):
        self.tc_gemm = tc_gemm
        self.policy = precision_policy

    def apply(self, A: torch.Tensor, B: torch.Tensor, gate: torch.Tensor,
              chi_max: int, bond_idx: int, step: int,
              hamiltonian_info: dict) -> Tuple[torch.Tensor, torch.Tensor, float]:
        d = A.shape[1]
        chi_l = A.shape[0]
        chi_r = B.shape[2]

        A_mat = A.reshape(chi_l * d, -1)
        B_mat = B.reshape(B.shape[0], -1)

        contraction_prec = self.policy.get_contraction_precision(
            bond_idx, step, hamiltonian_info)

        theta = self.tc_gemm.gemm(A_mat, B_mat,
                                   precision_override=contraction_prec.gemm_input_dtype)

        theta = theta.reshape(chi_l, d, d, chi_r)

        gate_mat = gate.reshape(d, d, d, d)
        if theta.is_complex() and gate_mat.dtype != theta.dtype:
            gate_mat = gate_mat.to(theta.dtype)
        elif not theta.is_complex():
            theta = theta.to(gate_mat.dtype)
        theta = torch.einsum("aijb,klij->aklb", theta, gate_mat)

        theta = theta.reshape(chi_l * d, d * chi_r)

        trunc_config = self.policy.get_truncation_config(
            bond_idx, step, hamiltonian_info)
        A_new, B_new, trunc_error = _dispatch_truncation(
            theta, chi_max, trunc_config, self.tc_gemm)

        if trunc_error is None:
            trunc_error = 0.0

        chi_new = A_new.shape[1]
        A_new = A_new.reshape(chi_l, d, chi_new)
        B_new = B_new.reshape(chi_new, d, chi_r)

        return A_new, B_new, trunc_error


class DistributedGateApplicator(GateApplicator):
    """Multi-GPU gate application for bond parallelism (strong scaling)."""

    def __init__(self, bond_comm, grid_rows: int, grid_cols: int,
                 tc_gemm: TensorCoreGEMM, precision_policy: PrecisionPolicy,
                 device: torch.device):
        self.tc_gemm = tc_gemm
        self.policy = precision_policy
        self.device = device
        self._bond_comm = bond_comm
        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._summa = None
        self._tsqr = None
        self._dist_trunc = None

    def set_distributed_ops(self, summa, tsqr, dist_trunc):
        """Set distributed operation objects after initialization."""
        self._summa = summa
        self._tsqr = tsqr
        self._dist_trunc = dist_trunc

    def apply(self, A: torch.Tensor, B: torch.Tensor, gate: torch.Tensor,
              chi_max: int, bond_idx: int, step: int,
              hamiltonian_info: dict) -> Tuple[torch.Tensor, torch.Tensor, float]:
        d = A.shape[1]
        chi_l = A.shape[0]
        chi_r = B.shape[2]

        contraction_prec = self.policy.get_contraction_precision(
            bond_idx, step, hamiltonian_info)

        A_mat = A.reshape(chi_l * d, -1)
        B_mat = B.reshape(B.shape[0], -1)

        if self._summa is not None and not (A_mat.is_complex() or B_mat.is_complex()):
            theta = self._summa.matmul(A_mat, B_mat)
        else:
            theta = self.tc_gemm.gemm(A_mat, B_mat,
                                       precision_override=contraction_prec.gemm_input_dtype)

        theta = theta.reshape(chi_l, d, d, chi_r)
        gate_mat = gate.reshape(d, d, d, d)
        if theta.is_complex() and gate_mat.dtype != theta.dtype:
            gate_mat = gate_mat.to(theta.dtype)
        elif not theta.is_complex():
            theta = theta.to(gate_mat.dtype)
        theta = torch.einsum("aijb,klij->aklb", theta, gate_mat)
        theta = theta.reshape(chi_l * d, d * chi_r)

        trunc_config = self.policy.get_truncation_config(
            bond_idx, step, hamiltonian_info)

        if self._dist_trunc is not None and trunc_config.method == "variational":
            A_new, B_new = self._dist_trunc.truncate(
                theta, chi_max, n_iter=trunc_config.n_iter)
            trunc_error = 0.0
        else:
            A_new, B_new, trunc_error = _dispatch_truncation(
                theta, chi_max, trunc_config, self.tc_gemm)

        if trunc_error is None:
            trunc_error = 0.0

        chi_new = A_new.shape[1]
        A_new = A_new.reshape(chi_l, d, chi_new)
        B_new = B_new.reshape(chi_new, d, chi_r)

        return A_new, B_new, trunc_error


__all__ = [
    "GateApplicator",
    "LocalGateApplicator",
    "DistributedGateApplicator",
]
