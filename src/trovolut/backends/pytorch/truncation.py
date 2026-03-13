"""Truncation methods for TEBD: SVD, randomized SVD, variational, QR.

The precision of GEMM operations is controlled via the TensorCoreGEMM interface,
allowing OpenEvolve to optimize which operations use FP4/FP8/FP16/FP32.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from trovolut.backends.pytorch.tc_ops import TensorCoreGEMM


def _promote_float(t: torch.Tensor) -> torch.Tensor:
    """Cast to float32, preserving complex dtype when present."""
    if t.is_complex():
        return t.to(torch.complex64)
    return t.float()


def svd_truncation(theta: torch.Tensor, chi_max: int,
                   compute_dtype: torch.dtype = torch.float32
                   ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Exact SVD truncation via cuSOLVER."""
    if theta.is_complex() and compute_dtype in (torch.float16, torch.float32):
        compute_dtype = torch.complex64
    elif theta.is_complex() and compute_dtype == torch.float64:
        compute_dtype = torch.complex128
    theta = theta.to(compute_dtype)
    try:
        U, S, Vh = torch.linalg.svd(theta, full_matrices=False, driver="gesvda")
    except torch._C._LinAlgError:
        U, S, Vh = torch.linalg.svd(theta, full_matrices=False)

    chi_new = min(chi_max, len(S))
    truncation_error = S[chi_new:].pow(2).sum().item() if chi_new < len(S) else 0.0

    U = U[:, :chi_new]
    S = S[:chi_new]
    Vh = Vh[:chi_new, :]

    sqrt_S = S.sqrt()
    A_new = U * sqrt_S.unsqueeze(0)
    B_new = sqrt_S.unsqueeze(1) * Vh

    return A_new, B_new, truncation_error


def randomized_svd_truncation(theta: torch.Tensor, chi_max: int,
                               tc_gemm: TensorCoreGEMM,
                               compute_dtype: torch.dtype = torch.float32,
                               oversampling: int = 10,
                               n_power_iter: int = 2
                               ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Randomized SVD using low-precision GEMMs for projections."""
    m, n = theta.shape
    k = min(chi_max + oversampling, min(m, n))
    is_complex = theta.is_complex()

    if is_complex and compute_dtype in (torch.float16, torch.float32):
        compute_dtype = torch.complex64
    elif is_complex and compute_dtype == torch.float64:
        compute_dtype = torch.complex128

    proj_dtype = torch.complex64 if is_complex else torch.float32
    omega = torch.randn(n, k, device=theta.device, dtype=proj_dtype)
    Y = tc_gemm.gemm(_promote_float(theta), omega)

    theta_f = _promote_float(theta)
    for _ in range(n_power_iter):
        Z = tc_gemm.gemm(theta_f.t().conj() if is_complex else theta_f.t(), Y)
        Z, _ = torch.linalg.qr(Z.to(compute_dtype))
        Y = tc_gemm.gemm(theta_f, _promote_float(Z))
        Y, _ = torch.linalg.qr(Y.to(compute_dtype))

    Q, _ = torch.linalg.qr(Y.to(compute_dtype))

    Q_f = _promote_float(Q)
    B_proj = tc_gemm.gemm(Q_f.t().conj() if is_complex else Q_f.t(),
                           theta_f)

    try:
        U_small, S, Vh = torch.linalg.svd(B_proj.to(compute_dtype).contiguous(),
                                            full_matrices=False, driver="gesvda")
    except torch._C._LinAlgError:
        U_small, S, Vh = torch.linalg.svd(B_proj.to(compute_dtype).contiguous(),
                                            full_matrices=False)

    chi_new = min(chi_max, len(S))
    truncation_error = S[chi_new:].pow(2).sum().item() if chi_new < len(S) else 0.0

    U_small = U_small[:, :chi_new]
    S = S[:chi_new]
    Vh = Vh[:chi_new, :]

    U_full = tc_gemm.gemm(Q_f, _promote_float(U_small))

    sqrt_S = S.sqrt()
    A_new = U_full * sqrt_S.unsqueeze(0)
    B_new = sqrt_S.unsqueeze(1) * Vh

    return A_new, B_new, truncation_error


def variational_truncation(theta: torch.Tensor, chi_max: int,
                            tc_gemm: TensorCoreGEMM,
                            n_iter: int = 8,
                            compute_dtype: torch.dtype = torch.float32
                            ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Variational (alternating least squares) truncation."""
    m, n = theta.shape
    chi = min(chi_max, min(m, n))
    is_complex = theta.is_complex()

    if is_complex and compute_dtype in (torch.float16, torch.float32):
        compute_dtype = torch.complex64
    elif is_complex and compute_dtype == torch.float64:
        compute_dtype = torch.complex128

    theta_f = _promote_float(theta)

    proj_dtype = torch.complex64 if is_complex else torch.float32
    omega = torch.randn(n, chi, device=theta.device, dtype=proj_dtype)
    Y = tc_gemm.gemm(theta_f, omega)
    Q, _ = torch.linalg.qr(Y.to(compute_dtype))
    A = _promote_float(Q[:, :chi])
    B = tc_gemm.gemm(A.t().conj() if is_complex else A.t(), theta_f)

    reg = 1e-6 * torch.eye(chi, device=theta.device, dtype=compute_dtype)

    for _ in range(n_iter):
        B_h = B.t().conj() if is_complex else B.t()
        BBh = tc_gemm.gemm(B, B_h)
        thetaBh = tc_gemm.gemm(theta_f, B_h)
        A = torch.linalg.solve((BBh.to(compute_dtype) + reg).t().conj(),
                                thetaBh.to(compute_dtype).t().conj()
                                ).t().conj()
        A = _promote_float(A)

        A_h = A.t().conj() if is_complex else A.t()
        AhA = tc_gemm.gemm(A_h, A)
        Ahtheta = tc_gemm.gemm(A_h, theta_f)
        B = _promote_float(torch.linalg.solve(AhA.to(compute_dtype) + reg,
                                               Ahtheta.to(compute_dtype)))

    AB = tc_gemm.gemm(A, B)
    error = (theta_f - AB).abs().pow(2).sum().item()

    return A, B, error


def qr_truncation(theta: torch.Tensor, chi_max: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, Optional[float]]:
    """QR-based truncation. Cheapest but least accurate."""
    Q, R = torch.linalg.qr(_promote_float(theta))
    chi_new = min(chi_max, Q.shape[1])
    A_new = Q[:, :chi_new]
    B_new = R[:chi_new, :]
    return A_new, B_new, None


__all__ = [
    "svd_truncation",
    "randomized_svd_truncation",
    "variational_truncation",
    "qr_truncation",
]
