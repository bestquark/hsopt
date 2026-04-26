"""
Editable QSVT dynamics method configuration for H2.

Uses double factorization (DF) with discretized Givens rotations and
coefficients to build the effective Hamiltonian, then applies QSVT
with a Jacobi-Anger polynomial to approximate e^{-iHt}.

Tunable parameters:
  - beth_per_leaf: rotation bit-width per DF leaf (controls Givens angle
    precision; more bits = smaller discretization error, larger circuits).
  - aleph: coefficient bit-width for PREPARE (controls eigenvalue rounding;
    more bits = smaller discretization error, larger circuits).
  - polynomial_precision: target precision for the Jacobi-Anger polynomial
    approximation (smaller = higher degree = more walk-operator applications).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QSVTConfig:
    name: str = "h2_df_qsvt_dynamics"
    factorization: str = "df"
    beth_per_leaf: tuple[int, ...] = (16, 16, 16)
    aleph: int = 10
    polynomial_precision: float = 1e-4


def get_method_config() -> QSVTConfig:
    return QSVTConfig()
