"""
Editable qubitization-QPE resource policy for H2 with double factorization.

The evaluator reads ``get_method_config()`` and uses its parameters to:
  - perform a DF factorization and discretize Givens angles / eigenvalue coefficients,
  - build the effective Hamiltonian,
  - count Toffoli gates via paper-exact cost formulas.

Autoresearch mutations should only change the numeric knobs below.

H2 in STO-3G (CAS(2e,2o)) has 2 spatial orbitals, which produces 3 DF leaves.
Each leaf gets its own Givens-angle bit-width (beth_per_leaf), and all
eigenvalue coefficients share a single fixed-point bit-width (aleph).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QubitizationConfig:
    """Discretization parameters for DF qubitization-QPE on H2."""

    name: str = "h2_df_qubitization_qpe"

    # Which two-body factorization to use: "df" (double factorization) or "thc"
    factorization: str = "df"

    # Givens-rotation angle bit-width per DF leaf.
    # H2 STO-3G yields 3 leaves from the double factorization.
    beth_per_leaf: tuple[int, ...] = (16, 16, 16)

    # Fixed-point bit-width for eigenvalue coefficients in PREPARE.
    aleph: int = 10

    # Phase-estimation energy resolution (Hartree).
    # QPE iterations = ceil(pi * lambda / (2 * epsilon_qpe)).
    epsilon_qpe: float = 0.001


def get_method_config() -> QubitizationConfig:
    """Return the current resource policy configuration."""
    return QubitizationConfig()
