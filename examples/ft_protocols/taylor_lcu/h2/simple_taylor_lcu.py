"""H2 Taylor/LCU method configuration using double factorization.

This method file provides a frozen dataclass with the factorization type,
discretization parameters, Taylor order, and error budget for the Taylor LCU
Hamiltonian simulation protocol applied to H2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaylorLCUConfig:
    """Configuration for Taylor LCU Hamiltonian simulation.

    Parameters
    ----------
    name : str
        Human-readable identifier for this configuration.
    factorization : str
        Hamiltonian factorization method: ``"df"`` (double factorization) or
        ``"thc"`` (tensor hypercontraction).
    beth_per_leaf : tuple[int, ...]
        Bit-widths for Givens rotation angles, one per DF leaf.  If fewer
        entries than leaves exist, the last value is repeated.
    aleph : int
        Bit-width for coefficient discretization in PREPARE.
    taylor_order : int
        Truncation order K of the Taylor series expansion.
    epsilon_taylor : float
        Total allowable Taylor truncation error across all segments.
    """
    name: str = "h2_df_taylor_lcu"
    factorization: str = "df"
    beth_per_leaf: tuple[int, ...] = (16, 16, 16)
    aleph: int = 10
    taylor_order: int = 8
    epsilon_taylor: float = 1e-4


def get_method_config() -> TaylorLCUConfig:
    """Return the Taylor LCU method configuration for H2."""
    return TaylorLCUConfig()
