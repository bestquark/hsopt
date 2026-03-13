"""TEBD sweep scheduler for Trotter decomposition.

Manages the ordering of gate applications within each Trotter step,
supporting second-order (Strang) and fourth-order (Suzuki-Trotter).
"""

from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import torch

from trovolut.backends.pytorch.gates import TrotterGates

# Fourth-order Suzuki-Trotter coefficient
_S4_P = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))


class TEBDSweep:
    """Manages the Trotter sweep ordering.

    For order=2 (Strang splitting):
        Half-step even bonds, full-step odd bonds, half-step even bonds.
    For order=4 (Suzuki-Trotter):
        Composed from five second-order steps with appropriate coefficients.
    """

    def __init__(self, n_sites: int, trotter_gates: TrotterGates, order: int = 2):
        self.n_sites = n_sites
        self.trotter_gates = trotter_gates
        self.order = order

    def get_half_sweep_bonds(self, parity: int) -> List[int]:
        """Return bond indices for this parity (0=even, 1=odd)."""
        return list(range(parity, self.n_sites - 1, 2))

    def get_trotter_step_schedule(self) -> List[Tuple[int, float]]:
        """Returns (parity, dt_fraction) pairs for one full Trotter step."""
        if self.order == 2:
            return [(0, 0.5), (1, 1.0), (0, 0.5)]
        elif self.order == 4:
            schedule = []
            coeffs = [_S4_P, _S4_P, 1.0 - 4.0 * _S4_P, _S4_P, _S4_P]
            for c in coeffs:
                schedule.extend([
                    (0, 0.5 * c),
                    (1, c),
                    (0, 0.5 * c),
                ])
            return schedule
        raise ValueError(f"Unsupported Trotter order: {self.order}")

    def iterate_step(self) -> Generator[Tuple[int, Optional[torch.Tensor], float], None, None]:
        """Yields (bond_idx, gate, dt_fraction) for one Trotter step."""
        schedule = self.get_trotter_step_schedule()
        for parity, dt_frac in schedule:
            for bond_idx in self.get_half_sweep_bonds(parity):
                gate = self.trotter_gates.get_gate(bond_idx, dt_frac)
                if gate is not None:
                    yield bond_idx, gate, dt_frac


__all__ = [
    "TEBDSweep",
]
