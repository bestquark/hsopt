"""Trotter gate construction from Hamiltonians.

Supports second-order (Strang splitting) and fourth-order (Suzuki-Trotter)
decompositions with position-dependent gates.
"""

from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import torch

# Fourth-order Suzuki-Trotter coefficients
_S4_P = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
_S4_COEFFS = [_S4_P, _S4_P, 1.0 - 4.0 * _S4_P, _S4_P, _S4_P]


def make_two_site_gate(h_local: torch.Tensor, dt: float,
                       order: int = 2) -> torch.Tensor:
    """Construct two-site time evolution gate exp(-i h_local dt).

    Args:
        h_local: Local Hamiltonian term, shape (d^2, d^2).
        dt: Time step.
        order: Trotter order (2 or 4).

    Returns:
        Unitary gate of shape (d^2, d^2).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(h_local)
    exp_diag = torch.exp(-1j * eigenvalues * dt)
    gate = eigenvectors @ torch.diag(exp_diag) @ eigenvectors.conj().T
    return gate.to(h_local.dtype)


class TrotterGates:
    """Collection of gates for a full Trotter step.

    Supports position-dependent gates (varying couplings at each bond)
    and second/fourth-order Suzuki-Trotter decomposition.

    The ``hamiltonian`` argument accepts any object with:
    - ``n_sites: int``
    - ``d: int``
    - ``get_local_term(bond_index: int) -> Optional[torch.Tensor]``

    This includes both ``trovolut.backends.pytorch.hamiltonians.Hamiltonian``
    and ``tebd.core.hamiltonians.Hamiltonian``.
    """

    def __init__(self, n_sites: int, d: int = 2, order: int = 2,
                 device: Optional[torch.device] = None):
        self.n_sites = n_sites
        self.d = d
        self.order = order
        self.device = device or torch.device("cpu")
        self._gates: dict[int, torch.Tensor] = {}
        self._frac_gates: dict[Tuple[int, float], torch.Tensor] = {}

    @classmethod
    def from_hamiltonian(cls, hamiltonian, dt: float,
                         order: int = 2,
                         device: Optional[torch.device] = None) -> "TrotterGates":
        """Build Trotter gates from a Hamiltonian."""
        dev = device or torch.device("cpu")
        tg = cls(hamiltonian.n_sites, hamiltonian.d, order, dev)

        schedule = tg.get_trotter_step_schedule()
        unique_fracs = sorted(set(frac for _, frac in schedule))

        for bond_idx in range(hamiltonian.n_sites - 1):
            h_local = hamiltonian.get_local_term(bond_idx)
            if h_local is not None:
                h_local = h_local.to(dev)
                tg._gates[bond_idx] = make_two_site_gate(h_local, dt, order)
                for frac in unique_fracs:
                    gate = make_two_site_gate(h_local, dt * frac)
                    tg._frac_gates[(bond_idx, frac)] = gate

        return tg

    def get_gate(self, bond_index: int,
                 dt_frac: Optional[float] = None) -> Optional[torch.Tensor]:
        """Get the gate for a specific bond and dt fraction."""
        if dt_frac is not None:
            frac_gate = self._frac_gates.get((bond_index, dt_frac))
            if frac_gate is not None:
                return frac_gate
        return self._gates.get(bond_index)

    def get_sweep_schedule(self) -> List[Tuple[int, List[int]]]:
        """Get the sweep schedule as (parity, bond_list) pairs."""
        even_bonds = [i for i in range(0, self.n_sites - 1, 2)]
        odd_bonds = [i for i in range(1, self.n_sites - 1, 2)]

        if self.order == 2:
            return [(0, even_bonds), (1, odd_bonds), (0, even_bonds)]
        elif self.order == 4:
            schedule = []
            for _ in range(5):
                schedule.extend([(0, even_bonds), (1, odd_bonds), (0, even_bonds)])
            return schedule
        return [(0, even_bonds), (1, odd_bonds)]

    def get_trotter_step_schedule(self) -> List[Tuple[int, float]]:
        """Returns (parity, dt_fraction) for one Trotter step."""
        if self.order == 2:
            return [(0, 0.5), (1, 1.0), (0, 0.5)]
        elif self.order == 4:
            schedule = []
            for coeff in _S4_COEFFS:
                schedule.extend([
                    (0, 0.5 * coeff),
                    (1, coeff),
                    (0, 0.5 * coeff),
                ])
            return schedule
        return [(0, 0.5), (1, 1.0), (0, 0.5)]

    def iterate_step(self) -> Generator[Tuple[int, torch.Tensor, float], None, None]:
        """Yields (bond_idx, gate, dt_fraction) for one Trotter step."""
        schedule = self.get_trotter_step_schedule()
        for parity, dt_frac in schedule:
            start = parity
            for bond_idx in range(start, self.n_sites - 1, 2):
                gate = self.get_gate(bond_idx, dt_frac)
                if gate is not None:
                    yield bond_idx, gate, dt_frac


__all__ = [
    "make_two_site_gate",
    "TrotterGates",
]
