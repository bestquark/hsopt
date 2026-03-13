"""PyTorch TEBD dynamics conforming to trovolut's BaseDynamics interface.

Provides TEBDEvolution which runs Time-Evolving Block Decimation using
the PyTorch tensor network engine with mixed-precision tensor core support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from trovolut.core.base_dynamics import BaseDynamics
from trovolut.core.types import StateVector, DynamicsConfig

from trovolut.backends.pytorch.mps import MPS
from trovolut.backends.pytorch.gates import TrotterGates
from trovolut.backends.pytorch.sweep import TEBDSweep
from trovolut.backends.pytorch.gate_applicator import LocalGateApplicator
from trovolut.backends.pytorch.tc_ops import get_tc_gemm
from trovolut.backends.pytorch.precision_policy import PrecisionPolicy


class TEBDEvolution(BaseDynamics):
    """TEBD time evolution using PyTorch tensor networks.

    Uses Matrix Product States (MPS) with configurable precision policies
    and tensor core acceleration for gate application and truncation.

    Parameters
    ----------
    chi_max : int
        Maximum bond dimension for MPS truncation.
    order : int
        Trotter order (2 or 4).
    precision_policy : PrecisionPolicy, optional
        Mixed-precision policy. If None, uses default policy.
    storage_dtype : str
        MPS storage format ("fp4", "fp8", "fp16", "fp32", "fp64").
    device : torch.device, optional
        Device for computation. Auto-detects CUDA if available.
    """

    def __init__(
        self,
        chi_max: int = 256,
        order: int = 2,
        precision_policy: Optional[PrecisionPolicy] = None,
        storage_dtype: str = "fp8",
        device: Optional[torch.device] = None,
    ):
        self.chi_max = chi_max
        self.order = order
        self.policy = precision_policy or PrecisionPolicy()
        self.policy.chi_max = chi_max
        self.storage_dtype = storage_dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tc_gemm = get_tc_gemm(self.device if self.device.type == "cuda" else None)
        self._last_mps: Optional[MPS] = None
        self._last_truncation_errors: List[float] = []

    def evolve(
        self,
        hamiltonian,
        initial_state,
        time: float,
        config: Optional[DynamicsConfig] = None,
        n_steps: Optional[int] = None,
        **kwargs,
    ) -> StateVector:
        """Evolve initial_state under hamiltonian for given time using TEBD.

        Parameters
        ----------
        hamiltonian : object
            Hamiltonian with ``n_sites``, ``d``, ``get_local_term(bond_idx)``
            and ``get_structure_info()`` methods.
        initial_state : StateVector, MPS, or torch.Tensor
            Initial quantum state. If a state vector, it will be converted
            to an MPS internally.
        time : float
            Total evolution time.
        config : DynamicsConfig, optional
            Additional configuration.
        n_steps : int, optional
            Number of Trotter steps. If None, defaults to ceil(10 * time).

        Returns
        -------
        StateVector
            Final state as a numpy array (for compatibility with trovolut).
        """
        if n_steps is None:
            n_steps = max(1, int(np.ceil(10 * abs(time))))

        dt = time / n_steps
        n_sites = hamiltonian.n_sites

        # Build Trotter gates
        trotter_gates = TrotterGates.from_hamiltonian(
            hamiltonian, dt, order=self.order, device=self.device
        )

        # Initialize MPS
        if isinstance(initial_state, MPS):
            mps = initial_state
        else:
            mps = self._state_to_mps(initial_state, n_sites, hamiltonian.d)

        # Setup gate applicator
        applicator = LocalGateApplicator(self._tc_gemm, self.policy)
        sweep = TEBDSweep(n_sites, trotter_gates, self.order)

        # Get Hamiltonian structure info for precision decisions
        ham_info = {}
        if hasattr(hamiltonian, "get_structure_info"):
            ham_info = hamiltonian.get_structure_info()
        ham_info["n_sites"] = n_sites
        ham_info["n_steps"] = n_steps

        # Run TEBD steps
        self._last_truncation_errors = []
        for step in range(n_steps):
            for bond_idx, gate, dt_frac in sweep.iterate_step():
                A = mps.get(bond_idx)
                B = mps.get(bond_idx + 1)

                A_new, B_new, trunc_error = applicator.apply(
                    A, B, gate, self.chi_max, bond_idx, step, ham_info
                )

                # Store with policy-determined precision
                storage = self.policy.get_storage_dtype(bond_idx, step)
                old_dtype = mps.storage_dtype
                mps.storage_dtype = storage
                mps.set(bond_idx, A_new)
                mps.set(bond_idx + 1, B_new)
                mps.storage_dtype = old_dtype

                self._last_truncation_errors.append(trunc_error)

            # Periodic canonicalization
            if self.policy.should_canonicalize(step):
                mps.canonicalize(n_sites // 2)

            # Periodic renormalization
            if self.policy.should_renormalize(step):
                mps.normalize()

        self._last_mps = mps

        # Convert MPS back to state vector
        return self._mps_to_state(mps)

    def _state_to_mps(self, state, n_sites: int, d: int) -> MPS:
        """Convert a state vector to MPS via successive SVDs."""
        if isinstance(state, np.ndarray):
            psi = torch.tensor(state, dtype=torch.complex64, device=self.device)
        elif isinstance(state, torch.Tensor):
            psi = state.to(dtype=torch.complex64, device=self.device)
        else:
            psi = torch.tensor(np.asarray(state), dtype=torch.complex64, device=self.device)

        psi = psi.reshape(-1)
        dim = d ** n_sites
        if psi.numel() != dim:
            raise ValueError(f"State dimension {psi.numel()} != d^n = {dim}")

        # Successive SVD decomposition into MPS
        tensors = []
        remaining = psi.reshape(1, -1)  # (1, d^n)

        for i in range(n_sites - 1):
            chi_l = remaining.shape[0]
            remaining = remaining.reshape(chi_l * d, -1)
            U, S, Vh = torch.linalg.svd(remaining, full_matrices=False)

            chi_new = min(self.chi_max, len(S))
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]

            tensors.append(U.reshape(chi_l, d, chi_new))
            remaining = torch.diag(S) @ Vh

        # Last tensor
        chi_l = remaining.shape[0]
        tensors.append(remaining.reshape(chi_l, d, 1))

        return MPS.from_tensors(tensors, storage_dtype=self.storage_dtype)

    def _mps_to_state(self, mps: MPS) -> StateVector:
        """Contract MPS into a state vector."""
        result = mps.get(0)  # (chi_l=1, d, chi_r)
        for i in range(1, mps.n_sites):
            t = mps.get(i)  # (chi_l, d, chi_r)
            result = torch.einsum("...i,ijk->...jk", result, t)

        # Result shape: (1, d, d, ..., d, 1) -> flatten to (d^n,)
        psi = result.reshape(-1)
        return psi.cpu().numpy()

    @property
    def last_mps(self) -> Optional[MPS]:
        """Return the MPS from the last evolution (for inspection)."""
        return self._last_mps

    @property
    def last_truncation_errors(self) -> List[float]:
        """Return truncation errors from the last evolution."""
        return self._last_truncation_errors


__all__ = [
    "TEBDEvolution",
]
