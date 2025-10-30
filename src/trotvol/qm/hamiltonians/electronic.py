"""Electronic-structure Hamiltonians"""

from __future__ import annotations
from typing import Any, Dict, Sequence, Union, Optional, Callable, List

import numpy as np
from openfermion import (
    QubitOperator,
    FermionOperator,
    jordan_wigner,
    bravyi_kitaev,
    parity_code,
    binary_code_transform,
    get_sparse_operator,
)

from trotvol.qm.hamiltonians._hamiltonian import Hamiltonian
from trotvol.qm.hamiltonians.pauli import PauliSumHamiltonian


def _parse_xyz(xyz_source: Union[str, Sequence[str]]) -> str:
    """Normalize XYZ input into a geometry string understood by PySCF.

    Args:
        xyz_source (Union[str, Sequence[str]]): Either a path to an XYZ file, a
            multiline XYZ string, or an iterable of XYZ lines.

    Returns:
        str: Semicolon-separated ``"Atom x y z"`` specification.

    Raises:
        ValueError: If the XYZ content is malformed or inconsistent.
    """
    if isinstance(xyz_source, str):
        if "\n" in xyz_source or "\r" in xyz_source:
            lines = xyz_source.strip().splitlines()
        else:
            with open(xyz_source, "r") as f:
                lines = f.read().strip().splitlines()
    else:
        lines = list(xyz_source)

    if len(lines) < 3:
        raise ValueError("XYZ appears too short")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("First line of xyz must be atom count")

    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError("XYZ atom count mismatch")

    geom_parts = []
    for L in atom_lines:
        toks = L.split()
        if len(toks) < 4:
            raise ValueError(f"Bad xyz atom line: {L}")
        sym = toks[0]
        x, y, z = toks[1:4]
        geom_parts.append(f"{sym} {x} {y} {z}")
    return "; ".join(geom_parts)


def _scf_with_optional_external(
    geom_str: str,
    basis: str,
    charge: int,
    spin_mult: int,
    external_potential=None,
    use_gpu4pyscf: bool = False,
):
    # try gpu
    from pyscf import gto, scf, qmmm as pyscf_qmmm, dft as pyscf_dft

    if use_gpu4pyscf:
        try:
            from gpu4pyscf import scf as gpu_scf

            have_gpu = True
        except ImportError:
            have_gpu = False
    else:
        have_gpu = False

    # spin multiplicity M = 2S+1  -> S=(M-1)/2  -> spin = 2S = #alpha-#beta
    S = (spin_mult - 1) / 2.0
    mol_spin = int(round(2 * S))

    mol = gto.Mole()
    mol.build(
        atom=geom_str,
        basis=basis,
        charge=charge,
        spin=mol_spin,
        unit="Ang",
    )

    if mol.spin == 0:
        mf = gpu_scf.RHF(mol) if have_gpu else scf.RHF(mol)
    else:
        mf = gpu_scf.UHF(mol) if have_gpu else scf.UHF(mol)

    if external_potential is not None:
        # ---- case A: list/ndarray of [x,y,z,q] point charges ----
        if not callable(external_potential):
            charges_arr = np.asarray(external_potential)
            if charges_arr.ndim != 2 or charges_arr.shape[1] != 4:
                raise ValueError("external_potential array must be (N,4): [x,y,z,q]")
            coords_pc = np.array(charges_arr[:, :3])
            qs = np.array(charges_arr[:, 3])
            mf = pyscf_qmmm.mm_charge(mf, coords_pc, qs)

        # ---- case B: callable V(r) ----
        else:
            orig_get_veff = mf.get_veff
            mol_ref = mf.mol

            # grid
            if hasattr(mf, "grids"):
                grids = mf.grids
                if not getattr(grids, "coords", None):
                    grids.build()
            else:
                grids = pyscf_dft.gen_grid.Grids(mol_ref)
                grids.build()

            coords_grid = grids.coords  # (Ng,3) in Bohr
            weights = grids.weights  # (Ng,)
            ni = pyscf_dft.numint.NumInt()
            ao = ni.eval_ao(mol_ref, coords_grid)  # (Ng, nao)

            # external_potential must take Bohr coords, return Hartree
            v_ext_vals = external_potential(coords_grid)  # (Ng,)

            wv = weights * v_ext_vals  # (Ng,)
            Vmat = ao.T @ (wv[:, None] * ao)  # (nao, nao)

            def get_veff_with_external(mol_=None, dm=None, *args, **kwargs):
                veff = orig_get_veff(mol_ or mol_ref, dm, *args, **kwargs)

                # RHF/RKS case
                if isinstance(veff, np.ndarray):
                    return veff + Vmat

                # UHF/UKS case
                va, vb = veff
                return (va + Vmat, vb + Vmat)

            mf.get_veff = get_veff_with_external

    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    return mf, mol


def _integrals_from_mf(mf, mol):
    """Extract spin-orbital integrals from a converged mean-field object.

    Args:
        mf: PySCF mean-field object with converged orbitals.
        mol: PySCF molecule corresponding to ``mf``.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: One-body integrals ``h1`` (``n_spin_orb, n_spin_orb``), antisymmetrized two-body integrals ``h2`` (``n_spin_orb``^4), and the nuclear repulsion constant.
    """
    from pyscf import ao2mo

    e_nuc = mol.energy_nuc()
    h_core_ao = mf.get_hcore()

    if mol.spin == 0:
        # RHF
        C = mf.mo_coeff  # (nao, nmo)
        h_core_mo = C.T @ h_core_ao @ C  # (nmo, nmo)
        nmo = C.shape[1]

        eri_mo_spatial_packed = ao2mo.kernel(mol, C)
        eri_mo_spatial = ao2mo.restore(1, eri_mo_spatial_packed, nmo)
        # eri_mo_spatial[p,q,r,s] = (pq|rs)

        n_spin_orb = 2 * nmo

        # h1 block-diag for spin
        h1 = np.zeros((n_spin_orb, n_spin_orb))
        h1[0::2, 0::2] = h_core_mo
        h1[1::2, 1::2] = h_core_mo

        # antisymmetrized h2 in spin-orbital basis
        h2 = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        pq_rs = eri_mo_spatial[p, q, r, s]  # (pq|rs)
                        pq_sr = eri_mo_spatial[p, q, s, r]  # (pq|sr)
                        for sp_p in (0, 1):
                            for sp_q in (0, 1):
                                for sp_r in (0, 1):
                                    for sp_s in (0, 1):
                                        P = 2 * p + sp_p
                                        Q = 2 * q + sp_q
                                        R = 2 * r + sp_r
                                        Sidx = 2 * s + sp_s
                                        term = 0.0
                                        if sp_p == sp_r and sp_q == sp_s:
                                            term += pq_rs  # Coulomb
                                        if sp_p == sp_s and sp_q == sp_r:
                                            term -= pq_sr  # Exchange
                                        if term != 0.0:
                                            h2[P, Q, R, Sidx] += term

    else:
        # UHF / open shell
        Ca, Cb = mf.mo_coeff  # (nao,nmo_a), (nao,nmo_b)
        h_core_mo_a = Ca.T @ h_core_ao @ Ca
        h_core_mo_b = Cb.T @ h_core_ao @ Cb

        nmo_a = Ca.shape[1]
        nmo_b = Cb.shape[1]
        n_spin_orb = nmo_a + nmo_b

        h1 = np.zeros((n_spin_orb, n_spin_orb))
        h1[:nmo_a, :nmo_a] = h_core_mo_a
        h1[nmo_a:, nmo_a:] = h_core_mo_b

        eri_aa_packed = ao2mo.kernel(mol, Ca)
        eri_bb_packed = ao2mo.kernel(mol, Cb)
        eri_ab_packed = ao2mo.kernel(mol, (Ca, Cb, Ca, Cb))

        eri_aa = ao2mo.restore(1, eri_aa_packed, nmo_a)
        eri_bb = ao2mo.restore(1, eri_bb_packed, nmo_b)
        eri_ab = ao2mo.restore(1, eri_ab_packed, (nmo_a, nmo_b, nmo_a, nmo_b))

        h2 = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))

        # alpha-alpha / beta-beta antisymmetrized
        for p in range(nmo_a):
            for q in range(nmo_a):
                for r in range(nmo_a):
                    for s in range(nmo_a):
                        coul = eri_aa[p, q, r, s]
                        exch = eri_aa[p, q, s, r]
                        val = coul - exch
                        h2[p, q, r, s] += val

        offset = nmo_a
        for p in range(nmo_b):
            for q in range(nmo_b):
                for r in range(nmo_b):
                    for s in range(nmo_b):
                        coul = eri_bb[p, q, r, s]
                        exch = eri_bb[p, q, s, r]
                        val = coul - exch
                        P = offset + p
                        Q = offset + q
                        R = offset + r
                        Sidx = offset + s
                        h2[P, Q, R, Sidx] += val

        # alpha-beta Coulomb only
        for p in range(nmo_a):
            for q in range(nmo_b):
                for r in range(nmo_a):
                    for s in range(nmo_b):
                        coul = eri_ab[p, q, r, s]
                        P = p
                        Q = offset + q
                        R = r
                        Sidx = offset + s
                        h2[P, Q, R, Sidx] += coul
                        h2[offset + q, p, offset + s, r] += coul

    constant = e_nuc
    return h1, h2, constant


def _fermion_operator_from_integrals(
    h1: np.ndarray,
    h2: np.ndarray,
    constant: float,
) -> "FermionOperator":
    """Build an OpenFermion FermionOperator from spin-orbital integrals.

    Uses sparsity of h1/h2 to avoid O(n_orb^4) pure-Python loops.
    """
    from openfermion import FermionOperator

    n_orb = h1.shape[0]
    H_f = FermionOperator()

    thresh = 1e-12

    # --- 1-body terms: sum_{pq} h1[p,q] a_p^\dag a_q
    nz_p, nz_q = np.nonzero(np.abs(h1) > thresh)
    for p_raw, q_raw in zip(nz_p, nz_q):
        p = int(p_raw)
        q = int(q_raw)
        c = h1[p, q]
        H_f += FermionOperator(
            ((p, 1), (q, 0)),  # a_p^\dag a_q
            float(c),
        )

    # --- 2-body terms: (1/2) sum_{pqrs} h2[p,q,r,s] a_p^\dag a_q^\dag a_r a_s
    # flatten (pq, rs) so we can nonzero() once
    h2_eff = 0.5 * h2
    n2 = n_orb * n_orb
    flat_h2 = h2_eff.reshape(n2, n2)

    nz_row, nz_col = np.nonzero(np.abs(flat_h2) > thresh)

    for row_raw, col_raw in zip(nz_row, nz_col):
        row = int(row_raw)
        col = int(col_raw)

        val = flat_h2[row, col]
        if abs(val) <= thresh:
            continue

        # decode row -> (p,q), col -> (r,s)
        p = row // n_orb
        q = row % n_orb
        r = col // n_orb
        s = col % n_orb

        # cast to plain Python int for OpenFermion
        p = int(p)
        q = int(q)
        r = int(r)
        s = int(s)

        H_f += FermionOperator(
            ((p, 1), (q, 1), (r, 0), (s, 0)),  # a_p^\dag a_q^\dag a_r a_s
            float(val),
        )

    # --- constant / nuclear repulsion etc.
    if abs(constant) > thresh:
        H_f += FermionOperator((), float(constant))

    return H_f


def _qubit_op_to_terms(
    qubit_op: "QubitOperator",
    n_qubits: int,
) -> List[Dict[str, Any]]:
    """Extract Pauli-term list *only*, skipping full matrix construction.

    Accepts sparse notation strings like "X0 Z1 Y2" (identity on all other qubits).
    """

    terms: List[Dict[str, Any]] = []
    for pauli_term, coeff in qubit_op.terms.items():
        if len(pauli_term) == 0:
            pstr = "I"
        else:
            sorted_ops = sorted(pauli_term, key=lambda t: t[0])
            pstr = " ".join(f"{op}{idx}" for idx, op in sorted_ops)
        terms.append({"coeff": complex(coeff), "pauli": pstr})

    return terms


def _apply_a(bitstring: int, p: int):
    """Apply the annihilation operator ``a_p`` to a bitstring Fock state.

    Args:
        bitstring (int): Integer encoding the occupation pattern.
        p (int): Orbital index.

    Returns:
        Tuple[Optional[int], float]: New bitstring (or ``None`` if the term annihilates the state) and the parity phase.
    """
    if ((bitstring >> p) & 1) == 0:
        return None, 0.0
    # fermionic sign = (-1)^(# of occupied modes < p)
    mask = (1 << p) - 1
    sign = (-1) ** (bin(bitstring & mask).count("1"))
    new_bs = bitstring ^ (1 << p)
    return new_bs, sign


def _apply_adag(bitstring: int, p: int):
    """Apply the creation operator ``a_p^†`` to a bitstring Fock state.

    Args:
        bitstring (int): Integer encoding the occupation pattern.
        p (int): Orbital index.

    Returns:
        Tuple[Optional[int], float]: New bitstring (or ``None`` if creation is blocked) and the parity phase.
    """
    if ((bitstring >> p) & 1) == 1:
        return None, 0.0
    mask = (1 << p) - 1
    sign = (-1) ** (bin(bitstring & mask).count("1"))
    new_bs = bitstring ^ (1 << p)
    return new_bs, sign


def _one_body_term_matrix(M: int, p: int, q: int, coeff: complex) -> np.ndarray:
    """Construct the matrix for ``coeff * a_p^† a_q`` in the Fock basis.

    Args:
        M (int): Number of spin orbitals.
        p (int): Creation index.
        q (int): Annihilation index.
        coeff (complex): Term coefficient.

    Returns:
        np.ndarray: Dense ``(2**M, 2**M)`` matrix block.
    """
    dim = 1 << M
    block = np.zeros((dim, dim), dtype=complex)
    for bs in range(dim):
        bs1, ph1 = _apply_a(bs, q)
        if bs1 is None:
            continue
        bs2, ph2 = _apply_adag(bs1, p)
        if bs2 is None:
            continue
        block[bs2, bs] += coeff * ph1 * ph2
    return block


def _two_body_term_matrix(M: int, p: int, q: int, r: int, s: int, coeff: complex) -> np.ndarray:
    """Construct the matrix for ``coeff * a_p^† a_q^† a_r a_s``.

    Args:
        M (int): Number of spin orbitals.
        p (int): First creation index.
        q (int): Second creation index.
        r (int): First annihilation index.
        s (int): Second annihilation index.
        coeff (complex): Term coefficient.

    Returns:
        np.ndarray: Dense ``(2**M, 2**M)`` representation.
    """
    dim = 1 << M
    block = np.zeros((dim, dim), dtype=complex)
    for bs in range(dim):
        bs1, ph1 = _apply_a(bs, s)
        if bs1 is None:
            continue
        bs2, ph2 = _apply_a(bs1, r)
        if bs2 is None:
            continue
        bs3, ph3 = _apply_adag(bs2, q)
        if bs3 is None:
            continue
        bs4, ph4 = _apply_adag(bs3, p)
        if bs4 is None:
            continue
        block[bs4, bs] += coeff * ph1 * ph2 * ph3 * ph4
    return block


def _pauli_string_to_matrix(pauli_ops, n_qubits: int) -> np.ndarray:
    """Build the matrix for a single Pauli string.

    Args:
        pauli_ops (Sequence[Tuple[int, str]]): Items like ``(index, "X")``.
        n_qubits (int): Register size.

    Returns:
        np.ndarray: Dense ``(2**n_qubits, 2**n_qubits)`` matrix.
    """
    # build one-term QubitOperator
    term = ()
    if len(pauli_ops) > 0:
        term = tuple((idx, op) for (idx, op) in pauli_ops)
    single = QubitOperator(term, 1.0)
    sparse_single = get_sparse_operator(single, n_qubits=n_qubits)
    return sparse_single.toarray()


def parity_mapping(fermion_operator, n_modes):
    return binary_code_transform(fermion_operator, parity_code(n_modes))


def _active_space_from_mf(
    mf,
    n_core_keep: int,
    n_virt_keep: int,
):
    """
    Build active-space spin-orbital integrals from a converged RHF `mf`,
    keeping only a subset of spatial MOs near the frontier.

    Returns:
        h1_spin, h2_spin, const_term, meta
    where meta includes active/frozen orbital info.
    """
    import numpy as _np
    from pyscf import ao2mo

    mol = mf.mol
    e_nuc = mol.energy_nuc()

    C = mf.mo_coeff  # (nao, nmo)
    eps = mf.mo_energy  # (nmo,)
    occ = mf.mo_occ  # (nmo,)
    nmo = C.shape[1]

    # occupied / virtual split
    occ_idx = [i for i in range(nmo) if occ[i] > 1e-8]
    virt_idx = [i for i in range(nmo) if occ[i] <= 1e-8]

    # sort within each block by energy
    occ_sorted = sorted(occ_idx, key=lambda i: eps[i])  # low -> high
    virt_sorted = sorted(virt_idx, key=lambda i: eps[i])  # low -> high

    # choose active occ/virt
    active_occ = occ_sorted[-n_core_keep:] if n_core_keep <= len(occ_sorted) else occ_sorted[:]
    active_virt = virt_sorted[:n_virt_keep] if n_virt_keep <= len(virt_sorted) else virt_sorted[:]

    active_indices = sorted(active_occ + active_virt)

    # frozen ones = everything else
    frozen_indices = sorted(set(range(nmo)) - set(active_indices))

    # transform core Hamiltonian and ERIs to MO basis
    h_core_ao = mf.get_hcore()
    h_core_mo = C.T @ h_core_ao @ C  # (nmo,nmo)

    eri_mo_packed = ao2mo.kernel(mol, C)
    eri_mo = ao2mo.restore(1, eri_mo_packed, nmo)  # (nmo,nmo,nmo,nmo)

    act = _np.array(active_indices, dtype=int)
    n_act = act.shape[0]

    # restricted 1e and 2e integrals in spatial active subspace
    h1_spatial = h_core_mo[_np.ix_(act, act)]  # (n_act,n_act)

    h2_spatial = eri_mo[
        act[:, None, None, None],
        act[None, :, None, None],
        act[None, None, :, None],
        act[None, None, None, :],
    ]

    # spin-orbital expansion
    n_spin_orb = 2 * n_act
    h1_spin = _np.zeros((n_spin_orb, n_spin_orb))
    h1_spin[0::2, 0::2] = h1_spatial
    h1_spin[1::2, 1::2] = h1_spatial

    h2_spin = _np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))

    for p_i in range(n_act):
        for q_i in range(n_act):
            for r_i in range(n_act):
                for s_i in range(n_act):
                    coul = h2_spatial[p_i, q_i, r_i, s_i]  # (pq|rs)
                    exch = h2_spatial[p_i, q_i, s_i, r_i]  # (pq|sr)

                    for sp_p in (0, 1):
                        for sp_q in (0, 1):
                            for sp_r in (0, 1):
                                for sp_s in (0, 1):
                                    P = 2 * p_i + sp_p
                                    Q = 2 * q_i + sp_q
                                    R = 2 * r_i + sp_r
                                    S = 2 * s_i + sp_s

                                    term = 0.0
                                    if sp_p == sp_r and sp_q == sp_s:
                                        term += coul
                                    if sp_p == sp_s and sp_q == sp_r:
                                        term -= exch
                                    if abs(term) > 1e-12:
                                        h2_spin[P, Q, R, S] += term

    # constant term: start with just nuclear repulsion.
    const_term = e_nuc

    meta = {
        # which MOs we kept (spatial)
        "active_indices": active_indices,
        "frozen_indices": frozen_indices,
        # label which part of active_indices came from occ/virt
        "active_occ": active_occ,
        "active_virt": active_virt,
        # orbital info for debugging / plotting frontier orbitals
        "orbital_energies": eps.tolist(),
        "orbital_occupations": occ.tolist(),
        # sizes
        "n_act_spatial": n_act,
        "n_spin_orb": n_spin_orb,
        "n_core_keep": int(n_core_keep),
        "n_virt_keep": int(n_virt_keep),
    }

    return h1_spin, h2_spin, const_term, meta


# ----------------------- main classes ----------------------- #


class ElectronicHamiltonian(Hamiltonian):
    """Second-quantized electronic Hamiltonian in a spin-orbital basis.

    Args:
        h1 (np.ndarray): One-body integrals ``(M, M)``.
        h2 (np.ndarray): Two-body integrals ``(M, M, M, M)``.
        constant (float): Scalar nuclear-repulsion/shift term.
        meta (Optional[Dict[str, Any]]): Optional metadata dictionary.

    Attributes:
        h1 (np.ndarray): Stored copy of the one-body integrals.
        h2 (np.ndarray): Stored copy of the two-body integrals.
        constant (float): Energy shift added to the Hamiltonian.
        M (int): Number of spin orbitals.
    """

    def __init__(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        constant: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.h1 = np.array(h1, dtype=float)
        self.h2 = np.array(h2, dtype=float)
        self.constant = float(constant)

        M = self.h1.shape[0]
        if self.h1.shape != (M, M):
            raise ValueError("h1 must be (M,M)")
        if self.h2.shape != (M, M, M, M):
            raise ValueError("h2 must be (M,M,M,M)")
        self.M = M
        self._hilbert_dim = 2**M  # spin-orbital occupation basis dim
        self.meta = meta or {}

    @property
    def hilbert_dim(self) -> int:
        """int: Dimension of the fermionic Fock space (``2**M``)."""
        return self._hilbert_dim

    @property
    def is_stochastic(self) -> bool:
        """bool: ElectronicHamiltonian instances are deterministic."""
        return False

    def to_matrix(self) -> np.ndarray:
        """Return the fermionic Hamiltonian matrix in Fock-space ordering.

        Returns:
            np.ndarray: Dense ``(2**M, 2**M)`` many-body matrix.
        """
        # brute-force build second-quantized matrix
        M = self.M
        dim = 1 << M  # 2**M
        Hmat = np.zeros((dim, dim), dtype=complex)

        # we'll apply creation/annihilation on bitstrings
        # a_p |n> = (-1)^{#occ< p} n_p |n with p flipped to 0> etc.
        # Note: this is standard Jordan–Wigner action in Fock space.
        # This is exponential, so OK only for small M.
        def occ(bitstring, p):
            return (bitstring >> p) & 1

        def phase(bitstring, p):
            # number of 1s below p
            mask = (1 << p) - 1
            return (-1) ** (bin(bitstring & mask).count("1"))

        def apply_a(bitstring, p):
            # annihilate at p
            if not occ(bitstring, p):
                return None, 0.0
            new_bs = bitstring ^ (1 << p)
            return new_bs, phase(bitstring, p)

        def apply_adag(bitstring, p):
            # create at p
            if occ(bitstring, p):
                return None, 0.0
            new_bs = bitstring ^ (1 << p)
            return new_bs, phase(bitstring, p)

        # 1-body: sum_{pq} h1[p,q] a_p^† a_q
        for p in range(M):
            for q in range(M):
                hpq = self.h1[p, q]
                if abs(hpq) < 1e-12:
                    continue
                for bs in range(dim):
                    # a_q on |bs>
                    bs1, phase1 = apply_a(bs, q)
                    if bs1 is None:
                        continue
                    # a_p^† on result
                    bs2, phase2 = apply_adag(bs1, p)
                    if bs2 is None:
                        continue
                    amp = hpq * phase1 * phase2
                    Hmat[bs2, bs] += amp

        # 2-body: (1/2) sum_{pqrs} h2[p,q,r,s] a_p^† a_q^† a_r a_s
        for p in range(M):
            for q in range(M):
                for r in range(M):
                    for s in range(M):
                        hpqrs = 0.5 * self.h2[p, q, r, s]
                        if abs(hpqrs) < 1e-12:
                            continue
                        for bs in range(dim):
                            # a_s
                            bs1, ph1 = apply_a(bs, s)
                            if bs1 is None:
                                continue
                            # a_r
                            bs2, ph2 = apply_a(bs1, r)
                            if bs2 is None:
                                continue
                            # a_q^†
                            bs3, ph3 = apply_adag(bs2, q)
                            if bs3 is None:
                                continue
                            # a_p^†
                            bs4, ph4 = apply_adag(bs3, p)
                            if bs4 is None:
                                continue
                            amp = hpqrs * ph1 * ph2 * ph3 * ph4
                            Hmat[bs4, bs] += amp

        # constant shift
        if abs(self.constant) > 1e-12:
            for bs in range(dim):
                Hmat[bs, bs] += self.constant

        return Hmat

    def decompose(self) -> Dict[str, Any]:
        """Provide lightweight metadata describing the Hamiltonian.

        Returns:
            Dict[str, Any]: JSON-serializable data with tensor shapes.
        """
        return {
            "type": "second_quantized",
            "one_body": self.h1,
            "two_body": self.h2,
            "constant": self.constant,
            "n_spin_orbs": self.M,
        }

    def to_qubits(
        self,
        mapping: Optional[str] = "jordan-wigner",
        sparse: bool = False,
    ) -> "QElectronicHamiltonian":
        H_f = _fermion_operator_from_integrals(self.h1, self.h2, self.constant)

        if mapping == "jordan-wigner":
            H_q = jordan_wigner(H_f)
        elif mapping == "bravyi-kitaev":
            H_q = bravyi_kitaev(H_f)
        elif mapping == "parity":
            H_q = parity_mapping(H_f, n_modes=self.M)
        else:
            raise ValueError(f"Unknown mapping {mapping}")

        n_qubits = self.M
        terms = _qubit_op_to_terms(H_q, n_qubits)

        return QElectronicHamiltonian(
            qubit_terms=terms,
            n_qubits=n_qubits,
            mapping=mapping,
            sparse=sparse,
            origin={"source": "ElectronicHamiltonian"},
        )

    def terms(self) -> List[Dict[str, Any]]:
        """Return the additive decomposition of the fermionic Hamiltonian.

        Returns:
            List[Dict[str, Any]]:
                Entries containing ``coef``, a dense matrix, the ``kind`` of contribution, and the tensor indices involved.

        Notes:
            The routine scales exponentially with the number of orbitals,
            matching [`to_matrix`][trotvol.qm.hamiltonians.electronic.ElectronicHamiltonian.to_matrix].
        """
        M = self.M
        dim = 1 << M
        out: List[Dict[str, Any]] = []

        # 1-body terms: h1[p,q] a_p^† a_q
        for p in range(M):
            for q in range(M):
                c = self.h1[p, q]
                if abs(c) < 1e-12:
                    continue
                block = _one_body_term_matrix(M, p, q, c)
                out.append(
                    {
                        "coef": c,
                        "matrix": block,
                        "kind": "1-body",
                        "indices": (p, q),
                    }
                )

        # 2-body terms: (1/2) h2[p,q,r,s] a_p^† a_q^† a_r a_s
        for p in range(M):
            for q in range(M):
                for r in range(M):
                    for s in range(M):
                        c = 0.5 * self.h2[p, q, r, s]
                        if abs(c) < 1e-12:
                            continue
                        block = _two_body_term_matrix(M, p, q, r, s, c)
                        out.append(
                            {
                                "coef": c,
                                "matrix": block,
                                "kind": "2-body",
                                "indices": (p, q, r, s),
                            }
                        )

        # constant shift: constant * I
        if abs(self.constant) > 1e-12:
            block = np.eye(dim, dtype=complex) * self.constant
            out.append(
                {
                    "coef": self.constant,
                    "matrix": block,
                    "kind": "constant",
                    "indices": (),
                }
            )

        return out

    @classmethod
    def from_xyz(
        cls,
        xyz_source: Union[str, Sequence[str]],
        basis: str = "sto-3g",
        charge: int = 0,
        spin_mult: int = 1,
        use_gpu4pyscf: bool = False,
        freeze_core: bool = False,
        n_core_keep: int = 4,
        n_virt_keep: int = 4,
    ) -> "ElectronicHamiltonian":
        """
        If freeze_core is False:
            build full Hamiltonian (all orbitals).
        If freeze_core is True:
            build an active-space Hamiltonian with only selected valence+low virtual
            and attach metadata listing which orbitals were kept/frozen.
        """
        geom_str = _parse_xyz(xyz_source)

        mf, mol = _scf_with_optional_external(
            geom_str=geom_str,
            basis=basis,
            charge=charge,
            spin_mult=spin_mult,
            external_potential=None,
            use_gpu4pyscf=use_gpu4pyscf,
        )

        if not freeze_core:
            h1, h2, constant = _integrals_from_mf(mf, mol)
            meta = {
                "active_indices": list(range(h1.shape[0] // 2)),  # spatial MOs ~ all of them
                "frozen_indices": [],
                "active_occ": [],
                "active_virt": [],
                "orbital_energies": getattr(mf, "mo_energy").tolist(),
                "orbital_occupations": getattr(mf, "mo_occ").tolist(),
                "n_act_spatial": h1.shape[0] // 2,
                "n_spin_orb": h1.shape[0],
                "n_core_keep": None,
                "n_virt_keep": None,
                "note": "full-space (no freezing)",
            }
            return cls(
                h1=h1,
                h2=h2,
                constant=constant,
                meta=meta,
            )

        # freeze_core=True path
        h1_act, h2_act, const_act, meta_act = _active_space_from_mf(
            mf,
            n_core_keep=n_core_keep,
            n_virt_keep=n_virt_keep,
        )

        meta_act["note"] = "active-space / frozen-core approximation"

        return cls(
            h1=h1_act,
            h2=h2_act,
            constant=const_act,
            meta=meta_act,
        )


class QElectronicHamiltonian(PauliSumHamiltonian):
    """Electronic Hamiltonian expressed as a Pauli sum.

    Args:
        qubit_terms (Sequence[Dict[str, Any]]): Terms with “coeff” and “pauli” keys.
        n_qubits (int): Number of qubits.
        mapping (str): Label of the fermion-to-qubit mapping used.
        sparse (bool): If True store data in sparse format.
        origin (Dict[str, Any] | None): Optional provenance metadata.
    """

    def __init__(
        self,
        qubit_terms: Sequence[Dict[str, Any]],
        n_qubits: int,
        mapping: str,
        sparse: bool = False,
        origin: Optional[Dict[str, Any]] = None,
    ):
        coeffs = [complex(t["coeff"]) for t in qubit_terms]
        pauli_strings = [t["pauli"].replace(" ", "") for t in qubit_terms]

        super().__init__(
            n_qubits=n_qubits,
            pauli_strings=pauli_strings,
            coeffs=coeffs,
            sparse=sparse,
        )

        self.mapping = mapping
        self.origin = origin or {}

    def decompose(self) -> Dict[str, Any]:
        base = super().decompose()
        base["type"] = "qubit_pauli_sum"
        base["mapping"] = self.mapping
        base["origin"] = self.origin
        return base


class QMMMHamiltonian(ElectronicHamiltonian):
    """Electronic Hamiltonian embedded in a classical environment.

    Args:
        h1 (np.ndarray): One-body integrals for the embedded system.
        h2 (np.ndarray): Two-body integrals for the embedded system.
        constant (float): Energy offset.
        mapping (str): Default fermion-to-qubit mapping tag.
        mm_env (Dict[str, Any]): Metadata describing the MM environment.
    """

    def __init__(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        constant: float,
        mm_env: Dict[str, Any],
    ):
        super().__init__(h1=h1, h2=h2, constant=constant)
        self.mm_env = mm_env  # e.g. {"type": "point_charges", "charges": ...}

    def decompose(self) -> Dict[str, Any]:
        """Extend [`decompose`][trotvol.qm.hamiltonians.electronic.ElectronicHamiltonian.decompose] with MM metadata.

        Returns:
            Dict[str, Any]: Metadata including ``mm_env``.
        """
        base = super().decompose()
        base["mm_env"] = self.mm_env
        base["type"] = "qmmm_second_quantized"
        return base

    @classmethod
    def from_xyz_with_env(
        cls,
        xyz_source: Union[str, Sequence[str]],
        mm_external: Union[
            Callable[[np.ndarray], np.ndarray],
            Sequence[Sequence[float]],
            np.ndarray,
        ],
        basis: str = "sto-3g",
        charge: int = 0,
        spin_mult: int = 1,
        use_gpu4pyscf: bool = False,
    ) -> "QMMMHamiltonian":
        """Construct a QM/MM Hamiltonian directly from XYZ data.

        Args:
            xyz_source (Union[str, Sequence[str]]): XYZ path, string, or lines.
            mm_external (Callable | Sequence | np.ndarray): Either a callable
                returning a potential ``V_ext(coords)`` or an ``(N, 4)`` array
                of point charges ``[x, y, z, q]``.
            basis (str): Basis set keyword for PySCF.
            charge (int): Molecular charge.
            spin_mult (int): Spin multiplicity ``2S + 1``.
            use_gpu4pyscf (bool): Attempt to use ``gpu4pyscf`` when available.

        Returns:
            QMMMHamiltonian: Embedded Hamiltonian with recorded MM metadata.
        """
        geom_str = _parse_xyz(xyz_source)

        mf, mol = _scf_with_optional_external(
            geom_str=geom_str,
            basis=basis,
            charge=charge,
            spin_mult=spin_mult,
            external_potential=mm_external,
            use_gpu4pyscf=use_gpu4pyscf,
        )

        h1, h2, constant = _integrals_from_mf(mf, mol)

        # record environment metadata for reproducibility
        if callable(mm_external):
            mm_env = {"type": "external_potential_fn"}
        else:
            mm_env = {
                "type": "point_charges",
                "charges": np.asarray(mm_external),
            }

        return cls(
            h1=h1,
            h2=h2,
            constant=constant,
            mm_env=mm_env,
        )

    def terms(self) -> List[Dict[str, Any]]:
        """Return additive terms while annotating them with ``mm_env`` info.

        Returns:
            List[Dict[str, Any]]:
                Same structure as [`terms`][trotvol.qm.hamiltonians.electronic.ElectronicHamiltonian.terms] but each term carries ``mm_env`` metadata for provenance.
        """
        base_terms = super().terms()
        # attach mm_env metadata for traceability
        for t in base_terms:
            t["mm_env"] = self.mm_env
        return base_terms
