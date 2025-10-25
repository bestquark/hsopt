"""Electronic-structure Hamiltonians and mappings to Pauli representations."""

from __future__ import annotations
from typing import Any, Dict, Sequence, Union, Optional, Callable, Tuple, List

import cupynumeric as np
import scipy.sparse as sp
from openfermion import (
    QubitOperator,
    FermionOperator,
    jordan_wigner,
    bravyi_kitaev,
    parity_mapping,
    get_sparse_operator,
)

from eqmmm.qm.hamiltonians._hamiltonian import Hamiltonian
from eqmmm.qm.hamiltonians.pauli import PauliSumHamiltonian


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
    external_potential: Optional[
        Union[
            Callable[[np.ndarray], np.ndarray],
            Sequence[Sequence[float]],
            np.ndarray,
        ]
    ] = None,
    use_gpu4pyscf: bool = False,
):
    """Run a mean-field calculation with optional external potential.

    Args:
        geom_str (str): Semi-colon-separated geometry specification.
        basis (str): Basis set keyword understood by PySCF.
        charge (int): Total molecular charge.
        spin_mult (int): Spin multiplicity ``2S + 1``.
        external_potential (Callable | Sequence | np.ndarray | None): Either a
            callable potential ``V(r)`` evaluated on grid points or an ``(N,4)``
            array of point charges ``[x, y, z, q]``.
        use_gpu4pyscf (bool): If ``True`` prefer ``gpu4pyscf`` SCF kernels.

    Returns:
        Tuple[Any, Any]: ``(mf, mol)`` objects from PySCF/gpu4pyscf.

    Raises:
        ValueError: If the external potential specification is invalid.
        RuntimeError: If the SCF procedure fails to converge.
    """
    # gpu4pyscf path first
    if use_gpu4pyscf:
        try:
            from gpu4pyscf import scf as gpu_scf
            from pyscf import gto, qmmm as pyscf_qmmm, dft as pyscf_dft

            have_gpu = True
        except ImportError:
            have_gpu = False
    else:
        have_gpu = False

    # base pyscf
    from pyscf import gto, scf, qmmm as pyscf_qmmm, dft as pyscf_dft

    # spin multiplicity M = 2S+1 → S=(M-1)/2 → spin = 2S
    S = (spin_mult - 1) / 2.0
    mol_spin = int(round(2 * S))

    mol = gto.Mole()
    mol.build(
        atom=geom_str,
        basis=basis,
        charge=charge,
        spin=mol_spin,  # = #alpha - #beta
        unit="Ang",
    )

    # RHF vs UHF
    if mol.spin == 0:
        mf = gpu_scf.RHF(mol) if have_gpu else scf.RHF(mol)
    else:
        mf = gpu_scf.UHF(mol) if have_gpu else scf.UHF(mol)

    # external classical environment
    if external_potential is not None:
        # point charges case
        if not callable(external_potential):
            charges_arr = np.asarray(external_potential)
            if charges_arr.ndim != 2 or charges_arr.shape[1] != 4:
                raise ValueError("external_potential array must be (N,4): [x,y,z,q]")
            coords = np.array(charges_arr[:, :3])
            vals = np.array(charges_arr[:, 3])
            mf = pyscf_qmmm.mm_charge(mf, coords, vals)

        # callable V(r) case
        else:
            original_get_veff = mf.get_veff

            def get_veff_with_external(mol_=None, dm=None, *args, **kwargs):
                if mol_ is None:
                    mol_ = mf.mol
                if dm is None:
                    dm = mf.make_rdm1()

                veff = original_get_veff(mol_, dm, *args, **kwargs)

                try:
                    if hasattr(mf, "grids"):
                        grids = mf.grids
                    else:
                        grids = pyscf_dft.gen_grid.Grids(mf.mol)
                        grids.build()

                    coords_grid = grids.coords  # (Ng,3)
                    weights = grids.weights  # (Ng,)
                    v_ext_vals = external_potential(coords_grid)  # (Ng,)
                    veff += (v_ext_vals * weights).sum()
                except Exception:
                    pass

                return veff

            mf.get_veff = get_veff_with_external

    # run SCF
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
    """Build an OpenFermion ``FermionOperator`` from spin-orbital data.

    Args:
        h1 (np.ndarray): One-body integrals ``h1[p, q]``.
        h2 (np.ndarray): Two-body integrals ``h2[p, q, r, s]`` before the
            ``1/2`` prefactor.
        constant (float): Scalar energy offset.

    Returns:
        FermionOperator: Second-quantized Hamiltonian suitable for mapping to qubits.
    """
    from openfermion import FermionOperator

    n_orb = h1.shape[0]
    H_f = FermionOperator()

    # one-body terms
    for p in range(n_orb):
        for q in range(n_orb):
            c = h1[p, q]
            if abs(c) > 1e-12:
                H_f += FermionOperator(((p, 1), (q, 0)), c)

    # two-body terms
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    c = 0.5 * h2[p, q, r, s]
                    if abs(c) > 1e-12:
                        H_f += FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), c)

    # constant shift
    if abs(constant) > 1e-12:
        H_f += FermionOperator((), constant)

    return H_f


def _qubit_op_to_matrix_and_terms(
    qubit_op: "QubitOperator",
    n_qubits: int,
    sparse: bool = False,
) -> Tuple[Union[np.ndarray, "sp.spmatrix"], List[Dict[str, Any]]]:
    """Convert an OpenFermion operator into matrix and Pauli metadata.

    Args:
        qubit_op (QubitOperator): Operator expressed as a Pauli sum.
        n_qubits (int): Number of qubits needed for the operator.
        sparse (bool): If ``True`` keep the matrix sparse.

    Returns:
        Tuple[Union[np.ndarray, sp.spmatrix], List[Dict[str, Any]]]: Sparse or dense matrix plus a list of ``{"coeff", "pauli"}`` dictionaries with human-readable strings like ``"X0 Z1"``.
    """

    # build sparse matrix from OpenFermion
    sp_op = get_sparse_operator(qubit_op, n_qubits=n_qubits)

    # convert to dense if requested
    mat = sp_op if sparse else sp_op.toarray()

    # term list
    terms: List[Dict[str, Any]] = []
    for pauli_term, coeff in qubit_op.terms.items():
        if len(pauli_term) == 0:
            pstr = "I"
        else:
            sorted_ops = sorted(pauli_term, key=lambda t: t[0])
            pstr = " ".join(f"{op}{idx}" for idx, op in sorted_ops)
        terms.append({"coeff": complex(coeff), "pauli": pstr})

    return mat, terms


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


# ----------------------- main classes ----------------------- #


class ElectronicHamiltonian(Hamiltonian):
    """Second-quantized electronic Hamiltonian in a spin-orbital basis.

    Args:
        h1 (np.ndarray): One-body integrals ``(M, M)``.
        h2 (np.ndarray): Two-body integrals ``(M, M, M, M)``.
        constant (float): Scalar nuclear-repulsion/shift term.
        mapping (str): Preferred fermion-to-qubit mapping label used by
            [`ElectronicHamiltonian.to_qubits`][].

    Attributes:
        h1 (np.ndarray): Stored copy of the one-body integrals.
        h2 (np.ndarray): Stored copy of the two-body integrals.
        constant (float): Energy shift added to the Hamiltonian.
        mapping (str): Default mapping name used by
            [`ElectronicHamiltonian.to_qubits`][].
        M (int): Number of spin orbitals.
    """

    def __init__(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        constant: float = 0.0,
        mapping: str = "jordan-wigner",
    ):
        self.h1 = np.array(h1, dtype=float)
        self.h2 = np.array(h2, dtype=float)
        self.constant = float(constant)
        self.mapping = mapping  # default suggestion, can be overridden

        M = self.h1.shape[0]
        if self.h1.shape != (M, M):
            raise ValueError("h1 must be (M,M)")
        if self.h2.shape != (M, M, M, M):
            raise ValueError("h2 must be (M,M,M,M)")
        self.M = M
        self._hilbert_dim = 2**M  # spin-orbital occupation basis dim

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
            Dict[str, Any]: JSON-serializable data with tensor shapes and mapping metadata.
        """
        return {
            "type": "second_quantized",
            "one_body": self.h1,
            "two_body": self.h2,
            "constant": self.constant,
            "n_spin_orbs": self.M,
            "mapping": self.mapping,
        }

    def to_qubits(
        self,
        mapping: Optional[str] = None,
        sparse: bool = False,
    ) -> "QElectronicHamiltonian":
        """Map the fermionic Hamiltonian to qubits via the chosen mapping.

        Args:
            mapping (str | None): Mapping name. Defaults to ``self.mapping``.
            sparse (bool): If ``True`` build a sparse Pauli representation.

        Returns:
            QElectronicHamiltonian: Pauli-sum Hamiltonian with metadata about the mapping origin.

        Raises:
            ValueError: If the mapping keyword is unknown.
        """
        use_mapping = mapping or self.mapping

        # 1. Build FermionOperator
        H_f = _fermion_operator_from_integrals(self.h1, self.h2, self.constant)

        # 2. Map to QubitOperator
        if use_mapping == "jordan-wigner":
            H_q = jordan_wigner(H_f)
        elif use_mapping == "bravyi-kitaev":
            H_q = bravyi_kitaev(H_f)
        elif use_mapping == "parity":
            H_q = parity_mapping(H_f)
        else:
            raise ValueError(f"Unknown mapping {use_mapping}")

        n_qubits = self.M

        # 3. Extract Pauli-sum terms (now supports sparse)
        _, terms = _qubit_op_to_matrix_and_terms(H_q, n_qubits, sparse=sparse)

        # 4. Construct QElectronicHamiltonian (sparse/dense internally handled)
        return QElectronicHamiltonian(
            qubit_terms=terms,
            n_qubits=n_qubits,
            mapping=use_mapping,
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
            matching [`ElectronicHamiltonian.to_matrix`][].
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
        mapping: str = "jordan-wigner",
        use_gpu4pyscf: bool = False,
    ) -> "ElectronicHamiltonian":
        """Construct an electronic Hamiltonian directly from XYZ input.

        Args:
            xyz_source (Union[str, Sequence[str]]): XYZ file path, multiline
                string, or sequence of lines.
            basis (str): Basis set keyword for PySCF.
            charge (int): Molecular charge.
            spin_mult (int): Spin multiplicity ``2S + 1``.
            mapping (str): Default mapping tag stored on the object.
            use_gpu4pyscf (bool): If ``True`` attempt to use ``gpu4pyscf``.

        Returns:
            ElectronicHamiltonian: Hamiltonian built from SCF integrals.
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
        h1, h2, constant = _integrals_from_mf(mf, mol)
        return cls(h1=h1, h2=h2, constant=constant, mapping=mapping)


class QElectronicHamiltonian(PauliSumHamiltonian):
    """Electronic Hamiltonian expressed as a Pauli sum.

    Args:
        qubit_terms (Sequence[Dict[str, Any]]): Terms with ``coeff`` and
            ``pauli`` keys as produced by :func:`_qubit_op_to_matrix_and_terms`.
        n_qubits (int): Number of qubits.
        mapping (str): Label of the fermion-to-qubit mapping used.
        sparse (bool): If ``True`` store data in sparse format.
        origin (Dict[str, Any] | None): Optional provenance metadata.
    """

    def __init__(
        self,
        qubit_terms: Sequence[Dict[str, Any]],
        n_qubits: int,
        mapping: str,
        sparse: bool = False,
        origin: Dict[str, Any] | None = None,
    ):
        # Extract coeffs and Pauli strings
        coeffs = [complex(t["coeff"]) for t in qubit_terms]
        pauli_strings = [t["pauli"].replace(" ", "") for t in qubit_terms]

        # Initialize as PauliSumHamiltonian
        super().__init__(
            n_qubits=n_qubits,
            pauli_strings=pauli_strings,
            coeffs=coeffs,
            sparse=sparse,
        )

        self.mapping = mapping
        self.origin = origin or {}

    def decompose(self) -> Dict[str, Any]:
        """Return Pauli metadata augmented with mapping information.

        Returns:
            Dict[str, Any]: Output of [`PauliSumHamiltonian.decompose`][] extended with ``type``, ``mapping``, and ``origin`` fields.
        """
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
        mapping: str,
        mm_env: Dict[str, Any],
    ):
        super().__init__(h1=h1, h2=h2, constant=constant, mapping=mapping)
        self.mm_env = mm_env  # e.g. {"type": "point_charges", "charges": ...}

    def decompose(self) -> Dict[str, Any]:
        """Extend [`ElectronicHamiltonian.decompose`][] with MM metadata.

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
        mapping: str = "jordan-wigner",
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
            mapping (str): Default mapping tag.
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
            mapping=mapping,
            mm_env=mm_env,
        )

    def terms(self) -> List[Dict[str, Any]]:
        """Return additive terms while annotating them with ``mm_env`` info.

        Returns:
            List[Dict[str, Any]]:
                Same structure as [`ElectronicHamiltonian.terms`][] but each term carries ``mm_env`` metadata for provenance.
        """
        base_terms = super().terms()
        # attach mm_env metadata for traceability
        for t in base_terms:
            t["mm_env"] = self.mm_env
        return base_terms
