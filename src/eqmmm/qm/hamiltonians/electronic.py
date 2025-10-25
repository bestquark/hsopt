from __future__ import annotations
from typing import Any, Dict, Sequence, Union, Optional, Callable, Tuple, List
import cupynumeric as np
from eqmmm.qm.hamiltonians._hamiltonian import BaseHamiltonian

from openfermion import (
    QubitOperator,
    FermionOperator,
    jordan_wigner,
    bravyi_kitaev,
    parity_mapping,
    get_sparse_operator,
)


def _parse_xyz(xyz_source: Union[str, Sequence[str]]) -> str:
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
    """
    Return spin-orbital one- and two-electron integrals and constant.
    h1: (n_spin_orb, n_spin_orb)
    h2: (n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)
    constant: float
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
    """
    Build an OpenFermion FermionOperator from spin-orbital integrals h1,h2.
    h1[p,q] a_p^† a_q
    (1/2) h2[p,q,r,s] a_p^† a_q^† a_r a_s
    + constant
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
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Convert an OpenFermion QubitOperator to:
      - dense matrix (2^n x 2^n)
      - list of {"coeff": ..., "pauli": "X0 Z1 ..."}
    """

    # dense matrix
    sparse = get_sparse_operator(qubit_op, n_qubits=n_qubits)
    mat = sparse.toarray()

    # term list
    terms: List[Dict[str, Any]] = []
    for pauli_term, coeff in qubit_op.terms.items():
        # pauli_term is tuple like ((qubit_idx, 'X'), (qubit_idx,'Z'), ...)
        if len(pauli_term) == 0:
            pstr = "I"
        else:
            # we'll write something readable like "X0 Z2 Y5"
            # (order them by qubit index for stability)
            sorted_ops = sorted(pauli_term, key=lambda t: t[0])
            pstr = " ".join(f"{op}{idx}" for idx, op in sorted_ops)
        terms.append({"coeff": complex(coeff), "pauli": pstr})

    return mat, terms


# ----------------------- main classes ----------------------- #


class ElectronicHamiltonian(BaseHamiltonian):
    """
    Second-quantized electronic Hamiltonian in spin-orbital basis.
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
        return self._hilbert_dim

    @property
    def is_stochastic(self) -> bool:
        return False

    def to_matrix(self) -> np.ndarray:
        """
        Return the full fermionic Hamiltonian matrix in the occupation-number
        basis of spin orbitals (dimension 2^M). This is basically FCI H.
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
    ) -> "QElectronicHamiltonian":
        """
        Map this fermionic Hamiltonian to a qubit Hamiltonian using the
        chosen mapping ("jordan-wigner", "bravyi-kitaev", "parity", ...),
        return a QElectronicHamiltonian that lives in qubit/Pauli space.
        """

        use_mapping = mapping or self.mapping

        # 1. build FermionOperator
        H_f = _fermion_operator_from_integrals(self.h1, self.h2, self.constant)

        # 2. map to qubits
        if use_mapping == "jordan-wigner":
            H_q = jordan_wigner(H_f)
        elif use_mapping == "bravyi-kitaev":
            H_q = bravyi_kitaev(H_f)
        elif use_mapping == "parity":
            H_q = parity_mapping(H_f)
        else:
            raise ValueError(f"Unknown mapping {use_mapping}")

        n_qubits = self.M  # spin-orbital count → qubit count

        # 3. turn QubitOperator → dense matrix + term list
        mat, terms = _qubit_op_to_matrix_and_terms(H_q, n_qubits)

        return QElectronicHamiltonian(
            qubit_terms=terms,
            qubit_matrix=mat,
            n_qubits=n_qubits,
            mapping=use_mapping,
        )

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


class QElectronicHamiltonian(BaseHamiltonian):
    """
    Deterministic qubit-space Hamiltonian:
        H = sum_j α_j P_j
    where P_j are n-qubit Pauli strings.

    We store both:
      - `qubit_terms`: [{"coeff": complex, "pauli": "X0 Z1 ..."}, ...]
      - `qubit_matrix`: dense (2^n, 2^n) ndarray
    """

    def __init__(
        self,
        qubit_terms: Sequence[Dict[str, Any]],
        qubit_matrix: np.ndarray,
        n_qubits: int,
        mapping: str,
    ):
        self.qubit_terms = list(qubit_terms)
        self._H = np.array(qubit_matrix, dtype=complex)
        self.n_qubits = int(n_qubits)
        self.mapping = mapping

        dim = 1 << self.n_qubits
        if self._H.shape != (dim, dim):
            raise ValueError("qubit_matrix has wrong shape for n_qubits")

        # optional Hermiticity check
        if not np.allclose(self._H, self._H.conj().T):
            raise ValueError("qubit_matrix must be Hermitian")

    @property
    def hilbert_dim(self) -> int:
        return 2**self.n_qubits

    @property
    def is_stochastic(self) -> bool:
        return False

    def to_matrix(self) -> np.ndarray:
        """
        Dense n-qubit matrix representation (computational basis).
        """
        return self._H

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "qubit_pauli_sum",
            "mapping": self.mapping,
            "n_qubits": self.n_qubits,
            "terms": self.qubit_terms,
        }


class QMMMHamiltonian(ElectronicHamiltonian):
    """
    QM/MM embedded electronic Hamiltonian.

    - Same data model as ElectronicHamiltonian (h1, h2, constant, mapping)
    - Plus mm_env metadata describing the classical environment

    Notes:
    - to_matrix() works the same: builds the FCI-like fermionic Hamiltonian
      for the *embedded* system (i.e. with MM effects already included
      in the mean-field that produced h1/h2).
    - to_qubits() also works the same and returns a QElectronicHamiltonian.
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
        """
        Build QM/MM Hamiltonian from xyz_source plus an MM environment.

        mm_external:
            - callable(coords) -> V_ext(coords)   (continuous embedding)
            - array-like (N,4): [x,y,z,q]         (point charges)
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
