from __future__ import annotations
from typing import Any, Dict, Sequence, Union, Optional, Callable
import cupynumeric as np
from eqmmm.qm.hamiltonians._hamiltonian import BaseHamiltonian


# ----------------------- helpers ----------------------- #


def _parse_xyz(xyz_source: Union[str, Sequence[str]]) -> str:
    """
    Returns PySCF geometry string "O x y z; H x y z; ..."
    from either a path to .xyz, an xyz string, or a list of lines.
    """
    if isinstance(xyz_source, str):
        # either a filename or inline xyz contents
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
    """
    Build pyscf/gpu4pyscf molecule, run RHF/UHF with optional external potential
    (QMMM point charges or custom V(r)), return (mf, mol).

    external_potential:
        - callable(coords) -> V_ext(coords)
        - array-like of shape (N,4): [x,y,z,q] point charges
        - None
    """
    # Try gpu4pyscf if requested. Fallback to CPU PySCF.
    if use_gpu4pyscf:
        try:
            from gpu4pyscf import scf as gpu_scf
            from pyscf import gto, qmmm as pyscf_qmmm, dft as pyscf_dft

            have_gpu = True
        except ImportError:
            have_gpu = False
    else:
        have_gpu = False

    # Base PySCF imports (always needed)
    from pyscf import gto, scf, qmmm as pyscf_qmmm, dft as pyscf_dft

    # spin multiplicity M = 2S+1 → S=(M-1)/2 → spin = 2S (alpha - beta)
    S = (spin_mult - 1) / 2.0
    mol_spin = int(round(2 * S))

    mol = gto.Mole()
    mol.build(
        atom=geom_str,
        basis=basis,
        charge=charge,
        spin=mol_spin,  # mol.spin = #alpha - #beta
        unit="Ang",
    )

    # choose RHF if spin=0 else UHF; prefer GPU if available
    if mol.spin == 0:
        mf = gpu_scf.RHF(mol) if have_gpu else scf.RHF(mol)
    else:
        mf = gpu_scf.UHF(mol) if have_gpu else scf.UHF(mol)

    # attach external potential if provided
    if external_potential is not None:
        # Case 1: array-like point charges
        if not callable(external_potential):
            charges_arr = np.asarray(external_potential)
            if charges_arr.ndim != 2 or charges_arr.shape[1] != 4:
                raise ValueError("external_potential array must be (N,4): [x,y,z,q]")
            coords = np.array(charges_arr[:, :3])
            vals = np.array(charges_arr[:, 3])
            mf = pyscf_qmmm.mm_charge(mf, coords, vals)

        # Case 2: general callable V(r)
        else:
            original_get_veff = mf.get_veff

            def get_veff_with_external(mol_=None, dm=None, *args, **kwargs):
                """
                Wrap mf.get_veff to include an arbitrary external scalar potential.
                NOTE: This is an approximate projection of V_ext onto AO basis.
                """
                if mol_ is None:
                    mol_ = mf.mol
                if dm is None:
                    dm = mf.make_rdm1()

                veff = original_get_veff(mol_, dm, *args, **kwargs)

                # crude AO projection via numerical grid sampling
                try:
                    if hasattr(mf, "grids"):
                        grids = mf.grids
                    else:
                        grids = pyscf_dft.gen_grid.Grids(mf.mol)
                        grids.build()

                    coords_grid = grids.coords  # (Ng,3)
                    weights = grids.weights  # (Ng,)
                    v_ext_vals = external_potential(coords_grid)  # (Ng,)
                    # Placeholder: just add scalar energy-like shift
                    veff += (v_ext_vals * weights).sum()
                except Exception:
                    # if anything goes wrong, just fall back
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
    Take a converged mf, build spin-orbital h1, h2, and constant.
    Returns (h1, h2, constant).

    h1: (n_spin_orb, n_spin_orb)
    h2: (n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)
    constant: float (nuclear repulsion etc.)
    """
    from pyscf import ao2mo

    e_nuc = mol.energy_nuc()
    h_core_ao = mf.get_hcore()

    # Closed-shell RHF branch
    if mol.spin == 0:
        C = mf.mo_coeff  # (nao, nmo)
        h_core_mo = C.T @ h_core_ao @ C  # (nmo, nmo)
        nmo = C.shape[1]

        # Two-electron ERIs in spatial MO basis
        eri_mo_spatial_packed = ao2mo.kernel(mol, C)
        eri_mo_spatial = ao2mo.restore(1, eri_mo_spatial_packed, nmo)
        # eri_mo_spatial[p,q,r,s] = (pq|rs)

        n_spin_orb = 2 * nmo

        # h1 in spin-orbital basis: block diag(h_core_mo, h_core_mo)
        h1 = np.zeros((n_spin_orb, n_spin_orb))
        h1[0::2, 0::2] = h_core_mo
        h1[1::2, 1::2] = h_core_mo

        # build antisymmetrized h2
        h2 = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        pq_rs = eri_mo_spatial[p, q, r, s]
                        pq_sr = eri_mo_spatial[p, q, s, r]
                        for sp_p in (0, 1):  # spin of p
                            for sp_q in (0, 1):
                                for sp_r in (0, 1):
                                    for sp_s in (0, 1):
                                        P = 2 * p + sp_p
                                        Q = 2 * q + sp_q
                                        R = 2 * r + sp_r
                                        Sidx = 2 * s + sp_s
                                        term = 0.0
                                        # Coulomb term
                                        if sp_p == sp_r and sp_q == sp_s:
                                            term += pq_rs
                                        # Exchange term
                                        if sp_p == sp_s and sp_q == sp_r:
                                            term -= pq_sr
                                        if term != 0.0:
                                            h2[P, Q, R, Sidx] += term

    # Open-shell UHF branch
    else:
        Ca, Cb = mf.mo_coeff  # (nao,nmo_a), (nao,nmo_b)
        h_core_mo_a = Ca.T @ h_core_ao @ Ca
        h_core_mo_b = Cb.T @ h_core_ao @ Cb

        nmo_a = Ca.shape[1]
        nmo_b = Cb.shape[1]
        n_spin_orb = nmo_a + nmo_b

        # pack spin orbitals as [alpha block || beta block]
        h1 = np.zeros((n_spin_orb, n_spin_orb))
        h1[:nmo_a, :nmo_a] = h_core_mo_a
        h1[nmo_a:, nmo_a:] = h_core_mo_b

        eri_aa_packed = ao2mo.kernel(mol, Ca)  # (aa|aa)
        eri_bb_packed = ao2mo.kernel(mol, Cb)  # (bb|bb)
        eri_ab_packed = ao2mo.kernel(mol, (Ca, Cb, Ca, Cb))  # (aa|bb)

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

        # alpha-beta Coulomb only (no exchange across spins)
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


# ----------------------- main classes ----------------------- #


class ElectronicHamiltonian(BaseHamiltonian):
    """
    Second-quantized electronic Hamiltonian in spin-orbital basis:

        H = sum_{p,q} h1[p,q] a_p^† a_q
          + 1/2 sum_{p,q,r,s} h2[p,q,r,s] a_p^† a_q^† a_r a_s
          + constant

    Internals:
      - h1: (n_spin_orb, n_spin_orb)
      - h2: (n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)
      - constant: float (nuclear repulsion + any core energy)
      - mapping: which fermion->qubit map you'll later apply
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
        self.mapping = mapping

        M = self.h1.shape[0]
        if self.h1.shape != (M, M):
            raise ValueError("h1 must be (M,M)")
        if self.h2.shape != (M, M, M, M):
            raise ValueError("h2 must be (M,M,M,M)")
        self.M = M
        self._hilbert_dim = 2**M

    @property
    def hilbert_dim(self) -> int:
        return self._hilbert_dim

    @property
    def is_stochastic(self) -> bool:
        return False

    def to_matrix(self) -> np.ndarray:
        raise NotImplementedError("Map to qubits first (JW/BK/etc.).")

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "second_quantized",
            "one_body": self.h1,
            "two_body": self.h2,
            "constant": self.constant,
            "n_spin_orbs": self.M,
            "mapping": self.mapping,
        }

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
        """
        Build a pure QM Hamiltonian from an .xyz.
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


class QMMMHamiltonian(ElectronicHamiltonian):
    """
    Same as ElectronicHamiltonian but embeds an external MM environment.
    The environment can be:
      - classical point charges (array-like [x,y,z,q])
      - arbitrary external potential callable V(r)
    We store that environment description for reproducibility.
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

        # record environment metadata
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
