import cupynumeric as np
from typing import Any, Dict, Sequence, Union
from eqmmm.qm.hamiltonians._hamiltonian import BaseHamiltonian


# ---------- ElectronicHamiltonian with .from_xyz() ----------
class ElectronicHamiltonian(BaseHamiltonian):
    """
    Second-quantized electronic Hamiltonian in spin-orbital basis:

        H = sum_{p,q} h1[p,q] a_p^† a_q
          + 1/2 sum_{p,q,r,s} h2[p,q,r,s] a_p^† a_q^† a_r a_s
          + constant

    Internals:
      - h1: (n_spin_orb, n_spin_orb)
      - h2: (n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)
        antisymmetrized spin-orbital integrals <pq||rs>
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
        raise NotImplementedError(
            "Map to qubits (JW/BK/etc.) before calling to_matrix()."
        )

    def decompose(self) -> Dict[str, Any]:
        return {
            "type": "second_quantized",
            "one_body": self.h1,
            "two_body": self.h2,
            "constant": self.constant,
            "n_spin_orbs": self.M,
            "mapping": self.mapping,
        }

    # -------- FACTORY: build from XYZ via PySCF --------
    @classmethod
    def from_xyz(
        cls,
        xyz_source: Union[str, Sequence[str]],
        basis: str = "sto-3g",
        charge: int = 0,
        spin_mult: int = 1,
        mapping: str = "jordan-wigner",
    ) -> "ElectronicHamiltonian":
        """
        xyz_source:
            - path to .xyz file
            - OR list[str] / str with xyz contents.
          xyz format:
              N
              comment
              El x y y
              ...

        basis: e.g. 'sto-3g', '6-31g*', etc.
        charge: total molecular charge
        spin_mult: 2S+1 (1 -> closed shell singlet, 2 -> doublet, etc.)

        Returns ElectronicHamiltonian in *spin-orbital* basis:
        - h1 (one-electron integrals in MO spin basis)
        - h2 (antisymmetrized two-electron integrals in MO spin basis)
        - constant = nuclear repulsion energy

        Requires pyscf.
        """

        # ---- 1. parse xyz into PySCF geometry string ----
        def _load_xyz_lines(xsrc):
            if isinstance(xsrc, str):
                # could be path or inline block
                if "\n" in xsrc or "\r" in xsrc:
                    return xsrc.strip().splitlines()
                else:
                    # treat as filename
                    with open(xsrc, "r") as f:
                        return f.read().strip().splitlines()
            else:
                # already list-like
                return list(xsrc)

        lines = _load_xyz_lines(xyz_source)
        if len(lines) < 3:
            raise ValueError("XYZ appears too short")

        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError("First line of xyz must be atom count")

        atom_lines = lines[2 : 2 + n_atoms]
        if len(atom_lines) != n_atoms:
            raise ValueError("XYZ atom count mismatch")

        # PySCF geometry string like: "O 0.0 0.0 0.0; H 0.0 0.0 0.96; H ..."
        geom_parts = []
        for L in atom_lines:
            toks = L.split()
            if len(toks) < 4:
                raise ValueError(f"Bad xyz atom line: {L}")
            sym = toks[0]
            x, y, z = toks[1:4]
            geom_parts.append(f"{sym} {x} {y} {z}")
        geom_str = "; ".join(geom_parts)

        # ---- 2. build PySCF molecule and run RHF/UHF ----
        try:
            from pyscf import gto, scf, ao2mo
        except ImportError:
            raise ImportError(
                "pyscf is required for ElectronicHamiltonian.from_xyz(). "
                "pip install pyscf"
            )

        # spin = #alpha - #beta
        # for given multiplicity M = 2S+1 -> S=(M-1)/2 -> spin = 2S
        S = (spin_mult - 1) / 2.0
        mol_spin = int(round(2 * S))

        mol = gto.Mole()
        mol.build(
            atom=geom_str,
            basis=basis,
            charge=charge,
            spin=mol_spin,  # 0 for closed shell
            unit="Ang",
        )

        # choose RHF if spin=0, else UHF
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.kernel()
        if not mf.converged:
            raise RuntimeError("SCF did not converge")

        # nuclear repulsion
        e_nuc = mol.energy_nuc()

        # ---- 3. get 1e and 2e integrals in MO basis (spin-orbital) ----
        # AO integrals:
        h_core_ao = mf.get_hcore()  # one-electron core Hamiltonian in AO basis
        # S_ao = mol.intor("int1e_ovlp")

        # MO coeffs:
        #   RHF: mf.mo_coeff is (nao, nmo)
        #   UHF: tuple (mo_a, mo_b)
        if mol.spin == 0:
            C = mf.mo_coeff  # (nao, nmo)
            # transform 1e integrals to MO spatial-orbital basis:
            h_core_mo = C.T @ h_core_ao @ C  # (nmo, nmo)

            # two-electron (chemist notation (pq|rs)):
            # ao2mo.kernel returns 2e integrals in MO basis, packed
            eri_mo_spatial_packed = ao2mo.kernel(mol, C)
            # unpack to (nmo,nmo,nmo,nmo)
            nmo = C.shape[1]
            eri_mo_spatial = ao2mo.restore(1, eri_mo_spatial_packed, nmo)

            # Now build spin-orbital integrals.
            # We double spatial orbitals: spin-up, spin-down.
            n_spin_orb = 2 * nmo

            # h1 spin-orbital: block diag(h_core_mo, h_core_mo)
            h1 = np.zeros((n_spin_orb, n_spin_orb))
            h1[0::2, 0::2] = h_core_mo
            h1[1::2, 1::2] = h_core_mo

            # For h2 we need antisymmetrized <pq||rs> in spin-orbital basis.
            # Spatial ERI: eri_mo_spatial[p,q,r,s] = (pq|rs)
            # Spin structure:
            # <pσ qτ || rσ' sτ'> =
            #   (pq|rs) * δ_{σ,σ'} δ_{τ,τ'} - (pq|sr) * δ_{σ,τ'} δ_{τ,σ'}
            #
            # We'll build full tensor brute force.
            h2 = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))
            for p in range(nmo):
                for q in range(nmo):
                    for r in range(nmo):
                        for s in range(nmo):
                            (pq_rs) = eri_mo_spatial[p, q, r, s]
                            (pq_sr) = eri_mo_spatial[p, q, s, r]
                            for sp_p in [0, 1]:  # spin of p
                                for sp_q in [0, 1]:  # spin of q
                                    for sp_r in [0, 1]:
                                        for sp_s in [0, 1]:
                                            P = 2 * p + sp_p
                                            Q = 2 * q + sp_q
                                            R = 2 * r + sp_r
                                            Sidx = 2 * s + sp_s
                                            term = 0.0
                                            # Coulomb term (same spin pairs)
                                            if sp_p == sp_r and sp_q == sp_s:
                                                term += pq_rs
                                            # Exchange term (swap r<->s)
                                            if sp_p == sp_s and sp_q == sp_r:
                                                term -= pq_sr
                                            if term != 0.0:
                                                h2[P, Q, R, Sidx] += term

        else:
            # UHF branch: build alpha/beta separately then expand to spin-orbitals.
            Ca, Cb = mf.mo_coeff  # each (nao, nmo_alpha/beta)
            # one-electron:
            h_core_mo_a = Ca.T @ h_core_ao @ Ca
            h_core_mo_b = Cb.T @ h_core_ao @ Cb

            nmo_a = Ca.shape[1]
            nmo_b = Cb.shape[1]
            n_spin_orb = nmo_a + nmo_b

            # pack spin orbitals as [all alpha then all beta]
            h1 = np.zeros((n_spin_orb, n_spin_orb))
            h1[:nmo_a, :nmo_a] = h_core_mo_a
            h1[nmo_a:, nmo_a:] = h_core_mo_b

            # two-electron integrals:
            # alpha-alpha, beta-beta, alpha-beta blocks
            # We'll do the brute-force generalization:
            eri_aa_packed = ao2mo.kernel(mol, Ca)  # (aa|aa)
            eri_bb_packed = ao2mo.kernel(mol, Cb)  # (bb|bb)
            eri_ab_packed = ao2mo.kernel(mol, (Ca, Cb, Ca, Cb))  # (aa|bb)

            eri_aa = ao2mo.restore(1, eri_aa_packed, nmo_a)
            eri_bb = ao2mo.restore(1, eri_bb_packed, nmo_b)
            eri_ab = ao2mo.restore(1, eri_ab_packed, (nmo_a, nmo_b, nmo_a, nmo_b))

            # build h2[P,Q,R,S] in spin-orbital index space (α block then β block).
            h2 = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))

            # αααα and ββββ need antisymmetrization
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

            # αβαβ / βαβα (no exchange between different spins, just Coulomb)
            for p in range(nmo_a):
                for q in range(nmo_b):
                    for r in range(nmo_a):
                        for s in range(nmo_b):
                            coul = eri_ab[p, q, r, s]
                            P = p
                            Q = offset + q
                            R = r
                            Sidx = offset + s
                            # Only Coulomb, no antisymmetrization across spin
                            h2[P, Q, R, Sidx] += coul
                            # and also symmetric block where we permute pairs:
                            # (β α | β α)
                            h2[offset + q, p, offset + s, r] += coul

        # total constant energy offset = nuclear repulsion + mean-field ref energy correction.
        # A standard simple choice:
        constant = e_nuc

        return cls(
            h1=h1,
            h2=h2,
            constant=constant,
            mapping=mapping,
        )
