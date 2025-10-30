"""Lattice-model Hamiltonians (Ising, Heisenberg, Hubbard) on 1D/2D grids."""

from __future__ import annotations
from typing import List, Tuple, Sequence


from trotvol.qm.hamiltonians.pauli import LatticeHamiltonian


def _linear_index_2d(ix: int, iy: int, Lx: int, Ly: int) -> int:
    # row-major
    return iy * Lx + ix


def _neighbor_pairs_1d(Lx: int, boundary: str) -> List[Tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(Lx - 1)]
    if boundary == "periodic" and Lx > 1:
        pairs.append((Lx - 1, 0))
    return pairs


def _neighbor_pairs_2d(Lx: int, Ly: int, boundary: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []

    # horizontal neighbors for each row
    for iy in range(Ly):
        for ix in range(Lx - 1):
            a = _linear_index_2d(ix, iy, Lx, Ly)
            b = _linear_index_2d(ix + 1, iy, Lx, Ly)
            pairs.append((a, b))
        if boundary == "periodic" and Lx > 1:
            a = _linear_index_2d(Lx - 1, iy, Lx, Ly)
            b = _linear_index_2d(0, iy, Lx, Ly)
            pairs.append((a, b))

    # vertical neighbors for each col
    for ix in range(Lx):
        for iy in range(Ly - 1):
            a = _linear_index_2d(ix, iy, Lx, Ly)
            b = _linear_index_2d(ix, iy + 1, Lx, Ly)
            pairs.append((a, b))
        if boundary == "periodic" and Ly > 1:
            a = _linear_index_2d(ix, Ly - 1, Lx, Ly)
            b = _linear_index_2d(ix, 0, Lx, Ly)
            pairs.append((a, b))

    # deduplicate undirected edges
    uniq = []
    seen = set()
    for i, j in pairs:
        key = (i, j) if i < j else (j, i)
        if key not in seen:
            uniq.append((i, j))
            seen.add(key)
    return uniq


# ---------- Spin models: Ising / Heisenberg ---------- #


def _ising_terms(
    n_sites: int,
    J: float,
    h: float,
    neighbor_pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[str], List[complex]]:
    r"""Return Pauli data for the transverse-field Ising model:

    $$H = -J \sum_{\langle ij\rangle} Z_i Z_j - h \sum_i X_i.$$
    """
    paulis: List[str] = []
    coeffs: List[complex] = []

    # -J Z_i Z_j
    for i, j in neighbor_pairs:
        paulis.append(f"Z{i} Z{j}")
        coeffs.append(-J)

    # -h X_i
    for i in range(n_sites):
        paulis.append(f"X{i}")
        coeffs.append(-h)

    return paulis, coeffs


def _heisenberg_terms(
    Jx: float,
    Jy: float,
    Jz: float,
    neighbor_pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[str], List[complex]]:
    r"""Return Pauli data for the anisotropic Heisenberg model:

    $$H = \sum_{\langle ij\rangle} \bigl(J_x S_i^x S_j^x + J_y S_i^y S_j^y + J_z S_i^z S_j^z\bigr).$$

    Here $S^\alpha = \tfrac{1}{2}\sigma^\alpha$, so $S_i^x S_j^x = \tfrac{1}{4} X_i X_j$, etc.
    """
    paulis: List[str] = []
    coeffs: List[complex] = []

    for i, j in neighbor_pairs:
        if abs(Jx) > 1e-12:
            paulis.append(f"X{i} X{j}")
            coeffs.append(Jx * 0.25)
        if abs(Jy) > 1e-12:
            paulis.append(f"Y{i} Y{j}")
            coeffs.append(Jy * 0.25)
        if abs(Jz) > 1e-12:
            paulis.append(f"Z{i} Z{j}")
            coeffs.append(Jz * 0.25)

    return paulis, coeffs


# ---------- Jordan–Wigner helpers for Hubbard ---------- #


def _jw_hop_terms(p: int, q: int) -> List[Tuple[str, complex]]:
    r"""
    JW for c†_p c_q + c†_q c_p.
    For p < q:

      c†_p c_q + c†_q c_p
        = 0.5 * [ X_p Z_{p+1} ... Z_{q-1} X_q
                + Y_p Z_{p+1} ... Z_{q-1} Y_q ]

    We'll return sparse strings like "X5 Z6 Z7 X9".
    """
    if p == q:
        return []

    if p > q:
        p, q = q, p

    middle = [f"Z{t}" for t in range(p + 1, q)]
    chain_X = " ".join([f"X{p}", *middle, f"X{q}"])
    chain_Y = " ".join([f"Y{p}", *middle, f"Y{q}"])

    return [
        (chain_X, 0.5),
        (chain_Y, 0.5),
    ]


def _hubbard_terms(
    L_sites: int,
    t_hop: float,
    U_onsite: float,
    neighbor_pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[str], List[complex]]:
    r"""Return Pauli data for the single-band Hubbard model:

    $$H = -t \sum_{\langle ij\rangle,\sigma} \left(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\right) + U \sum_i n_{i,\mathrm{up}} n_{i,\mathrm{dn}}.$$

    Mapping:
      site $i$ spin-up $\rightarrow$ qubit $2i$
      site $i$ spin-down $\rightarrow$ qubit $2i+1$

    $n_{i,\sigma} = \tfrac{1 - Z_k}{2}$ and $n_{i,\mathrm{up}} n_{i,\mathrm{dn}} = \tfrac{1}{4}\left(1 - Z_{\text{up}} - Z_{\text{dn}} + Z_{\text{up}} Z_{\text{dn}}\right)$.
    """
    paulis: List[str] = []
    coeffs: List[complex] = []

    # hopping
    for i, j in neighbor_pairs:
        for spin_offset in (0, 1):  # 0=↑,1=↓
            p = 2 * i + spin_offset
            q = 2 * j + spin_offset
            jw_terms = _jw_hop_terms(p, q)
            for pstr, base_coef in jw_terms:
                paulis.append(pstr)
                coeffs.append(-t_hop * base_coef)

    # onsite U
    for i in range(L_sites):
        up = 2 * i
        dn = 2 * i + 1

        # + (U/4) Z_up Z_dn
        paulis.append(f"Z{up} Z{dn}")
        coeffs.append(U_onsite * 0.25)

        # - (U/4) Z_up
        paulis.append(f"Z{up}")
        coeffs.append(-U_onsite * 0.25)

        # - (U/4) Z_dn
        paulis.append(f"Z{dn}")
        coeffs.append(-U_onsite * 0.25)

        # + (U/4) I
        paulis.append("I")
        coeffs.append(U_onsite * 0.25)

    return paulis, coeffs


# ---------- Public constructors returning LatticeHamiltonian ---------- #


def ising_1d(
    Lx: int,
    J: float = 1.0,
    h: float = 0.0,
    boundary: str = "open",
    sparse: bool = False,
) -> LatticeHamiltonian:
    r"""Construct a transverse-field Ising chain.

    $$H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i.$$
    """
    pairs = _neighbor_pairs_1d(Lx, boundary=boundary)
    paulis, coeffs = _ising_terms(Lx, J, h, pairs)
    return LatticeHamiltonian(
        n_qubits=Lx,
        pauli_strings=paulis,
        coeffs=coeffs,
        sparse=sparse,
        model="ising",
        dims=(Lx,),
        boundary=boundary,
        params={"J": J, "h": h},
        neighbor_pairs=pairs,
        notes="1D transverse-field Ising",
    )


def ising_2d(
    Lx: int,
    Ly: int,
    J: float = 1.0,
    h: float = 0.0,
    boundary: str = "open",
    sparse: bool = False,
) -> LatticeHamiltonian:
    r"""Construct a 2D transverse-field Ising model on an $L_x \times L_y$ grid.

    $$H = -J \sum_{\langle ij\rangle} Z_i Z_j - h \sum_i X_i.$$
    """
    pairs = _neighbor_pairs_2d(Lx, Ly, boundary=boundary)
    n_sites = Lx * Ly
    paulis, coeffs = _ising_terms(n_sites, J, h, pairs)
    return LatticeHamiltonian(
        n_qubits=n_sites,
        pauli_strings=paulis,
        coeffs=coeffs,
        sparse=sparse,
        model="ising",
        dims=(Lx, Ly),
        boundary=boundary,
        params={"J": J, "h": h},
        neighbor_pairs=pairs,
        notes="2D transverse-field Ising",
    )


def heisenberg_1d(
    Lx: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    boundary: str = "open",
    sparse: bool = False,
) -> LatticeHamiltonian:
    r"""Construct an anisotropic Heisenberg chain.

    $$H = \sum_i \bigl(J_x S_i^x S_{i+1}^x + J_y S_i^y S_{i+1}^y + J_z S_i^z S_{i+1}^z\bigr),$$

    with $S^\alpha = \tfrac{1}{2}\sigma^\alpha$.
    """
    pairs = _neighbor_pairs_1d(Lx, boundary=boundary)
    paulis, coeffs = _heisenberg_terms(Jx, Jy, Jz, pairs)
    return LatticeHamiltonian(
        n_qubits=Lx,
        pauli_strings=paulis,
        coeffs=coeffs,
        sparse=sparse,
        model="heisenberg",
        dims=(Lx,),
        boundary=boundary,
        params={"Jx": Jx, "Jy": Jy, "Jz": Jz},
        neighbor_pairs=pairs,
        notes="1D anisotropic Heisenberg (S=1/2)",
    )


def heisenberg_2d(
    Lx: int,
    Ly: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    boundary: str = "open",
    sparse: bool = False,
) -> LatticeHamiltonian:
    r"""Construct a 2D anisotropic Heisenberg model.

    $$H = \sum_{\langle ij\rangle} \bigl(J_x S_i^x S_j^x + J_y S_i^y S_j^y + J_z S_i^z S_j^z\bigr).$$
    """
    pairs = _neighbor_pairs_2d(Lx, Ly, boundary=boundary)
    n_sites = Lx * Ly
    paulis, coeffs = _heisenberg_terms(Jx, Jy, Jz, pairs)
    return LatticeHamiltonian(
        n_qubits=n_sites,
        pauli_strings=paulis,
        coeffs=coeffs,
        sparse=sparse,
        model="heisenberg",
        dims=(Lx, Ly),
        boundary=boundary,
        params={"Jx": Jx, "Jy": Jy, "Jz": Jz},
        neighbor_pairs=pairs,
        notes="2D anisotropic Heisenberg (S=1/2)",
    )


def hubbard_1d(
    Lx: int,
    t_hop: float = 1.0,
    U_onsite: float = 4.0,
    boundary: str = "open",
    sparse: bool = False,
) -> LatticeHamiltonian:
    r"""Construct a 1D single-band Fermi-Hubbard model.

    $$H = -t \sum_{\langle ij\rangle,\sigma} \left(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\right) + U \sum_i n_{i,\mathrm{up}} n_{i,\mathrm{dn}}.$$

    Jordan–Wigner mapping with two qubits per site: $i_{\mathrm{up}} \mapsto 2i$, $i_{\mathrm{dn}} \mapsto 2i+1$.
    """
    pairs = _neighbor_pairs_1d(Lx, boundary=boundary)
    paulis, coeffs = _hubbard_terms(Lx, t_hop, U_onsite, pairs)
    return LatticeHamiltonian(
        n_qubits=2 * Lx,
        pauli_strings=paulis,
        coeffs=coeffs,
        sparse=sparse,
        model="hubbard",
        dims=(Lx,),
        boundary=boundary,
        params={"t_hop": t_hop, "U_onsite": U_onsite},
        neighbor_pairs=pairs,
        notes="1D spinful Hubbard (JW, 2 qubits/site)",
    )


def hubbard_2d(
    Lx: int,
    Ly: int,
    t_hop: float = 1.0,
    U_onsite: float = 4.0,
    boundary: str = "open",
    sparse: bool = False,
) -> LatticeHamiltonian:
    r"""
    2D single-band Fermi-Hubbard model on an Lx x Ly square lattice.
    JW mapping, 2 qubits per site.
    """
    pairs = _neighbor_pairs_2d(Lx, Ly, boundary=boundary)
    n_sites = Lx * Ly
    paulis, coeffs = _hubbard_terms(n_sites, t_hop, U_onsite, pairs)
    return LatticeHamiltonian(
        n_qubits=2 * n_sites,
        pauli_strings=paulis,
        coeffs=coeffs,
        sparse=sparse,
        model="hubbard",
        dims=(Lx, Ly),
        boundary=boundary,
        params={"t_hop": t_hop, "U_onsite": U_onsite},
        neighbor_pairs=pairs,
        notes="2D spinful Hubbard (JW, 2 qubits/site)",
    )
