from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from pyscf import ao2mo, gto, lo, mcscf, scf


EXAMPLE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXAMPLE_ROOT / "pauli_coefficient_histogram.png"


@dataclass(frozen=True)
class MoleculeSpec:
    label: str
    latex_label: str
    geometry: str
    charge: int = 0
    spin: int = 0


MOLECULES = (
    MoleculeSpec(
        label="LiH",
        latex_label=r"\ce{LiH}",
        geometry="Li 0 0 0; H 0 0 1.595",
    ),
    MoleculeSpec(
        label="BeH2",
        latex_label=r"\ce{BeH2}",
        geometry="H 0 0 -1.326; Be 0 0 0; H 0 0 1.326",
    ),
    MoleculeSpec(
        label="BH3",
        latex_label=r"\ce{BH3}",
        geometry="B 0 0 0; H 1.190 0 0; H -0.595 1.0307 0; H -0.595 -1.0307 0",
    ),
    MoleculeSpec(
        label="CH4",
        latex_label=r"\ce{CH4}",
        geometry=(
            "C 0 0 0; "
            "H 0.629 0.629 0.629; "
            "H -0.629 -0.629 0.629; "
            "H -0.629 0.629 -0.629; "
            "H 0.629 -0.629 -0.629"
        ),
    ),
)


def _pipek_mezey_localize(mol: gto.Mole, coeff: np.ndarray) -> np.ndarray:
    if coeff.shape[1] <= 1:
        return coeff

    localizer = lo.PM(mol, coeff)
    localizer.verbose = 0
    return localizer.kernel(verbose=0)


def _localized_active_mo_coefficients(
    mf: scf.hf.RHF,
    *,
    active_occupied: int,
    active_virtual: int,
) -> tuple[np.ndarray, int, int]:
    occ_indices = [i for i, occ in enumerate(mf.mo_occ) if occ > 1e-8]
    virt_indices = [i for i, occ in enumerate(mf.mo_occ) if occ <= 1e-8]

    if len(occ_indices) < active_occupied:
        raise ValueError(f"requested {active_occupied} occupied orbitals, found {len(occ_indices)}")
    if len(virt_indices) < active_virtual:
        raise ValueError(f"requested {active_virtual} virtual orbitals, found {len(virt_indices)}")

    active_occ = occ_indices[-active_occupied:]
    active_virt = virt_indices[:active_virtual]
    active_set = set(active_occ + active_virt)

    frozen_core = [i for i in occ_indices if i not in active_occ]
    inactive = [
        i
        for i in range(mf.mo_coeff.shape[1])
        if i not in set(frozen_core) and i not in active_set
    ]

    localized_occ = _pipek_mezey_localize(mf.mol, mf.mo_coeff[:, active_occ])
    localized_virt = _pipek_mezey_localize(mf.mol, mf.mo_coeff[:, active_virt])

    mo_coeff = np.hstack(
        [
            mf.mo_coeff[:, frozen_core],
            localized_occ,
            localized_virt,
            mf.mo_coeff[:, inactive],
        ]
    )
    ncas = active_occupied + active_virtual
    nelecas = 2 * active_occupied

    return mo_coeff, ncas, nelecas


def _localized_active_interaction_operator(
    spec: MoleculeSpec,
    *,
    basis: str,
    active_occupied: int,
    active_virtual: int,
) -> tuple[InteractionOperator, dict[str, object]]:
    mol = gto.M(
        atom=spec.geometry,
        basis=basis,
        charge=spec.charge,
        spin=spec.spin,
        unit="Ang",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {spec.label}")

    mo_coeff, ncas, nelecas = _localized_active_mo_coefficients(
        mf,
        active_occupied=active_occupied,
        active_virtual=active_virtual,
    )
    ncore = (mol.nelectron - nelecas) // 2

    mc = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
    mc.verbose = 0
    h1eff, energy_core = mc.get_h1eff(mo_coeff=mo_coeff, ncas=ncas, ncore=ncore)
    h2eff = mc.get_h2eff(mo_coeff=mo_coeff)

    if len(h2eff.shape) != 4:
        h2_chemist = ao2mo.restore(1, h2eff.ravel(), ncas)
    else:
        h2_chemist = h2eff

    h2_physicist = np.transpose(h2_chemist, (0, 2, 1, 3))
    h1_spin, h2_spin = spinorb_from_spatial(h1eff, h2_physicist)

    interaction_operator = InteractionOperator(
        constant=energy_core,
        one_body_tensor=h1_spin,
        two_body_tensor=0.5 * h2_spin,
    )
    metadata = {
        "basis": basis,
        "localization": "pipek_mezey_split_occ_vir",
        "latex_label": spec.latex_label,
        "active_electrons": nelecas,
        "active_orbitals": ncas,
        "n_qubits": h1_spin.shape[0],
        "hf_energy_hartree": float(mf.e_tot),
    }

    return interaction_operator, metadata


def pauli_abs_coefficients(
    spec: MoleculeSpec,
    *,
    basis: str,
    active_occupied: int,
    active_virtual: int,
    include_identity: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    interaction_operator, metadata = _localized_active_interaction_operator(
        spec,
        basis=basis,
        active_occupied=active_occupied,
        active_virtual=active_virtual,
    )
    qubit_operator = jordan_wigner(get_fermion_operator(interaction_operator))

    coeffs = []
    identity_abs = None
    for pauli_term, coeff in qubit_operator.terms.items():
        abs_coeff = abs(complex(coeff))
        if len(pauli_term) == 0:
            identity_abs = abs_coeff
            if not include_identity:
                continue
        coeffs.append(abs_coeff)

    metadata = {
        **metadata,
        "pauli_terms": len(coeffs),
        "identity_abs_coefficient": identity_abs,
    }

    return np.asarray(coeffs, dtype=float), metadata


def plot_histogram(
    coefficient_sets: dict[str, np.ndarray],
    metadata: dict[str, dict[str, object]],
    *,
    output: Path,
    bins: int,
) -> None:
    positive_coeffs = np.concatenate([values[values > 0.0] for values in coefficient_sets.values()])
    bin_edges = np.geomspace(positive_coeffs.min(), positive_coeffs.max(), bins + 1)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[version=4]{mhchem}",
            "font.size": 24,
            "axes.labelsize": 26,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

    for color, (label, coeffs) in zip(colors, coefficient_sets.items()):
        latex_label = metadata[label]["latex_label"]
        ax.hist(
            coeffs,
            bins=bin_edges,
            histtype="stepfilled",
            alpha=0.35,
            linewidth=2.0,
            color=color,
            edgecolor=color,
            label=f"{latex_label} ({metadata[label]['pauli_terms']} terms)",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Pauli coefficient $|c_i|$")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build four localized active-space molecular Hamiltonians, expand them into "
            "Jordan-Wigner Pauli terms, and plot a shared |coefficient| histogram."
        )
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--active-occupied", type=int, default=2)
    parser.add_argument("--active-virtual", type=int, default=2)
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    coefficient_sets: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}

    for spec in MOLECULES:
        coeffs, mol_metadata = pauli_abs_coefficients(
            spec,
            basis=args.basis,
            active_occupied=args.active_occupied,
            active_virtual=args.active_virtual,
            include_identity=args.include_identity,
        )
        coefficient_sets[spec.label] = coeffs
        metadata[spec.label] = mol_metadata
        print(
            f"{spec.label}: {mol_metadata['pauli_terms']} terms, "
            f"{mol_metadata['n_qubits']} qubits, "
            f"{mol_metadata['localization']}, {mol_metadata['basis']}"
        )

    plot_histogram(
        coefficient_sets,
        metadata,
        output=args.output,
        bins=args.bins,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
