from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, lo, scf
from scipy.optimize import least_squares


EXAMPLE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXAMPLE_ROOT / "hamiltonian_factorization_histograms.pdf"
DEFAULT_TOLERANCE = 1e-12


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


@dataclass(frozen=True)
class ActiveSpaceIntegrals:
    one_body: np.ndarray
    two_body: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class FactorizationCoefficients:
    coefficients: dict[str, np.ndarray]
    counts: dict[str, int]
    thc_relative_error: float
    thc_rank: int


def _positive_abs(values: np.ndarray, *, tolerance: float) -> np.ndarray:
    abs_values = np.abs(np.asarray(values, dtype=np.float64)).ravel()
    return abs_values[abs_values > tolerance]


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


def _localized_active_integrals(
    spec: MoleculeSpec,
    *,
    basis: str,
    active_occupied: int,
    active_virtual: int,
) -> ActiveSpaceIntegrals:
    from pyscf import ao2mo, mcscf

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
    h1eff, _ = mc.get_h1eff(mo_coeff=mo_coeff, ncas=ncas, ncore=ncore)
    h2eff = mc.get_h2eff(mo_coeff=mo_coeff)
    if len(h2eff.shape) != 4:
        h2eff = ao2mo.restore(1, h2eff.ravel(), ncas)

    return ActiveSpaceIntegrals(
        one_body=np.asarray(h1eff, dtype=np.float64),
        two_body=np.asarray(h2eff, dtype=np.float64),
        metadata={
            "basis": basis,
            "localization": "pipek_mezey_split_occ_vir",
            "active_electrons": nelecas,
            "active_orbitals": ncas,
        },
    )


def _pair_matrix(two_body: np.ndarray) -> np.ndarray:
    n_orb = two_body.shape[0]
    matrix = np.asarray(two_body, dtype=np.float64).reshape(n_orb * n_orb, n_orb * n_orb)
    return 0.5 * (matrix + matrix.T)


def _single_factorization(two_body: np.ndarray, *, tolerance: float) -> list[np.ndarray]:
    matrix = _pair_matrix(two_body)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    n_orb = two_body.shape[0]

    factors: list[np.ndarray] = []
    for idx in order:
        eigenvalue = float(eigenvalues[idx])
        if abs(eigenvalue) <= tolerance:
            continue
        factor = np.sqrt(abs(eigenvalue)) * eigenvectors[:, idx].reshape(n_orb, n_orb)
        factors.append(0.5 * (factor + factor.T))
    return factors


def _double_factorization_coefficients(sf_factors: list[np.ndarray], *, tolerance: float) -> np.ndarray:
    eigenvalue_blocks = []
    for factor in sf_factors:
        symmetric_factor = 0.5 * (factor + factor.T)
        eigenvalues = np.linalg.eigvalsh(symmetric_factor)
        eigenvalue_blocks.append(_positive_abs(eigenvalues, tolerance=tolerance))
    if not eigenvalue_blocks:
        return np.array([], dtype=np.float64)
    return np.concatenate(eigenvalue_blocks)


def _initial_thc_orbital_factors(two_body: np.ndarray, rank: int, *, seed: int) -> np.ndarray:
    n_orb = two_body.shape[0]
    matrix = _pair_matrix(two_body)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(np.abs(eigenvalues))[::-1]

    columns = []
    for idx in order:
        pair_vector = eigenvectors[:, idx].reshape(n_orb, n_orb)
        pair_matrix = 0.5 * (pair_vector + pair_vector.T)
        values, vectors = np.linalg.eigh(pair_matrix)
        dominant = int(np.argmax(np.abs(values)))
        column = np.asarray(vectors[:, dominant], dtype=np.float64)
        if np.linalg.norm(column) > DEFAULT_TOLERANCE:
            columns.append(column / np.linalg.norm(column))
        if len(columns) == rank:
            break

    rng = np.random.default_rng(seed)
    while len(columns) < rank:
        column = rng.normal(size=n_orb)
        column_norm = np.linalg.norm(column)
        if column_norm > DEFAULT_TOLERANCE:
            columns.append(column / column_norm)

    return np.column_stack(columns[:rank])


def _normalize_thc_columns(raw: np.ndarray) -> np.ndarray:
    chi = np.asarray(raw, dtype=np.float64).copy()
    norms = np.linalg.norm(chi, axis=0)
    safe_norms = np.where(norms > DEFAULT_TOLERANCE, norms, 1.0)
    return chi / safe_norms


def _thc_basis(chi: np.ndarray) -> np.ndarray:
    n_orb, rank = chi.shape
    return np.einsum("pm,qm->pqm", chi, chi).reshape(n_orb * n_orb, rank)


def _solve_thc_zeta(pair_matrix: np.ndarray, chi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = _thc_basis(chi)
    basis_pinv = np.linalg.pinv(basis, rcond=1e-10)
    zeta = basis_pinv @ pair_matrix @ basis_pinv.T
    return zeta, basis


def _thc_zeta_coefficients(
    two_body: np.ndarray,
    *,
    rank: int,
    max_nfev: int,
    seed: int,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    matrix = _pair_matrix(two_body)
    matrix_norm = max(float(np.linalg.norm(matrix)), tolerance)
    initial_chi = _initial_thc_orbital_factors(two_body, rank, seed=seed)

    def residual(raw: np.ndarray) -> np.ndarray:
        chi = _normalize_thc_columns(raw.reshape(two_body.shape[0], rank))
        zeta, basis = _solve_thc_zeta(matrix, chi)
        return ((basis @ zeta @ basis.T - matrix) / matrix_norm).ravel()

    if max_nfev > 0:
        result = least_squares(
            residual,
            initial_chi.ravel(),
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        chi = _normalize_thc_columns(result.x.reshape(two_body.shape[0], rank))
    else:
        chi = initial_chi

    zeta, basis = _solve_thc_zeta(matrix, chi)
    zeta = 0.5 * (zeta + zeta.T)
    relative_error = float(np.linalg.norm(basis @ zeta @ basis.T - matrix) / matrix_norm)
    unique_zeta = zeta[np.triu_indices_from(zeta)]
    return _positive_abs(unique_zeta, tolerance=tolerance), relative_error


def _latex_scientific(value: float, *, precision: int = 1) -> str:
    if value == 0.0:
        return "0"
    mantissa, exponent = f"{value:.{precision}e}".split("e")
    return rf"{float(mantissa):.{precision}f}\times 10^{{{int(exponent)}}}"


def factorization_coefficients(
    integrals: ActiveSpaceIntegrals,
    *,
    thc_rank: int | None,
    thc_max_nfev: int,
    seed: int,
    include_one_body: bool,
    tolerance: float,
) -> FactorizationCoefficients:
    one_body = _positive_abs(integrals.one_body, tolerance=tolerance)
    two_body = np.asarray(integrals.two_body, dtype=np.float64)
    rank = int(thc_rank or (2 * two_body.shape[0]))

    raw_blocks = [_positive_abs(0.5 * two_body, tolerance=tolerance)]
    if include_one_body:
        raw_blocks.insert(0, one_body)
    raw = np.concatenate(raw_blocks)

    sf_factors = _single_factorization(two_body, tolerance=tolerance)
    sf_blocks = [_positive_abs(np.asarray(sf_factors), tolerance=tolerance)]
    if include_one_body:
        sf_blocks.insert(0, one_body)
    sf = np.concatenate(sf_blocks)

    df_blocks = [_double_factorization_coefficients(sf_factors, tolerance=tolerance)]
    if include_one_body:
        df_blocks.insert(0, one_body)
    df = np.concatenate(df_blocks)

    thc_zeta, thc_relative_error = _thc_zeta_coefficients(
        two_body,
        rank=rank,
        max_nfev=thc_max_nfev,
        seed=seed,
        tolerance=tolerance,
    )
    thc_blocks = [thc_zeta]
    if include_one_body:
        thc_blocks.insert(0, one_body)
    thc = np.concatenate(thc_blocks)

    coefficient_sets = {
        "O(n^4)": raw,
        "SF": sf,
        "DF": df,
        "THC": thc,
    }
    return FactorizationCoefficients(
        coefficients=coefficient_sets,
        counts={name: len(values) for name, values in coefficient_sets.items()},
        thc_relative_error=thc_relative_error,
        thc_rank=rank,
    )


def plot_factorization_histograms(
    all_coefficients: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, FactorizationCoefficients],
    latex_labels: dict[str, str],
    *,
    output: Path,
    bins: int,
) -> None:
    methods = ("O(n^4)", "SF", "DF", "THC")
    colors = ("#1B9E77", "#D95F02", "#7570B3", "#E7298A")

    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[version=4]{mhchem}",
            "font.size": 24,
            "axes.labelsize": 24,
            "axes.titlesize": 26,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(44.0, 9.0), sharey=True, constrained_layout=True)

    titles = {
        "O(n^4)": "Unfactorized",
        "SF": "Single Factorization",
        "DF": "Double Factorization",
        "THC": "THC",
    }

    for ax, method in zip(axes, methods):
        method_values = [
            molecule_data[method]
            for molecule_data in all_coefficients.values()
            if len(molecule_data[method]) > 0
        ]
        joined = np.concatenate(method_values)
        x_min = float(joined.min())
        x_max = float(joined.max())
        if np.isclose(x_min, x_max):
            x_min *= 0.5
            x_max *= 2.0
        else:
            log_min = np.log10(x_min)
            log_max = np.log10(x_max)
            padding = 0.05 * (log_max - log_min)
            x_min = 10 ** (log_min - padding)
            x_max = 10 ** (log_max + padding)
        bin_edges = np.geomspace(x_min, x_max, bins + 1)

        for color, (label, molecule_data) in zip(colors, all_coefficients.items()):
            values = molecule_data[method]
            if len(values) == 0:
                continue
            summary = summaries[label]
            legend_label = rf"{latex_labels[label]} ({summary.counts[method]})"
            if method == "THC":
                epsilon = _latex_scientific(summary.thc_relative_error)
                legend_label += rf", $\epsilon={epsilon}$"
            ax.hist(
                values,
                bins=bin_edges,
                histtype="stepfilled",
                alpha=0.28,
                linewidth=2.4,
                color=color,
                edgecolor=color,
                label=legend_label,
            )
        ax.set_xscale("log")
        ax.set_title(titles[method])
        ax.set_xlabel(r"$|\alpha_i|$")
        if method == "O(n^4)":
            ax.set_ylabel("Count")
        ax.legend(
            frameon=False,
            loc="upper left",
            handlelength=1.4,
            labelspacing=0.35,
            borderaxespad=0.2,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build localized active-space molecular Hamiltonians and compare absolute "
            "coefficient/data-entry histograms for unfactorized, SF, DF, and THC forms."
        )
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--active-occupied", type=int, default=2)
    parser.add_argument("--active-virtual", type=int, default=2)
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--thc-rank", type=int, default=None)
    parser.add_argument("--thc-max-nfev", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--no-one-body", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_coefficients: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, FactorizationCoefficients] = {}
    latex_labels: dict[str, str] = {}

    for molecule_index, spec in enumerate(MOLECULES):
        integrals = _localized_active_integrals(
            spec,
            basis=args.basis,
            active_occupied=args.active_occupied,
            active_virtual=args.active_virtual,
        )
        summary = factorization_coefficients(
            integrals,
            thc_rank=args.thc_rank,
            thc_max_nfev=args.thc_max_nfev,
            seed=args.seed + molecule_index,
            include_one_body=not args.no_one_body,
            tolerance=args.tolerance,
        )
        all_coefficients[spec.label] = summary.coefficients
        summaries[spec.label] = summary
        latex_labels[spec.label] = spec.latex_label
        counts = ", ".join(f"{name}={count}" for name, count in summary.counts.items())
        print(
            f"{spec.label}: {counts}; THC rank={summary.thc_rank}, "
            f"THC relerr={summary.thc_relative_error:.3e}; "
            f"{integrals.metadata['localization']}, {integrals.metadata['basis']}"
        )

    plot_factorization_histograms(
        all_coefficients,
        summaries,
        latex_labels,
        output=args.output,
        bins=args.bins,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
