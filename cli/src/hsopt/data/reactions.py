"""Generate reaction Hamiltonians from fixed geometries, without legacy checkouts.

PySCF supplies RHF frontier orbitals and frozen-core CAS integrals. OpenFermion
performs the Jordan--Wigner transform; dense references are built separately by
the existing baseline-calibration workflow.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

CONVENTIONS = ("physical", "historical-replay")
RECIPE = {
    "geometry_unit": "Angstrom",
    "mean_field": "RHF",
    "active_space_selection": "frontier_rhf_orbitals",
    "localization": "pipek_mezey_split_occ_vir",
    "population_method": "meta_lowdin",
    "scf_conv_tol": 1e-9,
    "scf_max_cycle": 50,
    "localization_conv_tol": 1e-6,
    "localization_max_cycle": 100,
    "localization_max_iters": 20,
    "localization_exponent": 2,
    "spin_orbital_order": "interleaved_alpha_beta",
    "pauli_label_order": "qubit_0_leftmost",
    "integral_cutoff": 1e-14,
    "pauli_cutoff": 1e-12,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_definitions(species_path: Path, reactions_path: Path) -> tuple[dict, dict]:
    """Require one consistent fixed species definition per stoichiometric term."""
    definitions = json.loads(species_path.read_text())
    reactions = json.loads(reactions_path.read_text())
    if definitions.get("schema") != 1 or definitions.get("recipe") != RECIPE:
        raise ValueError("unsupported species schema or chemistry recipe")
    keys = reactions["reaction_keys"]
    if len(set(keys)) != len(keys) or set(keys) != set(reactions["reactions"]):
        raise ValueError("reaction ordering and definitions disagree")
    required = set()
    for key in keys:
        row = reactions["reactions"][key]
        terms = row["terms"]
        if not terms or any(
            not np.isfinite(term["coefficient"]) or term["coefficient"] == 0 for term in terms
        ):
            raise ValueError(f"{key}: invalid stoichiometry")
        term_species = [term["species"] for term in terms]
        if len(set(term_species)) != len(term_species) or set(term_species) != set(row["species"]):
            raise ValueError(f"{key}: inconsistent species registry")
        required.update(term_species)
    if set(definitions["species"]) != required:
        raise ValueError("species definitions must exactly cover the reaction registry")
    for key, spec in definitions["species"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"invalid species key: {key}")
        for field in ("charge", "multiplicity", "active_electrons", "active_orbitals"):
            if type(spec[field]) is not int:
                raise ValueError(f"{key}: {field} must be an integer")
        electrons, orbitals = spec["active_electrons"], spec["active_orbitals"]
        if spec["multiplicity"] != 1 or electrons <= 0 or electrons % 2:
            raise ValueError(
                f"{key}: closed-shell RHF requires a singlet and even active electrons"
            )
        if orbitals <= 0 or electrons > 2 * orbitals:
            raise ValueError(f"{key}: active space cannot contain its electrons")
        if (
            not spec["geometry"].strip()
            or spec["basis"] != "sto-3g"
            or spec["localization"] != RECIPE["localization"]
        ):
            raise ValueError(f"{key}: unsupported fixed species definition")
    return definitions, reactions


def active_space_integrals(spec: dict, *, threads: int = 1) -> dict:
    """RHF -> frontier occupied/virtual blocks -> separate PM -> frozen-core CAS."""
    try:
        from pyscf import ao2mo, gto, lib, lo, mcscf, scf
    except ImportError as error:
        raise RuntimeError(
            "install chemistry dependencies with uv sync --extra chemistry"
        ) from error
    if threads < 1:
        raise ValueError("threads must be positive")
    previous_threads = lib.num_threads()
    lib.num_threads(threads)
    try:
        mol = gto.M(
            atom=spec["geometry"],
            unit=RECIPE["geometry_unit"],
            basis=spec["basis"],
            charge=spec["charge"],
            spin=spec["multiplicity"] - 1,
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = RECIPE["scf_conv_tol"]
        mf.max_cycle = RECIPE["scf_max_cycle"]
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF did not converge")
        electrons, orbitals = spec["active_electrons"], spec["active_orbitals"]
        n_core = (mol.nelectron - electrons) // 2
        n_occ = electrons // 2
        n_vir = orbitals - n_occ
        if (
            mol.nelectron % 2
            or n_core < 0
            or n_core + orbitals > mf.mo_coeff.shape[1]
            or np.count_nonzero(mf.mo_occ > 0) != n_core + n_occ
        ):
            raise ValueError("active space is incompatible with the RHF orbitals")
        occupied = mf.mo_coeff[:, n_core : n_core + n_occ]
        virtual = mf.mo_coeff[:, n_core + n_occ : n_core + orbitals]

        def localize(block: np.ndarray) -> np.ndarray:
            if block.shape[1] <= 1:
                return block
            pm = lo.PM(mol, block, pop_method=RECIPE["population_method"])
            pm.conv_tol = RECIPE["localization_conv_tol"]
            pm.conv_tol_grad = None  # PySCF uses sqrt(0.1 * conv_tol).
            pm.max_cycle = RECIPE["localization_max_cycle"]
            pm.max_iters = RECIPE["localization_max_iters"]
            pm.exponent = RECIPE["localization_exponent"]
            return pm.kernel()

        occupied, virtual = localize(occupied), localize(virtual)
        active = np.hstack([occupied, virtual]) if n_vir else occupied
        mo_final = np.hstack([mf.mo_coeff[:, :n_core], active, mf.mo_coeff[:, n_core + orbitals :]])
        mc = mcscf.CASCI(mf, orbitals, electrons)
        mc.mo_coeff, mc.ncore = mo_final, n_core
        one_body, constant = mc.get_h1eff()
        two_body = mc.get_h2eff()
        if two_body.ndim != 4:
            two_body = ao2mo.restore(1, two_body, orbitals)
        return {
            "one_body": np.asarray(one_body, dtype=np.float64),
            "two_body": np.asarray(two_body, dtype=np.float64),
            "constant": float(constant),
            "mo_active": np.asarray(active, dtype=np.float64),
            "frozen_core_orbitals": int(n_core),
            "rhf_energy_hartree": float(mf.e_tot),
        }
    finally:
        lib.num_threads(previous_threads)


def spatial_to_spin_integrals(
    one_body: np.ndarray, two_body: np.ndarray, *, convention: str = "physical"
) -> tuple[np.ndarray, np.ndarray]:
    """Convert chemist ERIs to coefficients of a†_p a†_q a_r a_s.

    The physical interaction is +1/2 (pr|qs) a†_p a†_q a_s a_r. Exchanging
    annihilators for OpenFermion's ordering contributes the minus sign. The
    archived generator instead used +(pr|qs), giving -2 times that interaction.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown Hamiltonian convention: {convention}")
    one_body, two_body = np.asarray(one_body), np.asarray(two_body)
    n = one_body.shape[0]
    if one_body.shape != (n, n) or two_body.shape != (n, n, n, n):
        raise ValueError("inconsistent spatial-integral dimensions")
    if not np.isfinite(one_body).all() or not np.isfinite(two_body).all():
        raise ValueError("integrals must be finite")
    one_spin = np.zeros((2 * n, 2 * n), dtype=np.float64)
    two_spin = np.zeros((2 * n,) * 4, dtype=np.float64)
    cutoff = RECIPE["integral_cutoff"]
    one = np.where(np.abs(one_body) > cutoff, one_body, 0.0)
    eri = np.where(np.abs(two_body) > cutoff, two_body, 0.0)
    # Apply the threshold to original integrals, before the physical factor.
    coefficient = (-0.5 if convention == "physical" else 1.0) * eri.transpose(0, 2, 1, 3)
    for spin_p in range(2):
        one_spin[spin_p::2, spin_p::2] = one
        for spin_q in range(2):
            two_spin[spin_p::2, spin_q::2, spin_p::2, spin_q::2] = coefficient
    return one_spin, two_spin


def pauli_decomposition(integrals: dict, *, convention: str = "physical") -> tuple[float, list]:
    """Return lexicographic Pauli labels with qubit zero at the left."""
    try:
        from openfermion import InteractionOperator, jordan_wigner
    except ImportError as error:
        raise RuntimeError(
            "install chemistry dependencies with uv sync --extra chemistry"
        ) from error
    one, two = spatial_to_spin_integrals(
        integrals["one_body"], integrals["two_body"], convention=convention
    )
    operator = jordan_wigner(InteractionOperator(float(integrals["constant"]), one, two))
    identity, terms = 0.0, []
    for term, value in operator.terms.items():
        if not np.isfinite(value) or abs(complex(value).imag) > 1e-10:
            raise ValueError(f"invalid JW coefficient for {term}: {value}")
        coefficient = float(np.real(value))
        if abs(coefficient) <= RECIPE["pauli_cutoff"]:
            continue
        if not term:
            identity = coefficient
            continue
        label = ["I"] * one.shape[0]
        for index, pauli in term:
            label[index] = pauli
        terms.append({"label": "".join(label), "coefficient": coefficient})
    return identity, sorted(terms, key=lambda item: item["label"])


def generate_reactions(
    species_path: Path,
    reactions_path: Path,
    output_root: Path,
    *,
    convention: str = "physical",
    species: list[str] | None = None,
    threads: int = 1,
) -> dict:
    """Publish a fresh, checksummed cache atomically; never alter existing inputs."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown Hamiltonian convention: {convention}")
    if threads < 1:
        raise ValueError("threads must be positive")
    definitions, reactions = load_definitions(species_path, reactions_path)
    selected = sorted(definitions["species"] if species is None else set(species))
    if not selected or set(selected) - set(definitions["species"]):
        raise ValueError("select one or more species from the fixed registry")
    output_root = output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"use a fresh output directory: {output_root}")
    output_root = output_root.resolve()
    source_hashes = {
        "species_sha256": _sha256(species_path),
        "reactions_sha256": _sha256(reactions_path),
        "generator_sha256": _sha256(Path(__file__)),
    }
    if convention == "historical-replay":
        print(
            "Historical replay uses the archived, incorrect two-electron conversion.",
            file=sys.stderr,
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary:
        staging = Path(temporary) / "generated"
        cache = staging / "problem_cache"
        cache.mkdir(parents=True)
        metadata, checksums = {}, {}
        for key in selected:
            print(f"Generating {key} ({convention})", file=sys.stderr, flush=True)
            spec = definitions["species"][key]
            integrals = active_space_integrals(spec, threads=threads)
            identity, terms = pauli_decomposition(integrals, convention=convention)
            norm = float(sum(abs(term["coefficient"]) for term in terms))
            if norm <= 0:
                raise ValueError(f"{key}: empty nonidentity Hamiltonian")
            row = {
                **spec,
                "schema": 1,
                "molecule": key,
                "geometry_unit": RECIPE["geometry_unit"],
                "hamiltonian_convention": convention,
                "recipe": RECIPE,
                "source": source_hashes,
                "cas_label": f"CAS({spec['active_electrons']}e,{spec['active_orbitals']}o)",
                "n_qubits": 2 * spec["active_orbitals"],
                "rhf_energy_hartree": integrals["rhf_energy_hartree"],
                "frozen_core_orbitals": integrals["frozen_core_orbitals"],
                "frozen_core_energy_hartree": integrals["constant"],
                "identity_coefficient": identity,
                "nonidentity_pauli_one_norm": norm,
                "total_pauli_rotation_terms": len(terms),
            }
            _write_json(cache / f"{key}.json", row)
            np.savez_compressed(
                cache / f"{key}.npz",
                one_body=integrals["one_body"],
                two_body=integrals["two_body"],
                frozen_core_energy=np.asarray(integrals["constant"]),
                mo_active=integrals["mo_active"],
                pauli_labels=np.asarray([t["label"] for t in terms], dtype=f"<U{row['n_qubits']}"),
                pauli_coefficients=np.asarray([t["coefficient"] for t in terms], dtype=np.float64),
            )
            for suffix in ("json", "npz"):
                name = f"problem_cache/{key}.{suffix}"
                checksums[name] = _sha256(staging / name)
            metadata[key] = {
                field: row[field]
                for field in (
                    "basis",
                    "active_electrons",
                    "active_orbitals",
                    "cas_label",
                    "n_qubits",
                )
            } | {
                "nonidentity_terms": len(terms),
                "metadata_sha256": checksums[f"problem_cache/{key}.json"],
            }
        (staging / "problem_cache.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items()))
        )
        complete = len(selected) == len(definitions["species"])
        provenance = {
            "schema": 1,
            "hamiltonian_convention": convention,
            "source": source_hashes,
            "original_source": definitions["provenance"],
            "recipe": RECIPE,
            "species": selected,
            "complete_reaction_set": complete,
            "reaction_keys": reactions["reaction_keys"] if complete else [],
            "threads": threads,
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pyscf", "openfermion")
            },
            "checksum_manifest_sha256": _sha256(staging / "problem_cache.sha256"),
        }
        _write_json(staging / "provenance.json", provenance)
        _write_json(
            staging / "problem_metadata.json",
            {
                "schema": 2,
                "hamiltonian_convention": convention,
                "source": source_hashes,
                "complete_reaction_set": complete,
                "phase_encoding": {
                    "method": "centered_nonidentity_pauli_one_norm",
                    "dimensionless_evolution_time": 1.0,
                    "physical_evolution_time": "inverse of the nonidentity Pauli one-norm",
                    "energy_recovery": "E_hartree = c0 + lambda * principal_phase",
                },
                "species": metadata,
            },
        )
        staging.rename(output_root)
    return {
        "output_root": str(output_root),
        "species_generated": len(selected),
        "hamiltonian_convention": convention,
        "complete_reaction_set": complete,
        "checksum_manifest": str(output_root / "problem_cache.sha256"),
    }
