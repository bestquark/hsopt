from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

from model_registry import SCRIPT_DIR, MoleculeSpec, get_spec

try:
    from openfermion import InteractionOperator, get_sparse_operator, jordan_wigner

    HAS_OPENFERMION = True
except ImportError:
    HAS_OPENFERMION = False


CACHE_ROOT = SCRIPT_DIR / "problem_cache"
CHEMICAL_ACCURACY_HARTREE = 1e-3


def single_qubit_paulis() -> Dict[str, np.ndarray]:
    return {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }


def jw_creation_operators(n_qubits: int) -> List[np.ndarray]:
    paulis = single_qubit_paulis()
    z_matrix = paulis["Z"]
    identity = paulis["I"]
    raise_op = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)

    operators: List[np.ndarray] = []
    for p in range(n_qubits):
        op = np.array([[1.0]], dtype=np.complex128)
        for site in range(n_qubits):
            if site < p:
                op = np.kron(op, z_matrix)
            elif site == p:
                op = np.kron(op, raise_op)
            else:
                op = np.kron(op, identity)
        operators.append(op)
    return operators


def hartree_fock_state(active_electrons: int, active_orbitals: int) -> np.ndarray:
    n_qubits = 2 * active_orbitals
    creation_ops = jw_creation_operators(n_qubits)
    vacuum = np.zeros(2 ** n_qubits, dtype=np.complex128)
    vacuum[0] = 1.0
    state = vacuum
    for spin_orbital in range(active_electrons):
        state = creation_ops[spin_orbital] @ state
    return state / np.linalg.norm(state)


def build_qubit_hamiltonian(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
) -> np.ndarray:
    n_orbitals = one_body.shape[0]
    n_qubits = 2 * n_orbitals
    creation_ops = jw_creation_operators(n_qubits)
    annihilation_ops = [op.conj().T for op in creation_ops]

    dim = 2 ** n_qubits
    hamiltonian = constant * np.eye(dim, dtype=np.complex128)

    for p in range(n_qubits):
        for q in range(n_qubits):
            orb_p, spin_p = divmod(p, 2)
            orb_q, spin_q = divmod(q, 2)
            if spin_p != spin_q:
                continue
            coefficient = one_body[orb_p, orb_q]
            if abs(coefficient) > 1e-14:
                hamiltonian += coefficient * (creation_ops[p] @ annihilation_ops[q])

    for p in range(n_qubits):
        for q in range(n_qubits):
            if p == q:
                continue
            for r in range(n_qubits):
                for s in range(n_qubits):
                    if r == s:
                        continue
                    orb_p, spin_p = divmod(p, 2)
                    orb_q, spin_q = divmod(q, 2)
                    orb_r, spin_r = divmod(r, 2)
                    orb_s, spin_s = divmod(s, 2)
                    if spin_p != spin_r or spin_q != spin_s:
                        continue
                    coefficient = two_body[orb_p, orb_r, orb_q, orb_s]
                    if abs(coefficient) > 1e-14:
                        hamiltonian += 0.5 * coefficient * (
                            creation_ops[p] @ creation_ops[q] @ annihilation_ops[s] @ annihilation_ops[r]
                        )

    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-10):
        raise ValueError("constructed Hamiltonian is not Hermitian")
    return hamiltonian


def pauli_decomposition(hamiltonian: np.ndarray) -> Tuple[float, List[Dict[str, object]]]:
    paulis = single_qubit_paulis()
    label_order = "IXYZ"
    label_to_matrix = [paulis["I"], paulis["X"], paulis["Y"], paulis["Z"]]

    n_qubits = int(round(math.log2(hamiltonian.shape[0])))
    dim = hamiltonian.shape[0]

    identity_coefficient = 0.0
    terms: List[Dict[str, object]] = []
    for pauli_indices in np.ndindex(*(4,) * n_qubits):
        matrix = np.array([[1.0]], dtype=np.complex128)
        for idx in pauli_indices:
            matrix = np.kron(matrix, label_to_matrix[idx])
        coefficient = np.trace(matrix @ hamiltonian) / float(dim)
        if abs(coefficient) <= 1e-12:
            continue
        label = "".join(label_order[idx] for idx in pauli_indices)
        coefficient_value = float(round(float(np.real_if_close(coefficient)), 15))
        if label == "I" * n_qubits:
            identity_coefficient = coefficient_value
            continue
        terms.append(
            {
                "label": label,
                "coefficient": coefficient_value,
                "support_size": sum(1 for ch in label if ch != "I"),
            }
        )
    return identity_coefficient, terms


def pauli_matrix_from_label(label: str) -> np.ndarray:
    paulis = single_qubit_paulis()
    matrix = np.array([[1.0]], dtype=np.complex128)
    for ch in label:
        matrix = np.kron(matrix, paulis[ch])
    return matrix


def apply_single_qubit_gate_to_state(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def apply_pauli_label_to_state(state: np.ndarray, label: str) -> np.ndarray:
    paulis = single_qubit_paulis()
    n_qubits = len(label)
    updated = np.asarray(state, dtype=np.complex128)
    for qubit, ch in enumerate(label):
        if ch == "I":
            continue
        updated = apply_single_qubit_gate_to_state(updated, paulis[ch], qubit, n_qubits)
    return updated


def apply_exact_pauli_rotation_to_state(state: np.ndarray, label: str, alpha: float) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128)
    rotated = apply_pauli_label_to_state(state, label)
    return math.cos(alpha) * state + 1j * math.sin(alpha) * rotated


def spatial_to_spin_integrals(one_body: np.ndarray, two_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_orbitals = one_body.shape[0]
    n_spin = 2 * n_orbitals

    one_body_spin = np.zeros((n_spin, n_spin), dtype=np.float64)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            coefficient = float(one_body[p, q])
            if abs(coefficient) <= 1e-14:
                continue
            one_body_spin[2 * p, 2 * q] = coefficient
            one_body_spin[2 * p + 1, 2 * q + 1] = coefficient

    two_body_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    coefficient = float(two_body[p, r, q, s])
                    if abs(coefficient) <= 1e-14:
                        continue
                    for spin_p in range(2):
                        for spin_q in range(2):
                            pi = 2 * p + spin_p
                            qi = 2 * q + spin_q
                            ri = 2 * r + spin_p
                            si = 2 * s + spin_q
                            two_body_spin[pi, qi, ri, si] = coefficient

    return one_body_spin, two_body_spin


def qubit_problem_from_integrals(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
) -> Tuple[np.ndarray, float, List[Dict[str, object]]]:
    if not HAS_OPENFERMION:
        raise RuntimeError("OpenFermion is required for the efficient JW build path")

    n_qubits = 2 * one_body.shape[0]
    one_body_spin, two_body_spin = spatial_to_spin_integrals(one_body, two_body)
    interaction = InteractionOperator(
        constant=float(constant),
        one_body_tensor=one_body_spin,
        two_body_tensor=two_body_spin,
    )
    qubit_op = jordan_wigner(interaction)
    hamiltonian = np.asarray(get_sparse_operator(qubit_op).toarray(), dtype=np.complex128)

    identity_coefficient = 0.0
    pauli_terms: List[Dict[str, object]] = []
    for term, coefficient in qubit_op.terms.items():
        coefficient_value = np.real_if_close(coefficient, tol=1_000)
        if isinstance(coefficient_value, complex) and abs(coefficient_value.imag) > 1e-10:
            raise ValueError(f"unexpected complex JW coefficient {coefficient} for term {term}")
        coefficient_float = float(np.real(coefficient_value))
        if abs(coefficient_float) <= 1e-12:
            continue

        label_chars = ["I"] * n_qubits
        for qubit_index, pauli_char in term:
            label_chars[qubit_index] = pauli_char
        label = "".join(label_chars)
        if label == "I" * n_qubits:
            identity_coefficient = coefficient_float
            continue
        pauli_terms.append(
            {
                "label": label,
                "coefficient": coefficient_float,
                "support_size": sum(1 for ch in label if ch != "I"),
            }
        )

    pauli_terms.sort(key=lambda item: item["label"])
    return hamiltonian, identity_coefficient, pauli_terms


def _build_active_space_data(spec: MoleculeSpec) -> Dict[str, object]:
    from pyscf import ao2mo, gto, lo, mcscf, scf
    from pyscf.mcscf import avas

    mol = gto.M(atom=spec.geometry, basis=spec.basis, charge=spec.charge, spin=spec.multiplicity - 1, verbose=0)
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {spec.key}")

    n_active_occ = spec.active_electrons // 2
    n_active_vir = spec.active_orbitals - n_active_occ

    if spec.aolabels:
        nocc = int(np.count_nonzero(mf.mo_occ > 0))
        avas_obj = avas.AVAS(mf, list(spec.aolabels), canonicalize=False)
        _ncas_avas, _nelecas_avas, mo_avas = avas_obj.kernel()
        occ_weights = avas_obj.occ_weights
        vir_weights = avas_obj.vir_weights

        ncas_occ_avas = int((occ_weights >= avas_obj.threshold).sum())
        ncas_vir_avas = int((vir_weights >= avas_obj.threshold).sum())
        n_inactive_occ = nocc - ncas_occ_avas
        if ncas_occ_avas < n_active_occ:
            raise ValueError(
                f"{spec.key} AVAS found only {ncas_occ_avas} occupied active orbitals "
                f"but {n_active_occ} are needed"
            )
        if ncas_vir_avas < n_active_vir:
            raise ValueError(
                f"{spec.key} AVAS found only {ncas_vir_avas} virtual active orbitals "
                f"but {n_active_vir} are needed"
            )

        active_start = n_inactive_occ
        active_occ_block = mo_avas[:, active_start : active_start + ncas_occ_avas]
        active_vir_block = mo_avas[
            :,
            active_start + ncas_occ_avas : active_start + ncas_occ_avas + ncas_vir_avas,
        ]

        active_occ_weights = occ_weights[-ncas_occ_avas:]
        occ_keep_idx = np.argsort(active_occ_weights)[-n_active_occ:]
        active_vir_weights = vir_weights[:ncas_vir_avas]
        vir_keep_idx = np.argsort(active_vir_weights)[-n_active_vir:]

        occ_discard = np.setdiff1d(np.arange(ncas_occ_avas), occ_keep_idx)
        vir_discard = np.setdiff1d(np.arange(ncas_vir_avas), vir_keep_idx)

        mo_inactive_occ = mo_avas[:, :n_inactive_occ]
        mo_inactive_vir = mo_avas[:, active_start + ncas_occ_avas + ncas_vir_avas :]

        mo_core = np.hstack([mo_inactive_occ, active_occ_block[:, occ_discard]]) if occ_discard.size > 0 else mo_inactive_occ
        mo_active_occ = active_occ_block[:, occ_keep_idx]
        mo_active_vir = active_vir_block[:, vir_keep_idx]
        mo_external = (
            np.hstack([active_vir_block[:, vir_discard], mo_inactive_vir]) if vir_discard.size > 0 else mo_inactive_vir
        )
        n_core = mo_core.shape[1]
    else:
        n_core = (mol.nelectron - spec.active_electrons) // 2
        total_orbitals = mf.mo_coeff.shape[1]
        if n_core < 0:
            raise ValueError(f"{spec.key} active electrons imply a negative frozen-core count")
        if n_core + spec.active_orbitals > total_orbitals:
            raise ValueError(
                f"{spec.key} {spec.cas_label} exceeds the available RHF orbital count {total_orbitals}"
            )

        mo_coeff = mf.mo_coeff
        mo_core = mo_coeff[:, :n_core]
        occ_start = n_core
        occ_stop = occ_start + n_active_occ
        vir_stop = n_core + spec.active_orbitals

        mo_active_occ = mo_coeff[:, occ_start:occ_stop]
        mo_active_vir = mo_coeff[:, occ_stop:vir_stop]
        mo_external = mo_coeff[:, vir_stop:]

    if n_active_occ > 1:
        mo_active_occ = lo.PM(mol, mo_active_occ).kernel()
    if n_active_vir > 1:
        mo_active_vir = lo.PM(mol, mo_active_vir).kernel()

    mo_active = np.hstack([mo_active_occ, mo_active_vir]) if n_active_vir > 0 else mo_active_occ
    mo_final = np.hstack([mo_core, mo_active, mo_external])

    mc = mcscf.CASCI(mf, spec.active_orbitals, spec.active_electrons)
    mc.mo_coeff = mo_final
    mc.ncore = n_core

    h1e, e_core = mc.get_h1eff()
    h2e = mc.get_h2eff()
    if getattr(h2e, "ndim", None) != 4:
        h2e = ao2mo.restore(1, h2e, spec.active_orbitals)

    return {
        "mol": mol,
        "mean_field": mf,
        "mo_active": np.asarray(mo_active, dtype=np.float64),
        "mo_active_occ": np.asarray(mo_active_occ, dtype=np.float64),
        "mo_active_vir": np.asarray(mo_active_vir, dtype=np.float64),
        "one_body": np.asarray(h1e, dtype=np.float64),
        "two_body": np.asarray(h2e, dtype=np.float64),
        "constant": float(e_core),
    }


def _active_space_integrals(spec: MoleculeSpec) -> Tuple[np.ndarray, np.ndarray, float]:
    payload = _build_active_space_data(spec)
    return payload["one_body"], payload["two_body"], payload["constant"]


def active_orbitals(molecule: str) -> Dict[str, object]:
    spec = get_spec(molecule)
    payload = _build_active_space_data(spec)
    return {
        "mol": payload["mol"],
        "mo_active": payload["mo_active"],
        "mo_active_occ": payload["mo_active_occ"],
        "mo_active_vir": payload["mo_active_vir"],
        "n_qubits": 2 * spec.active_orbitals,
    }


def cache_paths(spec: MoleculeSpec) -> Tuple[Path, Path]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"{spec.key}.json", CACHE_ROOT / f"{spec.key}.npz"


def build_problem_cache(spec: MoleculeSpec) -> Dict[str, object]:
    one_body, two_body, constant = _active_space_integrals(spec)
    if HAS_OPENFERMION:
        hamiltonian, identity_coefficient, pauli_terms = qubit_problem_from_integrals(one_body, two_body, constant)
    else:
        hamiltonian = build_qubit_hamiltonian(one_body, two_body, constant)
        identity_coefficient, pauli_terms = pauli_decomposition(hamiltonian)
    initial_state = hartree_fock_state(spec.active_electrons, spec.active_orbitals)
    exact_state = expm(1j * hamiltonian * spec.target_time) @ initial_state
    exact_state = exact_state / np.linalg.norm(exact_state)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    ground_state_energy = float(np.real(eigenvalues[0]))
    ground_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)

    identity = np.eye(hamiltonian.shape[0], dtype=np.complex128)
    centered_hamiltonian = hamiltonian - (np.trace(hamiltonian) / hamiltonian.shape[0]) * identity
    centered_norm = float(np.linalg.norm(centered_hamiltonian, ord=2))
    fidelity_floor = 1.0 - (CHEMICAL_ACCURACY_HARTREE / (2.0 * centered_norm)) ** 2
    abs_alphas = [abs(float(term["coefficient"])) * spec.target_time for term in pauli_terms]

    metadata = {
        "molecule": spec.key,
        "display_name": spec.display_name,
        "basis": spec.basis,
        "charge": spec.charge,
        "multiplicity": spec.multiplicity,
        "active_electrons": spec.active_electrons,
        "active_orbitals": spec.active_orbitals,
        "cas_label": spec.cas_label,
        "target_time": spec.target_time,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_fidelity_floor": fidelity_floor,
        "ground_state_energy_hartree": ground_state_energy,
        "centered_hamiltonian_norm": centered_norm,
        "identity_coefficient": identity_coefficient,
        "total_pauli_rotation_terms": len(pauli_terms),
        "max_abs_total_alpha": max(abs_alphas),
        "min_abs_total_alpha": min(abs_alphas),
        "n_qubits": 2 * spec.active_orbitals,
        "localization": spec.localization,
        "geometry": spec.geometry,
    }
    arrays = {
        "one_body": one_body,
        "two_body": two_body,
        "hamiltonian_real": np.real(hamiltonian),
        "hamiltonian_imag": np.imag(hamiltonian),
        "initial_state_real": np.real(initial_state),
        "initial_state_imag": np.imag(initial_state),
        "exact_state_real": np.real(exact_state),
        "exact_state_imag": np.imag(exact_state),
        "ground_state_real": np.real(ground_state),
        "ground_state_imag": np.imag(ground_state),
        "pauli_labels": np.asarray([item["label"] for item in pauli_terms], dtype=f"<U{metadata['n_qubits']}"),
        "pauli_coefficients": np.asarray([item["coefficient"] for item in pauli_terms], dtype=np.float64),
        "pauli_support_sizes": np.asarray([item["support_size"] for item in pauli_terms], dtype=np.int64),
    }
    json_path, npz_path = cache_paths(spec)
    json_path.write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(npz_path, **arrays)
    return {"metadata": metadata, "arrays": arrays}


@functools.lru_cache(maxsize=None)
def load_problem_data(molecule: str) -> Dict[str, object]:
    spec = get_spec(molecule)
    json_path, npz_path = cache_paths(spec)
    rebuild = not json_path.exists() or not npz_path.exists()
    if not rebuild:
        metadata = json.loads(json_path.read_text())
        rebuild = (
            metadata.get("target_time") != spec.target_time
            or metadata.get("active_electrons") != spec.active_electrons
            or metadata.get("active_orbitals") != spec.active_orbitals
            or metadata.get("basis") != spec.basis
            or metadata.get("localization") != spec.localization
            or "ground_state_energy_hartree" not in metadata
        )
    if rebuild:
        payload = build_problem_cache(spec)
        metadata = payload["metadata"]
        arrays = payload["arrays"]
    else:
        arrays = dict(np.load(npz_path, allow_pickle=False))

    hamiltonian = arrays["hamiltonian_real"] + 1j * arrays["hamiltonian_imag"]
    initial_state = arrays["initial_state_real"] + 1j * arrays["initial_state_imag"]
    exact_state = arrays["exact_state_real"] + 1j * arrays["exact_state_imag"]
    ground_state = arrays["ground_state_real"] + 1j * arrays["ground_state_imag"]

    pauli_terms = []
    build_dense_term_matrices = int(metadata["n_qubits"]) <= 8
    for label, coefficient, support_size in zip(
        arrays["pauli_labels"].tolist(),
        arrays["pauli_coefficients"].tolist(),
        arrays["pauli_support_sizes"].tolist(),
    ):
        matrix = pauli_matrix_from_label(str(label)) if build_dense_term_matrices else None
        pauli_terms.append(
            {
                "label": str(label),
                "coefficient": float(coefficient),
                "support_size": int(support_size),
                "matrix": matrix,
            }
        )

    return {
        "metadata": metadata,
        "hamiltonian": np.asarray(hamiltonian, dtype=np.complex128),
        "identity": np.eye(hamiltonian.shape[0], dtype=np.complex128),
        "initial_state": np.asarray(initial_state, dtype=np.complex128),
        "exact_state": np.asarray(exact_state, dtype=np.complex128),
        "ground_state": np.asarray(ground_state, dtype=np.complex128),
        "pauli_terms": pauli_terms,
        "apply_exact_pauli_rotation": apply_exact_pauli_rotation_to_state,
    }
