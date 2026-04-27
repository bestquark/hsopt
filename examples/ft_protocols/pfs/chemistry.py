from __future__ import annotations

import functools
import importlib.util
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

try:
    from .array_backend import array_backend_name, np as runtime_np
except ImportError:
    from array_backend import array_backend_name, np as runtime_np

try:
    from openfermion import InteractionOperator, get_sparse_operator, jordan_wigner

    HAS_OPENFERMION = True
except ImportError:
    HAS_OPENFERMION = False


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / "problem_cache"
CHEMICAL_ACCURACY_HARTREE = 1e-3
SPECTRAL_PEAK_WEIGHT_THRESHOLD = 1e-10


@dataclass(frozen=True)
class MoleculeSpec:
    key: str
    display_name: str
    geometry: str
    active_electrons: int
    active_orbitals: int
    aolabels: Tuple[str, ...] | None = None
    charge: int = 0
    multiplicity: int = 1
    basis: str = "sto-3g"
    target_time: float = 1.0
    target_iteration: int = 100
    max_parallel: int = 1
    localization: str = "pipek_mezey_split_occ_vir"

    @property
    def cas_label(self) -> str:
        return f"CAS({self.active_electrons}e,{self.active_orbitals}o)"

    @property
    def lane_dir(self) -> Path:
        return SCRIPT_DIR / self.key

    @property
    def editable_script(self) -> Path:
        return self.lane_dir / "simple_pf.py"


MOLECULE_SPECS: Dict[str, MoleculeSpec] = {
    "h2": MoleculeSpec(
        key="h2",
        display_name="H2 PF",
        geometry="""
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
""".strip(),
        active_electrons=2,
        active_orbitals=2,
    ),
    "bh": MoleculeSpec(
        key="bh",
        display_name="BH PF",
        geometry="""
B 0.000000 0.000000 0.000000
H 0.000000 0.000000 1.230000
""".strip(),
        active_electrons=2,
        active_orbitals=3,
    ),
    "lih": MoleculeSpec(
        key="lih",
        display_name="LiH PF",
        geometry="""
Li 0.000000 0.000000 0.000000
H  0.000000 0.000000 1.570000
""".strip(),
        active_electrons=2,
        active_orbitals=4,
    ),
    "dmds": MoleculeSpec(
        key="dmds",
        display_name="DMDS PF",
        geometry="""
C  -2.1538   -0.6230    0.3341
S  -0.6470    0.3161    0.7111
S   0.6470   -0.3160   -0.7111
C   2.1538    0.6231   -0.3341
H  -2.9509   -0.3548    1.0631
H  -2.4958   -0.3767   -0.6961
H  -1.9388   -1.7128    0.4040
H   2.9509    0.3549   -1.0631
H   1.9386    1.7128   -0.4047
H   2.4957    0.3775    0.6963
""".strip(),
        active_electrons=6,
        active_orbitals=5,
        aolabels=("S 3p", "S 3s"),
    ),
    "n2": MoleculeSpec(
        key="n2",
        display_name="N2 PF",
        geometry="""
N 0.000000 0.000000 0.000000
N 0.000000 0.000000 1.098000
""".strip(),
        active_electrons=6,
        active_orbitals=5,
    ),
    "co": MoleculeSpec(
        key="co",
        display_name="CO PF",
        geometry="""
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.128000
""".strip(),
        active_electrons=6,
        active_orbitals=5,
    ),
    "nh3": MoleculeSpec(
        key="nh3",
        display_name="NH3 PF",
        geometry="""
N 0.000000 0.000000 0.000000
H 0.000000 0.937000 0.382000
H 0.812000 -0.469000 0.382000
H -0.812000 -0.469000 0.382000
""".strip(),
        active_electrons=6,
        active_orbitals=5,
    ),
    "hcn": MoleculeSpec(
        key="hcn",
        display_name="HCN PF",
        geometry="""
H 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.064000
N 0.000000 0.000000 2.220000
""".strip(),
        active_electrons=6,
        active_orbitals=5,
    ),
}

ACTIVE_MOLECULES = tuple(MOLECULE_SPECS)


def get_spec(key: str) -> MoleculeSpec:
    try:
        return MOLECULE_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"unknown molecule key: {key}") from exc


class StateAction:
    def __init__(self, apply_fn):
        self._apply_fn = apply_fn

    def __matmul__(self, state: np.ndarray) -> np.ndarray:
        return self._apply_fn(runtime_np.asarray(state, dtype=runtime_np.complex128))


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


def electron_number_indices(n_qubits: int, n_electrons: int) -> np.ndarray:
    return np.asarray(
        [basis_index for basis_index in range(2 ** n_qubits) if bin(basis_index).count("1") == n_electrons],
        dtype=np.int64,
    )


def fixed_electron_sector_eigensystem(
    hamiltonian: np.ndarray,
    n_qubits: int,
    n_electrons: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sector_indices = electron_number_indices(n_qubits, n_electrons)
    if sector_indices.size == 0:
        raise ValueError(f"empty electron-number sector for n_qubits={n_qubits}, n_electrons={n_electrons}")

    sector_hamiltonian = hamiltonian[np.ix_(sector_indices, sector_indices)]
    eigenvalues, sector_eigenvectors = np.linalg.eigh(sector_hamiltonian)
    embedded_eigenvectors = np.zeros((hamiltonian.shape[0], sector_eigenvectors.shape[1]), dtype=np.complex128)
    embedded_eigenvectors[sector_indices, :] = sector_eigenvectors
    return (
        np.asarray(np.real(eigenvalues), dtype=np.float64),
        embedded_eigenvectors,
        sector_indices,
    )


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
    tensor = runtime_np.asarray(state, dtype=runtime_np.complex128).reshape((2,) * n_qubits)
    gate = runtime_np.asarray(gate, dtype=runtime_np.complex128)
    moved = runtime_np.moveaxis(tensor, qubit, 0)
    updated = runtime_np.tensordot(gate, moved, axes=([1], [0]))
    updated = runtime_np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def apply_cnot_to_state(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    tensor = runtime_np.asarray(state, dtype=runtime_np.complex128).reshape((2,) * n_qubits)
    moved = runtime_np.moveaxis(tensor, (control, target), (0, 1))
    updated = moved.copy()
    updated[1, 0] = moved[1, 1]
    updated[1, 1] = moved[1, 0]
    updated = runtime_np.moveaxis(updated, (0, 1), (control, target))
    return updated.reshape(-1)


def apply_pauli_label_to_state(state: np.ndarray, label: str) -> np.ndarray:
    paulis = single_qubit_paulis()
    n_qubits = len(label)
    updated = runtime_np.asarray(state, dtype=runtime_np.complex128)
    for qubit, ch in enumerate(label):
        if ch == "I":
            continue
        updated = apply_single_qubit_gate_to_state(updated, paulis[ch], qubit, n_qubits)
    return updated


def apply_exact_pauli_rotation_to_state(state: np.ndarray, label: str, alpha: float) -> np.ndarray:
    state = runtime_np.asarray(state, dtype=runtime_np.complex128)
    rotated = apply_pauli_label_to_state(state, label)
    return math.cos(alpha) * state + 1j * math.sin(alpha) * rotated


def load_method_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pf_method", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load method file: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for symbol in ("get_method_config", "run_product_formula"):
        if not hasattr(module, symbol):
            raise AttributeError(f"method file is missing required symbol {symbol!r}")
    return module


def patch_pygridsynth_for_python39() -> None:
    import pygridsynth.mymath
    import pygridsynth.ring

    def ntz_compat(n: int) -> int:
        n = int(n)
        if n == 0:
            return 0
        lowbit = n & -n
        return lowbit.bit_length() - 1

    pygridsynth.mymath.ntz = ntz_compat
    pygridsynth.ring.ntz = ntz_compat


def import_pygridsynth_backend():
    try:
        patch_pygridsynth_for_python39()
        from pygridsynth import gridsynth_gates
        from pygridsynth.unitary import DOmegaUnitary
    except ImportError as exc:
        raise ImportError(
            "pygridsynth is required for compiler-backed T counting. "
            "Install it with `pip install pygridsynth` or `uv sync`."
        ) from exc
    return gridsynth_gates, DOmegaUnitary


def pygridsynth_matrix_from_gates(gate_string: str, d_omega_unitary_cls) -> np.ndarray:
    matrix = d_omega_unitary_cls.from_gates(gate_string).to_complex_matrix
    return np.array(
        [[complex(matrix[row, col]) for col in range(2)] for row in range(2)],
        dtype=np.complex128,
    )


def rz_matrix(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-0.5j * theta), 0.0],
            [0.0, np.exp(0.5j * theta)],
        ],
        dtype=np.complex128,
    )


def unitary_error_up_to_global_phase(target: np.ndarray, approx: np.ndarray) -> float:
    phase = np.vdot(target[:, 0], approx[:, 0])
    if abs(phase) <= 1e-30:
        phase = 1.0 + 0.0j
    else:
        phase = phase / abs(phase)
    return float(np.linalg.norm(target - (approx / phase), ord=2))


def build_pauli_rotation_compiler(n_qubits: int):
    gridsynth_gates, d_omega_unitary_cls = import_pygridsynth_backend()
    hadamard = (1.0 / math.sqrt(2.0)) * runtime_np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=runtime_np.complex128,
    )
    s_dagger = runtime_np.array([[1.0, 0.0], [0.0, -1j]], dtype=runtime_np.complex128)
    s_gate = runtime_np.array([[1.0, 0.0], [0.0, 1j]], dtype=runtime_np.complex128)
    named_single_qubit_gates = {
        "H": hadamard,
        "S": s_gate,
        "Sdg": s_dagger,
    }

    @functools.lru_cache(maxsize=None)
    def term_scaffold(label: str) -> Tuple[Tuple[Tuple[int, str], ...], Tuple[Tuple[int, str], ...], Tuple[Tuple[int, int], ...], int]:
        support = [idx for idx, ch in enumerate(label) if ch != "I"]
        if not support:
            raise ValueError("cannot compile an identity Pauli rotation")
        pivot = support[-1]

        basis_forward: list[Tuple[int, str]] = []
        basis_inverse: list[Tuple[int, str]] = []
        for qubit, pauli in enumerate(label):
            if pauli == "X":
                basis_forward.append((qubit, "H"))
                basis_inverse.append((qubit, "H"))
            elif pauli == "Y":
                basis_forward.extend([(qubit, "Sdg"), (qubit, "H")])
                basis_inverse.extend([(qubit, "H"), (qubit, "S")])

        cnot_pairs = tuple((qubit, pivot) for qubit in support[:-1])
        return tuple(basis_forward), tuple(basis_inverse), cnot_pairs, pivot

    @functools.lru_cache(maxsize=None)
    def compile_rz(theta: float, epsilon: float) -> Dict[str, object]:
        if epsilon <= 0.0:
            raise ValueError(f"synthesis epsilon must be positive, got {epsilon}")

        target_unitary = rz_matrix(theta)
        candidates = []
        seen_gate_strings = set()
        max_attempts = 48
        stale_attempts = 0
        stale_limit = 12

        for _ in range(max_attempts):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"pygridsynth is .*",
                    category=UserWarning,
                )
                gate_string = gridsynth_gates(theta, epsilon)
            if gate_string in seen_gate_strings:
                stale_attempts += 1
                if stale_attempts >= stale_limit:
                    break
                continue
            stale_attempts = 0
            seen_gate_strings.add(gate_string)
            rz_unitary = pygridsynth_matrix_from_gates(gate_string, d_omega_unitary_cls)
            candidates.append(
                {
                    "gate_string": gate_string,
                    "gate_count": len(gate_string),
                    "t_gate_count": gate_string.count("T"),
                    "rz_unitary": rz_unitary,
                    "operator_error": unitary_error_up_to_global_phase(target_unitary, rz_unitary),
                }
            )

        if not candidates:
            raise RuntimeError(f"pygridsynth returned no candidates for theta={theta:.16e}, epsilon={epsilon:.16e}")

        best = min(
            candidates,
            key=lambda item: (
                item["t_gate_count"],
                item["operator_error"],
                item["gate_count"],
                item["gate_string"],
            ),
        )
        return {
            "theta": theta,
            "epsilon": epsilon,
            "gate_string": best["gate_string"],
            "gate_count": best["gate_count"],
            "t_gate_count": best["t_gate_count"],
            "operator_error": best["operator_error"],
            "rz_unitary": best["rz_unitary"],
        }

    @functools.lru_cache(maxsize=None)
    def compile_pauli_rotation(label: str, alpha: float, epsilon: float) -> Dict[str, object]:
        alpha = float(round(alpha, 15))
        theta = float(round(-2.0 * alpha, 15))
        rz_compilation = compile_rz(theta, epsilon)
        rz_unitary = rz_compilation["rz_unitary"]
        basis_forward, basis_inverse, cnot_pairs, pivot = term_scaffold(label)
        single_gate = runtime_np.array(
            [
                [rz_unitary[0, 0], rz_unitary[0, 1]],
                [rz_unitary[1, 0], rz_unitary[1, 1]],
            ],
            dtype=runtime_np.complex128,
        )

        if rz_compilation["gate_count"] == 0:
            def apply_state_identity(state: np.ndarray) -> np.ndarray:
                return runtime_np.asarray(state, dtype=runtime_np.complex128)

            return {
                "cache_key": f"{label}|{alpha:.16e}|{epsilon:.16e}",
                "label": label,
                "alpha": alpha,
                "theta": theta,
                "epsilon": epsilon,
                "gate_string": rz_compilation["gate_string"],
                "gate_count": rz_compilation["gate_count"],
                "t_gate_count": rz_compilation["t_gate_count"],
                "operator_error": rz_compilation["operator_error"],
                "unitary": StateAction(apply_state_identity),
                "state_action": apply_state_identity,
            }

        def apply_state(state: np.ndarray) -> np.ndarray:
            updated = runtime_np.asarray(state, dtype=runtime_np.complex128)
            for qubit, gate_name in basis_forward:
                updated = apply_single_qubit_gate_to_state(updated, named_single_qubit_gates[gate_name], qubit, n_qubits)
            for control, target in cnot_pairs:
                updated = apply_cnot_to_state(updated, control, target, n_qubits)
            updated = apply_single_qubit_gate_to_state(updated, single_gate, pivot, n_qubits)
            for control, target in cnot_pairs:
                updated = apply_cnot_to_state(updated, control, target, n_qubits)
            for qubit, gate_name in basis_inverse:
                updated = apply_single_qubit_gate_to_state(updated, named_single_qubit_gates[gate_name], qubit, n_qubits)
            return updated

        return {
            "cache_key": f"{label}|{alpha:.16e}|{epsilon:.16e}",
            "label": label,
            "alpha": alpha,
            "theta": theta,
            "epsilon": epsilon,
            "gate_string": rz_compilation["gate_string"],
            "gate_count": rz_compilation["gate_count"],
            "t_gate_count": rz_compilation["t_gate_count"],
            "operator_error": rz_compilation["operator_error"],
            "unitary": StateAction(apply_state),
            "state_action": apply_state,
        }

    return compile_pauli_rotation


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
    n_qubits = 2 * spec.active_orbitals
    if HAS_OPENFERMION:
        hamiltonian, identity_coefficient, pauli_terms = qubit_problem_from_integrals(one_body, two_body, constant)
    else:
        if n_qubits > 8:
            raise RuntimeError("OpenFermion is required to build cached PF problems with more than 8 qubits.")
        hamiltonian = build_qubit_hamiltonian(one_body, two_body, constant)
        identity_coefficient, pauli_terms = pauli_decomposition(hamiltonian)
    initial_state = hartree_fock_state(spec.active_electrons, spec.active_orbitals)
    exact_state = expm(1j * hamiltonian * spec.target_time) @ initial_state
    exact_state = exact_state / np.linalg.norm(exact_state)
    sector_eigenvalues, sector_eigenvectors, sector_indices = fixed_electron_sector_eigensystem(
        hamiltonian,
        n_qubits,
        spec.active_electrons,
    )
    ground_state_energy = float(sector_eigenvalues[0])
    ground_state = np.asarray(sector_eigenvectors[:, 0], dtype=np.complex128)
    hf_ground_state_overlap = float(abs(np.vdot(ground_state, initial_state)) ** 2)

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
        "ground_state_sector": "fixed_electron_number",
        "hf_ground_state_overlap": hf_ground_state_overlap,
        "spectral_peak_weight_threshold": SPECTRAL_PEAK_WEIGHT_THRESHOLD,
        "centered_hamiltonian_norm": centered_norm,
        "identity_coefficient": identity_coefficient,
        "total_pauli_rotation_terms": len(pauli_terms),
        "max_abs_total_alpha": max(abs_alphas),
        "min_abs_total_alpha": min(abs_alphas),
        "n_qubits": n_qubits,
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
        "sector_eigenvalues": sector_eigenvalues,
        "sector_eigenvectors_real": np.real(sector_eigenvectors),
        "sector_eigenvectors_imag": np.imag(sector_eigenvectors),
        "sector_indices": sector_indices,
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
        )
    if rebuild:
        payload = build_problem_cache(spec)
        metadata = payload["metadata"]
        arrays = payload["arrays"]
    else:
        arrays = dict(np.load(npz_path, allow_pickle=False))
    metadata = dict(metadata)

    hamiltonian = arrays["hamiltonian_real"] + 1j * arrays["hamiltonian_imag"]
    initial_state = arrays["initial_state_real"] + 1j * arrays["initial_state_imag"]
    exact_state = arrays["exact_state_real"] + 1j * arrays["exact_state_imag"]
    sector_keys = {
        "sector_eigenvalues",
        "sector_eigenvectors_real",
        "sector_eigenvectors_imag",
        "sector_indices",
    }
    if sector_keys.issubset(arrays):
        sector_eigenvalues = np.asarray(arrays["sector_eigenvalues"], dtype=np.float64)
        sector_eigenvectors = np.asarray(
            arrays["sector_eigenvectors_real"] + 1j * arrays["sector_eigenvectors_imag"],
            dtype=np.complex128,
        )
        sector_indices = np.asarray(arrays["sector_indices"], dtype=np.int64)
    else:
        sector_eigenvalues, sector_eigenvectors, sector_indices = fixed_electron_sector_eigensystem(
            hamiltonian,
            int(metadata["n_qubits"]),
            spec.active_electrons,
        )
    ground_state = np.asarray(sector_eigenvectors[:, 0], dtype=np.complex128)
    hf_ground_state_overlap = float(abs(np.vdot(ground_state, initial_state)) ** 2)
    metadata["ground_state_energy_hartree"] = float(sector_eigenvalues[0])
    metadata["ground_state_sector"] = "fixed_electron_number"
    metadata["hf_ground_state_overlap"] = hf_ground_state_overlap
    metadata.setdefault("spectral_peak_weight_threshold", SPECTRAL_PEAK_WEIGHT_THRESHOLD)

    pauli_terms = []
    build_dense_term_matrices = int(metadata["n_qubits"]) <= 8
    for label, coefficient, support_size in zip(
        arrays["pauli_labels"].tolist(),
        arrays["pauli_coefficients"].tolist(),
        arrays["pauli_support_sizes"].tolist(),
    ):
        matrix = pauli_matrix_from_label(str(label)) if build_dense_term_matrices else None
        runtime_matrix = (
            runtime_np.asarray(matrix, dtype=runtime_np.complex128)
            if matrix is not None
            else None
        )
        pauli_terms.append(
            {
                "label": str(label),
                "coefficient": float(coefficient),
                "support_size": int(support_size),
                "matrix": runtime_matrix,
            }
        )

    return {
        "metadata": metadata,
        "hamiltonian": runtime_np.asarray(hamiltonian, dtype=runtime_np.complex128),
        "identity": runtime_np.eye(hamiltonian.shape[0], dtype=runtime_np.complex128),
        "initial_state": runtime_np.asarray(initial_state, dtype=runtime_np.complex128),
        "exact_state": runtime_np.asarray(exact_state, dtype=runtime_np.complex128),
        "ground_state": runtime_np.asarray(ground_state, dtype=runtime_np.complex128),
        "sector_eigenvalues": runtime_np.asarray(sector_eigenvalues, dtype=runtime_np.float64),
        "sector_eigenvectors": runtime_np.asarray(sector_eigenvectors, dtype=runtime_np.complex128),
        "sector_indices": runtime_np.asarray(sector_indices, dtype=runtime_np.int64),
        "pauli_terms": pauli_terms,
        "apply_exact_pauli_rotation": apply_exact_pauli_rotation_to_state,
    }


def build_problem(molecule: str) -> Dict[str, object]:
    spec = get_spec(molecule)
    payload = load_problem_data(molecule)
    metadata = payload["metadata"]
    hamiltonian = payload["hamiltonian"]

    return {
        "experiment": f"pf_{molecule}",
        "molecule": molecule,
        "display_name": spec.display_name,
        "hamiltonian": hamiltonian,
        "identity": payload["identity"],
        "identity_coefficient": float(metadata["identity_coefficient"]),
        "pauli_terms": payload["pauli_terms"],
        "apply_exact_pauli_rotation": payload.get("apply_exact_pauli_rotation"),
        "initial_state": payload["initial_state"],
        "exact_state": payload["exact_state"],
        "ground_state": payload["ground_state"],
        "sector_eigenvalues": payload["sector_eigenvalues"],
        "sector_eigenvectors": payload["sector_eigenvectors"],
        "sector_indices": payload["sector_indices"],
        "target_time": float(spec.target_time),
        "time_sign": 1.0,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_fidelity_floor": float(metadata["chemical_accuracy_fidelity_floor"]),
        "ground_state_energy_hartree": float(metadata["ground_state_energy_hartree"]),
        "ground_state_sector": metadata["ground_state_sector"],
        "hf_ground_state_overlap": float(metadata["hf_ground_state_overlap"]),
        "spectral_peak_weight_threshold": float(
            metadata.get("spectral_peak_weight_threshold", SPECTRAL_PEAK_WEIGHT_THRESHOLD)
        ),
        "centered_hamiltonian_norm": float(metadata["centered_hamiltonian_norm"]),
        "total_pauli_rotation_terms": int(metadata["total_pauli_rotation_terms"]),
        "max_abs_total_alpha": float(metadata["max_abs_total_alpha"]),
        "min_abs_total_alpha": float(metadata["min_abs_total_alpha"]),
        "n_qubits": int(metadata["n_qubits"]),
        "compile_pauli_rotation": build_pauli_rotation_compiler(int(metadata["n_qubits"])),
        "compilation_backend": "pygridsynth",
        "array_backend": array_backend_name(),
        "cas_label": metadata["cas_label"],
        "basis": metadata["basis"],
        "localization": metadata["localization"],
    }


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    overlap = runtime_np.vdot(state_a, state_b)
    return float(abs(overlap) ** 2)


def unwrap_energy_near_reference(phase: float, target_time: float, reference_energy: float) -> float:
    principal_energy = float(phase / target_time)
    phase_period = float((2.0 * math.pi) / target_time)
    wrap_index = round((reference_energy - principal_energy) / phase_period)
    return principal_energy + wrap_index * phase_period


def estimate_ground_state_energy_from_overlap(
    reference_state: np.ndarray,
    propagated_state: np.ndarray,
    target_time: float,
    reference_energy: float,
    identity_shift: float = 0.0,
) -> Dict[str, object]:
    overlap = runtime_np.vdot(reference_state, propagated_state)
    overlap_magnitude = float(abs(overlap))
    if overlap_magnitude <= 1e-15:
        return {
            "phase_overlap_magnitude": overlap_magnitude,
            "energy_estimate_hartree": None,
            "energy_error_hartree": None,
        }

    centered_reference_energy = float(reference_energy - identity_shift)
    centered_energy_estimate = unwrap_energy_near_reference(
        float(runtime_np.angle(overlap)),
        target_time,
        centered_reference_energy,
    )
    energy_estimate = centered_energy_estimate + identity_shift
    return {
        "phase_overlap_magnitude": overlap_magnitude,
        "energy_estimate_hartree": float(energy_estimate),
        "energy_error_hartree": float(abs(energy_estimate - reference_energy)),
    }


def select_lowest_reference_peak(
    reference_state: np.ndarray,
    sector_eigenvalues: np.ndarray,
    sector_eigenvectors: np.ndarray,
    min_weight: float = SPECTRAL_PEAK_WEIGHT_THRESHOLD,
) -> Dict[str, object]:
    reference = runtime_np.asarray(reference_state, dtype=runtime_np.complex128)
    eigenvalues = runtime_np.asarray(sector_eigenvalues, dtype=runtime_np.float64)
    eigenvectors = runtime_np.asarray(sector_eigenvectors, dtype=runtime_np.complex128)
    amplitudes = eigenvectors.conj().T @ reference
    weights = runtime_np.abs(amplitudes) ** 2
    resolved = runtime_np.flatnonzero(weights >= float(min_weight))
    if resolved.size == 0:
        return {
            "peak_index": None,
            "peak_energy_hartree": None,
            "peak_weight": None,
            "peak_state": None,
            "reference_amplitude": None,
            "resolved_peak_count": 0,
            "max_peak_weight": float(runtime_np.max(weights)) if weights.size else 0.0,
        }

    selected = int(resolved[runtime_np.argmin(eigenvalues[resolved])])
    return {
        "peak_index": selected,
        "peak_energy_hartree": float(eigenvalues[selected]),
        "peak_weight": float(weights[selected]),
        "peak_state": runtime_np.asarray(eigenvectors[:, selected], dtype=runtime_np.complex128),
        "reference_amplitude": complex(amplitudes[selected]),
        "resolved_peak_count": int(resolved.size),
        "max_peak_weight": float(runtime_np.max(weights)),
    }


def estimate_reference_peak_energy_from_state(
    reference_state: np.ndarray,
    propagated_state: np.ndarray,
    sector_eigenvalues: np.ndarray,
    sector_eigenvectors: np.ndarray,
    target_time: float,
    identity_shift: float = 0.0,
    min_weight: float = SPECTRAL_PEAK_WEIGHT_THRESHOLD,
) -> Dict[str, object]:
    peak = select_lowest_reference_peak(
        reference_state,
        sector_eigenvalues,
        sector_eigenvectors,
        min_weight,
    )
    if peak["peak_state"] is None:
        return {
            **peak,
            "phase_overlap_magnitude": None,
            "energy_estimate_hartree": None,
            "energy_error_hartree": None,
        }

    peak_state = runtime_np.asarray(peak["peak_state"], dtype=runtime_np.complex128)
    reference_amplitude = complex(peak["reference_amplitude"])
    propagated_amplitude = runtime_np.vdot(peak_state, propagated_state)
    peak_weight = float(peak["peak_weight"])
    phase_component = runtime_np.conjugate(reference_amplitude) * propagated_amplitude / peak_weight
    phase_magnitude = float(abs(phase_component))
    if phase_magnitude <= 1e-15:
        return {
            **peak,
            "phase_overlap_magnitude": phase_magnitude,
            "energy_estimate_hartree": None,
            "energy_error_hartree": None,
        }

    reference_energy = float(peak["peak_energy_hartree"])
    centered_reference_energy = reference_energy - float(identity_shift)
    centered_energy_estimate = unwrap_energy_near_reference(
        float(runtime_np.angle(phase_component)),
        target_time,
        centered_reference_energy,
    )
    energy_estimate = centered_energy_estimate + float(identity_shift)
    return {
        **peak,
        "phase_overlap_magnitude": phase_magnitude,
        "energy_estimate_hartree": float(energy_estimate),
        "energy_error_hartree": float(abs(energy_estimate - reference_energy)),
    }


def score_from_error_budget(
    t_gate_count: int,
    allowed_error: float,
    actual_error: float | None,
) -> float:
    if actual_error is None or not math.isfinite(actual_error):
        return float(t_gate_count) + 1e9
    excess_error = max(0.0, actual_error - allowed_error)
    penalty = 1e9 * (excess_error / max(allowed_error, 1e-15))
    return float(t_gate_count) + penalty


def t_gate_metrics(run_result: Dict[str, object]) -> Dict[str, object]:
    return {
        "t_gate_count": int(run_result["t_gate_count"]),
        "nonzero_rotation_count": int(run_result["nonzero_rotation_count"]),
        "skipped_rotation_count": int(run_result["skipped_rotation_count"]),
        "synthesis_epsilon_histogram": run_result["synthesis_epsilon_histogram"],
        "unique_compilation_count": int(run_result["unique_compilation_count"]),
        "unique_compilations": run_result["unique_compilations"],
        "method_config": run_result["method_config"],
    }
