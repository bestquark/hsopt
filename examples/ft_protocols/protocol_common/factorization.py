"""Hamiltonian factorization utilities for fault-tolerant chemistry protocols.

Provides single factorization (SF), double factorization (DF), tensor
hypercontraction (THC), Givens decomposition, angle/coefficient rounding,
and effective-Hamiltonian reconstruction from discretized factorizations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from openfermion import InteractionOperator, get_sparse_operator, jordan_wigner
from scipy.linalg import expm


FACTORIZATION_TOLERANCE = 1e-12


def _pair_matrix(two_body: np.ndarray) -> np.ndarray:
    n_orb = two_body.shape[0]
    matrix = np.asarray(two_body, dtype=np.float64).reshape(n_orb * n_orb, n_orb * n_orb)
    return 0.5 * (matrix + matrix.T)


def single_factorization(
    two_body: np.ndarray,
    tolerance: float = FACTORIZATION_TOLERANCE,
) -> List[np.ndarray]:
    matrix = _pair_matrix(two_body)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    n_orb = two_body.shape[0]

    factors: List[np.ndarray] = []
    for idx in order:
        ev = float(eigenvalues[idx])
        if abs(ev) <= tolerance:
            continue
        factor = math.sqrt(abs(ev)) * eigenvectors[:, idx].reshape(n_orb, n_orb)
        factors.append(0.5 * (factor + factor.T))
    return factors


def double_factorization(
    two_body: np.ndarray,
    tolerance: float = FACTORIZATION_TOLERANCE,
) -> List[Dict[str, np.ndarray]]:
    sf_factors = single_factorization(two_body, tolerance)
    leaves: List[Dict[str, np.ndarray]] = []
    for sf_factor in sf_factors:
        symmetric = 0.5 * (sf_factor + sf_factor.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        keep = np.abs(eigenvalues) > tolerance
        if not np.any(keep):
            continue
        leaves.append({
            "eigenvalues": eigenvalues[keep],
            "eigenvectors": eigenvectors[:, keep],
            "sf_factor": sf_factor,
        })
    return leaves


def thc_factorization(
    two_body: np.ndarray,
    rank: int | None = None,
    max_nfev: int = 120,
    seed: int = 1234,
    tolerance: float = FACTORIZATION_TOLERANCE,
) -> Dict[str, object]:
    from scipy.optimize import least_squares

    n_orb = two_body.shape[0]
    if rank is None:
        rank = 2 * n_orb
    matrix = _pair_matrix(two_body)
    matrix_norm = max(float(np.linalg.norm(matrix)), tolerance)

    chi = _initial_thc_orbital_factors(two_body, rank, seed=seed)

    def residual(raw: np.ndarray) -> np.ndarray:
        c = _normalize_columns(raw.reshape(n_orb, rank))
        z, basis = _solve_thc_zeta(matrix, c)
        return ((basis @ z @ basis.T - matrix) / matrix_norm).ravel()

    if max_nfev > 0:
        result = least_squares(residual, chi.ravel(), max_nfev=max_nfev,
                               ftol=1e-8, xtol=1e-8, gtol=1e-8)
        chi = _normalize_columns(result.x.reshape(n_orb, rank))
    zeta, _ = _solve_thc_zeta(matrix, chi)
    zeta = 0.5 * (zeta + zeta.T)
    rel_err = float(np.linalg.norm(_thc_basis(chi) @ zeta @ _thc_basis(chi).T - matrix) / matrix_norm)
    return {"chi": chi, "zeta": zeta, "rank": rank, "relative_error": rel_err}


def _initial_thc_orbital_factors(two_body: np.ndarray, rank: int, seed: int) -> np.ndarray:
    n_orb = two_body.shape[0]
    matrix = _pair_matrix(two_body)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    columns = []
    for idx in order:
        pair_mat = eigenvectors[:, idx].reshape(n_orb, n_orb)
        pair_mat = 0.5 * (pair_mat + pair_mat.T)
        vals, vecs = np.linalg.eigh(pair_mat)
        dominant = int(np.argmax(np.abs(vals)))
        col = np.asarray(vecs[:, dominant], dtype=np.float64)
        norm = np.linalg.norm(col)
        if norm > FACTORIZATION_TOLERANCE:
            columns.append(col / norm)
        if len(columns) == rank:
            break
    rng = np.random.default_rng(seed)
    while len(columns) < rank:
        col = rng.normal(size=n_orb)
        norm = np.linalg.norm(col)
        if norm > FACTORIZATION_TOLERANCE:
            columns.append(col / norm)
    return np.column_stack(columns[:rank])


def _normalize_columns(raw: np.ndarray) -> np.ndarray:
    chi = np.asarray(raw, dtype=np.float64).copy()
    norms = np.linalg.norm(chi, axis=0)
    safe = np.where(norms > FACTORIZATION_TOLERANCE, norms, 1.0)
    return chi / safe


def _thc_basis(chi: np.ndarray) -> np.ndarray:
    n_orb, rank = chi.shape
    return np.einsum("pm,qm->pqm", chi, chi).reshape(n_orb * n_orb, rank)


def _solve_thc_zeta(matrix: np.ndarray, chi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    basis = _thc_basis(chi)
    pinv = np.linalg.pinv(basis, rcond=1e-10)
    zeta = pinv @ matrix @ pinv.T
    return zeta, basis


# ---------------------------------------------------------------------------
# Givens rotation decomposition
# ---------------------------------------------------------------------------

def givens_decomposition(U: np.ndarray) -> List[Tuple[int, int, float]]:
    """Decompose an orthogonal matrix U into Givens rotations.

    Returns a list of (i, j, theta) such that
        U = prod_{(i,j,theta)} G(i, j, theta)
    where G(i, j, theta) is the identity matrix with the (i,i), (i,j),
    (j,i), (j,j) block replaced by [[cos(theta), -sin(theta)],
    [sin(theta), cos(theta)]].

    The product is applied left-to-right (first rotation acts first).
    """
    n = U.shape[0]
    R = np.array(U, dtype=np.float64, copy=True)
    rotations: List[Tuple[int, int, float]] = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            if abs(R[row, col]) < 1e-15:
                continue
            a = R[row - 1, col]
            b = R[row, col]
            r = math.hypot(a, b)
            c = a / r
            s = b / r
            theta = math.atan2(s, c)
            G = np.eye(n)
            G[row - 1, row - 1] = c
            G[row - 1, row] = s
            G[row, row - 1] = -s
            G[row, row] = c
            R = G @ R
            rotations.append((row - 1, row, theta))
    return rotations


def reconstruct_from_givens(n: int, rotations: List[Tuple[int, int, float]]) -> np.ndarray:
    """Reconstruct an orthogonal matrix from Givens rotations."""
    U = np.eye(n, dtype=np.float64)
    for i, j, theta in reversed(rotations):
        G = np.eye(n)
        c = math.cos(theta)
        s = math.sin(theta)
        G[i, i] = c
        G[i, j] = -s
        G[j, i] = s
        G[j, j] = c
        U = G @ U
    return U


# ---------------------------------------------------------------------------
# Rounding utilities
# ---------------------------------------------------------------------------

def round_angle(theta: float, n_bits: int) -> float:
    """Round theta to the nearest multiple of 2*pi / 2^n_bits."""
    unit = 2.0 * math.pi / (2 ** n_bits)
    return round(theta / unit) * unit


def round_coefficient(value: float, n_bits: int, value_range: float) -> float:
    """Round a coefficient to n_bits fixed-point within [-value_range, value_range]."""
    if value_range <= 0:
        return 0.0
    unit = value_range / (2 ** (n_bits - 1))
    return round(value / unit) * unit


# ---------------------------------------------------------------------------
# Effective-Hamiltonian reconstruction
# ---------------------------------------------------------------------------

def df_lambda(
    one_body: np.ndarray,
    df_leaves: List[Dict[str, np.ndarray]],
) -> float:
    """Compute the 1-norm lambda for DF: lambda = Σ|T'_pq| + (1/4) Σ_l (Σ_k |f_k^l|)^2.

    Note: one_body here should be the modified T' = h - (1/2)Σ_r V_{prrq}.
    For simplicity we use the raw one-body and add a correction.
    """
    one_body_contribution = float(np.sum(np.abs(one_body)))
    two_body_contribution = 0.0
    for leaf in df_leaves:
        ev_sum = float(np.sum(np.abs(leaf["eigenvalues"])))
        two_body_contribution += ev_sum ** 2
    return one_body_contribution + 0.25 * two_body_contribution


def thc_lambda(
    one_body: np.ndarray,
    thc_data: Dict[str, object],
) -> float:
    """Compute the 1-norm for THC: lambda = Σ|t_l| + (1/2)Σ_{mu,nu}|zeta_{mu,nu}|."""
    eigenvalues = np.linalg.eigvalsh(one_body)
    one_body_contribution = float(np.sum(np.abs(eigenvalues)))
    zeta = np.asarray(thc_data["zeta"], dtype=np.float64)
    two_body_contribution = 0.5 * float(np.sum(np.abs(zeta)))
    return one_body_contribution + two_body_contribution


def _spatial_to_spin_integrals(
    one_body: np.ndarray,
    two_body: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert spatial-orbital integrals to spin-orbital integrals.

    Matches the convention in pfs/chemistry.py ``spatial_to_spin_integrals``:
      one_body_spin[2p+σ, 2q+σ]           = one_body[p, q]
      two_body_spin[2p+σ, 2q+τ, 2r+σ, 2s+τ] = two_body[p, r, q, s]
    """
    n_orb = one_body.shape[0]
    n_spin = 2 * n_orb

    one_body_spin = np.zeros((n_spin, n_spin), dtype=np.float64)
    one_body_spin[0::2, 0::2] = one_body
    one_body_spin[1::2, 1::2] = one_body

    two_body_reordered = np.transpose(two_body, (0, 2, 1, 3))
    two_body_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)
    for sigma in range(2):
        for tau in range(2):
            two_body_spin[sigma::2, tau::2, sigma::2, tau::2] = two_body_reordered

    return one_body_spin, two_body_spin


def compute_e_core(
    one_body: np.ndarray,
    two_body: np.ndarray,
    identity_coefficient: float,
    n_qubits: int,
) -> float:
    """Compute the core energy (nuclear repulsion + frozen core) from spatial integrals.

    The stored ``identity_coefficient`` is the Pauli-I coefficient after JW,
    which includes one-body and two-body trace contributions on top of e_core:
        identity_coefficient = e_core + Tr(H_electronic_JW) / 2^n_qubits
    """
    one_body_spin, two_body_spin = _spatial_to_spin_integrals(one_body, two_body)
    iop = InteractionOperator(0.0, one_body_spin, two_body_spin)
    qubit_op = jordan_wigner(iop)
    sparse_H = get_sparse_operator(qubit_op, n_qubits)
    dim = 2 ** n_qubits
    trace_per_dim = float(np.real(sparse_H.diagonal().sum())) / dim
    return identity_coefficient - trace_per_dim


def build_effective_hamiltonian_df(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
    df_leaves: List[Dict[str, np.ndarray]],
    beth_per_leaf: List[int] | int,
    aleph: int,
    n_qubits: int,
) -> np.ndarray:
    """Build the effective qubit Hamiltonian from a DF factorization with
    discretized Givens angles and coefficients.

    For each leaf l:
      1. Decompose eigenvectors U_l into Givens rotations
      2. Round each Givens angle to beth_per_leaf[l] bits
      3. Reconstruct the approximate U_tilde_l
      4. Round eigenvalues to aleph bits
      5. Rebuild the two-body integrals from the rounded factorization
    Then build the full qubit Hamiltonian via Jordan-Wigner.
    """
    n_orb = one_body.shape[0]
    if isinstance(beth_per_leaf, int):
        beth_per_leaf = [beth_per_leaf] * len(df_leaves)

    lambda_max = max(
        float(np.max(np.abs(leaf["eigenvalues"])))
        for leaf in df_leaves
    ) if df_leaves else 1.0

    V_tilde = np.zeros_like(two_body)
    for leaf_idx, leaf in enumerate(df_leaves):
        beth = beth_per_leaf[leaf_idx]
        eigvecs = leaf["eigenvectors"]
        eigvals = leaf["eigenvalues"]
        n_keep = eigvecs.shape[1]

        full_U = np.eye(n_orb)
        full_U[:, :n_keep] = eigvecs
        if n_keep < n_orb:
            Q, _ = np.linalg.qr(full_U)
            full_U = Q

        rotations = givens_decomposition(full_U)
        rounded_rotations = [(i, j, round_angle(theta, beth)) for i, j, theta in rotations]
        U_tilde = reconstruct_from_givens(n_orb, rounded_rotations)

        rounded_eigvals = np.array([
            round_coefficient(float(ev), aleph, lambda_max)
            for ev in eigvals
        ])

        padded_eigvals = np.zeros(n_orb)
        padded_eigvals[:n_keep] = rounded_eigvals

        W_tilde = U_tilde @ np.diag(padded_eigvals) @ U_tilde.T
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        V_tilde[p, q, r, s] += W_tilde[p, q] * W_tilde[r, s]

    return _build_qubit_hamiltonian(one_body, V_tilde, constant, n_qubits)


def build_effective_hamiltonian_thc(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
    thc_data: Dict[str, object],
    beth_per_mu: List[int] | int,
    aleph: int,
    n_qubits: int,
) -> np.ndarray:
    """Build the effective qubit Hamiltonian from a THC factorization with
    discretized orbital rotations and coefficients."""
    chi = np.asarray(thc_data["chi"], dtype=np.float64)
    zeta = np.asarray(thc_data["zeta"], dtype=np.float64)
    n_orb, rank = chi.shape

    if isinstance(beth_per_mu, int):
        beth_per_mu = [beth_per_mu] * rank

    zeta_max = max(float(np.max(np.abs(zeta))), FACTORIZATION_TOLERANCE)
    zeta_tilde = np.array([
        [round_coefficient(float(zeta[mu, nu]), aleph, zeta_max)
         for nu in range(rank)]
        for mu in range(rank)
    ])

    chi_tilde = np.zeros_like(chi)
    for mu in range(rank):
        beth = beth_per_mu[mu]
        col = chi[:, mu]
        norm = np.linalg.norm(col)
        if norm < FACTORIZATION_TOLERANCE:
            continue
        direction = col / norm
        full_U = np.eye(n_orb)
        full_U[:, 0] = direction
        Q, _ = np.linalg.qr(full_U)
        if np.dot(Q[:, 0], direction) < 0:
            Q[:, 0] = -Q[:, 0]
        rotations = givens_decomposition(Q)
        rounded = [(i, j, round_angle(theta, beth)) for i, j, theta in rotations]
        Q_tilde = reconstruct_from_givens(n_orb, rounded)
        chi_tilde[:, mu] = norm * Q_tilde[:, 0]

    V_tilde = np.einsum("pm,qm,mn,rn,sn->pqrs", chi_tilde, chi_tilde, zeta_tilde, chi_tilde, chi_tilde)
    return _build_qubit_hamiltonian(one_body, V_tilde, constant, n_qubits)


def _build_qubit_hamiltonian(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
    n_qubits: int,
) -> np.ndarray:
    """Build full qubit Hamiltonian from spatial integrals via Jordan-Wigner.

    The *constant* should be e_core (nuclear repulsion + frozen core energy),
    NOT the Pauli identity coefficient.  Use :func:`compute_e_core` to obtain it.
    """
    one_body_spin, two_body_spin = _spatial_to_spin_integrals(one_body, two_body)
    iop = InteractionOperator(constant, one_body_spin, two_body_spin)
    qubit_op = jordan_wigner(iop)
    sparse_H = get_sparse_operator(qubit_op, n_qubits)
    return np.asarray(sparse_H.toarray(), dtype=np.complex128)


def fixed_electron_sector_eigensystem(
    hamiltonian: np.ndarray,
    n_qubits: int,
    n_electrons: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) in the fixed-electron sector."""
    indices = np.array([
        i for i in range(2 ** n_qubits) if bin(i).count("1") == n_electrons
    ], dtype=np.int64)
    if indices.size == 0:
        raise ValueError(f"empty sector for {n_qubits} qubits, {n_electrons} electrons")
    sector_H = hamiltonian[np.ix_(indices, indices)]
    eigvals, sector_eigvecs = np.linalg.eigh(sector_H)
    full_eigvecs = np.zeros((hamiltonian.shape[0], sector_eigvecs.shape[1]), dtype=np.complex128)
    full_eigvecs[indices, :] = sector_eigvecs
    return np.real(eigvals).astype(np.float64), full_eigvecs


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    return float(abs(np.vdot(state_a, state_b)) ** 2)
