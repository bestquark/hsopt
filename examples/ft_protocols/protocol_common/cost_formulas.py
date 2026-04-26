"""Paper-exact T-gate and Toffoli cost formulas for fault-tolerant chemistry.

All formulas are self-contained (no Qualtran dependency).

References:
  [Lee2021]   Lee et al., PRX Quantum 2, 030305 (2021).  arXiv:2011.03494
  [Berry2019] Berry et al., Quantum 3, 208 (2019).       arXiv:1902.02134
  [vonBurg2021] von Burg et al., Phys. Rev. Res. 3, 033055 (2021). arXiv:2007.14460
  [Berry2015] Berry et al., Phys. Rev. Lett. 114, 090502 (2015). arXiv:1412.4687
"""

from __future__ import annotations

import math
from typing import Dict, List


# ---------------------------------------------------------------------------
# QROM primitives
# ---------------------------------------------------------------------------

def optimal_qrom_k(n_items: int, output_bits: int) -> int:
    """Optimal expansion factor for forward QROM (minimizes Toffoli count)."""
    if output_bits <= 0 or n_items <= 0:
        return 1
    k = max(1, math.isqrt(max(1, n_items // max(1, output_bits))))
    best_cost = _qrom_cost_raw(n_items, k, output_bits)
    for trial in [k - 1, k + 1, k + 2]:
        if trial < 1:
            continue
        cost = _qrom_cost_raw(n_items, trial, output_bits)
        if cost < best_cost:
            best_cost = cost
            k = trial
    return k


def optimal_inverse_qrom_k(n_items: int) -> int:
    """Optimal expansion factor for inverse (erasure) QROM."""
    if n_items <= 0:
        return 1
    k = max(1, math.isqrt(n_items))
    best_cost = math.ceil(n_items / k) + k
    for trial in [k - 1, k + 1, k + 2]:
        if trial < 1:
            continue
        cost = math.ceil(n_items / trial) + trial
        if cost < best_cost:
            best_cost = cost
            k = trial
    return k


def _qrom_cost_raw(n_items: int, k: int, output_bits: int) -> int:
    return math.ceil(n_items / k) + output_bits * (k - 1)


def qrom_cost(n_items: int, output_bits: int, k: int | None = None) -> int:
    """Toffoli cost of forward QROM: reads n_items entries of output_bits each.

    Cost = ceil(N/k) + b*(k-1)  where b = output_bits.
    [Lee2021 Appendix G, Berry2019 Section III.B]
    """
    if n_items <= 0 or output_bits <= 0:
        return 0
    if k is None:
        k = optimal_qrom_k(n_items, output_bits)
    return _qrom_cost_raw(n_items, k, output_bits)


def inverse_qrom_cost(n_items: int, k: int | None = None) -> int:
    """Toffoli cost of inverse (erasure) QROM.

    Cost = ceil(N/k') + k'.
    [Lee2021 Eq. below Eq. 33]
    """
    if n_items <= 0:
        return 0
    if k is None:
        k = optimal_inverse_qrom_k(n_items)
    return math.ceil(n_items / k) + k


# ---------------------------------------------------------------------------
# PREPARE via coherent alias sampling
# ---------------------------------------------------------------------------

def coherent_alias_sampling_cost(
    n_coefficients: int,
    aleph: int,
    *,
    k_forward: int | None = None,
    k_inverse: int | None = None,
) -> Dict[str, int]:
    """Cost of PREPARE + PREPARE^dagger via coherent alias sampling.

    Creates state |L> = sum_j sqrt(|alpha_j|/lambda) |j> using alias sampling.

    Registers: mu = ceil(log2(n_coefficients)) index bits, aleph keep-probability bits.
    QROM output = mu + aleph bits (alternate index + keep value).

    [Lee2021 Eqs. 27-33]
    """
    mu = math.ceil(math.log2(max(n_coefficients, 2)))
    output_bits = mu + aleph

    if k_forward is None:
        k_forward = optimal_qrom_k(n_coefficients, output_bits)
    if k_inverse is None:
        k_inverse = optimal_inverse_qrom_k(n_coefficients)

    forward = qrom_cost(n_coefficients, output_bits, k_forward)
    inverse = inverse_qrom_cost(n_coefficients, k_inverse)

    prepare = forward + aleph + mu
    prepare_inverse = inverse + aleph + mu
    return {
        "prepare_cost": prepare,
        "prepare_inverse_cost": prepare_inverse,
        "total": prepare + prepare_inverse,
        "qrom_forward": forward,
        "qrom_inverse": inverse,
        "mu_bits": mu,
        "aleph": aleph,
        "k_forward": k_forward,
        "k_inverse": k_inverse,
    }


# ---------------------------------------------------------------------------
# DF block encoding cost
# ---------------------------------------------------------------------------

def df_walk_operator_cost(
    n_orb: int,
    n_leaves: int,
    xi_per_leaf: List[int],
    beth_per_leaf: List[int] | int,
    aleph: int,
) -> Dict[str, int]:
    """Toffoli cost of one walk-operator step for DF qubitization.

    The walk operator = PREPARE^dag . SELECT . PREPARE + reflections.

    PREPARE selects over all (leaf, eigenvalue_index) pairs weighted by |f_k^l|.
    SELECT applies basis rotation (Givens), diagonal phase, inverse rotation.

    Parameters
    ----------
    n_orb : int
        Number of spatial orbitals.
    n_leaves : int
        Number of SF/DF leaves (L).
    xi_per_leaf : list[int]
        Number of kept eigenvalues per leaf.
    beth_per_leaf : list[int] or int
        Rotation bit-width per leaf.  If int, uniform across all leaves.
    aleph : int
        Coefficient bit-width for PREPARE.
    """
    if isinstance(beth_per_leaf, int):
        beth_per_leaf = [beth_per_leaf] * n_leaves

    total_coefficients = sum(xi_per_leaf)
    n_givens = n_orb * (n_orb - 1) // 2

    # -- PREPARE: coherent alias sampling over total_coefficients entries --
    prepare_data = coherent_alias_sampling_cost(total_coefficients, aleph)
    prepare_total = prepare_data["total"]

    # -- SELECT: Givens rotations + diagonal --
    # QROM loads Givens angles for the selected leaf.
    # Output width = n_givens * max(beth_per_leaf) bits per leaf.
    beth_max = max(beth_per_leaf) if beth_per_leaf else 1
    angle_output_bits = n_givens * beth_max
    k_angle = optimal_qrom_k(n_leaves, angle_output_bits)
    k_angle_inv = optimal_inverse_qrom_k(n_leaves)
    select_qrom = qrom_cost(n_leaves, angle_output_bits, k_angle)
    select_qrom_inv = inverse_qrom_cost(n_leaves, k_angle_inv)

    # Givens rotation execution: each rotation adds beth bits into a
    # phase-gradient register via (beth-2) Toffoli additions.
    # With mixed precision, the AVERAGE cost per step (over the PREPARE
    # superposition) is the weighted average of per-leaf costs.
    # For the worst-case (upper bound), use max beth.
    # The circuit must handle the max beth (phase gradient register width).
    givens_execution = n_givens * (beth_max - 2) if beth_max > 2 else 0
    givens_forward = givens_execution
    givens_inverse = givens_execution

    # Diagonal phase rotations: xi eigenvalues, each at aleph bits.
    xi_max = max(xi_per_leaf) if xi_per_leaf else 0
    diagonal_cost = xi_max * aleph

    select_total = (select_qrom + givens_forward + diagonal_cost
                    + givens_inverse + select_qrom_inv)

    # -- Reflections: 2 reflections, each costing O(ancilla_qubits) Toffoli --
    n_qubits = 2 * n_orb
    ancilla_bits = (prepare_data["mu_bits"] + aleph
                    + math.ceil(math.log2(max(n_leaves, 2)))
                    + beth_max + 5)
    reflection_cost = 2 * (n_qubits + ancilla_bits)

    total = prepare_total + select_total + reflection_cost

    return {
        "total_per_step": total,
        "prepare_total": prepare_total,
        "select_total": select_total,
        "select_qrom": select_qrom,
        "select_qrom_inv": select_qrom_inv,
        "givens_forward": givens_forward,
        "givens_inverse": givens_inverse,
        "diagonal_cost": diagonal_cost,
        "reflection_cost": reflection_cost,
        "n_givens": n_givens,
        "beth_max": beth_max,
        "angle_output_bits": angle_output_bits,
        "total_coefficients": total_coefficients,
    }


# ---------------------------------------------------------------------------
# THC block encoding cost
# ---------------------------------------------------------------------------

def thc_walk_operator_cost(
    n_orb: int,
    thc_rank: int,
    beth_per_mu: List[int] | int,
    aleph: int,
    b_r: int = 7,
) -> Dict[str, int]:
    """Toffoli cost of one walk-operator step for THC qubitization.

    Follows Lee et al. 2021 Eqs. 27-44.

    Parameters
    ----------
    n_orb : int
        Number of spatial orbitals (N/2 where N = spin-orbitals).
    thc_rank : int
        THC rank M.
    beth_per_mu : list[int] or int
        Rotation bit-width per auxiliary index mu.
    aleph : int
        Coefficient bits for PREPARE (the keep register).
    b_r : int
        Bits for amplitude rotation in PREPARE (default 7).
    """
    if isinstance(beth_per_mu, int):
        beth_per_mu = [beth_per_mu] * thc_rank

    M = thc_rank
    N_half = n_orb
    beth_max = max(beth_per_mu) if beth_per_mu else 1
    n_M = math.ceil(math.log2(M + 1))

    # Number of distinct coefficients for PREPARE
    # d = N/2 (one-body eigenvalues) + M*(M+1)/2 (unique zeta entries)
    d = N_half + M * (M + 1) // 2

    # QROM output size for PREPARE
    m = 2 * n_M + 2 + aleph

    # -- PREPARE cost [Lee2021 Eq. 33] --
    k_s1 = optimal_qrom_k(d, m)
    k_s2 = optimal_inverse_qrom_k(d)

    # Equal superposition creation + contiguous register
    superposition_cost = 10 * n_M + 2 * b_r - 9
    contiguous_cost = n_M * n_M + n_M - 1

    # QROM for alias sampling
    prepare_qrom = qrom_cost(d, m, k_s1)
    prepare_qrom_inv = inverse_qrom_cost(d, k_s2)

    # Inequality test + controlled swaps
    prepare_inequality = aleph
    prepare_swaps = 3 * n_M + 1

    prepare_cost = superposition_cost + contiguous_cost + prepare_qrom + prepare_inequality + prepare_swaps
    prepare_inv_cost = superposition_cost + contiguous_cost + prepare_qrom_inv + prepare_inequality + prepare_swaps

    # -- SELECT cost [Lee2021 Eq. 43] --
    # QROM for orbital rotation angles (chi_mu)
    rotation_output = N_half * beth_max
    k_r1 = optimal_qrom_k(M, rotation_output)
    k_r1_half = optimal_qrom_k(N_half, rotation_output)
    k_r2 = optimal_qrom_k(M, rotation_output)
    k_r2_inv = optimal_inverse_qrom_k(M)

    select_qrom_mu = qrom_cost(M, rotation_output, k_r1)
    select_qrom_mu_half = qrom_cost(N_half, rotation_output, k_r1_half)
    select_qrom_nu = qrom_cost(M, rotation_output, k_r2)
    select_qrom_nu_inv = inverse_qrom_cost(M, k_r2_inv)

    # Givens rotations for orbital embedding: N_half rotations at beth_max bits
    # Forward + inverse for mu, then same for nu
    givens_per_application = N_half * (beth_max - 2) if beth_max > 2 else 0

    # Spin-controlled swaps
    spin_swaps = N_half

    select_cost = (2 * M  # controlled operations
                   + 4 * givens_per_application
                   + select_qrom_mu + select_qrom_mu_half
                   + select_qrom_nu + select_qrom_nu_inv
                   + 2 * spin_swaps)

    # -- Reflections --
    n_qubits = 2 * n_orb
    ancilla_bits = 2 * n_M + beth_max + aleph + math.ceil(math.log2(max(d, 2))) + 10
    reflection_cost = 2 * (n_qubits + ancilla_bits)

    total = prepare_cost + prepare_inv_cost + select_cost + reflection_cost

    return {
        "total_per_step": total,
        "prepare_cost": prepare_cost,
        "prepare_inv_cost": prepare_inv_cost,
        "select_cost": select_cost,
        "reflection_cost": reflection_cost,
        "d": d,
        "n_M": n_M,
        "beth_max": beth_max,
    }


# ---------------------------------------------------------------------------
# QPE, QSVT, Taylor LCU wrappers
# ---------------------------------------------------------------------------

def qubitization_qpe_cost(
    walk_cost_per_step: int,
    lam: float,
    epsilon_qpe: float,
) -> Dict[str, object]:
    """Total Toffoli for qubitization-based QPE.

    n_iterations = ceil(pi * lambda / (2 * epsilon_qpe))
    total = n_iterations * walk_cost_per_step
    """
    n_iterations = math.ceil(math.pi * lam / (2.0 * epsilon_qpe))
    total = n_iterations * walk_cost_per_step
    return {
        "n_iterations": n_iterations,
        "walk_cost_per_step": walk_cost_per_step,
        "total_toffoli": total,
        "total_t_gates": 4 * total,
        "lambda": lam,
        "epsilon_qpe": epsilon_qpe,
    }


def qsvt_cost(
    walk_cost_per_step: int,
    lam: float,
    target_time: float,
    epsilon_poly: float,
) -> Dict[str, object]:
    """Total Toffoli for QSVT-based Hamiltonian simulation.

    The polynomial degree for Jacobi-Anger approximation is approximately:
      d ~ alpha*t + C * log(1/epsilon) / log(log(1/epsilon))
    where alpha = lambda, t = target_time.
    """
    tau = lam * target_time
    if epsilon_poly <= 0:
        raise ValueError("epsilon_poly must be positive")
    log_eps = math.log(max(1.0 / epsilon_poly, math.e))
    log_log_eps = max(math.log(max(log_eps, math.e)), 1.0)
    degree = int(math.ceil(tau + 0.5 * log_eps / log_log_eps))
    degree = max(degree, 1)

    total = degree * walk_cost_per_step
    return {
        "degree": degree,
        "walk_cost_per_step": walk_cost_per_step,
        "total_toffoli": total,
        "total_t_gates": 4 * total,
        "lambda": lam,
        "target_time": target_time,
        "tau": tau,
        "epsilon_poly": epsilon_poly,
    }


def taylor_lcu_cost(
    prepare_cost: int,
    select_cost: int,
    lam: float,
    target_time: float,
    taylor_order: int,
    epsilon_taylor: float,
) -> Dict[str, object]:
    """Total Toffoli for Taylor-LCU Hamiltonian simulation.

    The simulation is divided into S segments each of duration t/S.
    Within each segment, a Taylor expansion of order K is used.
    Oblivious amplitude amplification (OAA) boosts success probability.

    S is chosen so that lambda*t/S is small enough for order-K Taylor.
    Total queries per segment = K (each query = one SELECT).
    Total cost = S * (2*PREPARE + K*(PREPARE + SELECT)) * amplification.

    [Berry2015 Theorem 1]
    """
    K = taylor_order
    tau = lam * target_time

    # Choose S so that the Taylor remainder is below epsilon_taylor.
    # |e^x - T_K(x)| <= e^|x| * |x|^{K+1} / (K+1)!
    # With x = -i*lambda*t/S, |x| = lambda*t/S.
    S = 1
    while True:
        x = tau / S
        remainder = math.exp(x) * (x ** (K + 1)) / math.factorial(K + 1)
        total_err = S * remainder
        if total_err <= epsilon_taylor:
            break
        S += 1
        if S > 10000:
            break

    cost_per_segment = 2 * prepare_cost + K * (prepare_cost + select_cost)
    total = S * cost_per_segment

    return {
        "segments": S,
        "taylor_order": K,
        "cost_per_segment": cost_per_segment,
        "prepare_cost": prepare_cost,
        "select_cost": select_cost,
        "total_toffoli": total,
        "total_t_gates": 4 * total,
        "lambda": lam,
        "target_time": target_time,
        "tau": tau,
        "epsilon_taylor": epsilon_taylor,
        "actual_taylor_error": S * math.exp(tau / S) * (tau / S) ** (K + 1) / math.factorial(K + 1),
    }


# ---------------------------------------------------------------------------
# Logical qubit counts
# ---------------------------------------------------------------------------

def df_logical_qubits(
    n_orb: int,
    n_leaves: int,
    total_coefficients: int,
    beth_max: int,
    aleph: int,
) -> int:
    """Logical qubit count for DF qubitization-QPE."""
    n_system = 2 * n_orb
    mu_bits = math.ceil(math.log2(max(total_coefficients, 2)))
    leaf_bits = math.ceil(math.log2(max(n_leaves, 2)))
    ancilla = mu_bits + aleph + leaf_bits + beth_max + 10
    qpe_ancilla = math.ceil(math.log2(max(total_coefficients, 2))) + 5
    return n_system + ancilla + qpe_ancilla


def thc_logical_qubits(
    n_orb: int,
    thc_rank: int,
    d: int,
    beth_max: int,
    aleph: int,
    n_iterations: int,
) -> int:
    """Logical qubit count for THC qubitization-QPE.

    [Lee2021 Eq. 46]
    """
    N = 2 * n_orb
    n_M = math.ceil(math.log2(thc_rank + 1))
    qpe_bits = math.ceil(math.log2(max(n_iterations + 1, 2)))
    base = 2 * qpe_bits + N + 2 * n_M + beth_max + math.ceil(math.log2(max(d, 2))) + aleph + 5
    m = 2 * n_M + 2 + aleph
    k_s1 = optimal_qrom_k(d, m)
    option_a = m * k_s1 + math.ceil(math.log2(max(d // k_s1, 2)))
    option_b = m + beth_max * n_orb + beth_max - 2
    return base + max(option_a, option_b)
