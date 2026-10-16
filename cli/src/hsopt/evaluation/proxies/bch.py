"""Leading BCH energy-bias proxy adapted from the October 2, 2026 experiment.

For a symmetric, first-order-consistent schedule with +i time convention,
S(dt) = exp(i dt (H + dt**2 B + O(dt**4))). Estimate bias with
 dt**2 <reference|B|reference>, using capped selected-determinant Ritz vectors.
This is neither a rigorous bound nor a compiled-unitary/QPE T-cost score.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.linalg import eigh

from ..compiled_unitary import (
    DIFFERENCE_ERROR_CONTRACTS,
    _product_formula_schedule,
    load_policy,
    load_problem,
)

CONTRACT_VERSION = "bch_energy_bias_proxy_v1"
DEFAULT_REFERENCE_SIZE = 32

Pauli = tuple[int, int]
Operator = dict[Pauli, float]
State = dict[int, complex]


def encode(label: str) -> Pauli:
    x = z = 0
    for c in label:
        x = (x << 1) | (c in "XY")
        z = (z << 1) | (c in "YZ")
    return x, z


def add(*pieces: tuple[float, Operator]) -> Operator:
    out: Operator = {}
    for factor, op in pieces:
        for key, value in op.items():
            out[key] = out.get(key, 0.0) + factor * value
    return {k: v for k, v in out.items() if v != 0.0}


def lie(a: Operator, b: Operator) -> Operator:
    """Hermitian Lie bracket i[A,B], with Hermitian Pauli conventions."""
    if len(a) * len(b) > 2_000_000:
        raise ValueError("BCH expansion exceeds the sparse pair budget; use exact evaluation")
    out: Operator = {}
    for (x, z), c in a.items():
        for (u, v), d in b.items():
            if not (((x & v).bit_count() + (z & u).bit_count()) & 1):
                continue
            key = (x ^ u, z ^ v)
            power = (
                (x & z).bit_count()
                + (u & v).bit_count()
                - (key[0] & key[1]).bit_count()
                + 2 * (z & u).bit_count()
            ) % 4
            # Anticommuting Hermitian Paulis have product phase +/- i.
            factor = -2 if power == 1 else 2
            out[key] = out.get(key, 0.0) + factor * c * d
    return {k: v for k, v in out.items() if v != 0.0}


def bch_coefficients(gates: list[tuple[Pauli, float]]):
    """Chronological gates multiply on the LEFT of the accumulated circuit."""
    h: Operator = {}
    a: Operator = {}
    b: Operator = {}
    for p, coefficient in gates:
        g = {p: coefficient}
        comm = lie(g, h)
        b = add((1, b), (0.5, lie(g, a)), (1 / 12, lie(g, comm)), (-1 / 12, lie(h, comm)))
        a = add((1, a), (0.5, comm))
        h = add((1, h), (1, g))
    return h, a, b


def symmetric_gates(terms):
    half = [(p, c / 2) for p, c in terms]
    return half + half[::-1]


def apply(op: Operator, state: State) -> State:
    out: State = {}
    for (x, z), c in op.items():
        yphase = 1j ** (x & z).bit_count()
        for determinant, amplitude in state.items():
            target = determinant ^ x
            value = c * yphase * (-1) ** (z & determinant).bit_count() * amplitude
            out[target] = out.get(target, 0j) + value
    return {k: v for k, v in out.items() if v != 0j}


def expectation(op: Operator, state: State) -> float:
    applied = apply(op, state)
    value = sum(np.conj(v) * applied.get(k, 0j) for k, v in state.items())
    assert abs(value.imag) < 1e-9
    return float(value.real)


def projected_matrix(op: Operator, determinants: list[int]):
    matrix = np.zeros((len(determinants), len(determinants)), complex)
    indices = {d: i for i, d in enumerate(determinants)}
    for j, determinant in enumerate(determinants):
        for target, value in apply(op, {determinant: 1}).items():
            if target in indices:
                matrix[indices[target], j] = value
    assert np.max(np.abs(matrix - matrix.conj().T)) < 1e-10
    return matrix


def selected_reference(op: Operator, n: int, electrons: int, cap: int):
    hf = ((1 << electrons) - 1) << (n - electrons)
    determinants = [hf]
    while True:
        energies, vectors = eigh(projected_matrix(op, determinants))
        energy = float(energies[0])
        state = dict(zip(determinants, vectors[:, 0]))
        applied = apply(op, state)
        residual = dict(applied)
        for d, amplitude in state.items():
            residual[d] = residual.get(d, 0j) - energy * amplitude
        residual_norm = float(np.sqrt(sum(abs(v) ** 2 for v in residual.values())))
        available = [
            (abs(v), d)
            for d, v in residual.items()
            if d not in state and d.bit_count() == electrons and abs(v) > 1e-12
        ]
        if len(determinants) >= cap or not available:
            return state, energy, residual_norm
        available.sort(reverse=True)
        count = min(cap - len(determinants), max(1, len(determinants)))
        determinants += [d for _, d in available[:count]]


def evaluate_molecule(
    problem_path: Path,
    policy_path: Path,
    *,
    reference_size: int = DEFAULT_REFERENCE_SIZE,
) -> dict[str, Any]:
    """Estimate ideal PF bias for one symmetric, consistent candidate schedule."""
    if not 1 <= reference_size <= 512:
        raise ValueError("reference_size must be between 1 and 512 determinants")
    started = time.perf_counter()
    problem_path = problem_path.resolve()
    policy_path = policy_path.resolve()
    problem = load_problem(problem_path)
    _, rotations, metrics = _product_formula_schedule(problem, load_policy(policy_path))
    dt = metrics["step_time"]
    # Restore the physical frame before comparing to the original H. Symmetric
    # commuting-group stages need not be palindromic in their flat expansion.
    from ..clifford import conjugate, inverse
    frame = inverse(metrics["elote"]["basis_change"])
    rotations = [(dict(term, label=label), sign*angle) for term, angle in rotations
                 for sign, label in [conjugate(term["label"], frame)]]
    if not metrics["elote"].get("symmetric_splitting"):
        for (left, a), (right, b) in zip(rotations, reversed(rotations), strict=True):
            if left["label"] != right["label"] or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("BCH proxy requires a symmetric (palindromic) rotation schedule")
    op = add(*[(1.0, {encode(term["label"]): term["coefficient"]}) for term in problem["terms"]])
    gates = [(encode(term["label"]), angle / dt) for term, angle in rotations]
    first_order = add(*[(1.0, {p: coefficient}) for p, coefficient in gates])
    scale = max(1.0, sum(abs(c) for c in op.values()))
    if sum(abs(c) for c in add((1.0, first_order), (-1.0, op)).values()) > 1e-10 * scale:
        raise ValueError("BCH proxy requires a first-order-consistent Hamiltonian schedule")
    _, second, third = bch_coefficients(gates)
    if sum(abs(c) for c in second.values()) > 1e-10 * scale**2:
        raise ValueError("BCH proxy requires a vanishing second-order log coefficient")
    state, reference_energy, residual = selected_reference(
        op, int(problem["n_qubits"]), int(problem["electrons"]), reference_size
    )
    coefficient = expectation(third, state)
    bias = dt**2 * coefficient
    return {
        "status": "ok",
        "evaluation_method": "bch",
        "evaluation_contract_version": CONTRACT_VERSION,
        "observable": "leading_bch_ideal_pf_energy_bias",
        "compiled": False,
        "certificate": False,
        "score": None,
        "problem": problem["name"],
        "problem_path": str(problem_path),
        "policy_path": str(policy_path),
        "problem_sha256": hashlib.sha256(problem_path.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "n_qubits": problem["n_qubits"],
        "electrons": problem["electrons"],
        "target_time": problem["target_time"],
        "phase_encoding": problem["phase_encoding"],
        **metrics,
        "reference_cap": reference_size,
        "reference_determinants": len(state),
        "reference_energy_hartree": reference_energy + problem["identity_coefficient"],
        "reference_residual_hartree": residual,
        "bch_terms": len(third),
        "bch_coefficient_hartree_cubed": coefficient,
        "proxy_bias_hartree": bias,
        "absolute_proxy_bias_hartree": abs(bias),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "limitations": (
            "Leading-order estimate only; reference cap controls cost, not accuracy. "
            "Omits higher-order PF terms, rotation synthesis, HF peak selection and QPE cost. "
            "The schedule is treated as scaling linearly with step time."
        ),
    }


def evaluate_difference(
    reaction_path: Path,
    policy_path: Path,
    *,
    error_contract: str,
    reference_size: int = DEFAULT_REFERENCE_SIZE,
) -> dict[str, Any]:
    """Aggregate estimated biases; never claim either exact reaction contract passes."""
    if error_contract not in DIFFERENCE_ERROR_CONTRACTS:
        raise ValueError(f"error_contract must be one of {DIFFERENCE_ERROR_CONTRACTS}")
    started = time.perf_counter()
    reaction_path = reaction_path.resolve()
    reaction = json.loads(reaction_path.read_text())
    if not reaction["terms"]:
        raise ValueError("reaction must contain at least one species")
    results: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    coefficients: dict[str, float] = {}
    for item in reaction["terms"]:
        species = str(item["species"])
        coefficient = float(item["coefficient"])
        if not math.isfinite(coefficient):
            raise ValueError("reaction coefficients must be finite")
        path = (reaction_path.parent / item["problem"]).resolve()
        if not path.is_relative_to(reaction_path.parent):
            raise ValueError("reaction Hamiltonian escapes its input directory")
        if species in paths and paths[species] != path:
            raise ValueError(f"conflicting Hamiltonians for {species}")
        paths[species] = path
        coefficients[species] = coefficients.get(species, 0.0) + coefficient
        if species not in results:
            results[species] = evaluate_molecule(path, policy_path, reference_size=reference_size)
    contributions = {s: coefficients[s] * r["proxy_bias_hartree"] for s, r in results.items()}
    signed = math.fsum(contributions.values())
    individual = math.fsum(abs(value) for value in contributions.values())
    return {
        "status": "ok",
        "evaluation_method": "bch",
        "evaluation_contract_version": CONTRACT_VERSION,
        "observable": "leading_bch_ideal_pf_signed_energy_difference_bias",
        "compiled": False,
        "certificate": False,
        "score": None,
        "reaction": reaction["name"],
        "reaction_path": str(reaction_path),
        "reaction_sha256": hashlib.sha256(reaction_path.read_bytes()).hexdigest(),
        "error_contract": error_contract,
        "proxy_bias_hartree": signed,
        "sum_absolute_species_proxy_bias_hartree": individual,
        "contract_proxy_bias_hartree": individual
        if error_contract == "individual_errors"
        else abs(signed),
        "species": results,
        "evaluation_wall_seconds": time.perf_counter() - started,
    }
