"""Inspectable ingredients of an Error-Limited Optimized Trotter Evolution policy.

HSOpt searches; ELOTE is the resulting saved algorithm. Legacy flat schedules
remain valid. Explicit groups describe commuting Hamiltonian blocks; stages
specify their splitting. Nothing here declares a candidate accurate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from functools import lru_cache

from . import clifford

SCHEMA = "elote-product-formula-v1"


def digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def commute(a, b):
    return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b, strict=True)) % 2 == 0


def term_fields(label, coefficient):
    return {
        "label": label,
        "coefficient": coefficient,
        "support_size": len(label) - label.count("I"),
        **{axis.lower() + "_count": label.count(axis) for axis in "XYZ"},
    }


@lru_cache(maxsize=128)
def _fermionic_blocks(n_qubits, terms):
    try:
        from openfermion import (
            FermionOperator,
            QubitOperator,
            hermitian_conjugated,
            jordan_wigner,
            normal_ordered,
            reverse_jordan_wigner,
        )
    except ImportError as exc:
        raise RuntimeError("fermionic nesting needs `uv sync --extra chemistry`") from exc
    h = QubitOperator()
    for label, coefficient in terms:
        h.terms[tuple((q, c) for q, c in enumerate(label) if c != "I")] = coefficient
    f = normal_ordered(reverse_jordan_wigner(h, n_qubits=n_qubits))
    # Reverse JW can introduce floating-point high-body residue. Keep the
    # trusted Hamiltonian exact by explicitly restoring its Pauli residual below.
    threshold = 1e-12 * max(1.0, sum(abs(c) for _, c in terms))
    seen, blocks, reconstructed = set(), [], {}
    for key, coefficient in sorted(f.terms.items()):
        if not key or key in seen or abs(coefficient) <= threshold:
            continue
        if len(key) not in (2, 4) or sum(action for _, action in key) * 2 != len(key):
            raise ValueError("nesting requires a number-conserving one-/two-body JW Hamiltonian")
        operator = FermionOperator(key, coefficient)
        conjugate = normal_ordered(hermitian_conjugated(operator))
        self_adjoint = (operator - conjugate).induced_norm() <= threshold
        if not self_adjoint:
            operator += conjugate
        seen.add(key)
        seen.update(conjugate.terms)
        paulis = jordan_wigner(operator)
        entries = []
        for axes, value in sorted(paulis.terms.items()):
            if abs(value.imag) > threshold:
                raise ValueError("fermionic block is not Hermitian")
            if not axes or value.real == 0:
                continue
            label = ["I"] * n_qubits
            for q, axis in axes:
                label[q] = axis
            entries.append(term_fields("".join(label), float(value.real)))
            reconstructed.setdefault(axes, []).append(float(value.real))
        if entries:
            if any(not commute(a["label"], b["label"]) for a in entries for b in entries):
                raise ValueError("fermionic block needs a non-Pauli joint synthesis implementation")
            blocks.append({"orbitals": sorted({q for q, _ in key}), "terms": entries})
    # Do NOT subtract with OpenFermion arithmetic: its default small-number
    # pruning could hide missing input coefficients. Audit raw coefficient maps.
    residual = {
        axes: h.terms.get(axes, 0.0) - math.fsum(reconstructed.get(axes, []))
        for axes in h.terms.keys() | reconstructed.keys()
        if axes
    }
    residual_norm = math.fsum(abs(c) for c in residual.values())
    if residual_norm > 100 * threshold:
        raise ValueError(
            "JW block reconstruction failed; do not infer chemistry from this encoding"
        )
    # Explicit correction terms prevent even small changes to the input H.
    for axes, value in sorted(residual.items()):
        if value == 0:
            continue
        label = ["I"] * n_qubits
        for q, axis in axes:
            label[q] = axis
        blocks.append(
            {
                "orbitals": list(range(n_qubits)),
                "terms": [term_fields("".join(label), float(value.real))],
                "small_residual_correction": True,
            }
        )
    return {
        "method": "normal_ordered_inverse_jw_hermitian_blocks_v1",
        "blocks": blocks,
        "reconstruction_residual_l1_before_correction": residual_norm,
        "sha256": digest(blocks),
    }


def fermionic_blocks(problem):
    """Trusted, cached decomposition; no candidate changes H or its identity shift."""
    terms = tuple(sorted((t["label"], float(t["coefficient"])) for t in problem["terms"]))
    return copy.deepcopy(_fermionic_blocks(int(problem["n_qubits"]), terms))


def candidate_problem(problem, policy):
    result = copy.deepcopy({k: v for k, v in problem.items() if k != "_path"})
    representation = getattr(policy.ProductFormula, "hamiltonian_representation", "pauli")
    if representation not in ("pauli", "fermionic_jw"):
        raise ValueError("Hamiltonian representation must be pauli or fermionic_jw")
    if representation == "fermionic_jw":
        result["fermionic_decomposition"] = fermionic_blocks(problem)
    return result


def schedule(raw, trusted_terms, *, limit):
    """Expand explicit commuting groups/stages, retaining a replayable recipe."""
    recipe = {"schema": SCHEMA}
    if isinstance(raw, dict):
        if set(raw) - {"groups", "stages", "merge_adjacent"}:
            raise ValueError("explicit PF fields are groups, stages, merge_adjacent")
        groups, stages = raw.get("groups"), raw.get("stages")
        if not isinstance(groups, (list, tuple)) or not groups or len(groups) > limit:
            raise ValueError("groups must be a nonempty bounded list of commuting rotation lists")
        if not isinstance(stages, (list, tuple)) or not stages or len(stages) > limit:
            raise ValueError("stages must be a nonempty list of (group_index, multiplier)")
        parsed = []
        group_size = 0
        for group in groups:
            rotations = _pairs(group, trusted_terms, limit)
            group_size += len(rotations)
            if group_size > limit:
                raise ValueError("too many Hamiltonian-group rotations")
            if any(not commute(a[0]["label"], b[0]["label"]) for a in rotations for b in rotations):
                raise ValueError("terms in an explicit Hamiltonian group must commute")
            parsed.append(rotations)
        flattened, stage_rows = [], []
        for item in stages:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("a splitting stage is (group_index, multiplier)")
            index, weight = item
            if type(index) is not int or not 0 <= index < len(parsed):
                raise ValueError("splitting references an unknown group")
            weight = float(weight)
            if not math.isfinite(weight):
                raise ValueError("non-finite splitting weight")
            flattened.extend((term, weight * angle) for term, angle in parsed[index])
            if len(flattened) > limit:
                raise ValueError("ProductFormula.build_step() returned too many rotations")
            stage_rows.append([index, weight])
        recipe.update(
            {
                "grouping": "explicit_commuting_blocks",
                "groups": [[{"label": t["label"], "angle": a} for t, a in g] for g in parsed],
                "splitting": stage_rows,
            }
        )
        if type(raw.get("merge_adjacent", False)) is not bool:
            raise ValueError("merge_adjacent must be Boolean")
        recipe["merge_adjacent"] = raw.get("merge_adjacent", False)
        if recipe["merge_adjacent"]:
            merged = []
            for term, angle in flattened:
                if merged and merged[-1][0]["label"] == term["label"]:
                    previous, value = merged.pop()
                    merged.append((previous, value + angle))
                else:
                    merged.append((term, angle))
            flattened = [(t, a) for t, a in merged if a != 0]
    else:
        flattened = _pairs(raw, trusted_terms, limit)
        recipe.update({"grouping": "individual_pauli_rotations", "splitting": "flat_schedule"})
    if not flattened or any(not math.isfinite(a) for _, a in flattened):
        raise ValueError("PF must contain finite, nonempty rotations after expansion")
    recipe["rotation_schedule"] = [{"label": t["label"], "angle": a} for t, a in flattened]
    return flattened, recipe


def _pairs(raw, trusted, limit):
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("ProductFormula.build_step() must return a nonempty list or tuple")
    if len(raw) > limit:
        raise ValueError("ProductFormula.build_step() returned too many rotations")
    result = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"rotation {index} must be a (term, angle) pair")
        term, angle = item
        label = str(term.get("label", "")) if isinstance(term, dict) else str(term)
        if label not in trusted:
            raise ValueError(f"rotation {index} references unknown Pauli term {label!r}")
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError(f"rotation {index} has a non-finite angle")
        result.append((dict(trusted[label]), angle))
    return result


def classify_groups(recipe, terms, step_time):
    """Recognize symmetric consistent group schedules; never trust an order label."""
    if recipe["grouping"] != "explicit_commuting_blocks":
        return None
    totals = {}
    for index, weight in recipe["splitting"]:
        for row in recipe["groups"][index]:
            totals.setdefault(row["label"], []).append(weight * row["angle"])
    consistent = all(
        math.isclose(
            math.fsum(totals.get(t["label"], [])),
            t["coefficient"] * step_time,
            rel_tol=1e-10,
            abs_tol=1e-14,
        )
        for t in terms
    )
    symmetric = recipe["splitting"] == list(reversed(recipe["splitting"]))
    recipe["first_order_generator_matches_hamiltonian"] = consistent
    recipe["symmetric_splitting"] = symmetric
    if consistent and symmetric:
        return {
            "formula_name": "symmetric_commuting_group_splitting",
            "formula_order": 2,
            "term_order_strategy": "explicit_commuting_groups",
        }
    return None


def transformed_terms(problem, frame):
    return [
        {
            **term_fields(label, sign * float(t["coefficient"])),
            "original_label": t["label"],
            "frame_sign": sign,
        }
        for t in problem["terms"]
        for sign, label in [clifford.conjugate(t["label"], frame)]
    ]
