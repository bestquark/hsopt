#!/usr/bin/env python3
"""Post-hoc, first-order synthesis sensitivities; never scores or edits a campaign.

The full-cohort calculation substitutes the cached H ground input for the ideal
product-formula eigenvector. Small explicitly selected cases check that
approximation against the actual step spectrum and finite differences.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals, schur

from hsopt.campaign.protocol import bind_policy
from hsopt.evaluation import compiled_unitary as ev
from hsopt.evaluation.ground_branch import cached_reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PauliActions:
    """O(2**n) workspace, without a per-term index/phase cache."""

    def __init__(self, n):
        self.n = n
        self.indices = np.arange(1 << n, dtype=np.int64)
        self.parity = np.array([i.bit_count() % 2 for i in self.indices], dtype=np.int8)

    def apply(self, label, value):
        x = sum(1 << k for k, c in enumerate(reversed(label)) if c in "XY")
        z = sum(1 << k for k, c in enumerate(reversed(label)) if c in "YZ")
        source = self.indices ^ x
        phase = (1j ** label.count("Y")) * (1 - 2 * self.parity[source & z])
        if value.ndim == 2:
            phase = phase[:, None]
        return phase * value[source]

    def expectation(self, label, state):
        value = np.vdot(state, self.apply(label, state))
        if abs(value.imag) > 1e-10:
            raise ValueError("Hermitian Pauli expectation is not real")
        return float(value.real)


def propagate(actions, schedule, state, perturbation=None):
    """+i alpha P convention; an error is inserted AFTER the selected gadget."""
    value = state.copy()
    for j, (term, alpha) in enumerate(schedule):
        value = math.cos(alpha) * value + 1j * math.sin(alpha) * actions.apply(
            term["label"], value
        )
        if perturbation is not None and j == perturbation[0]:
            _, axis, delta = perturbation
            sign, label = ev._conjugated_pivot_pauli(term["label"], axis)
            value = math.cos(delta) * value + 1j * math.sin(delta) * sign * actions.apply(
                label, value
            )
    return value


def response(actions, schedule, state):
    """Return dimensionless signed eigenphase gradients at each rotation.

    Exact for an isolated step eigenvector. For a cached H eigenvector these
    are a ground-input approximation, rather than certified eigenphase gradients.
    X/Y/Z are errors on the compiled Rz pivot, conjugated by its real scaffold.
    A phase-aligned infinitesimal SU(2) error has norm ||delta||_2; its worst
    first-order phase change is ||b_j||_2 ||delta||_2.
    """
    value = state.copy()
    derivatives = []
    for term, alpha in schedule:
        value = math.cos(alpha) * value + 1j * math.sin(alpha) * actions.apply(
            term["label"], value
        )
        b = []
        for axis in "XYZ":
            sign, label = ev._conjugated_pivot_pauli(term["label"], axis)
            if abs(sign.imag) > 1e-12:
                raise ValueError("nonreal Hermitian scaffold sign")
            b.append(sign.real * actions.expectation(label, value))
        if np.linalg.norm(b) > 1 + 1e-9:
            raise ValueError("single-qubit Bloch response exceeds one")
        derivatives.append(b)
    return np.asarray(derivatives), value


def exact_step_target(actions, schedule, ground):
    """Only for small offline validation, never an evaluator fallback."""
    if actions.n > 8:
        raise ValueError("dense derivative validation is limited to eight qubits")
    unitary = propagate(actions, schedule, np.eye(ground.size, dtype=complex))
    triangular, vectors = schur(unitary, output="complex")
    spectrum = np.diag(triangular)
    weights = np.abs(vectors.conj().T @ ground) ** 2
    index = int(weights.argmax())
    phases = np.angle(spectrum * spectrum[index].conjugate())
    group = np.flatnonzero(np.abs(phases) < 1e-10)
    basis = vectors[:, group]
    target = basis @ (basis.conj().T @ ground)
    probability = float(np.vdot(target, target).real)
    target /= math.sqrt(probability)
    phase = float(np.angle(spectrum[index]))
    residual = float(np.linalg.norm(unitary @ target - np.exp(1j * phase) * target))
    return target, phase, probability, residual, len(group)


def validate_derivatives(actions, schedule, ground, approximate):
    target, phase, probability, residual, multiplicity = exact_step_target(
        actions, schedule, ground
    )
    exact, _ = response(actions, schedule, target)
    difference = approximate - exact
    result = {
        "qubits": actions.n,
        "step_eigenstate_ground_input_probability": probability,
        "step_eigenvector_residual": residual,
        "selected_phase_multiplicity": multiplicity,
        "ground_input_proxy_max_absolute_gradient_difference": float(abs(difference).max()),
        "ground_input_proxy_relative_gradient_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(exact), 1e-30)
        ),
        "angle_proxy_relative_gradient_l2_error": float(
            np.linalg.norm(difference[:, 2]) / max(np.linalg.norm(exact[:, 2]), 1e-30)
        ),
        "finite_differences": [],
    }
    # A degenerate eigenphase may split: its projected expectation is not a
    # unique branch derivative. Do not manufacture a finite-difference match.
    if multiplicity != 1:
        result["finite_difference_skip"] = "degenerate eigenphase requires subspace perturbation"
        return result
    selected = sorted({0, len(schedule) // 2})
    for j in selected:
        for axis_index, axis in enumerate("XYZ"):
            differences = []
            for delta in (1e-5, 1e-6):
                phases = []
                for sign in (-1, 1):
                    perturbed = propagate(
                        actions, schedule, np.eye(ground.size, dtype=complex),
                        (j, axis, sign * delta),
                    )
                    values = eigvals(perturbed)
                    offset = np.angle(values * np.exp(-1j * phase))
                    phases.append(float(offset[np.argmin(abs(offset))]))
                differences.append((phases[1] - phases[0]) / (2 * delta))
            analytic = float(exact[j, axis_index])
            result["finite_differences"].append({
                "rotation": j, "axis": axis, "analytic_phase_gradient": analytic,
                "central_gradients": differences,
                "absolute_error_at_1e_6": abs(differences[-1] - analytic),
            })
    return result


def collect_case(case, root, policy_path, expected, validate):
    source, reference_path = (root / case[key] for key in ("source", "reference"))
    if sha(source) != expected["problem_sha256"]:
        raise ValueError(f"Hamiltonian snapshot changed for {case['id']}")
    problem = ev.load_problem(source)
    _, ground = cached_reference(problem, json.loads(reference_path.read_text()))
    with tempfile.TemporaryDirectory(prefix="hsopt-sensitivity-") as tmp:
        bound = Path(tmp) / "bound.py"
        bind_policy(policy_path, case["parameters"], bound)
        policy = ev.load_policy(bound)
        formula, schedule, metrics = ev._product_formula_schedule(problem, policy)
    dt = metrics["step_time"]
    actions = PauliActions(problem["n_qubits"])
    gradients, final = response(actions, schedule, ground)
    amplitude = np.vdot(ground, final)
    phase = float(np.angle(amplitude))
    grouped = defaultdict(list)
    for j, (term, alpha) in enumerate(schedule):
        grouped[term["label"]].append((j, float(formula.synthesis_epsilon(term, alpha))))
    largest_coefficient = max(abs(t["coefficient"]) for t in problem["terms"])
    rows = []
    for term in problem["terms"]:
        occurrences = grouped[term["label"]]
        epsilons = [epsilon for _, epsilon in occurrences]
        if not np.allclose(epsilons, epsilons[0], atol=0, rtol=1e-12):
            raise ValueError("this term-level comparison requires tied occurrence tolerances")
        indices = [j for j, _ in occurrences]
        angle_weight = float(np.mean(abs(gradients[indices, 2])))
        su2_weight = float(np.mean(np.linalg.norm(gradients[indices], axis=1)))
        # Tolerances shared by repeated term occurrences: error weights SUM,
        # while equal logarithmic cost weights scale with their multiplicity.
        rows.append({
            "case_id": case["id"], "qubits": problem["n_qubits"],
            "label": term["label"], "support_size": term["support_size"],
            "off_diagonal": int(bool(term["x_count"] + term["y_count"])),
            "abs_coefficient_hartree": abs(term["coefficient"]),
            "relative_coefficient": abs(term["coefficient"]) / largest_coefficient,
            "occurrences": len(indices), "step_time": dt,
            "epsilon": epsilons[0], "baseline_epsilon": expected["baseline_epsilon"],
            "epsilon_ratio": epsilons[0] / expected["baseline_epsilon"],
            "angle_weight": angle_weight, "su2_weight": su2_weight,
            "s_angle_hartree": angle_weight / dt, "s_su2_hartree": su2_weight / dt,
            "zero_timestep_angle_weight": abs(actions.expectation(term["label"], ground)),
        })
    info = {
        "case_id": case["id"], "problem_sha256": sha(source),
        "reference_sha256": sha(reference_path), "qubits": problem["n_qubits"],
        "terms": len(rows), "rotations": len(schedule),
        "rotation_schedule_sha256": metrics["rotation_schedule_sha256"],
        "step_time": dt, "ground_input_survival_probability": float(abs(amplitude) ** 2),
        "ground_input_step_eigenvector_residual": float(
            np.linalg.norm(final - np.exp(1j * phase) * ground)
        ),
        "assumption": "cached H ground input substituted for ideal PF eigenvector",
    }
    if validate:
        info["validation"] = validate_derivatives(actions, schedule, ground, gradients)
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Frozen input suite root")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-case", action="append", default=[])
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output_dir.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("write analysis outside the campaign")
    suite = json.loads((root / "suite.json").read_text())
    snapshot = json.loads(args.snapshot.read_text())
    if sha(args.policy) != snapshot["policy_sha256"]:
        raise ValueError("policy differs from frozen precision-plot snapshot")
    expected = {case["case_id"]: case for case in snapshot["cases"]}
    if set(expected) != {case["id"] for case in suite["cases"]}:
        raise ValueError("suite differs from precision-plot snapshot")
    if suite.get("energy_target") != "ground_branch" or suite.get("kind") != "absolute":
        raise ValueError("this analysis requires the ground-input absolute protocol")
    unknown = set(args.validate_case) - set(expected)
    if unknown:
        raise ValueError(f"unknown derivative validation cases: {unknown}")
    output.mkdir(parents=True, exist_ok=True)
    all_rows, cases = [], []
    started = time.perf_counter()
    for index, case in enumerate(suite["cases"]):
        before = time.perf_counter()
        rows, info = collect_case(
            case, root, args.policy, expected[case["id"]], case["id"] in args.validate_case
        )
        info["wall_seconds"] = time.perf_counter() - before
        all_rows.extend(rows)
        cases.append(info)
        print(f"{index + 1}/{len(expected)} {case['id']} {info['wall_seconds']:.2f}s", flush=True)
    table = output / "term_sensitivities.csv.gz"
    with gzip.open(table, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "schema": 1, "analysis": "local_synthesis_sensitivity_ground_input_proxy_v1",
        "policy_sha256": sha(args.policy), "precision_snapshot_sha256": sha(args.snapshot),
        "suite_sha256": sha(root / "suite.json"), "script_sha256": sha(__file__),
        "source_seed": snapshot["source_seed"], "source_iteration": snapshot["source_iteration"],
        "source_score": snapshot["score"], "molecules": len(cases), "terms": len(all_rows),
        "wall_seconds": time.perf_counter() - started,
        "compiler_or_campaign_evaluator_run": False,
        "small_dense_spectra_only_for_derivative_validation": args.validate_case,
        "error_model": "independent infinitesimal gate errors; phase-aligned SU(2)",
        "not_certified": "No finite-error remainder, arbitrary U(2) phase, QPE weight, or T95 guarantee",
        "cases": cases,
    }
    (output / "sensitivity_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {table}", flush=True)


if __name__ == "__main__":
    main()
