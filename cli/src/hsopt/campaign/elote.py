"""Prepare and inspect explicit product formulas without changing old campaigns."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..evaluation import compiled_unitary as ev
from ..evaluation import ground_branch
from ..evaluation.elote import digest
from . import calibration

SCHEMA = "hsopt-nested-baseline-v1"
TEMPLATE = Path(__file__).resolve().parents[4] / "baseline/initial_nested_pf.py"


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bound(problem, r, epsilon):
    policy = ev.load_policy(TEMPLATE)
    policy.ProductFormula = type(
        "ProductFormula",
        (policy.ProductFormula,),
        {
            "baseline_parameters": {problem["name"]: (r, epsilon, 0.0)},
        },
    )
    return policy


def ideal_energy(problem, *, n_steps, exact_energy, exact_target, backend_name, xp):
    unitary, metrics = ev._build_exact_product_formula_step(
        problem,
        bound(problem, n_steps, 1e-10),
        backend_name=backend_name,
        xp=xp,
        matrix_free=backend_name == "cupy-lanczos",
    )
    if backend_name == "cupy-lanczos":
        from ..evaluation.gpu_circuit import ground_lanczos, certify_selected_phase

        rays, vectors, diagnostics = ground_lanczos(unitary, problem, metrics, exact_target)
        diagnostics["phase_guard"] = certify_selected_phase(rays, n_steps)
    else:
        rays, vectors = np.linalg.eig(unitary)
        diagnostics = {"verification": "small_system_dense_test_oracle"}
    vectors /= xp.linalg.norm(vectors, axis=0, keepdims=True)
    probabilities = xp.abs(vectors.conj().T @ exact_target) ** 2
    peak = ground_branch.select_peak(
        xp.angle(rays),
        probabilities,
        n_steps=n_steps,
        target_time=problem["target_time"],
        identity_coefficient=problem["identity_coefficient"],
    )
    energy = peak["compiled_pf_energy_hartree"]
    return {
        "n_steps": n_steps,
        "unsynthesized_pf_energy_hartree": energy,
        "absolute_trotter_error_hartree": abs(energy - exact_energy),
        "selected_exact_ground_overlap": peak["selected_exact_ground_overlap"],
        "numerical_verification": diagnostics,
    }


def prepare(
    problem_path, reference_path, output, *, baseline, backend="cupy-lanczos", max_steps=1 << 20
):
    """Fresh beta for THIS grouping; reuse H/reference, never old-order beta.

    Only the nested profile is new. The historical Pauli policies/calibrations
    remain byte-for-byte intact. An unchanged compiled iteration 0 must pass
    before the new record is marked usable for mutations.
    """
    if baseline not in ("loose", "tight"):
        raise ValueError("choose loose or tight")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("baseline output directory must be empty")
    problem_path, reference_path = Path(problem_path).resolve(), Path(reference_path).resolve()
    problem, reference = ev.load_problem(problem_path), _read(reference_path)
    exact_energy, _ = ground_branch.cached_reference(problem, reference)
    backend_name, xp = ev.array_backend(backend)
    _, rotations, recipe = ev._product_formula_schedule(problem, bound(problem, 1, 1e-10))
    # One forward traversal of the actual commuting groups. Its flattened
    # Strang formula is exactly the same ideal unitary (including the center).
    ordered = [
        dict(t, coefficient=a / problem["target_time"])
        for g in recipe["elote"]["groups"]
        for t, a in [(gterm, gterm["angle"]) for gterm in g]
    ]
    if baseline == "tight":
        beta = calibration._empirical_reference_beta(
            problem,
            exact_reference=reference,
            backend_name=backend_name,
            xp=xp,
            reference_error_hartree=calibration.PAPER_BETA_REFERENCE_ERROR_HARTREE,
            max_steps=max_steps,
            step_evaluator=ideal_energy,
        )
    else:
        beta = calibration._double_commutator_bound(
            problem,
            reference_error_hartree=calibration.PAPER_BETA_REFERENCE_ERROR_HARTREE,
            ordered_terms=ordered,
        )
    allocation, derivation = calibration.optimize_paper_error_allocation(
        {**problem, "terms": ordered},
        beta=beta["reference_trotter_number_beta"],
        reference_error_hartree=calibration.PAPER_BETA_REFERENCE_ERROR_HARTREE,
        total_error_hartree=calibration.BENCHMARK_TOTAL_ERROR_HARTREE,
        rotations_per_step=len(rotations),
    )
    _, final_rotations, _ = ev._product_formula_schedule(
        problem, bound(problem, allocation["n_steps"], allocation["uniform_epsilon"])
    )
    if len(final_rotations) != len(rotations):
        allocation, derivation = calibration.optimize_paper_error_allocation(
            {**problem, "terms": ordered},
            beta=beta["reference_trotter_number_beta"],
            reference_error_hartree=calibration.PAPER_BETA_REFERENCE_ERROR_HARTREE,
            total_error_hartree=calibration.BENCHMARK_TOTAL_ERROR_HARTREE,
            rotations_per_step=len(final_rotations),
        )
    allocation.update(
        {
            "term_order_strategy": "greedy_disjoint_spin_orbital_blocks",
            "step_structure": "symmetric_commuting_group_splitting",
        }
    )
    # Explicitly verify the allocated timestep, rather than assuming 1/r²
    # scaling always implies that this particular beta gives a passing start.
    ideal = ideal_energy(
        problem,
        n_steps=allocation["n_steps"],
        exact_energy=exact_energy,
        exact_target=xp.asarray([complex(a, b) for a, b in reference["exact_ground_state"]]),
        backend_name=backend_name,
        xp=xp,
    )
    if ideal["absolute_trotter_error_hartree"] > allocation["trotter_error_hartree"]:
        raise RuntimeError(
            "nested baseline misses its allocated Trotter budget; no usable baseline saved"
        )
    record = {
        "schema": SCHEMA,
        "baseline": baseline,
        "profile": "fermionic_nested",
        "problem": problem["name"],
        "problem_sha256": ev._sha256(problem_path),
        "reference_sha256": ev._sha256(reference_path),
        "reference": beta,
        "instantiated_policy": allocation,
        "allocation_derivation": derivation,
        "ideal_allocated_step_validation": ideal,
        "baseline_template_sha256": ev._sha256(TEMPLATE),
        "decomposition_sha256": recipe["elote"]["fermionic_decomposition_sha256"],
        "paper": {
            "doi": "10.1073/pnas.1619152114",
            "nesting": "Appendix E.2.b",
            "reproduction_scope": "greedy orbital-disjoint Hermitian blocks, PF2, Eq19/Eq34; no PAR or proprietary LIQUi optimizer",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    table = {
        problem["name"]: {
            **allocation,
            "reference_trotter_number_beta": beta["reference_trotter_number_beta"],
        }
    }
    from .protocol import bind_policy

    bind_policy(TEMPLATE, table, output / "initial_pf.py")
    compiled = ev.evaluate_molecule(
        problem_path,
        output / "initial_pf.py",
        backend=backend_name,
        reference_path=reference_path,
        energy_target="ground_branch",
        matrix_free=backend_name == "cupy-lanczos",
    )
    _write(output / "baseline_validation.json", compiled)
    if not compiled["passes_chemical_accuracy"]:
        raise RuntimeError("compiled nested iteration 0 failed accuracy; calibration NOT sealed")
    record["compiled_iteration_zero_passed"] = True
    record["validation_contract_version"] = compiled.get(
        "evaluation_contract_version", ground_branch.CONTRACT_VERSION
    )
    record["baseline_policy_sha256"] = ev._sha256(output / "initial_pf.py")
    record["calibration_sha256"] = digest(record)
    _write(output / "calibration.json", record)
    return record


def verify(record, problem_path, baseline, *, require_current_verifier=True):
    payload = dict(record)
    checksum = payload.pop("calibration_sha256", None)
    if (
        record.get("schema") != SCHEMA
        or checksum != digest(payload)
        or record.get("baseline") != baseline
        or record.get("problem_sha256") != ev._sha256(Path(problem_path))
        or record.get("baseline_template_sha256") != ev._sha256(TEMPLATE)
        or record.get("compiled_iteration_zero_passed") is not True
    ):
        raise ValueError(
            "nested calibration is stale, unvalidated, or for another problem/baseline"
        )
    if (
        require_current_verifier
        and record.get("validation_contract_version") != ground_branch.CONTRACT_VERSION
    ):
        raise ValueError(
            "nested validation uses an older verifier; refresh its compiled iteration zero"
        )


def refresh_validation(problem_path, reference_path, output, *, backend="cupy-lanczos"):
    """Recheck the unchanged policy after a verifier upgrade, reusing its beta.

    No parameters are retuned and no historical trajectory is rewritten. This
    operates only on a sealed preparation directory, before its new campaign.
    """
    problem_path, reference_path, output = Path(problem_path), Path(reference_path), Path(output)
    record = _read(output / "calibration.json")
    verify(record, problem_path, record["baseline"], require_current_verifier=False)
    if (
        ev._sha256(output / "initial_pf.py") != record["baseline_policy_sha256"]
        or ev._sha256(reference_path) != record["reference_sha256"]
    ):
        raise ValueError("cannot refresh a changed policy or exact reference")
    from ..evaluation.elote import fermionic_blocks

    if fermionic_blocks(ev.load_problem(problem_path))["sha256"] != record["decomposition_sha256"]:
        raise ValueError("cannot refresh a changed Hamiltonian decomposition")
    previous = _read(output / "baseline_validation.json")
    compiled = ev.evaluate_molecule(
        problem_path,
        output / "initial_pf.py",
        backend=backend,
        reference_path=reference_path,
        energy_target="ground_branch",
        matrix_free=True,
    )
    if not compiled["passes_chemical_accuracy"]:
        raise RuntimeError("refreshed compiled iteration zero failed accuracy")
    record.setdefault("validation_history", []).append(
        {
            "previous_contract_version": previous.get("evaluation_contract_version"),
            "previous_validation_sha256": ev._sha256(output / "baseline_validation.json"),
            "previous_calibration_sha256": record.pop("calibration_sha256"),
            "calibration_parameters_reused": True,
        }
    )
    record["validation_contract_version"] = compiled["evaluation_contract_version"]
    record["calibration_sha256"] = digest(record)
    _write(output / "baseline_validation.json", compiled)
    _write(output / "calibration.json", record)
    return record


def load_baselines(source, *, baseline, calibration_path, calibration_root):
    if calibration_path is None and calibration_root is None:
        raise ValueError("provide --calibration or --calibration-root")
    paths = (
        [Path(calibration_path)]
        if calibration_path
        else sorted(Path(calibration_root).rglob("*.json"))
    )
    catalog = {}
    for path in paths:
        payload = _read(path)
        if (
            isinstance(payload, dict)
            and payload.get("schema") == SCHEMA
            and payload.get("baseline") == baseline
        ):
            key = payload["problem_sha256"]
            if (
                key in catalog
                and catalog[key][1]["calibration_sha256"] != payload["calibration_sha256"]
            ):
                raise ValueError("multiple distinct nested calibrations for one Hamiltonian")
            catalog[key] = (path, payload)
    rows, used = {}, []
    problems = calibration._problem_paths(Path(source))
    for species, weight, problem_path, _payload in problems:
        key = ev._sha256(problem_path)
        if key not in catalog:
            raise ValueError(
                f"missing freshly prepared nested {baseline} calibration: {problem_path}"
            )
        path, payload = catalog[key]
        verify(payload, problem_path, baseline)
        allocation = dict(payload["instantiated_policy"])
        if weight is not None:
            if not math.isfinite(weight) or weight == 0:
                raise ValueError("reaction needs finite nonzero coefficients")
            target = calibration.BENCHMARK_TOTAL_ERROR_HARTREE / (len(problems) * abs(weight))
            problem = ev.load_problem(problem_path)
            modeled = {**problem, "terms": [None] * allocation["nonidentity_terms"]}
            allocation, _ = calibration.optimize_paper_error_allocation(
                modeled,
                beta=payload["reference"]["reference_trotter_number_beta"],
                reference_error_hartree=payload["reference"]["reference_error_hartree"],
                total_error_hartree=target,
                rotations_per_step=payload["instantiated_policy"]["scheduled_rotations_per_step"],
            )
            allocation.update(
                {
                    "reaction_coefficient": weight,
                    "species": species,
                    "reaction_species_count": len(problems),
                    "reaction_error_allocation": "equal_absolute_contribution_under_triangle_inequality",
                }
            )
        if payload["problem"] in rows:
            raise ValueError("duplicate evaluator-visible problem name")
        rows[payload["problem"]] = {
            **allocation,
            "baseline_source": "reiher_disjoint_orbital_nesting_eq19_eq34_v1",
            "baseline_variant": baseline,
            "problem_sha256": key,
            "reference_trotter_number_beta": payload["reference"]["reference_trotter_number_beta"],
            "calibration_sha256": payload["calibration_sha256"],
            "baseline": baseline,
        }
        used.append(path)
    return rows, used


def describe(problem_path, policy_path):
    """Resolve all ingredients without compiling or running a GPU solver."""
    problem, policy = ev.load_problem(problem_path), ev.load_policy(policy_path)
    formula, rotations, metrics = ev._product_formula_schedule(problem, policy)
    recipe = metrics["elote"]
    tolerances = [float(formula.synthesis_epsilon(dict(t), a)) for t, a in rotations]
    if any(not math.isfinite(e) or not 0 < e < 1 for e in tolerances):
        raise ValueError("invalid synthesis tolerance")
    recipe.update(
        {
            "synthesis_tolerances": tolerances,
            "policy_sha256": ev._sha256(policy_path),
            "problem_sha256": ev._sha256(problem_path),
        }
    )
    recipe["resolved_recipe_sha256"] = digest(recipe)
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--problem", type=Path, required=True)
    prep.add_argument("--exact-reference", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--baseline", choices=("loose", "tight"), required=True)
    prep.add_argument("--max-steps", type=int, default=1 << 20)
    refresh = sub.add_parser("refresh")
    refresh.add_argument("--problem", type=Path, required=True)
    refresh.add_argument("--exact-reference", type=Path, required=True)
    refresh.add_argument("--output-dir", type=Path, required=True)
    inspect = sub.add_parser("describe")
    inspect.add_argument("--problem", type=Path, required=True)
    inspect.add_argument("--policy-file", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        record = prepare(
            args.problem,
            args.exact_reference,
            args.output_dir,
            baseline=args.baseline,
            max_steps=args.max_steps,
        )
        print(
            json.dumps(
                {"output": str(args.output_dir), "calibration_sha256": record["calibration_sha256"]}
            )
        )
    elif args.action == "refresh":
        record = refresh_validation(args.problem, args.exact_reference, args.output_dir)
        print(
            json.dumps(
                {"output": str(args.output_dir), "calibration_sha256": record["calibration_sha256"]}
            )
        )
    else:
        _write(args.output, describe(args.problem, args.policy_file))
    return 0
