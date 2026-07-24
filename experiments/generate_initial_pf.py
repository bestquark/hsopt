"""Generate an immutable iteration-0 policy from paper-bound data.

The Hamiltonian-dependent beta(H) comes from the exact Pauli evaluation of the
Reiher et al. supplement's Eq. (19) coefficient bound. This command verifies
the sealed result and source-Hamiltonian checksum before embedding the fixed
PF2/Strang policy in ``initial_pf.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from hsopt.paper_baseline import (
    BENCHMARK_TOTAL_ERROR_HARTREE,
    PAPER_BETA_REFERENCE_ERROR_HARTREE,
    PAPER_ERROR_CONSTRAINT,
    PAPER_FORMULA_ORDER,
    PAPER_PHASE_ESTIMATION_ALPHA,
    PAPER_SYNTHESIS_DELTA_0,
    PAPER_SYNTHESIS_GAMMA,
    PAPER_TERM_ORDER,
    optimize_paper_error_allocation,
    verify_calibration,
)


PAPER_CONTRACT: dict[str, Any] = {
    "schema": 6,
    "baseline_id": "reiher_wiebe_svore_wecker_troyer_2017_eq19_eq34_rigorous_pf2",
    "citation": {
        "authors": [
            "Markus Reiher",
            "Nathan Wiebe",
            "Krysta M. Svore",
            "Dave Wecker",
            "Matthias Troyer",
        ],
        "title": "Elucidating Reaction Mechanisms on Quantum Computers",
        "journal": "Proceedings of the National Academy of Sciences",
        "year": 2017,
        "doi": "10.1073/pnas.1619152114",
        "arxiv": "1605.03590",
    },
    "paper_method": {
        "formula_order": PAPER_FORMULA_ORDER,
        "formula_name": "symmetric_second_order_trotter_suzuki",
        "step_structure": "forward_half_except_center_full_then_reverse_half",
        "term_order_strategy": PAPER_TERM_ORDER,
        "rotation_synthesis": "clifford_plus_t_pygridsynth_at_iteration_zero",
        "beta_reference_error_hartree": PAPER_BETA_REFERENCE_ERROR_HARTREE,
        "beta_selection": "exact Eq. (19) Pauli double-commutator coefficient bound",
        "phase_estimation_alpha": PAPER_PHASE_ESTIMATION_ALPHA,
        "synthesis_gamma": PAPER_SYNTHESIS_GAMMA,
        "synthesis_delta_0": PAPER_SYNTHESIS_DELTA_0,
        "allocation_equation": "supplement Eq. (34)",
        "error_constraint": PAPER_ERROR_CONSTRAINT,
        "synthesis_energy_bound": "(2M - 1) * Delta_synth / step_time",
    },
    "benchmark_adaptation": {
        "total_error_target_hartree": BENCHMARK_TOTAL_ERROR_HARTREE,
        "total_error_target": "1 kcal/mol",
        "target_time_hartree_inverse": 1.0,
        "allocation_cost": "supplement Eq. (34) with rigorous gamma=4 and delta_0=11",
        "realized_t_count": "direct PyGridSynth compilation during iteration 0",
    },
    "scope_note": (
        "beta(H) is obtained from the paper's rigorous Eq. (19) "
        "double-commutator coefficient bound for this Pauli decomposition. "
        "The separately cached exact energy is used only by the evaluator and "
        "does not enter beta, the Trotter step count, or the synthesis tolerance."
    ),
}
PAPER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(PAPER_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _problem_paths(
    source: Path,
) -> list[tuple[str | None, float | None, Path, dict[str, Any]]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "n_qubits" in payload and "terms" in payload:
        return [(None, None, source, payload)]
    rows: list[tuple[str | None, float | None, Path, dict[str, Any]]] = []
    for item in payload["terms"]:
        problem_path = (source.parent / str(item["problem"])).resolve()
        problem = json.loads(problem_path.read_text(encoding="utf-8"))
        rows.append(
            (
                str(item["species"]),
                float(item["coefficient"]),
                problem_path,
                problem,
            )
        )
    if not rows:
        raise ValueError(f"{source}: reaction has no Hamiltonians")
    return rows


def _read_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_calibration(payload)
    if payload["calibration_mode"] != "hamiltonian_bound":
        raise ValueError(f"{path}: expected canonical hamiltonian_bound calibration")
    return payload


def _calibration_catalog(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    catalog: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            verify_calibration(payload)
            if payload["calibration_mode"] != "hamiltonian_bound":
                continue
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        problem_sha256 = str(payload["problem_sha256"])
        if problem_sha256 in catalog:
            prior_path, prior = catalog[problem_sha256]
            if str(prior["calibration_sha256"]) != str(payload["calibration_sha256"]):
                raise ValueError(
                    f"conflicting calibrations for {problem_sha256}: {prior_path} and {path}"
                )
            continue
        catalog[problem_sha256] = (path, payload)
    if not catalog:
        raise ValueError(f"no sealed paper-baseline calibrations found under {root}")
    return catalog


def _policy_row(
    *,
    problem_path: Path,
    problem: Mapping[str, Any],
    calibration_path: Path,
    calibration: Mapping[str, Any],
    species: str | None,
    reaction_coefficient: float | None,
    reaction_species_count: int,
) -> tuple[str, dict[str, Any]]:
    verify_calibration(calibration, problem_path=problem_path)
    if calibration["calibration_mode"] != "hamiltonian_bound":
        raise ValueError(f"{calibration_path}: not a Hamiltonian-bound baseline")
    if reaction_coefficient is None:
        species_error_target = BENCHMARK_TOTAL_ERROR_HARTREE
    else:
        if reaction_species_count <= 0 or reaction_coefficient == 0.0:
            raise ValueError("reaction calibration requires nonzero signed terms")
        species_error_target = BENCHMARK_TOTAL_ERROR_HARTREE / (
            reaction_species_count * abs(reaction_coefficient)
        )
    policy, _derivation = optimize_paper_error_allocation(
        problem,
        beta=float(calibration["reference"]["reference_trotter_number_beta"]),
        reference_error_hartree=float(calibration["reference"]["reference_error_hartree"]),
        total_error_hartree=species_error_target,
    )
    if reaction_coefficient is None and policy != calibration["instantiated_policy"]:
        raise ValueError(
            f"{calibration_path}: generator Eq. (34) policy does not match "
            "the sealed calibration policy"
        )
    if reaction_coefficient is not None:
        policy["reaction_coefficient"] = reaction_coefficient
        policy["reaction_species_count"] = reaction_species_count
        policy["reaction_error_allocation"] = (
            "equal_absolute_contribution_under_triangle_inequality"
        )
        policy["reaction_total_error_target_hartree"] = BENCHMARK_TOTAL_ERROR_HARTREE
    if not math_isclose(float(problem["target_time"]), float(policy["target_time"])):
        raise ValueError(
            f"{problem_path}: source target_time={problem['target_time']} differs from "
            f"calibrated target_time={policy['target_time']}"
        )
    name = str(problem["name"])
    policy.update(
        {
            "baseline_source": str(PAPER_CONTRACT["baseline_id"]),
            "calibration_mode": str(calibration["calibration_mode"]),
            "calibration_sha256": str(calibration["calibration_sha256"]),
            "problem_sha256": str(calibration["problem_sha256"]),
            "reference_trotter_number_beta": float(
                calibration["reference"]["reference_trotter_number_beta"]
            ),
            "reference_beta_error_hartree": float(
                calibration["reference"]["reference_error_hartree"]
            ),
            "bound_method": str(calibration["reference"]["bound_method"]),
            "bound_equation": str(calibration["reference"]["bound_equation"]),
            "bound_term_count": int(calibration["reference"]["bound_term_count"]),
            "double_commutator_sum_upper": float(
                calibration["reference"]["double_commutator_sum_upper"]
            ),
        }
    )
    if species is not None:
        policy["species"] = species
    return name, policy


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-15 * max(1.0, abs(left), abs(right))


def load_baselines(
    source: Path,
    *,
    calibration_path: Path | None,
    calibration_root: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    problems = _problem_paths(source)
    if calibration_path is not None:
        if len(problems) != 1:
            raise ValueError("--calibration can only be used with one Hamiltonian")
        direct = _read_calibration(calibration_path)
        catalog = {str(direct["problem_sha256"]): (calibration_path, direct)}
    elif calibration_root is not None:
        catalog = _calibration_catalog(calibration_root)
    else:
        raise ValueError("provide --calibration or --calibration-root")

    rows: dict[str, dict[str, Any]] = {}
    used: list[Path] = []
    reaction_species_count = sum(species is not None for species, *_rest in problems)
    for species, reaction_coefficient, problem_path, problem in problems:
        digest = _sha256(problem_path)
        if digest not in catalog:
            raise ValueError(f"no calibration for {problem_path} (SHA-256 {digest})")
        resolved_calibration, calibration = catalog[digest]
        name, row = _policy_row(
            problem_path=problem_path,
            problem=problem,
            calibration_path=resolved_calibration,
            calibration=calibration,
            species=species,
            reaction_coefficient=reaction_coefficient,
            reaction_species_count=reaction_species_count,
        )
        if name in rows:
            raise ValueError(f"duplicate evaluator-visible problem name {name!r}")
        rows[name] = row
        used.append(resolved_calibration)
    return rows, used


def format_baselines(rows: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["{"]
    for name, row in sorted(rows.items()):
        lines.append(f"    {json.dumps(name)}: {{")
        for key, value in sorted(row.items()):
            literal = json.dumps(value) if isinstance(value, str) else repr(value)
            lines.append(f"        {json.dumps(key)}: {literal},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def render(
    source: Path,
    *,
    calibration_path: Path | None = None,
    calibration_root: Path | None = None,
) -> str:
    rows, _calibration_paths = load_baselines(
        source,
        calibration_path=calibration_path,
        calibration_root=calibration_root,
    )
    encoded = format_baselines(rows)
    source_sha = _sha256(source)
    calibration_lines = "\n".join(
        f"Calibration scientific SHA-256: {rows[name]['calibration_sha256']}"
        for name in sorted(rows)
    )
    scope_note = str(PAPER_CONTRACT["scope_note"])
    return f'''"""Generated calibrated paper PF initializer. Do not edit by hand.

Source: {source.name}
Source SHA-256: {source_sha}
Paper contract SHA-256: {PAPER_CONTRACT_SHA256}
{calibration_lines}
Generator: experiments/generate_initial_pf.py
Paper: Reiher et al., PNAS 114, 7555-7560 (2017), DOI 10.1073/pnas.1619152114

Paper structure: PF2/Strang, lexicographic order, Eq. (19) bound beta(H),
the paper's Trotter/QPE/synthesis error relation, and uniform Clifford+T
rotation tolerance.

Benchmark adaptation: beta is referenced to 0.1 mHa; the full error target is
1 kcal/mol; error allocation follows the supplement's Eq. (34) calculus using
the rigorous-row gamma=4 and delta_0=11. PyGridSynth is not run here: iteration
0 compiles the generated policy and records its realized T count. Reaction
initializers conservatively give each signed species term an equal absolute
share of the reaction-error target.

{scope_note}

This file is measured unchanged as iteration 0. Scientific acceptance is based
exclusively on exact diagonalization of the compiled PF unitary against the
separately cached U_exact reference.
"""

from dataclasses import dataclass


BASELINES = {encoded}


@dataclass(frozen=True)
class ProductFormulaConfig:
    formula_order: int
    n_steps: int
    zero_angle_threshold: float
    uniform_epsilon: float
    term_order_strategy: str
    baseline_source: str


def get_method_config(problem=None) -> ProductFormulaConfig:
    key = next(iter(BASELINES)) if problem is None else str(problem["name"])
    row = BASELINES[key]
    return ProductFormulaConfig(
        formula_order=int(row["formula_order"]),
        n_steps=int(row["n_steps"]),
        zero_angle_threshold=float(row["zero_angle_threshold"]),
        uniform_epsilon=float(row["uniform_epsilon"]),
        term_order_strategy=str(row["term_order_strategy"]),
        baseline_source=str(row["baseline_source"]),
    )


def choose_synthesis_epsilon(term, alpha, step_index, config) -> float:
    _ = (term, alpha, step_index)
    return config.uniform_epsilon
'''


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration", type=Path)
    calibration.add_argument("--calibration-root", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    expected = render(
        args.source.resolve(),
        calibration_path=args.calibration.resolve() if args.calibration else None,
        calibration_root=args.calibration_root.resolve() if args.calibration_root else None,
    )
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"generated baseline is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
