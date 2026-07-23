"""Generate the immutable iteration-0 product-formula policy.

The structural method comes from the checked-in Reiher--Wiebe--Svore--Wecker--
Troyer paper contract.  Benchmark-specific numerical choices that the paper
does not fix are recorded separately in that contract; this generator does not
derive or optimize them from the Hamiltonian under evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent
    / "paper_baselines"
    / "reiher_wiebe_svore_2017.json"
)


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    paper_method = contract["paper_method"]
    benchmark = contract["benchmark_instantiation"]
    if int(paper_method["formula_order"]) != 2:
        raise ValueError("the Reiher et al. contract must use symmetric PF2")
    if str(paper_method["term_order_strategy"]) != "lexicographic":
        raise ValueError("the Reiher et al. contract must use lexicographic ordering")
    if int(benchmark["n_steps"]) <= 0:
        raise ValueError("the baseline n_steps must be positive")
    if float(benchmark["total_synthesis_error_budget_hartree"]) <= 0.0:
        raise ValueError("the baseline synthesis budget must be positive")
    return contract


def baseline(
    problem: dict[str, Any],
    *,
    paper_contract: dict[str, Any],
) -> dict[str, Any]:
    paper_method = paper_contract["paper_method"]
    benchmark = paper_contract["benchmark_instantiation"]
    order = int(paper_method["formula_order"])
    n_steps = int(benchmark["n_steps"])
    term_count = len(problem["terms"])
    rotations_per_step = term_count if order == 1 else 2 * term_count - 1
    total_rotations = n_steps * rotations_per_step
    synthesis_budget = float(benchmark["total_synthesis_error_budget_hartree"])
    return {
        "baseline_source": str(paper_contract["baseline_id"]),
        "formula_order": order,
        "n_steps": n_steps,
        "nonidentity_terms": term_count,
        "scheduled_rotations": total_rotations,
        "synthesis_error_budget_hartree": synthesis_budget,
        "term_order_strategy": str(paper_method["term_order_strategy"]),
        "uniform_epsilon": synthesis_budget / total_rotations,
    }


def load_problems(source: Path) -> list[dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "terms" in payload and "n_qubits" in payload:
        return [payload]
    problems = []
    for item in payload["terms"]:
        path = (source.parent / str(item["problem"])).resolve()
        problems.append(json.loads(path.read_text(encoding="utf-8")))
    return problems


def format_baselines(rows: dict[str, dict[str, Any]]) -> str:
    lines = ["{"]
    items = sorted(rows.items())
    for name, row in items:
        lines.append(f"    {json.dumps(name)}: {{")
        for key, value in sorted(row.items()):
            literal = json.dumps(value) if isinstance(value, str) else repr(value)
            lines.append(f"        {json.dumps(key)}: {literal},")
        suffix = "," if len(items) > 1 else ""
        lines.append(f"    }}{suffix}")
    lines.append("}")
    return "\n".join(lines)


def seeded_reaction_baselines(
    source: Path,
    seed_validation: Path,
    contract: str,
    *,
    paper_contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate one fixed reaction budget policy on the paper PF structure."""

    if contract not in {"equal_share", "reaction_aware"}:
        raise ValueError("reaction contract must be equal_share or reaction_aware")
    reaction = json.loads(source.read_text(encoding="utf-8"))
    validation_rows = json.loads(seed_validation.read_text(encoding="utf-8"))
    reaction_key = str(reaction["reaction_key"])
    matches = [row for row in validation_rows if row.get("reaction_key") == reaction_key]
    if len(matches) != 1:
        raise ValueError(f"expected one paper seed for {reaction_key}, found {len(matches)}")
    seed = matches[0]
    point = seed["policies"][contract]["full_space_point"]
    order = int(paper_contract["paper_method"]["formula_order"])
    n_steps = int(paper_contract["benchmark_instantiation"]["n_steps"])
    rows: dict[str, dict[str, Any]] = {}
    for item in reaction["terms"]:
        species = str(item["species"])
        problem_path = (source.parent / str(item["problem"])).resolve()
        problem = json.loads(problem_path.read_text(encoding="utf-8"))
        rotations_per_step = (
            len(problem["terms"]) if order == 1 else 2 * len(problem["terms"]) - 1
        )
        synthesis_budget = float(
            point["species"][species]["synthesis_error_budget_hartree"]
        )
        uniform_epsilon = max(
            1.0e-15,
            synthesis_budget / float(n_steps * rotations_per_step),
        )
        rows[str(problem["name"])] = {
            "baseline_source": (
                f"{paper_contract['baseline_id']}_reaction_{contract}"
            ),
            "budget_source": "experiments/reactions10/inputs/seed_validation.json",
            "contract": contract,
            "formula_order": order,
            "n_steps": n_steps,
            "nonidentity_terms": len(problem["terms"]),
            "reaction_key": reaction_key,
            "species": species,
            "synthesis_error_budget_hartree": synthesis_budget,
            "term_order_strategy": str(
                paper_contract["paper_method"]["term_order_strategy"]
            ),
            "uniform_epsilon": uniform_epsilon,
        }
    return rows


def render(
    source: Path,
    *,
    paper_contract_path: Path = DEFAULT_CONTRACT,
    seed_validation: Path | None = None,
    contract: str | None = None,
) -> str:
    paper_contract = load_contract(paper_contract_path)
    if seed_validation is None and contract is None:
        rows = {
            problem["name"]: baseline(problem, paper_contract=paper_contract)
            for problem in load_problems(source)
        }
        provenance = ""
    elif seed_validation is not None and contract is not None:
        rows = seeded_reaction_baselines(
            source,
            seed_validation,
            contract,
            paper_contract=paper_contract,
        )
        provenance = (
            f"Paper reaction seed: {seed_validation.name}\n"
            f"Paper reaction contract: {contract}\n"
        )
    else:
        raise ValueError("--seed-validation and --contract must be provided together")
    encoded = format_baselines(rows)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    contract_sha = hashlib.sha256(paper_contract_path.read_bytes()).hexdigest()
    return f'''"""Generated standard PF initializer. Do not edit by hand.

Source: {source.name}
Source SHA-256: {source_sha}
Paper contract: {paper_contract_path.name}
Paper contract SHA-256: {contract_sha}
{provenance}\
Generator: examples/generate_initial_pf.py
Paper: Reiher et al., PNAS 114, 7555-7560 (2017), DOI 10.1073/pnas.1619152114

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
        zero_angle_threshold=0.0,
        uniform_epsilon=float(row["uniform_epsilon"]),
        term_order_strategy=str(row["term_order_strategy"]),
        baseline_source=str(row["baseline_source"]),
    )


def choose_synthesis_epsilon(term, alpha, step_index, config) -> float:
    _ = (term, alpha, step_index)
    return config.uniform_epsilon
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--seed-validation", type=Path)
    parser.add_argument("--contract", choices=("equal_share", "reaction_aware"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(
        args.source.resolve(),
        paper_contract_path=args.paper_contract.resolve(),
        seed_validation=args.seed_validation.resolve() if args.seed_validation else None,
        contract=args.contract,
    )
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"generated baseline is stale: {args.output}")
        return 0
    args.output.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
