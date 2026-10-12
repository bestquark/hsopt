"""Thin experiment-directory CLIs for the fixed exact HSOpt evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .compiled_unitary import (
    DIFFERENCE_ERROR_CONTRACTS,
    evaluate_difference,
    evaluate_molecule,
    load_problem,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a policy using exact compilation or an explicitly selected proxy."""
    from . import METHODS, evaluate_difference, evaluate_molecule

    parser = argparse.ArgumentParser(description=main.__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--problem", type=Path)
    source.add_argument("--reaction", type=Path)
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="exact")
    parser.add_argument("--exact-reference", type=Path)
    parser.add_argument(
        "--reference-size", type=int, help="BCH reference determinant cap (default 32)"
    )
    parser.add_argument("--error-contract", choices=DIFFERENCE_ERROR_CONTRACTS)
    parser.add_argument("--backend", choices=("numpy", "cupynumeric", "cupy-lanczos"), default="cupy-lanczos")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.method == "exact" and args.exact_reference is None:
        parser.error("--exact-reference is required for exact evaluation")
    if args.reaction is not None and args.error_contract is None:
        parser.error("--error-contract is required for a reaction")
    if args.problem is not None and args.error_contract is not None:
        parser.error("--error-contract applies only to a reaction")
    options = {
        "method": args.method,
        "backend": args.backend,
        "reference_path": args.exact_reference,
        "reference_size": args.reference_size,
    }
    if args.problem is not None:
        result = evaluate_molecule(args.problem, args.policy_file, **options)
    else:
        result = evaluate_difference(
            args.reaction, args.policy_file, error_contract=args.error_contract, **options
        )
    _write_result(result, args.output)
    return 0


def _write_result(result: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
        return
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)


def _required_external_path(
    explicit: Path | None,
    *,
    local_default: Path,
    flag: str,
) -> Path:
    path = explicit.resolve() if explicit is not None else local_default.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing coefficient-bearing input {path}; pass {flag} with the "
            "external campaign input"
        )
    return path


def evaluate_absolute_directory(
    experiment_dir: Path,
    argv: Sequence[str] | None = None,
) -> int:
    """Evaluate the Hamiltonian identified by one abs_energy directory."""

    experiment_dir = experiment_dir.resolve()
    metadata = json.loads((experiment_dir / "instance.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--policy-file", type=Path, default=experiment_dir / "initial_pf.py")
    parser.add_argument("--exact-reference", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupynumeric"), default="cupynumeric")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    problem_path = _required_external_path(
        args.problem,
        local_default=experiment_dir / "hamiltonian.json",
        flag="--problem",
    )
    problem = load_problem(problem_path)
    expected_name = f"HamLib {metadata['instance']}"
    if str(problem["name"]) != expected_name:
        raise ValueError(
            f"{problem_path} is {problem['name']!r}; this evaluator is sealed for {expected_name!r}"
        )
    if int(problem["n_qubits"]) != int(metadata["n_qubits"]):
        raise ValueError(f"{problem_path} has the wrong qubit count for this experiment")

    result = evaluate_molecule(
        problem_path,
        args.policy_file,
        backend=args.backend,
        reference_path=args.exact_reference,
    )
    _write_result(result, args.output)
    return 0


def evaluate_difference_directory(
    experiment_dir: Path,
    argv: Sequence[str] | None = None,
) -> int:
    """Evaluate the signed reaction identified by one diff_energy directory."""

    experiment_dir = experiment_dir.resolve()
    definition = json.loads((experiment_dir / "definition.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction", type=Path)
    parser.add_argument("--policy-file", type=Path, default=experiment_dir / "initial_pf.py")
    parser.add_argument("--exact-reference", type=Path, required=True)
    parser.add_argument(
        "--error-contract",
        choices=DIFFERENCE_ERROR_CONTRACTS,
        required=True,
        help=("individual_errors: sum_s |nu_s dE_s|; error_cancellation: |sum_s nu_s dE_s|"),
    )
    parser.add_argument("--backend", choices=("numpy", "cupynumeric"), default="cupynumeric")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    reaction_path = _required_external_path(
        args.reaction,
        local_default=experiment_dir / "reaction.json",
        flag="--reaction",
    )
    reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
    expected_species = {
        (str(item["species"]), float(item["coefficient"])) for item in definition["terms"]
    }
    actual_species = {
        (str(item["species"]), float(item["coefficient"])) for item in reaction["terms"]
    }
    if actual_species != expected_species:
        raise ValueError(
            f"{reaction_path} does not match the sealed reaction terms for {definition['instance']}"
        )
    if args.error_contract not in definition["contracts"]:
        raise ValueError(f"reaction-error contract {args.error_contract!r} is not sealed")

    result = evaluate_difference(
        reaction_path,
        args.policy_file,
        error_contract=args.error_contract,
        backend=args.backend,
        reference_path=args.exact_reference,
    )
    _write_result(result, args.output)
    return 0
