"""Evaluate one sealed absolute-energy experiment with the fixed exact evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from hsopt.compiled_unitary import evaluate_molecule, load_problem


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="One directory below experiments/abs_energy.",
    )
    parser.add_argument(
        "--problem",
        type=Path,
        required=True,
        help="External coefficient-bearing Hamiltonian JSON.",
    )
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--exact-reference", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupynumeric"), default="cupynumeric")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    experiment_dir = args.experiment_dir.resolve()
    metadata = json.loads((experiment_dir / "instance.json").read_text(encoding="utf-8"))
    problem_path = args.problem.resolve()
    problem = load_problem(problem_path)
    expected_name = f"HamLib {metadata['instance']}"
    if str(problem["name"]) != expected_name:
        raise ValueError(
            f"{problem_path} is {problem['name']!r}; this experiment is sealed "
            f"for {expected_name!r}"
        )
    if int(problem["n_qubits"]) != int(metadata["n_qubits"]):
        raise ValueError(f"{problem_path} has the wrong qubit count for this experiment")

    result = evaluate_molecule(
        problem_path,
        (args.policy_file or experiment_dir / "initial_pf.py").resolve(),
        backend=args.backend,
        reference_path=args.exact_reference.resolve(),
    )
    _write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
