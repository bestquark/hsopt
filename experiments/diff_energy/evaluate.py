"""Evaluate one sealed signed-energy experiment (exact by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from hsopt.evaluation import METHODS, evaluate_difference
from hsopt.evaluation.compiled_unitary import DIFFERENCE_ERROR_CONTRACTS


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
        help="One directory below experiments/diff_energy.",
    )
    parser.add_argument(
        "--reaction",
        type=Path,
        required=True,
        help="External reaction JSON and its coefficient-bearing Hamiltonians.",
    )
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--baseline", choices=("loose", "tight"), default="loose")
    parser.add_argument("--method", choices=METHODS, default="exact")
    parser.add_argument("--exact-reference", type=Path)
    parser.add_argument("--reference-size", type=int, help="BCH reference determinant cap")
    parser.add_argument(
        "--error-contract",
        choices=DIFFERENCE_ERROR_CONTRACTS,
        required=True,
        help=(
            "individual_errors enforces sum_s |nu_s dE_s| < chemical accuracy; "
            "error_cancellation enforces |sum_s nu_s dE_s| < chemical accuracy"
        ),
    )
    parser.add_argument("--backend", choices=("numpy", "cupynumeric", "cupy-lanczos"), default="cupy-lanczos")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.method == "exact" and args.exact_reference is None:
        parser.error("--exact-reference is required for exact evaluation")

    experiment_dir = args.experiment_dir.resolve()
    definition = json.loads((experiment_dir / "definition.json").read_text(encoding="utf-8"))
    reaction_path = args.reaction.resolve()
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
        raise ValueError(
            f"{args.error_contract!r} is not sealed by {experiment_dir / 'definition.json'}"
        )

    result = evaluate_difference(
        reaction_path,
        (args.policy_file or experiment_dir / f"initial_pf_{args.baseline}.py").resolve(),
        method=args.method,
        error_contract=args.error_contract,
        backend=args.backend,
        reference_path=args.exact_reference.resolve() if args.exact_reference else None,
        reference_size=args.reference_size,
    )
    _write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
