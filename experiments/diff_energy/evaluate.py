"""Evaluate one sealed signed-energy experiment with the fixed exact evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from hsopt.compiled_unitary import evaluate_difference


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
    parser.add_argument("--exact-reference", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupynumeric"), default="cupynumeric")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

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

    result = evaluate_difference(
        reaction_path,
        (args.policy_file or experiment_dir / f"initial_pf_{args.baseline}.py").resolve(),
        backend=args.backend,
        reference_path=args.exact_reference.resolve(),
    )
    _write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
