"""Evaluate one HamLib policy by diagonalizing its compiled PF unitary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsopt.evaluation.compiled_unitary import evaluate_molecule
from hsopt.campaign.queue import enqueue


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-file", type=Path, default=HERE / "initial_pf.py")
    parser.add_argument("--backend", choices=("cupynumeric", "numpy"), default="cupynumeric")
    parser.add_argument("--queue-dir", type=Path)
    args = parser.parse_args()
    if args.queue_dir:
        result = enqueue(
            args.queue_dir,
            kind="absolute",
            source=HERE / "hamiltonian.json",
            policy=args.policy_file,
        )
    else:
        result = evaluate_molecule(
            HERE / "hamiltonian.json",
            args.policy_file,
            backend=args.backend,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
