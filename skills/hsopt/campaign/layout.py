"""Validate the compact source registries and shared-protocol entry points."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from ..data.registry import validate_registries


def check_layout(repo: Path) -> dict:
    result = validate_registries(repo)
    for family in ("abs_protocol", "diff_protocol"):
        for name in ("mutate.py", "evaluate.py", "initial_pf.py"):
            path = repo / "experiments" / family / name
            ast.parse(path.read_text())
    for name in ("generate_initial_pf.py", "initial_nested_pf.py"):
        ast.parse((repo / "experiments" / "baseline" / name).read_text())
    for name in ("molecules.json", "reactions.json"):
        baseline = json.loads((repo / "experiments" / "baseline" / name).read_text())["baseline"]
        for source, key in [
            ("initial_nested_pf.py", "template_sha256"),
            ("generate_initial_pf.py", "compact_pauli_generator_sha256"),
        ]:
            if (
                hashlib.sha256(
                    (repo / "experiments" / "baseline" / source).read_bytes()
                ).hexdigest()
                != baseline[key]
            ):
                raise ValueError(f"baseline source hash mismatch: {source}")
    for family in ("abs_energy", "diff_energy"):
        if (repo / "experiments" / family).exists():
            raise ValueError(f"{family} belongs on the single_molecule branch")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    print(json.dumps(check_layout(args.repo_root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
