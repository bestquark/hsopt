"""Materialize the checked-in benchmark definitions and validate family evaluators."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any


ABS_FAMILY = "abs_energy"
DIFF_FAMILY = "diff_energy"
ABS_DESCRIPTOR = "instance.json"
DIFF_DESCRIPTOR = "definition.json"
INITIAL_POLICY = "initial_pf.py"
EVALUATOR = "evaluate.py"
ABS_INSTANCE_COUNT = 91
DIFF_INSTANCE_COUNT = 10
PHASE_ENCODING_DESCRIPTOR = {
    "method": "centered_nonidentity_pauli_one_norm",
    "dimensionless_evolution_time": 1.0,
    "physical_evolution_time": (
        "computed after parsing as inverse of the nonidentity Pauli one-norm"
    ),
    "energy_recovery": "E_hartree = c0 + lambda * principal_phase",
}


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _checksum_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        digest, separator, relative_text = raw.partition("  ")
        relative = PurePosixPath(relative_text)
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"{path}:{line_number}: malformed checksum row")
        if relative_text in records:
            raise ValueError(f"{path}:{line_number}: duplicate path {relative_text}")
        records[relative_text] = digest
    return records


def absolute_definitions(
    manifest_path: Path,
    checksum_path: Path,
) -> dict[str, dict[str, Any]]:
    """Build one deterministic definition for every selected HamLib Hamiltonian."""

    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != ABS_INSTANCE_COUNT:
        raise ValueError(
            f"absolute-energy manifest must contain {ABS_INSTANCE_COUNT} rows, found {len(rows)}"
        )
    checksums = _checksum_manifest(checksum_path)
    definitions: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance = str(row["lane_id"])
        if instance in definitions:
            raise ValueError(f"duplicate absolute-energy instance: {instance}")
        archive_name = str(row["file"]).replace(".hdf5", ".zip")
        matches = [
            relative
            for relative in checksums
            if PurePosixPath(relative).parts[0] == row["collection"]
            and PurePosixPath(relative).name == archive_name
        ]
        if len(matches) != 1:
            raise ValueError(f"{instance}: expected one archive checksum, found {matches}")
        archive = matches[0]
        one_norm = float(row["one_norm"])
        pauli_terms = int(row["terms"])
        definitions[instance] = {
            "schema": 1,
            "family": ABS_FAMILY,
            "instance": instance,
            "molecule": str(row["molecule"]),
            "n_qubits": int(row["nqubits"]),
            "electrons": int(row["electrons"]),
            "pauli_terms_including_identity": pauli_terms,
            "nonidentity_terms": pauli_terms - 1,
            "hamiltonian_one_norm": one_norm,
            "phase_encoding": PHASE_ENCODING_DESCRIPTOR,
            "source": {
                "library": "HamLib",
                "collection": str(row["collection"]),
                "archive": archive,
                "archive_sha256": checksums[archive],
                "member": str(row["file"]),
                "dataset": str(row["dataset"]),
                "mapping": str(row["mapping"]),
                "active_space_key": str(row["active_space_key"]),
            },
            "dense_exact_fit": {
                "fits": str(row["exact_fit"]).lower() == "true",
                "reason": str(row["fit_reason"]),
                "fixed_electron_sector_dimension": int(row["sector_dim"]),
                "estimated_state_workspace_gib": float(row["estimated_state_workspace_gib"]),
                "estimated_sector_workspace_gib": float(row["estimated_sector_workspace_gib"]),
            },
            "initial_pf_generator": "experiments/generate_initial_pf.py",
            "evaluator": "hsopt.compiled_unitary.evaluate_molecule",
        }
    return definitions


def reaction_definitions(
    manifest_path: Path,
    checksum_path: Path,
    metadata_path: Path,
) -> dict[str, dict[str, Any]]:
    """Build one deterministic definition for every selected reaction."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = _checksum_manifest(checksum_path)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    species_metadata = metadata_payload["species"]
    reaction_keys = [str(value) for value in manifest["reaction_keys"]]
    if len(reaction_keys) != DIFF_INSTANCE_COUNT or len(set(reaction_keys)) != len(reaction_keys):
        raise ValueError(
            f"difference-energy manifest must contain {DIFF_INSTANCE_COUNT} unique reactions"
        )
    definitions: dict[str, dict[str, Any]] = {}
    for order, reaction_key in enumerate(reaction_keys, start=1):
        source = manifest["reactions"][reaction_key]
        terms = []
        for term in source["terms"]:
            species = str(term["species"])
            metadata = f"problem_cache/{species}.json"
            coefficients = f"problem_cache/{species}.npz"
            if metadata not in checksums or coefficients not in checksums:
                raise ValueError(f"{reaction_key}: missing cache checksums for {species}")
            if species not in species_metadata:
                raise ValueError(f"{reaction_key}: missing compact metadata for {species}")
            compact = species_metadata[species]
            if compact["metadata_sha256"] != checksums[metadata]:
                raise ValueError(f"{reaction_key}: stale compact metadata for {species}")
            terms.append(
                {
                    "species": species,
                    "problem_name": f"Reaction seed {species}",
                    "coefficient": float(term["coefficient"]),
                    "n_qubits": int(compact["n_qubits"]),
                    "active_electrons": int(compact["active_electrons"]),
                    "active_orbitals": int(compact["active_orbitals"]),
                    "basis": str(compact["basis"]),
                    "cas_label": str(compact["cas_label"]),
                    "nonidentity_terms": int(compact["nonidentity_terms"]),
                    "phase_encoding": PHASE_ENCODING_DESCRIPTOR,
                    "external_hamiltonian_cache": {
                        "metadata": metadata,
                        "metadata_sha256": checksums[metadata],
                        "coefficients": coefficients,
                        "coefficients_sha256": checksums[coefficients],
                    },
                }
            )
        definitions[reaction_key] = {
            "schema": 1,
            "family": DIFF_FAMILY,
            "instance": reaction_key,
            "order": order,
            "name": str(source["label"]),
            "reactant": str(source["reactant"]),
            "product": str(source["product"]),
            "rationale": str(source["rationale"]),
            "terms": terms,
            "contracts": ["equal_share", "reaction_aware"],
            "reaction_error_target_hartree": float(manifest["reaction_target_hartree"]),
            "initial_pf_generator": "experiments/generate_initial_pf.py",
            "evaluator": "hsopt.compiled_unitary.evaluate_difference",
        }
    return definitions


def expected_layout(
    *,
    abs_manifest: Path,
    abs_checksums: Path,
    reaction_manifest: Path,
    reaction_checksums: Path,
    reaction_metadata: Path,
    abs_root: Path,
    diff_root: Path,
) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for instance, payload in absolute_definitions(abs_manifest, abs_checksums).items():
        expected[abs_root / instance / ABS_DESCRIPTOR] = _json_text(payload)
    for instance, payload in reaction_definitions(
        reaction_manifest,
        reaction_checksums,
        reaction_metadata,
    ).items():
        expected[diff_root / instance / DIFF_DESCRIPTOR] = _json_text(payload)
    return expected


def _instance_directories(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name != "inputs" and not path.name.startswith(".")
    }


def _validate_initial_policy(path: Path, *, expected_problems: int) -> None:
    """Validate the compact immutable contract without importing the policy."""

    if not path.is_file():
        raise ValueError(f"missing generated initial policy: {path}")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ProductFormula"
    ]
    if len(classes) != 1:
        raise ValueError(f"{path}: expected one ProductFormula class")
    policy_class = classes[0]
    parameter_nodes = [
        node.value
        for node in policy_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "parameters" for target in node.targets
        )
    ]
    if len(parameter_nodes) != 1:
        raise ValueError(f"{path}: expected one ProductFormula.parameters assignment")
    parameters = ast.literal_eval(parameter_nodes[0])
    if not isinstance(parameters, dict) or len(parameters) != expected_problems:
        raise ValueError(
            f"{path}: expected {expected_problems} problem policies, "
            f"found {len(parameters) if isinstance(parameters, dict) else 'invalid'}"
        )
    for problem, values in parameters.items():
        if (
            not isinstance(problem, str)
            or not isinstance(values, tuple)
            or len(values) != 2
            or not isinstance(values[0], int)
            or values[0] <= 0
            or float(values[1]) <= 0.0
        ):
            raise ValueError(f"{path}: invalid bound-derived policy for {problem!r}")
    methods = {node.name for node in policy_class.body if isinstance(node, ast.FunctionDef)}
    required = {"__init__", "repetitions", "build_step", "synthesis_epsilon"}
    if not required <= methods:
        raise ValueError(f"{path}: ProductFormula is missing {sorted(required - methods)}")


def sync_layout(
    *,
    abs_manifest: Path,
    abs_checksums: Path,
    reaction_manifest: Path,
    reaction_checksums: Path,
    reaction_metadata: Path,
    abs_root: Path,
    diff_root: Path,
    check: bool,
) -> dict[str, int]:
    """Write or verify the complete 91-Hamiltonian/10-reaction layout."""

    expected = expected_layout(
        abs_manifest=abs_manifest,
        abs_checksums=abs_checksums,
        reaction_manifest=reaction_manifest,
        reaction_checksums=reaction_checksums,
        reaction_metadata=reaction_metadata,
        abs_root=abs_root,
        diff_root=diff_root,
    )
    expected_abs = {
        path.parent.name for path in expected if path.parent.parent.resolve() == abs_root.resolve()
    }
    expected_diff = {
        path.parent.name for path in expected if path.parent.parent.resolve() == diff_root.resolve()
    }
    family_evaluators = [abs_root / EVALUATOR, diff_root / EVALUATOR]
    if check:
        mismatches = [
            str(path)
            for path, content in expected.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        missing_evaluators = [str(path) for path in family_evaluators if not path.is_file()]
        stale_instance_evaluators = [
            str(path)
            for root in (abs_root, diff_root)
            for path in sorted(root.glob(f"*/{EVALUATOR}"))
        ]
        actual_abs = _instance_directories(abs_root)
        actual_diff = _instance_directories(diff_root)
        invalid_policies: list[str] = []
        for instance in sorted(expected_abs):
            try:
                _validate_initial_policy(
                    abs_root / instance / INITIAL_POLICY,
                    expected_problems=1,
                )
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                invalid_policies.append(str(exc))
        for instance in sorted(expected_diff):
            definition = json.loads(expected[diff_root / instance / DIFF_DESCRIPTOR])
            try:
                _validate_initial_policy(
                    diff_root / instance / INITIAL_POLICY,
                    expected_problems=len(definition["terms"]),
                )
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                invalid_policies.append(str(exc))
        if (
            mismatches
            or missing_evaluators
            or stale_instance_evaluators
            or actual_abs != expected_abs
            or actual_diff != expected_diff
        ):
            raise ValueError(
                "repository benchmark layout is stale: "
                f"mismatched_files={mismatches}, "
                f"missing_family_evaluators={missing_evaluators}, "
                f"stale_instance_evaluators={stale_instance_evaluators}, "
                f"invalid_initial_policies={invalid_policies}, "
                f"unexpected_abs={sorted(actual_abs - expected_abs)}, "
                f"missing_abs={sorted(expected_abs - actual_abs)}, "
                f"unexpected_diff={sorted(actual_diff - expected_diff)}, "
                f"missing_diff={sorted(expected_diff - actual_diff)}"
            )
        if invalid_policies:
            raise ValueError(
                "repository benchmark policies are stale: "
                f"invalid_initial_policies={invalid_policies}"
            )
    else:
        for path, content in expected.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
    return {
        "absolute_hamiltonians": len(expected_abs),
        "difference_reactions": len(expected_diff),
        "definition_files": sum(
            path.name in {ABS_DESCRIPTOR, DIFF_DESCRIPTOR} for path in expected
        ),
        "evaluator_files": sum(path.is_file() for path in family_evaluators),
        "initial_policies": len(expected_abs) + len(expected_diff),
    }


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--abs-manifest",
        type=Path,
        default=repo_root / "experiments/abs_energy/inputs/instances.csv",
    )
    parser.add_argument(
        "--abs-checksums",
        type=Path,
        default=repo_root / "experiments/abs_energy/inputs/hamlib_zip_sha256.txt",
    )
    parser.add_argument(
        "--reaction-manifest",
        type=Path,
        default=repo_root / "experiments/diff_energy/inputs/reactions.json",
    )
    parser.add_argument(
        "--reaction-checksums",
        type=Path,
        default=repo_root / "experiments/diff_energy/inputs/problem_cache.sha256",
    )
    parser.add_argument(
        "--reaction-metadata",
        type=Path,
        default=repo_root / "experiments/diff_energy/inputs/problem_metadata.json",
    )
    parser.add_argument(
        "--abs-root",
        type=Path,
        default=repo_root / "experiments/abs_energy",
    )
    parser.add_argument(
        "--diff-root",
        type=Path,
        default=repo_root / "experiments/diff_energy",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = sync_layout(
        abs_manifest=args.abs_manifest.resolve(),
        abs_checksums=args.abs_checksums.resolve(),
        reaction_manifest=args.reaction_manifest.resolve(),
        reaction_checksums=args.reaction_checksums.resolve(),
        reaction_metadata=args.reaction_metadata.resolve(),
        abs_root=args.abs_root.resolve(),
        diff_root=args.diff_root.resolve(),
        check=args.check,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
