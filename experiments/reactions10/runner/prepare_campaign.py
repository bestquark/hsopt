#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pprint
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-runner-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

from campaign_common import (  # noqa: E402
    CHEMICAL_ACCURACY_HARTREE,
    CODEX_CLI_VERSION,
    CODEX_PACKAGE,
    EXPECTED_CACHE_FILES,
    LANE_COUNT,
    PAPER_MODEL,
    PAPER_REASONING_EFFORT,
    PROPOSAL_ITERATIONS,
    REACTION_COUNT,
    SCHEMA_VERSION,
    campaign_paths,
    copy_verified_cache,
    derive_campaign_id,
    file_record,
    manifest_sha256,
    now_iso,
    sha256_file,
    snapshot_files,
    snapshot_tree,
    validate_policy,
    validate_absolute_posix_path,
    verify_problem_cache,
    write_json,
)


RUNNER_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = RUNNER_ROOT.parent
REPO_ROOT = RUNNER_ROOT.parents[2]
INPUTS_ROOT = EXPERIMENT_ROOT / "inputs"
SEED_MANIFEST = INPUTS_ROOT / "seed_manifest.json"
SEED_VALIDATION = INPUTS_ROOT / "seed_validation.json"
CACHE_CHECKSUMS = INPUTS_ROOT / "problem_cache.sha256"
EXPECTED_INPUT_HASHES = {
    "seed_manifest.json": "7cd26b6d25117cbebcec93cd9b800c4ab473925adee895d147ddc3e135f45094",
    "seed_validation.json": "e2bee43bbf79fd5cfa8877c0be166a115443528ca39f2c53ec596b6239555fcc",
    "problem_cache.sha256": "cf521d0a98c7b9cb3478ef28349200f4ab98b171a79a915edca6459349f9c22f",
}
CORE_FILES = (
    Path("examples/pfs/array_backend.py"),
    Path("examples/pfs/chemistry.py"),
    Path("examples/pfs/reactions/evaluate.py"),
    Path("examples/pfs/reactions/simple_pf.py"),
    Path("experiments/reactions10/__init__.py"),
    Path("experiments/reactions10/benchmark.py"),
    Path("experiments/reactions10/inputs/seed_manifest.json"),
    Path("experiments/reactions10/inputs/seed_validation.json"),
    Path("experiments/reactions10/inputs/problem_cache.sha256"),
)


def render_policy(template: str, lane_id: str, policies: dict[str, dict[str, Any]]) -> str:
    rendered = template.replace("__LANE_ID__", lane_id).replace(
        "__SPECIES_POLICIES__", pprint.pformat(policies, sort_dicts=True, width=100)
    )
    if "__LANE_ID__" in rendered or "__SPECIES_POLICIES__" in rendered:
        raise RuntimeError("policy template contains unresolved placeholders")
    return rendered


def git_provenance(repo_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": __import__("hashlib").sha256(status.encode("utf-8")).hexdigest(),
    }


def validated_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing = [
        path for path in (SEED_MANIFEST, SEED_VALIDATION, CACHE_CHECKSUMS) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"fixed reaction inputs are missing: {missing}")
    for path in (SEED_MANIFEST, SEED_VALIDATION, CACHE_CHECKSUMS):
        actual_hash = sha256_file(path)
        expected_hash = EXPECTED_INPUT_HASHES[path.name]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"fixed input hash mismatch for {path.name}: expected {expected_hash}, "
                f"found {actual_hash}"
            )
    seed_manifest = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    validation = json.loads(SEED_VALIDATION.read_text(encoding="utf-8"))
    if not isinstance(validation, list) or len(validation) != REACTION_COUNT:
        raise RuntimeError(f"seed validation must contain exactly {REACTION_COUNT} reactions")
    reaction_keys = seed_manifest.get("reaction_keys")
    reactions = seed_manifest.get("reactions")
    if (
        not isinstance(reaction_keys, list)
        or len(reaction_keys) != REACTION_COUNT
        or len(set(reaction_keys)) != REACTION_COUNT
        or not isinstance(reactions, dict)
        or set(reactions) != set(reaction_keys)
    ):
        raise RuntimeError("seed manifest does not define the fixed ten-reaction set")
    by_reaction = {str(row.get("reaction_key")): row for row in validation}
    if set(by_reaction) != set(reaction_keys):
        raise RuntimeError("seed validation reaction keys do not match seed manifest")
    for key in reaction_keys:
        row = by_reaction[key]
        if not bool(row.get("validation_passes")):
            raise RuntimeError(f"seed validation did not pass for {key}")
        target = float(row["reaction_error_target_hartree"])
        if abs(target - CHEMICAL_ACCURACY_HARTREE) > 1.0e-15:
            raise RuntimeError(f"{key}: seed target differs from exact 1 kcal/mol")
        policies = row.get("policies", {})
        if set(policies) != {"equal_share", "reaction_aware"}:
            raise RuntimeError(f"{key}: expected equal_share and reaction_aware seeds")
        if not all(bool(policies[name].get("full_space_constraint_passes")) for name in policies):
            raise RuntimeError(f"{key}: one or more seed policies fail their fixed contract")
    return seed_manifest, validation


def build_lanes(
    *,
    root: Path,
    seed_manifest: dict[str, Any],
    validation: list[dict[str, Any]],
    template: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_reaction = {str(row["reaction_key"]): row for row in validation}
    lanes: list[dict[str, Any]] = []
    lane_states: dict[str, dict[str, Any]] = {}
    for reaction_key in seed_manifest["reaction_keys"]:
        reaction = seed_manifest["reactions"][reaction_key]
        validated = by_reaction[reaction_key]
        species = [str(item) for item in reaction["species"]]
        for contract in ("equal_share", "reaction_aware"):
            lane_id = f"{reaction_key}__{contract}"
            point = validated["policies"][contract]["full_space_point"]
            policies = {
                species_key: {
                    "formula_order": int(validated["formula_order"]),
                    "n_steps": int(validated["n_steps"]),
                    "zero_angle_threshold": 1.0e-15,
                    "synthesis_error_budget_hartree": float(
                        point["species"][species_key]["synthesis_error_budget_hartree"]
                    ),
                }
                for species_key in species
            }
            lane = {
                "lane_id": lane_id,
                "reaction_key": reaction_key,
                "reaction": str(reaction["label"]),
                "contract": contract,
                "species": species,
                "terms": reaction["terms"],
                "reaction_target_hartree": CHEMICAL_ACCURACY_HARTREE,
                "reaction_target_mha": 1000.0 * CHEMICAL_ACCURACY_HARTREE,
                "seed_total_t_gate_count": int(point["total_t_gate_count"]),
                "seed_validation_passes": bool(
                    validated["policies"][contract]["full_space_constraint_passes"]
                ),
                "seed_result_expectation": {
                    "formula_order": int(validated["formula_order"]),
                    "n_steps": int(validated["n_steps"]),
                    "signed_reaction_error_hartree": float(point["signed_reaction_error_hartree"]),
                    "absolute_reaction_error_hartree": float(
                        point["absolute_reaction_error_hartree"]
                    ),
                    "triangle_bound_hartree": float(point["triangle_bound_hartree"]),
                    "max_species_absolute_error_hartree": float(
                        point["max_species_absolute_error_hartree"]
                    ),
                    "equal_share_error_limit_hartree": float(
                        point["equal_share_error_limit_hartree"]
                    ),
                    "total_t_gate_count": int(point["total_t_gate_count"]),
                    "species": {
                        species_key: {
                            "signed_energy_error_hartree": float(
                                point["species"][species_key]["signed_energy_error_hartree"]
                            ),
                            "t_gate_count": int(point["species"][species_key]["t_gate_count"]),
                        }
                        for species_key in species
                    },
                },
            }
            lane_dir = root / "lanes" / lane_id
            lane_dir.mkdir(parents=True, exist_ok=False)
            initial_policy = lane_dir / "initial_simple_pf.py"
            initial_policy.write_text(render_policy(template, lane_id, policies), encoding="utf-8")
            valid, error = validate_policy(initial_policy, species)
            if not valid:
                raise RuntimeError(f"rendered seed policy is invalid for {lane_id}: {error}")
            policy_hash = sha256_file(initial_policy)
            lane["seed_policy_sha256"] = policy_hash
            current_policy = lane_dir / "simple_pf.py"
            shutil.copy2(initial_policy, current_policy)
            write_json(lane_dir / "metadata.json", lane)

            candidate = root / "iterations" / "iter_0000" / "candidates" / lane_id / "simple_pf.py"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(initial_policy, candidate)
            record_path = root / "iterations" / "iter_0000" / "mutations" / lane_id / "record.json"
            write_json(
                record_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "lane_id": lane_id,
                    "iteration": 0,
                    "kind": "baseline",
                    "status": "candidate_ready",
                    "parent_iteration": None,
                    "parent_policy_sha256": None,
                    "prompt_sha256": None,
                    "candidate_policy_sha256": policy_hash,
                    "seed_policy_sha256": policy_hash,
                    "codex_package": None,
                    "codex_cli_version": None,
                    "model": None,
                    "reasoning_effort": None,
                    "returncode": None,
                    "timed_out": False,
                    "changed": False,
                    "valid": True,
                    "validation_error": None,
                    "created_at": now_iso(),
                    "completed_at": now_iso(),
                },
            )
            lanes.append(lane)
            lane_states[lane_id] = {
                "last_collected_iteration": -1,
                "proposal_count": 0,
                "incumbent_iteration": None,
                "incumbent_policy_path": None,
                "incumbent_policy_sha256": None,
                "incumbent_t_gate_count": None,
                "incumbent_result_path": None,
                "status": "baseline_prepared",
                "last_result_sha256": None,
            }
    if len(lanes) != LANE_COUNT:
        raise RuntimeError(f"preparation produced {len(lanes)} lanes, expected {LANE_COUNT}")
    return lanes, lane_states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed 20-lane, 50-proposal reaction autoresearch campaign."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--problem-cache",
        type=Path,
        required=True,
        help="External directory containing the exact 21 JSON/NPZ Hamiltonian cache pairs.",
    )
    parser.add_argument("--cluster-root", required=True)
    parser.add_argument("--cluster-python", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.batch_size > LANE_COUNT:
        parser.error(f"--batch-size must be in [1, {LANE_COUNT}]")
    try:
        validate_absolute_posix_path(args.cluster_root, "--cluster-root")
        validate_absolute_posix_path(args.cluster_python, "--cluster-python")
    except ValueError as exc:
        parser.error(str(exc))
    root = args.campaign_root.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f"campaign destination must be empty: {root}")
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        parser.error("campaign root must be external to the repository checkout")

    repository = git_provenance(REPO_ROOT)
    if repository["dirty"]:
        parser.error(
            "reaction paper campaigns require a clean Git checkout; commit or stash all changes first"
        )
    if not repository["branch"]:
        parser.error("reaction paper campaigns require a named branch, not a detached HEAD")
    seed_manifest, validation = validated_inputs()
    expected_cache = verify_problem_cache(args.problem_cache, CACHE_CHECKSUMS)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("lanes", "iterations", "logs", "source", "receipts", "transactions", ".locks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    copy_verified_cache(args.problem_cache.resolve(), root / "problem_cache", expected_cache)

    source_hashes = snapshot_files(
        source_root=REPO_ROOT,
        destination_root=root / "source" / "repo",
        relative_paths=CORE_FILES,
    )
    runner_hashes = snapshot_tree(
        source_root=RUNNER_ROOT,
        destination_root=root / "runner",
        excluded_names={".venv", "__pycache__", ".pytest_cache", ".ruff_cache"},
    )
    template = (RUNNER_ROOT / "policy_template.py").read_text(encoding="utf-8")
    lanes, lane_states = build_lanes(
        root=root,
        seed_manifest=seed_manifest,
        validation=validation,
        template=template,
    )
    created_at = now_iso()
    preparation_nonce = secrets.token_hex(32)
    campaign_id = derive_campaign_id(preparation_nonce)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "reactions10_autoresearch",
        "created_at": created_at,
        "preparation_nonce": preparation_nonce,
        "campaign_id": campaign_id,
        "reaction_count": REACTION_COUNT,
        "lane_count": LANE_COUNT,
        "proposal_iterations": PROPOSAL_ITERATIONS,
        "evaluation_iterations": list(range(PROPOSAL_ITERATIONS + 1)),
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "evaluation_mode": "full_hilbert_space_compiled_one_step_unitary_diagonalization",
        "compilation_backend": "pygridsynth",
        "t_count_definition": "sum of compiled T counts for every unique species circuit once",
        "contracts": {
            "equal_share": "all_i |e_i| <= (1 kcal/mol) / sum_j |nu_j|",
            "reaction_aware": "|sum_i nu_i e_i| <= 1 kcal/mol",
        },
        "autoresearch_provenance": {
            "codex_package": CODEX_PACKAGE,
            "codex_cli_version": CODEX_CLI_VERSION,
            "model": PAPER_MODEL,
            "reasoning_effort": PAPER_REASONING_EFFORT,
        },
        "execution": {
            "cluster_root": args.cluster_root,
            "cluster_python": args.cluster_python,
            "batch_size": int(args.batch_size),
        },
        "repository": repository,
        "inputs": {
            "seed_manifest": file_record(SEED_MANIFEST),
            "seed_validation": file_record(SEED_VALIDATION),
            "problem_cache_checksums": file_record(CACHE_CHECKSUMS),
            "problem_cache_file_count": EXPECTED_CACHE_FILES,
            "problem_cache_files": expected_cache,
        },
        "source_files": source_hashes,
        "runner_files": runner_hashes,
        "lanes": lanes,
    }
    paths = campaign_paths(root)
    write_json(paths["manifest"], manifest)
    state = {
        "schema_version": SCHEMA_VERSION,
        "experiment": manifest["experiment"],
        "manifest_sha256": manifest_sha256(paths["manifest"]),
        "campaign_id": campaign_id,
        "created_at": created_at,
        "updated_at": created_at,
        "current_collected_iteration": -1,
        "next_iteration": 0,
        "lanes": lane_states,
        "batches": {},
        "last_validation_errors": {},
    }
    write_json(paths["state"], state)
    print(
        f"prepared lanes={len(lanes)} evaluations_per_lane={PROPOSAL_ITERATIONS + 1} "
        f"cache_files={len(expected_cache)} root={root}"
    )


if __name__ == "__main__":
    main()
