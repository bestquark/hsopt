from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_ROOT = REPO_ROOT / "experiments" / "reactions10" / "runner"
INPUTS_ROOT = REPO_ROOT / "experiments" / "reactions10" / "inputs"
TARGET_HARTREE = 1.0 / 627.5094740631
EXPECTED_INPUT_HASHES = {
    "seed_manifest.json": "7cd26b6d25117cbebcec93cd9b800c4ab473925adee895d147ddc3e135f45094",
    "seed_validation.json": "e2bee43bbf79fd5cfa8877c0be166a115443528ca39f2c53ec596b6239555fcc",
    "problem_cache.sha256": "cf521d0a98c7b9cb3478ef28349200f4ab98b171a79a915edca6459349f9c22f",
}


def _load_module(stem: str, *, campaign_common: ModuleType | None = None) -> ModuleType:
    path = RUNNER_ROOT / f"{stem}.py"
    module_name = f"_hsopt_reactions10_runner_test_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_common = sys.modules.get("campaign_common")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(RUNNER_ROOT))
        if campaign_common is not None:
            sys.modules["campaign_common"] = campaign_common
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        if campaign_common is not None:
            if previous_common is None:
                sys.modules.pop("campaign_common", None)
            else:
                sys.modules["campaign_common"] = previous_common
    return module


CAMPAIGN_COMMON = _load_module("campaign_common")
PREPARE_CAMPAIGN = _load_module("prepare_campaign", campaign_common=CAMPAIGN_COMMON)
MUTATE_ITERATION = _load_module("mutate_iteration", campaign_common=CAMPAIGN_COMMON)
SUBMIT_ITERATION = _load_module("submit_iteration", campaign_common=CAMPAIGN_COMMON)
COLLECT_ITERATION = _load_module("collect_iteration", campaign_common=CAMPAIGN_COMMON)
EXPORT_RESULTS = _load_module("export_results")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): (
            "directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(root.rglob("*"))
    }


def _write_synthetic_cache(root: Path, checksum_path: Path) -> list[str]:
    root.mkdir()
    names: list[str] = []
    lines: list[str] = []
    for index in range(CAMPAIGN_COMMON.EXPECTED_CACHE_SPECIES):
        stem = f"species_{index:02d}"
        for suffix in (".json", ".npz"):
            name = f"{stem}{suffix}"
            payload = f"synthetic-cache:{name}\n".encode()
            (root / name).write_bytes(payload)
            names.append(name)
            lines.append(f"{hashlib.sha256(payload).hexdigest()}  problem_cache/{name}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return names


def _lane(contract: str = "reaction_aware") -> dict[str, Any]:
    return {
        "lane_id": f"synthetic__{contract}",
        "reaction_key": "synthetic",
        "reaction": "A -> B",
        "contract": contract,
        "species": ["a", "b"],
        "terms": [
            {"coefficient": -1.0, "species": "a"},
            {"coefficient": 1.0, "species": "b"},
        ],
        "reaction_target_hartree": TARGET_HARTREE,
    }


def _result_for_lane(lane: dict[str, Any], *, iteration: int, digest: str) -> dict[str, Any]:
    signed = {"a": 2.0 * TARGET_HARTREE, "b": 2.25 * TARGET_HARTREE}
    exact = {"a": -1.0, "b": -2.0}
    pf = {species: exact[species] + signed[species] for species in signed}
    t_counts = {"a": 11, "b": 13}
    signed_reaction = -signed["a"] + signed["b"]
    triangle_bound = abs(signed["a"]) + abs(signed["b"])
    exact_reaction = -exact["a"] + exact["b"]
    pf_reaction = -pf["a"] + pf["b"]
    equal_limit = TARGET_HARTREE / 2.0
    passes_equal = False
    passes_reaction = True
    passes_contract = passes_equal if lane["contract"] == "equal_share" else passes_reaction
    violation_ratio = (
        max(abs(value) for value in signed.values()) / equal_limit
        if lane["contract"] == "equal_share"
        else abs(signed_reaction) / TARGET_HARTREE
    )
    score = (
        24.0
        if passes_contract
        else CAMPAIGN_COMMON.FAILURE_SCORE_BASE * (1.0 + max(1.0, violation_ratio)) ** 2 + 24.0
    )
    return {
        "status": "ok",
        "lane_id": lane["lane_id"],
        "reaction_key": lane["reaction_key"],
        "reaction": lane["reaction"],
        "reaction_terms": [
            {"coefficient": float(term["coefficient"]), "species": str(term["species"])}
            for term in lane["terms"]
        ],
        "species_order": list(lane["species"]),
        "contract": lane["contract"],
        "iteration": iteration,
        "policy_sha256": digest,
        "evaluation_mode": "full_hilbert_space_compiled_one_step_unitary_diagonalization",
        "ground_phase_matching": "maximum_ground_state_eigenvector_overlap",
        "compilation_backend": "pygridsynth",
        "reaction_target_hartree": TARGET_HARTREE,
        "reaction_target_mha": 1000.0 * TARGET_HARTREE,
        "stoichiometric_abs_coefficient_sum": 2.0,
        "equal_share_error_limit_hartree": equal_limit,
        "equal_share_error_limit_mha": 1000.0 * equal_limit,
        "exact_reaction_energy_hartree": exact_reaction,
        "pf_reaction_energy_hartree": pf_reaction,
        "signed_reaction_error_hartree": signed_reaction,
        "signed_reaction_error_mha": 1000.0 * signed_reaction,
        "absolute_reaction_error_hartree": abs(signed_reaction),
        "absolute_reaction_error_mha": 1000.0 * abs(signed_reaction),
        "triangle_bound_hartree": triangle_bound,
        "triangle_bound_mha": 1000.0 * triangle_bound,
        "triangle_inequality_holds": True,
        "triangle_inequality_slack_hartree": triangle_bound - abs(signed_reaction),
        "max_species_absolute_error_hartree": max(abs(value) for value in signed.values()),
        "max_species_absolute_error_mha": 1000.0 * max(abs(value) for value in signed.values()),
        "passes_equal_share_constraint": passes_equal,
        "passes_reaction_constraint": passes_reaction,
        "passes_contract": passes_contract,
        "total_t_gate_count": sum(t_counts.values()),
        "total_scheduled_rotation_count": 2,
        "t_count_aggregation": "sum_each_unique_species_once",
        "score": score,
        "species": {
            species: {
                "species": species,
                "exact_energy_hartree": exact[species],
                "pf_energy_estimate_hartree": pf[species],
                "signed_energy_error_hartree": value,
                "absolute_energy_error_hartree": abs(value),
                "formula_order": 1,
                "n_steps": 1,
                "zero_angle_threshold": 0.0,
                "method_config": {
                    "species": species,
                    "formula_order": 1,
                    "n_steps": 1,
                    "zero_angle_threshold": 0.0,
                },
                "total_pauli_rotation_terms": 1,
                "rotations_per_step": 1,
                "scheduled_rotation_count": 1,
                "nonzero_rotation_count": 1,
                "skipped_rotation_count": 0,
                "one_step_nonzero_rotation_count": 1,
                "one_step_skipped_rotation_count": 0,
                "one_step_t_gate_count": t_counts[species],
                "t_gate_count": t_counts[species],
                "one_step_skipped_absolute_angle_sum": 0.0,
                "skipped_absolute_angle_sum": 0.0,
                "one_step_skipped_rotation_operator_error_bound": 0.0,
                "skipped_rotation_operator_error_bound": 0.0,
                "stationary_step_policy_verified": True,
                "unitary_construction": (
                    "compiled_one_step_product_formula_applied_and_diagonalized_"
                    "in_full_hilbert_space"
                ),
                "phase_matching": "maximum_ground_state_eigenvector_overlap",
                "t_count_semantics": (
                    "exact_count_of_T_characters_in_selected_PyGridSynth_gate_strings_times_n_steps"
                ),
            }
            for species, value in signed.items()
        },
        "source_provenance": {
            "archived_benchmark_sha256": (CAMPAIGN_COMMON.ARCHIVED_REACTION_BENCHMARK_SHA256),
            "archived_evaluator_sha256": (CAMPAIGN_COMMON.ARCHIVED_REACTION_EVALUATOR_SHA256),
        },
    }


def _seed_expectation(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "formula_order": 1,
        "n_steps": 1,
        "signed_reaction_error_hartree": result["signed_reaction_error_hartree"],
        "absolute_reaction_error_hartree": result["absolute_reaction_error_hartree"],
        "triangle_bound_hartree": result["triangle_bound_hartree"],
        "max_species_absolute_error_hartree": result["max_species_absolute_error_hartree"],
        "equal_share_error_limit_hartree": result["equal_share_error_limit_hartree"],
        "total_t_gate_count": result["total_t_gate_count"],
        "species": {
            species: {
                "signed_energy_error_hartree": row["signed_energy_error_hartree"],
                "t_gate_count": row["t_gate_count"],
            }
            for species, row in result["species"].items()
        },
    }


def _export_mutation_fixture(
    root: Path,
    *,
    with_process: bool,
    recovered: bool,
    launch_status: str = "launched",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    campaign_root = root / "campaign"
    lane = _lane("reaction_aware")
    lane_id = lane["lane_id"]
    lane_spec = EXPORT_RESULTS.LaneSpec(
        lane_id=lane_id,
        reaction_key=lane["reaction_key"],
        reaction=lane["reaction"],
        contract=lane["contract"],
        species=tuple(lane["species"]),
        terms=tuple((str(term["species"]), float(term["coefficient"])) for term in lane["terms"]),
        seed_policy_sha256="0" * 64,
    )
    parent_policy = (
        campaign_root / "iterations" / "iter_0000" / "candidates" / lane_id / "simple_pf.py"
    )
    parent_policy.parent.mkdir(parents=True)
    parent_policy.write_text("# parent policy\n", encoding="utf-8")
    parent_hash = _sha256(parent_policy)
    mutation_root = campaign_root / "iterations" / "iter_0001" / "mutations" / lane_id
    work_dir = mutation_root / "work"
    work_dir.mkdir(parents=True)
    prompt = "synthetic exporter mutation prompt\n"
    prompt_path = mutation_root / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    parent_snapshot = mutation_root / "parent_simple_pf.py"
    parent_snapshot.write_bytes(parent_policy.read_bytes())
    prompt_hash = _sha256(prompt_path)
    intent_path = mutation_root / "intent.json"
    CAMPAIGN_COMMON.write_json(
        intent_path,
        {
            "schema_version": 1,
            "lane_id": lane_id,
            "iteration": 1,
            "parent_iteration": 0,
            "parent_policy_path": parent_policy.relative_to(campaign_root).as_posix(),
            "parent_policy_sha256": parent_hash,
            "prompt_sha256": prompt_hash,
            "codex_package": CAMPAIGN_COMMON.CODEX_PACKAGE,
            "codex_cli_version": CAMPAIGN_COMMON.CODEX_CLI_VERSION,
            "model": CAMPAIGN_COMMON.PAPER_MODEL,
            "reasoning_effort": CAMPAIGN_COMMON.PAPER_REASONING_EFFORT,
            "created_at": "2026-07-15T00:00:00+0000",
        },
    )
    last_message = mutation_root / "last_message.txt"
    sanitized_command = [
        "npx",
        "-y",
        CAMPAIGN_COMMON.CODEX_PACKAGE,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-s",
        "workspace-write",
        "-C",
        str(work_dir),
        "-m",
        CAMPAIGN_COMMON.PAPER_MODEL,
        "-c",
        f'model_reasoning_effort="{CAMPAIGN_COMMON.PAPER_REASONING_EFFORT}"',
        "-o",
        str(last_message),
        "<prompt recorded in prompt.md>",
    ]
    launch_command_hash = hashlib.sha256(
        "\0".join([*sanitized_command[:-1], prompt]).encode("utf-8")
    ).hexdigest()
    launch_intent_path = mutation_root / "launch_intent.json"
    CAMPAIGN_COMMON.write_json(
        launch_intent_path,
        {
            "schema_version": 1,
            "lane_id": lane_id,
            "iteration": 1,
            "command_sha256": launch_command_hash,
            "requested_at": "2026-07-15T00:00:01+0000",
        },
    )
    launch_receipt_path = mutation_root / "launch.json"
    CAMPAIGN_COMMON.write_json(
        launch_receipt_path,
        {
            "schema_version": 1,
            "pid": 4242,
            "pgid": 4242,
            "work_dir": str(work_dir),
            "command_sha256": launch_command_hash,
            "status": launch_status,
            "started_at_epoch": 1.0,
        },
    )
    process_path = mutation_root / "process.json"
    returncode: int | None = 7 if with_process else None
    if with_process:
        CAMPAIGN_COMMON.write_json(
            process_path,
            {
                "command": sanitized_command,
                "command_sha256": launch_command_hash,
                "returncode": returncode,
                "timed_out": False,
                "process_error": "RuntimeError: synthetic fallback",
                "pid": 4242,
                "launch_receipt_sha256": _sha256(launch_receipt_path),
                "started_at": "2026-07-15T00:00:01+0000",
                "completed_at": "2026-07-15T00:00:02+0000",
            },
        )
    record = {
        "schema_version": 1,
        "lane_id": lane_id,
        "iteration": 1,
        "kind": "autoresearch_proposal",
        "status": "candidate_ready",
        "mutation_outcome": "fallback_parent",
        "parent_iteration": 0,
        "parent_policy_sha256": parent_hash,
        "candidate_policy_sha256": parent_hash,
        "prompt_sha256": prompt_hash,
        "intent_sha256": _sha256(intent_path),
        "launch_command_sha256": launch_command_hash,
        "launch_intent_sha256": _sha256(launch_intent_path),
        "launch_receipt_sha256": _sha256(launch_receipt_path),
        "process_sha256": _sha256(process_path) if with_process else None,
        "codex_package": CAMPAIGN_COMMON.CODEX_PACKAGE,
        "codex_cli_version": CAMPAIGN_COMMON.CODEX_CLI_VERSION,
        "codex_version_output": "codex-cli 0.144.0",
        "model": CAMPAIGN_COMMON.PAPER_MODEL,
        "reasoning_effort": CAMPAIGN_COMMON.PAPER_REASONING_EFFORT,
        "sandbox": "workspace-write",
        "ephemeral": True,
        "ignored_user_config": True,
        "ignored_rules": True,
        "returncode": returncode,
        "timed_out": False,
        "changed": False,
        "valid": True,
        "unexpected_work_files": [],
        "recovered_after_interruption": recovered,
    }
    kwargs = {
        "lane": lane_spec,
        "iteration": 1,
        "expected_parent_iteration": 0,
        "expected_parent_sha256": parent_hash,
        "expected_parent_policy_path": parent_policy,
        "candidate_sha256": parent_hash,
        "mutation_root": mutation_root,
        "campaign_root": campaign_root,
    }
    paths = {
        "intent": intent_path,
        "launch_intent": launch_intent_path,
        "launch_receipt": launch_receipt_path,
        "process": process_path,
    }
    return record, kwargs, paths


def test_fixed_inputs_and_autoresearch_provenance_are_exactly_pinned() -> None:
    assert PREPARE_CAMPAIGN.EXPECTED_INPUT_HASHES == EXPECTED_INPUT_HASHES
    assert CAMPAIGN_COMMON.CODEX_PACKAGE == "@openai/codex@0.144.0"
    assert CAMPAIGN_COMMON.CODEX_CLI_VERSION == "0.144.0"
    assert CAMPAIGN_COMMON.CODEX_COMMAND == ("npx", "-y", "@openai/codex@0.144.0")
    assert CAMPAIGN_COMMON.PAPER_MODEL == "gpt-5.5"
    assert CAMPAIGN_COMMON.PAPER_REASONING_EFFORT == "high"
    assert EXPORT_RESULTS.PAPER_CODEX_PACKAGE == CAMPAIGN_COMMON.CODEX_PACKAGE
    assert EXPORT_RESULTS.PAPER_CODEX_CLI_VERSION == CAMPAIGN_COMMON.CODEX_CLI_VERSION
    assert EXPORT_RESULTS.PAPER_MODEL == CAMPAIGN_COMMON.PAPER_MODEL
    assert EXPORT_RESULTS.PAPER_REASONING_EFFORT == CAMPAIGN_COMMON.PAPER_REASONING_EFFORT

    for filename, expected in EXPECTED_INPUT_HASHES.items():
        assert _sha256(INPUTS_ROOT / filename) == expected
    seed_manifest, validation = PREPARE_CAMPAIGN.validated_inputs()
    assert len(seed_manifest["reaction_keys"]) == CAMPAIGN_COMMON.REACTION_COUNT
    assert len(validation) == CAMPAIGN_COMMON.REACTION_COUNT

    provenance = {
        "autoresearch_provenance": {
            "codex_package": CAMPAIGN_COMMON.CODEX_PACKAGE,
            "codex_cli_version": CAMPAIGN_COMMON.CODEX_CLI_VERSION,
            "model": CAMPAIGN_COMMON.PAPER_MODEL,
            "reasoning_effort": CAMPAIGN_COMMON.PAPER_REASONING_EFFORT,
        },
        "repository": {
            "branch": "codex/test",
            "commit": "a" * 40,
            "dirty": False,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
        },
    }
    assert EXPORT_RESULTS.validate_provenance(provenance) == {
        "repository_branch": "codex/test",
        "repository_commit": "a" * 40,
        "codex_cli_version": "0.144.0",
        "codex_package": "@openai/codex@0.144.0",
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "declared_runner_input_source_hashes": {},
    }
    provenance["autoresearch_provenance"]["model"] = "drifted-model"
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="provenance model changed"):
        EXPORT_RESULTS.validate_provenance(provenance)


def test_problem_cache_validation_is_exact_and_fail_closed(tmp_path: Path) -> None:
    cache_root = tmp_path / "problem_cache"
    checksums = tmp_path / "problem_cache.sha256"
    names = _write_synthetic_cache(cache_root, checksums)

    verified = CAMPAIGN_COMMON.verify_problem_cache(cache_root, checksums)
    assert set(verified) == set(names)
    assert len(verified) == CAMPAIGN_COMMON.EXPECTED_CACHE_FILES == 42

    original = (cache_root / names[0]).read_bytes()
    (cache_root / names[0]).write_bytes(original + b"drift")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        CAMPAIGN_COMMON.verify_problem_cache(cache_root, checksums)

    (cache_root / names[0]).write_bytes(original)
    (cache_root / "unexpected.bin").write_bytes(b"not declared")
    with pytest.raises(RuntimeError, match="file set mismatch"):
        CAMPAIGN_COMMON.verify_problem_cache(cache_root, checksums)

    unsafe = tmp_path / "unsafe.sha256"
    unsafe.write_text(f"{'0' * 64}  ../escape.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe cache path"):
        CAMPAIGN_COMMON.read_checksum_manifest(unsafe)


def test_preparer_shape_is_accepted_by_runner_and_strict_exporter(tmp_path: Path) -> None:
    seed_manifest, validation = PREPARE_CAMPAIGN.validated_inputs()
    template = (RUNNER_ROOT / "policy_template.py").read_text(encoding="utf-8")
    campaign_root = tmp_path / "campaign"
    lanes, lane_states = PREPARE_CAMPAIGN.build_lanes(
        root=campaign_root,
        seed_manifest=seed_manifest,
        validation=validation,
        template=template,
    )

    assert len(lanes) == CAMPAIGN_COMMON.LANE_COUNT == 20
    assert len(lane_states) == CAMPAIGN_COMMON.LANE_COUNT
    assert {lane["contract"] for lane in lanes} == {"equal_share", "reaction_aware"}
    assert all(len(lane["seed_policy_sha256"]) == 64 for lane in lanes)
    assert len(list((campaign_root / "lanes").glob("*/initial_simple_pf.py"))) == 20
    assert (
        len(list((campaign_root / "iterations" / "iter_0000").glob("candidates/*/simple_pf.py")))
        == 20
    )

    source_snapshot = campaign_root / "source" / "repo" / "sentinel.txt"
    runner_snapshot = campaign_root / "runner" / "sentinel.txt"
    source_snapshot.parent.mkdir(parents=True)
    runner_snapshot.parent.mkdir(parents=True)
    source_snapshot.write_text("fixed source snapshot\n", encoding="utf-8")
    runner_snapshot.write_text("fixed runner snapshot\n", encoding="utf-8")
    cache_root = campaign_root / "problem_cache"
    cache_names = _write_synthetic_cache(
        cache_root,
        tmp_path / "synthetic-cache.sha256",
    )
    cache_hashes = {name: _sha256(cache_root / name) for name in cache_names}

    manifest = {
        "schema_version": CAMPAIGN_COMMON.SCHEMA_VERSION,
        "experiment": "reactions10_autoresearch",
        "reaction_count": CAMPAIGN_COMMON.REACTION_COUNT,
        "lane_count": CAMPAIGN_COMMON.LANE_COUNT,
        "proposal_iterations": CAMPAIGN_COMMON.PROPOSAL_ITERATIONS,
        "evaluation_iterations": list(range(CAMPAIGN_COMMON.PROPOSAL_ITERATIONS + 1)),
        "chemical_accuracy_hartree": TARGET_HARTREE,
        "autoresearch_provenance": {
            "codex_package": CAMPAIGN_COMMON.CODEX_PACKAGE,
            "codex_cli_version": CAMPAIGN_COMMON.CODEX_CLI_VERSION,
            "model": CAMPAIGN_COMMON.PAPER_MODEL,
            "reasoning_effort": CAMPAIGN_COMMON.PAPER_REASONING_EFFORT,
        },
        "inputs": {"problem_cache_files": cache_hashes},
        "source_files": {
            "sentinel.txt": {
                "sha256": _sha256(source_snapshot),
                "size_bytes": source_snapshot.stat().st_size,
            }
        },
        "runner_files": {
            "sentinel.txt": {
                "sha256": _sha256(runner_snapshot),
                "size_bytes": runner_snapshot.stat().st_size,
            }
        },
        "lanes": lanes,
    }
    exporter_lanes = EXPORT_RESULTS.validate_manifest(manifest)
    assert set(exporter_lanes) == {lane["lane_id"] for lane in lanes}

    CAMPAIGN_COMMON.write_json(campaign_root / "manifest.json", manifest)
    state = {
        "schema_version": CAMPAIGN_COMMON.SCHEMA_VERSION,
        "manifest_sha256": CAMPAIGN_COMMON.manifest_sha256(campaign_root / "manifest.json"),
        "lanes": lane_states,
    }
    CAMPAIGN_COMMON.write_json(campaign_root / "state.json", state)
    loaded_manifest, loaded_state = CAMPAIGN_COMMON.require_campaign(campaign_root)
    assert loaded_manifest == manifest
    assert loaded_state == state

    transaction_path = campaign_root / "transactions" / "collect_iter_0000.json"
    CAMPAIGN_COMMON.write_json(
        transaction_path,
        {"schema_version": 1, "phase": "applying", "iteration": 0},
    )
    with pytest.raises(RuntimeError, match="must be replayed first"):
        CAMPAIGN_COMMON.require_campaign(campaign_root)
    CAMPAIGN_COMMON.require_campaign(campaign_root, allow_open_collect_transaction=True)
    CAMPAIGN_COMMON.write_json(
        transaction_path,
        {"schema_version": 1, "phase": "committed", "iteration": 0},
    )
    CAMPAIGN_COMMON.require_campaign(campaign_root)


def test_prepare_cli_refuses_to_put_campaign_inside_repository() -> None:
    forbidden = REPO_ROOT / ".reaction-campaign-must-stay-external"
    assert not forbidden.exists()
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_ROOT / "prepare_campaign.py"),
            "--campaign-root",
            str(forbidden),
            "--problem-cache",
            str(forbidden / "cache"),
            "--cluster-root",
            "/cluster/hsopt-test",
            "--cluster-python",
            "/cluster/python",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "external to the repository" in completed.stderr
    assert not forbidden.exists()


def test_exact_contract_semantics_distinguish_cancellation_from_equal_share() -> None:
    digest = "a" * 64
    reaction_lane = _lane("reaction_aware")
    reaction_result = _result_for_lane(reaction_lane, iteration=4, digest=digest)
    CAMPAIGN_COMMON.validate_result(
        reaction_result,
        lane=reaction_lane,
        iteration=4,
        candidate_hash=digest,
    )
    assert reaction_result["passes_reaction_constraint"] is True
    assert reaction_result["passes_equal_share_constraint"] is False
    assert reaction_result["passes_contract"] is True
    assert reaction_result["score"] == 24.0

    equal_lane = _lane("equal_share")
    equal_result = _result_for_lane(equal_lane, iteration=4, digest=digest)
    CAMPAIGN_COMMON.validate_result(
        equal_result,
        lane=equal_lane,
        iteration=4,
        candidate_hash=digest,
    )
    assert equal_result["passes_reaction_constraint"] is True
    assert equal_result["passes_equal_share_constraint"] is False
    assert equal_result["passes_contract"] is False
    assert equal_result["score"] > CAMPAIGN_COMMON.FAILURE_SCORE_BASE

    reaction_result["passes_contract"] = False
    with pytest.raises(ValueError, match="inconsistent passes_contract"):
        CAMPAIGN_COMMON.validate_result(
            reaction_result,
            lane=reaction_lane,
            iteration=4,
            candidate_hash=digest,
        )


def test_exporter_validates_one_small_diagnostic_result_without_weakening_paper_gate(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "simple_pf.py"
    policy.write_text("# diagnostic policy\n", encoding="utf-8")
    digest = _sha256(policy)
    lane = _lane("reaction_aware")
    lane_spec = EXPORT_RESULTS.LaneSpec(
        lane_id=lane["lane_id"],
        reaction_key=lane["reaction_key"],
        reaction=lane["reaction"],
        contract=lane["contract"],
        species=tuple(lane["species"]),
        terms=tuple((str(term["species"]), float(term["coefficient"])) for term in lane["terms"]),
        seed_policy_sha256=digest,
    )
    candidate = EXPORT_RESULTS.validate_result(
        _result_for_lane(lane, iteration=3, digest=digest),
        lane=lane_spec,
        iteration=3,
        policy_path=policy,
    )
    assert candidate.passes_contract is True
    assert candidate.passes_equal_share_constraint is False
    assert candidate.total_t_gate_count == 24
    assert EXPORT_RESULTS.PAPER_CANDIDATE_COUNT == 20 * 51
    assert EXPORT_RESULTS.PAPER_MUTATION_COUNT == 20 * 50

    tampered = _result_for_lane(lane, iteration=3, digest=digest)
    tampered["species"]["a"]["scheduled_rotation_count"] = 2
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="rotation-count|scheduled"):
        EXPORT_RESULTS.validate_result(
            tampered,
            lane=lane_spec,
            iteration=3,
            policy_path=policy,
        )


def test_exporter_rejects_baseline_drift_from_preserved_seed_expectation(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "simple_pf.py"
    policy.write_text("# baseline policy\n", encoding="utf-8")
    digest = _sha256(policy)
    lane = _lane("reaction_aware")
    result = _result_for_lane(lane, iteration=0, digest=digest)
    expectation = _seed_expectation(result)

    def lane_spec(seed_expectation: dict[str, Any]) -> Any:
        return EXPORT_RESULTS.LaneSpec(
            lane_id=lane["lane_id"],
            reaction_key=lane["reaction_key"],
            reaction=lane["reaction"],
            contract=lane["contract"],
            species=tuple(lane["species"]),
            terms=tuple(
                (str(term["species"]), float(term["coefficient"])) for term in lane["terms"]
            ),
            seed_policy_sha256=digest,
            seed_result_expectation=seed_expectation,
        )

    candidate = EXPORT_RESULTS.validate_result(
        result,
        lane=lane_spec(expectation),
        iteration=0,
        policy_path=policy,
    )
    EXPORT_RESULTS.validate_baseline_result(candidate)

    drifted = copy.deepcopy(expectation)
    drifted["species"]["a"]["t_gate_count"] += 1
    drifted_candidate = EXPORT_RESULTS.validate_result(
        result,
        lane=lane_spec(drifted),
        iteration=0,
        policy_path=policy,
    )
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="species T count drifted"):
        EXPORT_RESULTS.validate_baseline_result(drifted_candidate)


def test_exporter_validates_mutation_evidence_status_hashes_and_process_null_rule(
    tmp_path: Path,
) -> None:
    record, kwargs, paths = _export_mutation_fixture(
        tmp_path / "complete",
        with_process=True,
        recovered=False,
    )
    EXPORT_RESULTS.validate_mutation_record(record, **kwargs)

    paths["process"].write_text('{"returncode": 999}\n', encoding="utf-8")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="process receipt file hash"):
        EXPORT_RESULTS.validate_mutation_record(record, **kwargs)

    bad_status_record, bad_status_kwargs, _paths = _export_mutation_fixture(
        tmp_path / "bad-status",
        with_process=True,
        recovered=False,
        launch_status="exec_error",
    )
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="actual Codex launch"):
        EXPORT_RESULTS.validate_mutation_record(bad_status_record, **bad_status_kwargs)

    recovered_record, recovered_kwargs, _paths = _export_mutation_fixture(
        tmp_path / "recovered",
        with_process=False,
        recovered=True,
    )
    EXPORT_RESULTS.validate_mutation_record(recovered_record, **recovered_kwargs)

    missing_process_record, missing_process_kwargs, _paths = _export_mutation_fixture(
        tmp_path / "missing-process",
        with_process=False,
        recovered=False,
    )
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="may be absent only"):
        EXPORT_RESULTS.validate_mutation_record(
            missing_process_record,
            **missing_process_kwargs,
        )


def test_completed_mutation_record_is_idempotent_and_immutable(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    lane = _lane("reaction_aware")
    lane_id = lane["lane_id"]
    parent_hash = "b" * 64
    candidate = CAMPAIGN_COMMON.candidate_path(root, 1, lane_id)
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# immutable candidate\n", encoding="utf-8")
    directory = CAMPAIGN_COMMON.mutation_dir(root, 1, lane_id)
    directory.mkdir(parents=True)
    prompt = directory / "prompt.md"
    prompt.write_text("synthetic mutation prompt\n", encoding="utf-8")
    prompt_hash = CAMPAIGN_COMMON.sha256_text(prompt.read_text(encoding="utf-8"))
    intent = directory / "intent.json"
    CAMPAIGN_COMMON.write_json(
        intent,
        {
            "schema_version": 1,
            "lane_id": lane_id,
            "iteration": 1,
            "parent_policy_sha256": parent_hash,
            "prompt_sha256": prompt_hash,
        },
    )
    launch_command_hash = "c" * 64
    launch_intent = directory / "launch_intent.json"
    CAMPAIGN_COMMON.write_json(
        launch_intent,
        {"schema_version": 1, "command_sha256": launch_command_hash},
    )
    launch_receipt = directory / "launch.json"
    CAMPAIGN_COMMON.write_json(
        launch_receipt,
        {
            "schema_version": 1,
            "command_sha256": launch_command_hash,
            "status": "launched",
            "work_dir": str((directory / "work").resolve()),
        },
    )
    process = directory / "process.json"
    CAMPAIGN_COMMON.write_json(
        process,
        {
            "returncode": 0,
            "command_sha256": launch_command_hash,
            "launch_receipt_sha256": _sha256(launch_receipt),
        },
    )
    CAMPAIGN_COMMON.write_json(
        directory / "record.json",
        {
            "schema_version": CAMPAIGN_COMMON.SCHEMA_VERSION,
            "lane_id": lane_id,
            "iteration": 1,
            "status": "candidate_ready",
            "parent_policy_sha256": parent_hash,
            "prompt_sha256": prompt_hash,
            "intent_sha256": _sha256(intent),
            "launch_command_sha256": launch_command_hash,
            "launch_intent_sha256": _sha256(launch_intent),
            "launch_receipt_sha256": _sha256(launch_receipt),
            "process_sha256": _sha256(process),
            "candidate_policy_sha256": CAMPAIGN_COMMON.sha256_file(candidate),
        },
    )

    before = _tree_snapshot(root)
    assert MUTATE_ITERATION.mutation_record_is_complete(
        root=root,
        lane=lane,
        iteration=1,
        parent_hash=parent_hash,
    )
    assert MUTATE_ITERATION.mutation_record_is_complete(
        root=root,
        lane=lane,
        iteration=1,
        parent_hash=parent_hash,
    )
    assert _tree_snapshot(root) == before

    process.write_text('{"returncode": 7}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="record drift"):
        MUTATE_ITERATION.mutation_record_is_complete(
            root=root,
            lane=lane,
            iteration=1,
            parent_hash=parent_hash,
        )
    CAMPAIGN_COMMON.write_json(
        process,
        {
            "returncode": 0,
            "command_sha256": launch_command_hash,
            "launch_receipt_sha256": _sha256(launch_receipt),
        },
    )

    candidate.write_text("# drifted candidate\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="record drift"):
        MUTATE_ITERATION.mutation_record_is_complete(
            root=root,
            lane=lane,
            iteration=1,
            parent_hash=parent_hash,
        )


def test_interrupted_prelaunch_intent_is_relaunched_without_counting_a_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "campaign"
    lane = _lane("reaction_aware")
    lane["reaction_target_mha"] = 1000.0 * TARGET_HARTREE
    lane_id = lane["lane_id"]
    incumbent = root / "lanes" / lane_id / "simple_pf.py"
    incumbent.parent.mkdir(parents=True)
    incumbent.write_text("# incumbent\n", encoding="utf-8")
    incumbent_hash = _sha256(incumbent)
    lane_state = {
        "incumbent_policy_path": incumbent.relative_to(root).as_posix(),
        "incumbent_policy_sha256": incumbent_hash,
        "incumbent_iteration": 0,
        "incumbent_result_path": None,
    }
    prompt = MUTATE_ITERATION.mutation_prompt(lane, 1, None, None)
    directory = CAMPAIGN_COMMON.mutation_dir(root, 1, lane_id)
    CAMPAIGN_COMMON.write_json(
        directory / "intent.json",
        {
            "schema_version": CAMPAIGN_COMMON.SCHEMA_VERSION,
            "lane_id": lane_id,
            "iteration": 1,
            "parent_iteration": 0,
            "parent_policy_path": incumbent.relative_to(root).as_posix(),
            "parent_policy_sha256": incumbent_hash,
            "prompt_sha256": CAMPAIGN_COMMON.sha256_text(prompt),
            "codex_package": CAMPAIGN_COMMON.CODEX_PACKAGE,
            "codex_cli_version": CAMPAIGN_COMMON.CODEX_CLI_VERSION,
            "model": CAMPAIGN_COMMON.PAPER_MODEL,
            "reasoning_effort": CAMPAIGN_COMMON.PAPER_REASONING_EFFORT,
            "created_at": "2026-07-15T00:00:00+0000",
        },
    )
    MUTATE_ITERATION.ensure_mutation_workspace(
        root=root,
        directory=directory,
        intent=CAMPAIGN_COMMON.read_json(directory / "intent.json"),
        prompt_text=prompt,
    )
    assert not (directory / "launch_intent.json").exists()
    launch_attempted = False

    async def interrupt_relaunch(*_args: Any, **_kwargs: Any) -> None:
        nonlocal launch_attempted
        launch_attempted = True
        raise KeyboardInterrupt("injected second crash during safe relaunch")

    monkeypatch.setattr(
        MUTATE_ITERATION.asyncio,
        "create_subprocess_exec",
        interrupt_relaunch,
    )
    with pytest.raises(KeyboardInterrupt, match="injected second crash"):
        asyncio.run(
            MUTATE_ITERATION.mutate_one(
                root=root,
                lane=lane,
                lane_state=lane_state,
                iteration=1,
                codex_version_output="codex-cli 0.144.0",
                timeout_seconds=10,
                semaphore=asyncio.Semaphore(1),
            )
        )
    assert launch_attempted is True
    assert not (directory / "record.json").exists()
    assert not CAMPAIGN_COMMON.candidate_path(root, 1, lane_id).exists()


def test_pid_reuse_guard_rejects_generic_codex_process_without_exact_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_dir = tmp_path / "expected-work"
    work_dir.mkdir()
    receipt = {
        "pid": 4242,
        "pgid": 4242,
        "work_dir": str(work_dir.resolve()),
    }
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(MUTATE_ITERATION, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        MUTATE_ITERATION,
        "_process_group_commands",
        lambda _pgid: ["npx -y @openai/codex@0.144.0 exec --ephemeral"],
    )
    monkeypatch.setattr(
        MUTATE_ITERATION.os,
        "killpg",
        lambda pgid, signum: signals.append((pgid, signum)),
    )
    with pytest.raises(RuntimeError, match="possible PID-reuse"):
        MUTATE_ITERATION.terminate_recorded_process_group(receipt, work_dir)
    assert signals == []


def test_mutation_lane_failure_does_not_cancel_other_lanes_or_change_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane_ids = ["lane_fail", *(f"lane_ok_{index:02d}" for index in range(19))]
    manifest = {
        "lanes": [
            {
                "lane_id": lane_id,
                "reaction_key": lane_id,
                "contract": "reaction_aware",
            }
            for lane_id in lane_ids
        ]
    }
    state = {
        "next_iteration": 1,
        "current_collected_iteration": 0,
        "lanes": {
            lane_id: {
                "incumbent_policy_sha256": f"incumbent-{lane_id}",
                "incumbent_iteration": 0,
            }
            for lane_id in lane_ids
        },
    }
    original_state = copy.deepcopy(state)
    completed: list[str] = []
    cancelled: list[str] = []

    async def fake_mutate_one(**kwargs: Any) -> dict[str, Any]:
        lane_id = str(kwargs["lane"]["lane_id"])
        if lane_id == "lane_fail":
            await asyncio.sleep(0)
            raise RuntimeError("injected one-lane failure")
        try:
            await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            cancelled.append(lane_id)
            raise
        completed.append(lane_id)
        return {"lane_id": lane_id, "status": "mutated"}

    monkeypatch.setattr(MUTATE_ITERATION, "ensure_not_slurm", lambda: None)
    monkeypatch.setattr(
        MUTATE_ITERATION,
        "require_campaign",
        lambda _root: (manifest, state),
    )
    monkeypatch.setattr(MUTATE_ITERATION, "verify_codex_version", lambda: "codex-cli 0.144.0")
    monkeypatch.setattr(MUTATE_ITERATION, "mutate_one", fake_mutate_one)
    args = SimpleNamespace(
        campaign_root=tmp_path / "campaign",
        iteration=1,
        max_concurrent=3,
        timeout_seconds=10,
    )

    returncode = asyncio.run(MUTATE_ITERATION.run(args))
    assert returncode != 0
    assert set(completed) == set(lane_ids[1:])
    assert cancelled == []
    assert state == original_state


def test_stateful_orchestrators_share_one_campaign_wide_lock(tmp_path: Path) -> None:
    lock_names: dict[str, list[str]] = {}
    for stem in ("mutate_iteration", "submit_iteration", "collect_iteration"):
        source = (RUNNER_ROOT / f"{stem}.py").read_text(encoding="utf-8")
        lock_names[stem] = re.findall(r'campaign_lock\(root,\s*"([^"]+)"\)', source)
    assert lock_names == {
        "mutate_iteration": ["orchestration"],
        "submit_iteration": ["orchestration"],
        "collect_iteration": ["orchestration"],
    }

    with CAMPAIGN_COMMON.campaign_lock(tmp_path, "orchestration"):
        with pytest.raises(RuntimeError, match="already held"):
            with CAMPAIGN_COMMON.campaign_lock(tmp_path, "orchestration"):
                pytest.fail("the shared state lock must exclude a second orchestrator")


def test_submit_partition_is_complete_retryable_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    lane_ids = [f"lane_{index:02d}" for index in range(CAMPAIGN_COMMON.LANE_COUNT)]
    manifest = {
        "execution": {"cluster_root": "/cluster/reactions10", "batch_size": 6},
        "lanes": [{"lane_id": lane_id} for lane_id in lane_ids],
    }
    state: dict[str, Any] = {"batches": {}}
    for lane_id in lane_ids:
        candidate = CAMPAIGN_COMMON.candidate_path(root, 0, lane_id)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(f"# {lane_id}\n", encoding="utf-8")
        CAMPAIGN_COMMON.write_json(
            CAMPAIGN_COMMON.mutation_dir(root, 0, lane_id) / "record.json",
            {
                "lane_id": lane_id,
                "iteration": 0,
                "status": "candidate_ready",
                "candidate_policy_sha256": _sha256(candidate),
            },
        )

    SUBMIT_ITERATION._validate_candidates(root, manifest, 0)
    pending = SUBMIT_ITERATION.prepare_batch_attempts(
        root=root,
        manifest=manifest,
        state=state,
        iteration=0,
    )
    assert len(pending) == 4
    covered = [lane_id for item in pending for lane_id in item["attempt"]["lane_ids"]]
    assert len(covered) == len(set(covered)) == CAMPAIGN_COMMON.LANE_COUNT
    assert set(covered) == set(lane_ids)
    identities = {item["batch"]["batch_id"]: item["attempt"]["submission_key"] for item in pending}

    replay = SUBMIT_ITERATION.prepare_batch_attempts(
        root=root,
        manifest=manifest,
        state=state,
        iteration=0,
    )
    assert {
        item["batch"]["batch_id"]: item["attempt"]["submission_key"] for item in replay
    } == identities
    assert all(len(item["batch"]["attempts"]) == 1 for item in replay)

    first = replay[0]["batch"]
    first_attempt = first["attempts"][-1]
    first_attempt["state"] = "failed"
    first["state"] = "retry_ready"
    first["retry_lane_ids"] = [first["lane_ids"][0]]
    retry = SUBMIT_ITERATION.prepare_batch_attempts(
        root=root,
        manifest=manifest,
        state=state,
        iteration=0,
    )
    retried_first = next(item for item in retry if item["batch"]["batch_id"] == first["batch_id"])
    assert retried_first["attempt"]["attempt"] == 2
    assert retried_first["attempt"]["lane_ids"] == [first["lane_ids"][0]]
    assert retried_first["attempt"]["submission_key"] != identities[first["batch_id"]]

    retried_first["batch"]["state"] = "submitted"
    pending_after_submission = SUBMIT_ITERATION.prepare_batch_attempts(
        root=root,
        manifest=manifest,
        state=state,
        iteration=0,
    )
    assert all(item["batch"]["batch_id"] != first["batch_id"] for item in pending_after_submission)
    assert len(retried_first["batch"]["attempts"]) == 2


def test_collector_prevalidation_separates_retryable_output_from_fatal_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    lane_ids = [f"lane_{index:02d}" for index in range(CAMPAIGN_COMMON.LANE_COUNT)]
    lanes: list[dict[str, Any]] = []
    state_lanes: dict[str, dict[str, Any]] = {}
    missing_lane = lane_ids[-3]
    arithmetic_lane = lane_ids[-2]
    lineage_lane = lane_ids[-1]

    for lane_id in lane_ids:
        lane = _lane("reaction_aware")
        lane["lane_id"] = lane_id
        lane["reaction_key"] = f"reaction_{lane_id}"
        candidate = CAMPAIGN_COMMON.candidate_path(root, 0, lane_id)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(f"# baseline {lane_id}\n", encoding="utf-8")
        digest = _sha256(candidate)
        lane["seed_policy_sha256"] = digest
        lanes.append(lane)
        state_lanes[lane_id] = {"proposal_count": 0}
        CAMPAIGN_COMMON.write_json(
            CAMPAIGN_COMMON.mutation_dir(root, 0, lane_id) / "record.json",
            {
                "schema_version": CAMPAIGN_COMMON.SCHEMA_VERSION,
                "lane_id": lane_id,
                "iteration": 0,
                "kind": "baseline",
                "status": "candidate_ready",
                "parent_iteration": None,
                "parent_policy_sha256": None,
                "candidate_policy_sha256": ("f" * 64 if lane_id == lineage_lane else digest),
            },
        )
        if lane_id in {missing_lane, lineage_lane}:
            continue
        result = _result_for_lane(lane, iteration=0, digest=digest)
        lane["seed_result_expectation"] = _seed_expectation(result)
        if lane_id == arithmetic_lane:
            result["triangle_bound_hartree"] = 0.0
        CAMPAIGN_COMMON.write_json(CAMPAIGN_COMMON.result_path(root, 0, lane_id), result)

    results, candidate_hashes, retryable, fatal = COLLECT_ITERATION.prevalidate_iteration(
        root=root,
        manifest={"lanes": lanes},
        state={"lanes": state_lanes},
        iteration=0,
    )
    assert set(results) == set(lane_ids) - {
        missing_lane,
        arithmetic_lane,
        lineage_lane,
    }, fatal
    assert set(candidate_hashes) == set(lane_ids)
    assert set(retryable) == {missing_lane}
    assert set(fatal) == {arithmetic_lane, lineage_lane}
    assert "arithmetic mismatch" in fatal[arithmetic_lane]
    assert "immutable mutation record disagree" in fatal[lineage_lane]


def test_collector_rejects_iterative_proposal_without_durable_launch_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    lane = _lane("reaction_aware")
    lane_id = lane["lane_id"]
    parent = root / "lanes" / lane_id / "simple_pf.py"
    parent.parent.mkdir(parents=True)
    parent.write_text("# incumbent\n", encoding="utf-8")
    parent_hash = _sha256(parent)
    candidate = CAMPAIGN_COMMON.candidate_path(root, 1, lane_id)
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# unproven proposal\n", encoding="utf-8")
    candidate_hash = _sha256(candidate)
    CAMPAIGN_COMMON.write_json(
        CAMPAIGN_COMMON.mutation_dir(root, 1, lane_id) / "record.json",
        {
            "schema_version": CAMPAIGN_COMMON.SCHEMA_VERSION,
            "lane_id": lane_id,
            "iteration": 1,
            "kind": "autoresearch_proposal",
            "status": "candidate_ready",
            "parent_iteration": 0,
            "parent_policy_sha256": parent_hash,
            "candidate_policy_sha256": candidate_hash,
        },
    )
    results, _hashes, retryable, fatal = COLLECT_ITERATION.prevalidate_iteration(
        root=root,
        manifest={"lanes": [lane]},
        state={
            "lanes": {
                lane_id: {
                    "proposal_count": 0,
                    "incumbent_iteration": 0,
                    "incumbent_policy_sha256": parent_hash,
                }
            }
        },
        iteration=1,
    )
    assert results == {}
    assert retryable == {}
    assert set(fatal) == {lane_id}
    assert "durable prompt/launch evidence" in fatal[lane_id]


def test_collection_transaction_replays_after_crash_before_state_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "campaign"
    old_state = {"schema_version": 1, "current_collected_iteration": 0, "sentinel": "old"}
    CAMPAIGN_COMMON.write_json(root / "state.json", old_state)
    decisions: list[dict[str, Any]] = []
    desired_lanes: dict[str, dict[str, Any]] = {}
    for lane_id in ("lane_a", "lane_b"):
        candidate = CAMPAIGN_COMMON.candidate_path(root, 1, lane_id)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(f"# accepted {lane_id}\n", encoding="utf-8")
        result = CAMPAIGN_COMMON.result_path(root, 1, lane_id)
        CAMPAIGN_COMMON.write_json(result, {"lane_id": lane_id, "status": "ok"})
        candidate_hash = _sha256(candidate)
        result_hash = _sha256(result)
        decisions.append(
            {
                "lane_id": lane_id,
                "iteration": 1,
                "accepted": True,
                "candidate_policy_sha256": candidate_hash,
                "result_sha256": result_hash,
            }
        )
        desired_lanes[lane_id] = {
            "incumbent_policy_path": candidate.relative_to(root).as_posix(),
            "incumbent_policy_sha256": candidate_hash,
        }

    desired_state = {
        "schema_version": 1,
        "current_collected_iteration": 1,
        "next_iteration": 2,
        "lanes": desired_lanes,
        "batches": {},
    }
    transaction_path = root / "transactions" / "collect_iter_0001.json"
    CAMPAIGN_COMMON.write_json(
        transaction_path,
        {
            "schema_version": CAMPAIGN_COMMON.SCHEMA_VERSION,
            "phase": "prepared",
            "iteration": 1,
            "created_at": "2026-07-15T00:00:00+0000",
            "decisions": decisions,
            "desired_state": desired_state,
        },
    )

    def crash_before_state(_root: Path, _state: dict[str, Any]) -> None:
        raise RuntimeError("injected crash before state commit")

    monkeypatch.setattr(COLLECT_ITERATION, "write_state", crash_before_state)
    with pytest.raises(RuntimeError, match="injected crash"):
        COLLECT_ITERATION.replay_collect_transaction(root, transaction_path)

    assert CAMPAIGN_COMMON.read_json(transaction_path)["phase"] == "applying"
    assert CAMPAIGN_COMMON.read_json(root / "state.json") == old_state
    assert (CAMPAIGN_COMMON.iteration_dir(root, 1) / "decisions.json").is_file()
    for lane_id in desired_lanes:
        lane_root = root / "lanes" / lane_id
        assert (
            _sha256(lane_root / "simple_pf.py") == desired_lanes[lane_id]["incumbent_policy_sha256"]
        )
        assert (lane_root / "last_result.json").is_file()
        assert (lane_root / "best" / "simple_pf.py").is_file()
        assert (lane_root / "best" / "result.json").is_file()

    monkeypatch.setattr(COLLECT_ITERATION, "write_state", CAMPAIGN_COMMON.write_state)
    COLLECT_ITERATION.replay_collect_transaction(root, transaction_path)
    committed = CAMPAIGN_COMMON.read_json(transaction_path)
    state = CAMPAIGN_COMMON.read_json(root / "state.json")
    assert committed["phase"] == "committed"
    assert len(committed["desired_state_sha256"]) == 64
    assert state["current_collected_iteration"] == 1
    assert state["next_iteration"] == 2
    assert state["lanes"] == desired_lanes

    before = _tree_snapshot(root)
    COLLECT_ITERATION.replay_collect_transaction(root, transaction_path)
    assert _tree_snapshot(root) == before


def test_exporter_rejects_any_incomplete_paper_shape() -> None:
    incomplete = {
        "schema_version": 1,
        "experiment": "reactions10_autoresearch",
        "proposal_iterations": 50,
        "evaluation_iterations": {"first": 0, "last": 50},
        "lane_count": 1,
        "reaction_count": 10,
        "chemical_accuracy_hartree": TARGET_HARTREE,
        "lanes": [],
    }
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="exactly 20"):
        EXPORT_RESULTS.validate_manifest(incomplete)


def test_exporter_snapshot_tree_rejects_undeclared_files_and_symlinks(
    tmp_path: Path,
) -> None:
    for name in ("source", "runner"):
        snapshot = tmp_path / name
        snapshot.mkdir()
        declared = snapshot / "declared.py"
        declared.write_text(f"# fixed {name}\n", encoding="utf-8")
        records = {
            "declared.py": {
                "sha256": _sha256(declared),
                "size_bytes": declared.stat().st_size,
            }
        }
        EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)

        extra = snapshot / "undeclared.py"
        extra.write_text("# must not be silently ignored\n", encoding="utf-8")
        with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="file set differs"):
            EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)
        extra.unlink()

        link = snapshot / "undeclared-link.py"
        link.symlink_to(declared.name)
        with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="symlinks"):
            EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)
        link.unlink()


def test_every_reaction_runner_cli_supports_help() -> None:
    cli_scripts = sorted(
        path
        for path in RUNNER_ROOT.glob("*.py")
        if 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
    )
    assert {path.name for path in cli_scripts} >= {
        "evaluate_worker.py",
        "export_results.py",
        "mutate_iteration.py",
        "prepare_campaign.py",
    }
    for script in cli_scripts:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=RUNNER_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, f"{script.name}: {completed.stderr or completed.stdout}"
        assert "usage:" in completed.stdout.lower(), script.name
