from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
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


def test_runner_readme_uses_the_locked_runner_projects_for_every_stage() -> None:
    readme = (RUNNER_ROOT / "README.md").read_text(encoding="utf-8")

    assert 'uv sync --locked --project "$SOURCE_ENV"' in readme
    assert (
        'uv run --locked --project "$SOURCE_ENV" python "$SOURCE_RUNNER/prepare_campaign.py"'
        in readme
    )
    assert "uv sync --locked --project '$CLUSTER_ROOT/runner/environment'" in readme
    for script_name in (
        "verify_hamiltonians.py",
        "submit_iteration.py",
        "supervise_campaign.py",
        "export_results.py",
    ):
        assert 'uv run --locked --project "$CAMPAIGN_ENV" python' in readme
        assert f'"$CAMPAIGN_RUNNER/{script_name}"' in readme
    assert "python /external/reactions10_run/runner/" not in readme


def test_supervisor_shell_quotes_slurm_format_for_csh_login_shell(monkeypatch) -> None:
    supervisor = _load_module("supervise_campaign")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="123|RUNNING\n", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)

    assert supervisor.query_slurm_states("nvlogin", ["123"], 30) == {"123": "RUNNING"}
    assert commands == [["ssh", "nvlogin", "squeue -h -j 123 -o '%A|%T'"]]


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
TEST_PREPARATION_NONCE = "1" * 64
TEST_CAMPAIGN_ID = CAMPAIGN_COMMON.derive_campaign_id(TEST_PREPARATION_NONCE)
VERIFY_HAMILTONIANS = _load_module("verify_hamiltonians", campaign_common=CAMPAIGN_COMMON)
PREPARE_CAMPAIGN = _load_module("prepare_campaign", campaign_common=CAMPAIGN_COMMON)
MUTATE_ITERATION = _load_module("mutate_iteration", campaign_common=CAMPAIGN_COMMON)
SUBMIT_ITERATION = _load_module("submit_iteration", campaign_common=CAMPAIGN_COMMON)
_PREVIOUS_MUTATE_ITERATION = sys.modules.get("mutate_iteration")
sys.modules["mutate_iteration"] = MUTATE_ITERATION
try:
    COLLECT_ITERATION = _load_module("collect_iteration", campaign_common=CAMPAIGN_COMMON)
finally:
    if _PREVIOUS_MUTATE_ITERATION is None:
        sys.modules.pop("mutate_iteration", None)
    else:
        sys.modules["mutate_iteration"] = _PREVIOUS_MUTATE_ITERATION
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


def _write_pauli_verification_fixture(root: Path, *, tamper_pauli: bool) -> Path:
    root.mkdir()
    npz_path = root / "two_qubit.npz"
    metadata_path = npz_path.with_suffix(".json")
    identity = 0.5
    labels = np.asarray(["XI", "IY", "ZX"])
    coefficients = np.asarray([0.25, -0.4, 0.75], dtype=np.float64)
    paulis = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }
    hamiltonian = identity * np.eye(4, dtype=np.complex128) + sum(
        coefficient * np.kron(paulis[label[0]], paulis[label[1]])
        for label, coefficient in zip(labels.tolist(), coefficients.tolist(), strict=True)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    stored_coefficients = coefficients.copy()
    if tamper_pauli:
        stored_coefficients[0] += 0.125
    np.savez_compressed(
        npz_path,
        hamiltonian_real=np.real(hamiltonian),
        hamiltonian_imag=np.imag(hamiltonian),
        sector_indices=np.arange(4, dtype=np.int64),
        sector_eigenvalues=eigenvalues,
        sector_eigenvectors_real=np.real(eigenvectors),
        sector_eigenvectors_imag=np.imag(eigenvectors),
        ground_state_real=np.real(eigenvectors[:, 0]),
        ground_state_imag=np.imag(eigenvectors[:, 0]),
        pauli_labels=labels,
        pauli_coefficients=stored_coefficients,
    )
    CAMPAIGN_COMMON.write_json(
        metadata_path,
        {
            "molecule": "two_qubit",
            "n_qubits": 2,
            "active_electrons": 1,
            "identity_coefficient": identity,
            "total_pauli_rotation_terms": len(labels),
        },
    )
    return npz_path


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
        "campaign_id": TEST_CAMPAIGN_ID,
        "campaign_manifest_sha256": "d" * 64,
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
        "candidate_relative_path": (
            f"iterations/iter_{iteration:04d}/candidates/{lane['lane_id']}/simple_pf.py"
        ),
        "problem_cache_relative_path": "problem_cache",
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
    launch_status: str = "codex_started",
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
    work_policy = work_dir / "simple_pf.py"
    work_policy.write_bytes(parent_policy.read_bytes())
    candidate = campaign_root / "iterations" / "iter_0001" / "candidates" / lane_id / "simple_pf.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(parent_policy.read_bytes())
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
    last_message.write_text("synthetic Codex completion\n", encoding="utf-8")
    stdout_path = mutation_root / "codex.stdout"
    stderr_path = mutation_root / "codex.stderr"
    start_event = {"type": "thread.started", "thread_id": "thread-synthetic"}
    terminal_event = {
        "type": "turn.completed",
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    stdout_path.write_text(
        json.dumps(start_event, sort_keys=True)
        + "\n"
        + json.dumps(terminal_event, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")
    launch_nonce = "e" * 64
    start_event_hash = MUTATE_ITERATION.canonical_event_sha256(start_event)
    terminal_event_hash = MUTATE_ITERATION.canonical_event_sha256(terminal_event)
    usage = terminal_event["usage"]
    usage_hash = CAMPAIGN_COMMON.sha256_text(
        json.dumps(usage, sort_keys=True, separators=(",", ":"))
    )
    sanitized_command = [
        "npx",
        "-y",
        CAMPAIGN_COMMON.CODEX_PACKAGE,
        "exec",
        "--json",
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
            "launch_nonce": launch_nonce,
            "exec_ack_protocol": MUTATE_ITERATION.EXEC_ACK_PROTOCOL,
            "required_start_event": MUTATE_ITERATION.CODEX_START_EVENT,
            "required_terminal_event": MUTATE_ITERATION.CODEX_TERMINAL_EVENT,
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
            "exec_pid": 4243,
            "work_dir": str(work_dir),
            "command_sha256": launch_command_hash,
            "status": launch_status,
            "exec_ack_protocol": MUTATE_ITERATION.EXEC_ACK_PROTOCOL,
            "launch_nonce": launch_nonce,
            "low_level_exec_status": "exec_confirmed",
            "codex_thread_id": "thread-synthetic",
            "codex_start_event_sha256": start_event_hash,
            "started_at_epoch": 1.0,
            "exec_confirmed_at_epoch": 2.0,
            "codex_started_at_epoch": 3.0,
        },
    )
    process_path = mutation_root / "process.json"
    returncode: int | None = 0 if with_process else None
    if with_process:
        CAMPAIGN_COMMON.write_json(
            process_path,
            {
                "command": sanitized_command,
                "command_sha256": launch_command_hash,
                "returncode": returncode,
                "timed_out": False,
                "process_error": None,
                "lifecycle_error": None,
                "pid": 4242,
                "pgid": 4242,
                "exec_pid": 4243,
                "exec_acknowledgement_status": "codex_started",
                "exec_ack_protocol": MUTATE_ITERATION.EXEC_ACK_PROTOCOL,
                "launch_nonce": launch_nonce,
                "codex_thread_id": "thread-synthetic",
                "codex_start_event_sha256": start_event_hash,
                "codex_lifecycle_status": "turn_completed",
                "codex_terminal_event_sha256": terminal_event_hash,
                "codex_usage": usage,
                "codex_usage_sha256": usage_hash,
                "codex_stdout_sha256": _sha256(stdout_path),
                "codex_stderr_sha256": _sha256(stderr_path),
                "last_message_sha256": _sha256(last_message),
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
        "launch_nonce": launch_nonce,
        "launch_intent_sha256": _sha256(launch_intent_path),
        "launch_receipt_sha256": _sha256(launch_receipt_path),
        "exec_acknowledgement_sha256": _sha256(launch_receipt_path),
        "exec_acknowledgement_status": "codex_started",
        "exec_ack_protocol": MUTATE_ITERATION.EXEC_ACK_PROTOCOL,
        "codex_thread_id": "thread-synthetic",
        "codex_start_event_sha256": start_event_hash,
        "codex_lifecycle_status": "turn_completed",
        "codex_terminal_event_sha256": terminal_event_hash,
        "codex_usage": usage,
        "codex_usage_sha256": usage_hash,
        "archived_attempt_evidence": [],
        "archived_attempt_directory_paths": [],
        "process_sha256": _sha256(process_path) if with_process else None,
        "codex_stdout_sha256": _sha256(stdout_path),
        "codex_stderr_sha256": _sha256(stderr_path),
        "last_message_sha256": _sha256(last_message),
        "work_file_records": {
            "simple_pf.py": {
                "sha256": _sha256(work_policy),
                "size_bytes": work_policy.stat().st_size,
            }
        },
        "work_directory_paths": [],
        "work_policy_sha256": _sha256(work_policy),
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
        "validation_error": None,
        "unexpected_work_files": [],
        "recovered_after_interruption": recovered,
        "created_at": "2026-07-15T00:00:00+0000",
        "completed_at": "2026-07-15T00:00:02+0000",
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
        "preparation_nonce": TEST_PREPARATION_NONCE,
        "campaign_id": TEST_CAMPAIGN_ID,
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
        "campaign_id": TEST_CAMPAIGN_ID,
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


def test_campaign_cache_verification_requires_an_exact_regular_tree(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    cache_root = root / "problem_cache"
    cache_root.parent.mkdir(parents=True)
    names = _write_synthetic_cache(cache_root, tmp_path / "problem_cache.sha256")
    manifest = {
        "inputs": {"problem_cache_files": {name: _sha256(cache_root / name) for name in names}}
    }
    CAMPAIGN_COMMON.verify_campaign_cache(root, manifest)

    declared = cache_root / names[0]
    backup = tmp_path / "declared-backup"
    backup.write_bytes(declared.read_bytes())
    declared.unlink()
    declared.symlink_to(backup)
    with pytest.raises(RuntimeError, match="symlink|regular|tree|cache"):
        CAMPAIGN_COMMON.verify_campaign_cache(root, manifest)
    declared.unlink()
    declared.write_bytes(backup.read_bytes())

    empty = cache_root / "undeclared-empty-directory"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="director|tree|cache"):
        CAMPAIGN_COMMON.verify_campaign_cache(root, manifest)
    empty.rmdir()

    fifo = cache_root / "undeclared.fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(RuntimeError, match="special|tree|cache"):
            CAMPAIGN_COMMON.verify_campaign_cache(root, manifest)
    finally:
        fifo.unlink()


def test_runtime_snapshot_verification_requires_an_exact_regular_tree(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    source_root = root / "source" / "repo"
    runner_root = root / "runner"
    declared_paths = {
        "source_files": (source_root, (Path("package/core.py"),)),
        "runner_files": (
            runner_root,
            (Path("campaign_common.py"), Path("environment/uv.lock")),
        ),
    }
    manifest: dict[str, Any] = {}
    for key, (base, relative_paths) in declared_paths.items():
        records: dict[str, dict[str, Any]] = {}
        for relative in relative_paths:
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# fixed {key}: {relative.as_posix()}\n", encoding="utf-8")
            records[relative.as_posix()] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        manifest[key] = records

    # Explicit non-import runtime trees may be ignored, but bytecode caches
    # are executable input and must never be silently pruned.
    pytest_cache = source_root / "package" / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "README.md").write_text("mutable test cache\n", encoding="utf-8")
    venv_bin = runner_root / "environment" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("runtime environment\n", encoding="utf-8")
    CAMPAIGN_COMMON.verify_snapshot_records(root, manifest)

    pycache = source_root / "package" / "__pycache__"
    pycache.mkdir()
    (pycache / "core.cpython.pyc").write_bytes(b"untrusted executable bytecode")
    with pytest.raises(RuntimeError, match="undeclared|__pycache__"):
        CAMPAIGN_COMMON.verify_snapshot_records(root, manifest)
    (pycache / "core.cpython.pyc").unlink()
    pycache.rmdir()

    extra = source_root / "undeclared.py"
    extra.write_text("# must fail closed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="undeclared_files"):
        CAMPAIGN_COMMON.verify_snapshot_records(root, manifest)
    extra.unlink()

    unexpected_directory = runner_root / "unexpected-empty-directory"
    unexpected_directory.mkdir()
    with pytest.raises(RuntimeError, match="unexpected_directories"):
        CAMPAIGN_COMMON.verify_snapshot_records(root, manifest)
    unexpected_directory.rmdir()

    symlink = source_root / "undeclared-link.py"
    symlink.symlink_to("package/core.py")
    with pytest.raises(RuntimeError, match="contains symlinks"):
        CAMPAIGN_COMMON.verify_snapshot_records(root, manifest)
    symlink.unlink()

    runtime_named_symlink = source_root / "package" / ".venv"
    runtime_named_symlink.symlink_to(source_root / "package", target_is_directory=True)
    with pytest.raises(RuntimeError, match="contains symlinks"):
        CAMPAIGN_COMMON.verify_snapshot_records(root, manifest)
    runtime_named_symlink.unlink()

    fifo = runner_root / "undeclared.fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(RuntimeError, match="special entries"):
            CAMPAIGN_COMMON.verify_snapshot_records(root, manifest)
    finally:
        fifo.unlink()


@pytest.mark.parametrize(
    "script_name",
    [
        "prepare_campaign.py",
        "verify_hamiltonians.py",
        "mutate_iteration.py",
        "submit_iteration.py",
        "collect_iteration.py",
        "evaluate_worker.py",
        "supervise_campaign.py",
        "export_results.py",
    ],
)
def test_runner_cli_ignores_attacker_controlled_bytecode_prefix(
    tmp_path: Path,
    script_name: str,
) -> None:
    """A stale unchecked pyc must not execute before snapshot verification."""

    attacker_prefix = tmp_path / "attacker-pycache"
    marker = tmp_path / "malicious-bytecode-executed"
    campaign_common_path = RUNNER_ROOT / "campaign_common.py"
    malicious_source = tmp_path / "campaign_common_malicious.py"
    malicious_source.write_text(
        campaign_common_path.read_text(encoding="utf-8")
        + "\nfrom pathlib import Path as _AttackPath\n"
        + f"_AttackPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    previous_prefix = sys.pycache_prefix
    try:
        sys.pycache_prefix = str(attacker_prefix)
        malicious_pyc = Path(importlib.util.cache_from_source(str(campaign_common_path)))
    finally:
        sys.pycache_prefix = previous_prefix
    malicious_pyc.parent.mkdir(parents=True)
    py_compile.compile(
        str(malicious_source),
        cfile=str(malicious_pyc),
        dfile=str(campaign_common_path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )

    completed = subprocess.run(
        [sys.executable, str(RUNNER_ROOT / script_name), "--help"],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONPYCACHEPREFIX": str(attacker_prefix)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, f"{script_name}: {completed.stderr or completed.stdout}"
    assert "usage:" in completed.stdout.lower()
    assert not marker.exists(), "runner imported attacker-controlled campaign_common bytecode"


@pytest.mark.parametrize(
    "value",
    [
        "/",
        "relative/path",
        "/cluster//campaign",
        "/cluster/./campaign",
        "/cluster/../campaign",
        "/cluster/campaign/",
        "/cluster/campaign\nsecond-command",
        "/cluster/campaign\x00suffix",
    ],
)
def test_remote_cluster_path_guard_rejects_noncanonical_or_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError, match="absolute POSIX|canonical|component|newline|NUL|root"):
        CAMPAIGN_COMMON.validate_absolute_posix_path(value, "cluster_root")

    with pytest.raises(ValueError):
        SUBMIT_ITERATION._remote_claim_command(
            cluster_root=value,
            campaign_id=TEST_CAMPAIGN_ID,
            manifest_sha256="d" * 64,
        )


def test_remote_claim_is_atomic_idempotent_and_identity_bound(tmp_path: Path) -> None:
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    flock = stub_bin / "flock"
    flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    flock.chmod(0o755)
    env = {**os.environ, "PATH": f"{stub_bin}:{os.environ.get('PATH', '')}"}

    def run_claim(
        root: Path, campaign_id: str = TEST_CAMPAIGN_ID
    ) -> subprocess.CompletedProcess[str]:
        command = SUBMIT_ITERATION._remote_claim_command(
            cluster_root=root.as_posix(),
            campaign_id=campaign_id,
            manifest_sha256="d" * 64,
        )
        syntax = subprocess.run(
            ["bash", "-n"],
            input=command,
            text=True,
            capture_output=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr
        return subprocess.run(
            ["bash", "-c", command],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    claimed = tmp_path / "claimed"
    first = run_claim(claimed)
    assert first.returncode == 0, first.stderr
    claim_path = claimed / SUBMIT_ITERATION.REMOTE_CLAIM_NAME
    assert CAMPAIGN_COMMON.read_json(claim_path) == {
        "campaign_id": TEST_CAMPAIGN_ID,
        "manifest_sha256": "d" * 64,
        "schema_version": SUBMIT_ITERATION.REMOTE_CLAIM_SCHEMA_VERSION,
    }
    claim_hash = _sha256(claim_path)

    repeated = run_claim(claimed)
    assert repeated.returncode == 0, repeated.stderr
    assert _sha256(claim_path) == claim_hash

    mismatched = run_claim(claimed, campaign_id="e" * 64)
    assert mismatched.returncode != 0
    assert "claim mismatch" in mismatched.stderr
    assert _sha256(claim_path) == claim_hash

    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "unrelated.txt").write_text("do not overwrite\n", encoding="utf-8")
    refused = run_claim(stale)
    assert refused.returncode != 0
    assert "non-empty unclaimed" in refused.stderr
    assert not (stale / SUBMIT_ITERATION.REMOTE_CLAIM_NAME).exists()

    venv_only = tmp_path / "venv-only"
    (venv_only / ".venv").mkdir(parents=True)
    allowed = run_claim(venv_only)
    assert allowed.returncode == 0, allowed.stderr


def test_generated_remote_submission_command_is_valid_bash() -> None:
    campaign_id = TEST_CAMPAIGN_ID
    manifest_hash = "d" * 64
    submission_key = "e" * 64
    batch = {
        "batch_id": "iter0007_batch00",
        "iteration": 7,
        "campaign_id": campaign_id,
        "manifest_sha256": manifest_hash,
    }
    attempt = {
        "attempt": 1,
        "campaign_id": campaign_id,
        "manifest_sha256": manifest_hash,
        "submission_key": submission_key,
        "job_name": f"rx_{submission_key[:32]}",
        "lane_ids_path": ("iterations/iter_0007/batches/iter0007_batch00/attempt_01/lane_ids.txt"),
    }
    command = SUBMIT_ITERATION._remote_submission_command(
        cluster_root="/cluster/hsopt/reactions10",
        cluster_python="/cluster/hsopt/reactions10/runner/environment/.venv/bin/python",
        campaign_id=campaign_id,
        manifest_sha256=manifest_hash,
        batch=batch,
        attempt=attempt,
        account="research",
        partition="batch",
        workers=2,
        cpus_per_task=16,
        memory="64G",
        time_limit="04:00:00",
    )
    syntax = subprocess.run(
        ["bash", "-n"],
        input=command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    assert campaign_id in command
    assert manifest_hash in command
    assert submission_key in command


def test_hamiltonian_verification_proves_pauli_sum_matches_dense_matrix(
    tmp_path: Path,
) -> None:
    valid_path = _write_pauli_verification_fixture(tmp_path / "valid", tamper_pauli=False)
    valid = VERIFY_HAMILTONIANS.verify_one(valid_path, _sha256(valid_path))
    assert valid["verified"] is True
    assert valid["pauli_dense_max_abs_residual_hartree"] < 1.0e-14
    assert valid["pauli_dense_relative_frobenius_residual"] < 1.0e-14

    tampered_path = _write_pauli_verification_fixture(tmp_path / "tampered", tamper_pauli=True)
    tampered = VERIFY_HAMILTONIANS.verify_one(tampered_path, _sha256(tampered_path))
    assert tampered["verified"] is False
    assert tampered["pauli_dense_max_abs_residual_hartree"] == pytest.approx(0.125)
    assert (
        tampered["pauli_dense_relative_frobenius_residual"]
        > VERIFY_HAMILTONIANS.PAULI_DENSE_RESIDUAL_TOLERANCE
    )


def test_submission_requires_recorded_pauli_dense_residuals(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    report_path = root / "baselines" / "hamiltonian_verification.json"
    cache_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for index in range(CAMPAIGN_COMMON.EXPECTED_CACHE_SPECIES):
        stem = f"species_{index:02d}"
        npz_name = f"{stem}.npz"
        json_name = f"{stem}.json"
        npz_hash = hashlib.sha256(npz_name.encode()).hexdigest()
        json_hash = hashlib.sha256(json_name.encode()).hexdigest()
        cache_hashes[npz_name] = npz_hash
        cache_hashes[json_name] = json_hash
        rows.append(
            {
                "cache_file": npz_name,
                "cache_sha256": npz_hash,
                "metadata_sha256": json_hash,
                "verified": True,
                "ground_energy_delta_hartree": 0.0,
                "max_spectrum_delta_hartree": 0.0,
                "cached_ground_vector_residual": 0.0,
                "evaluator_ground_state_outside_sector_norm": 0.0,
                "evaluator_ground_state_residual": 0.0,
                "evaluator_recomputed_ground_eigenspace_overlap": 1.0,
                "evaluator_cached_ground_eigenspace_overlap": 1.0,
                "pauli_dense_max_abs_residual_hartree": 0.0,
                "pauli_dense_relative_frobenius_residual": 0.0,
            }
        )
    manifest_hash = "a" * 64
    manifest = {"inputs": {"problem_cache_files": cache_hashes}}
    state = {"manifest_sha256": manifest_hash}
    payload = {
        "all_verified": True,
        "hamiltonian_count": CAMPAIGN_COMMON.EXPECTED_CACHE_SPECIES,
        "manifest_sha256": manifest_hash,
        "rows": rows,
    }
    CAMPAIGN_COMMON.write_json(report_path, payload)
    state["hamiltonian_verification_sha256"] = _sha256(report_path)
    SUBMIT_ITERATION.validate_hamiltonian_verification(root, manifest, state)

    rows[0]["pauli_dense_max_abs_residual_hartree"] = 1.0e-6
    CAMPAIGN_COMMON.write_json(report_path, payload)
    state["hamiltonian_verification_sha256"] = _sha256(report_path)
    with pytest.raises(RuntimeError, match="row/hash mismatch"):
        SUBMIT_ITERATION.validate_hamiltonian_verification(root, manifest, state)


def test_exporter_independently_rejects_empty_or_unverified_hamiltonian_rows(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign"
    cache_root = campaign_root / "problem_cache"
    checksums = tmp_path / "problem_cache.sha256"
    cache_root.parent.mkdir(parents=True)
    cache_names = _write_synthetic_cache(cache_root, checksums)
    for metadata_name in sorted(name for name in cache_names if name.endswith(".json")):
        CAMPAIGN_COMMON.write_json(
            cache_root / metadata_name,
            {
                "molecule": Path(metadata_name).stem,
                "n_qubits": 2,
                "active_electrons": 1,
            },
        )
    cache_hashes = {name: _sha256(cache_root / name) for name in cache_names}
    manifest_hash = "a" * 64
    manifest = {"inputs": {"problem_cache_files": cache_hashes}}
    rows: list[dict[str, Any]] = []
    for npz_name in sorted(name for name in cache_names if name.endswith(".npz")):
        stem = Path(npz_name).stem
        rows.append(
            {
                "species": stem,
                "cache_file": npz_name,
                "cache_sha256": cache_hashes[npz_name],
                "metadata_sha256": cache_hashes[f"{stem}.json"],
                "n_qubits": 2,
                "active_electrons": 1,
                "sector_dimension": 2,
                "hermiticity_error": 0.0,
                "pauli_dense_max_abs_residual_hartree": 0.0,
                "pauli_dense_relative_frobenius_residual": 0.0,
                "cached_ground_energy_hartree": -1.0,
                "recomputed_ground_energy_hartree": -1.0,
                "ground_energy_delta_hartree": 0.0,
                "max_spectrum_delta_hartree": 0.0,
                "cached_ground_vector_residual": 0.0,
                "evaluator_ground_state_input_norm": 1.0,
                "evaluator_ground_state_outside_sector_norm": 0.0,
                "evaluator_ground_state_residual": 0.0,
                "ground_eigenspace_dimension": 1,
                "evaluator_recomputed_ground_eigenspace_overlap": 1.0,
                "evaluator_cached_ground_eigenspace_overlap": 1.0,
                "verified": True,
                "wall_seconds": 0.01,
            }
        )
    report = {
        "schema_version": 1,
        "experiment": "reactions10_hamiltonian_cache_verification",
        "diagonalization": "numpy.linalg.eigh on fixed-electron Hamiltonian sectors",
        "pauli_dense_check": (
            "identity_coefficient plus cached evaluator Pauli labels/coefficients "
            "against the cached dense Hamiltonian"
        ),
        "hamiltonian_count": CAMPAIGN_COMMON.EXPECTED_CACHE_SPECIES,
        "all_verified": True,
        "manifest_sha256": manifest_hash,
        "verified_at": "2026-07-15T00:00:00+0000",
        "rows": rows,
    }
    report_path = campaign_root / "baselines" / "hamiltonian_verification.json"
    CAMPAIGN_COMMON.write_json(report_path, report)
    EXPORT_RESULTS.validate_hamiltonian_verification(
        report_path,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        campaign_root=campaign_root,
    )

    empty_rows = copy.deepcopy(report)
    empty_rows["rows"] = [{} for _ in range(CAMPAIGN_COMMON.EXPECTED_CACHE_SPECIES)]
    CAMPAIGN_COMMON.write_json(report_path, empty_rows)
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="row|cache|species"):
        EXPORT_RESULTS.validate_hamiltonian_verification(
            report_path,
            manifest=manifest,
            manifest_sha256=manifest_hash,
            campaign_root=campaign_root,
        )

    unverified = copy.deepcopy(report)
    unverified["rows"][0]["verified"] = False
    CAMPAIGN_COMMON.write_json(report_path, unverified)
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="verified|verification"):
        EXPORT_RESULTS.validate_hamiltonian_verification(
            report_path,
            manifest=manifest,
            manifest_sha256=manifest_hash,
            campaign_root=campaign_root,
        )


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
        "preparation_nonce": TEST_PREPARATION_NONCE,
        "campaign_id": TEST_CAMPAIGN_ID,
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
        "execution": {
            "cluster_root": "/cluster/reactions10",
            "cluster_python": "/cluster/reactions10/runner/environment/.venv/bin/python",
            "batch_size": 6,
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
        "campaign_id": TEST_CAMPAIGN_ID,
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
        campaign_id=TEST_CAMPAIGN_ID,
        campaign_manifest_sha256="d" * 64,
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
        campaign_id=TEST_CAMPAIGN_ID,
        campaign_manifest_sha256="d" * 64,
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
            campaign_id=TEST_CAMPAIGN_ID,
            campaign_manifest_sha256="d" * 64,
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
        manifest_sha256="d" * 64,
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
            manifest_sha256="d" * 64,
        )

    for provenance_field in (
        "campaign_manifest_sha256",
        "candidate_relative_path",
        "problem_cache_relative_path",
    ):
        missing_provenance = _result_for_lane(lane, iteration=3, digest=digest)
        missing_provenance.pop(provenance_field)
        with pytest.raises(
            EXPORT_RESULTS.ExportValidationError,
            match=provenance_field,
        ):
            EXPORT_RESULTS.validate_result(
                missing_provenance,
                lane=lane_spec,
                iteration=3,
                policy_path=policy,
                manifest_sha256="d" * 64,
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
        manifest_sha256="d" * 64,
    )
    EXPORT_RESULTS.validate_baseline_result(candidate)

    drifted = copy.deepcopy(expectation)
    drifted["species"]["a"]["t_gate_count"] += 1
    drifted_candidate = EXPORT_RESULTS.validate_result(
        result,
        lane=lane_spec(drifted),
        iteration=0,
        policy_path=policy,
        manifest_sha256="d" * 64,
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

    tampered_process = CAMPAIGN_COMMON.read_json(paths["process"])
    tampered_process["completed_at"] = "2026-07-15T00:00:03+0000"
    CAMPAIGN_COMMON.write_json(paths["process"], tampered_process)
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="process receipt file hash"):
        EXPORT_RESULTS.validate_mutation_record(record, **kwargs)

    bad_status_record, bad_status_kwargs, _paths = _export_mutation_fixture(
        tmp_path / "bad-status",
        with_process=True,
        recovered=False,
        launch_status="codex_start_error",
    )
    with pytest.raises(
        EXPORT_RESULTS.ExportValidationError,
        match="actual Codex launch|codex_started|startup",
    ):
        EXPORT_RESULTS.validate_mutation_record(bad_status_record, **bad_status_kwargs)

    recovered_record, recovered_kwargs, _paths = _export_mutation_fixture(
        tmp_path / "recovered",
        with_process=False,
        recovered=True,
    )
    with pytest.raises(
        EXPORT_RESULTS.ExportValidationError,
        match="process|completion|lifecycle|returncode|provenance",
    ):
        EXPORT_RESULTS.validate_mutation_record(recovered_record, **recovered_kwargs)

    missing_process_record, missing_process_kwargs, _paths = _export_mutation_fixture(
        tmp_path / "missing-process",
        with_process=False,
        recovered=False,
    )
    with pytest.raises(
        EXPORT_RESULTS.ExportValidationError,
        match="process|completion|lifecycle|returncode|provenance",
    ):
        EXPORT_RESULTS.validate_mutation_record(
            missing_process_record,
            **missing_process_kwargs,
        )


def test_completed_mutation_record_is_idempotent_and_immutable(tmp_path: Path) -> None:
    record, kwargs, paths = _export_mutation_fixture(
        tmp_path / "complete-runner-record",
        with_process=True,
        recovered=False,
    )
    root = kwargs["campaign_root"]
    lane = _lane("reaction_aware")
    lane_id = lane["lane_id"]
    parent_hash = kwargs["expected_parent_sha256"]
    candidate = CAMPAIGN_COMMON.candidate_path(root, 1, lane_id)
    directory = CAMPAIGN_COMMON.mutation_dir(root, 1, lane_id)
    process = paths["process"]
    CAMPAIGN_COMMON.write_json(directory / "record.json", record)

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

    candidate_empty = candidate.parent / "undeclared-empty-directory"
    candidate_empty.mkdir()
    with pytest.raises(
        RuntimeError,
        match="candidate directory|record drift|isolated|symlink|evidence tree",
    ):
        MUTATE_ITERATION.mutation_record_is_complete(
            root=root,
            lane=lane,
            iteration=1,
            parent_hash=parent_hash,
        )
    candidate_empty.rmdir()

    candidate_link = candidate.parent / "policy-link.py"
    candidate_link.symlink_to("simple_pf.py")
    with pytest.raises(
        RuntimeError,
        match="candidate directory|record drift|isolated|symlink|evidence tree",
    ):
        MUTATE_ITERATION.mutation_record_is_complete(
            root=root,
            lane=lane,
            iteration=1,
            parent_hash=parent_hash,
        )
    candidate_link.unlink()

    original_process = process.read_bytes()
    process.write_text('{"returncode": 7}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="record drift"):
        MUTATE_ITERATION.mutation_record_is_complete(
            root=root,
            lane=lane,
            iteration=1,
            parent_hash=parent_hash,
        )
    process.write_bytes(original_process)

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


def test_shell_style_pre_codex_failure_cannot_satisfy_tool_acknowledgement(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    receipt_path = tmp_path / "launch.json"
    launch_nonce = "f" * 64
    command = ["/bin/sh", "-c", "exit 127"]
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_ROOT / "launch_codex.py"),
            "--receipt",
            str(receipt_path),
            "--work-dir",
            str(work_dir),
            "--launch-nonce",
            launch_nonce,
            "--",
            *command,
        ],
        text=True,
        capture_output=True,
        check=False,
        start_new_session=True,
    )
    assert completed.returncode == 127
    receipt = CAMPAIGN_COMMON.read_json(receipt_path)
    assert receipt["low_level_exec_status"] == "exec_confirmed"
    assert receipt["status"] == "codex_start_error"
    assert "codex_thread_id" not in receipt
    with pytest.raises(RuntimeError, match="Codex exec acknowledgement"):
        MUTATE_ITERATION.validate_exec_acknowledgement(
            receipt,
            command_sha256=hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
            work_dir=work_dir,
            launch_nonce=launch_nonce,
        )
    assert not (tmp_path / "record.json").exists()
    assert not (tmp_path / "candidate" / "simple_pf.py").exists()


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
        "campaign_id": TEST_CAMPAIGN_ID,
        "execution": {"cluster_root": "/cluster/reactions10", "batch_size": 6},
        "lanes": [{"lane_id": lane_id} for lane_id in lane_ids],
    }
    state: dict[str, Any] = {"batches": {}, "manifest_sha256": "d" * 64}
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


def test_submission_key_is_bound_to_campaign_and_manifest_identity(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    lane_id = "lane_00"
    candidate = CAMPAIGN_COMMON.candidate_path(root, 0, lane_id)
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# immutable candidate\n", encoding="utf-8")
    common = {
        "cluster_root": "/cluster/reactions10",
        "iteration": 0,
        "batch_id": "iter0000_batch00",
        "attempt": 1,
        "lane_ids": [lane_id],
        "root": root,
    }
    base_key, base_job = SUBMIT_ITERATION._batch_identity(
        campaign_id="a" * 64,
        manifest_sha256="b" * 64,
        **common,
    )
    campaign_key, campaign_job = SUBMIT_ITERATION._batch_identity(
        campaign_id="c" * 64,
        manifest_sha256="b" * 64,
        **common,
    )
    manifest_key, manifest_job = SUBMIT_ITERATION._batch_identity(
        campaign_id="a" * 64,
        manifest_sha256="d" * 64,
        **common,
    )
    assert len({base_key, campaign_key, manifest_key}) == 3
    assert all(re.fullmatch(r"[0-9a-f]{64}", key) for key in (base_key, campaign_key, manifest_key))
    assert base_job == f"rx_{base_key[:32]}"
    assert campaign_job == f"rx_{campaign_key[:32]}"
    assert manifest_job == f"rx_{manifest_key[:32]}"


def test_exporter_rejects_identity_free_noncollected_or_incomplete_batches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    lane_ids = [f"lane_{index:02d}" for index in range(CAMPAIGN_COMMON.LANE_COUNT)]
    manifest_hash = "d" * 64
    cluster_root = "/cluster/reactions10"
    cluster_python = "/cluster/reactions10/runner/environment/.venv/bin/python"
    batch_size = 6
    manifest = {
        "campaign_id": TEST_CAMPAIGN_ID,
        "execution": {
            "cluster_root": cluster_root,
            "cluster_python": cluster_python,
            "batch_size": batch_size,
        },
        "lanes": [{"lane_id": lane_id} for lane_id in lane_ids],
    }
    state: dict[str, Any] = {
        "campaign_id": TEST_CAMPAIGN_ID,
        "batches": {},
        "manifest_sha256": manifest_hash,
    }
    for lane_id in lane_ids:
        candidate = CAMPAIGN_COMMON.candidate_path(root, 0, lane_id)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(f"# immutable {lane_id}\n", encoding="utf-8")

    execution_spec = {
        "sync_host": "omni",
        "submit_host": "omni-submit",
        "account": "research",
        "partition": "batch",
        "workers": 2,
        "cpus_per_task": 16,
        "memory": "64G",
        "time_limit": "04:00:00",
        "cluster_root": cluster_root,
        "cluster_python": cluster_python,
    }
    pending = SUBMIT_ITERATION.prepare_batch_attempts(
        root=root,
        manifest=manifest,
        state=state,
        iteration=0,
        execution_spec=execution_spec,
    )
    for index, item in enumerate(pending, 1):
        batch = item["batch"]
        attempt = item["attempt"]
        attempt.update(
            {
                "state": "collected",
                "job_id": str(9000 + index),
                "submitted_at": "2026-07-15T00:00:01+0000",
                "collected_at": "2026-07-15T00:00:02+0000",
            }
        )
        batch.update(
            {
                "state": "collected",
                "retry_lane_ids": [],
                "collected_at": "2026-07-15T00:00:02+0000",
                "updated_at": "2026-07-15T00:00:02+0000",
            }
        )
        CAMPAIGN_COMMON.write_json(
            CAMPAIGN_COMMON.iteration_dir(root, 0)
            / "batches"
            / str(batch["batch_id"])
            / "batch.json",
            batch,
        )

    iteration_root = CAMPAIGN_COMMON.iteration_dir(root, 0)
    validation_kwargs = {
        "iteration": 0,
        "batches": state["batches"],
        "lane_ids": lane_ids,
        "campaign_id": TEST_CAMPAIGN_ID,
        "manifest_sha256": manifest_hash,
        "cluster_root": cluster_root,
        "cluster_python": cluster_python,
        "batch_size": batch_size,
    }
    EXPORT_RESULTS.validate_batch_structure(iteration_root, **validation_kwargs)

    first_batch_id = sorted(state["batches"])[0]
    first_batch = state["batches"][first_batch_id]
    original_first = copy.deepcopy(first_batch)
    first_batch.pop("campaign_id")
    CAMPAIGN_COMMON.write_json(
        iteration_root / "batches" / first_batch_id / "batch.json",
        first_batch,
    )
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="campaign|identity|schema"):
        EXPORT_RESULTS.validate_batch_structure(iteration_root, **validation_kwargs)

    state["batches"][first_batch_id] = copy.deepcopy(original_first)
    state["batches"][first_batch_id]["state"] = "submitted"
    state["batches"][first_batch_id]["attempts"][-1]["state"] = "submitted"
    CAMPAIGN_COMMON.write_json(
        iteration_root / "batches" / first_batch_id / "batch.json",
        state["batches"][first_batch_id],
    )
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="collected|state"):
        EXPORT_RESULTS.validate_batch_structure(iteration_root, **validation_kwargs)

    state["batches"][first_batch_id] = original_first
    CAMPAIGN_COMMON.write_json(
        iteration_root / "batches" / first_batch_id / "batch.json",
        original_first,
    )
    incomplete = dict(state["batches"])
    incomplete.pop(sorted(incomplete)[-1])
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="cover|partition|structure"):
        EXPORT_RESULTS.validate_batch_structure(
            iteration_root,
            **{**validation_kwargs, "batches": incomplete},
        )


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
        manifest={"campaign_id": TEST_CAMPAIGN_ID, "lanes": lanes},
        state={
            "campaign_id": TEST_CAMPAIGN_ID,
            "manifest_sha256": "d" * 64,
            "lanes": state_lanes,
        },
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
        manifest={"campaign_id": TEST_CAMPAIGN_ID, "lanes": [lane]},
        state={
            "campaign_id": TEST_CAMPAIGN_ID,
            "manifest_sha256": "d" * 64,
            "lanes": {
                lane_id: {
                    "proposal_count": 0,
                    "incumbent_iteration": 0,
                    "incumbent_policy_sha256": parent_hash,
                }
            },
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
        "execution": {
            "cluster_root": "/cluster/reactions10",
            "cluster_python": "/cluster/reactions10/runner/environment/.venv/bin/python",
            "batch_size": 6,
        },
        "lanes": [],
    }
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="exactly 20"):
        EXPORT_RESULTS.validate_manifest(incomplete)


def test_exporter_rejects_permissive_legacy_final_state_aliases(tmp_path: Path) -> None:
    lane = EXPORT_RESULTS.LaneSpec(
        lane_id="lane_00",
        reaction_key="synthetic",
        reaction="A -> B",
        contract="reaction_aware",
        species=("a", "b"),
        terms=(("a", -1.0), ("b", 1.0)),
        seed_policy_sha256="a" * 64,
        campaign_id=TEST_CAMPAIGN_ID,
    )
    candidate = EXPORT_RESULTS.Candidate(
        lane=lane,
        iteration=50,
        policy_sha256="a" * 64,
        policy_path=tmp_path / "simple_pf.py",
        result={},
        total_t_gate_count=1,
        signed_reaction_error_hartree=0.0,
        absolute_reaction_error_hartree=0.0,
        triangle_bound_hartree=0.0,
        equal_share_error_limit_hartree=TARGET_HARTREE / 2.0,
        max_species_absolute_error_hartree=0.0,
        passes_equal_share_constraint=True,
        passes_reaction_constraint=True,
        passes_contract=True,
    )
    manifest_hash = "d" * 64
    manifest = {
        "schema_version": 1,
        "experiment": "reactions10_autoresearch",
        "campaign_id": TEST_CAMPAIGN_ID,
    }
    state = {
        "schema_version": 1,
        "experiment": "reactions10_autoresearch",
        "campaign_id": TEST_CAMPAIGN_ID,
        "manifest_sha256": manifest_hash,
        "current_iteration": 50,
        "next_iteration": None,
        "lanes": {},
    }
    with pytest.raises(
        EXPORT_RESULTS.ExportValidationError,
        match="schema|current_collected_iteration|legacy",
    ):
        EXPORT_RESULTS.validate_state(
            state,
            manifest=manifest,
            manifest_sha256=manifest_hash,
            lanes={lane.lane_id: lane},
            incumbents={lane.lane_id: candidate},
            campaign_root=tmp_path,
        )

    state["current_collected_iteration"] = state.pop("current_iteration")
    state.pop("manifest_sha256")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="schema|manifest_sha256"):
        EXPORT_RESULTS.validate_state(
            state,
            manifest=manifest,
            manifest_sha256=manifest_hash,
            lanes={lane.lane_id: lane},
            incumbents={lane.lane_id: candidate},
            campaign_root=tmp_path,
        )


def test_exporter_rejects_open_collection_transaction(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_hash = "d" * 64
    transactions = campaign_root / "transactions"
    transactions.mkdir(parents=True)
    final_desired_state: dict[str, Any] | None = None
    for iteration in EXPORT_RESULTS.PAPER_EVALUATION_ITERATIONS:
        created_at = f"2026-07-15T00:{iteration:02d}:00+0000"
        decisions = [{"lane_id": f"lane_{index:02d}"} for index in range(20)]
        decision_payload = {
            "schema_version": 1,
            "iteration": iteration,
            "created_at": created_at,
            "decisions": decisions,
        }
        CAMPAIGN_COMMON.write_json(
            campaign_root / "iterations" / f"iter_{iteration:04d}" / "decisions.json",
            decision_payload,
        )
        desired_state = {
            "schema_version": 1,
            "experiment": "reactions10_autoresearch",
            "campaign_id": TEST_CAMPAIGN_ID,
            "manifest_sha256": manifest_hash,
            "current_collected_iteration": iteration,
            "next_iteration": iteration + 1 if iteration < 50 else None,
        }
        desired_state_sha256 = hashlib.sha256(
            json.dumps(desired_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if iteration == EXPORT_RESULTS.PAPER_PROPOSAL_ITERATIONS:
            final_desired_state = desired_state
        CAMPAIGN_COMMON.write_json(
            transactions / f"collect_iter_{iteration:04d}.json",
            {
                "schema_version": 1,
                "phase": "committed",
                "iteration": iteration,
                "manifest_sha256": manifest_hash,
                "created_at": created_at,
                "decisions": decisions,
                "desired_state": desired_state,
                "replayed_at": created_at,
                "committed_at": created_at,
                "desired_state_sha256": desired_state_sha256,
            },
        )

    assert final_desired_state is not None
    CAMPAIGN_COMMON.write_json(campaign_root / "state.json", final_desired_state)

    EXPORT_RESULTS.validate_collection_transactions(
        campaign_root,
        manifest_sha256=manifest_hash,
        campaign_id=TEST_CAMPAIGN_ID,
    )

    open_path = transactions / "collect_iter_0050.json"
    open_transaction = CAMPAIGN_COMMON.read_json(open_path)
    open_transaction["phase"] = "applying"
    CAMPAIGN_COMMON.write_json(open_path, open_transaction)
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="committed|open|phase"):
        EXPORT_RESULTS.validate_collection_transactions(
            campaign_root,
            manifest_sha256=manifest_hash,
            campaign_id=TEST_CAMPAIGN_ID,
        )


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
        with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="tree differs|file set"):
            EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)
        extra.unlink()

        link = snapshot / "undeclared-link.py"
        link.symlink_to(declared.name)
        with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="symlinks"):
            EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)
        link.unlink()

        empty = snapshot / "undeclared-empty-directory"
        empty.mkdir()
        with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="tree differs|director"):
            EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)
        empty.rmdir()

        runtime_named_link = snapshot / ".venv"
        runtime_named_link.symlink_to(snapshot, target_is_directory=True)
        with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="symlink"):
            EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)
        runtime_named_link.unlink()

        pycache = snapshot / "__pycache__"
        pycache.mkdir()
        (pycache / "declared.cpython.pyc").write_bytes(b"untrusted bytecode")
        with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="tree differs|__pycache__"):
            EXPORT_RESULTS.validate_snapshot_tree(snapshot, records, name)
        (pycache / "declared.cpython.pyc").unlink()
        pycache.rmdir()


def test_exporter_requires_real_exact_campaign_and_iteration_roots(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    for name in EXPORT_RESULTS.CAMPAIGN_TOP_FILES:
        (campaign / name).write_text("{}\n", encoding="utf-8")
    for name in EXPORT_RESULTS.CAMPAIGN_TOP_DIRECTORIES:
        (campaign / name).mkdir()
    (campaign / "source" / "repo").mkdir()
    (campaign / "baselines" / "hamiltonian_verification.json").write_text("{}\n", encoding="utf-8")
    EXPORT_RESULTS.validate_campaign_top_level(campaign)

    stray = campaign / "notes.txt"
    stray.write_text("not campaign state\n", encoding="utf-8")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="wrong exact structure"):
        EXPORT_RESULTS.validate_campaign_top_level(campaign)
    stray.unlink()

    campaign_link = tmp_path / "campaign-link"
    campaign_link.symlink_to(campaign.name, target_is_directory=True)
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="real directory"):
        EXPORT_RESULTS.validate_campaign_top_level(campaign_link)

    iterations = campaign / "iterations"
    for iteration in EXPORT_RESULTS.PAPER_EVALUATION_ITERATIONS:
        (iterations / f"iter_{iteration:04d}").mkdir()
    EXPORT_RESULTS.validate_iteration_directories(iterations)
    (iterations / "README.txt").write_text("opposite-type noise\n", encoding="utf-8")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="wrong exact structure"):
        EXPORT_RESULTS.validate_iteration_directories(iterations)


def test_exporter_requires_exact_iteration_lane_entry_types(tmp_path: Path) -> None:
    iteration = 7
    lane_id = "lane_a"
    iteration_root = tmp_path / f"iter_{iteration:04d}"
    for name in EXPORT_RESULTS.ITERATION_DIRECTORIES:
        (iteration_root / name).mkdir(parents=True, exist_ok=True)
    (iteration_root / "decisions.json").write_text("{}\n", encoding="utf-8")
    CAMPAIGN_COMMON.write_json(
        iteration_root / "validation.json",
        {
            "schema_version": 1,
            "iteration": iteration,
            "validated_lane_count": EXPORT_RESULTS.PAPER_LANE_COUNT,
            "retryable_errors": {},
            "fatal_errors": {},
            "validated_at": "2026-07-15T00:00:00+0000",
        },
    )
    candidate_root = iteration_root / "candidates" / lane_id
    candidate_root.mkdir()
    (candidate_root / "simple_pf.py").write_text("# candidate\n", encoding="utf-8")
    (iteration_root / "results" / f"{lane_id}.json").write_text("{}\n", encoding="utf-8")
    (iteration_root / "mutations" / lane_id).mkdir()

    EXPORT_RESULTS.require_exact_directory(
        iteration_root / "results",
        context=f"iteration {iteration} results",
        files={f"{lane_id}.json"},
    )
    opposite_type = iteration_root / "results" / "undeclared-result-directory"
    opposite_type.mkdir()
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="wrong exact structure"):
        EXPORT_RESULTS.require_exact_directory(
            iteration_root / "results",
            context=f"iteration {iteration} results",
            files={f"{lane_id}.json"},
        )
    opposite_type.rmdir()

    EXPORT_RESULTS.require_exact_directory(
        candidate_root,
        context=f"iteration {iteration}/{lane_id} candidate",
        files={"simple_pf.py"},
    )
    candidate_empty = candidate_root / "undeclared-empty-directory"
    candidate_empty.mkdir()
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="wrong exact structure"):
        EXPORT_RESULTS.require_exact_directory(
            candidate_root,
            context=f"iteration {iteration}/{lane_id} candidate",
            files={"simple_pf.py"},
        )
    candidate_empty.rmdir()

    candidate_link = candidate_root / "policy-link.py"
    candidate_link.symlink_to("simple_pf.py")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="symlink"):
        EXPORT_RESULTS.require_exact_directory(
            candidate_root,
            context=f"iteration {iteration}/{lane_id} candidate",
            files={"simple_pf.py"},
        )


def test_exporter_requires_exact_recorded_mutation_work_tree(tmp_path: Path) -> None:
    record, kwargs, _paths = _export_mutation_fixture(
        tmp_path / "mutation",
        with_process=True,
        recovered=False,
    )
    mutation_root = kwargs["mutation_root"]
    work = mutation_root / "work"
    CAMPAIGN_COMMON.write_json(mutation_root / "record.json", record)
    EXPORT_RESULTS.validate_mutation_directory_structure(
        mutation_root,
        iteration=1,
        lane_id=kwargs["lane"].lane_id,
        record=record,
    )

    undeclared_empty = work / "undeclared-empty-directory"
    undeclared_empty.mkdir()
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="director|tree"):
        EXPORT_RESULTS.validate_mutation_directory_structure(
            mutation_root,
            iteration=1,
            lane_id=kwargs["lane"].lane_id,
            record=record,
        )
    undeclared_empty.rmdir()

    nested = work / "scratch" / "diagnostic.txt"
    nested.parent.mkdir()
    nested.write_text("recorded fallback residue\n", encoding="utf-8")
    record["unexpected_work_files"] = ["scratch/diagnostic.txt"]
    record["work_file_records"]["scratch/diagnostic.txt"] = {
        "sha256": _sha256(nested),
        "size_bytes": nested.stat().st_size,
    }
    record["work_directory_paths"] = ["scratch"]
    EXPORT_RESULTS.validate_mutation_directory_structure(
        mutation_root,
        iteration=1,
        lane_id=kwargs["lane"].lane_id,
        record=record,
    )

    (work / "unrecorded.txt").write_text("must fail\n", encoding="utf-8")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="recorded file tree"):
        EXPORT_RESULTS.validate_mutation_directory_structure(
            mutation_root,
            iteration=1,
            lane_id=kwargs["lane"].lane_id,
            record=record,
        )

    (work / "unrecorded.txt").unlink()
    (work / "simple_pf.py").write_text("# tampered after recording\n", encoding="utf-8")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="hash|SHA|size|work policy"):
        EXPORT_RESULTS.validate_mutation_directory_structure(
            mutation_root,
            iteration=1,
            lane_id=kwargs["lane"].lane_id,
            record=record,
        )


def test_exporter_requires_exact_hashed_archived_attempt_evidence(tmp_path: Path) -> None:
    record, kwargs, _paths = _export_mutation_fixture(
        tmp_path / "archive",
        with_process=True,
        recovered=False,
    )
    mutation_root = kwargs["mutation_root"]
    failed_nonce = "f" * 64
    archive = mutation_root / "archived_attempts" / failed_nonce
    failure_path = archive / "failure.json"
    CAMPAIGN_COMMON.write_json(
        failure_path,
        {
            "schema_version": 1,
            "launch_nonce": failed_nonce,
            "reason": "synthetic pre-Codex startup failure",
            "archived_at": "2026-07-15T00:00:00+0000",
        },
    )
    stderr_path = archive / "codex.stderr"
    stderr_path.write_text("npx failed before Codex startup\n", encoding="utf-8")
    record["archived_attempt_evidence"] = [
        {
            "path": f"archived_attempts/{failed_nonce}/codex.stderr",
            "sha256": _sha256(stderr_path),
            "size_bytes": stderr_path.stat().st_size,
        },
        {
            "path": f"archived_attempts/{failed_nonce}/failure.json",
            "sha256": _sha256(failure_path),
            "size_bytes": failure_path.stat().st_size,
        },
    ]
    record["archived_attempt_directory_paths"] = [f"archived_attempts/{failed_nonce}"]
    CAMPAIGN_COMMON.write_json(mutation_root / "record.json", record)
    EXPORT_RESULTS.validate_mutation_directory_structure(
        mutation_root,
        iteration=1,
        lane_id=kwargs["lane"].lane_id,
        record=record,
    )

    empty_archive_directory = archive / "undeclared-empty-directory"
    empty_archive_directory.mkdir()
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="archive|director|tree"):
        EXPORT_RESULTS.validate_mutation_directory_structure(
            mutation_root,
            iteration=1,
            lane_id=kwargs["lane"].lane_id,
            record=record,
        )
    empty_archive_directory.rmdir()

    stderr_path.write_text("tampered archive\n", encoding="utf-8")
    with pytest.raises(EXPORT_RESULTS.ExportValidationError, match="archive|hash|SHA"):
        EXPORT_RESULTS.validate_mutation_directory_structure(
            mutation_root,
            iteration=1,
            lane_id=kwargs["lane"].lane_id,
            record=record,
        )


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
