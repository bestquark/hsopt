#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from campaign_common import (
    CODEX_CLI_VERSION,
    CODEX_COMMAND,
    CODEX_PACKAGE,
    PAPER_MODEL,
    PAPER_REASONING_EFFORT,
    SCHEMA_VERSION,
    campaign_lock,
    candidate_path,
    copy_file_atomic,
    expected_lane_ids,
    ensure_campaign_runner_runtime,
    lane_map,
    mutation_dir,
    now_iso,
    read_json,
    relative_to_campaign,
    require_campaign,
    result_path,
    sha256_file,
    sha256_text,
    validate_iteration,
    validate_policy,
    write_json,
)


def ensure_not_slurm() -> None:
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if job_id:
        raise RuntimeError(f"refusing to launch a mutation agent inside Slurm job {job_id}")


def verify_codex_version() -> str:
    completed = subprocess.run(
        [*CODEX_COMMAND, "--version"],
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    output = completed.stdout.strip()
    versions = re.findall(r"(?<![0-9.])([0-9]+\.[0-9]+\.[0-9]+)(?![0-9.])", output)
    if versions != [CODEX_CLI_VERSION]:
        raise RuntimeError(f"pinned {CODEX_PACKAGE} reported unexpected version output {output!r}")
    return output


def compact_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    compact_species = {
        key: {
            "signed_energy_error_mha": value.get("signed_energy_error_mha"),
            "absolute_energy_error_mha": value.get("absolute_energy_error_mha"),
            "t_gate_count": value.get("t_gate_count"),
            "method_config": value.get("method_config"),
            "synthesis_epsilon_histogram": value.get("synthesis_epsilon_histogram"),
        }
        for key, value in result.get("species", {}).items()
        if isinstance(value, dict)
    }
    fields = (
        "iteration",
        "accepted",
        "decision_reason",
        "passes_contract",
        "passes_equal_share_constraint",
        "passes_reaction_constraint",
        "absolute_reaction_error_mha",
        "triangle_bound_mha",
        "equal_share_error_limit_mha",
        "max_species_absolute_error_mha",
        "total_t_gate_count",
        "score",
    )
    return {
        **{field: result.get(field) for field in fields if field in result},
        "species": compact_species,
    }


def mutation_prompt(
    lane: dict[str, Any],
    iteration: int,
    incumbent: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> str:
    if lane["contract"] == "equal_share":
        contract = (
            "Every unique species must satisfy |e_i| <= (1 kcal/mol) / sum_j |nu_j|. "
            "Reaction-error cancellation cannot make this control lane pass."
        )
    else:
        contract = (
            "The signed stoichiometric error |sum_i nu_i e_i| must stay at or below "
            "1 kcal/mol. Individual errors and their triangle sum may be larger when signed "
            "errors cancel."
        )
    return f"""You are making proposal {iteration} of exactly 50 sequential HSOpt proposals for one reaction-energy lane.

Lane:
- lane_id: {lane["lane_id"]}
- reaction: {lane["reaction"]}
- contract: {lane["contract"]}
- stoichiometric terms: {json.dumps(lane["terms"], sort_keys=True)}
- species: {", ".join(lane["species"])}
- exact target: {lane["reaction_target_mha"]:.15g} mHa (1 kcal/mol)

Edit only `simple_pf.py` in this disposable directory. Make one coherent,
deterministic policy mutation. Do not create, rename, or delete any other file.

The fixed evaluator sets the active species before calling get_method_config() and
choose_synthesis_epsilon(). It builds the fixed product-formula circuit, compiles
single-qubit rotations with PyGridSynth, counts exact T characters, constructs the
compiled one-step unitary in the full Hilbert space, diagonalizes it, and extracts
each signed molecular energy error. Both contracts use the same evaluator and
T-count definition.

Contract:
{contract}

Objective and allowed policy changes:
- Minimize total_t_gate_count, summed over each unique species circuit once.
- A proposal is retained only if it passes this lane's contract and strictly lowers T count.
- Use species-specific budgets and Hamiltonian-aware precision logic based on label,
  coefficient, support_size, x_count, y_count, z_count, and abs(alpha).
- You may change PF1/PF2, n_steps, zero_angle_threshold, or deterministic epsilon logic.
- Keep choose_synthesis_epsilon independent of step_index.
- Return a positive finite epsilon for every nonzero rotation.

Hard constraints:
- Do not change molecules, geometries, basis sets, active spaces, target time,
  chemical accuracy, compiler behavior, evaluator code, or files outside simple_pf.py.
- Do not run the evaluator, inspect cluster files, install packages, access the network,
  or search outside this disposable directory. Use only the diagnostics below.
- Keep the policy auditable; do not add random search or large lookup tables.

Accepted incumbent result:
```json
{json.dumps(compact_result(incumbent), indent=2, sort_keys=True)}
```

Immediately previous attempted result:
```json
{json.dumps(compact_result(previous), indent=2, sort_keys=True)}
```

After editing, briefly state the single policy idea you changed.
"""


def load_optional_result(root: Path, path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = root / path_text
    if not path.is_file():
        raise FileNotFoundError(f"state points to missing result: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"result is not an object: {path}")
    return payload


def mutation_record_is_complete(
    *, root: Path, lane: dict[str, Any], iteration: int, parent_hash: str
) -> bool:
    directory = mutation_dir(root, iteration, str(lane["lane_id"]))
    record_path = directory / "record.json"
    if not record_path.is_file():
        return False
    record = read_json(record_path)
    candidate = candidate_path(root, iteration, str(lane["lane_id"]))
    prompt_path = directory / "prompt.md"
    intent_path = directory / "intent.json"
    launch_intent_path = directory / "launch_intent.json"
    launch_receipt_path = directory / "launch.json"
    process_path = directory / "process.json"
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("lane_id") != lane["lane_id"]
        or record.get("iteration") != iteration
        or record.get("status") != "candidate_ready"
        or record.get("parent_policy_sha256") != parent_hash
        or not candidate.is_file()
        or record.get("candidate_policy_sha256") != sha256_file(candidate)
        or not prompt_path.is_file()
        or not intent_path.is_file()
        or not launch_intent_path.is_file()
        or not launch_receipt_path.is_file()
        or record.get("prompt_sha256") != sha256_text(prompt_path.read_text(encoding="utf-8"))
        or read_json(intent_path).get("prompt_sha256") != record.get("prompt_sha256")
        or record.get("intent_sha256") != sha256_file(intent_path)
        or read_json(launch_intent_path).get("command_sha256")
        != record.get("launch_command_sha256")
        or record.get("launch_intent_sha256") != sha256_file(launch_intent_path)
        or read_json(launch_receipt_path).get("command_sha256")
        != record.get("launch_command_sha256")
        or read_json(launch_receipt_path).get("status") != "launched"
        or record.get("launch_receipt_sha256") != sha256_file(launch_receipt_path)
        or record.get("process_sha256")
        != (sha256_file(process_path) if process_path.is_file() else None)
        or (
            process_path.is_file()
            and read_json(process_path).get("command_sha256") != record.get("launch_command_sha256")
        )
    ):
        raise RuntimeError(
            f"immutable mutation record drift for {lane['lane_id']} iteration {iteration}"
        )
    candidate_files = [path.name for path in candidate.parent.iterdir() if path.is_file()]
    if candidate_files != ["simple_pf.py"]:
        raise RuntimeError(f"candidate directory is not isolated for {lane['lane_id']}")
    return True


def _process_group_commands(pgid: int) -> list[str]:
    completed = subprocess.run(
        ["ps", "-axo", "pgid=,command="],
        check=True,
        text=True,
        capture_output=True,
    )
    commands: list[str] = []
    for line in completed.stdout.splitlines():
        stripped = line.lstrip()
        fields = stripped.split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pgid:
            commands.append(fields[1])
    return commands


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError(f"cannot inspect mutation process group {pgid}") from exc
    return True


def terminate_recorded_process_group(receipt: dict[str, Any], work_dir: Path) -> None:
    pid = int(receipt.get("pid", -1))
    pgid = int(receipt.get("pgid", -1))
    if pid <= 1 or pgid != pid:
        raise RuntimeError(f"unsafe mutation launch receipt pid={pid} pgid={pgid}")
    if receipt.get("work_dir") != str(work_dir.resolve()):
        raise RuntimeError("mutation launch receipt points at a different work directory")
    if not _process_group_alive(pgid):
        return
    commands = _process_group_commands(pgid)
    if not commands or not any(str(work_dir.resolve()) in command for command in commands):
        raise RuntimeError(
            f"refusing to signal possible PID-reuse process group {pgid}: {commands}"
        )
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_group_alive(pgid):
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 10.0
        while _process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
    if _process_group_alive(pgid):
        raise RuntimeError(f"mutation process group {pgid} did not terminate")


def _find_unreceipted_process_groups(work_dir: Path) -> list[int]:
    completed = subprocess.run(
        ["ps", "-axo", "pgid=,command="],
        check=True,
        text=True,
        capture_output=True,
    )
    result: set[int] = set()
    needle = str(work_dir.resolve())
    for line in completed.stdout.splitlines():
        stripped = line.lstrip()
        fields = stripped.split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit() and needle in fields[1]:
            result.add(int(fields[0]))
    return sorted(result)


def _validate_no_intent_residue(directory: Path, *, parent_hash: str, prompt_hash: str) -> None:
    if not directory.exists() or not any(directory.iterdir()):
        return
    allowed = {"prompt.md", "parent_simple_pf.py", "work"}
    unexpected = sorted(path.name for path in directory.iterdir() if path.name not in allowed)
    prompt = directory / "prompt.md"
    parent = directory / "parent_simple_pf.py"
    work = directory / "work"
    if unexpected:
        raise RuntimeError(f"uncommitted mutation residue contains unexpected paths: {unexpected}")
    if prompt.exists() and sha256_text(prompt.read_text(encoding="utf-8")) != prompt_hash:
        raise RuntimeError("uncommitted prompt residue has unexpected bytes")
    if parent.exists() and sha256_file(parent) != parent_hash:
        raise RuntimeError("uncommitted parent residue has unexpected bytes")
    if work.exists():
        files = sorted(
            path.relative_to(work).as_posix() for path in work.rglob("*") if path.is_file()
        )
        if files not in ([], ["simple_pf.py"]):
            raise RuntimeError(f"uncommitted work residue contains unexpected files: {files}")
        if files and sha256_file(work / "simple_pf.py") != parent_hash:
            raise RuntimeError("uncommitted work residue is not the unchanged parent")


def ensure_mutation_workspace(
    *,
    root: Path,
    directory: Path,
    intent: dict[str, Any],
    prompt_text: str,
) -> Path:
    parent_source = root / str(intent["parent_policy_path"])
    if not parent_source.is_file() or sha256_file(parent_source) != intent["parent_policy_sha256"]:
        raise RuntimeError("intent parent source path/hash drift")
    prompt_path = directory / "prompt.md"
    if prompt_path.exists():
        if sha256_text(prompt_path.read_text(encoding="utf-8")) != intent["prompt_sha256"]:
            raise RuntimeError("intent prompt hash drift")
    else:
        prompt_path.write_text(prompt_text, encoding="utf-8")
    parent_snapshot = directory / "parent_simple_pf.py"
    if parent_snapshot.exists():
        if sha256_file(parent_snapshot) != intent["parent_policy_sha256"]:
            raise RuntimeError("intent parent snapshot hash drift")
    else:
        shutil.copy2(parent_source, parent_snapshot)
    work_dir = directory / "work"
    if not work_dir.exists():
        work_dir.mkdir(parents=True)
        shutil.copy2(parent_snapshot, work_dir / "simple_pf.py")
    elif not (work_dir / "simple_pf.py").exists():
        if any(work_dir.iterdir()):
            raise RuntimeError("mutation work directory lacks simple_pf.py but is not empty")
        shutil.copy2(parent_snapshot, work_dir / "simple_pf.py")
    return work_dir


def recover_interrupted_launch(
    directory: Path, work_dir: Path
) -> tuple[bool, dict[str, Any] | None]:
    launch_intent = directory / "launch_intent.json"
    receipt_path = directory / "launch.json"
    if not launch_intent.exists():
        return False, None
    deadline = time.monotonic() + 5.0
    while not receipt_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        expected_command_hash = read_json(launch_intent).get("command_sha256")
        if receipt.get("command_sha256") != expected_command_hash:
            raise RuntimeError("mutation launch command hash mismatch")
        if receipt.get("status") == "exec_error":
            work_policy = work_dir / "simple_pf.py"
            parent = directory / "parent_simple_pf.py"
            if not work_policy.is_file() or work_policy.read_bytes() != parent.read_bytes():
                raise RuntimeError("exec-failed launch unexpectedly changed the work policy")
            suffix = sha256_file(receipt_path)[:12]
            receipt_path.replace(directory / f"launch_exec_error_{suffix}.json")
            process_path = directory / "process.json"
            if process_path.is_file():
                process_path.replace(directory / f"process_exec_error_{suffix}.json")
            return False, None
        if receipt.get("status") != "launched":
            raise RuntimeError(f"invalid durable launch receipt status {receipt.get('status')!r}")
        terminate_recorded_process_group(receipt, work_dir)
        launched = True
    else:
        unreceipted = _find_unreceipted_process_groups(work_dir)
        if unreceipted:
            raise RuntimeError(
                f"found unreceipted mutation process groups for {work_dir}: {unreceipted}"
            )
        work_policy = work_dir / "simple_pf.py"
        parent = directory / "parent_simple_pf.py"
        if not work_policy.is_file() or work_policy.read_bytes() != parent.read_bytes():
            raise RuntimeError("mutation work changed without a durable launch receipt")
        launched = False
    process_path = directory / "process.json"
    process_record = read_json(process_path) if process_path.is_file() else None
    return launched, process_record


def finalize_mutation(
    *,
    root: Path,
    lane: dict[str, Any],
    iteration: int,
    codex_version_output: str,
    process_record: dict[str, Any] | None,
    recovered: bool,
) -> dict[str, Any]:
    lane_id = str(lane["lane_id"])
    directory = mutation_dir(root, iteration, lane_id)
    intent = read_json(directory / "intent.json")
    prompt_path = directory / "prompt.md"
    if not prompt_path.is_file() or sha256_text(
        prompt_path.read_text(encoding="utf-8")
    ) != intent.get("prompt_sha256"):
        raise RuntimeError(f"prompt snapshot hash drift for {lane_id}")
    parent = directory / "parent_simple_pf.py"
    if not parent.is_file() or sha256_file(parent) != intent.get("parent_policy_sha256"):
        raise RuntimeError(f"parent policy snapshot hash drift for {lane_id}")
    launch_intent_path = directory / "launch_intent.json"
    launch_receipt_path = directory / "launch.json"
    if not launch_intent_path.is_file() or not launch_receipt_path.is_file():
        raise RuntimeError(f"no durable Codex launch evidence for {lane_id}")
    launch_intent = read_json(launch_intent_path)
    launch_receipt = read_json(launch_receipt_path)
    if launch_intent.get("command_sha256") != launch_receipt.get("command_sha256"):
        raise RuntimeError(f"Codex launch command receipt mismatch for {lane_id}")
    if launch_receipt.get("status") != "launched":
        raise RuntimeError(f"Codex did not actually launch for {lane_id}")
    work_dir = directory / "work"
    work_policy = work_dir / "simple_pf.py"
    if not work_policy.is_file():
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent, work_policy)
    unexpected_files = sorted(
        path.relative_to(work_dir).as_posix()
        for path in work_dir.rglob("*")
        if path.is_file() and path.relative_to(work_dir).as_posix() != "simple_pf.py"
    )
    work_changed = work_policy.read_bytes() != parent.read_bytes()
    work_valid, validation_error = validate_policy(work_policy, lane["species"])
    returncode = None if process_record is None else process_record.get("returncode")
    timed_out = bool(process_record and process_record.get("timed_out"))
    use_mutation = (
        process_record is not None
        and returncode == 0
        and not timed_out
        and work_changed
        and work_valid
        and not unexpected_files
    )
    candidate_source = work_policy if use_mutation else parent
    outcome = "mutated" if use_mutation else "fallback_parent"
    candidate = candidate_path(root, iteration, lane_id)
    if candidate.parent.exists() and any(candidate.parent.iterdir()):
        if not candidate.is_file() or sha256_file(candidate) != sha256_file(candidate_source):
            raise RuntimeError(
                f"candidate snapshot already exists with different bytes for {lane_id}"
            )
    else:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        copy_file_atomic(candidate_source, candidate)
    candidate_files = sorted(
        path.relative_to(candidate.parent).as_posix()
        for path in candidate.parent.rglob("*")
        if path.is_file()
    )
    if candidate_files != ["simple_pf.py"]:
        raise RuntimeError(
            f"candidate directory contains unexpected files for {lane_id}: {candidate_files}"
        )
    candidate_hash = sha256_file(candidate)
    record = {
        "schema_version": SCHEMA_VERSION,
        "lane_id": lane_id,
        "iteration": iteration,
        "kind": "autoresearch_proposal",
        "status": "candidate_ready",
        "mutation_outcome": outcome,
        "parent_iteration": intent["parent_iteration"],
        "parent_policy_sha256": intent["parent_policy_sha256"],
        "prompt_sha256": intent["prompt_sha256"],
        "intent_sha256": sha256_file(directory / "intent.json"),
        "launch_command_sha256": (launch_intent.get("command_sha256")),
        "launch_intent_sha256": sha256_file(launch_intent_path),
        "launch_receipt_sha256": (sha256_file(launch_receipt_path)),
        "process_sha256": (
            sha256_file(directory / "process.json")
            if (directory / "process.json").is_file()
            else None
        ),
        "candidate_policy_sha256": candidate_hash,
        "codex_package": CODEX_PACKAGE,
        "codex_cli_version": CODEX_CLI_VERSION,
        "codex_version_output": codex_version_output,
        "model": PAPER_MODEL,
        "reasoning_effort": PAPER_REASONING_EFFORT,
        "sandbox": "workspace-write",
        "ephemeral": True,
        "ignored_user_config": True,
        "ignored_rules": True,
        "returncode": returncode,
        "timed_out": timed_out,
        "changed": work_changed,
        "valid": work_valid,
        "validation_error": validation_error,
        "unexpected_work_files": unexpected_files,
        "recovered_after_interruption": recovered,
        "created_at": intent["created_at"],
        "completed_at": now_iso(),
    }
    write_json(directory / "record.json", record)
    return record


async def mutate_one(
    *,
    root: Path,
    lane: dict[str, Any],
    lane_state: dict[str, Any],
    iteration: int,
    codex_version_output: str,
    timeout_seconds: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        lane_id = str(lane["lane_id"])
        parent_path_text = lane_state.get("incumbent_policy_path")
        parent_hash = lane_state.get("incumbent_policy_sha256")
        parent_iteration = lane_state.get("incumbent_iteration")
        if not parent_path_text or not parent_hash or parent_iteration is None:
            raise RuntimeError(f"{lane_id}: no accepted baseline/incumbent is available")
        parent_path = root / str(parent_path_text)
        if not parent_path.is_file() or sha256_file(parent_path) != parent_hash:
            raise RuntimeError(f"{lane_id}: incumbent policy path/hash drift")
        if mutation_record_is_complete(
            root=root, lane=lane, iteration=iteration, parent_hash=str(parent_hash)
        ):
            return {"lane_id": lane_id, "status": "already_prepared"}

        directory = mutation_dir(root, iteration, lane_id)
        directory.mkdir(parents=True, exist_ok=True)
        incumbent = load_optional_result(root, lane_state.get("incumbent_result_path"))
        previous_path = result_path(root, iteration - 1, lane_id)
        previous = read_json(previous_path) if previous_path.is_file() else None
        prompt_text = mutation_prompt(lane, iteration, incumbent, previous)
        intent_path = directory / "intent.json"
        had_intent = intent_path.is_file()
        if had_intent:
            intent = read_json(intent_path)
            if (
                intent.get("parent_policy_sha256") != parent_hash
                or intent.get("parent_iteration") != parent_iteration
                or intent.get("prompt_sha256") != sha256_text(prompt_text)
            ):
                raise RuntimeError(f"{lane_id}: interrupted mutation intent has a different parent")
        else:
            _validate_no_intent_residue(
                directory,
                parent_hash=str(parent_hash),
                prompt_hash=sha256_text(prompt_text),
            )
            intent = {
                "schema_version": SCHEMA_VERSION,
                "lane_id": lane_id,
                "iteration": iteration,
                "parent_iteration": parent_iteration,
                "parent_policy_path": relative_to_campaign(root, parent_path),
                "parent_policy_sha256": parent_hash,
                "prompt_sha256": sha256_text(prompt_text),
                "codex_package": CODEX_PACKAGE,
                "codex_cli_version": CODEX_CLI_VERSION,
                "model": PAPER_MODEL,
                "reasoning_effort": PAPER_REASONING_EFFORT,
                "created_at": now_iso(),
            }
            # The intent is durable before any new mutable residue is created.
            write_json(intent_path, intent)
        if had_intent:
            # Terminate a durably identified orphan before reading or repairing
            # any mutable work-tree contents.
            launched, process_record = recover_interrupted_launch(directory, directory / "work")
        else:
            launched, process_record = False, None
        work_dir = ensure_mutation_workspace(
            root=root,
            directory=directory,
            intent=intent,
            prompt_text=prompt_text,
        )
        if had_intent:
            if launched:
                record = finalize_mutation(
                    root=root,
                    lane=lane,
                    iteration=iteration,
                    codex_version_output=codex_version_output,
                    process_record=process_record,
                    recovered=True,
                )
                return {
                    "lane_id": lane_id,
                    "status": record["mutation_outcome"],
                    "recovered": True,
                }

        stdout_path = directory / "codex.stdout"
        stderr_path = directory / "codex.stderr"
        last_message_path = directory / "last_message.txt"
        command = [
            *CODEX_COMMAND,
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
            PAPER_MODEL,
            "-c",
            f'model_reasoning_effort="{PAPER_REASONING_EFFORT}"',
            "-o",
            str(last_message_path),
            prompt_text,
        ]
        started_at = now_iso()
        process_error = None
        timed_out = False
        returncode = 125
        process: asyncio.subprocess.Process | None = None
        cancellation: asyncio.CancelledError | None = None
        launch_receipt_path = directory / "launch.json"
        launch_intent_path = directory / "launch_intent.json"
        launch_intent_payload = {
            "schema_version": SCHEMA_VERSION,
            "lane_id": lane_id,
            "iteration": iteration,
            "command_sha256": sha256_text("\0".join(command)),
            "requested_at": now_iso(),
        }
        if launch_intent_path.is_file():
            existing_launch_intent = read_json(launch_intent_path)
            if (
                existing_launch_intent.get("schema_version") != SCHEMA_VERSION
                or existing_launch_intent.get("lane_id") != lane_id
                or existing_launch_intent.get("iteration") != iteration
                or existing_launch_intent.get("command_sha256")
                != launch_intent_payload["command_sha256"]
            ):
                raise RuntimeError("persisted launch intent drift")
        else:
            write_json(launch_intent_path, launch_intent_payload)
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(Path(__file__).resolve().parent / "launch_codex.py"),
                    "--receipt",
                    str(launch_receipt_path),
                    "--work-dir",
                    str(work_dir),
                    "--",
                    *command,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                receipt_deadline = asyncio.get_running_loop().time() + 5.0
                while (
                    not launch_receipt_path.exists()
                    and asyncio.get_running_loop().time() < receipt_deadline
                ):
                    await asyncio.sleep(0.02)
                if not launch_receipt_path.exists():
                    raise RuntimeError("durable Codex launch receipt was not created")
                launch_receipt = read_json(launch_receipt_path)
                if (
                    int(launch_receipt.get("pid", -1)) != process.pid
                    or int(launch_receipt.get("pgid", -1)) != process.pid
                    or launch_receipt.get("work_dir") != str(work_dir.resolve())
                ):
                    raise RuntimeError("Codex launch receipt identity mismatch")
                try:
                    returncode = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        await process.wait()
                    returncode = 124
        except asyncio.CancelledError as exc:
            cancellation = exc
            process_error = "CancelledError: operator cancellation"
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(asyncio.shield(process.wait()), timeout=10)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await asyncio.shield(process.wait())
            returncode = 130
        except Exception as exc:
            process_error = f"{type(exc).__name__}: {exc}"
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
        process_record = {
            "command": command[:-1] + ["<prompt recorded in prompt.md>"],
            "command_sha256": sha256_text("\0".join(command)),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "process_error": process_error,
            "pid": None if process is None else process.pid,
            "launch_receipt_sha256": sha256_file(launch_receipt_path)
            if launch_receipt_path.is_file()
            else None,
            "started_at": started_at,
            "completed_at": now_iso(),
        }
        write_json(directory / "process.json", process_record)
        if not launch_receipt_path.is_file():
            if cancellation is not None:
                raise cancellation
            raise RuntimeError(
                f"{lane_id}: Codex was not durably launched; rerun this same iteration"
            )
        launch_receipt = read_json(launch_receipt_path)
        if launch_receipt.get("status") == "exec_error":
            suffix = sha256_file(launch_receipt_path)[:12]
            launch_receipt_path.replace(directory / f"launch_exec_error_{suffix}.json")
            (directory / "process.json").replace(directory / f"process_exec_error_{suffix}.json")
            raise RuntimeError(
                f"{lane_id}: Codex exec failed before launch; rerun this same iteration"
            )
        if launch_receipt.get("status") != "launched":
            raise RuntimeError(f"{lane_id}: invalid launch receipt status")
        record = finalize_mutation(
            root=root,
            lane=lane,
            iteration=iteration,
            codex_version_output=codex_version_output,
            process_record=process_record,
            recovered=False,
        )
        if cancellation is not None:
            raise cancellation
        return {
            "lane_id": lane_id,
            "status": record["mutation_outcome"],
            "returncode": returncode,
            "changed": record["changed"],
            "valid": record["valid"],
        }


async def await_mutation_tasks(
    tasks: list[asyncio.Future | asyncio.Task | Any], lane_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Await every lane even when one raises; never cancel healthy sibling mutations."""
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for lane_id, outcome in zip(lane_ids, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            failures.append(
                {
                    "lane_id": lane_id,
                    "error": f"{type(outcome).__name__}: {outcome}",
                }
            )
        else:
            results.append(outcome)
    return results, failures


async def run(args: argparse.Namespace) -> int:
    ensure_not_slurm()
    validate_iteration(args.iteration, mutation=True)
    root = args.campaign_root.resolve()
    manifest, state = require_campaign(root)
    if int(state["next_iteration"]) != args.iteration:
        raise RuntimeError(
            f"iteration sequence violation: state expects {state['next_iteration']}, got {args.iteration}"
        )
    if int(state["current_collected_iteration"]) != args.iteration - 1:
        raise RuntimeError("previous iteration has not been fully collected")
    codex_version_output = verify_codex_version()
    lanes = lane_map(manifest)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    lane_ids = expected_lane_ids(manifest)
    tasks = [
        mutate_one(
            root=root,
            lane=lanes[lane_id],
            lane_state=state["lanes"][lane_id],
            iteration=args.iteration,
            codex_version_output=codex_version_output,
            timeout_seconds=args.timeout_seconds,
            semaphore=semaphore,
        )
        for lane_id in lane_ids
    ]
    results, failures = await await_mutation_tasks(tasks, lane_ids)
    fallback_count = 0
    for result in results:
        fallback_count += int(result["status"] == "fallback_parent")
        print(json.dumps(result, sort_keys=True), flush=True)
    for failure in failures:
        print(json.dumps({"status": "fatal_error", **failure}, sort_keys=True), flush=True)
    return 2 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create immutable candidates for one reaction autoresearch iteration."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.max_concurrent <= 0:
        parser.error("--max-concurrent must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    root = args.campaign_root.resolve()
    with campaign_lock(root, "orchestration"):
        manifest, _state = require_campaign(root)
        ensure_campaign_runner_runtime(root, Path(__file__), manifest)
        raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
