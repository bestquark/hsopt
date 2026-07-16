#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
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


EXEC_ACK_PROTOCOL = "codex_jsonl_lifecycle_v1"
CODEX_START_EVENT = "thread.started"
CODEX_TERMINAL_EVENT = "turn.completed"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_event_sha256(event: dict[str, Any]) -> str:
    return sha256_text(json.dumps(event, sort_keys=True, separators=(",", ":")))


def nonempty_text_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required text evidence is missing or unsafe: {path}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"required text evidence is not UTF-8: {path}") from exc
    if not text.strip():
        raise RuntimeError(f"required text evidence is empty: {path}")
    return sha256_file(path)


def validate_exec_acknowledgement(
    receipt: dict[str, Any],
    *,
    command_sha256: str,
    work_dir: Path,
    launcher_pid: int | None = None,
    launch_nonce: str | None = None,
) -> None:
    pid = int(receipt.get("pid", -1))
    pgid = int(receipt.get("pgid", -1))
    exec_pid = int(receipt.get("exec_pid", -1))
    try:
        started_at = float(receipt.get("started_at_epoch"))
        exec_confirmed_at = float(receipt.get("exec_confirmed_at_epoch"))
        codex_started_at = float(receipt.get("codex_started_at_epoch"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid Codex exec acknowledgement timestamps") from exc
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "codex_started"
        or receipt.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or receipt.get("low_level_exec_status") != "exec_confirmed"
        or receipt.get("command_sha256") != command_sha256
        or receipt.get("work_dir") != str(work_dir.resolve())
        or pid <= 1
        or pgid != pid
        or exec_pid <= 1
        or exec_pid == pid
        or (launcher_pid is not None and pid != launcher_pid)
        or SHA256_RE.fullmatch(str(receipt.get("launch_nonce", ""))) is None
        or (launch_nonce is not None and receipt.get("launch_nonce") != launch_nonce)
        or not isinstance(receipt.get("codex_thread_id"), str)
        or not receipt.get("codex_thread_id")
        or len(str(receipt.get("codex_thread_id"))) > 256
        or SHA256_RE.fullmatch(str(receipt.get("codex_start_event_sha256", ""))) is None
        or not all(map(math.isfinite, (started_at, exec_confirmed_at, codex_started_at)))
        or exec_confirmed_at < started_at
        or codex_started_at < exec_confirmed_at
    ):
        raise RuntimeError("invalid Codex exec acknowledgement")


def validate_process_crosslink(
    process_record: dict[str, Any],
    *,
    launch_receipt: dict[str, Any],
    launch_receipt_sha256: str,
    command_sha256: str,
) -> None:
    usage = process_record.get("codex_usage")
    usage_hash = (
        sha256_text(json.dumps(usage, sort_keys=True, separators=(",", ":")))
        if isinstance(usage, dict)
        else None
    )
    if (
        process_record.get("command_sha256") != command_sha256
        or process_record.get("launch_receipt_sha256") != launch_receipt_sha256
        or process_record.get("pid") != launch_receipt.get("pid")
        or process_record.get("pgid") != launch_receipt.get("pgid")
        or process_record.get("exec_pid") != launch_receipt.get("exec_pid")
        or process_record.get("exec_acknowledgement_status") != "codex_started"
        or process_record.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or process_record.get("launch_nonce") != launch_receipt.get("launch_nonce")
        or process_record.get("codex_thread_id") != launch_receipt.get("codex_thread_id")
        or process_record.get("codex_start_event_sha256")
        != launch_receipt.get("codex_start_event_sha256")
        or process_record.get("codex_lifecycle_status") != "turn_completed"
        or SHA256_RE.fullmatch(str(process_record.get("codex_terminal_event_sha256", ""))) is None
        or process_record.get("codex_usage_sha256") != usage_hash
        or not usage
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in usage.values()
        )
        or any(
            SHA256_RE.fullmatch(str(process_record.get(field, ""))) is None
            for field in (
                "codex_stdout_sha256",
                "codex_stderr_sha256",
                "last_message_sha256",
            )
        )
        or process_record.get("returncode") != 0
        or process_record.get("timed_out") is not False
        or not isinstance(process_record.get("last_message_sha256"), str)
    ):
        raise RuntimeError("Codex process receipt does not cross-link its exec acknowledgement")


def validate_codex_lifecycle(stdout_path: Path) -> dict[str, Any]:
    """Require one real Codex thread start and one successful terminal turn event."""

    if not stdout_path.is_file() or stdout_path.is_symlink():
        raise RuntimeError("Codex JSONL stdout is missing or unsafe")
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(stdout_path.read_bytes().splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Codex JSONL line {line_number} is invalid") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise RuntimeError(f"Codex JSONL line {line_number} is not an event object")
        events.append(event)
    starts = [
        (index, event) for index, event in enumerate(events) if event["type"] == CODEX_START_EVENT
    ]
    terminals = [
        (index, event)
        for index, event in enumerate(events)
        if event["type"] == CODEX_TERMINAL_EVENT
    ]
    failures = [event for event in events if event["type"] in {"turn.failed", "error"}]
    if (
        len(starts) != 1
        or len(terminals) != 1
        or failures
        or starts[0][0] >= terminals[0][0]
        or terminals[0][0] != len(events) - 1
    ):
        raise RuntimeError(
            "Codex JSONL does not prove one successful start-to-completion lifecycle"
        )
    start = starts[0][1]
    terminal = terminals[0][1]
    thread_id = start.get("thread_id")
    usage = terminal.get("usage")
    if (
        not isinstance(thread_id, str)
        or not thread_id
        or len(thread_id) > 256
        or not isinstance(usage, dict)
        or not usage
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in usage.values()
        )
    ):
        raise RuntimeError("Codex JSONL lifecycle identity or usage is malformed")
    return {
        "codex_lifecycle_status": "turn_completed",
        "codex_thread_id": thread_id,
        "codex_start_event_sha256": canonical_event_sha256(start),
        "codex_terminal_event_sha256": canonical_event_sha256(terminal),
        "codex_usage": usage,
        "codex_usage_sha256": sha256_text(json.dumps(usage, sort_keys=True, separators=(",", ":"))),
    }


def exact_regular_file_records(root: Path) -> dict[str, dict[str, Any]]:
    """Hash an exact evidence tree and reject links or non-regular entries."""

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"evidence root is missing, non-directory, or a symlink: {root}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"evidence tree contains a symlink: {relative}")
        if path.is_dir():
            continue
        mode = path.stat(follow_symlinks=False).st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"evidence tree contains a special entry: {relative}")
        records[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return records


def exact_directory_paths(root: Path) -> list[str]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"evidence root is missing, non-directory, or a symlink: {root}")
    directories: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"evidence tree contains a symlink: {relative}")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            directories.append(relative)
        elif not stat.S_ISREG(mode):
            raise RuntimeError(f"evidence tree contains a special entry: {relative}")
    return directories


def archived_attempt_evidence(directory: Path) -> list[dict[str, Any]]:
    archive_root = directory / "archived_attempts"
    if not archive_root.exists():
        return []
    records = exact_regular_file_records(archive_root)
    return [
        {"path": f"archived_attempts/{path}", **record} for path, record in sorted(records.items())
    ]


def archived_attempt_directory_paths(directory: Path) -> list[str]:
    archive_root = directory / "archived_attempts"
    if not archive_root.exists():
        return []
    directories = exact_directory_paths(archive_root)
    if not directories:
        raise RuntimeError("archived_attempts exists without a nonce evidence directory")
    return [f"archived_attempts/{path}" for path in directories]


def archive_failed_attempt(directory: Path, *, launch_nonce: str, reason: str) -> None:
    if SHA256_RE.fullmatch(launch_nonce) is None:
        raise RuntimeError("cannot archive an attempt with an invalid launch nonce")
    archive = directory / "archived_attempts" / launch_nonce
    archive.mkdir(parents=True, exist_ok=True)
    failure_path = archive / "failure.json"
    if failure_path.is_file():
        failure = read_json(failure_path)
        if failure.get("launch_nonce") != launch_nonce:
            raise RuntimeError("partial failed-attempt archive has a different nonce")
    else:
        write_json(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "launch_nonce": launch_nonce,
                "reason": reason,
                "archived_at": now_iso(),
            },
        )
    # Move the mutable work tree first and the durable launch intent last.  If
    # interrupted, the still-present intent causes recovery to resume this
    # idempotent archive instead of reusing partial failed work.
    work = directory / "work"
    archived_work = archive / "work"
    if work.exists():
        if archived_work.exists():
            raise RuntimeError("partial failed-attempt archive contains two work trees")
        if work.is_symlink() or not work.is_dir():
            raise RuntimeError("failed-attempt work evidence is not a real directory")
        exact_regular_file_records(work)
        exact_directory_paths(work)
        work.replace(archived_work)
    for name in (
        "launch.json",
        "process.json",
        "codex.stdout",
        "codex.stderr",
        "last_message.txt",
        "launch_intent.json",
    ):
        source = directory / name
        if source.exists():
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"failed-attempt evidence is unsafe: {source}")
            destination = archive / name
            if destination.exists():
                raise RuntimeError(f"partial failed-attempt archive duplicates {name}")
            source.replace(destination)
    exact_regular_file_records(archive)
    exact_directory_paths(archive)


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

The fixed evaluator sets the active species before calling the policy hooks,
compiles every selected single-qubit rotation with PyGridSynth, counts exact T
characters, and extracts each signed molecular energy error. Stationary Trotter
uses the fast one-step eigenspectrum path. randomized_trotter, qdrift, and custom
use an evaluator-seeded full explicit schedule on the GPU. Both contracts use the
same evaluator and T-count definition.

Contract:
{contract}

Objective and allowed policy changes:
- Minimize total_t_gate_count, summed over each unique species circuit once.
- A proposal is retained only if it passes this lane's contract and strictly lowers T count.
- Use species-specific budgets and Hamiltonian-aware precision logic based on label,
  coefficient, support_size, x_count, y_count, z_count, and abs(alpha).
- You may change PF1/PF2, n_steps, zero_angle_threshold, ordering, schedule, or
  deterministic epsilon logic.
- term_order_strategy="custom" may return any exact permutation through
  order_product_formula_terms(). schedule_strategy may be stationary_trotter,
  randomized_trotter, qdrift, or custom. A custom schedule returns a sequence of
  {{"term_index": int, "alpha": float, optional "epsilon": float}} items.
- For explicit schedules, choose_synthesis_epsilon may depend on the occurrence
  step_index, and a schedule item may override epsilon directly. The evaluator
  owns the recorded random seed; do not create another source of randomness.
- Return a positive finite epsilon for every nonzero rotation.

Hard constraints:
- Do not change molecules, geometries, basis sets, active spaces, target time,
  chemical accuracy, compiler behavior, evaluator code, or files outside simple_pf.py.
- Do not run the evaluator, inspect cluster files, install packages, access the network,
  or search outside this disposable directory. Use only the diagnostics below.
- Keep the policy auditable; do not add unseeded randomness or large lookup tables.

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
    work_dir = directory / "work"
    work_policy = work_dir / "simple_pf.py"
    parent_snapshot = directory / "parent_simple_pf.py"
    stdout_path = directory / "codex.stdout"
    stderr_path = directory / "codex.stderr"
    last_message_path = directory / "last_message.txt"
    launch_intent = read_json(launch_intent_path) if launch_intent_path.is_file() else {}
    launch_receipt = read_json(launch_receipt_path) if launch_receipt_path.is_file() else {}
    process_record = read_json(process_path) if process_path.is_file() else {}
    command_sha256 = str(record.get("launch_command_sha256", ""))
    launch_nonce = str(record.get("launch_nonce", ""))
    work_records = exact_regular_file_records(work_dir) if work_dir.is_dir() else {}
    work_directories = exact_directory_paths(work_dir) if work_dir.is_dir() else []
    unexpected_work_files = sorted(set(work_records) - {"simple_pf.py"})
    archived_evidence = archived_attempt_evidence(directory)
    archived_directories = archived_attempt_directory_paths(directory)
    changed = (
        work_policy.is_file()
        and parent_snapshot.is_file()
        and work_policy.read_bytes() != parent_snapshot.read_bytes()
    )
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
        or not process_path.is_file()
        or not stdout_path.is_file()
        or not stderr_path.is_file()
        or not last_message_path.is_file()
        or record.get("last_message_sha256") != nonempty_text_sha256(last_message_path)
        or record.get("prompt_sha256") != sha256_text(prompt_path.read_text(encoding="utf-8"))
        or read_json(intent_path).get("prompt_sha256") != record.get("prompt_sha256")
        or record.get("intent_sha256") != sha256_file(intent_path)
        or launch_intent.get("command_sha256") != command_sha256
        or launch_intent.get("launch_nonce") != launch_nonce
        or launch_intent.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or launch_intent.get("required_start_event") != CODEX_START_EVENT
        or launch_intent.get("required_terminal_event") != CODEX_TERMINAL_EVENT
        or record.get("launch_intent_sha256") != sha256_file(launch_intent_path)
        or launch_receipt.get("command_sha256") != command_sha256
        or launch_receipt.get("status") != "codex_started"
        or launch_receipt.get("launch_nonce") != launch_nonce
        or launch_receipt.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or record.get("launch_receipt_sha256") != sha256_file(launch_receipt_path)
        or record.get("exec_acknowledgement_sha256") != sha256_file(launch_receipt_path)
        or record.get("exec_acknowledgement_status") != "codex_started"
        or record.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or record.get("codex_thread_id") != launch_receipt.get("codex_thread_id")
        or record.get("codex_start_event_sha256") != launch_receipt.get("codex_start_event_sha256")
        or record.get("codex_terminal_event_sha256")
        != process_record.get("codex_terminal_event_sha256")
        or record.get("codex_lifecycle_status") != "turn_completed"
        or record.get("codex_usage") != process_record.get("codex_usage")
        or record.get("codex_usage_sha256") != process_record.get("codex_usage_sha256")
        or record.get("archived_attempt_evidence") != archived_evidence
        or record.get("archived_attempt_directory_paths") != archived_directories
        or record.get("process_sha256") != sha256_file(process_path)
        or record.get("work_file_records") != work_records
        or record.get("work_directory_paths") != work_directories
        or record.get("work_policy_sha256")
        != (sha256_file(work_policy) if work_policy.is_file() else None)
        or record.get("unexpected_work_files") != unexpected_work_files
        or record.get("changed") is not changed
        or record.get("codex_stdout_sha256") != sha256_file(stdout_path)
        or record.get("codex_stderr_sha256") != sha256_file(stderr_path)
        or process_record.get("codex_stdout_sha256") != sha256_file(stdout_path)
        or process_record.get("codex_stderr_sha256") != sha256_file(stderr_path)
        or process_record.get("last_message_sha256") != nonempty_text_sha256(last_message_path)
        or record.get("returncode") != process_record.get("returncode")
        or record.get("timed_out") != process_record.get("timed_out")
        or record.get("mutation_outcome") not in {"mutated", "fallback_parent"}
        or (
            record.get("mutation_outcome") == "mutated"
            and (
                not changed
                or record.get("valid") is not True
                or bool(unexpected_work_files)
                or bool(work_directories)
            )
        )
    ):
        raise RuntimeError(
            f"immutable mutation record drift for {lane['lane_id']} iteration {iteration}"
        )
    validate_exec_acknowledgement(
        launch_receipt,
        command_sha256=command_sha256,
        work_dir=work_dir,
        launch_nonce=launch_nonce,
    )
    validate_process_crosslink(
        process_record,
        launch_receipt=launch_receipt,
        launch_receipt_sha256=sha256_file(launch_receipt_path),
        command_sha256=command_sha256,
    )
    candidate_files = exact_regular_file_records(candidate.parent)
    candidate_directories = exact_directory_paths(candidate.parent)
    if list(candidate_files) != ["simple_pf.py"] or candidate_directories:
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
    if prompt_path.is_symlink():
        raise RuntimeError("intent prompt path is a symlink")
    if prompt_path.exists():
        if not prompt_path.is_file():
            raise RuntimeError("intent prompt path is not a regular file")
        if sha256_text(prompt_path.read_text(encoding="utf-8")) != intent["prompt_sha256"]:
            raise RuntimeError("intent prompt hash drift")
    else:
        prompt_path.write_text(prompt_text, encoding="utf-8")
    parent_snapshot = directory / "parent_simple_pf.py"
    if parent_snapshot.is_symlink():
        raise RuntimeError("intent parent snapshot is a symlink")
    if parent_snapshot.exists():
        if not parent_snapshot.is_file():
            raise RuntimeError("intent parent snapshot is not a regular file")
        if sha256_file(parent_snapshot) != intent["parent_policy_sha256"]:
            raise RuntimeError("intent parent snapshot hash drift")
    else:
        shutil.copy2(parent_source, parent_snapshot)
    work_dir = directory / "work"
    if work_dir.is_symlink():
        raise RuntimeError("mutation work path is a symlink")
    if not work_dir.exists():
        work_dir.mkdir(parents=True)
        shutil.copy2(parent_snapshot, work_dir / "simple_pf.py")
    elif not work_dir.is_dir():
        raise RuntimeError("mutation work path is not a real directory")
    elif not (work_dir / "simple_pf.py").exists():
        if any(work_dir.iterdir()):
            raise RuntimeError("mutation work directory lacks simple_pf.py but is not empty")
        shutil.copy2(parent_snapshot, work_dir / "simple_pf.py")
    work_policy = work_dir / "simple_pf.py"
    if work_policy.is_symlink() or not work_policy.is_file():
        raise RuntimeError("mutation work policy is not a regular file")
    return work_dir


def recover_interrupted_launch(
    directory: Path, work_dir: Path
) -> tuple[bool, dict[str, Any] | None]:
    launch_intent = directory / "launch_intent.json"
    receipt_path = directory / "launch.json"
    if not launch_intent.exists():
        return False, None
    intent_payload = read_json(launch_intent)
    expected_command_hash = str(intent_payload.get("command_sha256", ""))
    launch_nonce = str(intent_payload.get("launch_nonce", ""))
    if SHA256_RE.fullmatch(launch_nonce) is None:
        raise RuntimeError("interrupted launch intent has an invalid nonce")
    deadline = time.monotonic() + 5.0
    while not receipt_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("command_sha256") != expected_command_hash
            or receipt.get("launch_nonce") != launch_nonce
            or receipt.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        ):
            raise RuntimeError("mutation launch command hash mismatch")
        status = receipt.get("status")
        if status in {"exec_confirmed_pending_codex", "codex_started"}:
            if status == "codex_started":
                validate_exec_acknowledgement(
                    receipt,
                    command_sha256=expected_command_hash,
                    work_dir=work_dir,
                    launch_nonce=launch_nonce,
                )
            terminate_recorded_process_group(receipt, work_dir)
        elif status not in {"exec_error", "codex_start_error"}:
            raise RuntimeError(f"unknown interrupted Codex launch status {status!r}")
        process_path = directory / "process.json"
        process_record = read_json(process_path) if process_path.is_file() else None
        if status == "codex_started" and process_record is not None:
            validate_process_crosslink(
                process_record,
                launch_receipt=receipt,
                launch_receipt_sha256=sha256_file(receipt_path),
                command_sha256=expected_command_hash,
            )
            return True, process_record
        archive_failed_attempt(
            directory,
            launch_nonce=launch_nonce,
            reason=f"interrupted attempt never completed a Codex turn (status={status})",
        )
        return False, None
    else:
        unreceipted = _find_unreceipted_process_groups(work_dir)
        if unreceipted:
            raise RuntimeError(
                f"found unreceipted mutation process groups for {work_dir}: {unreceipted}"
            )
        archive_failed_attempt(
            directory,
            launch_nonce=launch_nonce,
            reason="interrupted attempt produced no durable launch receipt",
        )
        return False, None


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
    launch_nonce = str(launch_intent.get("launch_nonce", ""))
    if (
        launch_intent.get("command_sha256") != launch_receipt.get("command_sha256")
        or launch_receipt.get("launch_nonce") != launch_nonce
        or launch_intent.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
        or launch_intent.get("required_start_event") != CODEX_START_EVENT
        or launch_intent.get("required_terminal_event") != CODEX_TERMINAL_EVENT
    ):
        raise RuntimeError(f"Codex launch command receipt mismatch for {lane_id}")
    work_dir = directory / "work"
    validate_exec_acknowledgement(
        launch_receipt,
        command_sha256=str(launch_intent.get("command_sha256")),
        work_dir=work_dir,
        launch_nonce=launch_nonce,
    )
    launch_receipt_hash = sha256_file(launch_receipt_path)
    if process_record is None:
        raise RuntimeError(f"no completed Codex process evidence for {lane_id}")
    validate_process_crosslink(
        process_record,
        launch_receipt=launch_receipt,
        launch_receipt_sha256=launch_receipt_hash,
        command_sha256=str(launch_intent.get("command_sha256")),
    )
    work_policy = work_dir / "simple_pf.py"
    if not work_policy.is_file():
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent, work_policy)
    work_file_records = exact_regular_file_records(work_dir)
    work_directory_paths = exact_directory_paths(work_dir)
    if "simple_pf.py" not in work_file_records:
        raise RuntimeError(f"Codex work evidence has no simple_pf.py for {lane_id}")
    unexpected_files = sorted(set(work_file_records) - {"simple_pf.py"})
    work_changed = work_policy.read_bytes() != parent.read_bytes()
    work_valid, validation_error = validate_policy(work_policy, lane["species"])
    returncode = process_record.get("returncode")
    timed_out = bool(process_record.get("timed_out"))
    stdout_path = directory / "codex.stdout"
    stderr_path = directory / "codex.stderr"
    last_message_path = directory / "last_message.txt"
    try:
        last_message_hash = nonempty_text_sha256(last_message_path)
    except RuntimeError as exc:
        raise RuntimeError(f"completed Codex attempt evidence is missing for {lane_id}") from exc
    if not stdout_path.is_file() or not stderr_path.is_file() or not last_message_path.is_file():
        raise RuntimeError(f"completed Codex attempt evidence is missing for {lane_id}")
    lifecycle = validate_codex_lifecycle(stdout_path)
    for field in (
        "codex_thread_id",
        "codex_start_event_sha256",
        "codex_lifecycle_status",
        "codex_terminal_event_sha256",
        "codex_usage",
        "codex_usage_sha256",
    ):
        if process_record.get(field) != lifecycle.get(field):
            raise RuntimeError(f"Codex process lifecycle evidence drift for {lane_id}: {field}")
    output_hashes = {
        "codex_stdout_sha256": sha256_file(stdout_path),
        "codex_stderr_sha256": sha256_file(stderr_path),
        "last_message_sha256": last_message_hash,
    }
    if any(process_record.get(field) != digest for field, digest in output_hashes.items()):
        raise RuntimeError(f"Codex process output evidence drift for {lane_id}")
    use_mutation = (
        returncode == 0
        and not timed_out
        and work_changed
        and work_valid
        and not unexpected_files
        and not work_directory_paths
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
    candidate_files = exact_regular_file_records(candidate.parent)
    candidate_directories = exact_directory_paths(candidate.parent)
    if list(candidate_files) != ["simple_pf.py"] or candidate_directories:
        raise RuntimeError(
            f"candidate directory is not exact for {lane_id}: "
            f"files={sorted(candidate_files)}, directories={candidate_directories}"
        )
    candidate_hash = sha256_file(candidate)
    archived_evidence = archived_attempt_evidence(directory)
    archived_directories = archived_attempt_directory_paths(directory)
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
        "launch_nonce": launch_nonce,
        "launch_intent_sha256": sha256_file(launch_intent_path),
        "launch_receipt_sha256": launch_receipt_hash,
        "exec_acknowledgement_sha256": launch_receipt_hash,
        "exec_acknowledgement_status": "codex_started",
        "exec_ack_protocol": EXEC_ACK_PROTOCOL,
        "codex_thread_id": process_record["codex_thread_id"],
        "codex_start_event_sha256": process_record["codex_start_event_sha256"],
        "codex_lifecycle_status": process_record["codex_lifecycle_status"],
        "codex_terminal_event_sha256": process_record["codex_terminal_event_sha256"],
        "codex_usage": process_record["codex_usage"],
        "codex_usage_sha256": process_record["codex_usage_sha256"],
        "archived_attempt_evidence": archived_evidence,
        "archived_attempt_directory_paths": archived_directories,
        "process_sha256": sha256_file(directory / "process.json"),
        **output_hashes,
        "work_file_records": work_file_records,
        "work_directory_paths": work_directory_paths,
        "work_policy_sha256": sha256_file(work_policy),
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
        process_path = directory / "process.json"
        stale_attempt_paths = [
            path
            for path in (
                stdout_path,
                stderr_path,
                last_message_path,
                launch_receipt_path,
                process_path,
            )
            if path.exists() or path.is_symlink()
        ]
        if stale_attempt_paths:
            raise RuntimeError(f"unarchived mutation-attempt residue: {stale_attempt_paths}")
        if launch_intent_path.is_file():
            launch_intent_payload = read_json(launch_intent_path)
            if (
                launch_intent_payload.get("schema_version") != SCHEMA_VERSION
                or launch_intent_payload.get("lane_id") != lane_id
                or launch_intent_payload.get("iteration") != iteration
                or launch_intent_payload.get("command_sha256") != sha256_text("\0".join(command))
                or SHA256_RE.fullmatch(str(launch_intent_payload.get("launch_nonce", ""))) is None
                or launch_intent_payload.get("exec_ack_protocol") != EXEC_ACK_PROTOCOL
                or launch_intent_payload.get("required_start_event") != CODEX_START_EVENT
                or launch_intent_payload.get("required_terminal_event") != CODEX_TERMINAL_EVENT
            ):
                raise RuntimeError("persisted launch intent drift")
        else:
            launch_intent_payload = {
                "schema_version": SCHEMA_VERSION,
                "lane_id": lane_id,
                "iteration": iteration,
                "command_sha256": sha256_text("\0".join(command)),
                "launch_nonce": secrets.token_hex(32),
                "exec_ack_protocol": EXEC_ACK_PROTOCOL,
                "required_start_event": CODEX_START_EVENT,
                "required_terminal_event": CODEX_TERMINAL_EVENT,
                "requested_at": now_iso(),
            }
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
                    "--launch-nonce",
                    str(launch_intent_payload["launch_nonce"]),
                    "--",
                    *command,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                receipt_deadline = asyncio.get_running_loop().time() + 120.0
                launch_receipt: dict[str, Any] = {}
                terminal_launch_statuses = {"codex_started", "exec_error", "codex_start_error"}
                while asyncio.get_running_loop().time() < receipt_deadline:
                    if launch_receipt_path.is_file():
                        launch_receipt = read_json(launch_receipt_path)
                        if launch_receipt.get("status") in terminal_launch_statuses:
                            break
                    await asyncio.sleep(0.02)
                if launch_receipt.get("status") not in terminal_launch_statuses:
                    raise RuntimeError("Codex emitted no durable startup acknowledgement")
                if (
                    int(launch_receipt.get("pid", -1)) != process.pid
                    or int(launch_receipt.get("pgid", -1)) != process.pid
                    or launch_receipt.get("work_dir") != str(work_dir.resolve())
                    or launch_receipt.get("launch_nonce") != launch_intent_payload["launch_nonce"]
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
        receipt_for_process: dict[str, Any] = (
            read_json(launch_receipt_path) if launch_receipt_path.is_file() else {}
        )
        lifecycle: dict[str, Any] = {}
        lifecycle_error: str | None = None
        try:
            lifecycle = validate_codex_lifecycle(stdout_path)
        except Exception as exc:
            lifecycle_error = f"{type(exc).__name__}: {exc}"
        stdout_hash = sha256_file(stdout_path) if stdout_path.is_file() else None
        stderr_hash = sha256_file(stderr_path) if stderr_path.is_file() else None
        try:
            last_message_hash = nonempty_text_sha256(last_message_path)
        except RuntimeError:
            last_message_hash = None
        process_record = {
            "command": command[:-1] + ["<prompt recorded in prompt.md>"],
            "command_sha256": sha256_text("\0".join(command)),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "process_error": process_error,
            "lifecycle_error": lifecycle_error,
            "pid": receipt_for_process.get("pid"),
            "pgid": receipt_for_process.get("pgid"),
            "exec_pid": receipt_for_process.get("exec_pid"),
            "exec_acknowledgement_status": receipt_for_process.get("status"),
            "exec_ack_protocol": receipt_for_process.get("exec_ack_protocol"),
            "launch_nonce": receipt_for_process.get("launch_nonce"),
            "codex_thread_id": lifecycle.get("codex_thread_id"),
            "codex_start_event_sha256": lifecycle.get("codex_start_event_sha256"),
            "codex_lifecycle_status": lifecycle.get("codex_lifecycle_status", "invalid"),
            "codex_terminal_event_sha256": lifecycle.get("codex_terminal_event_sha256"),
            "codex_usage": lifecycle.get("codex_usage"),
            "codex_usage_sha256": lifecycle.get("codex_usage_sha256"),
            "codex_stdout_sha256": stdout_hash,
            "codex_stderr_sha256": stderr_hash,
            "last_message_sha256": last_message_hash,
            "launch_receipt_sha256": sha256_file(launch_receipt_path)
            if launch_receipt_path.is_file()
            else None,
            "started_at": started_at,
            "completed_at": now_iso(),
        }
        write_json(directory / "process.json", process_record)
        failure_reasons: list[str] = []
        if receipt_for_process.get("status") != "codex_started":
            failure_reasons.append(
                f"startup status was {receipt_for_process.get('status', 'missing')!r}"
            )
        if returncode != 0:
            failure_reasons.append(f"wrapper return code was {returncode}")
        if timed_out:
            failure_reasons.append("Codex attempt timed out")
        if lifecycle_error is not None:
            failure_reasons.append(lifecycle_error)
        if last_message_hash is None:
            failure_reasons.append("Codex produced no nonempty last message")
        if lifecycle and (
            lifecycle.get("codex_thread_id") != receipt_for_process.get("codex_thread_id")
            or lifecycle.get("codex_start_event_sha256")
            != receipt_for_process.get("codex_start_event_sha256")
        ):
            failure_reasons.append("Codex startup and terminal lifecycle evidence disagree")
        if failure_reasons:
            archive_failed_attempt(
                directory,
                launch_nonce=str(launch_intent_payload["launch_nonce"]),
                reason="; ".join(failure_reasons),
            )
            if cancellation is not None:
                raise cancellation
            raise RuntimeError(
                f"{lane_id}: Codex proposal did not complete; rerun this same iteration"
            )
        launch_receipt = receipt_for_process
        validate_exec_acknowledgement(
            launch_receipt,
            command_sha256=str(launch_intent_payload["command_sha256"]),
            work_dir=work_dir,
            launcher_pid=None if process is None else process.pid,
            launch_nonce=str(launch_intent_payload["launch_nonce"]),
        )
        validate_process_crosslink(
            process_record,
            launch_receipt=launch_receipt,
            launch_receipt_sha256=sha256_file(launch_receipt_path),
            command_sha256=str(launch_intent_payload["command_sha256"]),
        )
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
