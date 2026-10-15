"""Configure OpenShell, isolate OpenCode mutations, and record agent telemetry."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from ..evaluation.compiled_unitary import validate_policy_source
from .mutation_retry import AgentRuntimeError, record_mutation_failure, resume_mutations
from .research_context import (
    MUTATION_PROTOCOL_VERSION,
    POLICY_CONTRACT,
    benchmark_summary,
    duplicate_policy,
    lane_history,
)


NVIDIA_PROVIDER = "hsopt-nvidia-native"
NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
NEMOTRON_MODEL = os.environ.get("HSOPT_MUTATION_MODEL", "nvidia/nvidia/nemotron-3.5-lightning")
OPENCODE_MODEL = f"openshell/{NEMOTRON_MODEL}"
OPENCODE_AGENT = "hsopt-mutation"
# Limits describe the inference deployment, not the checkpoint's advertised size.
# The live Lightning deployment accepts 128K total context and 64K response requests.
MODEL_CONTEXT_LIMIT = int(os.environ.get("HSOPT_MODEL_CONTEXT_LIMIT", "131072"))
MODEL_OUTPUT_LIMIT = int(os.environ.get("HSOPT_MODEL_OUTPUT_LIMIT", "65536"))
ONLINE_TOOLS = frozenset({"webfetch", "websearch", "codesearch"})
RESEARCH_CONTAMINATION_MARKERS = ("bestquark", "hsopt", "initial_pf")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
HSOPT_SKILL_PATH = REPOSITORY_ROOT / "skills" / "hsopt" / "SKILL.md"
EVALUATOR_SOURCE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "compiled_unitary.py"
SANDBOX_CONTEXT_ROOT = PurePosixPath("/sandbox/work")
READABLE_CONTEXT_FILES = frozenset({
    "candidate.py", "evaluate.py", "protocol_evaluate.py", "problem.json", "reaction.json",
})
OPENCODE_NETWORK_BINARIES = (
    "/usr/lib/node_modules/@opencode/cli/bin/opencode.exe",
    "/usr/local/lib/node_modules/@opencode/cli/bin/opencode.exe",
    "/usr/lib/node_modules/@opencode/cli-linux-x64/bin/opencode",
    "/usr/local/lib/node_modules/@opencode/cli-linux-x64/bin/opencode",
)
RESEARCH_ENDPOINTS = (
    "mcp.exa.ai:443",
    "arxiv.org:443:read-only:rest:enforce",
    "export.arxiv.org:443:read-only:rest:enforce",
    "ar5iv.labs.arxiv.org:443:read-only:rest:enforce",
    "doi.org:443:read-only:rest:enforce",
    "journals.aps.org:443:read-only:rest:enforce",
    "quantum-journal.org:443:read-only:rest:enforce",
)
OPENCODE_FILE_LIST_SHIM = """#!/bin/sh
# OpenCode uses `rg --files` to discover skills. Glob/grep/list are denied to
# the model, so this deliberately implements only the file-listing operation
# needed by the skill loader and avoids an in-sandbox binary download.
exec /usr/bin/find . -type f ! -path '*/.git/*' -print
"""


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    timeout: int | float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_setup_command(sandbox_name: str, command: Sequence[str], run_dir: Path) -> None:
    """Verify a host setup command independently of OpenShell's SSH exit status.

    OpenShell 0.0.89 has returned 1 for marker commands explicitly exiting 0.
    A nonce-bound receipt captures the actual setup command's status before
    the wrapper exits. Never apply this to agent execution or accept a missing
    receipt, a command failure, or another transport error as success.
    """
    args = list(command)
    bundle_install = args == ["tar", "-xf", "/sandbox/hsopt-bootstrap.tar", "-C", "/sandbox"]
    config_switch = args == ["cp", "-f", "/sandbox/opencode.mutation.json", "/sandbox/opencode.json"]
    if not (bundle_install or config_switch) and (
        len(args) < 3
        or (args[0], args[1])
        not in {
            ("mkdir", "-p"),
            ("chmod", "0444"),
            ("chmod", "755"),
        }
    ):
        raise ValueError("verified setup only supports mkdir/chmod or fixed bundle operations")
    if not (bundle_install or config_switch) and any(
        not path.startswith("/sandbox/")
        or PurePosixPath(path) == PurePosixPath("/sandbox")
        or ".." in PurePosixPath(path).parts
        for path in args[2:]
    ):
        raise ValueError("setup paths must remain inside /sandbox")
    marker = "HSOPT_SETUP_" + uuid.uuid4().hex
    invocation = [
        "openshell",
        "sandbox",
        "exec",
        "--name",
        sandbox_name,
        "--no-tty",
        "--timeout",
        "30",
        "--",
        "sh",
        "-c",
        'marker="$1"; shift; "$@"; status=$?; printf "\\n%s=%s\\n" "$marker" "$status"; '
        'exit "$status"',
        "hsopt-setup",
        marker,
        *args,
    ]
    audit: dict[str, Any] = {
        "schema": 1,
        "receipt_protocol": "host-setup-v1",
        "at": datetime.now(UTC).isoformat(),
        "command": args,
        "receipt_valid": False,
        "command_returncode": None,
    }
    try:
        result = _run(invocation, check=False, timeout=45)
        audit["transport_returncode"] = result.returncode
        match = re.fullmatch(r"\n" + re.escape(marker) + r"=([0-9]{1,3})\n", result.stdout)
        status = int(match[1]) if match else None
        audit["receipt_valid"] = status is not None and status <= 255
        audit["command_returncode"] = status
        if not audit["receipt_valid"] or result.returncode not in (0, 1, status):
            raise AgentRuntimeError(
                "OpenShell setup command completion could not be verified",
                retryable=True,
                category="sandbox_transport",
            )
        if status != 0:
            raise subprocess.CalledProcessError(status, args, stderr=result.stderr)
    except Exception as exc:
        audit["error_type"] = type(exc).__name__
        raise
    finally:
        with (run_dir / "setup_commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, sort_keys=True) + "\n")


def _download_candidate(sandbox_name: str, run_dir: Path, *, timeout_seconds: int) -> None:
    """Retry only known SSH probe/tar failures while downloading the same file.

    Every accepted transfer must return zero. At most three transport attempts
    share the original download timeout; no agent turn or proposal is repeated.
    Candidate validation still runs afterwards, including duplicate rejection.
    """
    if timeout_seconds <= 0:
        raise ValueError("candidate download timeout must be positive")
    candidate = run_dir / "candidate.py"
    command = [
        "openshell",
        "sandbox",
        "download",
        sandbox_name,
        "/sandbox/work/candidate.py",
        str(candidate),
    ]
    deadline = time.monotonic() + timeout_seconds
    for attempt in range(1, 4):
        started = time.monotonic()
        remaining = deadline - started
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        audit: dict[str, Any] = {
            "schema": 1,
            "transfer_protocol": "candidate-download-v1",
            "at": datetime.now(UTC).isoformat(),
            "attempt": attempt,
            "status": "failed",
            "will_retry": False,
        }
        try:
            result = _run(command, check=False, timeout=remaining)
            audit["returncode"] = result.returncode
            if result.returncode == 0:
                audit["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                audit["status"] = "downloaded"
                return
            reason = re.search(
                r"(?m)(?:^|[ \t])ssh (probe|tar create) exited with status exit status: 1[ \t]*$",
                result.stderr,
            )
            audit["retry_reason"] = reason[1] if reason and result.returncode == 1 else None
            audit["will_retry"] = bool(
                audit["retry_reason"] and attempt < 3 and time.monotonic() < deadline
            )
            if not audit["will_retry"]:
                raise subprocess.CalledProcessError(
                    result.returncode, command, output=result.stdout, stderr=result.stderr
                )
        except Exception as exc:
            audit["error_type"] = type(exc).__name__
            raise
        finally:
            audit["seconds"] = time.monotonic() - started
            with (run_dir / "candidate_downloads.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(audit, sort_keys=True) + "\n")
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _configure_research_network(sandbox_name: str) -> dict[str, Any]:
    """Best-effort, binary-scoped egress for optional literature research."""

    command = ["openshell", "policy", "update", sandbox_name]
    for endpoint in RESEARCH_ENDPOINTS:
        command.extend(["--add-endpoint", endpoint])
    for binary in OPENCODE_NETWORK_BINARIES:
        command.extend(["--binary", binary])
    command.extend(["--wait", "--timeout", "60"])
    try:
        result = _run(command, check=False, timeout=75)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "optional_unavailable",
            "required": False,
            "endpoints": list(RESEARCH_ENDPOINTS),
            "binaries": list(OPENCODE_NETWORK_BINARIES),
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
    configured = result.returncode == 0
    return {
        "status": "configured" if configured else "optional_unavailable",
        "required": False,
        "endpoints": list(RESEARCH_ENDPOINTS),
        "binaries": list(OPENCODE_NETWORK_BINARIES),
        "returncode": result.returncode,
        "error": None if configured else result.stderr.strip()[:1000],
    }


def _write_opencode_file_list_shim(path: Path) -> None:
    """Supply only the `rg --files` behavior needed for skill discovery."""

    path.write_text(OPENCODE_FILE_LIST_SHIM, encoding="utf-8")
    path.chmod(0o755)


@contextmanager
def _mutation_phase(run_dir: Path, name: str):
    """Persist host phase timings, including failed attempts, without payloads."""
    started = time.monotonic()
    try:
        yield
    finally:
        path = run_dir / "mutation_timing.json"
        value = json.loads(path.read_text()) if path.exists() else {"schema": 1, "phases": {}}
        value["phases"][name] = time.monotonic() - started
        _write_json(path, value)


def _sandbox_setup_gate(run_dir: Path):
    """Let an external deployment bound setup independently of model inference."""
    hook = os.environ.get("HSOPT_SANDBOX_SETUP_GATE")
    if not hook:
        return nullcontext()
    module, function = hook.split(":", 1)
    return getattr(importlib.import_module(module), function)(run_dir)


def _build_sandbox_bundle(run_dir: Path, candidate: Path, context_files) -> Path:
    """Bundle only explicitly permitted inputs, preserving read-only file modes."""
    entries = {
        "work/candidate.py": (candidate, 0o644),
        "work/.opencode/skills/hsopt/SKILL.md": (run_dir / "hsopt.SKILL.md", 0o444),
        "work/.local/bin/rg": (run_dir / "rg", 0o755),
        "opencode.json": (run_dir / "opencode.preparation.json", 0o444),
        "opencode.mutation.json": (run_dir / "opencode.json", 0o444),
        "hsopt-preparation-prompt.txt": (run_dir / "preparation_prompt.md", 0o444),
        "hsopt-mutation-prompt.txt": (run_dir / "prompt.md", 0o444),
    }
    for source, relative in context_files:
        relative = PurePosixPath(relative)
        name = "work/" + relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or name in entries:
            raise ValueError("invalid or colliding sandbox context path")
        entries[name] = (source, 0o444)
    archive = run_dir / "sandbox-bootstrap.tar"
    manifest = {"schema": 1, "files": {}, "prompts": {}}
    with tarfile.open(archive, "w") as bundle:
        for name, (source, mode) in sorted(entries.items()):
            data = source.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            bundle.addfile(info, io.BytesIO(data))
            digest = hashlib.sha256(data).hexdigest()
            manifest["files"][name] = {"sha256": digest, "mode": oct(mode)}
            if name in {"hsopt-preparation-prompt.txt", "hsopt-mutation-prompt.txt"}:
                manifest["prompts"][digest] = "/sandbox/" + name
    _write_json(run_dir / "sandbox_bundle_manifest.json", manifest)
    return archive


def _stdin_prompt_command(command: Sequence[str], run_dir: Path) -> list[str]:
    """Stage OpenCode prompts as stdin, outside OpenShell's 32-KiB argv limit.

    OpenCode reads non-TTY stdin. A fixed shell redirection transports
    the complete UTF-8 prompt without shell interpolation or argv quoting.
    Keep the staged bytes with the audit artifacts; never truncate feedback.
    """
    args = list(command)
    if args[:3] != ["openshell", "sandbox", "exec"] or "--" not in args:
        return args
    separator = args.index("--")
    if args[separator + 1 : separator + 3] != ["opencode", "run"]:
        return args
    sandbox_name = args[args.index("--name") + 1]
    manifest_path = run_dir / "sandbox_bundle_manifest.json"
    remote = None
    if manifest_path.exists():
        digest = hashlib.sha256(args[-1].encode()).hexdigest()
        remote = json.loads(manifest_path.read_text()).get("prompts", {}).get(digest)
        if remote not in {
            None,
            "/sandbox/hsopt-preparation-prompt.txt",
            "/sandbox/hsopt-mutation-prompt.txt",
        }:
            raise ValueError("invalid bundled prompt path")
    if remote is None:
        return _upload_prompt_command(args, run_dir, sandbox_name, separator)
    return _redirect_prompt_command(args, separator, remote)


def _upload_prompt_command(args, run_dir, sandbox_name, separator):
    prompt_file = run_dir / f"transport-prompt-{uuid.uuid4().hex[:12]}.txt"
    prompt_file.write_text(args[-1], encoding="utf-8")
    remote = f"/sandbox/{prompt_file.name}"
    _run(
        [
            "openshell",
            "sandbox",
            "upload",
            "--no-git-ignore",
            sandbox_name,
            str(prompt_file),
            remote,
        ],
        timeout=45,
    )
    _run_setup_command(sandbox_name, ["chmod", "0444", remote], run_dir)
    return _redirect_prompt_command(args, separator, remote)


def _redirect_prompt_command(args, separator, remote):
    return [
        *args[: separator + 1],
        "sh",
        "-c",
        'prompt_file="$1"; shift; exec "$@" < "$prompt_file"',
        "hsopt-prompt",
        remote,
        *args[separator + 1 : -1],
    ]


def _run_logged(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    completed_edit_grace_seconds: float | None = None,
    required_reads_to_stop: Sequence[str] = (),
    append: bool = False,
) -> tuple[int, bool]:
    """Run a command with logs visible to external monitors while it is active."""

    command = _stdin_prompt_command(command, stdout_path.parent)
    with (
        stdout_path.open("a" if append else "w", encoding="utf-8") as stdout,
        stderr_path.open("a" if append else "w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            list(command),
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        deadline = time.monotonic() + timeout
        edit_completed_at: float | None = None
        reads_completed_at: float | None = None
        while True:
            returncode = process.poll()
            if returncode is not None:
                return returncode, False
            now = time.monotonic()
            if required_reads_to_stop and reads_completed_at is None:
                _, actions = summarize_opencode_events(stdout_path)
                completed = {a['target'] for a in actions
                             if a['tool'] == 'read' and a['status'] == 'completed'}
                if set(required_reads_to_stop) <= completed:
                    reads_completed_at = now
            if (
                completed_edit_grace_seconds is not None
                and edit_completed_at is None
                and _has_completed_candidate_edit(stdout_path)
            ):
                edit_completed_at = now
            if (
                (edit_completed_at is not None
                 and now - edit_completed_at >= completed_edit_grace_seconds)
                or (reads_completed_at is not None and now - reads_completed_at >= 1.0)
            ):
                process.terminate()
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait(timeout=5)
                return returncode, True
            if now >= deadline:
                process.kill()
                process.wait(timeout=5)
                raise subprocess.TimeoutExpired(list(command), timeout)
            time.sleep(0.25)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_sandbox_relative_path(raw_path: str) -> PurePosixPath:
    """Validate one reaction-relative JSON path before exposing it in the sandbox."""

    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or path.suffix != ".json"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe reaction problem path: {raw_path!r}")
    return path


def sandbox_context_files(
    *,
    kind: str,
    source: Path,
) -> list[tuple[Path, PurePosixPath]]:
    """Return the exact evaluator and benchmark files exposed read-only to OpenCode."""

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not EVALUATOR_SOURCE_PATH.is_file():
        raise FileNotFoundError(EVALUATOR_SOURCE_PATH)
    files = [(EVALUATOR_SOURCE_PATH, PurePosixPath("evaluate.py"))]
    if kind == "absolute":
        files.append((source, PurePosixPath("problem.json")))
    elif kind == "difference":
        reaction = json.loads(source.read_text(encoding="utf-8"))
        files.append((source, PurePosixPath("reaction.json")))
        seen: set[PurePosixPath] = set()
        for item in reaction["terms"]:
            relative = _safe_sandbox_relative_path(str(item["problem"]))
            if relative in seen:
                continue
            problem = (source.parent / Path(*relative.parts)).resolve()
            if not problem.is_file():
                raise FileNotFoundError(problem)
            files.append((problem, relative))
            seen.add(relative)
    elif kind == "protocol":
        suite = json.loads(source.read_text())
        if suite.get("schema") != "hsopt-shared-protocol-v1":
            raise ValueError("unsupported shared protocol suite")
        files.append((source, PurePosixPath("suite.json")))
        files.append((Path(__file__).parents[1] / "campaign/protocol.py", PurePosixPath("protocol_evaluate.py")))
        seen = set()
        for case in suite["cases"]:
            relative = _safe_sandbox_relative_path(case["source"])
            problem = (source.parent / relative).resolve()
            paths = [relative]
            if suite["kind"] == "difference":
                paths.extend(relative.parent / _safe_sandbox_relative_path(row["problem"])
                             for row in json.loads(problem.read_text())["terms"])
            for path in paths:
                if path in seen:
                    continue
                local = (source.parent / path).resolve()
                if not local.is_relative_to(source.parent.resolve()) or not local.is_file():
                    raise ValueError("shared protocol context escapes the sealed suite")
                files.append((local, path))
                seen.add(path)
    else:
        raise ValueError("kind must be 'absolute', 'difference' or 'protocol'")
    return files


def _tool_category(tool: str) -> str:
    """Describe the observed tool without inferring unrecorded activity."""
    if tool in {"bash", "terminal", "shell", "exec"}:
        return "terminal"
    if tool == "execute":
        return "code_execution"
    if any(name in tool for name in ("slurm_submit", "submit_job", "slurm_run")):
        return "job_submission"
    if tool in {"websearch", "codesearch"}:
        return "web_search"
    if tool == "webfetch":
        return "web_fetch"
    if tool in {"read", "glob", "grep", "list"}:
        return "file_read"
    if tool in {"write", "edit", "apply_patch", "patch"}:
        return "file_edit"
    return "skill" if tool == "skill" else "other"


def _safe_tool_target(tool: str, payload: Any) -> str | None:
    """Keep an audit-useful target without copying prompts or file contents."""

    if not isinstance(payload, dict):
        return None
    for key in ("filePath", "path", "url", "query", "name"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            continue
        if key == "name" and tool != "skill":
            continue
        if key in {"filePath", "path"}:
            normalized = value.replace("\\", "/")
            sandbox_prefix = f"{SANDBOX_CONTEXT_ROOT.as_posix()}/"
            if normalized.startswith(sandbox_prefix):
                return normalized.removeprefix(sandbox_prefix)
            if normalized.startswith("./"):
                return normalized.removeprefix("./")
            if not normalized.startswith("/"):
                return normalized
            return f"outside:{PurePosixPath(normalized).name}"
        return value[:200]
    if _tool_category(tool) == "terminal":
        command = payload.get("command")
        if isinstance(command, str):
            return re.sub(
                r"(?:nvapi-|sk-)[A-Za-z0-9_-]+|Bearer\s+\S+|(?i:api[_-]?key|token|password)=\S+",
                "[REDACTED]", command,
            )[:200]
    return None


def summarize_opencode_events(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize OpenCode NDJSON into counters and a redacted action timeline."""

    tokens: Counter[str] = Counter()
    action_order: list[str] = []
    actions_by_call: dict[str, dict[str, Any]] = {}
    session_ids: set[str] = set()
    inference_steps = 0
    started_steps = 0
    errors = 0
    if not path.is_file():
        return {
            "schema": 1,
            "raw_log_present": False,
            "tool_calls": 0,
            "by_tool": {},
            "by_status": {},
            "by_category": {},
            "terminal_calls": 0,
            "job_submission_calls": 0,
            "online_tool_calls": 0,
            "web_searches": 0,
            "inference_steps": 0,
            "errors": 0,
            "tokens": {},
            "sessions": [],
        }, []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        session_id = event.get("sessionID")
        if isinstance(session_id, str):
            session_ids.add(session_id)
        kind = event.get("type")
        part = event.get("part") or {}
        if kind == "step_start":
            started_steps += 1
            continue
        if kind == "step_finish":
            inference_steps += 1
            for key, value in (part.get("tokens") or {}).items():
                if key == "cache" and isinstance(value, dict):
                    for cache_key, cache_value in value.items():
                        if isinstance(cache_value, (int, float)):
                            tokens[f"cache_{cache_key}"] += int(cache_value)
                elif isinstance(value, (int, float)):
                    tokens[key] += int(value)
            continue
        if kind == "error":
            errors += 1
            continue
        if kind != "tool_use":
            continue
        call_id = str(part.get("callID") or part.get("id") or "")
        action_key = call_id or f"anonymous-{line_number}"
        previous = actions_by_call.get(action_key, {})
        if action_key not in actions_by_call:
            action_order.append(action_key)
        tool = str(part.get("tool") or previous.get("tool") or "unknown")
        state = part.get("state") or {}
        status = str(state.get("status") or previous.get("status") or "unknown")
        timing = state.get("time") or {}
        start = timing.get("start")
        end = timing.get("end")
        duration_ms = (
            max(0, int(end) - int(start))
            if isinstance(start, (int, float)) and isinstance(end, (int, float))
            else previous.get("duration_ms")
        )
        target = _safe_tool_target(tool, state.get("input")) or previous.get("target")
        # Only a frontend permission rejection proves the tool did not execute.
        # Other errors may occur after a partial write and must remain fatal.
        permission_denied = previous.get("permission_denied", False)
        if "error" in state:
            permission_denied = isinstance(state["error"], str) and bool(
                re.fullmatch(r"Permission denied: (?:edit|external_directory)", state["error"])
            )
        metadata = state.get("metadata") or {}
        unavailable_message = (
            f'No tool named "{tool}" is currently available. '
            "Please use a tool from the available tool list."
        )
        tool_unavailable = (
            state.get("error") == unavailable_message
            and (metadata.get("providerCall") or {}).get("executed") is False
            and (metadata.get("providerResult") or {}).get("executed") is False
        )
        actions_by_call[action_key] = {
            "timestamp": event.get("timestamp") or previous.get("timestamp"),
            "session_id": session_id or previous.get("session_id"),
            "call_id": call_id or None,
            "tool": tool,
            "category": _tool_category(tool),
            "status": status,
            "duration_ms": duration_ms,
            "target": target,
            "online": tool in ONLINE_TOOLS,
            "permission_denied": status == "error" and permission_denied,
            "tool_unavailable": status == "error" and tool_unavailable,
        }
        if tool == "execute":
            metadata = state.get("metadata") or {}
            # v2 puts Code Mode receipts inside the tool-result metadata, rather
            # than emitting each nested call as a separate top-level event.
            detail = metadata.get("metadata", metadata)
            if detail.get("error") is True and status == "completed":
                actions_by_call[action_key]["status"] = "error"
            for index, nested in enumerate(detail.get("toolCalls") or []):
                nested_key = f"{action_key}:nested:{index}"
                if nested_key not in actions_by_call:
                    action_order.append(nested_key)
                qualified = str(nested.get("tool") or "unknown")
                nested_tool = qualified.removeprefix("opencode.")
                actions_by_call[nested_key] = {
                    "timestamp": event.get("timestamp"),
                    "session_id": session_id,
                    "call_id": nested_key,
                    "tool": nested_tool,
                    "category": _tool_category(nested_tool),
                    "status": str(nested.get("status") or "unknown"),
                    "duration_ms": None,
                    "target": _safe_tool_target(nested_tool, nested.get("input")),
                    "online": nested_tool in ONLINE_TOOLS,
                    # Nested error receipts do not include a denial reason.
                    # Never infer a harmless rejection from status alone.
                    "permission_denied": False,
                    "parent_call_id": call_id or action_key,
                }

    actions = [actions_by_call[key] for key in action_order]
    by_tool = Counter(str(action["tool"]) for action in actions)
    by_status = Counter(str(action["status"]) for action in actions)
    metrics = {
        "schema": 1,
        "raw_log_present": True,
        "tool_calls": sum(by_tool.values()),
        "by_tool": dict(sorted(by_tool.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_category": dict(sorted(Counter(action["category"] for action in actions).items())),
        "terminal_calls": sum(action["category"] == "terminal" for action in actions),
        "job_submission_calls": sum(action["category"] == "job_submission" for action in actions),
        "online_tool_calls": sum(by_tool[tool] for tool in ONLINE_TOOLS),
        "successful_online_tool_calls": sum(
            action["online"] and action["status"] == "completed" for action in actions
        ),
        "by_tool_status": dict(
            sorted(Counter(f"{action['tool']}:{action['status']}" for action in actions).items())
        ),
        "web_searches": by_tool["websearch"],
        "inference_steps": max(started_steps, inference_steps),
        "finished_inference_steps": inference_steps,
        "token_usage_complete": started_steps == 0 or started_steps == inference_steps,
        "errors": errors,
        "tokens": dict(sorted(tokens.items())),
        "sessions": sorted(session_ids),
    }
    return metrics, actions


def _has_completed_candidate_edit(path: Path) -> bool:
    """Return whether OpenCode durably reported a completed candidate mutation."""

    _, actions = summarize_opencode_events(path)
    return any(
        action["tool"] in {"edit", "write", "apply_patch", "patch"}
        and action["status"] == "completed"
        and action["target"] == "candidate.py"
        for action in actions
    )


def _is_scratch_path(target: str) -> bool:
    return target.startswith("scratch/") and all(
        part not in {"", ".", ".."} for part in target.split("/")
    )


def _compact_protocol_benchmark(benchmark):
    """Keep every case's starting parameters and Hamiltonian features readable."""
    cases = []
    for case in benchmark["cases"]:
        systems = []
        for h in case["hamiltonians"]:
            row = {k: h[k] for k in ("n_qubits", "electrons", "term_count", "nonidentity_one_norm", "target_time", "pauli_families")}
            initial = case["parameters"][h["name"]]
            row.update(initial_r=initial["n_steps"], initial_epsilon=initial["uniform_epsilon"],
                       initial_beta=initial["reference_trotter_number_beta"])
            row["max_abs_coefficient"] = max(abs(t["coefficient"]) for t in h["largest_terms"])
            row["min_abs_coefficient"] = min(abs(t["coefficient"]) for t in h["smallest_terms"])
            systems.append(row)
        cases.append({**{k: v for k, v in case.items() if k != "parameters"}, "hamiltonians": systems})
    return {**benchmark, "cases": cases}


def _write_protocol_context(path, context):
    """One JSON line per case avoids consuming a read page with indentation."""
    marker = "__HSOPT_PROTOCOL_CASE_LINES__"
    benchmark = context["benchmark"]
    outer = {**context, "benchmark": {**benchmark, "cases": marker}}
    rows = "[\n" + ",\n".join(json.dumps(c, separators=(",", ":"), allow_nan=False)
                              for c in benchmark["cases"]) + "\n]"
    rendered = json.dumps(outer, indent=2, allow_nan=False).replace(json.dumps(marker), rows, 1)
    path.write_text(rendered + "\n")


def _validate_agent_action_contract(
    path: Path,
    *,
    required_reads: Sequence[str] = (),
    preparation: bool = False,
) -> None:
    """Reject a sandbox turn that did anything outside the one-file contract."""

    _, actions = summarize_opencode_events(path)
    allowed_tools = {
        "read",
        "edit",
        "write",
        "apply_patch",
        "patch",
        "skill",
        "execute",
        *ONLINE_TOOLS,
    }
    mutation_tools = {"edit", "write", "apply_patch", "patch"}
    mutation_actions = []
    completed_reads: set[str] = set()
    for action in actions:
        tool = str(action["tool"])
        if action.get("tool_unavailable"):
            # OpenCode conclusively rejected an invented tool before execution.
            # Retain the failed call in telemetry without discarding the proposal.
            continue
        if tool not in allowed_tools:
            raise RuntimeError(f"OpenCode attempted forbidden tool {tool!r}")
        if tool == "read":
            if action["status"] == "error":
                # No file contents reached the model; retain a recoverable
                # read-only session after a bad path or blocked read.
                continue
            target = str(action["target"])
            safe_json = (
                target.endswith(".json")
                and not target.startswith(("outside:", "/", "../"))
                and all(part not in {"", ".", ".."} for part in PurePosixPath(target).parts)
            )
            if target not in READABLE_CONTEXT_FILES and not safe_json and not _is_scratch_path(target):
                raise RuntimeError("OpenCode attempted to read outside the mutation context")
            if action["status"] == "completed":
                completed_reads.add(target)
        if tool in ONLINE_TOOLS and action["target"] is not None:
            research_target = str(action["target"]).casefold()
            if any(marker in research_target for marker in RESEARCH_CONTAMINATION_MARKERS):
                raise RuntimeError("OpenCode research attempted to inspect HSOpt baseline data")
        if tool in mutation_tools:
            if action["permission_denied"]:
                # Frontend rejection happened before execution, including a
                # denied write to a protected input or an undesignated scratch file.
                continue
            if _is_scratch_path(str(action["target"])):
                # Scratch notes/drafts are not the scored policy. They can be
                # changed repeatedly in either stage, with every call audited.
                continue
            if action["target"] != "candidate.py":
                raise RuntimeError("OpenCode attempted to mutate outside candidate.py")
            if preparation:
                raise RuntimeError("OpenCode attempted to edit during read-only preparation")
            missing = set(required_reads) - completed_reads
            if missing:
                raise RuntimeError(
                    "Read required benchmark context before editing: " + ", ".join(sorted(missing))
                )
            mutation_actions.append(action)
    if preparation:
        missing = set(required_reads) - completed_reads
        if missing:
            raise RuntimeError("Preparation missing required reads: " + ", ".join(sorted(missing)))
        return
    completed_mutations = [action for action in mutation_actions if action["status"] == "completed"]
    if len(mutation_actions) != 1 or len(completed_mutations) != 1:
        raise RuntimeError("OpenCode must complete exactly one candidate.py mutation")


def _raise_inference_error(path: Path) -> None:
    """A zero shell exit status does not imply a successful OpenCode/API turn."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "error":
            continue
        error = event.get("error") or {}
        data = error.get("data") or {}
        message = str(
            data.get("message") or error.get("message") or error.get("name") or "OpenCode error"
        )
        status = data.get("statusCode")
        retryable = data.get("isRetryable") is not False and status not in {400, 401, 403, 404, 422}
        # The inference connector wraps database saturation as Authentication Error.
        # Recognize only this specific backend condition; real auth/configuration
        # failures must still stop the campaign instead of retrying credentials.
        normalized = message.casefold()
        database_capacity_errors = (
            "too many clients already",
            "remaining connection slots are reserved for roles with the superuser attribute",
        )
        if "error querying the database" in normalized and any(
            capacity_error in normalized for capacity_error in database_capacity_errors
        ):
            retryable = True
        if "max_tokens cannot exceed" in message:
            retryable = False
        raise AgentRuntimeError(message[:600], retryable=retryable)


def _prepare_mutation_context(
    command, *, run_dir: Path, required_reads, prompt: str, deadline: float
):
    """Finish missing read receipts in the same read-only session, within the budget."""
    session = None
    for attempt in range(3):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise RuntimeError("Mutation time budget exhausted during preparation")
        turn = [*command[:-1], *(["--session", session] if session else []), prompt]
        turn[turn.index("--timeout") + 1] = str(remaining)
        stop_on_reads = os.environ.get("HSOPT_PREPARATION_READ_BARRIER") == "1"
        returncode, stopped_on_reads = _run_logged(
            turn,
            stdout_path=run_dir / "opencode.jsonl",
            stderr_path=run_dir / "opencode.stderr",
            timeout=remaining + 30,
            append=attempt > 0,
            **({"required_reads_to_stop": required_reads} if stop_on_reads else {}),
        )
        if stopped_on_reads and "--name" in command:
            _terminate_sandbox_opencode(command[command.index("--name") + 1])
        _raise_inference_error(run_dir / "opencode.jsonl")
        if returncode != 0 and not stopped_on_reads:
            raise RuntimeError(f"OpenCode preparation exited with status {returncode}")
        # Reject actual/uncertain writes and forbidden tools, but retain a session
        # whose candidate write was conclusively blocked by the permissions layer.
        _validate_agent_action_contract(run_dir / "opencode.jsonl", preparation=True)
        metrics, actions = summarize_opencode_events(run_dir / "opencode.jsonl")
        if len(metrics["sessions"]) != 1:
            raise RuntimeError("Preparation must identify exactly one OpenCode session")
        session = metrics["sessions"][0]
        completed = {
            a["target"] for a in actions if a["tool"] == "read" and a["status"] == "completed"
        }
        missing = sorted(set(required_reads) - completed)
        if not missing:
            _write_json(
                run_dir / "preparation_result.json",
                {
                    "status": "ready",
                    "session": session,
                    "required_reads": required_reads,
                    "read_only_turns": attempt + 1,
                    "host_read_barrier_handoff": stopped_on_reads,
                    "blocked_candidate_writes": sum(
                        a["category"] == "file_edit"
                        and a["target"] == "candidate.py"
                        and a["permission_denied"]
                        for a in actions
                    ),
                    "metrics": metrics,
                },
            )
            return session
        prompt = (
            "Editing is STILL DISABLED. The host is missing successful read receipts for: "
            + json.dumps(["/sandbox/work/" + p for p in missing])
            + ". Your next tool calls must read EACH listed file. Use read with limit=80 "
            "to inspect its header, then page additional ranges as needed. Do not substitute "
            "the summaries for reading these files. Do not edit, write, or patch. "
            "Only after these reads, return your hypothesis. Scratch notes and Code Mode "
            "data checks are allowed; candidate.py must remain unchanged."
        )
        (run_dir / f"preparation_followup_{attempt + 1}.md").write_text(prompt)
    raise RuntimeError("Preparation missing required reads: " + ", ".join(missing))


def _verify_preparation_candidate(sandbox_name: str, run_dir: Path, *, deadline: float) -> None:
    """Verify unchanged candidate bytes before unlocking the scored policy."""
    result = json.loads((run_dir / "preparation_result.json").read_text())
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise RuntimeError("Mutation time budget exhausted during preparation")
    _download_candidate(sandbox_name, run_dir, timeout_seconds=remaining)
    before = _sha256(run_dir / "candidate.before.py")
    after = _sha256(run_dir / "candidate.py")
    _write_json(run_dir / "preparation_integrity.json", {
        "schema": 1,
        "blocked_candidate_writes": result["blocked_candidate_writes"],
        "before_sha256": before,
        "after_sha256": after,
        "unchanged": before == after,
    })
    if before != after:
        raise RuntimeError("Candidate changed during read-only preparation")


_OCSF_RE = re.compile(r"\bOCSF(?:\s*\]\s*\[ocsf\])?\s+(?P<class>[A-Z]+):(?P<activity>[A-Z_]+)")


def summarize_openshell_log(path: Path) -> dict[str, Any]:
    """Count OpenShell security events without copying credentials or payloads."""

    classes: Counter[str] = Counter()
    activities: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    seen_ids: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.partition("OCSF-JSON ")[2] if "OCSF-JSON " in line else line
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict) and "class_uid" in event:
                event_id = str((event.get("metadata") or {}).get("uid") or "")
                if event_id and event_id in seen_ids:
                    continue
                if event_id:
                    seen_ids.add(event_id)
                event_class = {4001: "NET", 4002: "HTTP", 4007: "SSH", 1007: "PROC",
                               5019: "CONFIG", 6002: "LIFECYCLE", 2004: "FINDING"}.get(
                                   event["class_uid"], "EVENT")
                activity = str(event.get("activity_name") or event.get("activity_id", "unknown"))
                decision = str(event.get("disposition") or event.get("action") or "").lower()
                classes[event_class] += 1
                activities[f"{event_class}:{activity}"] += 1
                if decision in {"allowed", "denied", "blocked"}:
                    decisions[decision] += 1
                continue
            match = _OCSF_RE.search(line)
            if not match:
                continue
            event_class = match.group("class")
            activity = match.group("activity")
            decision = next(
                (value for value in ("ALLOWED", "DENIED", "BLOCKED") if value in line),
                None,
            )
            classes[event_class] += 1
            activities[f"{event_class}:{activity}"] += 1
            if decision:
                decisions[decision.lower()] += 1
    return {
        "schema": 1,
        "raw_log_present": path.is_file(),
        "source": path.name,
        "ocsf_events": sum(classes.values()),
        "by_class": dict(sorted(classes.items())),
        "by_activity": dict(sorted(activities.items())),
        "by_decision": dict(sorted(decisions.items())),
        "network_events": classes["NET"] + classes["HTTP"] + classes["SSH"],
        "process_events": classes["PROC"],
        "denied_events": decisions["denied"] + decisions["blocked"],
    }


def _capture_openshell_logs(sandbox_name: str, destination: Path) -> None:
    """Collect supported gateway logs and optional deployment-owned JSON audit."""

    collection = {}
    try:
        completed = _run(
            ["openshell", "logs", sandbox_name, "--source", "all", "-n", "10000"],
            check=False, timeout=45,
        )
        destination.write_text(completed.stdout, encoding="utf-8")
        if completed.stderr:
            with destination.open("a", encoding="utf-8") as stream:
                stream.write("\n== capture stderr ==\n" + completed.stderr)
        collection["gateway_log_returncode"] = completed.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        destination.write_text("", encoding="utf-8")
        collection["gateway_log_error_type"] = type(exc).__name__
    collector = os.environ.get("HSOPT_AUDIT_COLLECTOR")
    if collector:
        try:
            result = _run([collector, sandbox_name, str(destination.parent)], check=False, timeout=45)
            collection["structured_collector_returncode"] = result.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            collection["structured_collector_error_type"] = type(exc).__name__
    _write_json(destination.parent / "audit_collection.json", collection)


def write_agent_audit(run_dir: Path) -> dict[str, Any]:
    """Persist durable, redacted metrics beside the raw agent/sandbox logs."""

    opencode_metrics, actions = summarize_opencode_events(run_dir / "opencode.jsonl")
    structured = run_dir / "openshell_ocsf.jsonl"
    openshell_metrics = summarize_openshell_log(
        structured if structured.is_file() and structured.stat().st_size else run_dir / "openshell.log"
    )
    _write_json(run_dir / "opencode_metrics.json", opencode_metrics)
    _write_json(run_dir / "openshell_metrics.json", openshell_metrics)
    with (run_dir / "agent_actions.jsonl").open("w", encoding="utf-8") as stream:
        for action in actions:
            stream.write(json.dumps(action, sort_keys=True) + "\n")
    provenance: dict[str, Any] = {}
    runtime_versions = None
    for filename in ("mutation_result.json", "mutation_protocol.json", "runtime_versions.json"):
        source = run_dir / filename
        if source.is_file():
            record = json.loads(source.read_text())
            provenance.update({key: value for key, value in record.items() if key == "model"})
            if filename == "runtime_versions.json":
                runtime_versions = record
    audit = {
        "schema": 1,
        "model": provenance.get("model"),
        "runtime_versions": runtime_versions,
        "opencode": opencode_metrics,
        "openshell": openshell_metrics,
        "action_log": "agent_actions.jsonl",
        "raw_opencode_log": "opencode.jsonl",
        "raw_openshell_log": "openshell.log",
        "structured_openshell_log": structured.name if structured.is_file() else None,
        "collection": (
            json.loads((run_dir / "audit_collection.json").read_text())
            if (run_dir / "audit_collection.json").is_file() else None
        ),
    }
    _write_json(run_dir / "agent_audit.json", audit)
    return audit


def _capture_runtime_versions(sandbox_name: str, run_dir: Path) -> None:
    """Record the executables actually used by this mutation round."""
    identity: dict[str, Any] = {
        "model": NEMOTRON_MODEL,
        "opencode_model": OPENCODE_MODEL,
        "sandbox_image": os.environ.get("HSOPT_SANDBOX_IMAGE") or "community/base:latest",
    }
    commands = {
        "openshell": ["openshell", "--version"],
        "opencode": ["openshell", "sandbox", "exec", "--name", sandbox_name,
                     "--no-tty", "--timeout", "30", "--", "opencode", "--version"],
    }
    for name, command in commands.items():
        try:
            result = _run(command, check=False, timeout=45)
            identity[name] = {"version": result.stdout.strip()[:100], "returncode": result.returncode}
        except (OSError, subprocess.SubprocessError) as exc:
            identity[name] = {"version": None, "error_type": type(exc).__name__}
    _write_json(run_dir / "runtime_versions.json", identity)


def configure_inference() -> dict[str, Any]:
    """Configure OpenShell without placing the NVIDIA key in a sandbox."""

    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is not set")
    _run(["openshell", "status"])
    existing = _run(
        ["openshell", "provider", "get", NVIDIA_PROVIDER],
        check=False,
    )
    if existing.returncode == 0:
        _run(
            [
                "openshell",
                "provider",
                "update",
                NVIDIA_PROVIDER,
                "--credential",
                "NVIDIA_API_KEY",
                "--config",
                f"NVIDIA_BASE_URL={NVIDIA_BASE_URL}",
            ]
        )
    else:
        _run(
            [
                "openshell",
                "provider",
                "create",
                "--name",
                NVIDIA_PROVIDER,
                "--type",
                "hsopt-nvidia-native",
                "--credential",
                "NVIDIA_API_KEY",
                "--config",
                f"NVIDIA_BASE_URL={NVIDIA_BASE_URL}",
            ]
        )
    return {
        "status": "configured",
        "provider": NVIDIA_PROVIDER,
        "model": NEMOTRON_MODEL,
        "route": NVIDIA_BASE_URL,
    }


def opencode_config(*, preparation: bool = False) -> dict[str, Any]:
    """Native OpenCode v2 config for one isolated candidate mutation."""

    if not 0 < MODEL_OUTPUT_LIMIT < MODEL_CONTEXT_LIMIT:
        raise ValueError("Model output allowance must be positive and smaller than context")

    permissions = [{"action": "*", "resource": "*", "effect": "deny"}]
    for resource in ("candidate.py", "evaluate.py", "*.json", "**/*.json"):
        permissions.append({"action": "read", "resource": resource, "effect": "allow"})
    permissions.extend([
        {"action": "read", "resource": "scratch/*", "effect": "allow"},
        {"action": "edit", "resource": "scratch/*", "effect": "allow"},
        {"action": "execute", "resource": "*", "effect": "allow"},
        {"action": "edit", "resource": "candidate.py", "effect": "deny" if preparation else "allow"},
        {"action": "skill", "resource": "hsopt", "effect": "allow"},
        {"action": "webfetch", "resource": "*", "effect": "allow"},
        {"action": "websearch", "resource": "*", "effect": "allow"},
    ])
    system = (
        "Editing is now enabled. Your FIRST tool call in THIS continuation must read "
        "candidate.py AGAIN: OpenCode resets its file-read timestamps between processes, "
        "even when the conversation is resumed. Then perform exactly one candidate.py "
        "policy mutation. Verify that you have read "
        "research_context.json, and every required benchmark JSON listed there. "
        "Read them before editing; missing reads invalidate the proposal. Before "
        "editing, you may optionally use skill, websearch, webfetch, "
        "for product-formula research. Read candidate.py, "
        "the read-only evaluate.py implementation, and the read-only "
        "benchmark JSON files as needed. Prefer one complete write of candidate.py "
        "with your final policy, retaining all four interface methods. The host stops "
        "this turn as soon as the first candidate edit completes. Call one edit or write tool "
        "exactly once for candidate.py, then "
        "immediately return a final text response without calling "
        "another tool. ProductFormula.build_step() is the "
        "authoritative recurrence and may return any deterministic "
        "stationary Pauli-rotation schedule. Do not search for this "
        "HSOpt repository, campaign, or any sibling baseline policy."
    )
    if preparation:
        system = (
            "This is the read-only preparation stage of an HSOpt mutation. "
            "Your FIRST tool call must read /sandbox/work/candidate.py. "
            "Then read /sandbox/work/research_context.json and EVERY file in required_reads. "
            "Each listed path is a FILE, never a directory. Do not read or list directories. "
            "The host verifies successful read receipts before enabling editing. "
            "Do not edit, write, or patch candidate.py in this stage. Analyze the concrete "
            "Hamiltonians, policy API, prior rejected proposals, and a new scientific "
            "hypothesis. Optional general product-formula web research is allowed. "
            "Do not search for HSOpt or sibling baselines. Finish with a brief hypothesis; "
            "a separate continuation will enable exactly one candidate.py edit."
        )
    system += (
        " You may read/write scratch/* notes or draft tests in either stage. "
        "Use execute (Code Mode) for JavaScript calculations and small assertions "
        "on data returned by permitted read tools. Nested tools retain their permissions. "
        "Use Math/JSON, console.log and throw new Error for failed assertions; "
        "require, imports and console.assert are unavailable. "
        "There is no shell or Python execution; scratch Python files are drafts only. "
        "Scratch edits do not count as candidate.py edits. Do optional checks BEFORE "
        "the single candidate.py edit; do not run the scientific evaluator."
    )
    return {
        "$schema": "https://opencode.ai/config.json",
        "update": "disable",
        "share": "disabled",
        "snapshots": False,
        "compaction": {
            "auto": True,
            "keep": {"tokens": 15000},
            # Leave room for a complete tool response before the next check.
            "buffer": min(32768, MODEL_CONTEXT_LIMIT // 4),
        },
        "model": OPENCODE_MODEL,
        "enabled_providers": ["openshell"],
        "providers": {
            "openshell": {
                "name": "NVIDIA inference through OpenShell credential proxy",
                "env": ["NVIDIA_API_KEY"],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": NVIDIA_BASE_URL},
                "models": {
                    NEMOTRON_MODEL: {
                        "modelID": NEMOTRON_MODEL,
                        "name": NEMOTRON_MODEL.rsplit("/", 1)[-1],
                        "capabilities": {"tools": True},
                        "limit": {"context": MODEL_CONTEXT_LIMIT, "output": MODEL_OUTPUT_LIMIT},
                    }
                },
            }
        },
        "agents": {
            OPENCODE_AGENT: {
                "description": "Mutate only the isolated HSOpt candidate.py policy.",
                "mode": "primary",
                "model": OPENCODE_MODEL,
                "request": {"body": {"temperature": 1.0, "top_p": 0.95, "max_tokens": MODEL_OUTPUT_LIMIT}},
                "steps": 20,
                "system": system,
                "permissions": permissions,
            }
        },
    }


def mutation_prompt(
    feedback: dict[str, Any] | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> str:
    prompt = """Editing is now enabled after verified read-only preparation.
Your FIRST tool call in THIS continuation MUST read /sandbox/work/candidate.py
AGAIN, even though you read it during preparation. OpenCode resets its file-read
timestamps between processes, so an edit without this fresh read will fail.
After that fresh read, make exactly one candidate.py edit and return final text.
Prefer a complete write of your final policy. The host stops at the first
completed candidate edit, so prepare all changes before writing the file.

You are proposing one HSOpt product-formula policy mutation.

The working directory contains:

- candidate.py: the only editable scored policy;
- evaluate.py: the exact fixed evaluator implementation used for scoring;
- problem.json for an absolute-energy lane, or reaction.json plus its referenced
  Hamiltonian JSON files for a difference-energy lane.
- research_context.json: compact Hamiltonian summaries, the exact policy API,
  lane-local proposal history, and the required-read list.
- scratch/: optional disposable notes and test drafts, editable in either stage.
  execute (Code Mode) can read permitted files and run JavaScript data checks.
  Shell/Python execution is unavailable; Python files here are drafts only.

Before editing, successfully read candidate.py, research_context.json, and ALL
benchmark files listed in required_reads. Large raw JSON files may require
paging with read(offset=..., limit=...); inspect the relevant coefficient ranges.
Read those files so the mutation can use the concrete Hamiltonian
coefficients, Pauli labels, term metadata, and evaluator interface. The
evaluator and JSON inputs are read-only. Edit only candidate.py.

You may optionally use an installed skill or browse/search for product-formula
and synthesis ideas before editing. Web research is optional, and search
content is inspiration rather than an evaluator result or permission to change
the benchmark. Do not search for this HSOpt repository, its campaign outputs,
or another baseline initializer. If an optional research tool is unavailable,
continue with the mutation instead of retrying it.

The fixed evaluator compiles every Pauli rotation with CUDA-Q, constructs
the actual compiled product-formula unitary, diagonalizes it, and selects the
lowest-energy eigenphase with resolved weight from the fixed Hartree--Fock
input. The immutable problem uses the branch-safe encoding
U = exp(+i * (H - c0 I) / lambda), where lambda is the nonidentity Pauli
one-norm and problem["target_time"] = 1 / lambda. The evaluator decodes the
principal phase back to Hartree as c0 + lambda * phase.

The score multiplies the directly compiled T count of one product-formula
evolution by the paper's full-evolution QPE query bound at the remaining
chemical-accuracy precision, then by enough independent attempts for 95%
confidence. The confidence calculation includes both the resolved HF peak
weight and the paper's one-half phase-estimation success guarantee. An
absolute-energy lane is feasible only when its decoded energy error is within
chemical accuracy. A reaction lane names one immutable feedback contract:
individual_errors uses sum_s |nu_s delta E_s|, while error_cancellation uses
|sum_s nu_s delta E_s|. Both reaction metrics are reported, but only the named
difference_error_contract determines feasibility and acceptance in that
isolated lane. Small resolved HF peak weight is therefore penalized by a
larger repetition cost; it is not an energy average.

Make one coherent deterministic mutation intended to lower the
confidence-adjusted T cost while remaining accurate. Preserve the
ProductFormula class and its repetitions(), build_step(), and
synthesis_epsilon() methods. build_step() is the scientific definition of the
formula: it may implement PF1, PF2/Strang, a higher-order stationary
composition, grouping, or another deterministic recurrence. The evaluator
infers standard PF1/PF2 labels from the returned rotations; it never trusts a
declared order. synthesis_epsilon() selects the CUDA-Q tolerance for each
returned rotation. Inspect evaluate.py for the exact accepted types, angle
convention, schedule validation, and synthesis-tolerance constraints.

build_step() is called ONCE per species; the returned schedule is repeated
unchanged. Counters cannot alternate successive calls. Put every stage of a
composition inside that one schedule. Each returned alpha implements
exp(+i alpha P), with Rz(-2*alpha). You may choose custom angles independently
by Pauli label, coefficient, support, or species, including negative substeps.
The synthesis tolerance is a separate per-rotation choice. These freedoms do
not change the Hamiltonian or exact energy target.

Use recent rejected proposals and failures to choose a new hypothesis. Do not
repeat an already scored policy or make a formatting-only edit. Prefer a
coherent change that tests a formula, ordering, term-dependent angle, or
precision hypothesis over repeatedly changing one scalar without evidence.
When useful, compare PF1/PF2/higher-order compositions, commuting groups,
term-dependent angle corrections, and mixed synthesis precisions. None is
required to win. In your final text state the hypothesis and changed controls.
Only import math; no generators, private attributes (self._foo), set, getattr,
hasattr, isinstance, or external helpers. Use only the builtins explicitly
listed in research_context.json. Summary-only fields such as nonidentity_one_norm
are NOT keys of problem; derive quantities from problem['terms'] when needed.
Preserve valid Python indentation: four spaces per level, methods inside the
class, method bodies another four spaces deeper. Prefer one write of the COMPLETE
candidate.py file (plain Python, no Markdown fences or diff markers) if replacing
multiple blocks. Preserve all four interface methods and every unchanged section.
Check bracket balance and indentation mentally before your single edit.
The compact API is in research_context.json so reading
the entire evaluate.py is unnecessary; its read output can be truncated.
Use read for local files; webfetch accepts only HTTP(S) URLs.

Do not change any Hamiltonian, evaluator, source data, or benchmark definition.
Do not fabricate an evaluation result. Read the available context as needed.
Call one edit or write tool exactly once. After that tool succeeds, immediately
return a final text response without rewriting the file.
"""
    if context is not None and context.get("shared_protocol"):
        if context["shared_protocol"].get("energy_target") == "ground_branch":
            prompt = prompt.replace(
                "lowest-energy eigenphase with resolved weight from the fixed Hartree--Fock\ninput.",
                "dominant compiled eigenspace sampled from the cached EXACT GROUND STATE\ninput used for tight calibration; error is E_compiled,0 - E_0.",
            ).replace("resolved HF peak", "resolved exact-ground-input peak").replace(
                "Small resolved HF peak weight", "Small resolved ground-input peak weight"
            )
            prompt += (
                "\nGROUND-INPUT SHARED CONTRACT: GPU Lanczos must certify the compiled QPE peak "
                "by its overlap with the cached exact ground vector. It never compares to an "
                "HF-accessible excited-state energy. Ground-state preparation cost is excluded; "
                "the shared formula is scored for dynamics and T95. All cases must pass.\n"
            )
        prompt += (
            "\nSHARED PROTOCOL EXPERIMENT: this ONE candidate.py is evaluated on ALL suite cases. "
            "Optimize the geometric mean of case T95/baseline T95, with case-wise accuracy. "
            "Do not optimize a separate file per molecule. Read suite.json and its fixed baseline "
            "parameters; preserve the empty baseline_parameters placeholder. The host binds it. "
            "Actively propose mixed-precision synthesis schemes, not only global epsilon changes: "
            "coefficient/angle magnitude weighting, diagonal versus off-diagonal terms, Pauli "
            "support, or normalized error-budget weights. These are hypotheses, not guaranteed "
            "bounds; the unchanged CUDA-Q/energy judge decides feasibility. Use Code Mode to "
            "batch permitted reads/calculations across the suite; many direct read turns "
            "would exceed the agent step budget. Return compact features, not all raw JSON.\n"
        )
    if context is not None and context.get("shared_protocol"):
        # This continuation uses the verified preparation session, which has
        # already read the entire context. Repeating it can overflow the model
        # window before OpenCode gets a chance to compact the next tool turn.
        prompt += (
            "\nThe immutable policy contract, whole-suite features and recent trial "
            "history are in research_context.json, already read during preparation "
            "in this session. Re-read that file when needed, including after "
            "automatic compaction; do not discard failed-trial feedback.\n"
        )
    elif context is not None:
        prompt += (
            "\nRequired context and policy contract (immutable):\n"
            + json.dumps(context, sort_keys=True)
            + "\n"
        )
    if feedback is None:
        return prompt
    return (
        prompt
        + "\nHere is the immutable evaluator feedback for this proposal:\n\n"
        + json.dumps(feedback, indent=2, sort_keys=True)
        + "\n"
    )


def agent_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """Keep only evaluator facts that are useful for the next policy mutation."""

    scalar_keys = (
        "status",
        "signed_energy_error_hartree",
        "chemical_accuracy_hartree",
        "observable",
        "problem",
        "reaction",
        "n_qubits",
        "electrons",
        "exact_ground_energy_hartree",
        "compiled_pf_energy_hartree",
        "exact_difference_hartree",
        "compiled_pf_difference_hartree",
        "absolute_energy_error_mha",
        "absolute_difference_error_mha",
        "difference_error_contract",
        "sum_absolute_species_error_mha",
        "contract_error_mha",
        "error_cancellation_fraction",
        "passes_chemical_accuracy",
        "passes_individual_errors_contract",
        "passes_error_cancellation_contract",
        "compiled_evolution_t_count",
        "per_attempt_t_count",
        "per_qpe_run_t_count",
        "confidence_adjusted_t_count",
        "t_count",
        "score",
        "protocol_objective",
        "mean_relative_t95",
        "failed_case_ids",
        "case_metrics",
        "worst_normalized_error",
        "case_count",
        "formula_order",
        "n_steps",
        "rotations_per_step",
        "scheduled_rotation_count",
        "one_step_t_count",
        "qpe_repetitions",
        "qpe_energy_precision_hartree",
        "qpe_phase_precision_radians",
        "qpe_full_evolution_queries",
        "qpe_target_confidence",
        "qpe_achieved_confidence",
        "qpe_success_probability_from_hf",
        "qpe_success_probability_per_attempt",
    )
    result = {key: feedback[key] for key in scalar_keys if key in feedback}
    if isinstance(feedback.get("policy"), dict):
        result["policy"] = feedback["policy"]
    species = feedback.get("species")
    if isinstance(species, dict):
        result["species"] = {
            name: {
                key: row[key]
                for key in (
                    "n_qubits",
                    "electrons",
                    "absolute_energy_error_mha",
                    "signed_reaction_error_contribution_hartree",
                    "absolute_reaction_error_contribution_hartree",
                    "passes_chemical_accuracy",
                    "compiled_evolution_t_count",
                    "per_attempt_t_count",
                    "per_qpe_run_t_count",
                    "confidence_adjusted_t_count",
                    "t_count",
                    "formula_order",
                    "n_steps",
                    "rotations_per_step",
                    "scheduled_rotation_count",
                    "qpe_repetitions",
                    "qpe_energy_precision_hartree",
                    "qpe_phase_precision_radians",
                    "qpe_full_evolution_queries",
                    "qpe_target_confidence",
                    "qpe_achieved_confidence",
                    "qpe_success_probability_from_hf",
                    "qpe_success_probability_per_attempt",
                    "policy",
                )
                if key in row
            }
            for name, row in species.items()
            if isinstance(row, dict)
        }
    return result


def _validate_candidate(path: Path) -> None:
    try:
        validate_policy_source(path)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise RuntimeError(f"candidate violates the ProductFormula contract: {exc}") from exc


def _terminate_sandbox_opencode(sandbox_name: str) -> None:
    """Stop a timed-out OpenCode process before deleting its disposable sandbox."""

    _run(
        [
            "openshell",
            "sandbox",
            "exec",
            "--name",
            sandbox_name,
            "--timeout",
            "15",
            "--",
            "sh",
            "-lc",
            (
                "pkill -TERM -f '[o]pencode(\\.exe)? (run|serve)' 2>/dev/null || true; "
                "sleep 1; "
                "pkill -KILL -f '[o]pencode(\\.exe)? (run|serve)' 2>/dev/null || true"
            ),
        ],
        check=False,
        timeout=20,
    )


def _mutate_once(
    policy_file: Path,
    feedback_file: Path,
    run_dir: Path,
    *,
    kind: str,
    source: Path,
    timeout_seconds: int = 900,
    keep_sandbox: bool = False,
) -> dict[str, Any]:
    """Implement one isolated mutation after the public wrapper sets up logging."""

    policy_file = policy_file.resolve()
    feedback_file = feedback_file.resolve()
    run_dir = run_dir.resolve()
    context_files = sandbox_context_files(kind=kind, source=source)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate = run_dir / "candidate.py"
    shutil.copyfile(policy_file, run_dir / "candidate.before.py")
    shutil.copyfile(policy_file, candidate)
    shutil.copyfile(feedback_file, run_dir / "feedback.json")
    feedback = json.loads(feedback_file.read_text(encoding="utf-8"))
    prompt_feedback = agent_feedback(feedback)
    _write_json(run_dir / "agent_feedback.json", prompt_feedback)
    required_reads = [
        "candidate.py",
        "research_context.json",
        *[
            relative.as_posix()
            for _, relative in context_files
            if relative.as_posix() not in {"evaluate.py", "protocol_evaluate.py"}
        ],
    ]
    if kind == "protocol":
        # Read the entire suite summary, then representative raw systems.
        # All raw inputs remain available; acceptance still covers every case.
        suite = json.loads(source.read_text())
        def case_size(case):
            definition = json.loads((source.parent / case["source"]).read_text())
            if suite["kind"] == "absolute":
                return (definition["n_qubits"], len(definition["terms"]))
            systems = [json.loads((source.parent / Path(case["source"]).parent / row["problem"]).read_text())
                       for row in definition["terms"]]
            return (max(p["n_qubits"] for p in systems), sum(len(p["terms"]) for p in systems))
        ordered = sorted(suite["cases"], key=case_size)
        selected = [ordered[i] for i in sorted({0, len(ordered)//2, len(ordered)-1})]
        prefixes = {str(Path(case["source"]).parent) + "/" for case in selected}
        required_reads = [p for p in required_reads if p in {
            "candidate.py", "research_context.json", "suite.json"
        } or any(p.startswith(prefix) for prefix in prefixes)]
    if any(relative.as_posix() == "research_context.json" for _, relative in context_files):
        raise ValueError("benchmark path collides with research_context.json")
    research_context = {
        "mutation_protocol_version": MUTATION_PROTOCOL_VERSION,
        "required_reads": required_reads,
        "policy_contract": POLICY_CONTRACT,
        "benchmark": benchmark_summary(source, kind),
        "history": lane_history(policy_file.parent, run_dir),
    }
    if kind == "protocol":
        energy_target = research_context["benchmark"].get("energy_target", "hf_accessible")
        research_context["benchmark"] = _compact_protocol_benchmark(research_context["benchmark"])
        research_context["mutation_protocol_version"] += "_shared_representative_raw_reads_v3"
        research_context["mutation_protocol_version"] += "_context_reuse_v1"
        if energy_target == "ground_branch":
            research_context["mutation_protocol_version"] += "_shared_ground_input_v1"
            research_context["policy_contract"] = {
                **POLICY_CONTRACT,
                "objective": (
                    "Minimize geometric-mean baseline-relative T95 across ALL molecules, "
                    "with |E_compiled,0 - E_0| below chemical accuracy for EVERY molecule. "
                    "The initial state is the cached exact ground vector used for tight "
                    "calibration. Select its dominant compiled QPE eigenspace; do not use HF. "
                    "Include its probability in the 95% confidence cost. Ideal ground-state "
                    "preparation cost is excluded. Only feasible strict aggregate gains count."
                ),
            }
        research_context["shared_protocol"] = {
            "raw_context_strategy": "Read whole-suite features and small/median/large raw cases; all raw cases remain readable and all cases must be evaluated.",
            "required_raw_case_ids": [case["id"] for case in selected],
            "raw_input_paths": "Absolute: /sandbox/work/inputs/{case_id}/problem.json. Reaction: /sandbox/work/inputs/{case_id}/reaction.json; it lists its species paths.",
            "energy_target": energy_target,
            "objective": "geometric_mean_baseline_relative_t95",
            "accuracy": "Every selected molecule/reaction must pass its own fixed energy contract.",
            "baseline_binding": "Preserve baseline_parameters = {}. The host supplies each case's sealed (r, epsilon, beta) starting values. Change self.r/angles/tolerance rules, not this table.",
            "mixed_precision": "Explicitly explore mixed-precision rotation synthesis: coefficient-, Pauli-support-, angle-, species-feature-, and error-budget-weighted tolerances. Constant epsilon is only the initial policy. Test coherent schemes using synthesis_epsilon(term, angle); exact compiled-energy feasibility remains mandatory for every case.",
        }
    if kind == "protocol":
        _write_protocol_context(run_dir / "research_context.json", research_context)
    else:
        _write_json(run_dir / "research_context.json", research_context)
    context_files.append(
        (run_dir / "research_context.json", PurePosixPath("research_context.json"))
    )
    prompt = mutation_prompt(prompt_feedback, context=research_context)
    _write_json(
        run_dir / "mutation_protocol.json",
        {
            "version": research_context["mutation_protocol_version"],
            "sandbox_transport": "bundle-v2",
            "blocked_write_handling": "permission-denial-recovery-v1",
            "agent_action_contract": "candidate-once-scratch-code-v1",
            "shared_protocol": research_context.get("shared_protocol"),
            "model": NEMOTRON_MODEL,
            "required_reads": required_reads,
            "context_sha256": _sha256(run_dir / "research_context.json"),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "model_output_limit": MODEL_OUTPUT_LIMIT,
            "model_context_limit": MODEL_CONTEXT_LIMIT,
            "automatic_compaction": True,
            "preparation_handoff": "required-read-barrier-v1" if os.environ.get("HSOPT_PREPARATION_READ_BARRIER") == "1" else "model_turn_completion",
            "editing_handoff": "first-completed-candidate-edit-v1",
        },
    )
    (run_dir / "opencode.json").write_text(
        json.dumps(opencode_config(), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "opencode.preparation.json", opencode_config(preparation=True))
    preparation_prompt = (
        "Read-only preparation. FIRST read /sandbox/work/candidate.py, then "
        "/sandbox/work/research_context.json, then each remaining required FILE. "
        "Do not read directories or append filenames to these file paths. "
        "Editing is disabled until the host verifies all reads. "
        "Successfully read each of these files before returning: "
        + json.dumps(required_reads)
        + ". Inspect research_context.json for the exact hook contract, Hamiltonian "
        "summaries, and previous proposals. Choose a new scientific hypothesis. "
        "Do not edit candidate.py yet. Scratch notes and Code Mode data checks are "
        "allowed. End with a brief explanation of the planned mutation."
    )
    if kind == "protocol":
        preparation_prompt += (
            " This is one shared protocol across the WHOLE suite. "
            "Use the ordinary read tool. Read research_context.json with limit=2000 "
            "so the complete compact suite is visible, and use {path, limit:80} "
            "for the required raw Hamiltonian FILES. "
            "The required raw examples span system sizes; research_context.json has "
            "features for EVERY case. All other Hamiltonian files remain readable "
            "when useful, and EVERY case will be evaluated. "
            "execute accepts JavaScript ONLY, never Python. There is no import, open, "
            "os, subprocess, filesystem API, fetch, or HTTP file endpoint. "
            "Do not use execute for these required reads. Use it only for optional "
            "small JavaScript calculations from data you already read. "
            "Explicitly plan a mixed-precision rotation-synthesis hypothesis."
        )
    (run_dir / "preparation_prompt.md").write_text(preparation_prompt, encoding="utf-8")
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    _write_json(
        run_dir / "sandbox_context.json",
        {
            "schema": 1,
            "mutation_protocol_version": MUTATION_PROTOCOL_VERSION,
            "kind": kind,
            "files": [
                {
                    "sandbox_path": relative.as_posix(),
                    "sha256": _sha256(local),
                    "role": "fixed_evaluator"
                    if relative == PurePosixPath("evaluate.py")
                    else "benchmark_input",
                }
                for local, relative in context_files
            ],
        },
    )
    if not HSOPT_SKILL_PATH.is_file():
        raise RuntimeError(f"OpenCode hsopt skill is unavailable: {HSOPT_SKILL_PATH}")
    sandbox_skill = run_dir / "hsopt.SKILL.md"
    shutil.copyfile(HSOPT_SKILL_PATH, sandbox_skill)
    sandbox_file_list_shim = run_dir / "rg"
    _write_opencode_file_list_shim(sandbox_file_list_shim)

    sandbox_name = f"hsopt-{uuid.uuid4().hex[:12]}"
    if len(sandbox_name) > 19:
        raise AssertionError("OpenShell sandbox names may not exceed 19 characters")
    created = False
    audit: dict[str, Any] | None = None
    try:
        with _mutation_phase(run_dir, "setup_wait_and_setup"), _sandbox_setup_gate(run_dir):
            with _mutation_phase(run_dir, "sandbox_setup"):
                # The owned name may exist even if create fails during registration.
                created = True
                _run(
                    [
                        "openshell",
                        "sandbox",
                        "create",
                        "--name",
                        sandbox_name,
                        "--no-auto-providers",
                        "--provider",
                        NVIDIA_PROVIDER,
                        "--no-tty",
                        "--detach",
                        *(
                            ["--from", os.environ["HSOPT_SANDBOX_IMAGE"]]
                            if os.environ.get("HSOPT_SANDBOX_IMAGE")
                            else []
                        ),
                        *(
                            ["--policy", os.environ["HSOPT_SANDBOX_POLICY"]]
                            if os.environ.get("HSOPT_SANDBOX_POLICY")
                            else []
                        ),
                        "--",
                        "sleep",
                        "infinity",
                    ],
                    timeout=timeout_seconds,
                )
                _capture_runtime_versions(sandbox_name, run_dir)
                _write_json(
                    run_dir / "research_network.json",
                    _configure_research_network(sandbox_name),
                )
                archive = _build_sandbox_bundle(run_dir, candidate, context_files)
                try:
                    _run(
                        [
                            "openshell",
                            "sandbox",
                            "upload",
                            "--no-git-ignore",
                            sandbox_name,
                            str(archive),
                            "/sandbox/hsopt-bootstrap.tar",
                        ],
                        timeout=timeout_seconds,
                    )
                    _run_setup_command(
                        sandbox_name,
                        [
                            "tar",
                            "-xf",
                            "/sandbox/hsopt-bootstrap.tar",
                            "-C",
                            "/sandbox",
                        ],
                        run_dir,
                    )
                finally:
                    archive.unlink(missing_ok=True)
        command = [
            "openshell",
            "sandbox",
            "exec",
            "--name",
            sandbox_name,
            "--workdir",
            "/sandbox/work",
            "--timeout",
            str(timeout_seconds),
            "--env",
            "OPENCODE_CONFIG=/sandbox/opencode.json",
            "--env",
            "OPENCODE_DISABLE_AUTOUPDATE=true",
            "--env",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS=true",
            "--env",
            "OPENCODE_DISABLE_MODELS_FETCH=true",
            "--env",
            "OPENCODE_DISABLE_LSP_DOWNLOAD=true",
            "--env",
            "OPENCODE_ENABLE_EXA=1",
            "--env",
            "PATH=/sandbox/work/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--",
            "opencode",
            "run",
            "--standalone",
            "--format",
            "json",
            "--model",
            OPENCODE_MODEL,
            "--agent",
            OPENCODE_AGENT,
            prompt,
        ]
        stopped_after_edit = False
        try:
            deadline = time.monotonic() + timeout_seconds
            with _mutation_phase(run_dir, "preparation_agent"):
                session = _prepare_mutation_context(
                    command,
                    run_dir=run_dir,
                    required_reads=required_reads,
                    prompt=preparation_prompt,
                    deadline=deadline,
                )
            with _mutation_phase(run_dir, "preparation_integrity"):
                _verify_preparation_candidate(sandbox_name, run_dir, deadline=deadline)
            # OpenCode also discovers /sandbox/opencode.json as project config.
            # Replace that same file after verified reads, just as the original
            # transport did; a second config filename can retain the deny rules.
            with _mutation_phase(run_dir, "editing_config_switch"):
                if os.environ.get("HSOPT_PREPARATION_READ_BARRIER") == "1":
                    _terminate_sandbox_opencode(sandbox_name)
                _run_setup_command(sandbox_name, [
                    "cp", "-f", "/sandbox/opencode.mutation.json", "/sandbox/opencode.json",
                ], run_dir)
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise RuntimeError("Mutation time budget exhausted during preparation")
            continuation = [*command[:-1], "--session", session, prompt]
            continuation[continuation.index("--timeout") + 1] = str(remaining)
            with _mutation_phase(run_dir, "editing_agent"):
                returncode, stopped_after_edit = _run_logged(
                    continuation,
                    stdout_path=run_dir / "opencode.jsonl",
                    stderr_path=run_dir / "opencode.stderr",
                    timeout=remaining + 30,
                    completed_edit_grace_seconds=0,
                    append=True,
                )
        finally:
            if stopped_after_edit:
                _terminate_sandbox_opencode(sandbox_name)
            with _mutation_phase(run_dir, "telemetry_collection"):
                _capture_openshell_logs(sandbox_name, run_dir / "openshell.log")
                audit = write_agent_audit(run_dir)
        _raise_inference_error(run_dir / "opencode.jsonl")
        if returncode != 0 and not stopped_after_edit:
            raise RuntimeError(f"OpenCode exited with status {returncode}")
        with _mutation_phase(run_dir, "candidate_download"):
            _download_candidate(sandbox_name, run_dir, timeout_seconds=timeout_seconds)
    finally:
        if created and not (run_dir / "agent_audit.json").is_file():
            _capture_openshell_logs(sandbox_name, run_dir / "openshell.log")
            write_agent_audit(run_dir)
        if created and not keep_sandbox:
            cleanup_errors = []
            for cleanup in (
                lambda: _run(
                    ["openshell", "sandbox", "delete", sandbox_name],
                    check=False,
                    timeout=45,
                ),
            ):
                try:
                    with _mutation_phase(run_dir, "sandbox_cleanup"):
                        cleanup()
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}"[:500])
            if cleanup_errors:
                _write_json(run_dir / "cleanup_errors.json", {"errors": cleanup_errors})

    _validate_agent_action_contract(run_dir / "opencode.jsonl", required_reads=required_reads)
    _validate_candidate(candidate)
    duplicate = duplicate_policy(candidate, policy_file)
    if duplicate:
        raise RuntimeError(
            f"Duplicate policy: matches {duplicate}; propose a new scientific change"
        )
    return {
        "status": "mutated",
        "sandbox": sandbox_name,
        "model": NEMOTRON_MODEL,
        "inference_route": NVIDIA_BASE_URL,
        "candidate": str(candidate),
        "run_dir": str(run_dir),
        "agent_audit": audit,
        "agent_completion": ("completed_edit_observed" if stopped_after_edit else "normal_exit"),
        "mutation_protocol_version": research_context["mutation_protocol_version"],
    }


def mutate(
    policy_file: Path,
    feedback_file: Path,
    run_dir: Path,
    *,
    kind: str,
    source: Path,
    timeout_seconds: int = 900,
    keep_sandbox: bool = False,
) -> dict[str, Any]:
    """Create one external candidate and always persist its final outcome."""

    run_dir = run_dir.resolve()
    may_record_failure = not run_dir.exists() or not any(run_dir.iterdir())
    started_at = datetime.now(UTC).isoformat()
    try:
        result = _mutate_once(
            policy_file,
            feedback_file,
            run_dir,
            kind=kind,
            source=source,
            timeout_seconds=timeout_seconds,
            keep_sandbox=keep_sandbox,
        )
    except Exception as exc:
        if may_record_failure:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                run_dir / "mutation_result.json",
                {
                    "schema": 1,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                    "stderr": re.sub(
                        r"(?:nvapi-|sk-)[A-Za-z0-9_-]+|Bearer\s+\S+",
                        "[REDACTED]",
                        str(getattr(exc, "stderr", "") or ""),
                    )[:1000],
                    "retryable": bool(getattr(exc, "retryable", True)),
                    "error_category": str(getattr(exc, "category", "mutation_contract")),
                    "mutation_protocol_version": MUTATION_PROTOCOL_VERSION,
                    "agent_audit": (
                        "agent_audit.json" if (run_dir / "agent_audit.json").is_file() else None
                    ),
                },
            )
            # Shared protocol orchestration records its own failure once.
            # Its numeric iteration folder must not count the same error twice.
            if run_dir.name.isdigit() and kind != "protocol":
                record_mutation_failure(
                    policy_file.resolve().parent, exc, iteration=int(run_dir.name)
                )
        raise
    _write_json(
        run_dir / "mutation_result.json",
        {
            "schema": 1,
            "status": "mutated",
            "started_at": started_at,
            "mutation_protocol_version": result.get("mutation_protocol_version", MUTATION_PROTOCOL_VERSION),
            "finished_at": datetime.now(UTC).isoformat(),
            "model": result["model"],
            "candidate": "candidate.py",
            "agent_audit": "agent_audit.json",
            "agent_completion": result["agent_completion"],
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "configure",
        help="configure the host-side OpenShell NVIDIA inference provider",
    )

    mutate_parser = subparsers.add_parser(
        "mutate",
        help="run one isolated OpenCode mutation and persist its audit logs",
    )
    mutate_parser.add_argument("--policy-file", type=Path, required=True)
    mutate_parser.add_argument("--feedback-file", type=Path, required=True)
    mutate_parser.add_argument("--run-dir", type=Path, required=True)
    mutate_parser.add_argument("--kind", choices=("absolute", "difference", "protocol"), required=True)
    mutate_parser.add_argument("--source", type=Path, required=True)
    mutate_parser.add_argument("--timeout-seconds", type=int, default=900)
    mutate_parser.add_argument("--keep-sandbox", action="store_true")

    audit_parser = subparsers.add_parser(
        "audit",
        help="rebuild redacted tool and OpenShell metrics from one mutation run",
    )
    audit_parser.add_argument("--run-dir", type=Path, required=True)
    resume_parser = subparsers.add_parser(
        "resume", help="resume mutations after repairing a recorded runtime failure"
    )
    resume_parser.add_argument(
        "--path", type=Path, required=True, help="campaign root or lane directory"
    )
    resume_parser.add_argument("--reason", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "configure":
        result = configure_inference()
    elif args.command == "mutate":
        result = mutate(
            args.policy_file,
            args.feedback_file,
            args.run_dir,
            kind=args.kind,
            source=args.source,
            timeout_seconds=args.timeout_seconds,
            keep_sandbox=args.keep_sandbox,
        )
    elif args.command == "resume":
        result = resume_mutations(args.path.resolve(), reason=args.reason)
    else:
        result = write_agent_audit(args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
