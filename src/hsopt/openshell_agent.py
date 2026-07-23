"""Run an isolated OpenCode mutation through OpenShell and Nemotron 3 Super."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


NVIDIA_PROVIDER = "hsopt-nvidia"
NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nvidia/nemotron-3-super-v3"
OPENCODE_MODEL = "openshell/nemotron-3-super-v3"
OPENCODE_AGENT = "hsopt-mutation"
ONLINE_TOOLS = frozenset({"webfetch", "websearch", "codesearch"})


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_logged(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    completed_edit_grace_seconds: float | None = None,
) -> tuple[int, bool]:
    """Run a command with logs visible to external monitors while it is active."""

    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            list(command),
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
        deadline = time.monotonic() + timeout
        edit_completed_at: float | None = None
        while True:
            returncode = process.poll()
            if returncode is not None:
                return returncode, False
            now = time.monotonic()
            if (
                completed_edit_grace_seconds is not None
                and edit_completed_at is None
                and _has_completed_candidate_edit(stdout_path)
            ):
                edit_completed_at = now
            if (
                edit_completed_at is not None
                and now - edit_completed_at >= completed_edit_grace_seconds
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


def _safe_tool_target(tool: str, payload: Any) -> str | None:
    """Keep an audit-useful target without copying prompts or file contents."""

    if not isinstance(payload, dict):
        return None
    for key in ("filePath", "path", "url", "query"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            continue
        if key in {"filePath", "path"}:
            return Path(value).name
        return value[:200]
    if tool == "edit":
        return "candidate.py"
    return None


def summarize_opencode_events(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize OpenCode NDJSON into counters and a redacted action timeline."""

    tokens: Counter[str] = Counter()
    action_order: list[str] = []
    actions_by_call: dict[str, dict[str, Any]] = {}
    session_ids: set[str] = set()
    inference_steps = 0
    errors = 0
    if not path.is_file():
        return {
            "schema": 1,
            "tool_calls": 0,
            "by_tool": {},
            "by_status": {},
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
        actions_by_call[action_key] = {
            "timestamp": event.get("timestamp") or previous.get("timestamp"),
            "session_id": session_id or previous.get("session_id"),
            "call_id": call_id or None,
            "tool": tool,
            "status": status,
            "duration_ms": duration_ms,
            "target": target,
            "online": tool in ONLINE_TOOLS,
        }

    actions = [actions_by_call[key] for key in action_order]
    by_tool = Counter(str(action["tool"]) for action in actions)
    by_status = Counter(str(action["status"]) for action in actions)
    metrics = {
        "schema": 1,
        "tool_calls": sum(by_tool.values()),
        "by_tool": dict(sorted(by_tool.items())),
        "by_status": dict(sorted(by_status.items())),
        "online_tool_calls": sum(by_tool[tool] for tool in ONLINE_TOOLS),
        "web_searches": by_tool["websearch"],
        "inference_steps": inference_steps,
        "errors": errors,
        "tokens": dict(sorted(tokens.items())),
        "sessions": sorted(session_ids),
    }
    return metrics, actions


def _has_completed_candidate_edit(path: Path) -> bool:
    """Return whether OpenCode durably reported a completed candidate mutation."""

    _, actions = summarize_opencode_events(path)
    return any(
        action["tool"] in {"edit", "write", "apply_patch"}
        and action["status"] == "completed"
        and action["target"] in {None, "candidate.py"}
        for action in actions
    )


_OCSF_RE = re.compile(r"\bOCSF\s+(?P<class>[A-Z]+):(?P<activity>[A-Z_]+)")


def summarize_openshell_log(path: Path) -> dict[str, Any]:
    """Count OpenShell security events without copying credentials or payloads."""

    classes: Counter[str] = Counter()
    activities: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
        "ocsf_events": sum(classes.values()),
        "by_class": dict(sorted(classes.items())),
        "by_activity": dict(sorted(activities.items())),
        "by_decision": dict(sorted(decisions.items())),
        "network_events": classes["NET"] + classes["HTTP"] + classes["SSH"],
        "process_events": classes["PROC"],
        "denied_events": decisions["denied"] + decisions["blocked"],
    }


def _capture_openshell_logs(sandbox_name: str, destination: Path) -> None:
    """Copy the sandbox's rotating audit logs before the sandbox is deleted."""

    completed = _run(
        [
            "openshell",
            "sandbox",
            "exec",
            "--name",
            sandbox_name,
            "--timeout",
            "30",
            "--",
            "sh",
            "-lc",
            (
                "for file in /var/log/openshell*.log; do "
                "[ -f \"$file\" ] && { printf '\\n== %s ==\\n' \"$file\"; cat \"$file\"; }; "
                "done"
            ),
        ],
        check=False,
        timeout=45,
    )
    destination.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write("\n== capture stderr ==\n")
            stream.write(completed.stderr)


def write_agent_audit(run_dir: Path) -> dict[str, Any]:
    """Persist durable, redacted metrics beside the raw agent/sandbox logs."""

    opencode_metrics, actions = summarize_opencode_events(run_dir / "opencode.jsonl")
    openshell_metrics = summarize_openshell_log(run_dir / "openshell.log")
    _write_json(run_dir / "opencode_metrics.json", opencode_metrics)
    _write_json(run_dir / "openshell_metrics.json", openshell_metrics)
    with (run_dir / "agent_actions.jsonl").open("w", encoding="utf-8") as stream:
        for action in actions:
            stream.write(json.dumps(action, sort_keys=True) + "\n")
    audit = {
        "schema": 1,
        "opencode": opencode_metrics,
        "openshell": openshell_metrics,
        "action_log": "agent_actions.jsonl",
        "raw_opencode_log": "opencode.jsonl",
        "raw_openshell_log": "openshell.log",
    }
    _write_json(run_dir / "agent_audit.json", audit)
    return audit


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
                "nvidia",
                "--credential",
                "NVIDIA_API_KEY",
                "--config",
                f"NVIDIA_BASE_URL={NVIDIA_BASE_URL}",
            ]
        )
    _run(
        [
            "openshell",
            "inference",
            "set",
            "--provider",
            NVIDIA_PROVIDER,
            "--model",
            NEMOTRON_MODEL,
            "--timeout",
            "300",
        ]
    )
    return {
        "status": "configured",
        "provider": NVIDIA_PROVIDER,
        "model": NEMOTRON_MODEL,
        "route": "https://inference.local/v1",
    }


def opencode_config() -> dict[str, Any]:
    """Return an edit-only OpenCode config for one isolated candidate file."""

    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "enabled_providers": ["openshell"],
        "model": OPENCODE_MODEL,
        "provider": {
            "openshell": {
                "name": "OpenShell managed inference",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": "https://inference.local/v1",
                    "apiKey": "unused",
                },
                "models": {
                    "nemotron-3-super-v3": {
                        "name": "NVIDIA Nemotron 3 Super v3",
                    }
                },
            }
        },
        "agent": {
            OPENCODE_AGENT: {
                "description": "Mutate only the isolated HSOpt candidate.py policy.",
                "mode": "primary",
                "model": OPENCODE_MODEL,
                "temperature": 1.0,
                "top_p": 0.95,
                # Bound the model/tool loop. A successful mutation needs one read,
                # one edit, and a final response; five steps leave recovery room
                # without allowing repeated rewrites to run until the wall clock
                # timeout.
                "steps": 5,
                "prompt": (
                    "Perform exactly one candidate.py policy mutation. Read the "
                    "file at most once, call one edit or write tool exactly once, "
                    "then immediately return a final text response without calling "
                    "another tool. The only supported term_order_strategy values "
                    "are input, lexicographic, coefficient_desc, coefficient_asc, "
                    "support_desc, and custom."
                ),
                "permission": {
                    "*": "deny",
                    "bash": "deny",
                    "glob": "deny",
                    "grep": "deny",
                    "list": "deny",
                    "lsp": "deny",
                    "question": "deny",
                    "read": {
                        "*": "deny",
                        "candidate.py": "allow",
                        "/sandbox/work/candidate.py": "allow",
                    },
                    # OpenCode's edit permission gates edit, write, and patch.
                    # The OpenShell sandbox contains only the disposable candidate
                    # worktree, and only candidate.py is downloaded afterwards.
                    "edit": "allow",
                    "skill": "deny",
                    "task": "deny",
                    "todowrite": "deny",
                    "webfetch": "deny",
                    "websearch": "deny",
                },
            }
        },
    }


def mutation_prompt(feedback: dict[str, Any] | None = None) -> str:
    prompt = """You are proposing one HSOpt product-formula policy mutation.

Read and edit only candidate.py.

The fixed evaluator compiles every Pauli rotation with PyGridSynth, constructs
the actual compiled product-formula unitary, diagonalizes it, identifies the
eigenstate connected to a normalized deterministic-anchor projection onto the
complete exact ground eigenspace, and measures its exact eigenphase energy
error. The anchor is HF when it has ground support, otherwise the lowest
resolved fixed-electron determinant. The score is the exact compiled T count
only when the candidate is within chemical accuracy. HF overlap is diagnostic
QPE success probability, not an energy average.

Make one coherent deterministic mutation intended to lower T count while
remaining accurate. Preserve get_method_config(problem) and
choose_synthesis_epsilon(term, alpha, step_index, config). Do not change any
Hamiltonian, evaluator, source data, or benchmark definition. Do not fabricate
an evaluation result. Read candidate.py at most once. Call one edit or write
tool exactly once. After that tool succeeds, immediately return a final text
response without reading or rewriting the file. If changing term_order_strategy,
use only input, lexicographic, coefficient_desc, coefficient_asc, support_desc,
or custom. A custom order_product_formula_terms implementation must preserve
every Pauli term exactly once.
"""
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
        "passes_chemical_accuracy",
        "t_count",
        "score",
        "formula_order",
        "n_steps",
        "rotations_per_step",
        "scheduled_rotation_count",
        "one_step_t_count",
        "selected_ground_state_overlap",
        "qpe_success_probability_from_hf",
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
                    "passes_chemical_accuracy",
                    "t_count",
                    "formula_order",
                    "n_steps",
                    "rotations_per_step",
                    "scheduled_rotation_count",
                    "qpe_success_probability_from_hf",
                    "policy",
                )
                if key in row
            }
            for name, row in species.items()
            if isinstance(row, dict)
        }
    return result


def _validate_candidate(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = {"get_method_config", "choose_synthesis_epsilon"}
    missing = sorted(required - functions)
    if missing:
        raise RuntimeError(f"candidate is missing required functions: {missing}")


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
                "pkill -TERM -f '[o]pencode run' 2>/dev/null || true; "
                "sleep 1; "
                "pkill -KILL -f '[o]pencode run' 2>/dev/null || true"
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
    timeout_seconds: int = 900,
    keep_sandbox: bool = False,
) -> dict[str, Any]:
    """Implement one isolated mutation after the public wrapper sets up logging."""

    policy_file = policy_file.resolve()
    feedback_file = feedback_file.resolve()
    run_dir = run_dir.resolve()
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
    prompt = mutation_prompt(prompt_feedback)
    (run_dir / "opencode.json").write_text(
        json.dumps(opencode_config(), indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    sandbox_name = f"hsopt-{uuid.uuid4().hex[:12]}"
    if len(sandbox_name) > 19:
        raise AssertionError("OpenShell sandbox names may not exceed 19 characters")
    created = False
    audit: dict[str, Any] | None = None
    try:
        _run(
            [
                "openshell",
                "sandbox",
                "create",
                "--name",
                sandbox_name,
                "--no-auto-providers",
                "--no-tty",
                "--",
                "sh",
                "-lc",
                "true",
            ],
            timeout=timeout_seconds,
        )
        created = True
        _run(
            [
                "openshell",
                "sandbox",
                "upload",
                sandbox_name,
                str(candidate),
                "/sandbox/work/candidate.py",
            ],
            timeout=timeout_seconds,
        )
        _run(
            [
                "openshell",
                "sandbox",
                "upload",
                sandbox_name,
                str(run_dir / "opencode.json"),
                "/sandbox/opencode.json",
            ],
            timeout=timeout_seconds,
        )
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
            "--",
            "opencode",
            "run",
            "--format",
            "json",
            "--model",
            OPENCODE_MODEL,
            "--agent",
            OPENCODE_AGENT,
            "--dir",
            "/sandbox/work",
            prompt,
        ]
        stopped_after_edit = False
        try:
            returncode, stopped_after_edit = _run_logged(
                command,
                stdout_path=run_dir / "opencode.jsonl",
                stderr_path=run_dir / "opencode.stderr",
                timeout=timeout_seconds + 30,
                completed_edit_grace_seconds=10,
            )
        finally:
            if stopped_after_edit:
                _terminate_sandbox_opencode(sandbox_name)
            _capture_openshell_logs(sandbox_name, run_dir / "openshell.log")
            audit = write_agent_audit(run_dir)
        if returncode != 0 and not stopped_after_edit:
            raise RuntimeError(f"OpenCode exited with status {returncode}")
        _run(
            [
                "openshell",
                "sandbox",
                "download",
                sandbox_name,
                "/sandbox/work/candidate.py",
                str(candidate),
            ],
            timeout=timeout_seconds,
        )
    finally:
        if created and not keep_sandbox:
            _terminate_sandbox_opencode(sandbox_name)
            _run(
                ["openshell", "sandbox", "delete", sandbox_name],
                check=False,
                timeout=45,
            )

    _validate_candidate(candidate)
    if candidate.read_bytes() == (run_dir / "candidate.before.py").read_bytes():
        raise RuntimeError("OpenCode completed without changing candidate.py")
    return {
        "status": "mutated",
        "sandbox": sandbox_name,
        "model": NEMOTRON_MODEL,
        "inference_route": "https://inference.local/v1",
        "candidate": str(candidate),
        "run_dir": str(run_dir),
        "agent_audit": audit,
        "agent_completion": (
            "completed_edit_observed" if stopped_after_edit else "normal_exit"
        ),
    }


def mutate(
    policy_file: Path,
    feedback_file: Path,
    run_dir: Path,
    *,
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
                    "agent_audit": (
                        "agent_audit.json"
                        if (run_dir / "agent_audit.json").is_file()
                        else None
                    ),
                },
            )
        raise
    _write_json(
        run_dir / "mutation_result.json",
        {
            "schema": 1,
            "status": "mutated",
            "started_at": started_at,
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
    subparsers.add_parser("configure")

    mutate_parser = subparsers.add_parser("mutate")
    mutate_parser.add_argument("--policy-file", type=Path, required=True)
    mutate_parser.add_argument("--feedback-file", type=Path, required=True)
    mutate_parser.add_argument("--run-dir", type=Path, required=True)
    mutate_parser.add_argument("--timeout-seconds", type=int, default=900)
    mutate_parser.add_argument("--keep-sandbox", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "configure":
        result = configure_inference()
    else:
        result = mutate(
            args.policy_file,
            args.feedback_file,
            args.run_dir,
            timeout_seconds=args.timeout_seconds,
            keep_sandbox=args.keep_sandbox,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
