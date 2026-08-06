"""Configure OpenShell, isolate OpenCode mutations, and record agent telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .compiled_unitary import validate_policy_source


NVIDIA_PROVIDER = "hsopt-nvidia"
NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nvidia/nemotron-3-super-v3"
OPENCODE_MODEL = "openshell/nemotron-3-super-v3"
OPENCODE_AGENT = "hsopt-mutation"
ONLINE_TOOLS = frozenset({"webfetch", "websearch", "codesearch"})
RESEARCH_CONTAMINATION_MARKERS = ("bestquark", "hsopt", "initial_pf")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
HSOPT_SKILL_PATH = REPOSITORY_ROOT / "skills" / "hsopt" / "SKILL.md"
EVALUATOR_SOURCE_PATH = Path(__file__).resolve().with_name("compiled_unitary.py")
SANDBOX_CONTEXT_ROOT = PurePosixPath("/sandbox/work")
READABLE_CONTEXT_FILES = frozenset({"candidate.py", "evaluate.py", "problem.json", "reaction.json"})
OPENCODE_NETWORK_BINARIES = (
    "/usr/lib/node_modules/opencode-ai/bin/.opencode",
    "/usr/local/lib/node_modules/opencode-ai/bin/.opencode",
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
    else:
        raise ValueError("kind must be 'absolute' or 'difference'")
    return files


def _safe_tool_target(tool: str, payload: Any) -> str | None:
    """Keep an audit-useful target without copying prompts or file contents."""

    if not isinstance(payload, dict):
        return None
    for key in ("filePath", "path", "url", "query"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
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


def _validate_agent_action_contract(path: Path) -> None:
    """Reject a sandbox turn that did anything outside the one-file contract."""

    _, actions = summarize_opencode_events(path)
    allowed_tools = {
        "read",
        "edit",
        "write",
        "apply_patch",
        "patch",
        "skill",
        *ONLINE_TOOLS,
    }
    mutation_tools = {"edit", "write", "apply_patch", "patch"}
    mutation_actions = []
    for action in actions:
        tool = str(action["tool"])
        if tool not in allowed_tools:
            raise RuntimeError(f"OpenCode attempted forbidden tool {tool!r}")
        if tool == "read":
            target = str(action["target"])
            safe_json = (
                target.endswith(".json")
                and not target.startswith(("outside:", "/", "../"))
                and all(part not in {"", ".", ".."} for part in PurePosixPath(target).parts)
            )
            if target not in READABLE_CONTEXT_FILES and not safe_json:
                raise RuntimeError("OpenCode attempted to read outside the mutation context")
        if tool in ONLINE_TOOLS and action["target"] is not None:
            research_target = str(action["target"]).casefold()
            if any(marker in research_target for marker in RESEARCH_CONTAMINATION_MARKERS):
                raise RuntimeError("OpenCode research attempted to inspect HSOpt baseline data")
        if tool in mutation_tools:
            mutation_actions.append(action)
            if action["target"] != "candidate.py":
                raise RuntimeError("OpenCode attempted to mutate outside candidate.py")
    completed_mutations = [action for action in mutation_actions if action["status"] == "completed"]
    if len(mutation_actions) != 1 or len(completed_mutations) != 1:
        raise RuntimeError("OpenCode must complete exactly one candidate.py mutation")


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
                '[ -f "$file" ] && { printf \'\\n== %s ==\\n\' "$file"; cat "$file"; }; '
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
                # Leave room for repeated context reads and optional research
                # before the single edit while still bounding the model/tool loop.
                "steps": 20,
                "prompt": (
                    "Perform exactly one candidate.py policy mutation. Before "
                    "editing, you may optionally use skill, websearch, webfetch, "
                    "or codesearch for product-formula research. Read candidate.py, "
                    "the read-only evaluate.py implementation, and the read-only "
                    "benchmark JSON files as needed. Call one edit or write tool "
                    "exactly once, then "
                    "immediately return a final text response without calling "
                    "another tool. ProductFormula.build_step() is the "
                    "authoritative recurrence and may return any deterministic "
                    "stationary Pauli-rotation schedule. Do not search for this "
                    "HSOpt repository, campaign, or any sibling baseline policy."
                ),
                "permission": {
                    # OpenCode 1.2.18 currently rejects granular edit allows
                    # after merging the agent policy. The disposable sandbox
                    # therefore permits its edit primitive, then enforces the
                    # one-file contract against the raw action log before any
                    # candidate is accepted or returned to the host.
                    "*": "allow",
                    "bash": "deny",
                    "batch": "deny",
                    "doom_loop": "deny",
                    "external_directory": {
                        "*": "deny",
                        "/sandbox/work/candidate.py": "allow",
                    },
                    "glob": "deny",
                    "grep": "deny",
                    "list": "deny",
                    "lsp": "deny",
                    "question": "deny",
                    "read": {
                        "*": "deny",
                        "candidate.py": "allow",
                        "/sandbox/work/candidate.py": "allow",
                        "evaluate.py": "allow",
                        "/sandbox/work/evaluate.py": "allow",
                        "problem.json": "allow",
                        "/sandbox/work/problem.json": "allow",
                        "reaction.json": "allow",
                        "/sandbox/work/reaction.json": "allow",
                        "*.json": "allow",
                        "**/*.json": "allow",
                        "/sandbox/work/**/*.json": "allow",
                    },
                    # OpenCode's edit permission gates edit, write, and patch.
                    "edit": "allow",
                    "skill": {
                        "*": "deny",
                        "hsopt": "allow",
                    },
                    "task": "deny",
                    "todoread": "deny",
                    "todowrite": "deny",
                    "codesearch": "allow",
                    "webfetch": "allow",
                    "websearch": "allow",
                },
            }
        },
    }


def mutation_prompt(feedback: dict[str, Any] | None = None) -> str:
    prompt = """You are proposing one HSOpt product-formula policy mutation.

The working directory contains:

- candidate.py: the only editable file;
- evaluate.py: the exact fixed evaluator implementation used for scoring;
- problem.json for an absolute-energy lane, or reaction.json plus its referenced
  Hamiltonian JSON files for a difference-energy lane.

Read those files as needed so the mutation can use the concrete Hamiltonian
coefficients, Pauli labels, term metadata, and evaluator interface. The
evaluator and JSON inputs are read-only. Edit only candidate.py.

You may optionally use an installed skill or browse/search for product-formula
and synthesis ideas before editing. Web research is optional, and search
content is inspiration rather than an evaluator result or permission to change
the benchmark. Do not search for this HSOpt repository, its campaign outputs,
or another baseline initializer. If an optional research tool is unavailable,
continue with the mutation instead of retrying it.

The fixed evaluator compiles every Pauli rotation with PyGridSynth, constructs
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
weight and the paper's one-half phase-estimation success guarantee. It is
feasible only when the decoded energy is within chemical accuracy of the
cached exact ground energy. Small resolved HF peak weight is therefore
penalized by a larger repetition cost; it is not an energy average.

Make one coherent deterministic mutation intended to lower the
confidence-adjusted T cost while remaining accurate. Preserve the
ProductFormula class and its repetitions(), build_step(), and
synthesis_epsilon() methods. build_step() is the scientific definition of the
formula: it may implement PF1, PF2/Strang, a higher-order stationary
composition, grouping, or another deterministic recurrence. The evaluator
infers standard PF1/PF2 labels from the returned rotations; it never trusts a
declared order. synthesis_epsilon() selects the PyGridSynth tolerance for each
returned rotation. Inspect evaluate.py for the exact accepted types, angle
convention, schedule validation, and synthesis-tolerance constraints.

Do not change any Hamiltonian, evaluator, source data, or benchmark definition.
Do not fabricate an evaluation result. Read the available context as needed.
Call one edit or write tool exactly once. After that tool succeeds, immediately
return a final text response without rewriting the file.
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
        "compiled_evolution_t_count",
        "per_attempt_t_count",
        "per_qpe_run_t_count",
        "confidence_adjusted_t_count",
        "t_count",
        "score",
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
    prompt = mutation_prompt(prompt_feedback)
    (run_dir / "opencode.json").write_text(
        json.dumps(opencode_config(), indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    _write_json(
        run_dir / "sandbox_context.json",
        {
            "schema": 1,
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
        _write_json(
            run_dir / "research_network.json",
            _configure_research_network(sandbox_name),
        )
        _run(
            [
                "openshell",
                "sandbox",
                "exec",
                "--name",
                sandbox_name,
                "--timeout",
                "30",
                "--",
                "mkdir",
                "-p",
                "/sandbox/work/.opencode/skills/hsopt",
                "/sandbox/work/.local/bin",
                *sorted(
                    {
                        f"/sandbox/work/{relative.parent.as_posix()}"
                        for _, relative in context_files
                        if relative.parent != PurePosixPath(".")
                    }
                ),
            ],
            timeout=45,
        )
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
        for local, relative in context_files:
            remote = f"/sandbox/work/{relative.as_posix()}"
            _run(
                [
                    "openshell",
                    "sandbox",
                    "upload",
                    sandbox_name,
                    str(local),
                    remote,
                ],
                timeout=timeout_seconds,
            )
            _run(
                [
                    "openshell",
                    "sandbox",
                    "exec",
                    "--name",
                    sandbox_name,
                    "--timeout",
                    "30",
                    "--",
                    "chmod",
                    "0444",
                    remote,
                ],
                timeout=45,
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
        _run(
            [
                "openshell",
                "sandbox",
                "upload",
                sandbox_name,
                str(sandbox_skill),
                "/sandbox/work/.opencode/skills/hsopt/SKILL.md",
            ],
            timeout=timeout_seconds,
        )
        _run(
            [
                "openshell",
                "sandbox",
                "upload",
                sandbox_name,
                str(sandbox_file_list_shim),
                "/sandbox/work/.local/bin/rg",
            ],
            timeout=timeout_seconds,
        )
        _run(
            [
                "openshell",
                "sandbox",
                "exec",
                "--name",
                sandbox_name,
                "--timeout",
                "30",
                "--",
                "chmod",
                "755",
                "/sandbox/work/.local/bin/rg",
            ],
            timeout=45,
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
            "--env",
            "OPENCODE_ENABLE_EXA=1",
            "--env",
            "PATH=/sandbox/work/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
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

    _validate_agent_action_contract(run_dir / "opencode.jsonl")
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
        "agent_completion": ("completed_edit_observed" if stopped_after_edit else "normal_exit"),
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
                    "agent_audit": (
                        "agent_audit.json" if (run_dir / "agent_audit.json").is_file() else None
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
    mutate_parser.add_argument("--kind", choices=("absolute", "difference"), required=True)
    mutate_parser.add_argument("--source", type=Path, required=True)
    mutate_parser.add_argument("--timeout-seconds", type=int, default=900)
    mutate_parser.add_argument("--keep-sandbox", action="store_true")

    audit_parser = subparsers.add_parser(
        "audit",
        help="rebuild redacted tool and OpenShell metrics from one mutation run",
    )
    audit_parser.add_argument("--run-dir", type=Path, required=True)
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
    else:
        result = write_agent_audit(args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
