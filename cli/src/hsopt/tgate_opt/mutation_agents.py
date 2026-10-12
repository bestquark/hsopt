from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODEX_PACKAGE = "@openai/codex@0.144.0"
OPENCODE_PACKAGE = "@opencode/cli@2.0.26"
NEMOTRON_MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
NVIDIA_API_BASE = "https://inference-api.nvidia.com/v1"
OPENCODE_AGENT = "hsopt-mutation"


@dataclass(frozen=True)
class MutationInvocation:
    agent: str
    model: str
    package: str
    command: list[str]
    environment: dict[str, str]


@dataclass(frozen=True)
class AgentLifecycle:
    agent: str
    session_id: str
    final_text: str
    edit_count: int
    usage: dict[str, Any] | None


def normalize_agent(agent: str) -> str:
    selected = agent.strip().lower()
    if selected not in {"codex", "opencode"}:
        raise ValueError(f"unsupported mutation agent {agent!r}; use 'codex' or 'opencode'")
    return selected


def opencode_config(*, model: str = NEMOTRON_MODEL) -> dict[str, Any]:
    provider_model = model.removeprefix("nvidia/")
    return {
        "$schema": "https://opencode.ai/config.json",
        "update": "disable",
        "share": "disabled",
        "snapshots": False,
        "enabled_providers": ["nvidia"],
        "model": model,
        "providers": {
            "nvidia": {
                "name": "NVIDIA API Catalog",
                "env": ["NVIDIA_API_KEY"],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": NVIDIA_API_BASE},
                "models": {
                    provider_model: {
                        "modelID": model,
                        "name": "NVIDIA Nemotron 3.5 Super VL",
                        "capabilities": {"tools": True},
                    }
                },
            }
        },
        "agents": {
            OPENCODE_AGENT: {
                "description": "Edit only the isolated HSOpt simple_pf.py candidate.",
                "mode": "primary",
                "model": model,
                "request": {"body": {"temperature": 1.0, "top_p": 0.95}},
                "permissions": [
                    {"action": "*", "resource": "*", "effect": "deny"},
                    {"action": "read", "resource": "simple_pf.py", "effect": "allow"},
                    {"action": "edit", "resource": "simple_pf.py", "effect": "allow"},
                ],
            }
        },
    }


def build_mutation_invocation(
    *,
    agent: str,
    work_dir: Path,
    prompt: str,
    state_dir: Path,
    model: str | None = None,
    reasoning_effort: str = "high",
    command: str = "npx",
) -> MutationInvocation:
    selected = normalize_agent(agent)
    work_dir = work_dir.resolve()
    state_dir = state_dir.resolve()
    environment: dict[str, str] = {}
    if selected == "codex":
        selected_model = model or "gpt-5.5"
        return MutationInvocation(
            agent=selected,
            model=selected_model,
            package=CODEX_PACKAGE,
            command=[
                command,
                "-y",
                CODEX_PACKAGE,
                "exec",
                "--sandbox",
                "workspace-write",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--json",
                "-C",
                str(work_dir),
                "-m",
                selected_model,
                "-c",
                f'model_reasoning_effort="{reasoning_effort}"',
                prompt,
            ],
            environment=environment,
        )

    selected_model = model or NEMOTRON_MODEL
    environment.update(
        {
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                opencode_config(model=selected_model), sort_keys=True, separators=(",", ":")
            ),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            "OPENCODE_DISABLE_MODELS_FETCH": "true",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "true",
            "XDG_CONFIG_HOME": str(state_dir / "config"),
            "XDG_DATA_HOME": str(state_dir / "data"),
            "XDG_CACHE_HOME": str(state_dir / "cache"),
            "npm_config_cache": str(state_dir / "npm"),
            "npm_config_update_notifier": "false",
        }
    )
    return MutationInvocation(
        agent=selected,
        model=selected_model,
        package=OPENCODE_PACKAGE,
        command=[
            command,
            "-y",
            OPENCODE_PACKAGE,
            "run",
            "--standalone",
            "--format",
            "json",
            "--model",
            selected_model,
            "--agent",
            OPENCODE_AGENT,
            prompt,
        ],
        environment=environment,
    )


def invocation_environment(invocation: MutationInvocation) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(invocation.environment)
    return environment


def _jsonl_events(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"mutation agent lifecycle output is missing or unsafe: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid lifecycle JSON on line {line_number}: {exc}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise RuntimeError(f"lifecycle line {line_number} has no event type")
        events.append(event)
    if not events:
        raise RuntimeError("mutation agent emitted no lifecycle events")
    return events


def validate_agent_lifecycle(agent: str, path: Path) -> AgentLifecycle:
    selected = normalize_agent(agent)
    events = _jsonl_events(path)
    if selected == "codex":
        types = [str(event["type"]) for event in events]
        if "thread.started" not in types or "turn.completed" not in types:
            raise RuntimeError(f"incomplete Codex lifecycle: {types}")
        failures = [kind for kind in types if kind.endswith((".failed", ".error"))]
        if failures:
            raise RuntimeError(f"Codex lifecycle reported failures: {failures}")
        started = next(event for event in events if event["type"] == "thread.started")
        session_id = str(started.get("thread_id", ""))
        if not session_id:
            raise RuntimeError("Codex lifecycle has no thread ID")
        texts = [
            str(event.get("text", "")).strip()
            for event in events
            if event.get("type") in {"item.completed", "turn.completed"}
            and str(event.get("text", "")).strip()
        ]
        return AgentLifecycle(selected, session_id, texts[-1] if texts else "completed", 0, None)

    errors = [event for event in events if event["type"] == "error"]
    if errors:
        raise RuntimeError(f"OpenCode lifecycle reported {len(errors)} error event(s)")
    types = [str(event["type"]) for event in events]
    if "step_start" not in types or "step_finish" not in types:
        raise RuntimeError(f"incomplete OpenCode lifecycle: {types}")
    session_ids = {str(event.get("sessionID", "")) for event in events}
    session_ids.discard("")
    if len(session_ids) != 1:
        raise RuntimeError(f"OpenCode lifecycle has inconsistent session IDs: {session_ids}")
    edits = [
        event
        for event in events
        if event["type"] == "tool_use"
        and isinstance(event.get("part"), dict)
        and event["part"].get("tool") in {"edit", "write"}
        and isinstance(event["part"].get("state"), dict)
        and event["part"]["state"].get("status") == "completed"
    ]
    if not edits:
        raise RuntimeError("OpenCode lifecycle contains no completed edit of simple_pf.py")
    failed_tools = [
        event
        for event in events
        if event["type"] == "tool_use"
        and isinstance(event.get("part"), dict)
        and isinstance(event["part"].get("state"), dict)
        and event["part"]["state"].get("status") == "error"
    ]
    if failed_tools:
        raise RuntimeError(f"OpenCode lifecycle contains {len(failed_tools)} failed tool call(s)")
    texts = [
        str(event["part"].get("text", "")).strip()
        for event in events
        if event["type"] == "text" and isinstance(event.get("part"), dict)
    ]
    texts = [value for value in texts if value]
    if not texts:
        raise RuntimeError("OpenCode lifecycle contains no nonempty terminal text")
    finish = next(event for event in reversed(events) if event["type"] == "step_finish")
    usage = finish.get("part", {}).get("tokens") if isinstance(finish.get("part"), dict) else None
    return AgentLifecycle(
        selected,
        next(iter(session_ids)),
        texts[-1],
        len(edits),
        usage if isinstance(usage, dict) else None,
    )
