#!/usr/bin/env python3
"""Run pinned OpenCode/Nemotron behind the runner's durable JSONL envelope."""

from __future__ import annotations

import argparse
import json
import os
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


OPENCODE_PACKAGE = "opencode-ai@1.4.11"
OPENCODE_VERSION = "1.4.11"
DEFAULT_MODEL = "nvidia/nvidia/nemotron-3-super-v3"
AGENT_NAME = "hsopt-mutation"


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def config(model: str) -> dict[str, Any]:
    provider_model = model.removeprefix("nvidia/")
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "enabled_providers": ["nvidia"],
        "model": model,
        "provider": {
            "nvidia": {
                "name": "NVIDIA API Catalog",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": "https://inference-api.nvidia.com/v1",
                    "apiKey": "{env:NVIDIA_API_KEY}",
                },
                "models": {provider_model: {"name": "NVIDIA Nemotron 3 Super v3"}},
            }
        },
        "agent": {
            AGENT_NAME: {
                "description": "Edit only the isolated HSOpt simple_pf.py candidate.",
                "mode": "primary",
                "model": model,
                "temperature": 1.0,
                "top_p": 0.95,
                "permission": {
                    "*": "deny",
                    "read": {"*": "deny", "simple_pf.py": "allow"},
                    "edit": {"*": "deny", "simple_pf.py": "allow"},
                },
            }
        },
    }


def flattened_usage(tokens: object) -> dict[str, int]:
    values: dict[str, int] = {}
    if not isinstance(tokens, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    for key, value in tokens.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            values[f"{key}_tokens"] = value
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                if (
                    isinstance(child_value, int)
                    and not isinstance(child_value, bool)
                    and child_value >= 0
                ):
                    values[f"{key}_{child_key}_tokens"] = child_value
    return values or {"input_tokens": 0, "output_tokens": 0}


def write_last_message(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    if argv and argv[0] == "exec":
        argv = argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--sandbox", "-s")
    parser.add_argument("--ephemeral", action="store_true")
    parser.add_argument("--ignore-user-config", action="store_true")
    parser.add_argument("--ignore-rules", action="store_true")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("-C", "--work-dir", type=Path, required=True)
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-c", dest="config_overrides", action="append", default=[])
    parser.add_argument("-o", "--last-message", type=Path, required=True)
    parser.add_argument("prompt")
    return parser.parse_args(argv)


def main() -> int:
    if sys.argv[1:] == ["--version"]:
        completed = subprocess.run(
            ["npx", "-y", OPENCODE_PACKAGE, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0 or completed.stdout.strip() != OPENCODE_VERSION:
            print(completed.stderr or completed.stdout, file=sys.stderr, end="")
            return completed.returncode or 1
        print(f"opencode {OPENCODE_VERSION}")
        return 0

    args = parse_args(sys.argv[1:])
    work_dir = args.work_dir.resolve()
    if not os.environ.get("NVIDIA_API_KEY"):
        print(
            "NVIDIA_API_KEY is not set; load it from a protected environment file",
            file=sys.stderr,
        )
        return 78
    state_dir = Path(
        os.environ.get("HSOPT_MUTATION_STATE_DIR", str(work_dir.parent / ".opencode-state"))
    ).resolve()
    environment = os.environ.copy()
    environment.update(
        {
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                config(args.model), sort_keys=True, separators=(",", ":")
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
    state_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "npx",
        "-y",
        OPENCODE_PACKAGE,
        "run",
        "--format",
        "json",
        "--model",
        args.model,
        "--agent",
        AGENT_NAME,
        "--dir",
        str(work_dir),
        args.prompt,
    ]
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )
    assert process.stdout is not None
    session_id = ""
    started = False
    finished = False
    edit_count = 0
    final_text = ""
    usage: dict[str, int] = {}
    errors: list[str] = []
    for line_number, line in enumerate(process.stdout, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"invalid OpenCode JSONL line {line_number}")
            emit({"type": "opencode.invalid", "line_number": line_number})
            continue
        emit({"type": "opencode.event", "event": event})
        event_session = str(event.get("sessionID", ""))
        if event_session:
            if session_id and session_id != event_session:
                errors.append("OpenCode session identity changed")
            session_id = session_id or event_session
        event_type = event.get("type")
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if event_type == "step_start" and not started:
            started = True
            emit({"type": "thread.started", "thread_id": session_id})
        elif event_type == "step_finish":
            finished = True
            usage = flattened_usage(part.get("tokens"))
        elif event_type == "tool_use":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            if part.get("tool") in {"edit", "write"} and state.get("status") == "completed":
                edit_count += 1
            if state.get("status") == "error":
                errors.append(f"OpenCode tool {part.get('tool')!r} failed")
        elif event_type == "text":
            text = str(part.get("text", "")).strip()
            if text:
                final_text = text
        elif event_type == "error":
            errors.append("OpenCode reported a session error")
    returncode = process.wait()
    if returncode != 0:
        errors.append(f"OpenCode exited with status {returncode}")
    if not started or not session_id:
        errors.append("OpenCode emitted no step_start session identity")
    if not finished:
        errors.append("OpenCode emitted no step_finish")
    if edit_count == 0:
        errors.append("OpenCode completed no edit of simple_pf.py")
    if not final_text:
        errors.append("OpenCode emitted no terminal text")
    if errors:
        emit({"type": "turn.failed", "error": "; ".join(errors)})
        return returncode or 1
    write_last_message(args.last_message, final_text)
    emit(
        {
            "type": "turn.completed",
            "usage": usage,
            "mutation_agent": "opencode",
            "model": args.model,
            "edit_count": edit_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
