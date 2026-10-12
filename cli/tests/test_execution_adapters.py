from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsopt.tgate_opt.execution import (
    EvaluationRequest,
    build_local_evaluation,
)
from hsopt.tgate_opt.mutation_agents import (
    NEMOTRON_MODEL,
    build_mutation_invocation,
    validate_agent_lifecycle,
)


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def test_opencode_invocation_is_edit_only_and_reads_key_from_environment(tmp_path: Path) -> None:
    invocation = build_mutation_invocation(
        agent="opencode",
        work_dir=tmp_path,
        prompt="Edit simple_pf.py",
        state_dir=tmp_path / "state",
    )

    assert invocation.model == NEMOTRON_MODEL
    assert "--auto" not in invocation.command
    assert "--dangerously-skip-permissions" not in invocation.command
    assert invocation.command[-1] == "Edit simple_pf.py"
    config = json.loads(invocation.environment["OPENCODE_CONFIG_CONTENT"])
    assert config["providers"]["nvidia"]["env"] == ["NVIDIA_API_KEY"]
    assert config["providers"]["nvidia"]["models"][NEMOTRON_MODEL.removeprefix("nvidia/")]["modelID"] == NEMOTRON_MODEL
    assert "--standalone" in invocation.command
    assert "--dir" not in invocation.command
    assert invocation.package == "@opencode/cli@2.0.26"
    assert "sk-" not in json.dumps(config)
    rules = config["agents"]["hsopt-mutation"]["permissions"]
    assert rules[0] == {"action": "*", "resource": "*", "effect": "deny"}
    assert rules[1:] == [{"action": action, "resource": "simple_pf.py", "effect": "allow"} for action in ("read", "edit")]


def test_opencode_lifecycle_requires_completed_edit_and_terminal_text(tmp_path: Path) -> None:
    output = tmp_path / "opencode.jsonl"
    _write_jsonl(
        output,
        [
            {"type": "step_start", "sessionID": "ses_1", "part": {}},
            {
                "type": "tool_use",
                "sessionID": "ses_1",
                "part": {"tool": "edit", "state": {"status": "completed"}},
            },
            {
                "type": "step_finish",
                "sessionID": "ses_1",
                "part": {"reason": "stop", "tokens": {"input": 1, "output": 2}},
            },
            {
                "type": "text",
                "sessionID": "ses_1",
                "part": {"text": "Updated the policy."},
            },
        ],
    )

    lifecycle = validate_agent_lifecycle("opencode", output)
    assert lifecycle.session_id == "ses_1"
    assert lifecycle.final_text == "Updated the policy."
    assert lifecycle.edit_count == 1

    _write_jsonl(
        output,
        [
            {"type": "step_start", "sessionID": "ses_2", "part": {}},
            {"type": "step_finish", "sessionID": "ses_2", "part": {"reason": "stop"}},
            {"type": "text", "sessionID": "ses_2", "part": {"text": "No change."}},
        ],
    )
    with pytest.raises(RuntimeError, match="completed edit"):
        validate_agent_lifecycle("opencode", output)


def test_local_evaluation_can_use_numpy_or_one_local_gpu(tmp_path: Path) -> None:
    request = EvaluationRequest(
        evaluator=tmp_path / "evaluate.py",
        molecule="h2",
        script=tmp_path / "simple_pf.py",
    )
    cpu = build_local_evaluation(request, backend="numpy", python="/venv/bin/python")
    assert cpu.command[:2] == ["/venv/bin/python", str(request.evaluator)]
    assert cpu.environment["TROVOLUT_PFS_ARRAY_BACKEND"] == "numpy"

    gpu = build_local_evaluation(
        request,
        backend="cupynumeric",
        python="/venv/bin/python",
        legate="/venv/bin/legate",
    )
    assert gpu.command[:7] == [
        "/venv/bin/legate",
        "--gpus",
        "1",
        "--run-mode",
        "exec",
        "/venv/bin/python",
        str(request.evaluator),
    ]
    assert gpu.environment["TROVOLUT_PFS_ARRAY_BACKEND"] == "cupynumeric"
