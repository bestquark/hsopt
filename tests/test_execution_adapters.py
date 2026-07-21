from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hsopt.tgate_opt.execution import (
    EvaluationRequest,
    build_local_evaluation,
    build_slurm_submission,
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
    assert config["provider"]["nvidia"]["options"]["apiKey"] == "{env:NVIDIA_API_KEY}"
    assert "sk-" not in json.dumps(config)
    permissions = config["agent"]["hsopt-mutation"]["permission"]
    assert permissions["*"] == "deny"
    assert permissions["read"] == {"*": "deny", "simple_pf.py": "allow"}
    assert permissions["edit"] == {"*": "deny", "simple_pf.py": "allow"}


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


def test_slurm_transport_blocks_without_scheduler_polling(tmp_path: Path) -> None:
    request = EvaluationRequest(
        evaluator=tmp_path / "evaluate.py",
        molecule="h2",
        script=tmp_path / "simple_pf.py",
    )
    submission = build_slurm_submission(
        request,
        submit_host="nvlogin",
        remote_repo_root="/lustre/hsopt",
        account="acct",
        partition="backfill_gpu",
        python="/lustre/hsopt/.venv/bin/python",
        legate="/lustre/hsopt/.venv/bin/legate",
    )

    assert submission.command[:2] == ["ssh", "nvlogin"]
    assert "sbatch --wait --parsable" in submission.command[-1]
    combined = " ".join(submission.command) + submission.stdin
    assert "squeue" not in combined
    assert "sacct" not in combined
    assert "--gres=gpu:1" in submission.command[-1]
    assert "TROVOLUT_PFS_ARRAY_BACKEND=cupynumeric" in submission.stdin


@pytest.mark.parametrize(
    "wrapper",
    [
        Path("experiments/hamlib121/runner/scripts/opencode_codex_compat.py"),
        Path("experiments/reactions10/runner/opencode_codex_compat.py"),
    ],
)
def test_paper_opencode_wrapper_preserves_raw_events_and_requires_an_edit(
    tmp_path: Path, wrapper: Path
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_npx = fake_bin / "npx"
    fake_npx.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

if sys.argv[-1] == "--version":
    print("1.4.11")
    raise SystemExit(0)
work = pathlib.Path(sys.argv[sys.argv.index("--dir") + 1])
policy = work / "simple_pf.py"
policy.write_text(policy.read_text() + "\\n# fake OpenCode edit\\n")
events = [
    {"type": "step_start", "sessionID": "ses_fake", "part": {}},
    {"type": "tool_use", "sessionID": "ses_fake", "part": {"tool": "edit", "state": {"status": "completed"}}},
    {"type": "text", "sessionID": "ses_fake", "part": {"text": "Updated the policy."}},
    {"type": "step_finish", "sessionID": "ses_fake", "part": {"tokens": {"input": 3, "output": 2, "cache": {"read": 1}}}},
]
for event in events:
    print(json.dumps(event), flush=True)
""",
        encoding="utf-8",
    )
    fake_npx.chmod(0o755)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "simple_pf.py").write_text("VALUE = 1\n", encoding="utf-8")
    last_message = tmp_path / "last_message.txt"
    completed = subprocess.run(
        [
            sys.executable,
            str(wrapper.resolve()),
            "exec",
            "--json",
            "-s",
            "workspace-write",
            "-C",
            str(work_dir),
            "-m",
            NEMOTRON_MODEL,
            "-o",
            str(last_message),
            "Edit simple_pf.py",
        ],
        cwd=wrapper.resolve().parents[0],
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "NVIDIA_API_KEY": "test-only-not-a-real-key",
            "HSOPT_MUTATION_STATE_DIR": str(tmp_path / "state"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    assert events[-1]["type"] == "turn.completed"
    assert any(event["type"] == "opencode.event" for event in events)
    assert events[-1]["usage"] == {
        "cache_read_tokens": 1,
        "input_tokens": 3,
        "output_tokens": 2,
    }
    assert "fake OpenCode edit" in (work_dir / "simple_pf.py").read_text(encoding="utf-8")
    assert last_message.read_text(encoding="utf-8") == "Updated the policy.\n"
