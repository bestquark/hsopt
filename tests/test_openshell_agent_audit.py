from __future__ import annotations

import json
from pathlib import Path

import pytest

import hsopt.openshell_agent as openshell_agent
from hsopt.openshell_agent import (
    agent_feedback,
    mutate,
    opencode_config,
    summarize_opencode_events,
    summarize_openshell_log,
)


def test_opencode_audit_counts_unique_redacted_tool_calls(tmp_path: Path) -> None:
    log = tmp_path / "opencode.jsonl"
    events = [
        {
            "type": "tool_use",
            "timestamp": 1000,
            "sessionID": "session-1",
            "part": {
                "callID": "call-read",
                "tool": "read",
                "state": {
                    "status": "completed",
                    "input": {"filePath": "/sandbox/work/candidate.py"},
                    "output": "secret file contents",
                    "time": {"start": 900, "end": 1000},
                },
            },
        },
        {
            "type": "tool_use",
            "timestamp": 1100,
            "sessionID": "session-1",
            "part": {
                "callID": "call-read",
                "tool": "read",
                "state": {"status": "completed"},
            },
        },
        {
            "type": "tool_use",
            "timestamp": 1200,
            "sessionID": "session-1",
            "part": {
                "callID": "call-search",
                "tool": "websearch",
                "state": {
                    "status": "completed",
                    "input": {"query": "Pauli ordering"},
                },
            },
        },
        {
            "type": "step_finish",
            "sessionID": "session-1",
            "part": {
                "tokens": {
                    "total": 12,
                    "input": 10,
                    "output": 2,
                    "cache": {"read": 3, "write": 0},
                }
            },
        },
    ]
    log.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    metrics, actions = summarize_opencode_events(log)

    assert metrics["tool_calls"] == 2
    assert metrics["by_tool"] == {"read": 1, "websearch": 1}
    assert metrics["web_searches"] == 1
    assert metrics["online_tool_calls"] == 1
    assert metrics["inference_steps"] == 1
    assert metrics["tokens"]["cache_read"] == 3
    assert actions[0]["target"] == "candidate.py"
    assert actions[0]["duration_ms"] == 100
    assert "secret file contents" not in json.dumps(actions)


def test_openshell_audit_counts_ocsf_security_events(tmp_path: Path) -> None:
    log = tmp_path / "openshell.log"
    log.write_text(
        "\n".join(
            [
                "2026-01-01 OCSF NET:OPEN [INFO] ALLOWED opencode -> inference.local:443",
                "2026-01-01 OCSF HTTP:POST [INFO] ALLOWED POST https://inference.local/v1",
                "2026-01-01 OCSF NET:OPEN [MED] DENIED curl -> example.com:443",
                "2026-01-01 OCSF PROC:LAUNCH [INFO] opencode(12)",
            ]
        )
        + "\n"
    )

    metrics = summarize_openshell_log(log)

    assert metrics["ocsf_events"] == 4
    assert metrics["network_events"] == 3
    assert metrics["process_events"] == 1
    assert metrics["by_decision"] == {"allowed": 2, "denied": 1}
    assert metrics["denied_events"] == 1


def test_agent_feedback_omits_large_compilation_payloads() -> None:
    feedback = {
        "status": "ok",
        "observable": "compiled_unitary_ground_eigenphase",
        "t_count": 123,
        "score": 123,
        "absolute_energy_error_mha": 0.2,
        "policy": {"n_steps": 16, "uniform_epsilon": 1.0e-5},
        "unique_compilations": [{"gate_string_sha256": "a" * 64}] * 100,
        "problem_path": "/private/path/problem.json",
    }

    reduced = agent_feedback(feedback)

    assert reduced["t_count"] == 123
    assert reduced["policy"]["n_steps"] == 16
    assert "unique_compilations" not in reduced
    assert "problem_path" not in reduced


def test_opencode_policy_only_allows_candidate_file_mutations() -> None:
    permissions = opencode_config()["agent"]["hsopt-mutation"]["permission"]

    assert permissions["*"] == "deny"
    assert permissions["edit"] == {
        "*": "deny",
        "candidate.py": "allow",
        "/sandbox/work/candidate.py": "allow",
    }
    assert permissions["read"] == {
        "*": "deny",
        "candidate.py": "allow",
        "/sandbox/work/candidate.py": "allow",
    }
    assert permissions["webfetch"] == "deny"
    assert permissions["websearch"] == "deny"


def test_mutation_failure_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("agent did not mutate candidate.py")

    monkeypatch.setattr(openshell_agent, "_mutate_once", fail)
    run_dir = tmp_path / "iteration"

    with pytest.raises(RuntimeError, match="did not mutate"):
        mutate(tmp_path / "policy.py", tmp_path / "feedback.json", run_dir)

    outcome = json.loads((run_dir / "mutation_result.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["error_type"] == "RuntimeError"
    assert outcome["error"] == "agent did not mutate candidate.py"
