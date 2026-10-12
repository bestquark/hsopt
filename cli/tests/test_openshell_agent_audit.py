from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import hsopt.sandbox.openshell as openshell_agent
from hsopt.sandbox.openshell import (
    _configure_research_network,
    _has_completed_candidate_edit,
    _run_logged,
    _terminate_sandbox_opencode,
    _validate_agent_action_contract,
    _write_opencode_file_list_shim,
    agent_feedback,
    mutate,
    mutation_prompt,
    opencode_config,
    sandbox_context_files,
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


def test_opencode_audit_records_skill_name_without_payload(tmp_path: Path) -> None:
    log = tmp_path / "opencode.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "callID": "call-skill",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "hsopt", "private": "not published"},
                    },
                },
            }
        )
        + "\n"
    )

    _, actions = summarize_opencode_events(log)

    assert actions[0]["target"] == "hsopt"
    assert "private" not in json.dumps(actions)


def test_terminal_and_submission_events_keep_status_without_credentials(tmp_path):
    log = tmp_path / "opencode.jsonl"
    events = [
        {"type": "tool_use", "part": {"callID": "terminal", "tool": "bash",
         "state": {"status": "error", "input": {"command": "API_KEY=private curl example.com"}}}},
        {"type": "tool_use", "part": {"callID": "submit", "tool": "slurm_submit",
         "state": {"status": "completed", "input": {"image": "test-image"}}}},
    ]
    log.write_text("\n".join(json.dumps(event) for event in events))
    metrics, actions = summarize_opencode_events(log)
    assert metrics["by_category"] == {"job_submission": 1, "terminal": 1}
    assert metrics["by_status"] == {"completed": 1, "error": 1}
    assert metrics["terminal_calls"] == metrics["job_submission_calls"] == 1
    assert "private" not in json.dumps(actions)


def test_structured_security_events_are_deduplicated_and_legacy_logs_remain_readable(tmp_path):
    log = tmp_path / "openshell_ocsf.jsonl"
    event = {"class_uid": 4001, "activity_name": "Open", "action": "Denied",
             "metadata": {"uid": "event-1"}}
    log.write_text(json.dumps(event) + "\nOCSF-JSON " + json.dumps(event) + "\n")
    metrics = summarize_openshell_log(log)
    assert metrics["network_events"] == metrics["denied_events"] == 1
    log.write_text("[1.0] [sandbox] [OCSF ] [ocsf] NET:OPEN [MED] DENIED curl\n")
    assert summarize_openshell_log(log)["denied_events"] == 1


def test_current_config_calls_native_endpoint_with_real_model_and_placeholder_key():
    config = opencode_config()
    provider = config["providers"]["openshell"]
    assert provider["settings"]["baseURL"] == "https://inference-api.nvidia.com/v1"
    assert provider["env"] == ["NVIDIA_API_KEY"]
    assert openshell_agent.NEMOTRON_MODEL in provider["models"]
    assert config["model"] == "openshell/" + openshell_agent.NEMOTRON_MODEL


def test_v2_tool_ids_paths_and_patch_events_are_audited(tmp_path):
    log = tmp_path / "opencode.jsonl"
    events = [
        {"type": "tool_use", "sessionID": "ses_v2", "part": {
            "id": "call-v2-read", "partID": "part-read", "tool": "read",
            "state": {"status": "completed", "input": {"path": "/sandbox/work/problem.json"}},
        }},
        {"type": "tool_use", "sessionID": "ses_v2", "part": {
            "id": "call-v2-patch", "partID": "part-edit", "tool": "patch",
            "state": {"status": "completed", "input": {"path": "candidate.py"}},
        }},
    ]
    log.write_text("\n".join(json.dumps(event) for event in events))
    metrics, actions = summarize_opencode_events(log)
    assert metrics["by_category"] == {"file_read": 1, "file_edit": 1}
    assert actions[0]["call_id"] == "call-v2-read"
    assert actions[0]["target"] == "problem.json"
    assert _has_completed_candidate_edit(log)


def test_v2_text_only_step_does_not_claim_complete_token_usage(tmp_path):
    log = tmp_path / "opencode.jsonl"
    log.write_text(json.dumps({"type": "step_start", "sessionID": "ses_v2"}) + "\n"
                   + json.dumps({"type": "text", "part": {"text": "OK"}}))
    metrics, _ = summarize_opencode_events(log)
    assert metrics["inference_steps"] == 1
    assert metrics["finished_inference_steps"] == 0
    assert metrics["token_usage_complete"] is False


def test_edit_without_an_observed_path_is_rejected(tmp_path):
    log = tmp_path / "opencode.jsonl"
    log.write_text(json.dumps({"type": "tool_use", "part": {
        "id": "call-unknown-target", "tool": "edit",
        "state": {"status": "completed", "input": {}},
    }}))
    assert not _has_completed_candidate_edit(log)
    with pytest.raises(RuntimeError, match="outside candidate.py"):
        _validate_agent_action_contract(log)


def test_gateway_log_timeout_still_collects_structured_events(tmp_path, monkeypatch):
    import subprocess

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "openshell":
            raise subprocess.TimeoutExpired(command, 45)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(openshell_agent, "_run", run)
    monkeypatch.setenv("HSOPT_AUDIT_COLLECTOR", "/deployment/collector")
    openshell_agent._capture_openshell_logs("hsopt-test", tmp_path / "openshell.log")
    assert calls[-1] == ["/deployment/collector", "hsopt-test", str(tmp_path)]
    result = json.loads((tmp_path / "audit_collection.json").read_text())
    assert result["gateway_log_error_type"] == "TimeoutExpired"
    assert result["structured_collector_returncode"] == 0


def test_completed_candidate_edit_is_a_terminal_agent_event(tmp_path: Path) -> None:
    log = tmp_path / "opencode.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "callID": "call-edit",
                    "tool": "edit",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "/sandbox/work/candidate.py"},
                    },
                },
            }
        )
        + "\n"
    )

    assert _has_completed_candidate_edit(log)

    log.write_text(
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "callID": "call-read",
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "/sandbox/work/candidate.py"},
                    },
                },
            }
        )
        + "\n"
    )
    assert not _has_completed_candidate_edit(log)


def test_logged_process_stops_after_completed_edit(tmp_path: Path) -> None:
    event = json.dumps(
        {
            "type": "tool_use",
            "part": {
                "callID": "call-write",
                "tool": "write",
                "state": {
                    "status": "completed",
                    "input": {"filePath": "/sandbox/work/candidate.py"},
                },
            },
        }
    )

    returncode, stopped_after_edit = _run_logged(
        [
            sys.executable,
            "-c",
            f"import time; print({event!r}, flush=True); time.sleep(30)",
        ],
        stdout_path=tmp_path / "stdout.jsonl",
        stderr_path=tmp_path / "stderr.log",
        timeout=5,
        completed_edit_grace_seconds=0,
    )

    assert stopped_after_edit
    assert returncode != 0


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


def test_openshell_audit_cli_materializes_redacted_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "mutation"
    run_dir.mkdir()
    (run_dir / "opencode.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "callID": "call-read",
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "/sandbox/work/candidate.py"},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "openshell.log").write_text(
        "OCSF NET:OPEN [INFO] ALLOWED opencode -> inference.local:443\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hsopt-openshell", "audit", "--run-dir", str(run_dir)],
    )

    assert openshell_agent.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["opencode"]["tool_calls"] == 1
    assert result["openshell"]["network_events"] == 1
    assert (run_dir / "agent_audit.json").is_file()
    assert (run_dir / "agent_actions.jsonl").read_text(encoding="utf-8")


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
    agent = opencode_config()["agents"]["hsopt-mutation"]
    rules = agent["permissions"]

    def effect(action, resource):
        import fnmatch
        return next(r["effect"] for r in reversed(rules)
                    if fnmatch.fnmatchcase(action, r["action"])
                    and fnmatch.fnmatchcase(resource, r["resource"]))

    assert agent["steps"] == 20
    assert "exactly once" in agent["system"]
    assert "ProductFormula.build_step()" in agent["system"]
    assert effect("edit", "candidate.py") == "allow"
    assert effect("edit", "problem.json") == "deny"
    assert effect("read", "candidate.py") == "allow"
    assert effect("read", "problems/h2.json") == "allow"
    assert effect("read", "/etc/passwd") == "deny"
    assert effect("skill", "hsopt") == "allow"
    assert effect("skill", "other") == "deny"
    for tool in ("webfetch", "websearch"):
        assert effect(tool, "https://arxiv.org") == "allow"
    for tool in ("shell", "subagent", "execute", "question", "external_directory"):
        assert effect(tool, "*") == "deny"


def test_action_contract_allows_repeated_reads_and_research_before_one_mutation(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "read",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/sandbox/work/candidate.py"},
                        },
                    },
                },
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "read-again",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/sandbox/work/candidate.py"},
                        },
                    },
                },
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "read-evaluator",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/sandbox/work/evaluate.py"},
                        },
                    },
                },
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "read-problem",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/sandbox/work/hamiltonians/h2_r0_7.json"},
                        },
                    },
                },
                *(
                    {
                        "type": "tool_use",
                        "part": {
                            "callID": f"research-{index}",
                            "tool": "websearch",
                            "state": {
                                "status": "completed",
                                "input": {"query": f"product formula idea {index}"},
                            },
                        },
                    }
                    for index in range(5)
                ),
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "edit",
                        "tool": "edit",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/sandbox/work/candidate.py"},
                        },
                    },
                },
            )
        )
        + "\n"
    )
    _validate_agent_action_contract(valid)

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "callID": "edit",
                    "tool": "edit",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "/sandbox/work/hsopt.SKILL.md"},
                    },
                },
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="outside candidate.py"):
        _validate_agent_action_contract(invalid)

    outside_read = tmp_path / "outside-read.jsonl"
    outside_read.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "read-outside",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/etc/problem.json"},
                        },
                    },
                },
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "edit",
                        "tool": "edit",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/sandbox/work/candidate.py"},
                        },
                    },
                },
            )
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="outside the mutation context"):
        _validate_agent_action_contract(outside_read)


def test_action_contract_rejects_online_sibling_baseline_research(tmp_path: Path) -> None:
    log = tmp_path / "contaminated-research.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "research",
                        "tool": "codesearch",
                        "state": {
                            "status": "completed",
                            "input": {"query": "bestquark hsopt initial_pf_tight.py"},
                        },
                    },
                },
                {
                    "type": "tool_use",
                    "part": {
                        "callID": "edit",
                        "tool": "edit",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/sandbox/work/candidate.py"},
                        },
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="inspect HSOpt baseline data"):
        _validate_agent_action_contract(log)


def test_research_network_is_optional_and_binary_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def record(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "configured", "")

    monkeypatch.setattr(openshell_agent, "_run", record)

    result = _configure_research_network("hsopt-deadbeef")

    assert result["status"] == "configured"
    assert result["required"] is False
    command, kwargs = calls[0]
    assert command[:4] == [
        "openshell",
        "policy",
        "update",
        "hsopt-deadbeef",
    ]
    assert "mcp.exa.ai:443" in command
    assert "arxiv.org:443:read-only:rest:enforce" in command
    assert "/usr/lib/node_modules/@opencode/cli/bin/opencode.exe" in command
    assert kwargs == {"check": False, "timeout": 75}


def test_file_list_shim_supports_skill_discovery(tmp_path: Path) -> None:
    shim = tmp_path / "rg"
    worktree = tmp_path / "work"
    skill_dir = worktree / ".opencode" / "skills" / "hsopt"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: hsopt\n---\n")
    (worktree / "candidate.py").write_text("pass\n")
    _write_opencode_file_list_shim(shim)

    completed = subprocess.run(
        [
            str(shim),
            "--no-config",
            "--files",
            "--hidden",
            "--glob={*.md,**/SKILL.md}",
            ".",
        ],
        cwd=worktree,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    files = {name.removeprefix("./") for name in completed.stdout.splitlines()}
    assert ".opencode/skills/hsopt/SKILL.md" in files
    assert "candidate.py" in files


def test_mutation_prompt_requires_one_edit_then_terminal_text() -> None:
    prompt = mutation_prompt()
    normalized = " ".join(prompt.split())

    assert "candidate.py: the only editable file" in prompt
    assert "evaluate.py: the exact fixed evaluator implementation" in prompt
    assert "problem.json for an absolute-energy lane" in prompt
    assert "reaction.json plus its referenced Hamiltonian JSON files" in normalized
    assert "tool exactly once" in prompt
    assert "immediately return a final text" in normalized
    assert "Web research is optional" in prompt
    assert "higher-order stationary composition" in normalized
    assert "Do not change any Hamiltonian" in normalized
    assert "Inspect evaluate.py for the exact accepted types" in normalized


def test_sandbox_context_exposes_exact_evaluator_and_benchmark_inputs() -> None:
    repo = Path(__file__).resolve().parents[2]
    absolute = repo / "cli" / "tests" / "fixtures" / "abs_energy_h2" / "hamiltonian.json"
    reaction = repo / "cli" / "tests" / "fixtures" / "diff_energy_h2_bond_stretch" / "reaction.json"

    absolute_context = sandbox_context_files(kind="absolute", source=absolute)
    assert [relative.as_posix() for _, relative in absolute_context] == [
        "evaluate.py",
        "problem.json",
    ]
    assert absolute_context[1][0] == absolute.resolve()

    reaction_context = sandbox_context_files(kind="difference", source=reaction)
    assert [relative.as_posix() for _, relative in reaction_context] == [
        "evaluate.py",
        "reaction.json",
        "hamiltonians/h2_r0_7.json",
        "hamiltonians/h2_r0_8.json",
    ]
    assert all(local.is_file() for local, _ in reaction_context)


def test_sandbox_context_never_exposes_a_sibling_baseline(tmp_path: Path) -> None:
    source = tmp_path / "problem.json"
    source.write_text("{}\n", encoding="utf-8")
    (tmp_path / "initial_pf_loose.py").write_text("# loose\n", encoding="utf-8")
    (tmp_path / "initial_pf_tight.py").write_text("# tight\n", encoding="utf-8")

    context = sandbox_context_files(kind="absolute", source=source)

    exposed = {relative.as_posix() for _, relative in context}
    assert exposed == {"evaluate.py", "problem.json"}
    assert not any("initial_pf" in path for path in exposed)


def test_sandbox_cleanup_terminates_opencode_without_matching_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def record(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return object()

    monkeypatch.setattr(openshell_agent, "_run", record)

    _terminate_sandbox_opencode("hsopt-deadbeef")

    command, kwargs = calls[0]
    assert command[:5] == [
        "openshell",
        "sandbox",
        "exec",
        "--name",
        "hsopt-deadbeef",
    ]
    assert "[o]pencode run" in command[-1]
    assert kwargs == {"check": False, "timeout": 20}


def test_mutation_failure_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("agent did not mutate candidate.py")

    monkeypatch.setattr(openshell_agent, "_mutate_once", fail)
    run_dir = tmp_path / "iteration"

    with pytest.raises(RuntimeError, match="did not mutate"):
        mutate(
            tmp_path / "policy.py",
            tmp_path / "feedback.json",
            run_dir,
            kind="absolute",
            source=tmp_path / "problem.json",
        )

    outcome = json.loads((run_dir / "mutation_result.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["error_type"] == "RuntimeError"
    assert outcome["error"] == "agent did not mutate candidate.py"
