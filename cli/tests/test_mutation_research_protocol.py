from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hsopt.campaign.loop import propose_lane
from hsopt.evaluation.compiled_unitary import EVALUATION_CONTRACT_VERSION
from hsopt.sandbox import openshell
from hsopt.sandbox.mutation_retry import (
    AgentRuntimeError,
    clear_mutation_retry,
    mutation_readiness,
    record_mutation_failure,
    resume_mutations,
)
from hsopt.sandbox.research_context import benchmark_summary, duplicate_policy, lane_history

FIXTURES = Path(__file__).parent / "fixtures"
POLICY = """class ProductFormula:
    def __init__(self, problem):
        self.n = 2
    def repetitions(self):
        return self.n
    def build_step(self, terms, step_time):
        return [(term, term['coefficient'] * step_time) for term in terms]
    def synthesis_epsilon(self, term, angle):
        return 1e-6
"""


def write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def lane_at(root: Path, name: str = "a") -> Path:
    lane = root / "lanes" / name
    write(
        lane / "manifest.json",
        {
            "queue_dir": str(root / "queue"),
            "family": "abs_energy",
            "iterations_target": 50,
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        },
    )
    write(
        lane / "state.json",
        {"pending_task_id": None, "pending_kind": None, "iterations_completed": 0},
    )
    (lane / "incumbent.py").write_text(POLICY)
    return lane


def event(tool, target, status="completed"):
    return {
        "type": "tool_use",
        "part": {
            "callID": f"{tool}:{target}",
            "tool": tool,
            "state": {"status": status, "input": {"filePath": target}},
        },
    }


@pytest.mark.parametrize("failure", ["missing", "failed", "after_edit"])
def test_benchmark_reads_must_succeed_before_mutation(tmp_path, failure):
    reads = [event("read", "candidate.py"), event("read", "research_context.json")]
    edit = event("edit", "candidate.py")
    benchmark = event("read", "problem.json", "error" if failure == "failed" else "completed")
    events = reads + (
        [edit, benchmark]
        if failure == "after_edit"
        else ([benchmark, edit] if failure == "failed" else [edit])
    )
    log = tmp_path / "opencode.jsonl"
    log.write_text("\n".join(json.dumps(x) for x in events))
    with pytest.raises(RuntimeError, match="Read required benchmark context"):
        openshell._validate_agent_action_contract(
            log, required_reads=["candidate.py", "research_context.json", "problem.json"]
        )


def test_all_reaction_species_are_required(tmp_path):
    required = [
        "candidate.py",
        "research_context.json",
        "reaction.json",
        "hamiltonians/a.json",
        "hamiltonians/b.json",
    ]
    log = tmp_path / "opencode.jsonl"
    events = [event("read", target) for target in required[:-1]] + [event("edit", "candidate.py")]
    log.write_text("\n".join(json.dumps(x) for x in events))
    with pytest.raises(RuntimeError, match="hamiltonians/b.json"):
        openshell._validate_agent_action_contract(log, required_reads=required)
    events.insert(-1, event("read", required[-1]))
    log.write_text("\n".join(json.dumps(x) for x in events))
    openshell._validate_agent_action_contract(log, required_reads=required)


def test_context_contains_actual_hamiltonians_and_reaction_coefficients():
    source = FIXTURES / "diff_energy_h2_bond_stretch" / "reaction.json"
    context = benchmark_summary(source, "difference")
    assert len(context["hamiltonians"]) == 2
    assert context["reaction"]["terms"] == json.loads(source.read_text())["terms"]
    for row in context["hamiltonians"]:
        original = json.loads((source.parent / row["source"]).read_text())
        assert row["nonidentity_one_norm"] == pytest.approx(
            sum(abs(t["coefficient"]) for t in original["terms"])
        )
        assert row["term_count"] == len(original["terms"])
        assert (
            row["largest_terms"][0]["coefficient"]
            == max(original["terms"], key=lambda t: abs(t["coefficient"]))["coefficient"]
        )


def test_rejected_proposal_and_evaluator_failure_are_visible_without_sibling_history(tmp_path):
    lane = lane_at(tmp_path)
    other = lane_at(tmp_path, "other")
    for root, score in [(lane, 99), (other, 9999)]:
        it = root / "iterations" / "0001"
        write(
            it / "result.json",
            {
                "status": "ok",
                "score": score,
                "passes_chemical_accuracy": False,
                "absolute_energy_error_mha": 3.0,
            },
        )
        write(
            it / "decision.json",
            {"accepted": False, "acceptance_reason": "chemical_accuracy_failed"},
        )
        (it / "candidate.before.py").write_text(POLICY)
        (it / "candidate.py").write_text(POLICY.replace("self.n = 2", "self.n = 1"))
    failed = lane / "mutation_failures" / "0002-attempt-001"
    write(failed / "mutation_result.json", {"status": "mutated"})
    write(failed / "result.json", {"status": "error", "error": "invalid synthesis epsilon"})
    history = lane_history(lane, lane / "iterations" / "0002")
    assert history["recent_proposals"][0]["score"] == 99
    assert history["recent_proposals"][0]["accepted"] is False
    assert "self.n = 1" in history["recent_proposals"][0]["change"]
    assert history["recent_failures"][0]["error"] == "invalid synthesis epsilon"
    assert "9999" not in json.dumps(history)


def test_duplicate_rejected_policy_is_detected_despite_formatting(tmp_path):
    lane = lane_at(tmp_path)
    previous = lane / "iterations" / "0001"
    write(previous / "result.json", {"status": "ok", "passes_chemical_accuracy": False})
    modified = POLICY.replace("self.n = 2", "self.n = 1")
    (previous / "candidate.py").write_text(modified)
    candidate = tmp_path / "candidate.py"
    candidate.write_text('# Different commentary\n"new docstring"\n' + modified)
    assert duplicate_policy(candidate, lane / "incumbent.py") == "scored iteration 0001"
    candidate.write_text(modified.replace("self.n = 1", "self.n = 3"))
    assert duplicate_policy(candidate, lane / "incumbent.py") is None


def test_output_limit_error_is_preserved_and_blocks_other_lanes(tmp_path):
    lane = lane_at(tmp_path)
    other = lane_at(tmp_path, "other")
    log = tmp_path / "opencode.jsonl"
    write(
        log,
        {
            "type": "error",
            "error": {
                "name": "APIError",
                "data": {
                    "statusCode": 400,
                    "isRetryable": False,
                    "message": "max_tokens cannot exceed the configured model output limit of 16384",
                    "responseHeaders": {"Authorization": "not-for-the-summary"},
                },
            },
        },
    )
    with pytest.raises(AgentRuntimeError, match="max_tokens") as raised:
        openshell._raise_inference_error(log)
    assert raised.value.retryable is False
    record_mutation_failure(lane, raised.value, iteration=1)
    assert mutation_readiness(other)["status"] == "needs_attention"
    assert "not-for-the-summary" not in json.dumps(mutation_readiness(other))
    assert propose_lane(other)["status"] == "needs_attention"
    write(
        tmp_path / "campaign.json",
        {"lanes": [{"lane_dir": "lanes/a"}, {"lane_dir": "lanes/other"}]},
    )
    resume_mutations(tmp_path, reason="Explicit output limit configured and validated")
    assert mutation_readiness(lane) is None
    assert mutation_readiness(other) is None
    assert (tmp_path / "supervisor" / "mutation_recovery.jsonl").is_file()


def test_transient_failure_has_backoff_then_stops_after_three_attempts(tmp_path, monkeypatch):
    lane = lane_at(tmp_path)
    other = lane_at(tmp_path, "other")
    monkeypatch.setattr("hsopt.sandbox.mutation_retry.time.time", lambda: 1000.0)
    failure = AgentRuntimeError("HTTP 503", retryable=True)
    record_mutation_failure(lane, failure, iteration=1)
    assert mutation_readiness(lane)["retry_after_epoch"] == 1030.0
    assert mutation_readiness(other) is None
    record_mutation_failure(lane, failure, iteration=1)
    assert mutation_readiness(lane)["retry_after_epoch"] == 1060.0
    record_mutation_failure(lane, failure, iteration=1)
    assert mutation_readiness(lane)["status"] == "needs_attention"
    clear_mutation_retry(lane)
    assert mutation_readiness(lane) is None


def test_lightning_deployment_limits_and_compaction_are_explicit():
    config=openshell.opencode_config()
    model = openshell.opencode_config()["providers"]["openshell"]["models"][openshell.NEMOTRON_MODEL]
    assert openshell.NEMOTRON_MODEL == 'nvidia/nvidia/nemotron-3.5-lightning'
    assert model["limit"] == {"context":131072,"output":65536}
    assert config['agents'][openshell.OPENCODE_AGENT]['request']['body']['max_tokens']==65536
    assert config['compaction']['auto'] is True


def test_output_reservation_cannot_exhaust_context(monkeypatch):
    monkeypatch.setattr(openshell,'MODEL_OUTPUT_LIMIT',openshell.MODEL_CONTEXT_LIMIT)
    with pytest.raises(ValueError,match='smaller than context'):
        openshell.opencode_config()


def test_metrics_distinguish_failed_fetch_from_successful_search(tmp_path):
    log = tmp_path / "opencode.jsonl"
    events = [
        event("webfetch", "https://example.com", "error"),
        event("websearch", "product formulas"),
    ]
    log.write_text("\n".join(json.dumps(x) for x in events))
    metrics, _ = openshell.summarize_opencode_events(log)
    assert metrics["online_tool_calls"] == 2
    assert metrics["successful_online_tool_calls"] == 1
    assert metrics["by_tool_status"]["webfetch:error"] == 1
    missing, _ = openshell.summarize_opencode_events(tmp_path / "missing.jsonl")
    assert missing["raw_log_present"] is False


def test_cleanup_timeout_does_not_hide_permanent_inference_error(tmp_path, monkeypatch):
    feedback = tmp_path / "feedback.json"
    write(feedback, {"status": "ok", "score": 100})

    def run(command, **kwargs):
        if command[:3] == ["openshell", "sandbox", "delete"]:
            raise subprocess.TimeoutExpired(command, 45)
        return subprocess.CompletedProcess(command, 0, "", "")

    def logged(command, *, stdout_path, **kwargs):
        write(
            stdout_path,
            {
                "type": "error",
                "error": {
                    "name": "APIError",
                    "data": {
                        "statusCode": 400,
                        "isRetryable": False,
                        "message": "max_tokens cannot exceed 16384",
                    },
                },
            },
        )
        return 0, False

    monkeypatch.setattr(openshell, "_run", run)
    monkeypatch.setattr(openshell, "_run_setup_command", lambda *args: None)
    monkeypatch.setattr(openshell, "_run_logged", logged)
    monkeypatch.setattr(openshell, "_capture_openshell_logs", lambda *args: None)
    monkeypatch.setattr(openshell, "_terminate_sandbox_opencode", lambda *args: None)
    run_dir = tmp_path / "trial"
    with pytest.raises(AgentRuntimeError, match="max_tokens"):
        openshell.mutate(
            FIXTURES / "abs_energy_h2" / "initial_pf.py",
            feedback,
            run_dir,
            kind="absolute",
            source=FIXTURES / "abs_energy_h2" / "hamiltonian.json",
        )
    assert json.loads((run_dir / "mutation_result.json").read_text())["retryable"] is False
    assert (run_dir / "cleanup_errors.json").is_file()
    protocol = json.loads((run_dir / "mutation_protocol.json").read_text())
    assert protocol["required_reads"] == ["candidate.py", "research_context.json", "problem.json"]


def test_preparation_disables_edits_and_requires_all_reads(tmp_path):
    config = openshell.opencode_config(preparation=True)
    assert next(r["effect"] for r in config["agents"][openshell.OPENCODE_AGENT]["permissions"] if r["action"] == "edit" and r["resource"] == "candidate.py") == "deny"
    assert (
        next(r["effect"] for r in openshell.opencode_config()["agents"][openshell.OPENCODE_AGENT]["permissions"] if r["action"] == "edit" and r["resource"] == "candidate.py")
        == "allow"
    )
    log = tmp_path / "events.jsonl"
    log.write_text(json.dumps(event("read", "candidate.py")) + "\n")
    with pytest.raises(RuntimeError, match="Preparation missing required reads"):
        openshell._validate_agent_action_contract(
            log, required_reads=["candidate.py", "problem.json"], preparation=True
        )
    with log.open("a") as stream:
        stream.write(json.dumps(event("read", "problem.json")) + "\n")
    openshell._validate_agent_action_contract(
        log, required_reads=["candidate.py", "problem.json"], preparation=True
    )
    with log.open("a") as stream:
        stream.write(json.dumps(event("edit", "candidate.py", "error")) + "\n")
    with pytest.raises(RuntimeError, match="read-only preparation"):
        openshell._validate_agent_action_contract(log, preparation=True)


def test_logged_continuation_preserves_preparation(tmp_path):
    import sys

    stdout, stderr = tmp_path / "out", tmp_path / "err"
    for message, append in [("prepare", False), ("edit", True)]:
        result = openshell._run_logged(
            [sys.executable, "-c", f"print({message!r})"],
            stdout_path=stdout,
            stderr_path=stderr,
            timeout=5,
            append=append,
        )
        assert result == (0, False)
    assert stdout.read_text().splitlines() == ["prepare", "edit"]


@pytest.mark.parametrize("complete_preparation", [False, True])
@pytest.mark.parametrize("blocked_write", [False, True])
def test_host_unlocks_edit_only_after_preparation_receipts(
    tmp_path, monkeypatch, complete_preparation, blocked_write
):
    feedback = tmp_path / "feedback.json"
    write(feedback, {"status": "ok", "score": 100})
    calls, uploads, configs = [], [], {}
    run_dir = tmp_path / "trial"

    def run(command, **kwargs):
        if command[:3] == ["openshell", "sandbox", "upload"]:
            assert "--no-git-ignore" in command
        if command[:3] == ["openshell", "sandbox", "upload"]:
            import tarfile
            with tarfile.open(command[-2]) as archive:
                for name in ["opencode.json", "opencode.mutation.json"]:
                    configs["/sandbox/" + name] = json.load(archive.extractfile(name))
            uploads.append(command[-1])
        if command[:3] == ["openshell", "sandbox", "download"]:
            target = Path(command[-1])
            if len(calls) == 1:
                target.write_bytes((run_dir / "candidate.before.py").read_bytes())
            else:
                target.write_text(POLICY.replace("self.n = 2", "self.n = 3"))
        return subprocess.CompletedProcess(command, 0, "", "")

    def logged(command, *, stdout_path, append=False, **kwargs):
        calls.append(command)
        selected = next(item.split("=", 1)[1] for item in command if item.startswith("OPENCODE_CONFIG="))
        edit_permission = next(rule["effect"] for rule in configs[selected]["agents"][openshell.OPENCODE_AGENT]["permissions"] if rule["action"] == "edit" and rule["resource"] == "candidate.py")
        if len(calls) == 1 or not complete_preparation:
            assert uploads == ["/sandbox/hsopt-bootstrap.tar"]
            assert edit_permission == "deny"
            targets = (
                ["candidate.py", "research_context.json", "problem.json"]
                if complete_preparation
                else ["candidate.py"]
            )
            events = [event("read", target) for target in targets]
            if blocked_write:
                denied = event("edit", "candidate.py", "error")
                denied["part"]["callID"] = f"blocked-{len(calls)}"
                denied["part"]["state"]["error"] = "Permission denied: edit"
                events.insert(0, denied)
        else:
            assert uploads == ["/sandbox/hsopt-bootstrap.tar"]
            assert edit_permission == "allow"
            assert command[command.index("--session") + 1] == "ses_prepare"
            assert append
            events = [event("edit", "candidate.py")]
        with stdout_path.open("a" if append else "w") as stream:
            for item in events:
                stream.write(json.dumps({**item, "sessionID": "ses_prepare"}) + "\n")
        return 0, False

    monkeypatch.setattr(openshell, "_run", run)
    def setup(sandbox, command, run_dir):
        if command[0] == "cp":
            assert complete_preparation
            assert len(calls) == 1
            configs["/sandbox/opencode.json"] = configs["/sandbox/opencode.mutation.json"]

    monkeypatch.setattr(openshell, "_run_setup_command", setup)
    monkeypatch.setattr(openshell, "_run_logged", logged)
    monkeypatch.setattr(openshell, "_capture_openshell_logs", lambda *args: None)
    monkeypatch.setattr(openshell, "_terminate_sandbox_opencode", lambda *args: None)
    args = (FIXTURES / "abs_energy_h2" / "initial_pf.py", feedback, run_dir)
    kwargs = {"kind": "absolute", "source": FIXTURES / "abs_energy_h2" / "hamiltonian.json"}
    if complete_preparation:
        assert openshell.mutate(*args, **kwargs)["status"] == "mutated"
        assert len(calls) == 2
        assert json.loads((run_dir / "opencode_metrics.json").read_text())["tool_calls"] == (
            5 if blocked_write else 4
        )
        if blocked_write:
            assert json.loads((run_dir / "preparation_integrity.json").read_text())["unchanged"]
    else:
        with pytest.raises(RuntimeError, match="Preparation missing required reads"):
            openshell.mutate(*args, **kwargs)
        assert len(calls) == 3
        assert uploads == ["/sandbox/hsopt-bootstrap.tar"]


@pytest.mark.parametrize("status", [401, 500])
@pytest.mark.parametrize(
    "database_error",
    [
        "sorry, too many clients already",
        "remaining connection slots are reserved for roles with the SUPERUSER attribute",
    ],
)
def test_database_capacity_error_is_transient_not_campaign_auth_failure(
    tmp_path, status, database_error
):
    log = tmp_path / "opencode.jsonl"
    write(
        log,
        {
            "type": "error",
            "error": {
                "data": {
                    "statusCode": status,
                    "isRetryable": False,
                    "message": "Authentication Error, Error in connector: Error querying the database: FATAL: "
                    + database_error,
                }
            },
        },
    )
    with pytest.raises(AgentRuntimeError) as raised:
        openshell._raise_inference_error(log)
    assert raised.value.retryable is True
    lane = lane_at(tmp_path)
    other = lane_at(tmp_path, "other")
    record_mutation_failure(lane, raised.value, iteration=1)
    assert mutation_readiness(other) is None


@pytest.mark.parametrize(
    "message",
    [
        "Authentication Error: invalid API key",
        "Authentication Error: remaining connection slots are reserved for roles with the SUPERUSER attribute",
        "Authentication Error, Error querying the database: password authentication failed",
    ],
)
def test_real_auth_error_still_blocks(tmp_path, message):
    log = tmp_path / "opencode.jsonl"
    write(
        log,
        {
            "type": "error",
            "error": {
                "data": {
                    "statusCode": 401,
                    "message": message,
                }
            },
        },
    )
    with pytest.raises(AgentRuntimeError) as raised:
        openshell._raise_inference_error(log)
    assert raised.value.retryable is False


def test_missing_reads_are_recovered_read_only_in_same_session(tmp_path, monkeypatch):
    import time

    calls = []
    required = ["candidate.py", "research_context.json", "reaction.json", "hamiltonians/a.json"]

    def logged(command, *, stdout_path, append, **kwargs):
        calls.append(command)
        reads = required[:2] if len(calls) == 1 else required[2:]
        with stdout_path.open("a" if append else "w") as stream:
            for path in reads:
                stream.write(json.dumps({**event("read", path), "sessionID": "ses_same"}) + "\n")
        return 0, False

    monkeypatch.setattr(openshell, "_run_logged", logged)
    session = openshell._prepare_mutation_context(
        ["openshell", "--timeout", "120", "opencode", "run", "unused"],
        run_dir=tmp_path,
        required_reads=required,
        prompt="read only",
        deadline=time.monotonic() + 120,
    )
    assert session == "ses_same"
    assert len(calls) == 2
    assert "--session" in calls[1]
    assert "Editing is STILL DISABLED" in calls[1][-1]
    assert "/sandbox/work/hamiltonians/a.json" in calls[1][-1]
    assert json.loads((tmp_path / "preparation_result.json").read_text())["read_only_turns"] == 2


def test_documented_builtins_match_immutable_evaluator():
    from hsopt.evaluation.compiled_unitary import _SAFE_POLICY_BUILTINS
    from hsopt.sandbox.research_context import POLICY_CONTRACT

    assert set(POLICY_CONTRACT["available_builtins"]) == {
        key for key in _SAFE_POLICY_BUILTINS if not key.startswith("_")
    }


@pytest.mark.parametrize("message", ["Permission denied: edit", "Permission denied: external_directory"])
@pytest.mark.parametrize("target", ["candidate.py", "problem.json", "scratch_test.py"])
def test_blocked_write_keeps_preparation_session_and_reads(tmp_path, monkeypatch, message, target):
    import time

    calls = []
    required = ["candidate.py", "research_context.json", "problem.json"]

    def logged(command, *, stdout_path, append, **kwargs):
        calls.append(command)
        events = [event("read", target) for target in (required[:1] if len(calls) == 1 else required[1:])]
        if len(calls) == 1:
            denied = event("write", target, "error")
            denied["part"]["state"]["error"] = message
            events.append(denied)
        with stdout_path.open("a" if append else "w") as stream:
            for item in events:
                stream.write(json.dumps({**item, "sessionID": "ses_same"}) + "\n")
        return 0, False

    monkeypatch.setattr(openshell, "_run_logged", logged)
    session = openshell._prepare_mutation_context(
        ["openshell", "--timeout", "120", "opencode", "run", "unused"],
        run_dir=tmp_path, required_reads=required, prompt="read only",
        deadline=time.monotonic() + 120,
    )
    assert session == "ses_same"
    assert len(calls) == 2
    assert calls[1][calls[1].index("--session") + 1] == session
    result = json.loads((tmp_path / "preparation_result.json").read_text())
    assert result["blocked_candidate_writes"] == (1 if target == "candidate.py" else 0)
    # A later successful edit counts once, despite the earlier denial.
    with (tmp_path / "opencode.jsonl").open("a") as stream:
        stream.write(json.dumps(event("edit", "candidate.py")) + "\n")
    openshell._validate_agent_action_contract(tmp_path / "opencode.jsonl", required_reads=required)


@pytest.mark.parametrize("status,error,target", [
    ("completed", None, "candidate.py"),
    ("running", None, "candidate.py"),
    ("error", "disk full", "candidate.py"),
    ("error", "Permission denied: edit after write", "candidate.py"),
])
def test_recovery_does_not_accept_actual_uncertain_or_outside_writes(tmp_path, status, error, target):
    attempt = event("edit", target, status)
    attempt["part"]["state"]["error"] = error
    log = tmp_path / "opencode.jsonl"
    log.write_text(json.dumps(attempt))
    with pytest.raises(RuntimeError):
        openshell._validate_agent_action_contract(log, preparation=True)


def test_failed_outside_read_keeps_context_receipts_but_successful_outside_read_is_rejected(tmp_path):
    required = ["candidate.py", "problem.json"]
    events = [event("read", p) for p in required]
    events.insert(1, event("read", "/etc/unknown", "error"))
    log = tmp_path / "events.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in events))
    openshell._validate_agent_action_contract(log, required_reads=required, preparation=True)
    events[1]["part"]["state"]["status"] = "completed"
    log.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(RuntimeError, match="outside the mutation context"):
        openshell._validate_agent_action_contract(log, required_reads=required, preparation=True)


@pytest.mark.parametrize("changed", [False, True])
def test_recovered_preparation_requires_unchanged_candidate_bytes(tmp_path, monkeypatch, changed):
    import time

    (tmp_path / "candidate.before.py").write_text(POLICY)
    write(tmp_path / "preparation_result.json", {"blocked_candidate_writes": 1})

    def download(*args, **kwargs):
        (tmp_path / "candidate.py").write_text(POLICY + ("# changed\n" if changed else ""))

    monkeypatch.setattr(openshell, "_download_candidate", download)
    if changed:
        with pytest.raises(RuntimeError, match="changed during read-only"):
            openshell._verify_preparation_candidate("test", tmp_path, deadline=time.monotonic() + 120)
    else:
        openshell._verify_preparation_candidate("test", tmp_path, deadline=time.monotonic() + 120)
    assert json.loads((tmp_path / "preparation_integrity.json").read_text())["unchanged"] is not changed


def test_scratch_and_code_mode_do_not_count_as_candidate_mutations(tmp_path):
    log = tmp_path / "opencode.jsonl"
    actions = [event("write", "scratch/smoke.js"), event("read", "scratch/smoke.js"),
               event("execute", "*"), event("edit", "scratch/smoke.js"),
               event("read", "candidate.py"), event("read", "problem.json")]
    log.write_text("\n".join(json.dumps(a) for a in actions))
    openshell._validate_agent_action_contract(log, required_reads=["candidate.py", "problem.json"], preparation=True)
    assert not openshell._has_completed_candidate_edit(log)
    with pytest.raises(RuntimeError, match="exactly one"):
        openshell._validate_agent_action_contract(log)
    log.write_text(log.read_text() + "\n" + json.dumps(event("edit", "candidate.py")))
    openshell._validate_agent_action_contract(log, required_reads=["candidate.py", "problem.json"])
    assert openshell._has_completed_candidate_edit(log)


@pytest.mark.parametrize("target", ["scratch/../candidate.py", "scratch//escape.py", "scratchpad/notes.py"])
def test_scratch_must_remain_in_designated_directory(tmp_path, target):
    log = tmp_path / "opencode.jsonl"
    log.write_text(json.dumps(event("write", target)))
    with pytest.raises(RuntimeError, match="outside candidate.py"):
        openshell._validate_agent_action_contract(log, preparation=True)


def test_code_mode_nested_receipts_enforce_reads_and_candidate_edit(tmp_path):
    log = tmp_path / "opencode.jsonl"
    calls = [{"tool": "opencode.read", "status": "completed", "input": {"path": f"/sandbox/work/{name}"}}
             for name in ["candidate.py", "problem.json"]]
    wrapper = event("execute", "*")
    wrapper["part"]["state"]["metadata"] = {"metadata": {"toolCalls": calls}}
    log.write_text(json.dumps(wrapper))
    openshell._validate_agent_action_contract(log, required_reads=["candidate.py", "problem.json"], preparation=True)
    calls.append({"tool": "opencode.edit", "status": "completed", "input": {"path": "/sandbox/work/candidate.py"}})
    log.write_text(json.dumps(wrapper))
    openshell._validate_agent_action_contract(log, required_reads=["candidate.py", "problem.json"])
    assert openshell._has_completed_candidate_edit(log)
    with pytest.raises(RuntimeError, match="read-only preparation"):
        openshell._validate_agent_action_contract(log, preparation=True)
    calls.append({"tool": "opencode.read", "status": "completed", "input": {"path": "/etc/passwd"}})
    log.write_text(json.dumps(wrapper))
    with pytest.raises(RuntimeError, match="outside the mutation context"):
        openshell._validate_agent_action_contract(log)


def test_code_mode_calculation_errors_are_audited_and_recoverable(tmp_path):
    log = tmp_path / "opencode.jsonl"
    wrapper = event("execute", "*")
    wrapper["part"]["state"]["metadata"] = {"metadata": {"toolCalls": [], "error": True}}
    log.write_text(json.dumps(wrapper))
    openshell._validate_agent_action_contract(log, preparation=True)
    _, actions = openshell.summarize_opencode_events(log)
    assert actions[0]["status"] == "error"


def test_compact_shared_context_preserves_every_case_and_starting_parameters(tmp_path):
    h = {"name": "H2", "n_qubits": 4, "electrons": 2, "term_count": 14,
         "nonidentity_one_norm": 2.0, "target_time": .5,
         "pauli_families": {"diagonal": 10, "off_diagonal": 4},
         "largest_terms": [{"coefficient": -.4, "label": "IIIZ"}],
         "smallest_terms": [{"coefficient": .01, "label": "XXXX"}]}
    cases = [{"case_id": f"case-{i}", "hamiltonians": [h],
              "parameters": {"H2": {"n_steps": i+2, "uniform_epsilon": 1e-7,
                                    "reference_trotter_number_beta": 7.0}}} for i in range(91)]
    original = {"energy_target": "ground_branch", "cases": cases}
    compact = openshell._compact_protocol_benchmark(original)
    path = tmp_path / "research_context.json"
    context = {"benchmark": compact, "required_reads": ["candidate.py", "suite.json"]}
    openshell._write_protocol_context(path, context)
    saved = json.loads(path.read_text())
    assert saved == context
    assert [c["case_id"] for c in saved["benchmark"]["cases"]] == [c["case_id"] for c in cases]
    for index, c in enumerate(saved["benchmark"]["cases"]):
        system = c["hamiltonians"][0]
        assert (system["initial_r"], system["initial_epsilon"], system["initial_beta"]) == (index+2, 1e-7, 7.0)
        assert system["nonidentity_one_norm"] == 2.0
    assert path.stat().st_size < 50*1024
    assert max(map(len, path.read_text().splitlines())) < 2000
    assert original["cases"][0]["parameters"]["H2"]["n_steps"] == 2
