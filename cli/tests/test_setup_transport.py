"""Short setup commands need a receipt; transport success alone is insufficient."""

import json
import subprocess

import pytest

from hsopt.sandbox import openshell
from hsopt.sandbox.mutation_retry import AgentRuntimeError


def mock_exec(monkeypatch, raw_status, command_status, *, output="receipt"):
    def run(invocation, **kwargs):
        assert kwargs == {"check": False, "timeout": 45}
        assert "--no-tty" in invocation
        marker = invocation[invocation.index("hsopt-setup") + 1]
        stdout = f"\n{marker}={command_status}\n"
        if output == "absent":
            stdout = ""
        elif output == "wrong_nonce":
            stdout = "\nHSOPT_SETUP_other=0\n"
        elif output == "duplicate":
            stdout *= 2
        elif output == "extra":
            stdout += "unexpected output\n"
        return subprocess.CompletedProcess(invocation, raw_status, stdout, "diagnostic")

    monkeypatch.setattr(openshell, "_run", run)


@pytest.mark.parametrize("raw_status", [0, 1])
def test_setup_success_requires_actual_zero_receipt(tmp_path, monkeypatch, raw_status):
    mock_exec(monkeypatch, raw_status, 0)
    openshell._run_setup_command("test", ["chmod", "0444", "/sandbox/context.json"], tmp_path)
    audit = json.loads((tmp_path / "setup_commands.jsonl").read_text())
    assert audit["receipt_valid"] and audit["command_returncode"] == 0
    assert audit["transport_returncode"] == raw_status
    assert "error_type" not in audit


@pytest.mark.parametrize("raw_status", [0, 1, 7])
def test_actual_failure_is_never_accepted(tmp_path, monkeypatch, raw_status):
    mock_exec(monkeypatch, raw_status, 7)
    with pytest.raises(subprocess.CalledProcessError) as error:
        openshell._run_setup_command("test", ["mkdir", "-p", "/sandbox/work/a"], tmp_path)
    assert error.value.returncode == 7
    assert json.loads((tmp_path / "setup_commands.jsonl").read_text())["command_returncode"] == 7


@pytest.mark.parametrize("output", ["absent", "wrong_nonce", "duplicate", "extra"])
@pytest.mark.parametrize("raw_status", [0, 1])
def test_missing_or_ambiguous_receipt_fails_closed(tmp_path, monkeypatch, output, raw_status):
    mock_exec(monkeypatch, raw_status, 0, output=output)
    with pytest.raises(AgentRuntimeError) as error:
        openshell._run_setup_command("test", ["chmod", "0444", "/sandbox/context.json"], tmp_path)
    assert error.value.retryable and error.value.category == "sandbox_transport"
    assert not json.loads((tmp_path / "setup_commands.jsonl").read_text())["receipt_valid"]


@pytest.mark.parametrize("raw_status", [-9, 74, 124, 255])
def test_other_transport_failures_are_not_masked(tmp_path, monkeypatch, raw_status):
    mock_exec(monkeypatch, raw_status, 0)
    with pytest.raises(AgentRuntimeError):
        openshell._run_setup_command("test", ["chmod", "0444", "/sandbox/context.json"], tmp_path)


@pytest.mark.parametrize("status", [-1, 256, "invalid"])
def test_invalid_receipt_status_is_rejected(tmp_path, monkeypatch, status):
    mock_exec(monkeypatch, 0, status)
    with pytest.raises(AgentRuntimeError):
        openshell._run_setup_command("test", ["chmod", "0444", "/sandbox/context.json"], tmp_path)


@pytest.mark.parametrize(
    "command",
    [
        ["opencode", "run", "prompt"],
        ["sh", "-c", "true"],
        ["chmod", "777", "/sandbox/a"],
        ["chmod", "0444", "/etc/passwd"],
        ["mkdir", "-p", "/sandbox/../outside"],
        ["chmod", "0444", "/sandbox//"],
    ],
)
def test_agent_commands_and_untrusted_paths_are_not_wrapped(tmp_path, command):
    with pytest.raises(ValueError):
        openshell._run_setup_command("test", command, tmp_path)
    assert not list(tmp_path.iterdir())


def test_wrapper_passes_paths_as_data_and_preserves_permissions(tmp_path, monkeypatch):
    target = tmp_path / 'literal $(touch NEVER); `false` "file"'
    target.write_text("fixture")
    monkeypatch.chdir(tmp_path)

    def run(invocation, **kwargs):
        shell = invocation[invocation.index("--") + 1 :]
        shell[-1] = str(target)
        return subprocess.run(shell, capture_output=True, text=True)

    monkeypatch.setattr(openshell, "_run", run)
    openshell._run_setup_command("test", ["chmod", "0444", "/sandbox/context.json"], tmp_path)
    assert target.stat().st_mode & 0o777 == 0o444
    assert not (tmp_path / "NEVER").exists()


def test_timeouts_are_logged_but_never_retried_inside_setup(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 45)

    monkeypatch.setattr(openshell, "_run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        openshell._run_setup_command("test", ["mkdir", "-p", "/sandbox/work/a"], tmp_path)
    assert len(calls) == 1
    assert (
        json.loads((tmp_path / "setup_commands.jsonl").read_text())["error_type"]
        == "TimeoutExpired"
    )
