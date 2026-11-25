"""Long prompts must survive OpenShell transport without executing their text."""

import subprocess
from pathlib import Path

from hsopt.sandbox import openshell


def test_long_unicode_prompt_is_staged_without_truncation(tmp_path, monkeypatch):
    prompt = 'Hamiltonian Δ; "quotes"; $HOME; $(touch NEVER); `false`\n' * 4000
    calls = []
    monkeypatch.setattr(openshell, "_run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(
        openshell, "_run_setup_command", lambda name, args, path: calls.append(args)
    )
    command = [
        "openshell",
        "sandbox",
        "exec",
        "--name",
        "hsopt-test",
        "--timeout",
        "900",
        "--",
        "opencode",
        "run",
        "--format",
        "json",
        "--session",
        "ses_test",
        prompt,
    ]
    actual = openshell._stdin_prompt_command(command, tmp_path)
    files = list(tmp_path.glob("transport-prompt-*.txt"))
    assert len(files) == 1
    assert files[0].read_bytes() == prompt.encode("utf-8")
    assert prompt not in actual
    assert max(len(arg.encode("utf-8")) for arg in actual) < 32768
    assert actual[-6:] == ["opencode", "run", "--format", "json", "--session", "ses_test"]
    assert calls[0] == [
        "openshell",
        "sandbox",
        "upload",
        "--no-git-ignore",
        "hsopt-test",
        str(files[0]),
        "/sandbox/" + files[0].name,
    ]
    assert calls[1][-3:] == ["chmod", "0444", "/sandbox/" + files[0].name]

    # Exercise the actual fixed shell wrapper; the prompt is data, never code.
    shell = actual[actual.index("--") + 1 :]
    shell[4] = str(files[0])
    result = subprocess.run(shell[:5] + ["/bin/cat"], capture_output=True, check=True)
    assert result.stdout == prompt.encode("utf-8")
    assert not Path("NEVER").exists()


def test_each_turn_retains_its_own_prompt_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(openshell, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(openshell, "_run_setup_command", lambda *args: None)
    prefix = ["openshell", "sandbox", "exec", "--name", "test", "--", "opencode", "run"]
    first = openshell._stdin_prompt_command([*prefix, "read only"], tmp_path)
    second = openshell._stdin_prompt_command(
        [*prefix, "--session", "ses_test", "edit now"], tmp_path
    )
    assert first != second
    assert {p.read_text() for p in tmp_path.glob("transport-prompt-*.txt")} == {
        "read only",
        "edit now",
    }


def test_unrelated_commands_are_unchanged(tmp_path):
    for command in (
        ["python", "-c", "print('ok')"],
        ["openshell", "sandbox", "exec", "--", "true"],
    ):
        assert openshell._stdin_prompt_command(command, tmp_path) == command
    assert not list(tmp_path.iterdir())
