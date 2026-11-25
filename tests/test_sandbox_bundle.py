import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from hsopt.sandbox import openshell


def bundle(tmp_path):
    for name in [
        "candidate.py",
        "hsopt.SKILL.md",
        "rg",
        "opencode.preparation.json",
        "opencode.json",
        "preparation_prompt.md",
        "prompt.md",
        "problem.json",
    ]:
        (tmp_path / name).write_text(name + "\n")
    (tmp_path / "private.txt").write_text("must never be uploaded")
    return openshell._build_sandbox_bundle(
        tmp_path,
        tmp_path / "candidate.py",
        [(tmp_path / "problem.json", PurePosixPath("problem.json"))],
    )


def test_bundle_preserves_exact_inputs_modes_and_excludes_unlisted_files(tmp_path):
    archive = bundle(tmp_path)
    with tarfile.open(archive) as data:
        names = data.getnames()
        assert "private.txt" not in names
        assert data.getmember("work/problem.json").mode == 0o444
        assert data.getmember("work/candidate.py").mode == 0o644
        assert data.getmember("work/.local/bin/rg").mode == 0o755
        assert data.extractfile("work/problem.json").read() == b"problem.json\n"
        assert data.extractfile("opencode.json").read() == b"opencode.preparation.json\n"
        assert data.extractfile("opencode.mutation.json").read() == b"opencode.json\n"


def test_bundled_prompts_skip_extra_transfer_but_followups_keep_verified_transport(
    tmp_path, monkeypatch
):
    bundle(tmp_path)
    calls = []
    monkeypatch.setattr(openshell, "_run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(openshell, "_run_setup_command", lambda *args: calls.append(args))
    prefix = ["openshell", "sandbox", "exec", "--name", "hsopt-test", "--", "opencode", "run"]
    result = openshell._stdin_prompt_command([*prefix, "prompt.md\n"], tmp_path)
    assert calls == []
    assert "/sandbox/hsopt-mutation-prompt.txt" in result
    openshell._stdin_prompt_command([*prefix, "missing read followup"], tmp_path)
    assert len(calls) == 2


@pytest.mark.parametrize("name", ["../secret", "/absolute", "candidate.py"])
def test_bundle_rejects_escaping_or_colliding_context(tmp_path, name):
    bundle(tmp_path)
    with pytest.raises(ValueError):
        openshell._build_sandbox_bundle(
            tmp_path, tmp_path / "candidate.py", [(tmp_path / "problem.json", PurePosixPath(name))]
        )


def test_timing_survives_failed_phase(tmp_path):
    with pytest.raises(RuntimeError):
        with openshell._mutation_phase(tmp_path, "sandbox_setup"):
            raise RuntimeError("failure")
    assert (
        json.loads((tmp_path / "mutation_timing.json").read_text())["phases"]["sandbox_setup"] >= 0
    )


def test_failed_creation_cleans_its_owned_partial_sandbox(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["openshell", "sandbox", "create"]:
            raise subprocess.CalledProcessError(1, command, stderr="registration timed out")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(openshell, "_run", run)
    monkeypatch.setattr(openshell, "_capture_openshell_logs", lambda *args: None)
    fixture = Path(__file__).parent / "fixtures" / "abs_energy_h2"
    feedback = tmp_path / "feedback.json"
    feedback.write_text(json.dumps({"status": "ok", "score": 100}))
    with pytest.raises(subprocess.CalledProcessError):
        openshell.mutate(
            fixture / "initial_pf.py",
            feedback,
            tmp_path / "trial",
            kind="absolute",
            source=fixture / "hamiltonian.json",
        )
    create = next(c for c in calls if c[:3] == ["openshell", "sandbox", "create"])
    name = create[create.index("--name") + 1]
    assert ["openshell", "sandbox", "delete", name] in calls
