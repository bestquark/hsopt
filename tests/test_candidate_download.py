"""File transport retries never grant another mutation or accept an error exit."""

import hashlib
import json
import subprocess

import pytest

from hsopt.sandbox import openshell


def audit_rows(directory):
    return [
        json.loads(row)
        for row in (directory / "candidate_downloads.jsonl").read_text().splitlines()
    ]


def download_stub(tmp_path, monkeypatch, outcomes, *, duration=0):
    calls = []
    clock = [0.0]
    monkeypatch.setattr(openshell.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        openshell.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command == [
            "openshell",
            "sandbox",
            "download",
            "test-sandbox",
            "/sandbox/work/candidate.py",
            str(tmp_path / "candidate.py"),
        ]
        assert kwargs["check"] is False
        clock[0] += duration
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        status, stderr = outcome
        if status == 0:
            (tmp_path / "candidate.py").write_text("downloaded fixture\n")
        return subprocess.CompletedProcess(command, status, "", stderr)

    monkeypatch.setattr(openshell, "_run", run)
    return calls


def test_success_needs_one_download_and_records_hash(tmp_path, monkeypatch):
    calls = download_stub(tmp_path, monkeypatch, [(0, "")])
    openshell._download_candidate("test-sandbox", tmp_path, timeout_seconds=10)
    assert len(calls) == 1
    (row,) = audit_rows(tmp_path)
    assert row["status"] == "downloaded" and row["returncode"] == 0
    assert not row["will_retry"]
    assert row["candidate_sha256"] == hashlib.sha256(b"downloaded fixture\n").hexdigest()


@pytest.mark.parametrize("reason", ["probe", "tar create"])
@pytest.mark.parametrize("failures", [1, 2])
def test_known_errors_retry_identical_transfer_with_shared_timeout(
    tmp_path, monkeypatch, reason, failures
):
    (tmp_path / "candidate.py").write_text("stale local candidate")
    error = f"Downloading sandbox:/sandbox/work/candidate.py\nError:   × ssh {reason} exited with status exit status: 1\n\n"
    calls = download_stub(tmp_path, monkeypatch, [(1, error)] * failures + [(0, "")], duration=1)
    openshell._download_candidate("test-sandbox", tmp_path, timeout_seconds=10)
    assert len(calls) == failures + 1
    assert all(call[0] == calls[0][0] for call in calls)
    timeouts = [call[1]["timeout"] for call in calls]
    assert timeouts[0] == 10 and all(a > b for a, b in zip(timeouts, timeouts[1:]))
    assert all(row["status"] == "failed" and row["will_retry"] for row in audit_rows(tmp_path)[:-1])
    assert audit_rows(tmp_path)[-1]["status"] == "downloaded"
    assert (tmp_path / "candidate.py").read_text() == "downloaded fixture\n"


def test_exhausted_transfer_failure_is_not_success(tmp_path, monkeypatch):
    error = "Error: × ssh probe exited with status exit status: 1\n"
    calls = download_stub(tmp_path, monkeypatch, [(1, error)] * 3)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        openshell._download_candidate("test-sandbox", tmp_path, timeout_seconds=10)
    assert exc.value.returncode == 1 and exc.value.stderr == error
    assert len(calls) == 3
    assert all(row["status"] == "failed" for row in audit_rows(tmp_path))
    assert not audit_rows(tmp_path)[-1]["will_retry"]


@pytest.mark.parametrize(
    "status,stderr",
    [
        (1, "Permission denied"),
        (1, "sandbox not found"),
        (1, "ssh probe exited with status exit status: 10\n"),
        (1, "ssh probe exited with status exit status: 1 but permission denied"),
        (2, "ssh probe exited with status exit status: 1\n"),
        (255, "ssh tar create exited with status exit status: 1\n"),
        (-9, ""),
    ],
)
def test_other_failures_are_not_retried(tmp_path, monkeypatch, status, stderr):
    calls = download_stub(tmp_path, monkeypatch, [(status, stderr)])
    with pytest.raises(subprocess.CalledProcessError):
        openshell._download_candidate("test-sandbox", tmp_path, timeout_seconds=10)
    assert len(calls) == 1 and not audit_rows(tmp_path)[0]["will_retry"]


@pytest.mark.parametrize(
    "error", [subprocess.TimeoutExpired(["download"], 10), OSError("fixture failure")]
)
def test_timeout_and_os_errors_are_preserved_without_retry(tmp_path, monkeypatch, error):
    calls = download_stub(tmp_path, monkeypatch, [error])
    with pytest.raises(type(error)):
        openshell._download_candidate("test-sandbox", tmp_path, timeout_seconds=10)
    assert len(calls) == 1
    assert audit_rows(tmp_path)[0]["error_type"] == type(error).__name__


def test_failure_at_deadline_has_no_additional_attempt(tmp_path, monkeypatch):
    error = "ssh probe exited with status exit status: 1\n"
    calls = download_stub(tmp_path, monkeypatch, [(1, error)], duration=10)
    with pytest.raises(subprocess.CalledProcessError):
        openshell._download_candidate("test-sandbox", tmp_path, timeout_seconds=10)
    assert len(calls) == 1 and not audit_rows(tmp_path)[0]["will_retry"]


@pytest.mark.parametrize("timeout", [0, -1])
def test_invalid_budget_does_not_run_download(tmp_path, monkeypatch, timeout):
    calls = download_stub(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError):
        openshell._download_candidate("test-sandbox", tmp_path, timeout_seconds=timeout)
    assert not calls and not list(tmp_path.iterdir())
