from __future__ import annotations

import os
import importlib.util
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .common import REPO_ROOT


ARRAY_BACKEND_ENV = "TROVOLUT_PFS_ARRAY_BACKEND"


@dataclass(frozen=True)
class EvaluationRequest:
    evaluator: Path
    molecule: str
    script: Path
    epsilon: float | None = None


@dataclass(frozen=True)
class EvaluationInvocation:
    transport: str
    backend: str
    command: list[str]
    environment: dict[str, str]
    stdin: str = ""


def _evaluator_arguments(
    request: EvaluationRequest, *, evaluator: str | None = None, script: str | None = None
) -> list[str]:
    arguments = [
        evaluator or str(request.evaluator),
        "--molecule",
        request.molecule,
        "--script",
        script or str(request.script),
    ]
    if request.epsilon is not None:
        arguments.extend(["--epsilon", str(request.epsilon)])
    return arguments


def resolve_local_backend(backend: str) -> str:
    selected = backend.strip().lower()
    if selected not in {"auto", "numpy", "cupynumeric"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'cupynumeric'")
    if selected != "auto":
        return selected
    return (
        "cupynumeric"
        if shutil.which("legate") and importlib.util.find_spec("cupynumeric") is not None
        else "numpy"
    )


def build_local_evaluation(
    request: EvaluationRequest,
    *,
    backend: str = "auto",
    python: str | None = None,
    legate: str | None = None,
) -> EvaluationInvocation:
    selected = resolve_local_backend(backend)
    python = python or sys.executable
    arguments = _evaluator_arguments(request)
    environment = {ARRAY_BACKEND_ENV: selected}
    if selected == "numpy":
        command = [python, *arguments]
    else:
        legate = legate or shutil.which("legate")
        if not legate:
            raise RuntimeError(
                "cuPyNumeric evaluation requested but 'legate' is not installed; "
                "install hsopt[cupynumeric] or select --backend numpy"
            )
        command = [legate, "--gpus", "1", "--run-mode", "exec", python, *arguments]
    return EvaluationInvocation("local", selected, command, environment)


def _remote_path(path: Path, remote_repo_root: str) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        relative = Path(path.name)
    return str(Path(remote_repo_root) / relative)


def build_slurm_submission(
    request: EvaluationRequest,
    *,
    submit_host: str,
    remote_repo_root: str,
    account: str,
    partition: str,
    python: str,
    legate: str,
    proxy_jump_host: str | None = None,
    time_limit: str = "04:00:00",
    memory: str = "64G",
    cpus_per_task: int = 8,
) -> EvaluationInvocation:
    if not submit_host or any(character.isspace() for character in submit_host):
        raise ValueError("submit_host must be a nonempty SSH alias without whitespace")
    if not remote_repo_root.startswith("/"):
        raise ValueError("remote_repo_root must be absolute")
    remote_evaluator = _remote_path(request.evaluator, remote_repo_root)
    remote_script = _remote_path(request.script, remote_repo_root)
    evaluator_command = [
        legate,
        "--gpus",
        "1",
        "--run-mode",
        "exec",
        python,
        *_evaluator_arguments(request, evaluator=remote_evaluator, script=remote_script),
    ]
    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            f"cd {shlex.quote(remote_repo_root)}",
            f"export {ARRAY_BACKEND_ENV}=cupynumeric",
            "export PYTHONUNBUFFERED=1",
            f"exec {shlex.join(evaluator_command)}",
            "",
        ]
    )
    remote = shlex.join(
        [
            "sbatch",
            "--wait",
            "--parsable",
            f"--account={account}",
            f"--partition={partition}",
            "--nodes=1",
            "--gres=gpu:1",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={memory}",
            f"--time={time_limit}",
        ]
    )
    command = ["ssh"]
    if proxy_jump_host:
        command.extend(["-J", proxy_jump_host])
    command.extend([submit_host, remote])
    return EvaluationInvocation("slurm", "cupynumeric", command, {}, script)


def merged_environment(invocation: EvaluationInvocation) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(invocation.environment)
    return environment
