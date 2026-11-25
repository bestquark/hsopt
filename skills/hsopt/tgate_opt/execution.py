from __future__ import annotations

import os
import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


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


def merged_environment(invocation: EvaluationInvocation) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(invocation.environment)
    return environment
