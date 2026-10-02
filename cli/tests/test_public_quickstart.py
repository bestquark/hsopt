"""Exercise the installed commands and README examples without private services."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = sorted(
    entry.name
    for entry in importlib.metadata.distribution("hsopt").entry_points
    if entry.group == "console_scripts"
)


@pytest.mark.parametrize("command", COMMANDS)
def test_installed_command_help(command):
    result = subprocess.run(
        [str(Path(sys.executable).parent / command), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize(
    "fixture,arguments",
    [
        ("abs_energy_h2", []),
        ("diff_energy_h2_bond_stretch", ["--error-contract", "error_cancellation"]),
    ],
)
def test_readme_cpu_example(fixture, arguments):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/tests/fixtures" / fixture / "evaluate.py"),
            "--backend",
            "numpy",
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["passes_chemical_accuracy"] is True
    assert payload["confidence_adjusted_t_count"] > 0
