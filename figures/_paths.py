"""Portable paths for the current manuscript, without an Overleaf checkout."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/manuscript.json"


def data_path(original_paper_path: str, expected_sha256: str | None = None) -> Path:
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["data_files"][original_paper_path]
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        record = manifest["historical_source_files"][expected_sha256]
        if record["logical_path"] != original_paper_path:
            raise ValueError("Historical source identity does not match the requested input")
    path = ROOT / record["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Frozen manuscript data changed: {path}")
    return path


def output_path(name: str) -> Path:
    root = Path(os.environ.get("HSOPT_FIGURE_OUTPUT", ROOT / "figures/generated"))
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
