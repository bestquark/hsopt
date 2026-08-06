"""Load the canonical baseline implementation from ``generate_initial_pf.py``.

The scientific definition deliberately lives with the checked-in experiment
generator.  This module is only a runtime adapter for campaign commands that
need to validate or calibrate an initializer; it contains no baseline math.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType


GENERATOR_ENV = "HSOPT_INITIAL_PF_GENERATOR"


def _default_generator() -> Path:
    configured = os.environ.get(GENERATOR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    repository_candidate = (
        Path(__file__).resolve().parents[4] / "experiments" / "generate_initial_pf.py"
    )
    if repository_candidate.is_file():
        return repository_candidate
    return (Path.cwd() / "experiments" / "generate_initial_pf.py").resolve()


@lru_cache(maxsize=None)
def _load_baseline(path_text: str) -> ModuleType:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(
            f"canonical initial-PF generator not found: {path}; "
            f"set {GENERATOR_ENV} when running outside the repository"
        )
    module_name = f"_hsopt_initial_pf_{abs(hash(path_text))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_module(generator: Path | None = None) -> ModuleType:
    """Return the canonical generator module used by campaign infrastructure."""

    path = (generator or _default_generator()).expanduser().resolve()
    return _load_baseline(str(path))


def calibration_main() -> int:
    """Compatibility entry point for the former paper-baseline command."""

    return int(baseline_module().calibration_main())
