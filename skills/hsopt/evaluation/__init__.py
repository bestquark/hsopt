"""Selectable exact evaluation and approximate diagnostics."""

from pathlib import Path
from typing import Any

from . import compiled_unitary

METHODS = ("exact", "bch")


def _validate_options(method: str, reference_path: Path | None, reference_size: int | None) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "bch" and reference_path is not None:
        raise ValueError("BCH uses a selected-determinant reference, not --exact-reference")
    if method == "exact" and reference_size is not None:
        raise ValueError("--reference-size applies only to the BCH proxy")


def evaluate_molecule(
    problem_path: Path,
    policy_path: Path,
    *,
    method: str = "exact",
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
    reference_size: int | None = None,
) -> dict[str, Any]:
    """Select an evaluator without changing the exact production contract."""
    _validate_options(method, reference_path, reference_size)
    if method == "exact":
        return compiled_unitary.evaluate_molecule(
            problem_path, policy_path, backend=backend, reference_path=reference_path
        )
    from .proxies import bch

    return bch.evaluate_molecule(
        problem_path,
        policy_path,
        reference_size=reference_size if reference_size is not None else bch.DEFAULT_REFERENCE_SIZE,
    )


def evaluate_difference(
    reaction_path: Path,
    policy_path: Path,
    *,
    error_contract: str,
    method: str = "exact",
    backend: str = "cupynumeric",
    reference_path: Path | None = None,
    reference_size: int | None = None,
) -> dict[str, Any]:
    """Select exact reaction scoring or aggregation of estimated species biases."""
    _validate_options(method, reference_path, reference_size)
    if method == "exact":
        return compiled_unitary.evaluate_difference(
            reaction_path,
            policy_path,
            error_contract=error_contract,
            backend=backend,
            reference_path=reference_path,
        )
    from .proxies import bch

    return bch.evaluate_difference(
        reaction_path,
        policy_path,
        error_contract=error_contract,
        reference_size=reference_size if reference_size is not None else bch.DEFAULT_REFERENCE_SIZE,
    )
