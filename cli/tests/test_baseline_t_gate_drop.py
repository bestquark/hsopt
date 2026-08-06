from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from hsopt.baseline_t_gate_drop import (
    OUTPUT_NAME,
    REACTION_OUTPUT_TEMPLATE,
    BaselineTGateDropError,
    materialize_baseline_t_gate_drop,
)
from hsopt.compiled_unitary import DIFFERENCE_ERROR_CONTRACTS


CSV_FIELDS = (
    "family",
    "instance",
    "difference_error_contract",
    "task_id",
    "t95",
    "passes_chemical_accuracy",
    "execution_backend",
    "execution_accelerator",
    "policy_sha256",
    "reference_sha256",
    "result_sha256",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository(root: Path) -> Path:
    repository = root / "repo"
    _write_json(repository / "experiments/abs_energy/molecule/instance.json", {})
    _write_json(repository / "experiments/diff_energy/reaction/definition.json", {})
    return repository


def _census(
    root: Path,
    *,
    variant: str | None,
    absolute_t95: int,
    reaction_t95: int,
    backend: str = "cupynumeric",
    accelerator: str = "gpu",
    reaction_passes: bool = True,
) -> Path:
    label = variant or "legacy"
    census = root / f"{label}-census"
    campaign = root / f"{label}-campaign"
    _write_json(campaign / "campaign.json", {"baseline_variant": variant})
    rows = []
    tasks = []
    absolute_reference_sha256 = ("3" if variant == "tight" else "c") * 64
    reaction_reference_sha256 = ("4" if variant == "tight" else "d") * 64
    observations = [
        (
            "abs_energy",
            "molecule",
            None,
            absolute_t95,
            "a" * 64,
            absolute_reference_sha256,
        ),
        *[
            (
                "diff_energy",
                "reaction",
                contract,
                reaction_t95,
                "b" * 64,
                reaction_reference_sha256,
            )
            for contract in DIFFERENCE_ERROR_CONTRACTS
        ],
    ]
    for (
        family,
        instance,
        difference_error_contract,
        t95,
        source_sha256,
        reference_sha256,
    ) in observations:
        task_id = f"{label}-{family}-{difference_error_contract or 'absolute'}"
        policy_sha256 = ("e" if variant == "loose" else "f") * 64
        result_sha256 = ("1" if family == "abs_energy" else "2") * 64
        tasks.append(
            {
                "family": family,
                "instance": instance,
                "difference_error_contract": difference_error_contract,
                "task_id": task_id,
                "source_sha256": source_sha256,
                "policy_sha256": policy_sha256,
                "reference_sha256": reference_sha256,
                "reference_scientific_sha256": ("8" if family == "abs_energy" else "9") * 64,
            }
        )
        rows.append(
            {
                "family": family,
                "instance": instance,
                "difference_error_contract": difference_error_contract or "",
                "task_id": task_id,
                "t95": t95,
                "passes_chemical_accuracy": (family != "diff_energy" or reaction_passes),
                "execution_backend": backend,
                "execution_accelerator": accelerator,
                "policy_sha256": policy_sha256,
                "reference_sha256": reference_sha256,
                "result_sha256": result_sha256,
            }
        )
    manifest = census / "manifest.json"
    _write_json(
        manifest,
        {
            "evaluation_contract_version": "compiled-unitary-v1",
            "baseline_variant": variant,
            "campaign_root": str(campaign.resolve()),
            "campaign_manifest_sha256": _sha256(campaign / "campaign.json"),
            "tasks": tasks,
        },
    )
    csv_path = census / "results/iteration0_t95.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        census / "results/iteration0_t95_manifest.json",
        {
            "evaluation_contract_version": "compiled-unitary-v1",
            "census_manifest_sha256": _sha256(manifest),
            "row_count": len(rows),
            "csv": csv_path.name,
            "csv_sha256": _sha256(csv_path),
        },
    )
    return census


def _materialize(
    loose: Path,
    tight: Path,
    repository: Path,
    **kwargs: object,
) -> dict[str, object]:
    return materialize_baseline_t_gate_drop(
        loose,
        tight,
        repository,
        loose_placement=str(kwargs.pop("loose_placement", "omni-l4")),
        tight_placement=str(kwargs.pop("tight_placement", "omni-l4")),
        **kwargs,
    )


def test_materializes_one_objective_comparable_number_per_experiment(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    loose = _census(
        tmp_path,
        variant="loose",
        absolute_t95=1000,
        reaction_t95=2500,
    )
    tight = _census(
        tmp_path,
        variant="tight",
        absolute_t95=600,
        reaction_t95=2000,
    )
    audit = tmp_path / "audit.json"

    result = _materialize(loose, tight, repository, audit_output=audit)

    assert result["status"] == "materialized"
    assert result["objective"] == "confidence_adjusted_t_count"
    assert result["instance_count"] == 2
    assert result["comparison_count"] == 3
    assert (
        repository / "experiments/abs_energy/molecule" / OUTPUT_NAME
    ).read_text() == "40.000000\n"
    for contract in DIFFERENCE_ERROR_CONTRACTS:
        assert (
            repository
            / "experiments/diff_energy/reaction"
            / REACTION_OUTPUT_TEMPLATE.format(contract=contract)
        ).read_text() == "20.000000\n"
    assert json.loads(audit.read_text())["records"][0]["loose_t95"] == 1000

    checked = _materialize(loose, tight, repository, check=True, audit_output=audit)
    assert checked["status"] == "verified"


def test_refuses_cross_placement_or_execution_backend_comparisons(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    loose = _census(
        tmp_path,
        variant="loose",
        absolute_t95=1000,
        reaction_t95=2500,
    )
    tight = _census(
        tmp_path,
        variant="tight",
        absolute_t95=600,
        reaction_t95=2000,
        accelerator="cpu",
    )
    with pytest.raises(BaselineTGateDropError, match="refusing to mix placements"):
        _materialize(loose, tight, repository, tight_placement="a100")
    with pytest.raises(BaselineTGateDropError, match="backends/accelerators"):
        _materialize(loose, tight, repository)


def test_refuses_incomplete_or_inaccurate_censuses_without_partial_outputs(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    loose = _census(
        tmp_path,
        variant="loose",
        absolute_t95=1000,
        reaction_t95=2500,
    )
    tight = _census(
        tmp_path,
        variant="tight",
        absolute_t95=600,
        reaction_t95=2000,
        reaction_passes=False,
    )
    with pytest.raises(BaselineTGateDropError, match="must pass chemical accuracy"):
        _materialize(loose, tight, repository)
    assert not list(repository.rglob(OUTPUT_NAME))

    rows = list(csv.DictReader((tight / "results/iteration0_t95.csv").open(encoding="utf-8")))
    rows.pop()
    with (tight / "results/iteration0_t95.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    results_manifest = tight / "results/iteration0_t95_manifest.json"
    payload = json.loads(results_manifest.read_text())
    payload["csv_sha256"] = _sha256(tight / "results/iteration0_t95.csv")
    _write_json(results_manifest, payload)
    with pytest.raises(BaselineTGateDropError, match="incomplete census row grid"):
        _materialize(loose, tight, repository)
    assert not list(repository.rglob(OUTPUT_NAME))


def test_legacy_loose_requires_an_explicit_audited_override(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    loose = _census(
        tmp_path,
        variant=None,
        absolute_t95=1000,
        reaction_t95=2500,
    )
    tight = _census(
        tmp_path,
        variant="tight",
        absolute_t95=600,
        reaction_t95=2000,
    )
    with pytest.raises(BaselineTGateDropError, match="not sourced from a loose campaign"):
        _materialize(loose, tight, repository)
    result = _materialize(loose, tight, repository, allow_legacy_loose=True)
    assert result["instance_count"] == 2
