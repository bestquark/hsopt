from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from hsopt.compiled_unitary import (
    DIFFERENCE_ERROR_CONTRACTS,
    DIFFERENCE_ERROR_CONTRACT_VERSION,
    EVALUATION_CONTRACT_VERSION,
    evaluate_exact_difference_reference,
    evaluate_exact_reference,
)
from hsopt.exact_queue import merge_batch, run_batch


REPO = Path(__file__).resolve().parents[2]
ABSOLUTE = REPO / "cli" / "tests" / "fixtures" / "abs_energy_h2"
DIFFERENCE = REPO / "cli" / "tests" / "fixtures" / "diff_energy_h2_bond_stretch"
SPEC = importlib.util.spec_from_file_location(
    "compute_iteration0_t95",
    REPO / "plots" / "compute_iteration0_t95.py",
)
assert SPEC is not None and SPEC.loader is not None
CENSUS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CENSUS
SPEC.loader.exec_module(CENSUS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lane(
    campaign: Path,
    *,
    family: str,
    instance: str,
    replicate: str,
    source: Path,
    policy: Path,
    reference: Path,
) -> dict[str, str | None]:
    lane_dir = campaign / "lanes" / family / instance / replicate
    lane_dir.mkdir(parents=True)
    lane_policy = lane_dir / "initial_pf.py"
    lane_reference = lane_dir / "exact_reference.json"
    lane_policy.write_bytes(policy.read_bytes())
    lane_reference.write_bytes(reference.read_bytes())
    kind = "absolute" if family == "abs_energy" else "difference"
    manifest = {
        "schema": 3,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "family": family,
        "instance": instance,
        "replicate": replicate,
        "kind": kind,
        "difference_error_contract": replicate if kind == "difference" else None,
        "difference_error_contract_version": (
            DIFFERENCE_ERROR_CONTRACT_VERSION if kind == "difference" else None
        ),
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "initial_policy_sha256": _sha256(lane_policy),
        "exact_reference_sha256": _sha256(lane_reference),
    }
    _write_json(lane_dir / "manifest.json", manifest)
    return {
        "family": family,
        "instance": instance,
        "replicate": replicate,
        "lane_dir": str(lane_dir.relative_to(campaign)),
        "reference_task_id": None,
    }


def _campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign"
    absolute_reference = tmp_path / "absolute-reference.json"
    difference_reference = tmp_path / "difference-reference.json"
    _write_json(
        absolute_reference,
        evaluate_exact_reference(ABSOLUTE / "hamiltonian.json", backend="numpy"),
    )
    _write_json(
        difference_reference,
        evaluate_exact_difference_reference(DIFFERENCE / "reaction.json", backend="numpy"),
    )
    lanes = [
        _lane(
            campaign,
            family="abs_energy",
            instance="h2",
            replicate=replicate,
            source=ABSOLUTE / "hamiltonian.json",
            policy=ABSOLUTE / "initial_pf.py",
            reference=absolute_reference,
        )
        for replicate in ("run_01", "run_02")
    ]
    lanes.extend(
        _lane(
            campaign,
            family="diff_energy",
            instance="stretch",
            replicate=replicate,
            source=DIFFERENCE / "reaction.json",
            policy=DIFFERENCE / "initial_pf.py",
            reference=difference_reference,
        )
        for replicate in DIFFERENCE_ERROR_CONTRACTS
    )
    _write_json(
        campaign / "campaign.json",
        {
            "schema": 1,
            "generator_sha256": "fixture-generator",
            "lanes": lanes,
        },
    )
    return campaign


def test_census_deduplicates_iteration_zero_and_exports_exact_t95(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    census = tmp_path / "census"
    manifest = CENSUS.prepare_census(
        campaign_root=campaign,
        census_root=census,
        expected_absolute=1,
        expected_reactions=1,
    )

    assert manifest["counts"] == {
        "absolute": 1,
        "reactions": 1,
        "reaction_contracts": 2,
        "total": 3,
    }
    assert manifest["baseline_variant"] is None
    assert all(len(record["reference_scientific_sha256"]) == 64 for record in manifest["tasks"])
    assert {record["source_lane_count"] for record in manifest["tasks"]} == {1, 2}
    batches = CENSUS.prepare_balanced_batches(census, jobs=2)
    assert len(batches) == 2
    assert sorted(len(_read_json(batch / "batch.json")["task_ids"]) for batch in batches) == [1, 2]
    assert CENSUS.census_status(census)["by_family"] == {
        "abs_energy": {"batched": 1},
        "diff_energy": {"batched": 2},
    }

    for batch in batches:
        finished = run_batch(batch, backend="numpy", accelerator="cpu")
        assert finished["failed"] == 0
        merge_batch(census / "queue", batch)
    summary = CENSUS.export_census(census)

    assert summary["row_count"] == 3
    with (census / "results" / "iteration0_t95.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["family"] for row in rows} == {"abs_energy", "diff_energy"}
    assert all(int(row["t95"]) > 0 for row in rows)
    assert all(row["execution_accelerator"] == "cpu" for row in rows)
    reaction_rows = [row for row in rows if row["family"] == "diff_energy"]
    assert {row["difference_error_contract"] for row in reaction_rows} == set(
        DIFFERENCE_ERROR_CONTRACTS
    )
    assert all(row["absolute_signed_sum_error_mha"] for row in reaction_rows)
    assert all(row["sum_absolute_species_error_mha"] for row in reaction_rows)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
