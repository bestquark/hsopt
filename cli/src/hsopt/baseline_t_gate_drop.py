"""Materialize the loose-to-tight iteration-0 T95 drop per experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Mapping

from .compiled_unitary import DIFFERENCE_ERROR_CONTRACTS


FAMILIES = ("abs_energy", "diff_energy")
OUTPUT_NAME = "baseline_t_gate_drop_percent.txt"
REACTION_OUTPUT_TEMPLATE = "baseline_t_gate_drop_{contract}_percent.txt"
OBJECTIVE_FIELD = "t95"
PERCENT_QUANTUM = Decimal("0.000001")


class BaselineTGateDropError(RuntimeError):
    """Raised when two censuses cannot support an isolated baseline comparison."""


@dataclass(frozen=True)
class CensusObservation:
    family: str
    instance: str
    difference_error_contract: str | None
    task_id: str
    source_sha256: str
    policy_sha256: str
    reference_sha256: str
    reference_scientific_sha256: str
    result_sha256: str
    backend: str
    accelerator: str
    evaluation_contract_version: str
    t95: int
    passes_chemical_accuracy: bool


@dataclass(frozen=True)
class Census:
    root: Path
    manifest_path: Path
    results_manifest_path: Path
    csv_path: Path
    campaign_variant: str | None
    observations: dict[tuple[str, str, str | None], CensusObservation]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BaselineTGateDropError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineTGateDropError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineTGateDropError(f"expected a JSON object at {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_component(value: object, *, label: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise BaselineTGateDropError(f"unsafe {label}: {text!r}")
    return text


def _positive_integer(value: object, *, label: str) -> int:
    try:
        count = int(str(value))
    except (TypeError, ValueError) as exc:
        raise BaselineTGateDropError(f"{label} must be a positive integer") from exc
    if count <= 0 or str(count) != str(value):
        raise BaselineTGateDropError(f"{label} must be a positive integer")
    return count


def _boolean(value: object, *, label: str) -> bool:
    text = str(value).lower()
    if text not in {"true", "false"}:
        raise BaselineTGateDropError(f"{label} must be true or false")
    return text == "true"


def _digest(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BaselineTGateDropError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _campaign_variant(manifest: Mapping[str, Any], *, census_manifest: Path) -> str | None:
    if "baseline_variant" in manifest:
        value = manifest["baseline_variant"]
        return str(value) if value is not None else None
    campaign_root = Path(str(manifest.get("campaign_root") or "")).expanduser()
    if not campaign_root.is_absolute():
        raise BaselineTGateDropError(f"{census_manifest}: campaign_root must be absolute")
    campaign_path = campaign_root / "campaign.json"
    campaign = _read_json(campaign_path)
    expected_sha256 = str(manifest.get("campaign_manifest_sha256") or "")
    if _sha256(campaign_path) != expected_sha256:
        raise BaselineTGateDropError(f"{census_manifest}: source campaign changed after census")
    value = campaign.get("baseline_variant")
    return str(value) if value is not None else None


def _load_census(root: Path) -> Census:
    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    results_manifest_path = root / "results/iteration0_t95_manifest.json"
    manifest = _read_json(manifest_path)
    results_manifest = _read_json(results_manifest_path)
    if results_manifest.get("census_manifest_sha256") != _sha256(manifest_path):
        raise BaselineTGateDropError(f"{results_manifest_path}: census manifest hash mismatch")
    contract = str(results_manifest.get("evaluation_contract_version") or "")
    if not contract or contract != str(manifest.get("evaluation_contract_version") or ""):
        raise BaselineTGateDropError(f"{results_manifest_path}: evaluator contract mismatch")
    csv_path = root / "results" / str(results_manifest.get("csv") or "")
    if not csv_path.is_file() or results_manifest.get("csv_sha256") != _sha256(csv_path):
        raise BaselineTGateDropError(f"{results_manifest_path}: result CSV is missing or changed")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise BaselineTGateDropError(f"{manifest_path}: census has no tasks")
    tasks_by_id: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise BaselineTGateDropError(f"{manifest_path}: task is not an object")
        task_id = _safe_component(task.get("task_id"), label="task id")
        if task_id in tasks_by_id:
            raise BaselineTGateDropError(f"{manifest_path}: duplicate task {task_id}")
        tasks_by_id[task_id] = task

    observations: dict[tuple[str, str, str | None], CensusObservation] = {}
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(results_manifest.get("row_count", -1)) or len(rows) != len(tasks_by_id):
        raise BaselineTGateDropError(f"{csv_path}: incomplete census row grid")
    for row in rows:
        task_id = _safe_component(row.get("task_id"), label="task id")
        if task_id not in tasks_by_id:
            raise BaselineTGateDropError(f"{csv_path}: unknown task {task_id}")
        task = tasks_by_id.pop(task_id)
        family = _safe_component(row.get("family"), label="family")
        instance = _safe_component(row.get("instance"), label="instance")
        if family not in FAMILIES:
            raise BaselineTGateDropError(f"{csv_path}: unsupported family {family!r}")
        if task.get("family") != family or task.get("instance") != instance:
            raise BaselineTGateDropError(f"{csv_path}: task identity disagrees with manifest")
        raw_difference_contract = str(row.get("difference_error_contract") or "")
        difference_error_contract = raw_difference_contract or None
        if family == "diff_energy":
            if difference_error_contract not in DIFFERENCE_ERROR_CONTRACTS:
                raise BaselineTGateDropError(
                    f"{csv_path}: {instance} has unknown reaction-error contract "
                    f"{difference_error_contract!r}"
                )
        elif difference_error_contract is not None:
            raise BaselineTGateDropError(
                f"{csv_path}: absolute experiment {instance} has a reaction-error contract"
            )
        if task.get("difference_error_contract") != difference_error_contract:
            raise BaselineTGateDropError(
                f"{csv_path}: task reaction-error contract disagrees with result row"
            )
        key = (family, instance, difference_error_contract)
        if key in observations:
            raise BaselineTGateDropError(f"{csv_path}: duplicate experiment {key}")
        policy_sha256 = _digest(row.get("policy_sha256"), label=f"{key} policy hash")
        reference_sha256 = _digest(row.get("reference_sha256"), label=f"{key} reference hash")
        if policy_sha256 != task.get("policy_sha256"):
            raise BaselineTGateDropError(f"{csv_path}: {key} policy hash disagrees with manifest")
        if reference_sha256 != task.get("reference_sha256"):
            raise BaselineTGateDropError(
                f"{csv_path}: {key} reference hash disagrees with manifest"
            )
        observations[key] = CensusObservation(
            family=family,
            instance=instance,
            difference_error_contract=difference_error_contract,
            task_id=task_id,
            source_sha256=_digest(task.get("source_sha256"), label=f"{key} source hash"),
            policy_sha256=policy_sha256,
            reference_sha256=reference_sha256,
            reference_scientific_sha256=_digest(
                task.get("reference_scientific_sha256"),
                label=f"{key} scientific reference hash",
            ),
            result_sha256=_digest(row.get("result_sha256"), label=f"{key} result hash"),
            backend=_safe_component(row.get("execution_backend"), label=f"{key} backend"),
            accelerator=_safe_component(
                row.get("execution_accelerator"), label=f"{key} accelerator"
            ),
            evaluation_contract_version=contract,
            t95=_positive_integer(row.get("t95"), label=f"{key} T95"),
            passes_chemical_accuracy=_boolean(
                row.get("passes_chemical_accuracy"), label=f"{key} accuracy status"
            ),
        )
    if tasks_by_id:
        raise BaselineTGateDropError(f"{csv_path}: missing tasks {sorted(tasks_by_id)}")
    return Census(
        root=root,
        manifest_path=manifest_path,
        results_manifest_path=results_manifest_path,
        csv_path=csv_path,
        campaign_variant=_campaign_variant(manifest, census_manifest=manifest_path),
        observations=observations,
    )


def _expected_instances(repository_root: Path) -> set[tuple[str, str, str | None]]:
    experiments = repository_root / "experiments"
    expected: set[tuple[str, str]] = set()
    for family, descriptor in (
        ("abs_energy", "instance.json"),
        ("diff_energy", "definition.json"),
    ):
        family_root = experiments / family
        if not family_root.is_dir():
            raise BaselineTGateDropError(f"missing experiment family: {family_root}")
        for path in family_root.glob(f"*/{descriptor}"):
            if not path.is_file():
                continue
            if family == "abs_energy":
                expected.add((family, path.parent.name, None))
            else:
                expected.update(
                    (family, path.parent.name, contract) for contract in DIFFERENCE_ERROR_CONTRACTS
                )
    if not expected:
        raise BaselineTGateDropError(f"no experiment instances found under {experiments}")
    return expected


def _format_percent(loose_t95: int, tight_t95: int) -> tuple[Decimal, str]:
    percentage = (
        Decimal(100) * (Decimal(loose_t95) - Decimal(tight_t95)) / Decimal(loose_t95)
    ).quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_EVEN)
    return percentage, f"{percentage:f}\n"


def materialize_baseline_t_gate_drop(
    loose_census_root: Path,
    tight_census_root: Path,
    repository_root: Path,
    *,
    loose_placement: str,
    tight_placement: str,
    allow_legacy_loose: bool = False,
    check: bool = False,
    audit_output: Path | None = None,
) -> dict[str, Any]:
    """Write one signed loose-to-tight T95 percentage per scientific contract."""

    repository_root = repository_root.expanduser().resolve()
    loose_placement = _safe_component(loose_placement, label="loose placement")
    tight_placement = _safe_component(tight_placement, label="tight placement")
    if loose_placement != tight_placement:
        raise BaselineTGateDropError(
            f"refusing to mix placements {loose_placement!r} and {tight_placement!r}"
        )
    loose = _load_census(loose_census_root)
    tight = _load_census(tight_census_root)
    if loose.root == tight.root:
        raise BaselineTGateDropError("loose and tight census roots must be different")
    if loose.campaign_variant != "loose" and not (
        allow_legacy_loose and loose.campaign_variant is None
    ):
        raise BaselineTGateDropError(
            f"{loose.manifest_path}: census is not sourced from a loose campaign"
        )
    if tight.campaign_variant != "tight":
        raise BaselineTGateDropError(
            f"{tight.manifest_path}: census is not sourced from a tight campaign"
        )
    if set(loose.observations) != set(tight.observations):
        missing_tight = sorted(set(loose.observations) - set(tight.observations))
        missing_loose = sorted(set(tight.observations) - set(loose.observations))
        raise BaselineTGateDropError(
            f"census grids differ: missing_tight={missing_tight}, missing_loose={missing_loose}"
        )
    expected = _expected_instances(repository_root)
    if set(loose.observations) != expected:
        missing = sorted(expected - set(loose.observations))
        unexpected = sorted(set(loose.observations) - expected)
        raise BaselineTGateDropError(
            f"baseline comparison is incomplete: missing={missing}, unexpected={unexpected}"
        )

    records: list[dict[str, Any]] = []
    writes: list[tuple[Path, str]] = []
    for key in sorted(expected):
        loose_row = loose.observations[key]
        tight_row = tight.observations[key]
        if loose_row.source_sha256 != tight_row.source_sha256:
            raise BaselineTGateDropError(f"{key}: loose and tight source hashes differ")
        if loose_row.reference_scientific_sha256 != tight_row.reference_scientific_sha256:
            raise BaselineTGateDropError(
                f"{key}: loose and tight scientific exact references differ"
            )
        if loose_row.evaluation_contract_version != tight_row.evaluation_contract_version:
            raise BaselineTGateDropError(f"{key}: evaluator contract versions differ")
        if (loose_row.backend, loose_row.accelerator) != (
            tight_row.backend,
            tight_row.accelerator,
        ):
            raise BaselineTGateDropError(f"{key}: refusing to mix execution backends/accelerators")
        if not loose_row.passes_chemical_accuracy or not tight_row.passes_chemical_accuracy:
            raise BaselineTGateDropError(
                f"{key}: both baselines must pass chemical accuracy before comparison"
            )
        percentage, encoded = _format_percent(loose_row.t95, tight_row.t95)
        family, instance, difference_error_contract = key
        output_name = (
            OUTPUT_NAME
            if difference_error_contract is None
            else REACTION_OUTPUT_TEMPLATE.format(contract=difference_error_contract)
        )
        output = repository_root / "experiments" / family / instance / output_name
        if check:
            if not output.is_file() or output.read_text(encoding="utf-8") != encoded:
                raise BaselineTGateDropError(f"stale baseline T-gate drop: {output}")
        else:
            writes.append((output, encoded))
        records.append(
            {
                "family": family,
                "instance": instance,
                "difference_error_contract": difference_error_contract,
                "placement": loose_placement,
                "execution_backend": loose_row.backend,
                "execution_accelerator": loose_row.accelerator,
                "evaluation_contract_version": loose_row.evaluation_contract_version,
                "source_sha256": loose_row.source_sha256,
                "reference_scientific_sha256": loose_row.reference_scientific_sha256,
                "loose_reference_sha256": loose_row.reference_sha256,
                "tight_reference_sha256": tight_row.reference_sha256,
                "loose_policy_sha256": loose_row.policy_sha256,
                "loose_result_sha256": loose_row.result_sha256,
                "loose_t95": loose_row.t95,
                "tight_policy_sha256": tight_row.policy_sha256,
                "tight_result_sha256": tight_row.result_sha256,
                "tight_t95": tight_row.t95,
                "drop_percent": float(percentage),
                "output": str(output.relative_to(repository_root)),
            }
        )

    for output, encoded in writes:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)

    result = {
        "schema": 2,
        "status": "verified" if check else "materialized",
        "objective": "confidence_adjusted_t_count",
        "definition": "100 * (T95_loose - T95_tight) / T95_loose",
        "placement": loose_placement,
        "loose_census_root": str(loose.root),
        "loose_census_manifest_sha256": _sha256(loose.manifest_path),
        "loose_results_manifest_sha256": _sha256(loose.results_manifest_path),
        "tight_census_root": str(tight.root),
        "tight_census_manifest_sha256": _sha256(tight.manifest_path),
        "tight_results_manifest_sha256": _sha256(tight.results_manifest_path),
        "repository_root": str(repository_root),
        "instance_count": len({(row["family"], row["instance"]) for row in records}),
        "comparison_count": len(records),
        "records": records,
    }
    if audit_output is not None:
        audit_output = audit_output.expanduser().resolve()
        audit_payload = {**result, "status": "complete"}
        encoded = json.dumps(audit_payload, indent=2, sort_keys=True) + "\n"
        if check:
            if not audit_output.is_file() or audit_output.read_text(encoding="utf-8") != encoded:
                raise BaselineTGateDropError(f"stale baseline T-gate audit: {audit_output}")
        else:
            audit_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = audit_output.with_suffix(audit_output.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(audit_output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loose-census-root", type=Path, required=True)
    parser.add_argument("--tight-census-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--loose-placement", required=True)
    parser.add_argument("--tight-placement", required=True)
    parser.add_argument("--allow-legacy-loose", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = materialize_baseline_t_gate_drop(
        args.loose_census_root,
        args.tight_census_root,
        args.repository_root,
        loose_placement=args.loose_placement,
        tight_placement=args.tight_placement,
        allow_legacy_loose=args.allow_legacy_loose,
        check=args.check,
        audit_output=args.audit_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
