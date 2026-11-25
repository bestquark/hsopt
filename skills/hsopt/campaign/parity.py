"""Certify completed CPU and A100 campaigns regardless of replay direction."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


ENERGY_FIELDS = (
    "exact_unitary_ground_energy_hartree",
    "exact_hf_accessible_energy_hartree",
    "compiled_pf_energy_hartree",
    "absolute_energy_error_hartree",
    "exact_difference_hartree",
    "compiled_pf_difference_hartree",
    "signed_difference_error_hartree",
    "absolute_difference_error_hartree",
    "sum_absolute_species_error_hartree",
    "contract_error_hartree",
    "error_cancellation_gain_hartree",
)
EXACT_METADATA_FIELDS = (
    "evaluation_contract_version",
    "problem_sha256",
    "observable",
    "unitary_definition",
    "target_state_definition",
    "target_anchor_kind",
    "target_anchor_basis_index",
    "exact_ground_subspace_dimension",
)
COMPILED_FIELDS = (
    "status",
    "evaluation_contract_version",
    "rotation_synthesis",
    "observable",
    "policy",
    "phase_encoding",
    "compiled_evolution_t_count",
    "per_attempt_t_count",
    "per_qpe_run_t_count",
    "qpe_energy_precision_hartree",
    "qpe_phase_precision_radians",
    "qpe_full_evolution_queries",
    "qpe_repetitions",
    "qpe_success_probability_per_attempt",
    "confidence_adjusted_t_count",
    "t_count",
    "score",
    "passes_chemical_accuracy",
    "difference_error_contract",
    "difference_error_contract_version",
    "passes_individual_errors_contract",
    "passes_error_cancellation_contract",
    "synthesis_epsilon_histogram",
    "scheduled_rotation_count",
    "skipped_rotation_count",
    "one_step_t_count",
    "dense_matrix_dimension",
    "unique_compilations",
)
DECISION_FIELDS = (
    "accepted",
    "acceptance_reason",
    "candidate_score",
    "previous_incumbent_score",
    "passes_chemical_accuracy",
)


class ParityAuditPending(RuntimeError):
    """Raised while one or both campaigns still have unfinished lanes."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or unsafe campaign record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _lane_map(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    campaign = _read_json(root / "campaign.json")
    lanes: dict[str, Path] = {}
    for row in campaign.get("lanes", []):
        relative = Path(str(row["lane_dir"]))
        lane = (root / relative).resolve()
        if not lane.is_relative_to(root):
            raise RuntimeError(f"lane escapes campaign root: {relative}")
        key = relative.as_posix()
        if key in lanes:
            raise RuntimeError(f"duplicate campaign lane: {key}")
        lanes[key] = lane
    if not lanes:
        raise RuntimeError(f"campaign has no lanes: {root}")
    return campaign, lanes


def _require_complete(root: Path, lanes: Mapping[str, Path]) -> None:
    if not (root / "campaign_complete.json").is_file():
        raise ParityAuditPending(f"campaign is not complete: {root}")
    for key, lane in lanes.items():
        manifest = _read_json(lane / "manifest.json")
        state = _read_json(lane / "state.json")
        if int(state.get("iterations_completed", -1)) != int(manifest["iterations_target"]):
            raise ParityAuditPending(f"lane is incomplete: {key}")
        if state.get("pending_kind") is not None or state.get("pending_task_id") is not None:
            raise ParityAuditPending(f"lane still has pending work: {key}")


def _maximum_qubits(result: Mapping[str, Any]) -> int | None:
    n_qubits = result.get("n_qubits")
    if isinstance(n_qubits, int):
        return n_qubits
    values = [
        row.get("n_qubits")
        for row in (result.get("species") or {}).values()
        if isinstance(row, Mapping) and isinstance(row.get("n_qubits"), int)
    ]
    return max(values, default=None)


def _timing_value(result: Mapping[str, Any]) -> float | None:
    value = result.get("task_wall_seconds", result.get("evaluation_wall_seconds"))
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _timing_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int | None], list[tuple[float, float]]] = {}
    for row in pairs:
        cpu_seconds = _timing_value(row["cpu"])
        gpu_seconds = _timing_value(row["gpu"])
        if cpu_seconds is None or gpu_seconds is None:
            continue
        key = (str(row["stage"]), _maximum_qubits(row["cpu"]))
        groups.setdefault(key, []).append((cpu_seconds, gpu_seconds))
    summaries = []
    for (stage, n_qubits), values in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or -1)
    ):
        cpu = [row[0] for row in values]
        gpu = [row[1] for row in values]
        ratios = [cpu_seconds / gpu_seconds for cpu_seconds, gpu_seconds in values]
        summaries.append(
            {
                "stage": stage,
                "n_qubits": n_qubits,
                "matched_evaluations": len(values),
                "cpu_median_seconds": _median(cpu),
                "gpu_median_seconds": _median(gpu),
                "median_cpu_over_gpu_speedup": _median(ratios),
            }
        )
    return {
        "matched_timed_evaluations": sum(int(row["matched_evaluations"]) for row in summaries),
        "by_stage_and_qubits": summaries,
    }


def _compare_exact(
    *,
    key: str,
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
    energy_atol_hartree: float,
    mismatches: list[dict[str, Any]],
) -> None:
    for field in EXACT_METADATA_FIELDS:
        if cpu.get(field) != gpu.get(field):
            mismatches.append(
                {
                    "record": key,
                    "field": field,
                    "cpu": cpu.get(field),
                    "gpu": gpu.get(field),
                }
            )
    _compare_energies(
        key=key,
        cpu=cpu,
        gpu=gpu,
        energy_atol_hartree=energy_atol_hartree,
        mismatches=mismatches,
    )


def _compare_energies(
    *,
    key: str,
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
    energy_atol_hartree: float,
    mismatches: list[dict[str, Any]],
) -> None:
    for field in ENERGY_FIELDS:
        cpu_value = cpu.get(field)
        gpu_value = gpu.get(field)
        if not isinstance(cpu_value, (int, float)) and not isinstance(gpu_value, (int, float)):
            continue
        if not isinstance(cpu_value, (int, float)) or not isinstance(gpu_value, (int, float)):
            mismatches.append(
                {
                    "record": key,
                    "field": field,
                    "cpu": cpu_value,
                    "gpu": gpu_value,
                }
            )
            continue
        difference = abs(float(cpu_value) - float(gpu_value))
        if difference > energy_atol_hartree:
            mismatches.append(
                {
                    "record": key,
                    "field": field,
                    "cpu": cpu_value,
                    "gpu": gpu_value,
                    "absolute_difference_hartree": difference,
                }
            )


def _compare_compiled(
    *,
    key: str,
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
    energy_atol_hartree: float,
    mismatches: list[dict[str, Any]],
) -> None:
    for field in COMPILED_FIELDS:
        if cpu.get(field) != gpu.get(field):
            mismatches.append(
                {
                    "record": key,
                    "field": field,
                    "cpu": cpu.get(field),
                    "gpu": gpu.get(field),
                }
            )
    _compare_energies(
        key=key,
        cpu=cpu,
        gpu=gpu,
        energy_atol_hartree=energy_atol_hartree,
        mismatches=mismatches,
    )


def audit_campaigns(
    cpu_root: Path,
    gpu_root: Path,
    *,
    energy_atol_hartree: float = 1.0e-10,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Return a strict cross-backend parity audit for two campaign roots."""

    if energy_atol_hartree < 0:
        raise ValueError("energy_atol_hartree must be nonnegative")
    cpu_root = cpu_root.expanduser().resolve()
    gpu_root = gpu_root.expanduser().resolve()
    cpu_campaign, cpu_lanes = _lane_map(cpu_root)
    gpu_campaign, gpu_lanes = _lane_map(gpu_root)
    if require_complete:
        _require_complete(cpu_root, cpu_lanes)
        _require_complete(gpu_root, gpu_lanes)

    mismatches: list[dict[str, Any]] = []
    if set(cpu_lanes) != set(gpu_lanes):
        mismatches.append(
            {
                "record": "campaign",
                "field": "lane_grid",
                "cpu_only": sorted(set(cpu_lanes) - set(gpu_lanes)),
                "gpu_only": sorted(set(gpu_lanes) - set(cpu_lanes)),
            }
        )
    for field in ("schema", "generator_sha256", "iterations", "hamlib", "reactions"):
        if cpu_campaign.get(field) != gpu_campaign.get(field):
            mismatches.append(
                {
                    "record": "campaign",
                    "field": field,
                    "cpu": cpu_campaign.get(field),
                    "gpu": gpu_campaign.get(field),
                }
            )

    timing_pairs: list[dict[str, Any]] = []
    timed_reference_keys: set[tuple[str, str]] = set()
    records_checked = 0
    iteration_records_checked = 0
    for lane_key in sorted(set(cpu_lanes) & set(gpu_lanes)):
        cpu_lane = cpu_lanes[lane_key]
        gpu_lane = gpu_lanes[lane_key]
        cpu_manifest = _read_json(cpu_lane / "manifest.json")
        gpu_manifest = _read_json(gpu_lane / "manifest.json")
        for field in (
            "kind",
            "source_sha256",
            "initial_policy_sha256",
            "iterations_target",
            "family",
            "replicate",
            "instance",
        ):
            if cpu_manifest.get(field) != gpu_manifest.get(field):
                mismatches.append(
                    {
                        "record": f"{lane_key}/manifest",
                        "field": field,
                        "cpu": cpu_manifest.get(field),
                        "gpu": gpu_manifest.get(field),
                    }
                )
        cpu_state = _read_json(cpu_lane / "state.json")
        gpu_state = _read_json(gpu_lane / "state.json")
        for field in ("iterations_completed", "accepted_mutations", "incumbent_score"):
            if cpu_state.get(field) != gpu_state.get(field):
                mismatches.append(
                    {
                        "record": f"{lane_key}/state",
                        "field": field,
                        "cpu": cpu_state.get(field),
                        "gpu": gpu_state.get(field),
                    }
                )
        for policy_name in ("initial_pf.py",):
            cpu_hash = _sha256(cpu_lane / policy_name)
            gpu_hash = _sha256(gpu_lane / policy_name)
            if cpu_hash != gpu_hash:
                mismatches.append(
                    {
                        "record": f"{lane_key}/{policy_name}",
                        "field": "sha256",
                        "cpu": cpu_hash,
                        "gpu": gpu_hash,
                    }
                )
        cpu_incumbent_hash = _sha256(cpu_lane / "incumbent.py")
        gpu_incumbent_hash = _sha256(gpu_lane / "incumbent.py")
        if cpu_incumbent_hash != gpu_incumbent_hash:
            mismatches.append(
                {
                    "record": f"{lane_key}/incumbent.py",
                    "field": "sha256",
                    "cpu": cpu_incumbent_hash,
                    "gpu": gpu_incumbent_hash,
                }
            )

        cpu_reference = _read_json(cpu_lane / "exact_reference.json")
        gpu_reference = _read_json(gpu_lane / "exact_reference.json")
        _compare_exact(
            key=f"{lane_key}/exact_reference",
            cpu=cpu_reference,
            gpu=gpu_reference,
            energy_atol_hartree=energy_atol_hartree,
            mismatches=mismatches,
        )
        reference_key = (
            str(cpu_manifest.get("family")),
            str(cpu_manifest.get("instance")),
        )
        if reference_key not in timed_reference_keys:
            timing_pairs.append(
                {
                    "stage": "exact_reference",
                    "cpu": cpu_reference,
                    "gpu": gpu_reference,
                }
            )
            timed_reference_keys.add(reference_key)
        records_checked += 1

        cpu_baseline = _read_json(cpu_lane / "baseline_result.json")
        gpu_baseline = _read_json(gpu_lane / "baseline_result.json")
        _compare_compiled(
            key=f"{lane_key}/baseline",
            cpu=cpu_baseline,
            gpu=gpu_baseline,
            energy_atol_hartree=energy_atol_hartree,
            mismatches=mismatches,
        )
        timing_pairs.append({"stage": "baseline", "cpu": cpu_baseline, "gpu": gpu_baseline})
        records_checked += 1

        iterations = int(cpu_manifest["iterations_target"])
        for iteration in range(1, iterations + 1):
            relative = Path("iterations") / f"{iteration:04d}"
            cpu_iteration = cpu_lane / relative
            gpu_iteration = gpu_lane / relative
            cpu_candidate_hash = _sha256(cpu_iteration / "candidate.py")
            gpu_candidate_hash = _sha256(gpu_iteration / "candidate.py")
            if cpu_candidate_hash != gpu_candidate_hash:
                mismatches.append(
                    {
                        "record": f"{lane_key}/{relative}/candidate.py",
                        "field": "sha256",
                        "cpu": cpu_candidate_hash,
                        "gpu": gpu_candidate_hash,
                    }
                )
            cpu_result = _read_json(cpu_iteration / "result.json")
            gpu_result = _read_json(gpu_iteration / "result.json")
            _compare_compiled(
                key=f"{lane_key}/{relative}/result",
                cpu=cpu_result,
                gpu=gpu_result,
                energy_atol_hartree=energy_atol_hartree,
                mismatches=mismatches,
            )
            cpu_decision = _read_json(cpu_iteration / "decision.json")
            gpu_decision = _read_json(gpu_iteration / "decision.json")
            for field in DECISION_FIELDS:
                if cpu_decision.get(field) != gpu_decision.get(field):
                    mismatches.append(
                        {
                            "record": f"{lane_key}/{relative}/decision",
                            "field": field,
                            "cpu": cpu_decision.get(field),
                            "gpu": gpu_decision.get(field),
                        }
                    )
            validation_paths = {
                "cpu": cpu_iteration / "replay_validation.json",
                "gpu": gpu_iteration / "replay_validation.json",
            }
            validations = {
                side: _read_json(path) for side, path in validation_paths.items() if path.is_file()
            }
            valid_replay = any(
                validation.get("status") == "ok" for validation in validations.values()
            )
            if not valid_replay:
                mismatches.append(
                    {
                        "record": f"{lane_key}/{relative}/replay_validation",
                        "field": "status",
                        "cpu": validations.get("cpu", {}).get("status"),
                        "gpu": validations.get("gpu", {}).get("status"),
                    }
                )
            timing_pairs.append({"stage": "candidate", "cpu": cpu_result, "gpu": gpu_result})
            records_checked += 1
            iteration_records_checked += 1

    return {
        "schema": 1,
        "status": "ok" if not mismatches else "mismatch",
        "audited_at": _now(),
        "cpu_campaign_root": str(cpu_root),
        "gpu_campaign_root": str(gpu_root),
        "energy_atol_hartree": energy_atol_hartree,
        "lane_count": len(set(cpu_lanes) & set(gpu_lanes)),
        "records_checked": records_checked,
        "iteration_records_checked": iteration_records_checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "timings": _timing_summary(timing_pairs),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-campaign-root", type=Path, required=True)
    parser.add_argument("--gpu-campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-atol-hartree", type=float, default=1.0e-10)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")
    while True:
        try:
            result = audit_campaigns(
                args.cpu_campaign_root,
                args.gpu_campaign_root,
                energy_atol_hartree=args.energy_atol_hartree,
            )
        except ParityAuditPending as exc:
            if not args.wait:
                print(json.dumps({"status": "pending", "reason": str(exc)}, sort_keys=True))
                return 2
            time.sleep(args.interval_seconds)
            continue
        _write_json_atomic(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
