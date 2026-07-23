"""Create checksummed final exports from completed exact-unitary campaigns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


FAMILIES = ("hamlib121", "reactions10")
SAFE_COMPONENT = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class ExactExportError(RuntimeError):
    """Raised when a campaign cannot prove that a final export is complete."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ExactExportError(f"missing or unsafe campaign record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_component(value: object, label: str) -> str:
    text = str(value)
    if not text or any(character not in SAFE_COMPONENT for character in text):
        raise ExactExportError(f"unsafe {label}: {text!r}")
    return text


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file() and not source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _audit_fields(iteration_dir: Path) -> dict[str, Any]:
    path = iteration_dir / "agent_audit.json"
    if not path.is_file():
        return {
            "agent_tool_calls": 0,
            "agent_web_searches": 0,
            "agent_online_tool_calls": 0,
            "agent_inference_steps": 0,
            "agent_total_tokens": 0,
            "openshell_denied_events": 0,
        }
    audit = _read_json(path)
    opencode = audit.get("opencode") or {}
    openshell = audit.get("openshell") or {}
    return {
        "agent_tool_calls": int(opencode.get("tool_calls", 0)),
        "agent_web_searches": int(opencode.get("web_searches", 0)),
        "agent_online_tool_calls": int(opencode.get("online_tool_calls", 0)),
        "agent_inference_steps": int(opencode.get("inference_steps", 0)),
        "agent_total_tokens": int((opencode.get("tokens") or {}).get("total", 0)),
        "openshell_denied_events": int(openshell.get("denied_events", 0)),
    }


def _policy_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    policy = result.get("policy") or {}
    return {
        "formula_order": policy.get("formula_order", result.get("formula_order", "")),
        "n_steps": policy.get("n_steps", result.get("n_steps", "")),
        "schedule_strategy": policy.get("schedule_strategy", "stationary_trotter"),
        "term_order_strategy": policy.get("term_order_strategy", ""),
        "qdrift_samples": policy.get("qdrift_samples", ""),
        "uniform_epsilon": policy.get("uniform_epsilon", ""),
        "zero_angle_threshold": policy.get("zero_angle_threshold", ""),
    }


def _mutation_summary(result: Mapping[str, Any]) -> str:
    fields = _policy_fields(result)
    return (
        f"PF{fields['formula_order']}; n_steps={fields['n_steps']}; "
        f"order={fields['term_order_strategy']}; epsilon={fields['uniform_epsilon']}"
    )


def _decision_reason(decision: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    recorded = decision.get("acceptance_reason")
    if recorded:
        return str(recorded)
    if bool(decision.get("accepted")):
        return "strict_confidence_adjusted_t_gate_improvement"
    if result.get("status") != "ok":
        return "evaluator_error"
    if not bool(result.get("passes_chemical_accuracy")):
        return "chemical_accuracy_failed"
    return "no_strict_confidence_adjusted_t_gate_improvement"


def _confidence_cost(result: Mapping[str, Any]) -> Any:
    for key in ("confidence_adjusted_t_count", "score", "t_count"):
        value = result.get(key)
        if value is not None:
            return value
    return ""


def _campaign_lanes(campaign_root: Path, family: str) -> list[tuple[Path, dict[str, Any]]]:
    campaign = _read_json(campaign_root / "campaign.json")
    lanes: list[tuple[Path, dict[str, Any]]] = []
    for row in campaign.get("lanes", []):
        if row.get("family") != family:
            continue
        lane_dir = (campaign_root / str(row["lane_dir"])).resolve()
        if not lane_dir.is_relative_to(campaign_root):
            raise ExactExportError(f"lane escapes campaign root: {row['lane_dir']}")
        lanes.append((lane_dir, row))
    if not lanes:
        raise ExactExportError(f"campaign has no {family} lanes")
    return lanes


def _validate_lane(
    lane_dir: Path, campaign_row: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(lane_dir / "manifest.json")
    state = _read_json(lane_dir / "state.json")
    target = int(manifest["iterations_target"])
    if int(state.get("iterations_completed", -1)) != target:
        raise ExactExportError(f"incomplete lane: {lane_dir}")
    if state.get("pending_kind") is not None or state.get("pending_task_id") is not None:
        raise ExactExportError(f"lane still has pending work: {lane_dir}")
    for required in (
        "initial_pf.py",
        "incumbent.py",
        "baseline_result.json",
        "exact_reference.json",
    ):
        if not (lane_dir / required).is_file():
            raise ExactExportError(f"completed lane lacks {required}: {lane_dir}")
    if manifest.get("family") != campaign_row.get("family"):
        raise ExactExportError(f"lane family disagrees with campaign: {lane_dir}")
    for iteration in range(1, target + 1):
        iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
        for required in ("candidate.py", "result.json", "decision.json", "mutation_result.json"):
            if not (iteration_dir / required).is_file():
                raise ExactExportError(f"lane iteration {iteration} lacks {required}: {lane_dir}")
    return manifest, state


def _copy_lane_records(
    lane_dir: Path,
    output_root: Path,
    *,
    instance: str,
    replicate: str,
    iterations: int,
) -> None:
    destination = output_root / "records" / instance / replicate
    for name in (
        "manifest.json",
        "state.json",
        "initial_pf.py",
        "incumbent.py",
        "baseline_result.json",
        "incumbent_result.json",
        "exact_reference.json",
    ):
        _copy_if_present(lane_dir / name, destination / name)
    for iteration in range(1, iterations + 1):
        source = lane_dir / "iterations" / f"{iteration:04d}"
        target = destination / "iterations" / f"{iteration:04d}"
        for name in (
            "candidate.py",
            "result.json",
            "decision.json",
            "mutation_result.json",
            "agent_audit.json",
            "agent_actions.jsonl",
        ):
            _copy_if_present(source / name, target / name)


def _hamlib_candidate_row(
    *,
    instance: str,
    replicate: str,
    iteration: int,
    iteration_dir: Path,
) -> dict[str, Any]:
    result = _read_json(iteration_dir / "result.json")
    decision = _read_json(iteration_dir / "decision.json")
    policy = _policy_fields(result)
    candidate = iteration_dir / "candidate.py"
    return {
        "run": replicate,
        "lane_id": instance,
        "iteration": iteration,
        "accepted": int(bool(decision.get("accepted"))),
        "decision_reason": _decision_reason(decision, result),
        "passes_accuracy": int(bool(result.get("passes_chemical_accuracy"))),
        **policy,
        "per_attempt_t_gates_qpe": result.get("t_count", ""),
        "qpe_peak_probability_from_hf": result.get("qpe_success_probability_from_hf", ""),
        "qpe_repetitions": result.get("qpe_repetitions", ""),
        "confidence_adjusted_t_gates_qpe": _confidence_cost(result),
        "total_t_gates_qpe": _confidence_cost(result),
        "ground_state_energy_hartree": result.get(
            "exact_unitary_ground_energy_hartree",
            result.get("exact_ground_energy_hartree", ""),
        ),
        "ground_state_energy_estimate_hartree": result.get("compiled_pf_energy_hartree", ""),
        "ground_state_energy_error_hartree": result.get("absolute_energy_error_hartree", ""),
        "mutation_summary": _mutation_summary(result),
        "task_id": result.get("task_id", decision.get("task_id", "")),
        "backend": result.get("backend", ""),
        "candidate_sha256": _sha256(candidate),
        **_audit_fields(iteration_dir),
    }


def _hamlib_summary_row(
    *,
    instance: str,
    replicate: str,
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    lane_dir: Path,
) -> dict[str, Any]:
    baseline = _read_json(lane_dir / "baseline_result.json")
    accepted_iterations: list[int] = []
    for iteration in range(1, int(manifest["iterations_target"]) + 1):
        decision = _read_json(lane_dir / "iterations" / f"{iteration:04d}" / "decision.json")
        if bool(decision.get("accepted")):
            accepted_iterations.append(iteration)
    best_iteration = accepted_iterations[-1] if accepted_iterations else 0
    best = (
        baseline
        if best_iteration == 0
        else _read_json(lane_dir / "iterations" / f"{best_iteration:04d}" / "result.json")
    )
    problem = _read_json(Path(str(manifest["source"])))
    baseline_policy = _policy_fields(baseline)
    best_policy = _policy_fields(best)
    baseline_t = int(_confidence_cost(baseline))
    best_t = int(_confidence_cost(best))
    return {
        "run": replicate,
        "lane_id": instance,
        "molecule": problem.get("name", instance),
        "nqubits": problem["n_qubits"],
        "electrons": problem["electrons"],
        "terms": len(problem["terms"]),
        "accepted": int(state.get("accepted_mutations", len(accepted_iterations))),
        "rejected": int(manifest["iterations_target"]) - len(accepted_iterations),
        **{f"baseline_{key}": value for key, value in baseline_policy.items()},
        "baseline_per_attempt_t_gates_qpe": baseline.get("t_count", ""),
        "baseline_qpe_peak_probability_from_hf": baseline.get(
            "qpe_success_probability_from_hf", ""
        ),
        "baseline_qpe_repetitions": baseline.get("qpe_repetitions", ""),
        "baseline_confidence_adjusted_t_gates_qpe": baseline_t,
        "baseline_total_t_gates_qpe": baseline_t,
        "baseline_ground_state_energy_hartree": baseline["exact_unitary_ground_energy_hartree"],
        "baseline_ground_state_energy_estimate_hartree": baseline["compiled_pf_energy_hartree"],
        "baseline_ground_state_energy_error_hartree": baseline["absolute_energy_error_hartree"],
        "best_iteration": best_iteration,
        **{f"best_{key}": value for key, value in best_policy.items()},
        "best_per_attempt_t_gates_qpe": best.get("t_count", ""),
        "best_qpe_peak_probability_from_hf": best.get("qpe_success_probability_from_hf", ""),
        "best_qpe_repetitions": best.get("qpe_repetitions", ""),
        "best_confidence_adjusted_t_gates_qpe": best_t,
        "best_total_t_gates_qpe": best_t,
        "best_ground_state_energy_hartree": best["exact_unitary_ground_energy_hartree"],
        "best_ground_state_energy_estimate_hartree": best["compiled_pf_energy_hartree"],
        "best_ground_state_energy_error_hartree": best["absolute_energy_error_hartree"],
        "reduction_percent": 100.0 * (baseline_t - best_t) / baseline_t,
    }


def _species_json(result: Mapping[str, Any]) -> str:
    species: dict[str, Any] = {}
    for name, item in sorted((result.get("species") or {}).items()):
        policy = _policy_fields(item)
        species[name] = {
            **policy,
            "per_attempt_t_gate_count": item.get("t_count", ""),
            "qpe_peak_probability_from_hf": item.get("qpe_success_probability_from_hf", ""),
            "qpe_repetitions": item.get("qpe_repetitions", ""),
            "confidence_adjusted_t_gate_count": _confidence_cost(item),
            "t_gate_count": _confidence_cost(item),
        }
    return json.dumps(species, sort_keys=True, separators=(",", ":"))


def _reaction_row(
    *,
    instance: str,
    contract: str,
    iteration: int,
    result: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    iteration_dir: Path | None,
) -> dict[str, Any]:
    accepted = iteration == 0 or bool((decision or {}).get("accepted"))
    species_errors = [
        float(item["absolute_energy_error_hartree"])
        for item in (result.get("species") or {}).values()
        if item.get("absolute_energy_error_hartree") is not None
    ]
    row = {
        "reaction_key": instance,
        "source_contract": contract,
        "source_lane": f"{instance}_{contract}",
        "iteration": iteration,
        "accepted": str(accepted).lower(),
        "decision_reason": (
            "initial_baseline" if iteration == 0 else _decision_reason(decision or {}, result)
        ),
        "mutation_summary": (
            "Immutable paper baseline"
            if iteration == 0
            else _mutation_summary(next(iter((result.get("species") or {}).values()), {}))
        ),
        "species_json": _species_json(result),
        "per_attempt_t_gate_count": result.get("t_count", ""),
        "confidence_adjusted_t_gate_count": _confidence_cost(result),
        "total_t_gate_count": _confidence_cost(result),
        "exact_reaction_energy_hartree": result.get("exact_difference_hartree", ""),
        "pf_reaction_energy_hartree": result.get("compiled_pf_difference_hartree", ""),
        "signed_reaction_error_hartree": result.get("signed_difference_error_hartree", ""),
        "absolute_reaction_error_hartree": result.get("absolute_difference_error_hartree", ""),
        "max_species_absolute_error_hartree": max(species_errors, default=math.nan),
        "passes_contract": str(bool(result.get("passes_chemical_accuracy"))).lower(),
        "task_id": result.get("task_id", (decision or {}).get("task_id", "")),
        "backend": result.get("backend", ""),
    }
    if iteration_dir is not None:
        row.update(_audit_fields(iteration_dir))
        row["candidate_sha256"] = _sha256(iteration_dir / "candidate.py")
    else:
        row.update(_audit_fields(Path("/nonexistent")))
        row["candidate_sha256"] = ""
    return row


def _checksums(output_root: Path) -> None:
    checksum = output_root / "SHA256SUMS"
    lines = [
        f"{_sha256(path)}  {path.relative_to(output_root).as_posix()}"
        for path in sorted(item for item in output_root.rglob("*") if item.is_file())
        if path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_family(campaign_root: Path, family: str, output_dir: Path) -> dict[str, Any]:
    """Export one completed family without modifying the source campaign."""

    if family not in FAMILIES:
        raise ValueError(f"unsupported family: {family}")
    campaign_root = campaign_root.expanduser().resolve()
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists():
        raise ExactExportError(f"refusing to overwrite export: {output_dir}")
    lanes = _campaign_lanes(campaign_root, family)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        candidates: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        instances: set[str] = set()
        replicates: set[str] = set()
        target_iterations: set[int] = set()
        reaction_names: dict[str, str] = {}
        reaction_final: dict[tuple[str, str], dict[str, Any]] = {}
        for lane_dir, campaign_row in sorted(lanes, key=lambda item: str(item[0])):
            manifest, state = _validate_lane(lane_dir, campaign_row)
            instance = _safe_component(manifest["instance"], "instance")
            replicate = _safe_component(manifest["replicate"], "replicate")
            iterations = int(manifest["iterations_target"])
            instances.add(instance)
            replicates.add(replicate)
            target_iterations.add(iterations)
            _copy_lane_records(
                lane_dir,
                temporary,
                instance=instance,
                replicate=replicate,
                iterations=iterations,
            )
            if family == "hamlib121":
                summaries.append(
                    _hamlib_summary_row(
                        instance=instance,
                        replicate=replicate,
                        manifest=manifest,
                        state=state,
                        lane_dir=lane_dir,
                    )
                )
                for iteration in range(1, iterations + 1):
                    iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
                    candidates.append(
                        _hamlib_candidate_row(
                            instance=instance,
                            replicate=replicate,
                            iteration=iteration,
                            iteration_dir=iteration_dir,
                        )
                    )
            else:
                reaction = _read_json(Path(str(manifest["source"])))
                reaction_names[instance] = str(reaction["name"])
                baseline = _read_json(lane_dir / "baseline_result.json")
                candidates.append(
                    _reaction_row(
                        instance=instance,
                        contract=replicate,
                        iteration=0,
                        result=baseline,
                        decision=None,
                        iteration_dir=None,
                    )
                )
                incumbent = baseline
                incumbent_iteration = 0
                for iteration in range(1, iterations + 1):
                    iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
                    result = _read_json(iteration_dir / "result.json")
                    decision = _read_json(iteration_dir / "decision.json")
                    candidates.append(
                        _reaction_row(
                            instance=instance,
                            contract=replicate,
                            iteration=iteration,
                            result=result,
                            decision=decision,
                            iteration_dir=iteration_dir,
                        )
                    )
                    if bool(decision.get("accepted")):
                        incumbent = result
                        incumbent_iteration = iteration
                reaction_final[(instance, replicate)] = {
                    "selected_source_contract": replicate,
                    "selected_source_iteration": incumbent_iteration,
                    "per_attempt_t_gate_count": incumbent["t_count"],
                    "confidence_adjusted_t_gate_count": _confidence_cost(incumbent),
                    "total_t_gate_count": _confidence_cost(incumbent),
                    "absolute_reaction_error_hartree": incumbent[
                        "absolute_difference_error_hartree"
                    ],
                    "max_species_absolute_error_hartree": max(
                        float(item["absolute_energy_error_hartree"])
                        for item in incumbent["species"].values()
                    ),
                }
        if len(target_iterations) != 1:
            raise ExactExportError("family lanes have inconsistent iteration targets")
        iterations = target_iterations.pop()
        expected_lanes = len(instances) * len(replicates)
        if len(lanes) != expected_lanes:
            raise ExactExportError(
                f"family lane grid is incomplete: found {len(lanes)}, expected {expected_lanes}"
            )

        if family == "hamlib121":
            _write_csv(
                temporary / "candidate_log.csv",
                candidates,
                (
                    "run",
                    "lane_id",
                    "iteration",
                    "accepted",
                    "decision_reason",
                    "passes_accuracy",
                    "formula_order",
                    "n_steps",
                    "schedule_strategy",
                    "term_order_strategy",
                    "qdrift_samples",
                    "uniform_epsilon",
                    "zero_angle_threshold",
                    "per_attempt_t_gates_qpe",
                    "qpe_peak_probability_from_hf",
                    "qpe_repetitions",
                    "confidence_adjusted_t_gates_qpe",
                    "total_t_gates_qpe",
                    "ground_state_energy_hartree",
                    "ground_state_energy_estimate_hartree",
                    "ground_state_energy_error_hartree",
                    "mutation_summary",
                    "task_id",
                    "backend",
                    "candidate_sha256",
                    "agent_tool_calls",
                    "agent_web_searches",
                    "agent_online_tool_calls",
                    "agent_inference_steps",
                    "agent_total_tokens",
                    "openshell_denied_events",
                ),
            )
            _write_csv(
                temporary / "lane_run_summary.csv",
                summaries,
                (
                    "run",
                    "lane_id",
                    "molecule",
                    "nqubits",
                    "electrons",
                    "terms",
                    "accepted",
                    "rejected",
                    "baseline_formula_order",
                    "baseline_n_steps",
                    "baseline_schedule_strategy",
                    "baseline_term_order_strategy",
                    "baseline_qdrift_samples",
                    "baseline_uniform_epsilon",
                    "baseline_zero_angle_threshold",
                    "baseline_per_attempt_t_gates_qpe",
                    "baseline_qpe_peak_probability_from_hf",
                    "baseline_qpe_repetitions",
                    "baseline_confidence_adjusted_t_gates_qpe",
                    "baseline_total_t_gates_qpe",
                    "baseline_ground_state_energy_hartree",
                    "baseline_ground_state_energy_estimate_hartree",
                    "baseline_ground_state_energy_error_hartree",
                    "best_iteration",
                    "best_formula_order",
                    "best_n_steps",
                    "best_schedule_strategy",
                    "best_term_order_strategy",
                    "best_qdrift_samples",
                    "best_uniform_epsilon",
                    "best_zero_angle_threshold",
                    "best_per_attempt_t_gates_qpe",
                    "best_qpe_peak_probability_from_hf",
                    "best_qpe_repetitions",
                    "best_confidence_adjusted_t_gates_qpe",
                    "best_total_t_gates_qpe",
                    "best_ground_state_energy_hartree",
                    "best_ground_state_energy_estimate_hartree",
                    "best_ground_state_energy_error_hartree",
                    "reduction_percent",
                ),
            )
        else:
            _write_csv(
                temporary / "candidate_log.csv",
                candidates,
                (
                    "reaction_key",
                    "source_contract",
                    "source_lane",
                    "iteration",
                    "accepted",
                    "decision_reason",
                    "mutation_summary",
                    "species_json",
                    "per_attempt_t_gate_count",
                    "confidence_adjusted_t_gate_count",
                    "total_t_gate_count",
                    "exact_reaction_energy_hartree",
                    "pf_reaction_energy_hartree",
                    "signed_reaction_error_hartree",
                    "absolute_reaction_error_hartree",
                    "max_species_absolute_error_hartree",
                    "passes_contract",
                    "task_id",
                    "backend",
                    "candidate_sha256",
                    "agent_tool_calls",
                    "agent_web_searches",
                    "agent_online_tool_calls",
                    "agent_inference_steps",
                    "agent_total_tokens",
                    "openshell_denied_events",
                ),
            )
            pooled_rows = []
            for instance in sorted(instances):
                pooled_rows.append(
                    {
                        "reaction_key": instance,
                        "reaction": reaction_names[instance],
                        **{
                            contract: reaction_final[(instance, contract)]
                            for contract in sorted(replicates)
                        },
                    }
                )
            _write_json(temporary / "reaction_summary.json", {"reactions": pooled_rows})
            _write_json(
                temporary / "summary.json",
                {
                    "status": "complete",
                    "candidate_record_count": len(candidates),
                    "reaction_count": len(instances),
                    "contract_count": len(replicates),
                    "iterations": iterations,
                },
            )
        manifest = {
            "schema": 1,
            "status": "complete",
            "family": family,
            "created_at": _now(),
            "campaign_root": str(campaign_root),
            "campaign_sha256": _sha256(campaign_root / "campaign.json"),
            "instance_count": len(instances),
            "lane_count": len(lanes),
            "replicates": sorted(replicates),
            "replicate_count": len(replicates),
            "iterations": iterations,
            "candidate_record_count": len(candidates),
            "raw_agent_logs_retained_at": str(campaign_root / "lanes"),
        }
        _write_json(temporary / "export_manifest.json", manifest)
        (temporary / "FINAL_EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
        _checksums(temporary)
        temporary.rename(output_dir)
        return {"status": "exported", "output_dir": str(output_dir), **manifest}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_family(args.campaign_root, args.family, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
