from __future__ import annotations

import json
from pathlib import Path

from hsopt.exact_export import export_family
from hsopt.tgate_opt.reporting import (
    _hamlib_extended,
    _hamlib_summary,
    _reaction_extended,
    _reaction_summary,
    _validate_hamlib,
    _validate_reactions,
    generate_reports,
    verify_strict_export,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _policy() -> dict[str, object]:
    return {
        "formula_order": 2,
        "n_steps": 8,
        "term_order_strategy": "lexicographic",
        "uniform_epsilon": 1.0e-8,
        "zero_angle_threshold": 0.0,
    }


def _absolute_result(t_count: int, estimate: float) -> dict[str, object]:
    return {
        "status": "ok",
        "backend": "cupynumeric",
        "task_id": f"task-{t_count}",
        "passes_chemical_accuracy": True,
        "score": t_count,
        "per_attempt_t_count": t_count,
        "qpe_success_probability_from_hf": 1.0,
        "qpe_target_confidence": 0.95,
        "qpe_achieved_confidence": 1.0,
        "qpe_repetitions": 1,
        "confidence_adjusted_t_count": t_count,
        "t_count": t_count,
        "policy": _policy(),
        "exact_unitary_ground_energy_hartree": -1.0,
        "compiled_pf_energy_hartree": estimate,
        "absolute_energy_error_hartree": abs(estimate + 1.0),
    }


def _reaction_result(t_count: int, estimate: float) -> dict[str, object]:
    species = _absolute_result(t_count, -1.0 + 0.5 * (estimate + 1.0))
    return {
        "status": "ok",
        "backend": "cupynumeric",
        "task_id": f"reaction-{t_count}",
        "passes_chemical_accuracy": True,
        "score": t_count,
        "per_attempt_t_count": t_count,
        "confidence_adjusted_t_count": t_count,
        "t_count": t_count,
        "exact_difference_hartree": -1.0,
        "compiled_pf_difference_hartree": estimate,
        "signed_difference_error_hartree": estimate + 1.0,
        "absolute_difference_error_hartree": abs(estimate + 1.0),
        "species": {"fixture_species": species},
    }


def _complete_lane(
    lane_dir: Path,
    *,
    family: str,
    instance: str,
    replicate: str,
    kind: str,
    source: Path,
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> None:
    lane_dir.mkdir(parents=True)
    manifest = {
        "schema": 1,
        "family": family,
        "instance": instance,
        "replicate": replicate,
        "kind": kind,
        "source": str(source),
        "iterations_target": 1,
    }
    state = {
        "schema": 1,
        "iterations_completed": 1,
        "pending_kind": None,
        "pending_task_id": None,
        "accepted_mutations": 1,
    }
    _write_json(lane_dir / "manifest.json", manifest)
    _write_json(lane_dir / "state.json", state)
    _write_json(lane_dir / "baseline_result.json", baseline)
    _write_json(lane_dir / "incumbent_result.json", candidate)
    _write_json(lane_dir / "exact_reference.json", {"status": "ok"})
    (lane_dir / "initial_pf.py").write_text("# initial\n", encoding="utf-8")
    (lane_dir / "incumbent.py").write_text("# incumbent\n", encoding="utf-8")
    iteration = lane_dir / "iterations" / "0001"
    iteration.mkdir(parents=True)
    (iteration / "candidate.py").write_text("# candidate\n", encoding="utf-8")
    _write_json(iteration / "result.json", candidate)
    _write_json(
        iteration / "decision.json",
        {"iteration": 1, "accepted": True, "task_id": candidate["task_id"]},
    )
    _write_json(iteration / "mutation_result.json", {"status": "mutated"})
    _write_json(
        iteration / "agent_audit.json",
        {
            "opencode": {
                "tool_calls": 2,
                "web_searches": 0,
                "online_tool_calls": 0,
                "inference_steps": 3,
                "tokens": {"total": 100},
            },
            "openshell": {"denied_events": 1},
        },
    )


def test_completed_hamlib_campaign_exports_dynamic_strict_report_tables(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    source = campaign / "problem.json"
    _write_json(
        source,
        {
            "name": "Fixture H2",
            "n_qubits": 4,
            "electrons": 2,
            "terms": [{"label": "ZIII", "coefficient": 1.0}],
        },
    )
    lane_relative = Path("lanes/abs_energy/fixture/run_01")
    _complete_lane(
        campaign / lane_relative,
        family="abs_energy",
        instance="fixture",
        replicate="run_01",
        kind="absolute",
        source=source,
        baseline=_absolute_result(100, -0.999),
        candidate=_absolute_result(80, -0.9995),
    )
    _write_json(
        campaign / "campaign.json",
        {
            "schema": 1,
            "lanes": [
                {
                    "family": "abs_energy",
                    "instance": "fixture",
                    "replicate": "run_01",
                    "lane_dir": lane_relative.as_posix(),
                }
            ],
        },
    )

    output = tmp_path / "hamlib-export"
    exported = export_family(campaign, "abs_energy", output)
    verify_strict_export(output)
    candidates, summaries = _validate_hamlib(output)

    assert exported["instance_count"] == 1
    assert len(candidates) == 1
    assert len(summaries) == 1
    assert candidates[0]["agent_tool_calls"] == "2"
    assert "T95 (U x M x R)" in _hamlib_summary("fixture", summaries)
    assert "T95 (U x M x R)" in _hamlib_extended("fixture", summaries, candidates)
    assert (output / "records/fixture/run_01/iterations/0001/agent_actions.jsonl").exists() is False
    reports = tmp_path / "hamlib-reports"
    generate_reports("abs_energy", output, reports, engine="reportlab")
    assert (reports / "hamiltonians/fixture/summary.pdf").read_bytes().startswith(b"%PDF-")
    assert (reports / "hamiltonians/fixture/extended_log.pdf").read_bytes().startswith(b"%PDF-")


def test_completed_reaction_campaign_exports_both_contracts(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    source = campaign / "reaction.json"
    _write_json(source, {"name": "Fixture reaction", "terms": []})
    rows = []
    for contract in ("equal_share", "reaction_aware"):
        lane_relative = Path("lanes/diff_energy/fixture") / contract
        _complete_lane(
            campaign / lane_relative,
            family="diff_energy",
            instance="fixture",
            replicate=contract,
            kind="difference",
            source=source,
            baseline=_reaction_result(100, -0.999),
            candidate=_reaction_result(80, -0.9995),
        )
        rows.append(
            {
                "family": "diff_energy",
                "instance": "fixture",
                "replicate": contract,
                "lane_dir": lane_relative.as_posix(),
            }
        )
    _write_json(campaign / "campaign.json", {"schema": 1, "lanes": rows})

    output = tmp_path / "reaction-export"
    export_family(campaign, "diff_energy", output)
    verify_strict_export(output)
    candidates, pooled = _validate_reactions(output)

    assert len(candidates) == 4
    assert set(pooled["fixture"]) >= {"equal_share", "reaction_aware", "reaction"}
    reaction_rows = [row for row in candidates if row["reaction_key"] == "fixture"]
    assert "T95 (sum U x M x R)" in _reaction_summary(
        "fixture", reaction_rows, pooled["fixture"]
    )
    assert "T95 (sum U x M x R)" in _reaction_extended(
        "fixture", reaction_rows, pooled["fixture"]
    )
    reports = tmp_path / "reaction-reports"
    generate_reports("diff_energy", output, reports, engine="reportlab")
    assert (reports / "reactions/fixture/summary.pdf").read_bytes().startswith(b"%PDF-")
    assert (reports / "reactions/fixture/extended_log.pdf").read_bytes().startswith(b"%PDF-")
