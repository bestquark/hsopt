from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsopt.evaluation.compiled_unitary import EVALUATION_CONTRACT_VERSION
from hsopt.campaign.parity import ParityAuditPending, audit_campaigns


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _reference(*, energy: float, seconds: float) -> dict[str, object]:
    return {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "observable": "exact_unitary_ground_eigenphase",
        "unitary_definition": "U_exact = exp(+i (H - c0 I) / lambda)",
        "target_state_definition": "anchored_ground",
        "target_anchor_kind": "hartree_fock",
        "target_anchor_basis_index": 12,
        "exact_ground_subspace_dimension": 1,
        "problem_sha256": "a" * 64,
        "exact_unitary_ground_energy_hartree": energy,
        "n_qubits": 4,
        "task_wall_seconds": seconds,
    }


def _compiled(*, energy: float, seconds: float) -> dict[str, object]:
    return {
        "status": "ok",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "observable": "compiled_unitary_lowest_hf_accessible_eigenphase",
        "policy": {"formula_order": 2, "n_steps": 8, "uniform_epsilon": 1.0e-8},
        "per_attempt_t_count": 80,
        "qpe_repetitions": 1,
        "confidence_adjusted_t_count": 80,
        "t_count": 80,
        "score": 80,
        "passes_chemical_accuracy": True,
        "synthesis_epsilon_histogram": {"1.000e-08": 8},
        "scheduled_rotation_count": 16,
        "skipped_rotation_count": 0,
        "one_step_t_count": 10,
        "dense_matrix_dimension": 16,
        "unique_compilations": [{"gate_string_sha256": "b" * 64, "t_count": 10}],
        "exact_unitary_ground_energy_hartree": -1.0,
        "compiled_pf_energy_hartree": energy,
        "absolute_energy_error_hartree": abs(energy + 1.0),
        "n_qubits": 4,
        "task_wall_seconds": seconds,
    }


def _campaign(root: Path, *, accelerator: str) -> Path:
    lane_relative = Path("lanes/abs_energy/fixture/run_01")
    lane = root / lane_relative
    iteration = lane / "iterations" / "0001"
    iteration.mkdir(parents=True)
    source = root / "problem.json"
    source.write_text("{}\n", encoding="utf-8")
    candidate = "def make_policy():\n    return {'n_steps': 8}\n"
    (lane / "initial_pf.py").write_text(candidate, encoding="utf-8")
    (lane / "incumbent.py").write_text(candidate, encoding="utf-8")
    (iteration / "candidate.py").write_text(candidate, encoding="utf-8")
    _write_json(
        root / "campaign.json",
        {
            "schema": 1,
            "generator_sha256": "c" * 64,
            "iterations": 1,
            "hamlib": {"instances": 1, "replicates": 1},
            "reactions": {"instances": 0, "replicates": 0},
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
    _write_json(
        lane / "manifest.json",
        {
            "kind": "absolute",
            "source_sha256": "d" * 64,
            "initial_policy_sha256": "e" * 64,
            "iterations_target": 1,
            "family": "abs_energy",
            "replicate": "run_01",
            "instance": "fixture",
        },
    )
    _write_json(
        lane / "state.json",
        {
            "iterations_completed": 1,
            "pending_kind": None,
            "pending_task_id": None,
        },
    )
    speed = 4.0 if accelerator == "cpu" else 1.0
    _write_json(lane / "exact_reference.json", _reference(energy=-1.0, seconds=speed))
    _write_json(
        lane / "baseline_result.json",
        _compiled(energy=-0.9995, seconds=speed),
    )
    _write_json(iteration / "result.json", _compiled(energy=-0.9997, seconds=speed))
    _write_json(
        iteration / "decision.json",
        {
            "accepted": True,
            "acceptance_reason": "strict_confidence_adjusted_t_gate_improvement",
            "candidate_score": 80,
            "previous_incumbent_score": 100,
            "passes_chemical_accuracy": True,
        },
    )
    if accelerator == "gpu":
        _write_json(iteration / "replay_validation.json", {"status": "ok"})
    _write_json(root / "campaign_complete.json", {"status": "complete"})
    return lane


def test_completed_cpu_and_gpu_campaigns_produce_strict_parity_audit(
    tmp_path: Path,
) -> None:
    cpu = tmp_path / "cpu"
    gpu = tmp_path / "gpu"
    _campaign(cpu, accelerator="cpu")
    _campaign(gpu, accelerator="gpu")

    result = audit_campaigns(cpu, gpu)

    assert result["status"] == "ok"
    assert result["lane_count"] == 1
    assert result["iteration_records_checked"] == 1
    assert result["mismatch_count"] == 0
    timings = result["timings"]["by_stage_and_qubits"]
    assert {row["stage"] for row in timings} == {
        "baseline",
        "candidate",
        "exact_reference",
    }
    assert all(row["median_cpu_over_gpu_speedup"] == 4.0 for row in timings)


def test_parity_audit_surfaces_candidate_hash_mismatch(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu"
    gpu = tmp_path / "gpu"
    _campaign(cpu, accelerator="cpu")
    gpu_lane = _campaign(gpu, accelerator="gpu")
    (gpu_lane / "iterations/0001/candidate.py").write_text(
        "# drifted candidate\n",
        encoding="utf-8",
    )

    result = audit_campaigns(cpu, gpu)

    assert result["status"] == "mismatch"
    assert any(
        row["field"] == "sha256" and row["record"].endswith("candidate.py")
        for row in result["mismatches"]
    )


def test_parity_audit_accepts_cpu_replay_of_canonical_gpu_campaign(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu"
    gpu = tmp_path / "gpu"
    cpu_lane = _campaign(cpu, accelerator="cpu")
    gpu_lane = _campaign(gpu, accelerator="gpu")
    (gpu_lane / "iterations/0001/replay_validation.json").unlink()
    _write_json(cpu_lane / "iterations/0001/replay_validation.json", {"status": "ok"})

    result = audit_campaigns(cpu, gpu)

    assert result["status"] == "ok"
    assert result["mismatch_count"] == 0


def test_parity_audit_waits_for_both_completion_markers(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu"
    gpu = tmp_path / "gpu"
    _campaign(cpu, accelerator="cpu")
    _campaign(gpu, accelerator="gpu")
    (gpu / "campaign_complete.json").unlink()

    with pytest.raises(ParityAuditPending, match="campaign is not complete"):
        audit_campaigns(cpu, gpu)
