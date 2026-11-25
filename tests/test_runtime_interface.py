from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hsopt.campaign import interface, loop
from hsopt.campaign.loop import initialize_lane
from hsopt.evaluation.compiled_unitary import (
    evaluate_exact_difference_reference,
    evaluate_exact_reference,
)

FIXTURE = Path(__file__).parent / "fixtures" / "abs_energy_h2"


@pytest.fixture
def lane(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            evaluate_exact_reference(
                FIXTURE / "hamiltonian.json",
                backend="numpy",
            )
        )
    )
    root = tmp_path / "lane"
    initialize_lane(
        root,
        queue_dir=tmp_path / "ledger",
        kind="absolute",
        source=FIXTURE / "hamiltonian.json",
        initial_policy=FIXTURE / "initial_pf.py",
        exact_reference_path=reference,
        iterations=1,
    )
    return root


@pytest.fixture
def agent(monkeypatch):
    calls = []

    def mutate(policy, feedback, output, **kwargs):
        calls.append(json.loads(feedback.read_text()))
        output.mkdir()
        candidate = output / "candidate.py"
        candidate.write_text(policy.read_text().replace("(15,", "(14,"))
        return {"candidate": str(candidate)}

    monkeypatch.setattr(loop, "mutate", mutate)
    return calls


def baseline(lane):
    request = interface.advance_lane(lane)["request"]
    return request, interface.run_evaluation(request, backend="numpy", accelerator="cpu")


def test_result_records_acceptance_then_wakes_exactly_one_next_mutation(lane, agent):
    request, receipt = baseline(lane)
    assert not agent
    step = interface.deliver_result(lane, receipt)
    assert step["recorded"]
    assert len(agent) == 1
    assert (lane / "baseline_result.json").is_file()
    assert step["request"]["task_id"] != request["task_id"]
    again = interface.deliver_result(lane, receipt)
    assert not again["recorded"]
    assert len(agent) == 1
    mutation = interface.run_evaluation(step["request"], backend="numpy")
    final = interface.deliver_result(lane, mutation)
    assert final["request"] is None
    assert final["state"]["status"] == "complete"
    decision = json.loads((lane / "iterations/0001/decision.json").read_text())
    assert decision["task_id"] == mutation["task_id"]
    assert len(agent) == 1


def test_request_and_evaluation_are_portable_across_staging_paths(lane, tmp_path):
    request = interface.advance_lane(lane)["request"]
    staged = tmp_path / "remote" / "task"
    shutil.copytree(request["task_dir"], staged)
    receipt = interface.run_evaluation(request, task_dir=staged, backend="numpy")
    assert receipt["request_sha256"] == request["request_sha256"]
    assert receipt["result"]["status"] == "ok"


@pytest.mark.parametrize("contract", ["individual_errors", "error_cancellation"])
def test_reaction_handoff_preserves_species_and_scoring_contract(tmp_path, contract):
    fixture = FIXTURE.parent / "diff_energy_h2_bond_stretch"
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(evaluate_exact_difference_reference(fixture / "reaction.json", backend="numpy"))
    )
    lane = tmp_path / "lane"
    initialize_lane(
        lane,
        queue_dir=tmp_path / "ledger",
        kind="difference",
        source=fixture / "reaction.json",
        initial_policy=fixture / "initial_pf.py",
        exact_reference_path=reference,
        difference_error_contract=contract,
        iterations=1,
    )
    request = interface.advance_lane(lane)["request"]
    assert any(name.startswith("problems/") for name in request["files"])
    staged = tmp_path / "worker-task"
    shutil.copytree(request["task_dir"], staged)
    receipt = interface.run_evaluation(request, task_dir=staged, backend="numpy")
    assert receipt["result"]["status"] == "ok"
    assert receipt["result"]["difference_error_contract"] == contract
    assert interface.deliver_result(lane, receipt, mutate=False)["recorded"]
    assert (lane / "baseline_result.json").is_file()


def test_changed_candidate_is_rejected_before_evaluation(lane, monkeypatch):
    request = interface.advance_lane(lane)["request"]
    (Path(request["task_dir"]) / "policy.py").write_text("changed")
    monkeypatch.setattr(interface, "evaluate_task", lambda *a, **k: pytest.fail("evaluated"))
    with pytest.raises(ValueError, match="differ"):
        interface.run_evaluation(request)


@pytest.mark.parametrize("field", ["request_sha256", "evaluator_sha256", "task_id"])
def test_foreign_receipt_cannot_update_lane(lane, field):
    _, receipt = baseline(lane)
    receipt[field] = "0" * (24 if field == "task_id" else 64)
    before = (lane / "state.json").read_bytes()
    with pytest.raises(ValueError):
        interface.deliver_result(lane, receipt, mutate=False)
    assert (lane / "state.json").read_bytes() == before
    assert not (lane / "baseline_result.json").exists()


def test_conflicting_callback_cannot_overwrite_recorded_result(lane):
    _, receipt = baseline(lane)
    interface.deliver_result(lane, receipt, mutate=False)
    before = (lane / "baseline_result.json").read_bytes()
    changed = copy.deepcopy(receipt)
    changed["result"]["score"] += 1
    with pytest.raises(ValueError, match="conflicting"):
        interface.deliver_result(lane, changed, mutate=False)
    assert (lane / "baseline_result.json").read_bytes() == before


def test_concurrent_callbacks_do_not_duplicate_mutations(lane, agent):
    _, receipt = baseline(lane)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: interface.deliver_result(lane, receipt), range(2)))
    assert sum(step["recorded"] for step in responses) == 1
    assert len(agent) == 1


def test_saved_result_survives_interrupted_agent_wakeup(lane, monkeypatch, agent):
    _, receipt = baseline(lane)
    original = loop.mutate

    def interrupted(*args, **kwargs):
        raise RuntimeError("agent unavailable")

    monkeypatch.setattr(loop, "mutate", interrupted)
    with pytest.raises(RuntimeError, match="agent unavailable"):
        interface.deliver_result(lane, receipt)
    assert (lane / "baseline_result.json").is_file()
    monkeypatch.setattr(loop, "mutate", original)
    step = interface.advance_lane(lane)
    assert step["request"] is not None
    assert len(agent) == 1


@pytest.mark.parametrize(
    "change", ["contract", "nonfinite", "evaluation_request_sha256", "evaluator_sha256"]
)
def test_invalid_score_receipt_is_rejected_before_storage(lane, change):
    _, receipt = baseline(lane)
    if change == "contract":
        receipt["result"]["evaluation_contract_version"] = "another-evaluator"
    elif change == "nonfinite":
        receipt["result"]["score"] = float("nan")
    else:
        receipt["result"][change] = "0" * 64
    with pytest.raises(ValueError):
        interface.deliver_result(lane, receipt, mutate=False)
    assert not (lane / "baseline_result.json").exists()


def test_completed_task_cannot_wake_an_unrelated_lane(lane, tmp_path):
    _, receipt = baseline(lane)
    interface.deliver_result(lane, receipt, mutate=False)
    policy = tmp_path / "other-policy.py"
    policy.write_text((FIXTURE / "initial_pf.py").read_text().replace("(15,", "(14,"))
    other = tmp_path / "other-lane"
    initialize_lane(
        other,
        queue_dir=tmp_path / "ledger",
        kind="absolute",
        source=FIXTURE / "hamiltonian.json",
        initial_policy=policy,
        exact_reference_path=tmp_path / "reference.json",
        iterations=1,
    )
    interface.advance_lane(other)
    before = (other / "state.json").read_bytes()
    with pytest.raises(ValueError, match="this lane"):
        interface.deliver_result(other, receipt)
    assert (other / "state.json").read_bytes() == before
