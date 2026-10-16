"""No overlap, deterministic seed splits, no test feedback or test-side search."""

import json
from pathlib import Path

import pytest

from hsopt.campaign import generalization as g, protocol as p
from hsopt.evaluation import compiled_unitary as ev


def selection(tmp_path, count=100):
    cases = []
    for index in range(count):
        source = tmp_path / f"source-{index}.json"
        source.write_text(json.dumps({"example": index}))
        reference = tmp_path / f"reference-{index}.json"
        reference.write_text("{}")
        cases.append({"id": f"mol-{index:03d}", "source": str(source), "reference": str(reference)})
    path = tmp_path / "selection.json"
    p.write(path, {"cases": cases})
    return path


def test_eighty_twenty_partition_is_disjoint_exhaustive_and_reproducible(tmp_path):
    source = selection(tmp_path)
    first = g.split(source, tmp_path / "one", seed=1)
    repeat = g.split(source, tmp_path / "repeat", seed=1)
    other = g.split(source, tmp_path / "other", seed=2)
    assert first == repeat
    assert len(first["training_ids"]) == 80 and len(first["test_ids"]) == 20
    assert not set(first["training_ids"]) & set(first["test_ids"])
    assert len(set(first["training_ids"]) | set(first["test_ids"])) == 100
    assert first["training_ids"] != other["training_ids"]
    assert g.verify_split(tmp_path / "one") == first
    assert g.split(source, tmp_path / "one", seed=1) == first
    with pytest.raises(ValueError, match="replace a sealed split"):
        g.split(source, tmp_path / "one", seed=3)


def test_changed_inputs_wrong_size_and_duplicate_membership_fail_closed(tmp_path):
    source = selection(tmp_path)
    with pytest.raises(ValueError, match="dataset size"):
        g.split(source, tmp_path / "bad", seed=1, train_size=79)
    g.split(source, tmp_path / "split", seed=1)
    (tmp_path / "source-0.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="dataset input changed"):
        g.verify_split(tmp_path / "split")
    data = p.read(source)
    data["cases"][1]["source"] = data["cases"][0]["source"]
    p.write(source, data)
    with pytest.raises(ValueError, match="duplicate Hamiltonian"):
        g.split(source, tmp_path / "duplicate", seed=1)


@pytest.fixture
def prepared(tmp_path):
    cases = []
    fixtures = Path(__file__).parent / "fixtures/diff_energy_h2_bond_stretch/hamiltonians"
    for index, name in enumerate(["h2_r0_7", "h2_r0_8"]):
        source = fixtures / (name + ".json")
        reference = tmp_path / f"ref-{index}.json"
        cal = tmp_path / f"cal-{index}.json"
        p.write(reference, ev.evaluate_exact_reference(source, backend="numpy"))
        p.write(
            cal, p.calibration.calibrate_problem(source, backend="numpy", mode="empirical_exact")
        )
        cases.append(
            {
                "id": name,
                "source": str(source),
                "reference": str(reference),
                "calibration": str(cal),
            }
        )
    source = tmp_path / "selection.json"
    p.write(source, {"cases": cases})
    directory = tmp_path / "partition"
    g.split(source, directory, seed=1, train_size=1, test_size=1)
    train = tmp_path / "training"
    g.prepare_training(
        directory,
        train,
        iterations=1,
        replicate="seed-01",
        energy_target="hf_accessible",
        policy_profile="pauli",
    )
    return directory, train, tmp_path / "heldout"


def deliver(run, cost=100, error=1e-5):
    for row in p._pending(run)[1]["requests"]:
        request = row["request"]
        receipt = {
            key: request[key] for key in ["schema", "task_id", "request_sha256", "evaluator_sha256"]
        }
        receipt["result"] = {
            "status": "ok",
            "task_id": request["task_id"],
            "evaluation_contract_version": ev.EVALUATION_CONTRACT_VERSION,
            "confidence_adjusted_t_count": cost,
            "absolute_energy_error_hartree": error,
            "passes_chemical_accuracy": error < ev.CHEMICAL_ACCURACY_HARTREE,
            "difference_error_contract": None,
            "execution_backend": "cupy-lanczos",
        }
        p.record_receipt(run, row["case_id"], receipt)


def test_only_training_inputs_reach_mutation_and_test_waits_for_completion(prepared):
    directory, train, test = prepared
    split = p.read(directory / "split.json")
    assert [c["id"] for c in p.suite_at(train)["cases"]] == split["training_ids"]
    assert not set(split["test_ids"]) & {x.name for x in (train / "inputs").iterdir()}
    with pytest.raises(ValueError, match="training budget must finish"):
        g.prepare_test(directory, train, test)
    assert not test.exists()


def finish_training(train):
    deliver(train)
    p.advance(train, propose=False)
    candidate = train / "fixed-winner.py"
    candidate.write_text(
        (train / "incumbent.py")
        .read_text()
        .replace("return self.epsilon", "return self.epsilon * 1.1")
    )
    with p._lane_lock(train):
        p._enqueue_round(train, 1, candidate)
    deliver(train, cost=70)
    p.advance(train, propose=False)


def test_final_test_reports_failures_without_choosing_a_different_policy(prepared, monkeypatch):
    directory, train, test = prepared
    finish_training(train)
    before = {
        name: p.sha(train / name)
        for name in ["incumbent.py", "incumbent_result.json", "state.json"]
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("A held-out test must never mutate")

    monkeypatch.setattr(p, "mutate", forbidden)
    g.prepare_test(directory, train, test)
    assert p.sha(test / "frozen_protocol.py") == before["incumbent.py"]
    deliver(test)
    g.advance_test(test)
    assert p.read(test / "state.json")["pending_round"] == 1
    deliver(test, cost=60, error=0.002)
    assert g.advance_test(test)["status"] == "complete"
    report = p.read(test / "generalization_report.json")
    assert not report["result"]["passes_chemical_accuracy"]
    assert report["result"]["score"] == pytest.approx(0.6)
    assert p.sha(test / "incumbent.py") != report["frozen_candidate_sha256"]
    assert {name: p.sha(train / name) for name in before} == before
    assert g.prepare_test(directory, train, test)["status"] == "already_prepared"
