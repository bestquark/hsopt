"""Suite membership, mixed precision, portable handoff and shared acceptance."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hsopt.campaign import protocol as p
from hsopt.evaluation import compiled_unitary as judge
from hsopt.sandbox.openshell import agent_feedback, mutation_prompt, sandbox_context_files
from hsopt.sandbox.research_context import benchmark_summary

FIXTURES = Path(__file__).parent / "fixtures"


def result(cost=100, error=1e-5, *, contract=None):
    return {
        "status": "ok",
        "evaluation_contract_version": judge.EVALUATION_CONTRACT_VERSION,
        "confidence_adjusted_t_count": cost,
        "absolute_energy_error_hartree": error,
        "passes_chemical_accuracy": error < judge.CHEMICAL_ACCURACY_HARTREE,
        "difference_error_contract": contract,
        "execution_backend": "cupy-lanczos",
    }


def receipt(request, cost=100, error=1e-5, *, contract=None):
    return {
        key: request[key] for key in ["schema", "task_id", "request_sha256", "evaluator_sha256"]
    } | {
        "execution": {"host": "test-fixture"},
        "result": result(cost, error, contract=contract) | {"task_id": request["task_id"]},
    }


@pytest.fixture
def prepared(tmp_path):
    # Small real cache/calibration fixtures; no GPUs, inference or campaign jobs.
    sources = [
        FIXTURES / "diff_energy_h2_bond_stretch/hamiltonians/h2_r0_7.json",
        FIXTURES / "diff_energy_h2_bond_stretch/hamiltonians/h2_r0_8.json",
    ]
    cases = []
    for index, source in enumerate(sources):
        reference = tmp_path / f"reference-{index}.json"
        p.write(reference, judge.evaluate_exact_reference(source, backend="numpy"))
        calibration = tmp_path / f"calibration-{index}.json"
        p.write(
            calibration,
            p.calibration.calibrate_problem(source, backend="numpy", mode="empirical_exact"),
        )
        cases.append(
            {
                "id": f"h2_{index}",
                "source": str(source),
                "reference": str(reference),
                "calibration": str(calibration),
            }
        )
    selection = tmp_path / "selection.json"
    p.write(selection, {"cases": cases})
    run = tmp_path / "shared"
    p.prepare(
        selection, run, family="abs_protocol", baseline="tight", iterations=2, replicate="seed-01"
    )
    return run


def pending(run):
    return p._pending(run)[1]["requests"]


def deliver_all(run, costs=(100, 100), errors=(1e-5, 1e-5)):
    for index, row in enumerate(pending(run)):
        p.record_receipt(run, row["case_id"], receipt(row["request"], costs[index], errors[index]))


def test_one_shared_proposal_waits_for_every_case_and_accepts_only_feasible_improvement(
    prepared, monkeypatch
):
    run = prepared
    calls = []
    first = pending(run)[0]
    p.record_receipt(run, first["case_id"], receipt(first["request"]))
    assert p.advance(run, propose=False)["status"] == "awaiting_evaluation"
    deliver_all(run)
    assert p.advance(run, propose=False)["status"] == "ready_to_mutate"
    assert p.read(run / "incumbent_result.json")["score"] == 1

    def mutate(policy, feedback, folder, **kwargs):
        calls.append(kwargs)
        assert kwargs["kind"] == "protocol"
        folder.mkdir()
        (folder / "candidate.py").write_text(
            policy.read_text().replace("return self.r", f"return {len(calls) + 1} * self.r")
        )
        assert p.read(feedback)["case_count"] == 2
        return {"status": "mutated"}

    monkeypatch.setattr(p, "mutate", mutate)
    request = p.advance(run)
    assert request["case_count"] == 2 and len(calls) == 1
    # A large T95 gain cannot compensate for one accuracy failure.
    deliver_all(run, costs=(10, 10), errors=(1e-5, 0.01))
    p.advance(run, propose=False)
    assert p.read(run / "iterations/0001/decision.json")["accepted"] is False
    assert p.read(run / "incumbent_result.json")["score"] == 1
    p.advance(run)
    deliver_all(run, costs=(50, 80))
    assert p.advance(run)["status"] == "complete"
    assert len(calls) == 2
    assert p.read(run / "incumbent_result.json")["score"] == pytest.approx((0.5 * 0.8) ** 0.5)
    assert len(list((run / "iterations").glob("*/case_results.json"))) == 3
    assert not list((run / "queue/pending").iterdir())


def test_shared_mutation_context_contains_entire_suite_and_explicit_mixed_precision(prepared):
    files = sandbox_context_files(kind="protocol", source=prepared / "suite.json")
    names = {relative.as_posix() for _, relative in files}
    assert "suite.json" in names and "protocol_evaluate.py" in names
    assert {"inputs/h2_0/problem.json", "inputs/h2_1/problem.json"} <= names
    summary = benchmark_summary(prepared / "suite.json", "protocol")
    assert len(summary["cases"]) == 2
    assert all(case["parameters"] for case in summary["cases"])
    prompt = mutation_prompt(context={"shared_protocol": {"objective": p.OBJECTIVE}})
    assert "ALL suite cases" in prompt
    assert "mixed-precision synthesis schemes" in prompt
    assert "baseline_parameters placeholder" in prompt
    assert (
        agent_feedback(p.aggregate({"a": result(50)}, {"a": result(100)}))["case_metrics"][0][
            "relative_t95"
        ]
        == 0.5
    )


def test_receipts_are_idempotent_and_refuse_wrong_backend_case_or_changed_inputs(prepared):
    row = pending(prepared)[0]
    good = receipt(row["request"])
    wrong = copy.deepcopy(good)
    wrong["request_sha256"] = "wrong"
    with pytest.raises(ValueError, match="foreign"):
        p.record_receipt(prepared, row["case_id"], wrong)
    wrong = copy.deepcopy(good)
    wrong["result"]["execution_backend"] = "numpy"
    with pytest.raises(ValueError, match="Lanczos"):
        p.record_receipt(prepared, row["case_id"], wrong)
    p.record_receipt(prepared, row["case_id"], good)
    p.record_receipt(prepared, row["case_id"], good)
    wrong = copy.deepcopy(good)
    wrong["result"]["confidence_adjusted_t_count"] = 2
    with pytest.raises(ValueError, match="conflicting"):
        p.record_receipt(prepared, row["case_id"], wrong)
    source = prepared / "inputs/h2_0/problem.json"
    source.chmod(0o644)
    source.write_text("{}")
    with pytest.raises(ValueError, match="sealed input"):
        p.suite_at(prepared)


def test_worker_can_partition_cases_but_never_fall_back_to_dense(prepared, monkeypatch):
    calls = []

    def worker(request, **kwargs):
        calls.append(kwargs)
        return receipt(request)

    monkeypatch.setattr(p, "run_evaluation", worker)
    with pytest.raises(ValueError, match="no full-diagonalization"):
        p.evaluate_pending(prepared, backend="numpy")
    p.evaluate_pending(prepared, case_ids=["h2_0"])
    assert len(calls) == 1 and calls[0]["backend"] == "cupy-lanczos"
    assert p.advance(prepared, propose=False)["missing_case_ids"] == ["h2_1"]
    p.evaluate_pending(prepared, case_ids=["h2_0", "h2_1"])
    assert len(calls) == 2


def test_infeasible_iteration_zero_holds_the_whole_suite(prepared, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cannot mutate before iteration zero passes")

    monkeypatch.setattr(p, "mutate", forbidden)
    deliver_all(prepared, errors=(1e-5, 0.01))
    state = p.advance(prepared)
    assert state["status"] == "needs_attention"
    assert state["failed_case_ids"] == ["h2_1"]
    assert p.read(prepared / "iterations/0000/decision.json")["accepted"] is False
    assert p.read(prepared / "state.json")["iterations_completed"] == 0
    assert p.advance(prepared)["status"] == "needs_attention"


def test_whole_suite_is_required_and_zero_cost_is_not_a_math_crash():
    baseline = {"a": result(100), "b": result(100)}
    with pytest.raises(ValueError, match="incomplete"):
        p.aggregate({"a": result()}, baseline)
    candidate = p.aggregate({"a": result(0), "b": result(50)}, baseline)
    assert candidate["score"] == 0 and candidate["passes_chemical_accuracy"]
    broken = result()
    broken["status"] = "error"
    broken["confidence_adjusted_t_count"] = None
    failed = p.aggregate({"a": broken, "b": result()}, baseline)
    assert failed["status"] == "error" and not p.accept(failed, candidate)[0]


def test_bound_protocol_starts_with_same_pf_as_prepared_baseline(prepared, tmp_path):
    from hsopt.campaign.baseline import baseline_module

    suite = p.suite_at(prepared)
    case = suite["cases"][0]
    problem = judge.load_problem(prepared / case["source"])
    bound = tmp_path / "bound.py"
    p.bind_policy(prepared / "initial_pf.py", case["parameters"], bound)
    shared = judge.load_policy(bound).ProductFormula(problem)
    individual = baseline_module().product_formula(problem, case["parameters"])
    assert shared.beta == case["parameters"][problem["name"]]["reference_trotter_number_beta"]
    assert shared.repetitions() == individual.repetitions()
    dt = problem["target_time"] / shared.repetitions()
    rotations = shared.build_step(problem["terms"], dt)
    assert rotations == individual.build_step(problem["terms"], dt)
    assert [shared.synthesis_epsilon(t, a) for t, a in rotations] == [
        individual.synthesis_epsilon(t, a) for t, a in rotations
    ]


@pytest.mark.parametrize("contract", judge.DIFFERENCE_ERROR_CONTRACTS)
def test_reaction_protocol_keeps_contract_and_species_bindings_isolated(tmp_path, contract):
    source = FIXTURES / "diff_energy_h2_bond_stretch/reaction.json"
    calibration_root = tmp_path / "calibrations"
    calibration_root.mkdir()
    for index, item in enumerate(p.read(source)["terms"]):
        problem = source.parent / item["problem"]
        p.write(
            calibration_root / f"{index}.json",
            p.calibration.calibrate_problem(problem, backend="numpy", mode="empirical_exact"),
        )
    reference = tmp_path / "reference.json"
    p.write(reference, judge.evaluate_exact_difference_reference(source, backend="numpy"))
    selection = tmp_path / "selection.json"
    p.write(
        selection,
        {
            "cases": [
                {
                    "id": "stretch",
                    "source": str(source),
                    "reference": str(reference),
                    "calibration_root": str(calibration_root),
                }
            ]
        },
    )
    run = tmp_path / contract
    p.prepare(
        selection,
        run,
        family="diff_protocol",
        baseline="tight",
        iterations=1,
        replicate="r1",
        error_contract=contract,
    )
    suite = p.suite_at(run)
    assert suite["difference_error_contract"] == contract
    assert len(suite["cases"][0]["parameters"]) == 2
    files = {
        rel.as_posix()
        for _, rel in sandbox_context_files(kind="protocol", source=run / "suite.json")
    }
    assert len([name for name in files if "/problems/" in name]) == 2
    assert len(benchmark_summary(run / "suite.json", "protocol")["cases"][0]["hamiltonians"]) == 2
    row = pending(run)[0]
    assert row["request"]["task"]["difference_error_contract"] == contract
    wrong = receipt(
        row["request"], contract=next(c for c in judge.DIFFERENCE_ERROR_CONTRACTS if c != contract)
    )
    with pytest.raises(ValueError, match="reaction contract"):
        p.record_receipt(run, "stretch", wrong)
    good = receipt(row["request"], contract=contract)
    good["result"]["contract_error_hartree"] = good["result"].pop("absolute_energy_error_hartree")
    p.record_receipt(run, "stretch", good)
    assert p.advance(run, propose=False)["status"] == "ready_to_mutate"


def test_round_cannot_silently_drop_a_case_from_requests(prepared):
    folder, bundle = p._pending(prepared)
    bundle["requests"] = bundle["requests"][:1]
    p.write(folder / "requests.json", bundle)
    with pytest.raises(ValueError, match="request bundle changed"):
        p.advance(prepared, propose=False)
    state = p.read(prepared / "state.json")
    state["pending_request_sha256"] = p.sha(folder / "requests.json")
    p.write(prepared / "state.json", state)
    with pytest.raises(ValueError, match="every suite case"):
        p.advance(prepared, propose=False)


def test_interrupted_successful_mutation_does_not_call_another_agent(prepared, monkeypatch):
    deliver_all(prepared)
    p.advance(prepared, propose=False)
    folder = prepared / "iterations/0001"
    folder.mkdir()
    (folder / "candidate.py").write_text(
        (prepared / "incumbent.py").read_text().replace("return self.r", "return 2 * self.r")
    )
    p.write(folder / "mutation_result.json", {"status": "mutated"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed proposal must be reused")

    monkeypatch.setattr(p, "mutate", forbidden)
    assert p.advance(prepared)["status"] == "awaiting_evaluation"
    assert p.read(folder / "protocol_validation.json")["status"] == "passed"


def test_reserved_baseline_edits_fail_before_any_candidate_evaluation(prepared, monkeypatch):
    deliver_all(prepared)
    p.advance(prepared, propose=False)

    def mutate(policy, feedback, folder, **kwargs):
        folder.mkdir()
        (folder / "candidate.py").write_text(
            policy.read_text().replace(
                "baseline_parameters = {}", "baseline_parameters = {'changed': (1, 1e-3, 1)}"
            )
        )
        p.write(folder / "mutation_result.json", {"status": "mutated"})

    monkeypatch.setattr(p, "mutate", mutate)
    with pytest.raises(ValueError, match="preserve baseline_parameters"):
        p.advance(prepared)
    assert p.read(prepared / "state.json")["pending_round"] is None
    assert p.read(prepared / "iterations/0001/protocol_validation.json")["status"] == "failed"
    assert not list((prepared / "queue/pending").iterdir())


def test_rejected_case_feedback_is_in_next_mutation_history(prepared):
    from hsopt.sandbox.research_context import lane_history

    deliver_all(prepared)
    p.advance(prepared, propose=False)
    folder = prepared / "iterations/0001"
    folder.mkdir()
    shutil_source = (prepared / "incumbent.py").read_text()
    (folder / "candidate.before.py").write_text(shutil_source)
    (folder / "candidate.py").write_text(
        shutil_source.replace("return self.r", "return 2 * self.r")
    )
    summary = p.aggregate(
        {"h2_0": result(10), "h2_1": result(10, error=0.01)}, {"h2_0": result(), "h2_1": result()}
    )
    p.write(folder / "result.json", summary)
    p.write(folder / "decision.json", {"accepted": False})
    history = lane_history(prepared, prepared / "iterations/0002")
    last = history["recent_proposals"][-1]
    assert last["failed_case_ids"] == ["h2_1"]
    assert last["worst_cases"][0]["case_id"] == "h2_1"
    context = benchmark_summary(prepared / "suite.json", "protocol")
    assert set(next(iter(context["cases"][0]["parameters"].values()))) == {
        "n_steps",
        "uniform_epsilon",
        "reference_trotter_number_beta",
        "target_time",
    }
