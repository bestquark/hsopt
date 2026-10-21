"""Strategy isolation, real policy export, budgets, numerical holds and tool wiring."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from hsopt.campaign import protocol as p
from hsopt.sandbox import openshell as sandbox
from hsopt.search import experiment as e
from hsopt.search.tolerance import (
    expression_source,
    feature_vector,
    tolerance_policy,
    training_features,
)

FIXTURES = Path(__file__).parent / "fixtures/diff_energy_h2_bond_stretch/hamiltonians"


@pytest.fixture
def comparison(tmp_path):
    rows = []
    for index, filename in enumerate(["h2_r0_7.json", "h2_r0_8.json"]):
        source = FIXTURES / filename
        ref, calib = tmp_path / f"ref{index}.json", tmp_path / f"cal{index}.json"
        p.write(ref, p.judge.evaluate_exact_reference(source, backend="numpy"))
        p.write(
            calib, p.calibration.calibrate_problem(source, backend="numpy", mode="empirical_exact")
        )
        rows.append(
            {
                "id": f"h2_{index}",
                "source": str(source),
                "reference": str(ref),
                "calibration": str(calib),
            }
        )
    selection, root = tmp_path / "selection.json", tmp_path / "comparison"
    p.write(selection, {"cases": rows})
    e.prepare_comparison(
        selection,
        root,
        seeds=[1],
        train_size=1,
        test_size=1,
        budget=2,
        baseline="tight",
        policy_profile="pauli",
        energy_target="hf_accessible",
    )
    return root


@pytest.fixture
def gpu_worker(monkeypatch):
    settings = {"cost": 100, "error": 1e-5, "status": "ok", "calls": 0}

    def evaluate(request, **kwargs):
        settings["calls"] += 1
        assert kwargs["backend"] == "cupy-lanczos"
        return {
            key: request[key] for key in ["schema", "task_id", "request_sha256", "evaluator_sha256"]
        } | {
            "execution": {"host": "contract-test"},
            "result": {
                "task_id": request["task_id"],
                "status": settings["status"],
                "evaluation_contract_version": request["task"]["evaluation_contract_version"],
                "confidence_adjusted_t_count": settings["cost"],
                "absolute_energy_error_hartree": settings["error"],
                "passes_chemical_accuracy": settings["error"] < p.judge.CHEMICAL_ACCURACY_HARTREE,
                "execution_backend": "cupy-lanczos",
                "difference_error_contract": None,
                "error": "unresolved Ritz pair" if settings["status"] != "ok" else None,
            },
        }

    monkeypatch.setattr(p, "run_evaluation", evaluate)
    return settings


def test_methods_share_split_judge_and_start_but_keep_separate_queues(comparison):
    manifest = p.read(comparison / "comparison.json")
    runs = [comparison / name for name in manifest["runs"]]
    assert len(runs) == 3
    assert len({p.sha(run / "suite.json") for run in runs}) == 1
    assert len({p.sha(run / "initial_pf.py") for run in runs}) == 1
    for run in runs:
        p.suite_at(run)
        train = {row["id"] for row in p.read(run / "suite.json")["cases"]}
        split = p.read(comparison / "splits/seed-01/split.json")
        assert train == set(split["training_ids"])
        assert not train & set(split["test_ids"])
        assert not (run / "held_out").exists()
    assert all((run / "queue/pending").is_dir() for run in runs)


def test_budget_counts_every_probe_and_reuses_only_identical_source(comparison, gpu_worker):
    run = comparison / "agent_pysr/seed-01"
    oracle = e.TrainingOracle(run, worker="local")
    oracle.initialize()
    base = (run / "initial_pf.py").read_text()
    first_source = tolerance_policy(base, "1.0")
    gpu_worker["cost"] = 80
    first = oracle.evaluate(first_source, origin={"tool": "hsopt_pysr"})
    assert first["loss"] == pytest.approx(0.8)
    assert oracle.remaining == 1
    assert not first["cached"]
    assert oracle.evaluate(first_source, origin={"tool": "opencode"})["cached"]
    assert oracle.remaining == 1 and gpu_worker["calls"] == 2  # baseline + one proposal
    gpu_worker["cost"] = 90
    second = oracle.evaluate(tolerance_policy(base, "2.0"), origin={"tool": "opencode"})
    assert second["loss"] == pytest.approx(0.9)
    assert p.read(run / "incumbent_result.json")["score"] == pytest.approx(0.8)
    with pytest.raises(e.BudgetExhausted):
        oracle.evaluate(tolerance_policy(base, "3.0"), origin={"tool": "hsopt_pysr"})
    assert oracle.evaluate(first_source, origin={})["cached"]


def test_physical_failure_is_rejected_but_numerical_failure_holds_search(comparison, gpu_worker):
    run = comparison / "pysr_only/seed-01"
    oracle = e.TrainingOracle(run, worker="local")
    oracle.initialize()
    base = (run / "initial_pf.py").read_text()
    gpu_worker.update(cost=1, error=0.01)
    bad = oracle.evaluate(tolerance_policy(base, "4"), origin={})
    assert not bad["feasible"] and bad["loss"] > 1e6
    assert p.read(run / "incumbent_result.json")["score"] == 1
    gpu_worker.update(status="error", error=1e-5)
    with pytest.raises(e.NumericalEvaluationError, match="uncertified"):
        oracle.evaluate(tolerance_policy(base, "5"), origin={})
    with pytest.raises(e.NumericalEvaluationError, match="unresolved evaluator"):
        oracle.evaluate(tolerance_policy(base, "6"), origin={})
    with pytest.raises(e.NumericalEvaluationError, match="unresolved evaluator"):
        oracle.initialize()  # Resuming at an exhausted budget must not bypass the hold.


def test_symbolic_only_cannot_change_structure_and_agent_only_cannot_search(comparison, gpu_worker):
    oracle = e.TrainingOracle(comparison / "agent_only/seed-01", worker="local")
    with pytest.raises(ValueError, match="disabled"):
        e.SymbolicSearch(oracle).search({})
    run = comparison / "pysr_only/seed-01"
    oracle = e.TrainingOracle(run, worker="local")
    changed = (run / "initial_pf.py").read_text().replace("return self.r", "return 2 * self.r")
    with pytest.raises(ValueError, match="initial splitting"):
        e.SymbolicSearch(oracle).search({"base_policy": changed})


def test_exported_tolerance_matches_features_and_preserves_schedule(comparison, tmp_path):
    run = comparison / "agent_pysr/seed-01"
    case = p.suite_at(run)["cases"][0]
    base = run / "initial_pf.py"
    proposal = tolerance_policy(base.read_text(), "d - lc + 0.2 * w + 0.1 * a")
    unbound, bound, original = (
        tmp_path / "candidate.py",
        tmp_path / "bound.py",
        tmp_path / "original.py",
    )
    unbound.write_text(proposal)
    p.bind_policy(unbound, case["parameters"], bound)
    p.bind_policy(base, case["parameters"], original)
    problem = p.judge.load_problem(run / case["source"])
    initial_pf, initial_rotations, _ = p.judge._product_formula_schedule(
        problem, p.judge.load_policy(original)
    )
    pf, rotations, _ = p.judge._product_formula_schedule(problem, p.judge.load_policy(bound))
    assert rotations == initial_rotations and pf.repetitions() == initial_pf.repetitions()
    for term, angle in rotations:
        c, w, a, d, xy, lc = feature_vector(problem, term, angle, pf.repetitions())
        expected = pf.epsilon * math.exp(max(-8, min(16, d - lc + 0.2 * w + 0.1 * a)))
        assert pf.synthesis_epsilon(term, angle) == pytest.approx(expected)
    assert training_features(run, base, tmp_path / "features")
    twice = tolerance_policy(proposal, "0.0")
    assert twice.count("self.symbolic_target_time =") == 1


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os')",
        "c.__class__",
        "open('x')",
        "c ** 99",
        "float('nan')",
        "[c]",
        "sqrt(c, w)",
        "unknown + 1",
    ],
)
def test_symbolic_language_rejects_code_and_unknown_features(source):
    with pytest.raises(ValueError):
        expression_source(source)


def test_agent_only_removes_inherited_capability_and_hybrid_is_stage_gated(monkeypatch):
    monkeypatch.setenv("HSOPT_ENABLE_PYSR", "1")
    monkeypatch.setenv("HSOPT_PYSR_URL", "http://runtime.example:8787")
    monkeypatch.setenv("HSOPT_PYSR_TOKEN", "test-token")
    with e.agent_capability("agent_only"):
        assert sandbox._pysr_settings() is None
        assert not sandbox.opencode_config().get("plugins")
    assert sandbox._pysr_settings()
    with e.agent_capability("agent_pysr", "http://runtime.example:8787", "new-token"):
        normal = sandbox.opencode_config()
        preparation = sandbox.opencode_config(preparation=True)
        assert normal["plugins"][-1]["package"] == "/sandbox/hsopt-pysr"
        assert normal["plugins"][-1]["options"]["token"] == "new-token"
        perms = normal["agents"][sandbox.OPENCODE_AGENT]["permissions"]
        pre = preparation["agents"][sandbox.OPENCODE_AGENT]["permissions"]
        assert next(p for p in perms if p["action"] == "hsopt_pysr")["effect"] == "allow"
        assert next(p for p in pre if p["action"] == "hsopt_pysr")["effect"] == "deny"
    assert sandbox._pysr_settings()["token"] == "test-token"


def test_v2_plugin_registers_and_calls_budgeted_tool_over_http(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node is needed to exercise the JavaScript plugin")
    requests = []

    class Search:
        def search(self, request):
            requests.append(request)
            return {
                "status": "complete",
                "new_suite_evaluations": 3,
                "best": {"source": "returned policy", "loss": 0.8},
            }

    server = e.tool_server(Search(), token="scope-token")
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(endpoint + "/search", data=b"{}"))
        assert error.value.code == 403
        plugin = tmp_path / "plugin.mjs"
        plugin.write_text((Path(e.__file__).parent / "opencode-pysr.js").read_text())
        script = """
          const plugin = (await import(process.argv[1])).default;
          let tool;
          await plugin.setup({options: {endpoint: process.argv[2], token: "scope-token"},
            tool: {async transform(f) { f({add(def) { tool = def; }}); }}});
          if (tool.name !== "hsopt_pysr") throw new Error("tool did not register");
          const result = await tool.execute({base_policy: "proposed PF", max_evals: 3});
          console.log(result.content);
        """
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script, plugin.as_uri(), endpoint],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert json.loads(result.stdout)["best"]["loss"] == 0.8
        assert requests == [{"base_policy": "proposed PF", "max_evals": 3}]
    finally:
        server.shutdown()
        server.server_close()


def test_pysr_calls_get_their_own_telemetry_and_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("HSOPT_ENABLE_PYSR", "1")
    monkeypatch.setenv("HSOPT_PYSR_URL", "http://runtime.example:8787")
    monkeypatch.setenv("HSOPT_PYSR_TOKEN", "test-token")
    log = tmp_path / "events.jsonl"
    events = []
    for number, (tool, data) in enumerate(
        [
            ("read", {"path": "candidate.py"}),
            ("hsopt_pysr", {"base_policy": "private policy content"}),
            ("write", {"path": "candidate.py"}),
        ]
    ):
        events.append(
            {
                "type": "tool_use",
                "part": {
                    "id": str(number),
                    "tool": tool,
                    "state": {"status": "completed", "input": data},
                },
            }
        )
    log.write_text("\n".join(json.dumps(event) for event in events))
    metrics, actions = sandbox.summarize_opencode_events(log)
    assert metrics["by_category"]["symbolic_search"] == 1
    assert "private policy content" not in json.dumps(actions)
    sandbox._validate_agent_action_contract(log, required_reads=["candidate.py"])
    with pytest.raises(RuntimeError, match="required reads"):
        sandbox._validate_agent_action_contract(log, required_reads=["research_context.json"])
    with pytest.raises(RuntimeError, match="editing stage"):
        sandbox._validate_agent_action_contract(log, preparation=True)


def test_hybrid_can_adopt_its_newly_scored_incumbent_without_duplicate_rejection(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HSOPT_ENABLE_PYSR", "1")
    monkeypatch.setenv("HSOPT_PYSR_URL", "http://runtime.example:8787")
    monkeypatch.setenv("HSOPT_PYSR_TOKEN", "test-token")
    before, candidate, incumbent = [
        tmp_path / name for name in ["candidate.before.py", "candidate.py", "incumbent.py"]
    ]
    before.write_text("x = 1\n")
    candidate.write_text("x = 2\n")
    incumbent.write_text("x = 2\n")
    (tmp_path / "opencode.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "id": "search",
                    "tool": "hsopt_pysr",
                    "state": {"status": "completed", "input": {}},
                },
            }
        )
    )
    assert sandbox._adopts_symbolic_incumbent(candidate, incumbent, tmp_path)
    before.write_text("x = 2\n")
    assert not sandbox._adopts_symbolic_incumbent(candidate, incumbent, tmp_path)
    monkeypatch.setenv("HSOPT_ENABLE_PYSR", "0")
    assert not sandbox._adopts_symbolic_incumbent(candidate, incumbent, tmp_path)


def test_hybrid_retains_certified_tool_work_if_final_agent_edit_fails(
    comparison, gpu_worker, monkeypatch
):
    run = comparison / "agent_pysr/seed-01"
    oracle = e.TrainingOracle(run, worker="local")
    base = (run / "initial_pf.py").read_text()

    class Symbolic:
        max_evals = 1

    class Server:
        def shutdown(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(e, "tool_server", lambda *a, **k: Server())

    def mutate(policy, feedback, folder, **kwargs):
        folder.mkdir()
        gpu_worker["cost"] = 80 if oracle.remaining == 2 else 70
        oracle.evaluate(
            tolerance_policy(base, str(oracle.remaining)), origin={"tool": "hsopt_pysr"}
        )
        raise RuntimeError("Duplicate final edit after successful inner search")

    monkeypatch.setattr(e, "mutate", mutate)
    assert (
        e.run_search(oracle, Symbolic(), tool_url="http://runtime.example:8787", port=8787)[
            "status"
        ]
        == "complete"
    )
    assert p.read(run / "state.json")["iterations_completed"] == 2
    assert p.read(run / "incumbent_result.json")["score"] == pytest.approx(0.7)
    assert len(list((run / "agent_turns").glob("*/controller_progress.json"))) == 2


def test_complete_training_freezes_test_policy_without_agent_selection(comparison, gpu_worker):
    run = comparison / "pysr_only/seed-01"
    oracle = e.TrainingOracle(run, worker="local")
    oracle.initialize()
    base = (run / "initial_pf.py").read_text()

    class Symbolic:
        max_evals = 1

        def search(self, request):
            gpu_worker["cost"] = 80 if oracle.remaining == 2 else 70
            oracle.evaluate(
                tolerance_policy(base, str(oracle.remaining)), origin={"tool": "hsopt_pysr"}
            )
            if oracle.remaining == 0:
                gpu_worker["error"] = 0.01  # held-out baseline will deliberately hold

    with pytest.raises(e.NumericalEvaluationError):
        # A failed TEST baseline holds instead of silently substituting a result.
        e.run_search(oracle, Symbolic())


def test_held_out_failure_is_reported_without_selecting_another_policy(
    comparison, gpu_worker, monkeypatch
):
    run = comparison / "pysr_only/seed-01"
    oracle = e.TrainingOracle(run, worker="local")
    base = (run / "initial_pf.py").read_text()
    evaluate = p.evaluate_pending

    def worker(directory):
        if directory.name == "held_out":
            gpu_worker["error"] = (
                0.01 if p.read(directory / "state.json")["pending_round"] == 1 else 1e-5
            )
        return evaluate(directory)

    monkeypatch.setattr(p, "evaluate_pending", worker)

    class Symbolic:
        max_evals = 1

        def search(self, request):
            gpu_worker["cost"] = 80 if oracle.remaining == 2 else 70
            oracle.evaluate(
                tolerance_policy(base, str(oracle.remaining)), origin={"tool": "hsopt_pysr"}
            )

    assert e.run_search(oracle, Symbolic())["status"] == "complete"
    report = p.read(run / "held_out/generalization_report.json")
    assert not report["result"]["passes_chemical_accuracy"]
    assert report["frozen_candidate_sha256"] == p.sha(run / "incumbent.py")
    assert p.sha(run / "incumbent.py") == p.sha(run / "held_out/frozen_protocol.py")
    assert (
        e.comparison_report(comparison)["runs"][1]["held_out"]["passes_chemical_accuracy"] is False
    )


@pytest.mark.skipif(
    not os.environ.get("HSOPT_TEST_PYSR_PYTHON"), reason="optional real Julia integration"
)
def test_real_julia_worker_waits_for_delayed_fitness_reply(tmp_path):
    worker = Path(e.__file__).with_name("pysr_worker.py")
    config = {
        "features": [[1, 1, 1, 1, 0, 0], [0.1, 0.5, 0.1, 0, 0.5, -2.3]],
        "feature_names": ["c", "w", "a", "d", "xy", "lc"],
        "max_evals": 5,
        "seconds": 30,
        "seed": 9,
        "output_directory": str(tmp_path / "pysr"),
    }
    with (tmp_path / "worker.stderr").open("w") as errors:
        child = subprocess.Popen(
            [os.environ["HSOPT_TEST_PYSR_PYTHON"], "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            bufsize=1,
        )
        try:
            child.stdin.write(json.dumps(config) + "\n")
            child.stdin.flush()
            calls = 0
            for line in child.stdout:
                assert not line.startswith("HSOPT_ERROR "), line
                if line.startswith("HSOPT_FITNESS "):
                    calls += 1
                    if calls == 2:
                        break
                    # Previously libuv made this pipe nonblocking and Python
                    # misread the wait for a physical evaluation as EOF.
                    time.sleep(0.3)
                    child.stdin.write('{"loss":1.0}\n')
                    child.stdin.flush()
            assert calls == 2
        finally:
            child.terminate()
            child.wait(timeout=15)
            child.stdin.close()
            child.stdout.close()
