"""Figure 4 uses the scored production measurements and one complete winner."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLOTS = Path(__file__).parents[2] / "plots"
sys.path.insert(0, str(PLOTS))
try:
    spec = importlib.util.spec_from_file_location(
        "plot_elote_delta_reactions", PLOTS / "plot_elote_delta_reactions.py"
    )
    figure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figure)
finally:
    sys.path.pop(0)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def result(cost, score=1.0):
    return {
        "status": "ok",
        "passes_chemical_accuracy": True,
        "failed_case_ids": [],
        "unresolved_case_ids": [],
        "score": score,
        "case_metrics": [
            {"case_id": case, "t95": cost, "passes_chemical_accuracy": True}
            for case in figure.REACTION_ORDER
        ],
    }


def run(tmp_path):
    baseline = tmp_path / "baseline"
    transfer = tmp_path / "transfer"
    delta = tmp_path / "delta"
    suite = {
        "kind": "difference",
        "energy_target": "ground_branch",
        "difference_error_contract": "error_cancellation",
        "cases": [{"id": case} for case in figure.REACTION_ORDER],
    }
    for path in (baseline, transfer):
        save(path / "suite.json", suite)
        save(path / "iterations/0000/result.json", result(1000))
    save(transfer / "iterations/0001/result.json", result(200, .2))
    policy = b"class ProductFormula: pass\n"
    (transfer / "iterations/0001/candidate.py").write_bytes(policy)
    policy_sha = hashlib.sha256(policy).hexdigest()
    for index in range(1, 6):
        seed = delta / f"seed-{index:02d}"
        save(seed / "suite.json", suite)
        save(seed / "manifest.json", {
            "iterations_target": 50,
            "initial_policy_kind": "warm_start",
            "initial_policy_sha256": policy_sha,
        })
        save(seed / "state.json", {"iterations_completed": 50})
        save(seed / "iterations/0000/result.json", result(201))
        save(seed / "incumbent_result.json", result(190-index, .19-index*.01))
        (seed / "incumbent.py").write_bytes(f"# seed {index}\n".encode())
    return baseline, transfer, delta


def test_figure_uses_a100_elote_and_one_suite_winner(tmp_path):
    baseline, transfer, delta = run(tmp_path)
    summary = figure.load(baseline, transfer, delta)
    assert set(summary["baseline_t95"].values()) == {1000}
    assert set(summary["elote_t95"].values()) == {201}
    assert set(summary["elote_delta_t95"].values()) == {185}
    assert summary["elote_delta_seed"] == "seed-05"


def test_figure_rejects_cross_device_or_cross_seed_mismatch(tmp_path):
    baseline, transfer, delta = run(tmp_path)
    save(delta / "seed-02/iterations/0000/result.json", result(203))
    with pytest.raises(ValueError, match="disagree"):
        figure.load(baseline, transfer, delta)
    save(delta / "seed-02/iterations/0000/result.json", result(201))
    for index in range(1, 6):
        save(delta / f"seed-{index:02d}/iterations/0000/result.json", result(250))
    with pytest.raises(ValueError, match="over 1%"):
        figure.load(baseline, transfer, delta)
