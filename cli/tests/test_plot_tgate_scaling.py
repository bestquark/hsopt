"""The bounded-memory PF2 counter matches the trusted full-schedule counter."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


NUMERICS = Path(__file__).parents[2] / "cli" / "tests" / "helpers" / "figure_numerics"
sys.path.insert(0, str(NUMERICS))
try:
    spec = importlib.util.spec_from_file_location(
        "plot_tgate_scaling", NUMERICS / "compiled_cost.py"
    )
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
finally:
    sys.path.pop(0)


# The recovered, accepted precision-only ProductFormula, with comments omitted.
# Its archived source SHA256 is 38d86d98abf0c927571e460bd2ec6324c6d569759cb06b8b1510163683a81eb1.
# Keep tests portable: no dataset or private campaign path is needed to run them.
PRECISION_SOURCE = """class ProductFormula:
    baseline_parameters = {}
    def __init__(self, problem):
        self.r, self.epsilon, self.beta = self.baseline_parameters[problem["name"]]
        self.max_coeff = max(abs(t["coefficient"]) for t in problem["terms"])
    def repetitions(self):
        return self.r
    def build_step(self, terms, step_time):
        ordered = sorted(terms, key=lambda term: term["label"])
        half = [(term, 0.5 * term["coefficient"] * step_time) for term in ordered[:-1]]
        center = [(ordered[-1], ordered[-1]["coefficient"] * step_time)]
        return half + center + list(reversed(half))
    def synthesis_epsilon(self, term, angle):
        coeff = abs(term["coefficient"])
        support = term.get("support_size", 1)
        x, y, z = term.get("x_count", 0), term.get("y_count", 0), term.get("z_count", 0)
        n_axes = (1 if x > 0 else 0) + (1 if y > 0 else 0) + (1 if z > 0 else 0)
        if n_axes <= 1:
            factor = 1.0
        elif n_axes == 2:
            factor = 1.008
        else:
            factor = 1.015
        base = max(self.beta, 1e-10) * support / max(coeff, 1e-10)
        coeff_ratio = coeff / max(self.max_coeff, 1e-10)
        p = 0.3 + 0.1 * (1 - coeff_ratio)**0.8
        eps = self.epsilon * base**p * factor
        if any(c in "XY" for c in term.get("label", "")):
            eps *= 1.002
        else:
            eps *= 0.998
        if eps <= 0:
            eps = 1e-12
        if eps >= 1.0:
            eps = 1.0 - 1e-12
        return eps
"""


@pytest.fixture
def policies(tmp_path):
    precision = tmp_path / "precision.py"
    precision.write_text(PRECISION_SOURCE)
    uniform = Path(__file__).parents[2] / "experiments/abs_protocol/initial_pf.py"
    return uniform, precision


def synthetic_case(n_qubits):
    # Source order deliberately differs from lexicographic order. Include all
    # axis types, both coefficient signs, an identity shift and a dropped zero.
    text = "\n".join(
        [
            ".01 [X0 Y1 Z2] +",
            "-2.0 [Z0] +",
            ".000001 [Y1] +",
            "0.5 [X0] +",
            "1.0 [Z0 Z1] +",
            "3.0 [] +",
            "0.0 [X2]",
        ]
    )
    statistics = plotting.inspect_text(text, n_qubits, expected_terms=7)
    problem = {
        "name": "normalized-toy",
        "n_qubits": n_qubits,
        "identity_coefficient": statistics["identity_coefficient"],
        "target_time": 1 / statistics["one_norm_nonidentity_hartree"],
        "terms": list(plotting.iter_terms(text, n_qubits)),
    }
    allocation = {
        "n_steps": 7,
        "uniform_epsilon": 2e-7,
        "reference_trotter_number_beta": 9.0,
        "policy_profile": "pauli",
    }
    return text, problem, statistics, allocation


@pytest.mark.parametrize("n_qubits", [4, 16, 24, 120])
@pytest.mark.parametrize("gamma,offset", [(3.0, 0.0), (4.0, 11.0), (3.1, -2.0)])
def test_streamed_actual_precision_methods_match_trusted_schedule(
    policies, n_qubits, gamma, offset
):
    text, problem, statistics, allocation = synthetic_case(n_qubits)
    uniform, precision = policies
    plotting.streamable_pf2(uniform, precision)
    stream = plotting.streamed_analytic_pair(
        text,
        problem,
        statistics,
        allocation,
        uniform,
        precision,
        {"gamma": gamma, "offset": offset},
    )
    for label, policy in (("baseline", uniform), ("policy", precision)):
        trusted = plotting.count_policy(
            problem, policy, allocation, method="analytic_ross_selinger", gamma=gamma, offset=offset
        )
        assert stream[label + "_t_count"] == pytest.approx(
            trusted["estimated_evolution_t_count"], rel=1e-13
        )
        assert stream[label + "_repetitions"] == trusted["n_steps"] == 7
        assert stream[label + "_epsilon_min"] == trusted["tolerance"]["minimum"]
        assert stream[label + "_epsilon_max"] == trusted["tolerance"]["maximum"]
        assert stream["rotations_per_step"] == trusted["rotations_per_step"] == 9
    assert stream["count_method"] == "analytic_ross_selinger"
    assert stream["exact_compiled_count"] is False


def test_streamed_counts_follow_changed_repetitions_and_angle_tolerances(policies):
    text, problem, statistics, allocation = synthetic_case(6)
    uniform, candidate = policies
    source = candidate.read_text().replace("return self.r", "return self.r * 2")
    start = source.index("    def synthesis_epsilon")
    candidate.write_text(
        source[:start]
        + """    def synthesis_epsilon(self, term, angle):
        return self.epsilon * (1 + abs(angle))
"""
    )
    plotting.streamable_pf2(uniform, candidate)
    stream = plotting.streamed_analytic_pair(
        text, problem, statistics, allocation, uniform, candidate, {"gamma": 3, "offset": 0}
    )
    trusted = plotting.count_policy(problem, candidate, allocation, method="analytic_ross_selinger")
    assert stream["baseline_repetitions"] == 7
    assert stream["policy_repetitions"] == trusted["n_steps"] == 14
    assert stream["policy_t_count"] == pytest.approx(trusted["estimated_evolution_t_count"])


def test_streaming_rejects_custom_or_framed_schedule(policies):
    uniform, candidate = policies
    candidate.write_text(
        candidate.read_text().replace(
            "return half + center + list(reversed(half))", "return half + center"
        )
    )
    with pytest.raises(ValueError, match="canonical lexicographic PF2"):
        plotting.streamable_pf2(uniform, candidate)
    candidate.write_text(
        PRECISION_SOURCE
        + """    def basis_change(self):
        return [("H", 0)]
"""
    )
    with pytest.raises(ValueError, match="Clifford-frame"):
        plotting.streamable_pf2(uniform, candidate)


def test_streaming_rejects_stateful_tolerance_that_depends_on_rotation_order(policies):
    uniform, candidate = policies
    source = uniform.read_text().replace(
        'self.r, self.epsilon, self.beta = self.baseline_parameters[problem["name"]]',
        'self.r, self.epsilon, self.beta = self.baseline_parameters[problem["name"]]\n'
        "        self.counter = 0",
    )
    source = source.replace("return self.epsilon", "self.counter += 1\n        return self.epsilon")
    candidate.write_text(source)
    with pytest.raises(ValueError, match="stateful|stateless|full-schedule"):
        plotting.streamable_pf2(uniform, candidate)


def test_streamed_counter_rejects_invalid_precision(policies):
    text, problem, statistics, allocation = synthetic_case(4)
    uniform, candidate = policies
    candidate.write_text(candidate.read_text().replace("return eps", "return 0"))
    with pytest.raises(ValueError, match="invalid synthesis tolerance"):
        plotting.streamed_analytic_pair(
            text, problem, statistics, allocation, uniform, candidate, {"gamma": 3, "offset": 0}
        )


@pytest.mark.parametrize("different_policy", [False, True])
def test_census_keeps_new_policies_separate_from_archived_counts(
    tmp_path, policies, monkeypatch, different_policy
):
    text, problem, statistics, allocation = synthetic_case(4)
    uniform, candidate = policies
    if different_policy:
        candidate.write_text(candidate.read_text().replace("return self.r", "return self.r * 2"))
    training = []
    for i, epsilon in enumerate((1e-3, 1e-4, 1e-5)):
        training.append(
            {
                "collection": "chem",
                "file": "toy.hdf5",
                "dataset": str(i),
                "case_id": str(i),
                "problem_name": problem["name"],
                "n_qubits": "4",
                "nonidentity_terms": str(statistics["nonidentity_terms"]),
                "target_time": str(problem["target_time"]),
                "reference_trotter_number_beta": "9",
                "molecule": "toy" + str(i),
                "uniform_epsilon": str(epsilon),
                "baseline_one_step_t_count": str(9 * 3 * -math.log2(epsilon)),
                "baseline_compiled_evolution_t_count": "1000000",
                "learned_compiled_evolution_t_count": "900000",
                "baseline_passed": "True",
                "learned_passed": "True",
            }
        )
    files = {}
    for name in ("catalog", "training_table", "calibrations", "policy_provenance"):
        files[name] = tmp_path / (name + ".json")
        files[name].write_text("{}")
    files["calibrations"].write_text(json.dumps({"0": allocation}))
    files["policy_provenance"].write_text(json.dumps({"policy_sha256": plotting.sha(candidate)}))
    monkeypatch.setattr(plotting, "read_csv", lambda path: training)
    catalog = {
        **training[0],
        "nqubits": "4",
        "terms": "7",
        "effective_electrons_for_algorithm": "2",
    }
    monkeypatch.setattr(plotting, "resolve_archives", lambda *args: [catalog])
    monkeypatch.setattr(plotting, "load_text", lambda *args, **kwargs: (text, "archive-hash"))
    monkeypatch.setattr(
        plotting.ev, "_compile_rotations", lambda *args: pytest.fail("compiled proxy")
    )
    args = SimpleNamespace(
        **files,
        archive_root=tmp_path,
        archive_cache=None,
        baseline_policy=uniform,
        policy=candidate,
        output_dir=tmp_path / "output",
        training_max_qubits=14,
        max_qubits=24,
        profile="pauli",
        method="analytic_ross_selinger",
        full_schedule=True,
        allow_different_policy=different_policy,
        case_id=None,
        policy_label="Frozen test protocol",
        shard_index=0,
        shard_count=1,
    )
    plotting.count(args)
    manifest = json.loads((args.output_dir / "scaling_manifest.json").read_text())
    row = json.loads((args.output_dir / "scaling_counts.jsonl").read_text())
    assert manifest["archived_policy_comparison"] is (not different_policy)
    assert row["baseline_repetitions"] == 7
    assert row["policy_repetitions"] == (14 if different_policy else 7)
    assert ("archived_policy_t_count" in row) is (not different_policy)
    assert ("archived_baseline_t_count" in row) is (not different_policy)
    assert ("archived_energy_accuracy_passed" in row) is (not different_policy)
    assert row["policy_t_count"] < 900000
    # Output cannot silently resume under a new interpretation of its archive.
    args.allow_different_policy = not different_policy
    with pytest.raises(ValueError, match="another configuration"):
        plotting.count(args)


def test_final_plot_rejects_missing_catalog_ids_even_if_summary_says_complete(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(NUMERICS))
    import _paper_plot

    def must_not_render():
        pytest.fail("incomplete census reached plotting before coverage validation")

    monkeypatch.setattr(_paper_plot, "apply_paper_style", must_not_render)
    (tmp_path / "scaling_manifest.json").write_text(
        json.dumps(
            {
                "selected_case_ids": ["a", "b"],
                "training_max_qubits": 14,
                "policy_label": "Frozen protocol",
            }
        )
    )
    (tmp_path / "scaling_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "selected": 2,
                "completed": 2,
                "failed": 0,
            }
        )
    )
    (tmp_path / "scaling_counts.jsonl").write_text(
        json.dumps(
            {
                "case_id": "a",
                "status": "ok",
                "n_qubits": 4,
                "baseline_t_count": 100,
                "policy_t_count": 90,
                "exact_compiled_count": True,
                "allocation_extrapolated": False,
            }
        )
        + "\n"
    )
    args = SimpleNamespace(input_dir=tmp_path, output_dir=tmp_path / "figures")
    with pytest.raises(ValueError, match="catalog|coverage|census|selected|missing"):
        plotting.plot(args)


def test_synthesis_model_uses_actual_nested_rotation_counts():
    gamma, offset = 3.0, 4.0
    rows = []
    for i, rotations in enumerate((9, 15, 21, 27, 33)):
        epsilon = 10.0 ** (-3 - i)
        rows.append(
            {
                "uniform_epsilon": epsilon,
                "nonidentity_terms": 5,
                "scheduled_rotations_per_step": rotations,
                "baseline_one_step_t_count": rotations * (gamma * math.log2(1 / epsilon) + offset),
            }
        )
    model = plotting.synthesis_model(rows)
    assert model["gamma"] == pytest.approx(gamma)
    assert model["offset"] == pytest.approx(offset)
    assert model["per_rotation_training_rmse"] < 1e-12
