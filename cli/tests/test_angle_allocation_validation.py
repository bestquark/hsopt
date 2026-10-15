"""Post-hoc allocation binding and full-suite feasibility reporting."""

import importlib.util
from pathlib import Path

import pytest

from hsopt.campaign.protocol import bind_policy
from hsopt.evaluation import compiled_unitary as ev


PLOTS = Path(__file__).parents[2] / "plots"
spec = importlib.util.spec_from_file_location("angle_validation", PLOTS / "validate_angle_allocation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_derived_binding_preserves_schedule_and_sealed_repetitions(tmp_path):
    source = tmp_path / "shared.py"
    source.write_text('''class ProductFormula:
    baseline_parameters = {}

    def __init__(self, problem):
        self.r, self.epsilon, self.beta = self.baseline_parameters[problem["name"]]

    def repetitions(self):
        return self.r

    def build_step(self, terms, step_time):
        return [(t, t["coefficient"] * step_time) for t in terms]

    def synthesis_epsilon(self, term, angle):
        return self.epsilon
''')
    parameters = {"toy": {"n_steps": 7, "uniform_epsilon": 1e-5,
                           "reference_trotter_number_beta": .2}}
    problem = {"name": "toy", "terms": [{"label": "X", "coefficient": .3},
                                         {"label": "Z", "coefficient": -.7}]}
    original, derived = tmp_path / "original.py", tmp_path / "derived.py"
    bind_policy(source, parameters, original)
    epsilon = {"X": 3e-5, "Z": 2e-6}
    validation.derived_policy(source, parameters, epsilon, derived)
    first = ev.load_policy(original).ProductFormula(problem)
    second = ev.load_policy(derived).ProductFormula(problem)
    assert first.repetitions() == second.repetitions() == 7
    assert first.build_step(problem["terms"], .1) == second.build_step(problem["terms"], .1)
    assert first.epsilon == second.epsilon == 1e-5
    assert first.beta == second.beta == .2
    for term in problem["terms"]:
        for angle in (-.3, .2):
            assert second.synthesis_epsilon(term, angle) == epsilon[term["label"]]


@pytest.mark.parametrize("mode", ["pending", "physical_failure", "uncertified", "pass"])
def test_full_suite_score_requires_every_case_to_pass(mode):
    baseline = {key: {"t_count": 100, "score": 200} for key in ("a", "b")}
    agent = {key: {"t_count": 90, "score": 180} for key in baseline}
    compilation = {"agent_total_compiled_evolution_t_count": 180,
                   "derived_total_compiled_evolution_t_count": 160}
    passed = {"status": "ok", "passes_chemical_accuracy": True, "score": 160,
              "t_count": 80, "absolute_energy_error_mha": .1}
    results = {"a": dict(passed)}
    if mode == "physical_failure":
        results["b"] = {**passed, "passes_chemical_accuracy": False, "score": None,
                        "absolute_energy_error_mha": 2.0}
    elif mode == "uncertified":
        results["b"] = {"status": "uncertified", "error": "no certified peak"}
    elif mode == "pass":
        results["b"] = dict(passed)
    report = validation.summarize(results, baseline, agent, compilation)
    assert report["passes_entire_suite"] == (mode == "pass")
    assert report["agent_geomean_relative_t95"] == pytest.approx(.9)
    if mode == "pass":
        assert report["derived_geomean_relative_t95"] == pytest.approx(.8)
        assert report["derived_t95_reduction_vs_baseline"] == pytest.approx(.2)
    else:
        assert report["derived_geomean_relative_t95"] is None
        assert report["derived_t95_reduction_vs_baseline"] is None
    assert report["physical_accuracy_failures"] == (["b"] if mode == "physical_failure" else [])
    assert report["uncertified_cases"] == (["b"] if mode == "uncertified" else [])
