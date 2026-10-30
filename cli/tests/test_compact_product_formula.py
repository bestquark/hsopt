"""Preserve sealed starting policies while exposing a small, mutable PF object."""

from pathlib import Path


from hsopt.campaign.baseline import baseline_module
from hsopt.evaluation import compiled_unitary as evaluator

GENERATOR = baseline_module()
REPO = Path(__file__).resolve().parents[2]


def test_factory_returns_same_schedule_as_emitted_policy(tmp_path):
    problem = evaluator.load_problem(REPO / "cli/tests/fixtures/abs_energy_h2/hamiltonian.json")
    rows = {problem["name"]: {"n_steps": 13, "uniform_epsilon": 1e-6}}
    formula = GENERATOR.product_formula(problem, rows)
    path = tmp_path / "candidate.py"
    path.write_text(
        GENERATOR._policy_source().replace(
            "parameters = {}", "parameters = " + GENERATOR.format_parameters(rows)
        )
    )
    emitted = evaluator.load_policy(path).ProductFormula(problem)
    dt = problem["target_time"] / formula.repetitions()
    assert formula.repetitions() == emitted.repetitions() == 13
    assert formula.build_step(problem["terms"], dt) == emitted.build_step(problem["terms"], dt)
    assert formula.synthesis_epsilon(problem["terms"][0], dt) == emitted.synthesis_epsilon(
        problem["terms"][0], dt
    )


def test_agent_can_change_r_schedule_and_precision_independently(tmp_path, monkeypatch):
    problem = evaluator.load_problem(REPO / "cli/tests/fixtures/abs_energy_h2/hamiltonian.json")
    rows = {problem["name"]: {"n_steps": 3, "uniform_epsilon": 1e-6}}
    source = GENERATOR._policy_source().replace(
        "parameters = {}", "parameters = " + GENERATOR.format_parameters(rows)
    )
    # One candidate changes step count, ordering and synthesis precision together.
    source = source.replace("return self.n_steps", "return 2 * self.n_steps")
    source = source.replace(
        'key=lambda term: term["label"]', 'key=lambda term: -abs(term["coefficient"])'
    )
    source = source.replace(
        "return self.epsilon",
        'return self.epsilon * (2.0 if term["x_count"] + term["y_count"] == 0 else 1.0)',
    )
    path = tmp_path / "candidate.py"
    path.write_text(source)
    evaluator.validate_policy_source(path)
    formula = evaluator.load_policy(path).ProductFormula(problem)
    dt = problem["target_time"] / formula.repetitions()
    rotations = formula.build_step(problem["terms"], dt)
    assert formula.repetitions() == 6
    assert {formula.synthesis_epsilon(term, angle) for term, angle in rotations} == {1e-6, 2e-6}
    # Test the real compiler handoff, without timing hardware or scoring a proposal.
    monkeypatch.setattr(evaluator, "_cudaq_synth", lambda: (None, None))
    monkeypatch.setattr(evaluator, "_synthesis_worker_count", lambda: 1)
    monkeypatch.setattr(evaluator, "_compile_rz", lambda theta, eps: {"epsilon": eps})
    keys, _, _ = evaluator._compile_rotations(rotations, formula)
    assert {epsilon for theta, epsilon in keys} == {1e-6, 2e-6}
