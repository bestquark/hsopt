"""Independent finite-difference checks of the post-hoc perturbation model."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PLOTS = Path(__file__).parents[2] / "plots"
spec = importlib.util.spec_from_file_location("policy_sensitivity", PLOTS / "analyze_policy_sensitivity.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_pauli_action_matches_explicit_kronecker_products():
    matrices = {
        "I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
    }
    rng = np.random.default_rng(5)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    for label in ("XYZ", "YIY", "IYY", "ZXI"):
        matrix = np.kron(np.kron(matrices[label[0]], matrices[label[1]]), matrices[label[2]])
        np.testing.assert_allclose(analysis.PauliActions(3).apply(label, state), matrix @ state)


def test_all_error_axes_match_eigenphase_finite_differences():
    # Noncommuting schedule with mixed X/Y/Z scaffold conventions.
    schedule = [({"label": "XI"}, .31), ({"label": "ZZ"}, .27),
                ({"label": "IY"}, -.17), ({"label": "ZI"}, .23)]
    actions = analysis.PauliActions(2)
    unitary = analysis.propagate(actions, schedule, np.eye(4, dtype=complex))
    _, vectors = np.linalg.eig(unitary)
    target = vectors[:, 1] / np.linalg.norm(vectors[:, 1])
    gradients, _ = analysis.response(actions, schedule, target)
    result = analysis.validate_derivatives(actions, schedule, target, gradients)
    assert result["selected_phase_multiplicity"] == 1
    assert result["step_eigenvector_residual"] < 1e-12
    assert result["ground_input_proxy_relative_gradient_l2_error"] < 1e-12
    assert len(result["finite_differences"]) == 6
    assert max(row["absolute_error_at_1e_6"] for row in result["finite_differences"]) < 1e-7


def test_tied_tolerance_optimum_counts_rotation_multiplicity(monkeypatch):
    monkeypatch.syspath_prepend(str(PLOTS))
    plot_spec = importlib.util.spec_from_file_location("sensitivity_plot", PLOTS / "plot_policy_sensitivity.py")
    plot = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(plot)
    rows = [
        {"case_id": "a", "occurrences": 2, "epsilon": 1e-4, "angle_weight": .9, "su2_weight": .95},
        {"case_id": "a", "occurrences": 1, "epsilon": 2e-4, "angle_weight": .1, "su2_weight": .5},
        {"case_id": "a", "occurrences": 1, "epsilon": 1e-4, "angle_weight": 0., "su2_weight": .3},
    ]
    selected = plot.allocation_shapes(rows)
    assert len(selected) == 2  # Zero first-order response has no finite interior optimum.
    for model in ("angle", "su2"):
        weights = np.array([row["occurrences"] for row in selected])
        sensitivity = np.array([row[f"{model}_weight"] for row in selected])
        before = np.array([row["epsilon"] for row in selected])
        after = np.array([row[f"{model}_local_budget_matched_epsilon"] for row in selected])
        assert np.sum(weights * sensitivity * after) == pytest.approx(np.sum(weights * sensitivity * before))
        assert sensitivity[0] * after[0] == pytest.approx(sensitivity[1] * after[1])
        assert np.sum(weights * np.log(1 / after)) <= np.sum(weights * np.log(1 / before))
        assert plot.geometric_mean(np.array([row[f"{model}_shape"] for row in selected]), weights) == pytest.approx(1)
