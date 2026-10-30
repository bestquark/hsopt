"""Independent finite-difference checks of the post-hoc perturbation model."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PLOTS = Path(__file__).parents[2] / "plots"
spec = importlib.util.spec_from_file_location(
    "policy_sensitivity", PLOTS / "analyze_policy_sensitivity.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_pauli_action_matches_explicit_kronecker_products():
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    rng = np.random.default_rng(5)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    for label in ("XYZ", "YIY", "IYY", "ZXI"):
        matrix = np.kron(np.kron(matrices[label[0]], matrices[label[1]]), matrices[label[2]])
        np.testing.assert_allclose(analysis.PauliActions(3).apply(label, state), matrix @ state)


def test_all_error_axes_match_eigenphase_finite_differences():
    # Noncommuting schedule with mixed X/Y/Z scaffold conventions.
    schedule = [
        ({"label": "XI"}, 0.31),
        ({"label": "ZZ"}, 0.27),
        ({"label": "IY"}, -0.17),
        ({"label": "ZI"}, 0.23),
    ]
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
    plot_spec = importlib.util.spec_from_file_location(
        "sensitivity_plot", PLOTS / "plot_policy_sensitivity.py"
    )
    plot = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(plot)
    rows = [
        {
            "case_id": "a",
            "occurrences": 2,
            "epsilon": 1e-4,
            "angle_weight": 0.9,
            "su2_weight": 0.95,
        },
        {"case_id": "a", "occurrences": 1, "epsilon": 2e-4, "angle_weight": 0.1, "su2_weight": 0.5},
        {"case_id": "a", "occurrences": 1, "epsilon": 1e-4, "angle_weight": 0.0, "su2_weight": 0.3},
    ]
    selected = plot.allocation_shapes(rows)
    assert len(selected) == 2  # Zero first-order response has no finite interior optimum.
    for model in ("angle", "su2"):
        weights = np.array([row["occurrences"] for row in selected])
        sensitivity = np.array([row[f"{model}_weight"] for row in selected])
        before = np.array([row["epsilon"] for row in selected])
        after = np.array([row[f"{model}_local_budget_matched_epsilon"] for row in selected])
        assert np.sum(weights * sensitivity * after) == pytest.approx(
            np.sum(weights * sensitivity * before)
        )
        assert sensitivity[0] * after[0] == pytest.approx(sensitivity[1] * after[1])
        assert np.sum(weights * np.log(1 / after)) <= np.sum(weights * np.log(1 / before))
        assert plot.geometric_mean(
            np.array([row[f"{model}_shape"] for row in selected]), weights
        ) == pytest.approx(1)


def test_bounded_angle_allocation_satisfies_budget_and_kkt_conditions(monkeypatch):
    monkeypatch.syspath_prepend(str(PLOTS))
    from compile_angle_allocation import bounded_allocation

    weights = np.array([2.0, 1.0, 3.0, 1.0])
    response = np.array([1.0, 0.2, 1e-12, 0.0])
    agent = np.array([2e-6, 3e-6, 2e-6, 8e-6])
    derived, info = bounded_allocation(weights, response, agent)
    assert len(derived) == len(agent)
    assert np.all((derived > 0) & (derived <= agent.max()))
    assert np.sum(weights * response * derived) == pytest.approx(
        np.sum(weights * response * agent), rel=1e-12
    )
    interior = derived < agent.max() * (1 - 1e-12)
    np.testing.assert_allclose(response[interior] * derived[interior], info["scale"], rtol=1e-12)
    assert np.all(response[~interior] * derived[~interior] <= info["scale"] * (1 + 1e-12))
    assert derived[-1] == agent.max()  # Zero sensitivity is retained with finite tolerance.
    assert info["derived_log_cost"] <= info["agent_log_cost"]


def test_zero_angle_response_uses_finite_cap_for_every_term(monkeypatch):
    monkeypatch.syspath_prepend(str(PLOTS))
    from compile_angle_allocation import bounded_allocation

    derived, info = bounded_allocation([1, 2], [0, 0], [1e-5, 2e-5])
    np.testing.assert_array_equal(derived, [2e-5, 2e-5])
    assert info["derived_dimensionless_budget"] == 0


def test_invalid_synthesis_tolerance_is_not_silently_clipped(monkeypatch):
    monkeypatch.syspath_prepend(str(PLOTS))
    from compile_angle_allocation import bounded_allocation

    with pytest.raises(ValueError, match="invalid"):
        bounded_allocation([1], [1], [1])


def test_three_axis_response_captures_error_missed_by_angle_only_model():
    # This nondegenerate conjugated-Z circuit has a first gadget whose angle
    # derivative vanishes, although another synthesis-error axis moves its phase.
    actions = analysis.PauliActions(1)
    schedule = [({"label": "X"}, np.pi / 4), ({"label": "Z"}, 0.2), ({"label": "X"}, -np.pi / 4)]
    _, vectors = np.linalg.eig(analysis.propagate(actions, schedule, np.eye(2, dtype=complex)))
    target = vectors[:, 0]
    gradients, _ = analysis.response(actions, schedule, target)
    assert abs(gradients[0, 2]) < 1e-12
    assert abs(gradients[0, 0]) == pytest.approx(1)
    assert np.linalg.norm(gradients[0]) == pytest.approx(1)
    check = analysis.validate_derivatives(actions, schedule, target, gradients)
    assert max(row["absolute_error_at_1e_6"] for row in check["finite_differences"]) < 1e-7


def test_baseline_overlay_preserves_actual_tolerance_multipliers(monkeypatch):
    monkeypatch.syspath_prepend(str(PLOTS))
    plot_spec = importlib.util.spec_from_file_location(
        "sensitivity_plot", PLOTS / "plot_policy_sensitivity.py"
    )
    plot = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(plot)
    rows = [
        {
            "case_id": "a",
            "relative_coefficient": 0.01,
            "occurrences": 2,
            "baseline_epsilon": 1e-6,
            "epsilon": 3e-6,
            "derived_epsilon": 5e-6,
            "derived_shape": 1.0,
            "learned_shape": 1.0,
        },
        {
            "case_id": "b",
            "relative_coefficient": 1.0,
            "occurrences": 1,
            "baseline_epsilon": 1e-8,
            "epsilon": 2e-8,
            "derived_epsilon": 4e-8,
            "derived_shape": 1.0,
            "learned_shape": 1.0,
        },
    ]
    x, derived, agent = plot.overlay_coordinates(
        rows, response_model="su2", normalization="baseline"
    )
    np.testing.assert_allclose(x, [0.01, 1])
    np.testing.assert_allclose(derived, [5, 4])
    np.testing.assert_allclose(agent, [3, 2])
