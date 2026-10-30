"""Count-only analysis preserves schedule semantics without a spectral solve."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from hsopt.evaluation import compiled_unitary as ev
from hsopt.evaluation.elote import term_fields


spec = importlib.util.spec_from_file_location(
    "tgate_scaling", Path(__file__).parents[2] / "figures/validation/_tgate_scaling.py"
)
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


@pytest.fixture
def problem():
    return {
        "name": "toy",
        "n_qubits": 2,
        "target_time": 0.5,
        "identity_coefficient": 0.0,
        "terms": [term_fields("XI", 0.7), term_fields("ZZ", 1.3)],
    }


@pytest.fixture
def allocation():
    return {
        "n_steps": 3,
        "uniform_epsilon": 1e-5,
        "reference_trotter_number_beta": 2.0,
        "policy_profile": "pauli",
    }


@pytest.fixture
def policy(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("""class ProductFormula:
    baseline_parameters = {}
    def __init__(self, problem):
        self.r, self.epsilon, self.beta = self.baseline_parameters[problem["name"]]
    def repetitions(self):
        return self.r
    def build_step(self, terms, step_time):
        return [(terms[0], terms[0]["coefficient"] * step_time),
                (terms[1], terms[1]["coefficient"] * step_time),
                (terms[0], terms[0]["coefficient"] * step_time)]
    def synthesis_epsilon(self, term, angle):
        return self.epsilon * (2 if term["label"] == "XI" else 1)
""")
    return path


@pytest.fixture
def forbid_physical_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("count-only analysis attempted a matrix, GPU backend, or eigensolve")

    for name in (
        "array_backend",
        "evaluate_exact_reference",
        "evaluate_molecule",
        "_build_compiled_step",
        "_build_exact_product_formula_step",
    ):
        monkeypatch.setattr(ev, name, forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eig", forbidden)


def test_compiler_count_respects_occurrences_repetitions_and_tolerances(
    problem, policy, allocation, monkeypatch, forbid_physical_evaluation
):
    def compiler(theta, epsilon):
        return {"t_count": 7 if epsilon == 2e-5 else 11}

    monkeypatch.setattr(ev, "_cudaq_synth", lambda: None)
    monkeypatch.setattr(ev, "_compile_rz", compiler)
    result = scaling.count_policy(problem, policy, allocation)
    assert result["compiled_evolution_t_count"] == (7 + 11 + 7) * 3
    assert result["unique_rotation_count"] == 2
    assert result["scheduled_rotation_count"] == 9
    assert result["exact_compiled_count"] and not result["analytic_estimate"]
    assert result["estimated_evolution_t_count"] is None
    assert not result["energy_evaluated"] and not result["accuracy_certified"]
    assert result["tolerance"]["minimum"] == 1e-5
    assert result["tolerance"]["maximum"] == 2e-5


def test_proxy_never_calls_compiler_or_physical_evaluator(
    problem, policy, allocation, monkeypatch, forbid_physical_evaluation
):
    monkeypatch.setattr(
        ev, "_compile_rotations", lambda *a: pytest.fail("compiled analytical proxy")
    )
    # Count-only scales in the term list even far beyond the statevector loader's cap.
    problem["n_qubits"] = 200
    problem["terms"] = [term_fields("XI" + "I" * 198, 0.7), term_fields("ZZ" + "I" * 198, 1.3)]
    result = scaling.count_policy(
        problem, policy, allocation, method="analytic_ross_selinger", gamma=4.0, offset=11.0
    )
    # This policy keys mixed precision on the full label, so all three now use epsilon0.
    expected = 9 * (4 * math.log2(1e5) + 11)
    assert result["estimated_evolution_t_count"] == pytest.approx(expected)
    assert result["compiled_evolution_t_count"] is None
    assert result["analytic_estimate"] and not result["exact_compiled_count"]


def test_future_grouped_frame_policy_uses_trusted_schedule(
    problem, allocation, tmp_path, forbid_physical_evaluation
):
    policy = tmp_path / "grouped.py"
    policy.write_text("""class ProductFormula:
    parameters = {}
    def __init__(self, problem):
        self.r, self.epsilon = self.parameters[problem["name"]]
    def repetitions(self):
        return self.r
    def basis_change(self):
        return [("H", 0)]
    def build_step(self, terms, step_time):
        return {"groups": [[(terms[0], step_time)], [(terms[1], step_time)]],
                "stages": [(0, .5), (1, 1), (0, .5)]}
    def synthesis_epsilon(self, term, angle):
        return self.epsilon
""")
    result = scaling.count_policy(problem, policy, allocation, method="analytic_ross_selinger")
    assert result["rotations_per_step"] == 3
    assert result["grouping"] == "explicit_commuting_blocks"
    assert result["basis_change"] == [("H", 0)]
    assert result["n_steps"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_steps", 0),
        ("n_steps", 1.5),
        ("n_steps", True),
        ("uniform_epsilon", 0),
        ("uniform_epsilon", 1),
        ("uniform_epsilon", math.nan),
        ("reference_trotter_number_beta", -1),
    ],
)
def test_reject_invalid_sealed_parameters(problem, policy, allocation, field, value):
    allocation[field] = value
    with pytest.raises(ValueError):
        scaling.count_policy(problem, policy, allocation, method="analytic_ross_selinger")


def test_reject_candidate_nonfinite_tolerance(problem, allocation, policy):
    policy.write_text(policy.read_text().replace("return self.epsilon *", "return 0 *"))
    with pytest.raises(ValueError, match="synthesis_epsilon"):
        scaling.count_policy(problem, policy, allocation, method="analytic_ross_selinger")


def test_resolved_hash_includes_tolerance_not_just_angle(problem, allocation, policy):
    a = scaling.count_policy(problem, policy, allocation, method="analytic_ross_selinger")
    allocation["uniform_epsilon"] *= 2
    b = scaling.count_policy(problem, policy, allocation, method="analytic_ross_selinger")
    assert a["rotation_schedule_sha256"] == b["rotation_schedule_sha256"]
    assert a["resolved_schedule_sha256"] != b["resolved_schedule_sha256"]


def calibration_rows():
    rows = []
    for index in range(8):
        n, terms, norm = 4 + index * 2, 20 + index**2, 1.2 + index * 0.3
        time = 1 / norm
        steps = 2 * n**0.4 * terms**0.1 * norm**0.2
        rows.append(
            {
                "name": str(index),
                "molecule": str(index // 2),
                "n_qubits": n,
                "nonidentity_terms": terms,
                "one_norm": norm,
                "target_time": time,
                "reference_trotter_number_beta": steps / time,
            }
        )
    return rows


def test_beta_model_fits_normalized_reference_steps_and_reports_cv():
    rows = calibration_rows()
    model = scaling.fit_beta_model(rows, ridge=0)
    for row in rows:
        result = scaling.predict_beta(row, model)
        assert result["reference_trotter_number_beta"] == pytest.approx(
            row["reference_trotter_number_beta"], rel=1e-10
        )
        assert result["predicted_reference_steps"] == pytest.approx(
            row["reference_trotter_number_beta"] * row["target_time"]
        )
    assert model["leave_one_out"]["log_rmse"] < 1e-9
    assert model["leave_one_molecule_out"]["log_rmse"] < 1e-8
    assert not model["accuracy_certified"]
    assert model["training_count"] == 8
    prediction = scaling.predict_beta({**rows[-1], "n_qubits": 100}, model)
    assert "log_n_qubits" in prediction["outside_training_features"]


def test_sealed_allocation_overrides_prediction_and_profiles_stay_separate(problem, allocation):
    # The bogus model must never be touched when a sealed row exists.
    result = scaling.allocate_parameters(problem, sealed=allocation, beta_model={"schema": "wrong"})
    assert result["n_steps"] == allocation["n_steps"]
    assert result["uniform_epsilon"] == allocation["uniform_epsilon"]
    assert not result["allocation_extrapolated"]
    assert result["allocation_source"] == "sealed_calibration"
    with pytest.raises(ValueError, match="profile"):
        scaling.allocate_parameters(problem, sealed=allocation, profile="fermionic_nested")
    rows = calibration_rows()
    rows[0]["policy_profile"] = "fermionic_nested"
    with pytest.raises(ValueError, match="profile"):
        scaling.fit_beta_model(rows)


def test_beta_prediction_allocation_does_not_calibrate_energy(problem, forbid_physical_evaluation):
    model = scaling.fit_beta_model(calibration_rows())
    result = scaling.allocate_parameters(problem, beta_model=model)
    assert result["allocation_extrapolated"]
    assert result["allocation_source"] == "extrapolated_tight_beta"
    assert result["n_steps"] >= 1
    assert 0 < result["uniform_epsilon"] < 1
    assert not result["beta_prediction"]["accuracy_certified"]
    assert result["error_constraint_value_hartree"] == pytest.approx(
        scaling.BENCHMARK_TOTAL_ERROR_HARTREE
    )
    with pytest.raises(ValueError, match="profile"):
        scaling.allocate_parameters(problem, beta_model=model, profile="fermionic_nested")


def test_nested_extrapolation_requires_its_own_beta_fit_and_actual_decomposition(
    problem, forbid_physical_evaluation
):
    rows = [{**row, "policy_profile": "fermionic_nested"} for row in calibration_rows()]
    model = scaling.fit_beta_model(rows, profile="fermionic_nested")
    with pytest.raises(ValueError, match="allocation_terms"):
        scaling.allocate_parameters(problem, beta_model=model, profile="fermionic_nested")
    actual_rotations = 7
    result = scaling.allocate_parameters(
        problem,
        beta_model=model,
        profile="fermionic_nested",
        allocation_terms=problem["terms"],
        rotations_per_step=actual_rotations,
    )
    assert result["scheduled_rotations_per_step"] == actual_rotations
    assert result["uniform_epsilon"] == pytest.approx(
        result["rotation_synthesis_energy_error_hartree"]
        * problem["target_time"]
        / result["n_steps"]
        / actual_rotations
    )
    assert result["allocation_extrapolated"]
