"""Known spectral measures check one-sided rejection without an eigensolver."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from hsopt.evaluation import spectral_window as window
from hsopt.evaluation import gpu_circuit, compiled_unitary as ev
from hsopt.campaign import protocol, queue


@pytest.fixture
def cpu_array_api(monkeypatch):
    class Scalar(float):
        def get(self):
            return float(self)
    cp = ModuleType("cupy")
    cp.zeros_like = np.zeros_like
    cp.linalg = SimpleNamespace(norm=lambda value: Scalar(np.linalg.norm(value)))
    monkeypatch.setitem(sys.modules, "cupy", cp)


class DiagonalUnitary:
    def __init__(self, phases, action_error=1e-13):
        self.values = np.exp(1j * np.asarray(phases))
        self.shape = (len(phases), len(phases))
        self.action_error_bound = action_error
    def __matmul__(self, vector):
        return self.values * vector


def measure(phases, weights, **kwargs):
    unitary = DiagonalUnitary(phases)
    state = np.sqrt(np.asarray(weights)).astype(complex)
    problem = {"target_time": 1.0, "identity_coefficient": 0.0}
    return window.certify_infeasible(unitary, problem, {"n_steps": 1}, state, 0.0, **kwargs)


def test_bad_dominant_peak_is_certified_without_inventing_energy(cpu_array_api):
    certificate = measure([0.0, 0.7], [0.2, 0.8])
    assert certificate["verified"] is True
    assert 0.2 <= certificate["accurate_window_probability_upper_bound"] < 0.5
    assert certificate["acceptance_permitted"] is False
    assert certificate["full_diagonalization_permitted"] is False
    assert "compiled_pf_energy_hartree" not in certificate


@pytest.mark.parametrize("weight", [0.50001, 0.6, 0.9, 1.0])
def test_accurate_majority_is_never_rejected(cpu_array_api, weight):
    assert measure([0.0, 0.7], [weight, 1 - weight]) is None


def test_chemical_accuracy_boundary_is_included_in_the_window(cpu_array_api):
    phase = ev.CHEMICAL_ACCURACY_HARTREE * (1 - 1e-12)
    assert measure([phase, -0.8], [0.6, 0.4]) is None


def test_bound_exceeds_exact_window_weight_for_random_spectral_measures(cpu_array_api):
    rng = np.random.RandomState(17)
    for _ in range(25):
        phases = rng.uniform(-0.8, 0.8, 16)
        phases[:3] = rng.uniform(-0.001, 0.001, 3)
        weights = rng.uniform(0, 1, 16); weights /= weights.sum()
        certificate = measure(phases, weights)
        if certificate:
            actual = weights[np.abs(phases) <= ev.CHEMICAL_ACCURACY_HARTREE].sum()
            assert certificate["accurate_window_probability_upper_bound"] >= actual - 1e-13


def test_wrong_operator_norm_cannot_produce_a_certificate(cpu_array_api):
    unitary = DiagonalUnitary([0.0, 0.7])
    unitary.values *= 0.9
    result = window.certify_infeasible(unitary, {"target_time": 1, "identity_coefficient": 0},
                                      {"n_steps": 1}, np.array([1.0, 0j]), 0.0)
    assert result is None


def test_missing_roundoff_contract_stays_unresolved(cpu_array_api):
    unitary = DiagonalUnitary([0.7, 0.8])
    del unitary.action_error_bound
    assert window.certify_infeasible(unitary, {}, {}, np.array([1, 0]), 0.0) is None


def test_phase_wrap_is_not_reinterpreted_as_an_accurate_branch(cpu_array_api):
    unitary = DiagonalUnitary([0.3, 0.4])
    result = window.certify_infeasible(unitary, {"target_time": 1, "identity_coefficient": 0},
                                      {"n_steps": 2}, np.array([1, 0], dtype=complex), 4.0)
    assert result is None


def rejection():
    return {"method": window.VERSION, "verified": True,
            "constraint": "accurate_majority_ground_input_qpe_peak",
            "accurate_window_probability_upper_bound": 0.2,
            "acceptance_permitted": False}


def test_lanczos_failure_can_be_resolved_as_certified_infeasible(monkeypatch):
    monkeypatch.setattr(gpu_circuit, "_ground_lanczos", lambda *a, **k: (_ for _ in ()).throw(gpu_circuit.GroundLanczosInconclusive([], 12)))
    monkeypatch.setattr(window, "certify_infeasible", lambda *a, **k: rejection())
    with pytest.raises(window.CertifiedGroundInputInfeasible) as error:
        gpu_circuit.ground_lanczos(object(), {}, {}, object(), reference_energy=0.0)
    assert error.value.certificate["previous_lanczos_matvec_calls"] == 12


def test_queue_distinguishes_certificate_from_unresolved_numerics(tmp_path, monkeypatch):
    import json
    (tmp_path / "task.json").write_text(json.dumps({
        "task_id": "x", "source": "problem.json", "kind": "absolute",
        "energy_target": "ground_branch",
        "evaluation_contract_version": queue.ground_branch.CONTRACT_VERSION,
    }))
    def fail(*args, **kwargs):
        raise window.CertifiedGroundInputInfeasible(rejection())
    monkeypatch.setattr(queue, "evaluate_molecule", fail)
    result = queue.evaluate_task(tmp_path, backend="cupy-lanczos", accelerator="test")
    assert result["status"] == "infeasible"
    assert result["energy_estimate_available"] is False
    assert result["rejection_certificate"]["verified"] is True
    monkeypatch.setattr(queue, "evaluate_molecule", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unresolved")))
    assert queue.evaluate_task(tmp_path, backend="cupy-lanczos")["status"] == "error"


def test_shared_policy_can_be_rejected_by_one_certificate_without_faking_scores():
    result = {"status": "infeasible", "rejection_certificate": rejection()}
    summary = protocol.aggregate({"molecule": result}, {"molecule": {"confidence_adjusted_t_count": 100}})
    assert summary["status"] == "infeasible"
    assert summary["score"] is None
    assert summary["unresolved_case_ids"] == []
    assert summary["case_metrics"][0]["error_hartree"] is None
    assert protocol.accept(summary, {"passes_chemical_accuracy": True}) == (False, "certified_qpe_energy_contract_violation")


def test_an_inconclusive_or_malformed_certificate_stays_an_evaluation_failure():
    result = {"status": "infeasible", "rejection_certificate": {**rejection(), "accurate_window_probability_upper_bound": 0.9}}
    summary = protocol.aggregate({"molecule": result}, {"molecule": {"confidence_adjusted_t_count": 100}})
    assert summary["status"] == "error"
    assert protocol.accept(summary, {}) == (False, "evaluation_failed")
