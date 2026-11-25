"""Regression tests distinguishing simulation error from state-preparation bias."""

import json

import numpy as np
import pytest

from hsopt.evaluation import compiled_unitary as ev


def problem_file(tmp_path):
    path = tmp_path / "hamiltonian.json"
    coefficients = [1.0, 0.7, 0.9, 0.1]
    payload = {
        "name": "HF cannot reach the ground determinant",
        "n_qubits": 4,
        "electrons": 2,
        "identity_coefficient": 0.0,
        "target_time": 1 / sum(coefficients),
        "terms": [
            {"label": "I" * i + "Z" + "I" * (3 - i), "coefficient": c}
            for i, c in enumerate(coefficients)
        ],
    }
    payload["phase_encoding"] = ev.qpe_phase_encoding(0.0, payload["terms"])
    path.write_text(json.dumps(payload))
    return path


def test_exact_hf_reference_can_be_excited_and_preserves_ground_diagnostic(tmp_path):
    path = problem_file(tmp_path)
    result = ev.evaluate_exact_reference(path, backend="numpy")
    assert result["exact_unitary_ground_energy_hartree"] == pytest.approx(-1.1)
    assert result["exact_hf_accessible_energy_hartree"] == pytest.approx(-0.7)
    assert result["exact_hf_accessible_probability"] == pytest.approx(1.0)
    assert result["hf_ground_subspace_probability"] == pytest.approx(0.0)
    assert result["exact_hf_accessible_ground_gap_hartree"] == pytest.approx(0.4)
    assert result["hf_reference_solver"]["full_diagonalization_permitted"] is False


def test_candidate_scores_fixed_hf_dynamics_without_recomputing_reference(tmp_path, monkeypatch):
    path = problem_file(tmp_path)
    reference = ev.evaluate_exact_reference(path, backend="numpy")
    problem = ev.load_problem(path)
    matrix = ev._dense_hamiltonian(problem)
    # Shift only HF's compiled energy by 10 microhartree. It remains 0.4 Ha
    # above the ground energy, but accurately reproduces the HF QPE outcome.
    hf = ev._hartree_fock_basis_index(problem)
    matrix[hf, hf] += 1e-5
    unitary = np.diag(np.exp(1j * np.diag(matrix) * problem["target_time"]))
    monkeypatch.setattr(ev, "load_policy", lambda path: object())
    monkeypatch.setattr(
        ev,
        "_build_compiled_step",
        lambda *a, **kw: (
            unitary,
            {"n_steps": 1, "t_count": 100},
        ),
    )
    monkeypatch.setattr(
        ev,
        "evaluate_exact_reference",
        lambda *a, **kw: pytest.fail("candidate recomputed its exact reference"),
    )
    result = ev.evaluate_molecule(
        path, tmp_path / "policy.py", backend="numpy", reference_result=reference
    )
    assert result["passes_chemical_accuracy"] is True
    assert result["absolute_energy_error_hartree"] == pytest.approx(1e-5)
    assert result["ground_state_energy_error_hartree"] == pytest.approx(0.40001)
    assert result["exact_reference_energy_hartree"] == pytest.approx(-0.7)


def test_legacy_ground_only_reference_cannot_silently_score_new_campaign(tmp_path):
    path = problem_file(tmp_path)
    result = ev.evaluate_exact_reference(path, backend="numpy")
    result.pop("reference_target")
    with pytest.raises(ValueError, match="HF-accessible reference preparation"):
        ev._reference_energy(ev.load_problem(path), result)


def test_ground_supported_reference_reuses_cache_without_solving(tmp_path, monkeypatch):
    path = problem_file(tmp_path)
    old = ev.evaluate_exact_reference(path, backend="numpy")
    old.pop("reference_target")
    old["hf_ground_subspace_probability"] = 0.9
    monkeypatch.setattr(
        ev,
        "_fixed_electron_hamiltonian",
        lambda *a: pytest.fail("unnecessary Hamiltonian reconstruction"),
    )
    new = ev.prepare_hf_reference(path, old)
    assert new["exact_hf_accessible_energy_hartree"] == old["exact_unitary_ground_energy_hartree"]
    assert old["exact_hf_accessible_energy_hartree"] == pytest.approx(-0.7)


def test_degenerate_qpe_peak_uses_projector_probability():
    result = ev._select_lowest_hf_accessible_peak(
        [-0.2, -0.2, 0.3],
        [0.2, 0.3, 0.5],
        n_steps=1,
        target_time=1.0,
        identity_coefficient=0.0,
    )
    assert result["compiled_pf_energy_hartree"] == pytest.approx(-0.2)
    assert result["qpe_success_probability_from_hf"] == pytest.approx(0.5)


def test_reaction_reference_and_contract_use_hf_energies(tmp_path, monkeypatch):
    path = problem_file(tmp_path)
    reference = ev.evaluate_exact_reference(path, backend="numpy")
    reaction = tmp_path / "reaction.json"
    reaction.write_text(
        json.dumps(
            {
                "name": "HF reference contract",
                "terms": [
                    {"species": "A", "coefficient": 2.0, "problem": path.name},
                ],
            }
        )
    )
    assembled = ev.assemble_exact_difference_reference(
        reaction,
        {"A": reference},
        backend="numpy",
    )
    assert assembled["exact_unitary_difference_hartree"] == pytest.approx(-1.4)
    assert assembled["exact_ground_difference_hartree"] == pytest.approx(-2.2)
    stored = tmp_path / "reference.json"
    stored.write_text(json.dumps(assembled))
    monkeypatch.setattr(
        ev,
        "evaluate_molecule",
        lambda *a, **kw: {
            "exact_reference_energy_hartree": -0.7,
            "exact_ground_energy_hartree": -1.1,
            "compiled_pf_energy_hartree": -0.7 + 1e-5,
            "qpe_success_probability_from_hf": 1.0,
            "t_count": 100,
            "evaluation_wall_seconds": 0.1,
            "phase_encoding": {"physical_evolution_time_hartree_inverse": 0.1},
        },
    )
    for contract in ev.DIFFERENCE_ERROR_CONTRACTS:
        result = ev.evaluate_difference(
            reaction,
            tmp_path / "policy.py",
            backend="numpy",
            error_contract=contract,
            reference_path=stored,
        )
        assert result["passes_chemical_accuracy"] is True
        assert result["contract_error_hartree"] == pytest.approx(2e-5)
