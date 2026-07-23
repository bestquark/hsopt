from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hsopt.compiled_unitary import (
    TARGET_STATE_DEFINITION,
    _compile_rz,
    _hf_projected_ground_state,
    evaluate_exact_reference,
    evaluate_molecule,
)


REPO = Path(__file__).resolve().parents[1]
ABSOLUTE = REPO / "examples" / "abs_energy" / "hamlib_h2_r0_7"
DIFFERENCE = REPO / "examples" / "diff_energy" / "h2_bond_stretch"
GENERATOR = REPO / "examples" / "generate_initial_pf.py"


@pytest.mark.parametrize(
    ("source", "output"),
    [
        (ABSOLUTE / "hamiltonian.json", ABSOLUTE / "initial_pf.py"),
        (DIFFERENCE / "reaction.json", DIFFERENCE / "initial_pf.py"),
    ],
)
def test_initial_policy_is_exactly_generated_from_paper_baseline(
    source: Path, output: Path
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            str(source),
            "--output",
            str(output),
            "--check",
        ],
        check=True,
    )
    generated = output.read_text()
    assert "DOI 10.1073/pnas.1619152114" in generated
    assert "separately cached U_exact reference" in generated
    assert not (output.parent / "optimized_pf.py").exists()


def test_pygridsynth_compilation_is_reproducible_and_within_tolerance() -> None:
    first = _compile_rz(0.2718281828, 1.0e-5)
    second = _compile_rz(0.2718281828, 1.0e-5)

    assert first["gate_string"] == second["gate_string"]
    assert first["t_count"] > 0
    assert first["operator_error"] <= 1.0e-5


def test_exact_reference_is_obtained_from_the_exact_unitary_once() -> None:
    result = evaluate_exact_reference(
        ABSOLUTE / "hamiltonian.json",
        backend="numpy",
    )

    assert result["observable"] == "exact_unitary_ground_eigenphase"
    assert result["unitary_definition"] == "U_exact = exp(+i H t)"
    assert result["exact_unitary_ground_energy_hartree"] == pytest.approx(
        -1.144979079498297
    )
    assert result["hamiltonian_ground_energy_hartree"] == pytest.approx(
        result["exact_unitary_ground_energy_hartree"]
    )
    assert result["unitary_hamiltonian_consistency_error_hartree"] < 1.0e-12
    assert result["selected_ground_state_overlap"] == pytest.approx(1.0)
    assert result["exact_ground_subspace_dimension"] == 1
    assert result["target_state_definition"] == TARGET_STATE_DEFINITION


def test_hf_projected_target_is_invariant_to_degenerate_eigenbasis_rotation() -> None:
    problem = {"n_qubits": 4, "electrons": 1}
    indices = np.asarray([1, 2, 4, 8], dtype=np.int64)
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    eigenvectors = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [inverse_sqrt_two, inverse_sqrt_two, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [inverse_sqrt_two, -inverse_sqrt_two, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    energy, target, probability, dimension = _hf_projected_ground_state(
        problem,
        indices_host=indices,
        eigenvalues=np.asarray([0.0, 0.0, 2.0, 2.0]),
        eigenvectors=eigenvectors,
        xp=np,
    )

    assert energy == 0.0
    assert dimension == 2
    assert probability == pytest.approx(1.0)
    assert target == pytest.approx(np.asarray([0.0, 0.0, 0.0, 1.0]))


def test_degenerate_exact_reference_serializes_unique_hf_ground_component(
    tmp_path: Path,
) -> None:
    problem_path = tmp_path / "degenerate.json"
    problem_path.write_text(
        json.dumps(
            {
                "name": "degenerate HF target",
                "n_qubits": 4,
                "electrons": 1,
                "target_time": 0.2,
                "identity_coefficient": 0.0,
                "terms": [
                    {"label": "ZIII", "coefficient": 1.0},
                    {"label": "IIZI", "coefficient": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_exact_reference(problem_path, backend="numpy")
    target = np.asarray(
        [complex(real, imaginary) for real, imaginary in result["exact_ground_state"]]
    )

    assert result["exact_ground_subspace_dimension"] == 2
    assert result["target_state_definition"] == TARGET_STATE_DEFINITION
    assert result["qpe_success_probability_from_hf"] == pytest.approx(1.0)
    assert target[8] == pytest.approx(1.0)
    assert np.linalg.norm(np.delete(target, 8)) == pytest.approx(0.0)


def test_absolute_baseline_uses_cached_exact_reference_and_compiled_eigenphase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = evaluate_exact_reference(
        ABSOLUTE / "hamiltonian.json",
        backend="numpy",
    )
    monkeypatch.setattr(
        "hsopt.compiled_unitary.evaluate_exact_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the candidate evaluator recomputed U_exact")
        ),
    )
    result = evaluate_molecule(
        ABSOLUTE / "hamiltonian.json",
        ABSOLUTE / "initial_pf.py",
        backend="numpy",
        reference_result=reference,
    )

    assert result["observable"] == "compiled_unitary_ground_eigenphase"
    assert result["exact_ground_energy_hartree"] == pytest.approx(-1.144979079498297)
    assert result["compiled_pf_energy_hartree"] == pytest.approx(-1.1449932429897123)
    assert result["passes_chemical_accuracy"] is True
    assert result["formula_order"] == 2
    assert result["n_steps"] == 512
    assert result["policy"]["term_order_strategy"] == "lexicographic"
    assert result["t_count"] == 1_072_128
    assert result["qpe_success_probability_from_hf"] == pytest.approx(0.9890367894564321)
    assert all("gate_string" not in row for row in result["unique_compilations"])
    assert all(len(row["gate_string_sha256"]) == 64 for row in result["unique_compilations"])
