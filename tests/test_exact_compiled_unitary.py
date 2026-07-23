from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hsopt.compiled_unitary import (
    DENSE_EVALUATOR_MAX_QUBITS,
    TARGET_STATE_DEFINITION,
    _anchored_ground_state,
    _apply_compiled_rotation,
    _compile_rz,
    _dense_hamiltonian,
    _fixed_electron_hamiltonian,
    evaluate_exact_reference,
    evaluate_molecule,
    load_problem,
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


def test_fixed_electron_hamiltonian_matches_projected_full_matrix() -> None:
    problem = {
        "n_qubits": 4,
        "electrons": 2,
        "identity_coefficient": -0.25,
        "terms": [
            {"label": "ZIII", "coefficient": 0.3},
            {"label": "XXII", "coefficient": -0.2},
            {"label": "XYZI", "coefficient": 0.1},
        ],
    }

    indices, sector = _fixed_electron_hamiltonian(problem)
    expected = _dense_hamiltonian(problem)[np.ix_(indices, indices)]

    assert sector == pytest.approx(expected)


@pytest.mark.parametrize("label", ["XII", "IYI", "XYZ", "ZYX"])
def test_compiled_rotation_workspace_matches_dense_left_multiplication(
    label: str,
) -> None:
    rng = np.random.default_rng(20260723)
    dimension = 1 << len(label)
    unitary = (
        rng.normal(size=(dimension, dimension))
        + 1.0j * rng.normal(size=(dimension, dimension))
    ).astype(np.complex128)
    workspace = np.empty_like(unitary)
    compiled_rz = np.asarray(
        [
            [np.exp(-0.5j * 0.317), 0.0],
            [0.0, np.exp(0.5j * 0.317)],
        ],
        dtype=np.complex128,
    )

    actual, returned_workspace = _apply_compiled_rotation(
        unitary,
        workspace,
        label=label,
        compiled_rz=compiled_rz,
        xp=np,
    )

    pauli = np.asarray([[1.0 + 0.0j]])
    for character in label:
        pauli = np.kron(
            pauli,
            {
                "I": np.eye(2, dtype=np.complex128),
                "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
                "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
                "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
            }[character],
        )
    expected = (
        np.cos(0.317 / 2.0) * np.eye(dimension)
        - 1.0j * np.sin(0.317 / 2.0) * pauli
    ) @ unitary

    assert actual == pytest.approx(expected, abs=1.0e-12)
    assert returned_workspace is unitary


@pytest.mark.parametrize(
    ("n_qubits", "accepted"),
    [
        (DENSE_EVALUATOR_MAX_QUBITS, True),
        (DENSE_EVALUATOR_MAX_QUBITS + 1, False),
    ],
)
def test_dense_evaluator_qubit_limit(
    tmp_path: Path,
    n_qubits: int,
    accepted: bool,
) -> None:
    problem_path = tmp_path / f"q{n_qubits}.json"
    problem_path.write_text(
        json.dumps(
            {
                "name": "limit fixture",
                "n_qubits": n_qubits,
                "electrons": 1,
                "target_time": 0.1,
                "identity_coefficient": 0.0,
                "terms": [{"label": "Z" + "I" * (n_qubits - 1), "coefficient": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    if accepted:
        assert load_problem(problem_path)["n_qubits"] == n_qubits
    else:
        with pytest.raises(ValueError, match=f"limited to {DENSE_EVALUATOR_MAX_QUBITS}"):
            load_problem(problem_path)


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


def test_anchored_target_is_invariant_to_degenerate_eigenbasis_rotation() -> None:
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

    (
        energy,
        target,
        hf_ground_probability,
        hf_target_probability,
        dimension,
        anchor_kind,
        anchor_basis_index,
    ) = _anchored_ground_state(
        problem,
        indices_host=indices,
        eigenvalues=np.asarray([0.0, 0.0, 2.0, 2.0]),
        eigenvectors=eigenvectors,
        xp=np,
    )

    assert energy == 0.0
    assert dimension == 2
    assert hf_ground_probability == pytest.approx(1.0)
    assert hf_target_probability == pytest.approx(1.0)
    assert anchor_kind == "hartree_fock"
    assert anchor_basis_index == 8
    assert target == pytest.approx(np.asarray([0.0, 0.0, 0.0, 1.0]))


def test_anchored_target_falls_back_when_hf_has_no_ground_support() -> None:
    problem = {"n_qubits": 4, "electrons": 1}
    indices = np.asarray([1, 2, 4, 8], dtype=np.int64)

    (
        energy,
        target,
        hf_ground_probability,
        hf_target_probability,
        dimension,
        anchor_kind,
        anchor_basis_index,
    ) = _anchored_ground_state(
        problem,
        indices_host=indices,
        eigenvalues=np.asarray([0.0, 0.0, 2.0, 2.0]),
        eigenvectors=np.eye(4, dtype=np.complex128),
        xp=np,
    )

    assert energy == 0.0
    assert dimension == 2
    assert hf_ground_probability == pytest.approx(0.0)
    assert hf_target_probability == pytest.approx(0.0)
    assert anchor_kind == "lowest_resolved_fixed_electron_determinant"
    assert anchor_basis_index == 1
    assert target == pytest.approx(np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_degenerate_exact_reference_serializes_unique_anchored_ground_component(
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
    assert result["target_anchor_kind"] == "hartree_fock"
    assert result["target_anchor_basis_index"] == 8
    assert result["hf_ground_subspace_probability"] == pytest.approx(1.0)
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
