from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hsopt.compiled_unitary import (
    DENSE_EVALUATOR_MAX_QUBITS,
    EVALUATION_CONTRACT_VERSION,
    QPE_TARGET_CONFIDENCE,
    TARGET_STATE_DEFINITION,
    _anchored_ground_state,
    _apply_compiled_rotation,
    _build_compiled_step,
    _compile_rz,
    _dense_hamiltonian,
    _fixed_electron_hamiltonian,
    _rotation_fence_interval,
    _qpe_repetitions,
    _select_lowest_hf_accessible_peak,
    evaluate_exact_reference,
    evaluate_molecule,
    load_policy,
    load_problem,
)
from hsopt.paper_baseline import calibrate_problem
from hsopt.repository_layout import sync_layout


REPO = Path(__file__).resolve().parents[1]
ABSOLUTE = REPO / "tests" / "fixtures" / "abs_energy_h2"
DIFFERENCE = REPO / "tests" / "fixtures" / "diff_energy_h2_bond_stretch"
GENERATOR = REPO / "experiments" / "generate_initial_pf.py"


def test_experiments_tree_is_the_only_checked_in_benchmark_tree() -> None:
    experiments = REPO / "experiments"
    directories = {
        path.name for path in experiments.iterdir() if path.is_dir() and path.name != "__pycache__"
    }

    assert directories == {"abs_energy", "diff_energy"}
    assert not (REPO / "examples").exists()


def test_abs_energy_manifest_is_exactly_the_91_instance_dataset() -> None:
    manifest = REPO / "experiments" / "abs_energy" / "inputs" / "instances.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 91
    assert len({row["lane_id"] for row in rows}) == 91
    assert Counter(int(row["nqubits"]) for row in rows) == {
        4: 20,
        6: 2,
        8: 24,
        10: 11,
        12: 25,
        14: 9,
    }


def test_checked_in_benchmark_layout_matches_the_91_and_10_registries() -> None:
    result = sync_layout(
        abs_manifest=REPO / "experiments/abs_energy/inputs/instances.csv",
        abs_checksums=REPO / "experiments/abs_energy/inputs/hamlib_zip_sha256.txt",
        reaction_manifest=REPO / "experiments/diff_energy/inputs/reactions.json",
        reaction_checksums=REPO / "experiments/diff_energy/inputs/problem_cache.sha256",
        reaction_metadata=REPO / "experiments/diff_energy/inputs/problem_metadata.json",
        abs_root=REPO / "experiments/abs_energy",
        diff_root=REPO / "experiments/diff_energy",
        check=True,
    )

    assert result == {
        "absolute_hamiltonians": 91,
        "difference_reactions": 10,
        "definition_files": 101,
    }


@pytest.mark.parametrize(
    ("source", "output"),
    [
        (ABSOLUTE / "hamiltonian.json", ABSOLUTE / "initial_pf.py"),
        (DIFFERENCE / "reaction.json", DIFFERENCE / "initial_pf.py"),
    ],
)
def test_initial_policy_is_exactly_generated_from_paper_baseline(
    source: Path, output: Path, tmp_path: Path
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "n_qubits" in payload:
        problem_paths = [source]
    else:
        problem_paths = [
            (source.parent / str(item["problem"])).resolve() for item in payload["terms"]
        ]
    calibration_root = tmp_path / "calibrations"
    calibration_root.mkdir()
    for index, problem_path in enumerate(problem_paths):
        calibration = calibrate_problem(
            problem_path,
            backend="numpy",
            mode="empirical_exact",
            target_time=1.0,
        )
        (calibration_root / f"{index:03d}.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            str(source),
            "--output",
            str(output),
            "--calibration-root",
            str(calibration_root),
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


def test_rotation_fences_bound_deferred_memory_at_large_dimensions() -> None:
    assert _rotation_fence_interval(1 << 10) == 32
    assert _rotation_fence_interval(1 << 12) == 8
    assert _rotation_fence_interval(1 << 14) == 1
    assert _rotation_fence_interval(1 << 15) == 1


@pytest.mark.parametrize("label", ["XII", "IYI", "XYZ", "ZYX"])
def test_compiled_rotation_workspace_matches_dense_left_multiplication(
    label: str,
) -> None:
    rng = np.random.default_rng(20260723)
    dimension = 1 << len(label)
    unitary = (
        rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(size=(dimension, dimension))
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
        np.cos(0.317 / 2.0) * np.eye(dimension) - 1.0j * np.sin(0.317 / 2.0) * pauli
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
    assert result["exact_unitary_ground_energy_hartree"] == pytest.approx(-1.144979079498297)
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
    # Candidate peak selection must not depend on the exact ground-state vector.
    reference.pop("exact_ground_state")
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

    assert result["evaluation_contract_version"] == EVALUATION_CONTRACT_VERSION
    assert result["observable"] == "compiled_unitary_lowest_hf_accessible_eigenphase"
    assert result["exact_ground_energy_hartree"] == pytest.approx(-1.144979079498297)
    assert result["compiled_pf_energy_hartree"] == pytest.approx(-1.1444310142762317)
    assert result["passes_chemical_accuracy"] is True
    assert result["formula_order"] == 2
    assert result["n_steps"] == 2
    assert result["policy"]["term_order_strategy"] == "lexicographic"
    assert result["t_count"] == 3_068
    assert result["per_attempt_t_count"] == result["t_count"]
    assert result["qpe_repetitions"] == 1
    assert result["qpe_target_confidence"] == QPE_TARGET_CONFIDENCE
    assert result["qpe_achieved_confidence"] >= QPE_TARGET_CONFIDENCE
    assert result["confidence_adjusted_t_count"] == result["t_count"]
    assert result["score"] == result["confidence_adjusted_t_count"]
    assert result["evaluation_wall_seconds"] > 0.0
    assert result["timing_seconds"]["compiled_unitary_build"] > 0.0
    assert result["timing_seconds"]["compiled_unitary_eigensolve"] > 0.0
    assert result["rotation_fence_interval"] == 32
    assert result["qpe_success_probability_from_hf"] == pytest.approx(0.9884325016034219)
    assert all("gate_string" not in row for row in result["unique_compilations"])
    assert all(len(row["gate_string_sha256"]) == 64 for row in result["unique_compilations"])


def test_step_eigendecomposition_equals_explicit_full_compiled_pf_product() -> None:
    problem = load_problem(ABSOLUTE / "hamiltonian.json")
    policy = load_policy(ABSOLUTE / "initial_pf.py")
    reference = evaluate_exact_reference(
        ABSOLUTE / "hamiltonian.json",
        backend="numpy",
    )
    evaluated = evaluate_molecule(
        ABSOLUTE / "hamiltonian.json",
        ABSOLUTE / "initial_pf.py",
        backend="numpy",
        reference_result=reference,
    )
    step, metrics = _build_compiled_step(
        problem,
        policy,
        backend_name="numpy",
        xp=np,
    )
    full_compiled_pf = np.linalg.matrix_power(step, int(metrics["n_steps"]))
    eigenvalues, eigenvectors = np.linalg.eig(full_compiled_pf)
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    hf = np.zeros(1 << int(problem["n_qubits"]), dtype=np.complex128)
    hf_index = sum(
        1 << (int(problem["n_qubits"]) - qubit - 1) for qubit in range(int(problem["electrons"]))
    )
    hf[hf_index] = 1.0
    explicit = _select_lowest_hf_accessible_peak(
        np.angle(eigenvalues),
        np.abs(eigenvectors.conj().T @ hf) ** 2,
        n_steps=1,
        target_time=float(problem["target_time"]),
        identity_coefficient=float(problem["identity_coefficient"]),
        exact_energy=float(reference["exact_unitary_ground_energy_hartree"]),
    )

    assert explicit["compiled_pf_energy_hartree"] == pytest.approx(
        evaluated["compiled_pf_energy_hartree"],
        abs=1.0e-10,
    )


def test_lowest_hf_accessible_peak_is_not_the_largest_probability_peak() -> None:
    selected = _select_lowest_hf_accessible_peak(
        [0.2, -0.4, -0.8],
        [0.80, 0.19, 1.0e-16],
        n_steps=1,
        target_time=1.0,
        identity_coefficient=0.0,
        exact_energy=0.0,
    )

    assert selected["selected_eigen_index"] == 1
    assert selected["compiled_pf_energy_hartree"] == pytest.approx(-0.4)
    assert selected["qpe_success_probability_from_hf"] == pytest.approx(0.19)


def test_qpe_repetition_cost_reaches_95_percent_without_a_physical_p_min() -> None:
    assert _qpe_repetitions(1.0) == 1
    assert _qpe_repetitions(0.5) == 5
    tiny_probability_repetitions = _qpe_repetitions(1.0e-6)

    assert tiny_probability_repetitions == 2_995_731
    assert 1.0 - (1.0 - 1.0e-6) ** tiny_probability_repetitions >= 0.95
