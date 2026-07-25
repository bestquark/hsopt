from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
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
    _build_exact_product_formula_step,
    _compile_rz,
    _dense_hamiltonian,
    _fixed_electron_hamiltonian,
    _rotation_fence_interval,
    _qpe_repetitions,
    _product_formula_schedule,
    _select_lowest_hf_accessible_peak,
    evaluate_exact_reference,
    evaluate_molecule,
    load_policy,
    load_problem,
    qpe_phase_encoding,
    validate_policy_source,
)
from hsopt.campaign_setup import normalize_materialized_inputs
from hsopt.experiment_evaluator import evaluate_absolute_directory
from hsopt.paper_baseline import (
    BENCHMARK_TOTAL_ERROR_HARTREE,
    PAPER_ALLOCATION_METHOD,
    PAPER_SYNTHESIS_DELTA_0,
    PAPER_SYNTHESIS_GAMMA,
    _double_commutator_bound,
    calibrate_problem,
    optimize_paper_error_allocation,
    seal_calibration,
)
from hsopt.paper_baseline_batch import build_manifest, generate_policies, run_batch
from hsopt.repository_layout import sync_layout


REPO = Path(__file__).resolve().parents[1]
ABSOLUTE = REPO / "tests" / "fixtures" / "abs_energy_h2"
DIFFERENCE = REPO / "tests" / "fixtures" / "diff_energy_h2_bond_stretch"
GENERATOR = REPO / "experiments" / "generate_initial_pf.py"


def _encoded_problem(problem: dict[str, object]) -> dict[str, object]:
    """Attach the same phase encoding required of production Hamiltonians."""

    encoding = qpe_phase_encoding(
        float(problem["identity_coefficient"]),
        problem["terms"],
    )
    problem["phase_encoding"] = encoding
    problem["target_time"] = encoding["physical_evolution_time_hartree_inverse"]
    return problem


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
        "evaluator_files": 2,
        "initial_policies": 101,
    }
    for family in ("abs_energy", "diff_energy"):
        root = REPO / "experiments" / family
        evaluator = root / "evaluate.py"
        assert evaluator.is_file()
        assert not list(root.glob("*/evaluate.py"))
        source = evaluator.read_text(encoding="utf-8")
        assert "from hsopt.compiled_unitary import" in source
        assert "hsopt.experiment_evaluator" not in source


def test_normalize_materialized_inputs_copies_only_fresh_phase_encoded_inputs(
    tmp_path: Path,
) -> None:
    abs_inputs = tmp_path / "source/abs_energy"
    abs_problem = abs_inputs / "h2/hamiltonian.json"
    abs_problem.parent.mkdir(parents=True)
    abs_problem.write_bytes((ABSOLUTE / "hamiltonian.json").read_bytes())
    source_abs_sha256 = hashlib.sha256(abs_problem.read_bytes()).hexdigest()
    (abs_inputs / "manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "family": "abs_energy",
                "instances": [
                    {
                        "instance": "h2",
                        "problem": "h2/hamiltonian.json",
                        "problem_sha256": source_abs_sha256,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reaction_inputs = tmp_path / "source/diff_energy"
    shutil.copytree(DIFFERENCE, reaction_inputs / "stretch")
    (reaction_inputs / "stretch/paper_seed.json").write_text(
        '{"obsolete": true}\n',
        encoding="utf-8",
    )
    (reaction_inputs / "manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "family": "diff_energy",
                "instances": [
                    {
                        "instance": "stretch",
                        "reaction": "stretch/reaction.json",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = normalize_materialized_inputs(
        abs_inputs,
        reaction_inputs,
        tmp_path / "normalized",
    )
    normalized_abs = tmp_path / "normalized/abs_energy/h2/hamiltonian.json"
    normalized_manifest = json.loads(
        (tmp_path / "normalized/abs_energy/manifest.json").read_text(encoding="utf-8")
    )

    assert result["absolute_hamiltonians"] == 1
    assert result["reaction_instances"] == 1
    assert result["reaction_hamiltonian_files"] == 2
    assert result["unique_hamiltonians"] == 2
    assert load_problem(normalized_abs)["target_time"] == pytest.approx(
        0.5127568450530159
    )
    assert normalized_manifest["instances"][0]["source_problem_sha256"] == (
        source_abs_sha256
    )
    assert not (tmp_path / "normalized/diff_energy/stretch/paper_seed.json").exists()


def test_directory_evaluator_uses_its_sealed_hamiltonian_identity(tmp_path: Path) -> None:
    problem = json.loads((ABSOLUTE / "hamiltonian.json").read_text(encoding="utf-8"))
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "instance.json").write_text(
        json.dumps(
            {
                "instance": str(problem["name"]).removeprefix("HamLib "),
                "n_qubits": int(problem["n_qubits"]),
            }
        ),
        encoding="utf-8",
    )
    reference_path = tmp_path / "exact-reference.json"
    reference_path.write_text(
        json.dumps(
            evaluate_exact_reference(
                ABSOLUTE / "hamiltonian.json",
                backend="numpy",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    status = evaluate_absolute_directory(
        experiment,
        [
            "--problem",
            str(ABSOLUTE / "hamiltonian.json"),
            "--policy-file",
            str(ABSOLUTE / "initial_pf.py"),
            "--exact-reference",
            str(reference_path),
            "--backend",
            "numpy",
            "--output",
            str(output),
        ],
    )

    assert status == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["problem"] == problem["name"]
    assert result["formula_order"] == 2
    assert result["policy"]["class"] == "ProductFormula"


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
            mode="hamiltonian_bound",
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
    assert "Reiher et al. Eq. (19)/(34) rigorous PF2" in generated
    assert "adapted to 1 kcal/mol" in generated
    assert "class ProductFormula" in generated
    assert "def repetitions(self)" in generated
    assert "def build_step(self, terms, step_time)" in generated
    assert "def synthesis_epsilon(self, term, angle)" in generated
    assert "BASELINES" not in generated
    assert "get_method_config" not in generated
    assert not (output.parent / "optimized_pf.py").exists()


def test_eq19_pauli_bound_matches_direct_ordered_triple_sum() -> None:
    problem = {
        "target_time": 1.0,
        "terms": [
            {"label": "XI", "coefficient": 0.25},
            {"label": "IZ", "coefficient": -0.5},
            {"label": "XX", "coefficient": 0.75},
            {"label": "YY", "coefficient": -0.125},
        ],
    }
    terms = sorted(problem["terms"], key=lambda term: str(term["label"]))

    def anticommutes(left: str, right: str) -> bool:
        mismatches = sum(a != b and a != "I" and b != "I" for a, b in zip(left, right, strict=True))
        return bool(mismatches % 2)

    expected_terms: list[float] = []
    for alpha, left in enumerate(terms):
        for beta, middle in enumerate(terms):
            for alpha_prime, right in enumerate(terms):
                ordering = int(alpha > beta and alpha_prime > beta) + int(
                    beta > alpha and alpha_prime == alpha
                )
                inner = anticommutes(str(middle["label"]), str(right["label"]))
                outer = anticommutes(str(left["label"]), str(middle["label"])) != anticommutes(
                    str(left["label"]), str(right["label"])
                )
                if ordering and inner and outer:
                    expected_terms.append(
                        4.0
                        * abs(float(left["coefficient"]))
                        * abs(float(middle["coefficient"]))
                        * abs(float(right["coefficient"]))
                        * ordering
                    )
    expected = sum(expected_terms)
    result = _double_commutator_bound(
        problem,
        reference_error_hartree=1.0e-4,
    )

    assert result["bound_method"] == "exact_vectorized_pauli_sum"
    assert result["double_commutator_sum_float"] == pytest.approx(expected)
    assert result["double_commutator_sum_upper"] >= expected
    assert result["reference_trotter_number_beta"] == pytest.approx(
        np.sqrt(result["double_commutator_sum_upper"] / 1.0e-4)
    )


def test_bound_beta_is_independent_of_the_cached_exact_reference() -> None:
    calibration = calibrate_problem(
        ABSOLUTE / "hamiltonian.json",
        backend="numpy",
    )

    assert calibration["calibration_mode"] == "hamiltonian_bound"
    assert calibration["reference"]["exact_ground_energy_hartree"] is None
    assert calibration["exact_reference"]["observable"] == "exact_unitary_ground_eigenphase"
    assert calibration["timing_seconds"]["exact_reference"] >= 0.0


def test_policy_only_and_exact_calibrations_have_the_same_scientific_signature() -> None:
    policy_only = calibrate_problem(
        ABSOLUTE / "hamiltonian.json",
        backend="numpy",
        include_exact_reference=False,
    )
    with_reference = calibrate_problem(
        ABSOLUTE / "hamiltonian.json",
        backend="numpy",
        include_exact_reference=True,
    )

    assert policy_only["instantiated_policy"] == with_reference["instantiated_policy"]
    assert policy_only["calibration_sha256"] == with_reference["calibration_sha256"]


def test_initial_policy_uses_paper_eq34_calculus_before_iteration_zero() -> None:
    problem = json.loads((ABSOLUTE / "hamiltonian.json").read_text(encoding="utf-8"))
    bound = _double_commutator_bound(
        problem,
        reference_error_hartree=1.0e-4,
    )
    policy, derivation = optimize_paper_error_allocation(
        problem,
        beta=float(bound["reference_trotter_number_beta"]),
        reference_error_hartree=1.0e-4,
        total_error_hartree=BENCHMARK_TOTAL_ERROR_HARTREE,
    )

    assert policy["error_allocation"] == PAPER_ALLOCATION_METHOD
    assert policy["paper_synthesis_gamma"] == PAPER_SYNTHESIS_GAMMA == 4.0
    assert policy["paper_synthesis_delta_0"] == PAPER_SYNTHESIS_DELTA_0 == 11.0
    assert policy["phase_estimation_error_hartree"] == pytest.approx(
        2.0 * policy["trotter_error_hartree"]
    )
    assert (
        policy["phase_estimation_error_hartree"]
        + policy["trotter_error_hartree"]
        + policy["rotation_synthesis_energy_error_hartree"]
    ) == pytest.approx(BENCHMARK_TOTAL_ERROR_HARTREE)
    assert policy["uniform_epsilon"] == pytest.approx(
        policy["rotation_synthesis_energy_error_hartree"]
        * policy["step_time"]
        / policy["scheduled_rotations_per_step"]
    )
    assert policy["realized_t_count"] is None
    assert policy["realized_t_count_stage"] == ("iteration_0_exact_compiled_unitary_evaluation")
    assert derivation["paper_equation"].endswith("Eq_34")
    assert abs(derivation["root_residual_hartree"]) < 1.0e-14


def test_batch_generator_can_materialize_a_checked_in_policy(
    tmp_path: Path,
) -> None:
    abs_inputs = tmp_path / "inputs/abs_energy"
    reaction_inputs = tmp_path / "inputs/diff_energy"
    abs_inputs.mkdir(parents=True)
    reaction_inputs.mkdir(parents=True)
    problem = abs_inputs / "h2.json"
    problem.write_bytes((ABSOLUTE / "hamiltonian.json").read_bytes())
    (abs_inputs / "manifest.json").write_text(
        json.dumps(
            {
                "family": "abs_energy",
                "instances": [
                    {
                        "instance": "h2",
                        "problem": "h2.json",
                        "n_qubits": 4,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (reaction_inputs / "manifest.json").write_text(
        json.dumps({"family": "diff_energy", "instances": []}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = build_manifest(abs_inputs, reaction_inputs, manifest_path)

    calibration = calibrate_problem(problem, backend="numpy")
    calibration.update(
        {
            "dataset_key": "h2",
            "dataset_family": "abs_energy",
            "n_qubits": 4,
            "consumers": ["abs_energy/h2"],
        }
    )
    calibration = seal_calibration(calibration)
    calibration_dir = tmp_path / "calibrations"
    calibration_path = calibration_dir / "abs_energy/h2.json"
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repository_root = tmp_path / "repo"
    (repository_root / "experiments").mkdir(parents=True)
    (repository_root / "experiments/generate_initial_pf.py").write_text(
        "# repository marker\n",
        encoding="utf-8",
    )
    result = generate_policies(
        manifest_path,
        calibration_dir,
        tmp_path / "external-policies",
        generator=GENERATOR,
        repository_root=repository_root,
    )

    checked_in = repository_root / "experiments/abs_energy/h2/initial_pf.py"
    external = tmp_path / "external-policies/abs_energy/h2/initial_pf.py"
    assert manifest["absolute_hamiltonians"] == 1
    assert result["absolute_initial_policies"] == 1
    assert checked_in.read_bytes() == external.read_bytes()
    assert result["policies"][0]["checked_in_policy"] == ("experiments/abs_energy/h2/initial_pf.py")


def test_batch_reuses_exact_reference_when_allocation_schema_changes(
    tmp_path: Path,
) -> None:
    abs_inputs = tmp_path / "inputs/abs_energy"
    reaction_inputs = tmp_path / "inputs/diff_energy"
    abs_inputs.mkdir(parents=True)
    reaction_inputs.mkdir(parents=True)
    problem = abs_inputs / "h2.json"
    problem.write_bytes((ABSOLUTE / "hamiltonian.json").read_bytes())
    (abs_inputs / "manifest.json").write_text(
        json.dumps(
            {
                "family": "abs_energy",
                "instances": [
                    {
                        "instance": "h2",
                        "problem": "h2.json",
                        "n_qubits": 4,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (reaction_inputs / "manifest.json").write_text(
        json.dumps({"family": "diff_energy", "instances": []}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    build_manifest(abs_inputs, reaction_inputs, manifest_path)
    output_dir = tmp_path / "results"
    stale = calibrate_problem(problem, backend="numpy")
    stale["schema"] = 3
    stale = seal_calibration(stale)
    stale_path = output_dir / "abs_energy/h2.json"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = run_batch(
        manifest_path,
        output_dir,
        backend="numpy",
        mode="hamiltonian_bound",
        target_time=None,
        reference_error_hartree=1.0e-4,
        total_error_hartree=BENCHMARK_TOTAL_ERROR_HARTREE,
        shard_index=0,
        shard_count=1,
        max_steps=1 << 20,
        include_exact_reference=True,
    )
    refreshed = json.loads(stale_path.read_text(encoding="utf-8"))

    assert summary["completed"] == 1
    assert summary["exact_references_reused"] == 1
    assert refreshed["schema"] == 5
    assert refreshed["exact_reference_reused"] is True
    assert refreshed["exact_reference"] == stale["exact_reference"]


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
            _encoded_problem({
                "name": "limit fixture",
                "n_qubits": n_qubits,
                "electrons": 1,
                "identity_coefficient": 0.0,
                "terms": [{"label": "Z" + "I" * (n_qubits - 1), "coefficient": 1.0}],
            })
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
    assert result["unitary_definition"] == "U_exact = exp(+i (H - c0 I) / lambda)"
    assert result["phase_encoding"]["encoded_phase_bound_radians"] == [-1.0, 1.0]
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
            _encoded_problem({
                "name": "degenerate HF target",
                "n_qubits": 4,
                "electrons": 1,
                "identity_coefficient": 0.0,
                "terms": [
                    {"label": "ZIII", "coefficient": 1.0},
                    {"label": "IIZI", "coefficient": 1.0},
                ],
            })
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
    assert result["compiled_pf_energy_hartree"] == pytest.approx(-1.144983078779298)
    assert result["passes_chemical_accuracy"] is True
    assert result["formula_order"] == 2
    assert result["n_steps"] == 15
    assert result["policy"]["term_order_strategy"] == "lexicographic"
    assert result["t_count"] == 30_420
    assert result["compiled_evolution_t_count"] == result["t_count"]
    assert result["per_attempt_t_count"] == 938_000_700
    assert result["qpe_repetitions"] == 5
    assert result["qpe_target_confidence"] == QPE_TARGET_CONFIDENCE
    assert result["qpe_achieved_confidence"] >= QPE_TARGET_CONFIDENCE
    assert result["qpe_full_evolution_queries"] == 30_835
    assert result["confidence_adjusted_t_count"] == 4_690_003_500
    assert result["score"] == result["confidence_adjusted_t_count"]
    assert result["evaluation_wall_seconds"] > 0.0
    assert result["timing_seconds"]["compiled_unitary_build"] > 0.0
    assert result["timing_seconds"]["compiled_unitary_eigensolve"] > 0.0
    assert result["rotation_fence_interval"] == 32
    assert result["qpe_success_probability_from_hf"] == pytest.approx(0.9890307462934342)
    assert result["qpe_success_probability_per_attempt"] == pytest.approx(
        0.5 * result["qpe_success_probability_from_hf"]
    )
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
    )

    assert explicit["compiled_pf_energy_hartree"] == pytest.approx(
        evaluated["compiled_pf_energy_hartree"],
        abs=1.0e-10,
    )


def test_product_formula_schedule_has_exact_and_compiled_realizations() -> None:
    problem = load_problem(ABSOLUTE / "hamiltonian.json")
    policy = load_policy(ABSOLUTE / "initial_pf.py")
    exact_step, exact_metrics = _build_exact_product_formula_step(
        problem,
        policy,
        backend_name="numpy",
        xp=np,
    )
    _formula, rotations, schedule_metrics = _product_formula_schedule(problem, policy)

    paulis = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }
    dimension = 1 << int(problem["n_qubits"])
    explicit = np.eye(dimension, dtype=np.complex128)
    identity = np.eye(dimension, dtype=np.complex128)
    for term, angle in rotations:
        dense_pauli = paulis[term["label"][0]]
        for symbol in term["label"][1:]:
            dense_pauli = np.kron(dense_pauli, paulis[symbol])
        rotation = math.cos(angle) * identity + 1.0j * math.sin(angle) * dense_pauli
        explicit = rotation @ explicit

    assert exact_step == pytest.approx(explicit, abs=1.0e-12)
    assert exact_metrics["rotation_implementation"] == "ideal_pauli_rotations"
    assert exact_metrics["rotation_schedule_sha256"] == schedule_metrics["rotation_schedule_sha256"]
    assert schedule_metrics["formula_order"] == 2
    assert schedule_metrics["term_order_strategy"] == "lexicographic"


def test_product_formula_can_define_a_custom_recurrence_without_declared_order(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        """class ProductFormula:
    def __init__(self, problem):
        self.epsilon = 1.0e-6

    def repetitions(self):
        return 3

    def build_step(self, terms, step_time):
        first = sorted(terms, key=lambda term: -abs(float(term["coefficient"])))[0]
        return [(first, float(first["coefficient"]) * step_time)]

    def synthesis_epsilon(self, term, angle):
        return self.epsilon
""",
        encoding="utf-8",
    )
    problem = load_problem(ABSOLUTE / "hamiltonian.json")
    policy = load_policy(candidate)
    _formula, rotations, metrics = _product_formula_schedule(problem, policy)

    assert len(rotations) == 1
    assert metrics["n_steps"] == 3
    assert metrics["formula_name"] == "custom_stationary_product_formula"
    assert metrics["formula_order"] is None
    assert metrics["term_order_strategy"] == "custom"


def test_product_formula_can_import_math(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        """import math

class ProductFormula:
    def __init__(self, problem):
        self.epsilon = math.exp(-14.0)

    def repetitions(self):
        return 1

    def build_step(self, terms, step_time):
        first = terms[0]
        angle = math.copysign(abs(float(first["coefficient"]) * step_time), step_time)
        return [(first, angle)]

    def synthesis_epsilon(self, term, angle):
        return self.epsilon * math.exp(-abs(angle))
""",
        encoding="utf-8",
    )

    problem = load_problem(ABSOLUTE / "hamiltonian.json")
    policy = load_policy(candidate)
    formula, rotations, metrics = _product_formula_schedule(problem, policy)

    assert formula.epsilon == pytest.approx(math.exp(-14.0))
    assert len(rotations) == 1
    assert metrics["formula_order"] is None
    assert formula.synthesis_epsilon(*rotations[0]) == pytest.approx(
        math.exp(-14.0 - abs(rotations[0][1]))
    )


@pytest.mark.parametrize(
    "source",
    [
        "import os\nclass ProductFormula: pass\n",
        "from math import exp\nclass ProductFormula: pass\n",
        """class ProductFormula:
    def __init__(self, problem):
        self.problem = problem
    def repetitions(self):
        return 1
    def build_step(self, terms, step_time):
        open("/tmp/escape", "w")
        return []
    def synthesis_epsilon(self, term, angle):
        return 1.0e-6
""",
    ],
)
def test_product_formula_ast_contract_rejects_external_capabilities(
    source: str,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_policy_source(candidate)


def test_lowest_hf_accessible_peak_is_not_the_largest_probability_peak() -> None:
    selected = _select_lowest_hf_accessible_peak(
        [0.2, -0.4, -0.8],
        [0.80, 0.19, 1.0e-16],
        n_steps=1,
        target_time=1.0,
        identity_coefficient=0.0,
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
