"""Independent Hamiltonian, circuit and splitting oracles for explicit ELOTEs."""

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from hsopt.campaign import elote as prep
from hsopt.evaluation import clifford, elote
from hsopt.evaluation import compiled_unitary as ev
from hsopt.evaluation.gpu_circuit import gate_parameters

H2 = Path(__file__).parent / "fixtures/diff_energy_h2_bond_stretch/hamiltonians/h2_r0_7.json"


def pauli(label):
    value = np.ones((1, 1), dtype=complex)
    for axis in label:
        value = np.kron(value, ev._PAULIS[axis])
    return value


def frame_matrix(frame, n):
    result = np.eye(1 << n, dtype=complex)
    for name, *qubits in frame:
        if name == "CX":
            matrix = np.zeros_like(result)
            for i in range(1 << n):
                j = i ^ ((1 << (n - 1 - qubits[1])) if i & (1 << (n - 1 - qubits[0])) else 0)
                matrix[j, i] = 1
        else:
            matrix = np.ones((1, 1), dtype=complex)
            for q in range(n):
                matrix = np.kron(matrix, clifford.MATRICES[name] if q == qubits[0] else np.eye(2))
        result = matrix @ result
    return result


@pytest.mark.parametrize(
    "frame", [[("H", 0), ("S", 1), ("CX", 0, 1)], [("SDG", 0), ("CX", 1, 0), ("H", 1)]]
)
def test_clifford_frame_is_an_exact_signed_pauli_mapping(frame):
    c = frame_matrix(frame, 2)
    for axes in itertools.product("IXYZ", repeat=2):
        label = "".join(axes)
        sign, transformed = clifford.conjugate(label, frame)
        np.testing.assert_allclose(
            c @ pauli(label) @ c.conj().T, sign * pauli(transformed), atol=1e-14
        )
        reverse_sign, original = clifford.conjugate(transformed, clifford.inverse(frame))
        assert (sign * reverse_sign, original) == (1, label)


def test_literal_synthesis_errors_and_global_phase_survive_frame_conjugation():
    frame = [("H", 0), ("S", 1), ("CX", 0, 1)]
    c = frame_matrix(frame, 2)
    for label in ("XY", "ZZ", "IX"):
        native = ev._gate_sequence_matrix("WHTSTXHT")
        unframed, _ = ev._apply_compiled_rotation(
            np.eye(4), np.empty((4, 4), complex), label=label, compiled_rz=native, xp=np
        )
        framed, _ = ev._apply_compiled_rotation(
            np.eye(4),
            np.empty((4, 4), complex),
            label=label,
            compiled_rz=native,
            xp=np,
            frame=frame,
        )
        np.testing.assert_allclose(framed, c.conj().T @ unframed @ c, atol=1e-14)
        args = gate_parameters(label, native, frame)
        coefficients = np.array(args[6:]).reshape(4, 2) @ [1, 1j]
        actual = coefficients[0] * np.eye(4, dtype=complex)
        for index in range(3):
            x, z = (int(v) for v in args[2 * index : 2 * index + 2])
            for row in range(4):
                src = row ^ x
                actual[row, src] += coefficients[index + 1] * (-1) ** ((src & z).bit_count())
        np.testing.assert_allclose(actual, framed, atol=1e-14)


def test_group_strang_splitting_matches_exponentials_and_has_cubic_local_error(tmp_path):
    problem = {
        "name": "test",
        "n_qubits": 2,
        "identity_coefficient": 0.0,
        "target_time": 0.1,
        "terms": [
            elote.term_fields("XI", 0.7),
            elote.term_fields("IX", 0.2),
            elote.term_fields("ZZ", 0.4),
        ],
    }
    policy = tmp_path / "candidate.py"
    policy.write_text("""class ProductFormula:
    def __init__(self, problem):
        pass
    def repetitions(self):
        return 1
    def build_step(self, terms, step_time):
        return {"groups": [[(terms[0], .7*step_time), (terms[1], .2*step_time)],
                           [(terms[2], .4*step_time)]],
                "stages": [(0, .5), (1, 1), (0, .5)], "merge_adjacent": True}
    def synthesis_epsilon(self, term, angle):
        return 1e-8
""")
    h0, h1 = 0.7 * pauli("XI") + 0.2 * pauli("IX"), 0.4 * pauli("ZZ")
    errors = []
    for time in (0.1, 0.05):
        problem["target_time"] = time
        u, metrics = ev._build_exact_product_formula_step(
            problem, ev.load_policy(policy), backend_name="numpy", xp=np
        )
        expected = expm(0.5j * time * h0) @ expm(1j * time * h1) @ expm(0.5j * time * h0)
        np.testing.assert_allclose(u, expected, atol=1e-14)
        errors.append(np.linalg.norm(u - expm(1j * time * (h0 + h1))))
    assert errors[0] / errors[1] == pytest.approx(8, rel=0.01)
    assert metrics["elote"]["splitting"] == [[0, 0.5], [1, 1.0], [0, 0.5]]


def test_noncommuting_group_or_unknown_label_is_rejected():
    terms = {"X": elote.term_fields("X", 1), "Z": elote.term_fields("Z", 0.5)}
    with pytest.raises(ValueError, match="must commute"):
        elote.schedule({"groups": [[("X", 0.1), ("Z", 0.2)]], "stages": [(0, 1)]}, terms, limit=32)
    with pytest.raises(ValueError, match="unknown Pauli"):
        elote.schedule([("Y", 0.1)], terms, limit=32)
    with pytest.raises(ValueError, match="unknown group"):
        elote.schedule({"groups": [[("X", 0.1)]], "stages": [(1, 1)]}, terms, limit=32)


def test_fermionic_blocks_reconstruct_trusted_h_and_nested_initial_step(tmp_path):
    pytest.importorskip("openfermion")
    problem = ev.load_problem(H2)
    before = copy.deepcopy(problem)
    decomposition = elote.fermionic_blocks(problem)
    h = sum(t["coefficient"] * pauli(t["label"]) for t in problem["terms"])
    reconstructed = sum(
        t["coefficient"] * pauli(t["label"]) for b in decomposition["blocks"] for t in b["terms"]
    )
    np.testing.assert_allclose(h, reconstructed, atol=1e-13)
    assert problem == before
    policy = prep.bound(problem, 1, 1e-8)
    _, _, metrics = ev._product_formula_schedule(problem, policy)
    for group in metrics["elote"]["groups"]:
        assert all(elote.commute(a["label"], b["label"]) for a in group for b in group)
    u, _ = ev._build_exact_product_formula_step(problem, policy, backend_name="numpy", xp=np)
    oracle = np.eye(len(h), dtype=complex)
    for index, weight in metrics["elote"]["splitting"]:
        oracle = (
            expm(
                1j
                * weight
                * sum(t["angle"] * pauli(t["label"]) for t in metrics["elote"]["groups"][index])
            )
            @ oracle
        )
    np.testing.assert_allclose(u, oracle, atol=1e-13)
    policy.ProductFormula.basis_change = lambda self: [("H", 0), ("CX", 0, 2)]
    framed, _ = ev._build_exact_product_formula_step(problem, policy, backend_name="numpy", xp=np)
    np.testing.assert_allclose(u, framed, atol=1e-13)


def test_clifford_cancellation_keeps_the_exact_t_count():
    rotation = (elote.term_fields("ZZ", 1.0), 0.1)
    result = clifford.circuit_metrics([rotation, rotation], ["T", "T"])
    assert result["gate_counts"] == {"CX": 2, "T": 2}
    assert result["cancelled_clifford_gates"] == 2
    assert result["one_step_t_depth"] == 2


@pytest.mark.parametrize("baseline", ["tight", "loose"])
def test_new_calibration_isolated_and_shared_profile_uses_nested_template(
    tmp_path, monkeypatch, baseline
):
    from hsopt.campaign import protocol

    pytest.importorskip("openfermion")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(ev.evaluate_exact_reference(H2, backend="numpy")))
    # The production path is tested on the L4. This unit test isolates sealing,
    # profile selection and handoff from the native compiler/GPU dependency.
    monkeypatch.setattr(ev, "evaluate_molecule", lambda *a, **k: {"passes_chemical_accuracy": True})
    record = prep.prepare(H2, reference, tmp_path / "baseline", baseline=baseline, backend="numpy")
    prep.verify(record, H2, baseline)
    with pytest.raises(ValueError, match="another problem/baseline"):
        prep.verify(record, H2, "loose" if baseline == "tight" else "tight")
    changed = copy.deepcopy(record)
    changed["instantiated_policy"]["n_steps"] += 1
    with pytest.raises(ValueError, match="stale"):
        prep.verify(changed, H2, baseline)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "h2",
                        "source": str(H2.resolve()),
                        "reference": str(reference),
                        "calibration": str(tmp_path / "baseline/calibration.json"),
                    }
                ]
            }
        )
    )
    run = tmp_path / "run"
    protocol.prepare(
        selection,
        run,
        family="abs_protocol",
        baseline=baseline,
        iterations=2,
        replicate="seed-01",
        energy_target="ground_branch",
        policy_profile="fermionic_nested",
    )
    assert (run / "initial_pf.py").read_bytes() == prep.TEMPLATE.read_bytes()
    assert json.loads((run / "suite.json").read_text())["policy_profile"] == "fermionic_nested"
    bound = run / "iterations/0000/bound_policies/h2.py"
    recipe = prep.describe(H2, bound)
    assert recipe["grouping"] == "explicit_commuting_blocks"
    assert recipe["formula_order"] == 2
    assert len(recipe["synthesis_tolerances"]) == len(recipe["rotation_schedule"])
    assert recipe["basis_change"] == []


def test_failed_compiled_baseline_is_not_sealed(tmp_path, monkeypatch):
    pytest.importorskip("openfermion")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(ev.evaluate_exact_reference(H2, backend="numpy")))
    monkeypatch.setattr(
        ev, "evaluate_molecule", lambda *a, **k: {"passes_chemical_accuracy": False}
    )
    with pytest.raises(RuntimeError, match="iteration 0 failed"):
        prep.prepare(H2, reference, tmp_path / "baseline", baseline="tight", backend="numpy")
    assert not (tmp_path / "baseline/calibration.json").exists()


def test_verifier_refresh_keeps_calibrated_beta_policy_and_reference(tmp_path, monkeypatch):
    pytest.importorskip("openfermion")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(ev.evaluate_exact_reference(H2, backend="numpy")))
    from hsopt.evaluation import ground_branch
    from hsopt.campaign import calibration

    result = {
        "passes_chemical_accuracy": True,
        "evaluation_contract_version": ground_branch.CONTRACT_VERSION,
    }
    monkeypatch.setattr(ev, "evaluate_molecule", lambda *a, **k: dict(result))
    folder = tmp_path / "baseline"
    record = prep.prepare(H2, reference, folder, baseline="tight", backend="numpy")
    original_policy = (folder / "initial_pf.py").read_bytes()
    original_parameters = copy.deepcopy(record["instantiated_policy"])
    record.pop("calibration_sha256")
    record["validation_contract_version"] = "older-verifier"
    record["calibration_sha256"] = elote.digest(record)
    (folder / "calibration.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="older verifier"):
        prep.verify(record, H2, "tight")

    def forbidden(*args, **kwargs):
        pytest.fail("refresh recalibrated beta")

    monkeypatch.setattr(calibration, "_empirical_reference_beta", forbidden)
    updated = prep.refresh_validation(H2, reference, folder)
    prep.verify(updated, H2, "tight")
    assert updated["instantiated_policy"] == original_parameters
    assert (folder / "initial_pf.py").read_bytes() == original_policy
    assert updated["validation_history"][0]["calibration_parameters_reused"]
    reference.write_text(reference.read_text() + "\n")
    with pytest.raises(ValueError, match="changed policy or exact reference"):
        prep.refresh_validation(H2, reference, folder)


def test_host_keeps_its_decomposition_when_a_constructor_mutates_its_copy(tmp_path):
    pytest.importorskip("openfermion")
    policy = tmp_path / "candidate.py"
    policy.write_text("""class ProductFormula:
    hamiltonian_representation = "fermionic_jw"
    def __init__(self, problem):
        problem["fermionic_decomposition"]["blocks"][0]["terms"].append(
            {"label": "XIII", "coefficient": 100})
    def repetitions(self):
        return 1
    def build_step(self, terms, step_time):
        return [("XIII", 1)]
    def synthesis_epsilon(self, term, angle):
        return 1e-8
""")
    with pytest.raises(ValueError, match="unknown Pauli"):
        ev._product_formula_schedule(ev.load_problem(H2), ev.load_policy(policy))
