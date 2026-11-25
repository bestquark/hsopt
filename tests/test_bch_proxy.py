"""Check proxy algebra and integration independently on small CPU problems."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from hsopt.evaluation import evaluate_molecule
from hsopt.evaluation.compiled_unitary import _fixed_electron_hamiltonian, load_problem
from hsopt.evaluation.experiment import main
from hsopt.evaluation.proxies import bch

ROOT = Path(__file__).resolve().parents[1]
ABS = ROOT / "tests/fixtures/abs_energy_h2"
DIFF = ROOT / "tests/fixtures/diff_energy_h2_bond_stretch"
PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def matrix(label):
    result = np.ones((1, 1))
    for char in label:
        result = np.kron(result, PAULIS[char])
    return result


def test_pauli_actions_and_bracket_signs_against_independent_matrices():
    labels = ["".join(chars) for chars in itertools.product("IXYZ", repeat=2)]
    basis = list(range(4))
    for left in labels:
        got = bch.projected_matrix({bch.encode(left): 1.0}, basis)
        np.testing.assert_allclose(got, matrix(left))
        for right in labels:
            got = bch.projected_matrix(
                bch.lie({bch.encode(left): 1.0}, {bch.encode(right): 1.0}), basis
            )
            want = 1j * (matrix(left) @ matrix(right) - matrix(right) @ matrix(left))
            np.testing.assert_allclose(got, want, atol=1e-14)


def test_bch_sign_and_step_scaling_against_exact_ideal_h2(tmp_path):
    problem = load_problem(ABS / "hamiltonian.json")
    indices, sector = _fixed_electron_hamiltonian(problem)
    energies, vectors = np.linalg.eigh(sector)
    ground = np.zeros(1 << problem["n_qubits"], complex)
    ground[indices] = vectors[:, 0]
    errors = []
    biases = []
    for steps in [8, 16]:
        policy = tmp_path / f"pf-{steps}.py"
        policy.write_text((ABS / "initial_pf.py").read_text().replace("(15,", f"({steps},"))
        result = evaluate_molecule(ABS / "hamiltonian.json", policy, method="bch")
        dt = problem["target_time"] / steps
        # Independently construct the Strang step, without the proxy's Pauli algebra.
        terms = sorted(problem["terms"], key=lambda term: term["label"])
        gates = [(t, 0.5 * t["coefficient"] * dt) for t in terms[:-1]]
        gates += [(terms[-1], terms[-1]["coefficient"] * dt)] + gates[::-1]
        unitary = np.eye(len(ground), dtype=complex)
        for term, angle in gates:
            unitary = expm(1j * angle * matrix(term["label"])) @ unitary
        values, eigenvectors = np.linalg.eig(unitary)
        branch = np.argmax(np.abs(ground.conj() @ eigenvectors) ** 2)
        exact_bias = np.angle(values[branch]) / dt + problem["identity_coefficient"] - energies[0]
        assert result["reference_energy_hartree"] == pytest.approx(energies[0], abs=1e-12)
        assert result["reference_residual_hartree"] < 1e-10
        assert result["proxy_bias_hartree"] == pytest.approx(exact_bias, rel=0.01)
        assert result["score"] is None
        assert result["certificate"] is False
        assert "t_count" not in result
        assert "passes_chemical_accuracy" not in result
        errors.append(abs(exact_bias - result["proxy_bias_hartree"]))
        biases.append(result["proxy_bias_hartree"])
    assert biases[0] / biases[1] == pytest.approx(4.0)
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.1)


def test_proxy_never_compiles_or_diagonalizes_a_full_unitary(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("proxy invoked exact compilation/eigenphase evaluation")

    monkeypatch.setattr("hsopt.evaluation.compiled_unitary._compile_rz", forbidden)
    monkeypatch.setattr(np.linalg, "eig", forbidden)
    result = evaluate_molecule(
        ABS / "hamiltonian.json", ABS / "initial_pf.py", method="bch", reference_size=1
    )
    assert result["reference_determinants"] == 1
    assert result["reference_residual_hartree"] > 0
    assert result["compiled"] is False


def test_clifford_frame_preserves_the_physical_bch_prediction(tmp_path):
    source = (ABS / "initial_pf.py").read_text()
    source = source.replace('term["label"]', 'term["original_label"]')
    source = source.replace(
        "    def build_step(",
        "    def basis_change(self):\n"
        '        return [("H", 0), ("S", 2), ("CX", 1, 3)]\n\n'
        "    def build_step(",
    )
    candidate = tmp_path / "framed.py"
    candidate.write_text(source)
    reference = bch.evaluate_molecule(ABS / "hamiltonian.json", ABS / "initial_pf.py")
    framed = bch.evaluate_molecule(ABS / "hamiltonian.json", candidate)
    assert framed["proxy_bias_hartree"] == pytest.approx(reference["proxy_bias_hartree"], abs=1e-14)


def test_bch_accepts_a_symmetric_commuting_group_expansion(tmp_path):
    from hsopt.campaign.elote import TEMPLATE
    from hsopt.campaign.protocol import bind_policy

    pytest.importorskip("openfermion")
    problem = load_problem(DIFF / "hamiltonians/h2_r0_7.json")
    candidate = tmp_path / "nested.py"
    bind_policy(
        TEMPLATE,
        {
            problem["name"]: {
                "n_steps": 8,
                "uniform_epsilon": 1e-8,
                "reference_trotter_number_beta": 1.0,
            }
        },
        candidate,
    )
    result = bch.evaluate_molecule(DIFF / "hamiltonians/h2_r0_7.json", candidate)
    assert result["elote"]["symmetric_splitting"]
    assert result["elote"]["first_order_generator_matches_hamiltonian"]
    assert result["score"] is None


@pytest.mark.parametrize("mutation,reason", [("asymmetric", "symmetric"), ("scaled", "consistent")])
def test_unsupported_schedules_are_rejected(tmp_path, mutation, reason):
    source = (ABS / "initial_pf.py").read_text()
    if mutation == "asymmetric":
        source = source.replace(
            "return half + center + list(reversed(half))", "return half + center"
        )
    else:
        source = source.replace("0.5 * float", "0.25 * float")
    policy = tmp_path / "candidate.py"
    policy.write_text(source)
    with pytest.raises(ValueError, match=reason):
        evaluate_molecule(ABS / "hamiltonian.json", policy, method="bch")


@pytest.mark.parametrize("cap", [0, 513])
def test_invalid_reference_caps_are_rejected(cap):
    with pytest.raises(ValueError, match="reference_size"):
        bch.evaluate_molecule(ABS / "hamiltonian.json", ABS / "initial_pf.py", reference_size=cap)


def test_reaction_contracts_aggregate_signed_proxy_bias():
    results = {
        contract: bch.evaluate_difference(
            DIFF / "reaction.json", DIFF / "initial_pf.py", error_contract=contract
        )
        for contract in ["individual_errors", "error_cancellation"]
    }
    result = results["individual_errors"]
    contributions = [
        -result["species"]["h2_r0_7"]["proxy_bias_hartree"],
        result["species"]["h2_r0_8"]["proxy_bias_hartree"],
    ]
    assert result["proxy_bias_hartree"] == pytest.approx(sum(contributions))
    assert result["contract_proxy_bias_hartree"] == pytest.approx(sum(map(abs, contributions)))
    assert results["error_cancellation"]["contract_proxy_bias_hartree"] == pytest.approx(
        abs(sum(contributions))
    )
    assert all(r["score"] is None for r in results.values())


def test_cli_proxy_output_and_exact_reference_requirement(tmp_path):
    output = tmp_path / "proxy.json"
    arguments = [
        "--problem",
        str(ABS / "hamiltonian.json"),
        "--policy-file",
        str(ABS / "initial_pf.py"),
    ]
    assert main([*arguments, "--method", "bch", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["evaluation_method"] == "bch"
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    with pytest.raises(ValueError, match="not --exact-reference"):
        evaluate_molecule(
            ABS / "hamiltonian.json", ABS / "initial_pf.py", method="bch", reference_path=output
        )


def test_exact_is_default_and_retains_its_result(monkeypatch):
    sentinel = {"score": 123, "evaluation_contract_version": "exact"}
    monkeypatch.setattr(
        "hsopt.evaluation.compiled_unitary.evaluate_molecule", lambda *a, **k: sentinel
    )
    assert evaluate_molecule(ABS / "hamiltonian.json", ABS / "initial_pf.py") is sentinel


@pytest.mark.parametrize("family", ["abs_energy", "diff_energy"])
def test_public_evaluator_entry_point_accepts_bch(tmp_path, family):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    if family == "abs_energy":
        name = json.loads((ABS / "hamiltonian.json").read_text())["name"]
        (experiment / "instance.json").write_text(json.dumps({"instance": name[7:], "n_qubits": 4}))
        arguments = [
            "--problem",
            str(ABS / "hamiltonian.json"),
            "--policy-file",
            str(ABS / "initial_pf.py"),
        ]
    else:
        reaction = json.loads((DIFF / "reaction.json").read_text())
        (experiment / "definition.json").write_text(
            json.dumps(
                {"instance": "test", "terms": reaction["terms"], "contracts": ["individual_errors"]}
            )
        )
        arguments = [
            "--reaction",
            str(DIFF / "reaction.json"),
            "--policy-file",
            str(DIFF / "initial_pf.py"),
            "--error-contract",
            "individual_errors",
        ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hsopt.evaluation.experiment import main; raise SystemExit(main())",
            "--method",
            "bch",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["score"] is None
