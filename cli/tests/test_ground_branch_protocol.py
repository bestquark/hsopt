"""Ground-input dynamics contract, degeneracy and immutable request isolation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hsopt.campaign import interface, protocol, queue
from hsopt.evaluation import compiled_unitary as ev, gpu_spectrum, ground_branch as g
from hsopt.evaluation import gpu_circuit
from hsopt.sandbox.openshell import mutation_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "abs_energy_h2"


def test_numerical_verifier_upgrade_isolates_ground_task_receipts(monkeypatch):
    args = ("absolute", FIXTURE / "hamiltonian.json")
    kwargs = {"stage": "candidate", "policy": FIXTURE / "initial_pf.py"}
    ground_before = queue._task_id(*args, **kwargs, energy_target="ground_branch")
    hf_before = queue._task_id(*args, **kwargs)
    monkeypatch.setattr(gpu_circuit, "CERTIFIER_VERSION", "another_numerical_verifier")
    assert queue._task_id(*args, **kwargs, energy_target="ground_branch") != ground_before
    assert queue._task_id(*args, **kwargs) == hf_before


def test_ground_input_selects_its_branch_not_the_lowest_hf_peak():
    phases, weights = [-0.7, -0.1, 0.2], [0.01, 0.98, 0.01]
    args = dict(n_steps=2, target_time=1.0, identity_coefficient=-1.0)
    peak = g.select_peak(phases, weights, **args)
    hf = ev._select_lowest_hf_accessible_peak(phases, weights, **args)
    assert peak["compiled_pf_energy_hartree"] == pytest.approx(-1.2)
    assert hf["compiled_pf_energy_hartree"] == pytest.approx(-2.4)
    assert peak["qpe_success_probability_from_ground_input"] == pytest.approx(0.98)


def test_ground_peak_is_basis_independent_in_a_degenerate_eigenspace():
    args = dict(n_steps=1, target_time=1.0, identity_coefficient=0.0)
    for weights in ([0.3, 0.3, 0.05], [0.6, 0.0, 0.05]):
        peak = g.select_peak([-0.2, -0.2, 0.0], weights, **args)
        assert peak["selected_eigen_indices"] == [0, 1]
        assert peak["qpe_success_probability_from_ground_input"] == pytest.approx(0.6)
    with pytest.raises(RuntimeError, match="dominant"):
        g.select_peak([-0.2, 0.0], [0.3, 0.4], **args)


def test_ground_scoring_reuses_cached_input_and_excludes_hf_bias(monkeypatch):
    source, policy = FIXTURE / "hamiltonian.json", FIXTURE / "initial_pf.py"
    ref = ev.evaluate_exact_reference(source, backend="numpy")
    ref["exact_hf_accessible_energy_hartree"] = 10.0  # deliberately irrelevant HF diagnostic
    build = ev._build_compiled_step
    monkeypatch.setattr(ev, "array_backend", lambda _: ("cupy-lanczos", np))
    monkeypatch.setattr(ev, "_fence", lambda _: None)
    monkeypatch.setattr(ev, "_build_compiled_step", lambda p, pf, **kw: build(p, pf, backend_name="numpy", xp=np))
    seen = []

    def known_spectrum(unitary, problem, metrics, *, iterative, target_state, reference_energy):
        # Tiny full spectrum is an independent test oracle, never a production fallback.
        seen.append(target_state)
        assert reference_energy == ref["exact_unitary_ground_energy_hartree"]
        values, vectors = np.linalg.eig(unitary)
        return values, vectors, {"method": "test_oracle", "spectrum_complete": True}

    monkeypatch.setattr(gpu_spectrum, "eigensolve", known_spectrum)
    monkeypatch.setattr(ev, "evaluate_exact_reference", lambda *a, **k: pytest.fail("recomputed H"))
    result = ev.evaluate_molecule(source, policy, backend="cupy-lanczos",
                                  reference_result=ref, energy_target="ground_branch")
    assert result["evaluation_contract_version"] == g.CONTRACT_VERSION
    assert result["reference_target"] == g.REFERENCE_TARGET
    assert result["passes_chemical_accuracy"]
    assert result["signed_energy_error_hartree"] == pytest.approx(
        result["compiled_pf_energy_hartree"] - ref["exact_unitary_ground_energy_hartree"]
    )
    assert result["qpe_success_probability_from_ground_input"] > 0.99
    assert "qpe_success_probability_from_hf" not in result
    assert "exact_hf_accessible_energy_hartree" not in result
    np.testing.assert_allclose(seen[0], np.array(ref["exact_ground_state"]) @ [1.0, 1j])


def test_ground_input_has_no_cpu_or_reference_regeneration_fallback():
    args = (FIXTURE / "hamiltonian.json", FIXTURE / "initial_pf.py")
    for kwargs in ({"backend": "numpy"}, {"backend": "cupy-lanczos"}):
        with pytest.raises(ValueError, match="cached reference and GPU Lanczos"):
            ev.evaluate_molecule(*args, energy_target="ground_branch", **kwargs)


def test_ground_reference_rejects_wrong_inputs_and_bad_normalization():
    source = FIXTURE / "hamiltonian.json"
    problem = ev.load_problem(source)
    ref = ev.evaluate_exact_reference(source, backend="numpy")
    energy, state = g.cached_reference(problem, ref)
    assert energy == ref["exact_unitary_ground_energy_hartree"]
    assert np.linalg.norm(state) == pytest.approx(1.0)
    wrong = copy.deepcopy(ref)
    wrong["exact_ground_state"][0][0] = 2.0
    with pytest.raises(ValueError, match="normalized"):
        g.cached_reference(problem, wrong)
    wrong = dict(ref, problem_sha256="wrong")
    with pytest.raises(ValueError, match="does not match"):
        g.cached_reference(problem, wrong)


def test_task_and_request_fingerprints_separate_ground_from_hf(tmp_path):
    source, policy = FIXTURE / "hamiltonian.json", FIXTURE / "initial_pf.py"
    reference = tmp_path / "ref.json"
    protocol.write(reference, ev.evaluate_exact_reference(source, backend="numpy"))
    calls = []
    for target in ("hf_accessible", "ground_branch"):
        item = queue.enqueue(tmp_path / "queue", kind="absolute", source=source, policy=policy,
                             reference=reference, energy_target=target)
        calls.append(interface.evaluation_request(Path(item["path"])))
    hf, ground = calls
    assert hf["task_id"] != ground["task_id"]
    assert hf["task"]["evaluation_contract_version"] == ev.EVALUATION_CONTRACT_VERSION
    assert ground["task"]["evaluation_contract_version"] == g.CONTRACT_VERSION
    assert ground["task"]["energy_target"] == "ground_branch"
    rejected = interface.run_evaluation(ground, backend="numpy")
    assert rejected["result"]["status"] == "error"
    assert rejected["result"]["evaluation_contract_version"] == g.CONTRACT_VERSION
    assert rejected["result"]["energy_target"] == "ground_branch"


def test_mutation_prompt_explains_ground_input_and_mixed_precision():
    prompt = mutation_prompt(context={"shared_protocol": {"energy_target": "ground_branch"}})
    assert "E_compiled,0 - E_0" in prompt
    assert "Ground-state preparation cost is excluded" in prompt
    assert "mixed-precision synthesis schemes" in prompt
    assert "fixed Hartree--Fock\ninput" not in prompt


def test_shared_prepare_preserves_beta_and_requires_ground_receipts(tmp_path):
    source = FIXTURE / "hamiltonian.json"
    reference, calibration = tmp_path / "reference.json", tmp_path / "calibration.json"
    ref = ev.evaluate_exact_reference(source, backend="numpy")
    # HF preparation is not a prerequisite for this explicitly ground-input suite.
    for key in list(ref):
        if key.startswith("exact_hf_") or key in {"reference_target", "hf_basis_index"}:
            ref.pop(key)
    protocol.write(reference, ref)
    protocol.write(calibration, protocol.calibration.calibrate_problem(
        source, backend="numpy", mode="empirical_exact"))
    selection = tmp_path / "selection.json"
    protocol.write(selection, {"cases": [{"id": "h2", "source": str(source),
                                         "reference": str(reference), "calibration": str(calibration)}]})
    run = tmp_path / "shared"
    protocol.prepare(selection, run, family="abs_protocol", baseline="tight", iterations=100,
                     replicate="seed-01", energy_target="ground_branch")
    suite = protocol.suite_at(run)
    manifest = protocol.read(run / "manifest.json")
    assert suite["energy_target"] == "ground_branch"
    assert manifest["evaluation_contract_version"] == g.CONTRACT_VERSION
    assert manifest["iterations_target"] == 100
    parameters = suite["cases"][0]["parameters"]
    expected, _ = protocol.calibration.load_baselines(source, baseline="tight",
                                                    calibration_path=calibration, calibration_root=None)
    assert parameters == expected
    row = protocol._pending(run)[1]["requests"][0]
    request = row["request"]
    wrong = {k: request[k] for k in ("schema", "task_id", "request_sha256", "evaluator_sha256")}
    wrong["result"] = {"task_id": request["task_id"],
                       "evaluation_contract_version": ev.EVALUATION_CONTRACT_VERSION}
    with pytest.raises(ValueError, match="evaluator contract"):
        protocol.record_receipt(run, "h2", wrong)


def test_reaction_ground_branch_uses_same_exact_input_for_signed_difference(tmp_path, monkeypatch):
    source = FIXTURE / "hamiltonian.json"
    reaction = tmp_path / "reaction.json"
    local_problem = tmp_path / "h2.json"
    local_problem.write_bytes(source.read_bytes())
    reaction.write_text(json.dumps({
        "name": "ground-input check",
        "terms": [{"species": "h2", "coefficient": 1.0, "problem": local_problem.name}],
    }))
    # Reference hashes are bound to the exact local Hamiltonian file bytes.
    reference = ev.evaluate_exact_reference(local_problem, backend="numpy")
    assembled = ev.assemble_exact_difference_reference(
        reaction, {"h2": reference}, backend="numpy", energy_target="ground_branch"
    )
    assert assembled["reference_target"] == g.REFERENCE_TARGET
    assert assembled["evaluation_contract_version"] == g.DIFFERENCE_CONTRACT_VERSION
    assert assembled["exact_unitary_difference_hartree"] == pytest.approx(
        reference["exact_unitary_ground_energy_hartree"]
    )
    stored = tmp_path / "reference.json"
    stored.write_text(json.dumps(assembled))
    protocol._validate_reference("difference", reaction, stored, "ground_branch")
    seen = []

    def evaluate(*args, **kwargs):
        seen.append(kwargs["energy_target"])
        return {
            "exact_reference_energy_hartree": assembled["exact_unitary_difference_hartree"],
            "compiled_pf_energy_hartree": assembled["exact_unitary_difference_hartree"] + 1e-5,
            "qpe_success_probability_from_ground_input": 0.9,
            "t_count": 100,
            "evaluation_wall_seconds": 0.1,
            "phase_encoding": {"physical_evolution_time_hartree_inverse": 0.1},
        }

    monkeypatch.setattr(ev, "evaluate_molecule", evaluate)
    result = ev.evaluate_difference(
        reaction, tmp_path / "policy.py", error_contract="error_cancellation",
        backend="cupy-lanczos", reference_path=stored, energy_target="ground_branch"
    )
    assert seen == ["ground_branch"]
    assert result["energy_target"] == "ground_branch"
    assert result["evaluation_contract_version"] == g.DIFFERENCE_CONTRACT_VERSION
    assert result["passes_chemical_accuracy"]
    assert result["absolute_difference_error_hartree"] == pytest.approx(1e-5)
    queued = queue.enqueue(
        tmp_path / "queue", kind="difference", source=reaction,
        policy=FIXTURE / "initial_pf.py", reference=stored,
        difference_error_contract="error_cancellation", energy_target="ground_branch"
    )
    request = interface.evaluation_request(Path(queued["path"]))
    assert request["task"]["energy_target"] == "ground_branch"
    assert request["task"]["evaluation_contract_version"] == g.DIFFERENCE_CONTRACT_VERSION
