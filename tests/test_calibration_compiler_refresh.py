from pathlib import Path
from hsopt.campaign.baseline import baseline_module


def test_legacy_compiler_label_refresh_preserves_all_numerical_policy_inputs():
    g = baseline_module()
    path = Path(__file__).parent / "fixtures/abs_energy_h2/hamiltonian.json"
    current = g.calibrate_problem(
        path, backend="numpy", mode="hamiltonian_bound", include_exact_reference=False
    )
    old = dict(current)
    old["instantiated_policy"] = dict(current["instantiated_policy"])
    old["instantiated_policy"]["rotation_synthesis"] = (
        "clifford_plus_t_pygridsynth_at_iteration_zero"
    )
    old = g.seal_calibration(old)
    fresh = g.refresh_synthesis_provenance(old, problem_path=path)
    g.verify_calibration(fresh, problem_path=path)
    assert fresh["parent_calibration_sha256"] == old["calibration_sha256"]
    assert fresh["reference"] == old["reference"]
    assert fresh["instantiated_policy"] == current["instantiated_policy"]
    assert old["instantiated_policy"]["rotation_synthesis"].startswith(
        "clifford_plus_t_pygridsynth"
    )
