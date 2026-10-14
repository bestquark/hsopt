import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("shared_figures", Path(__file__).parents[2] / "plots/plot_shared_protocol.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot():
    return {"case_count": 2, "seeds": [
        {"seed": f"seed-{s}", "trajectory": {
            "status": "ready", "iterations": [0, 1, 2] if s else [0, 1],
            "chemical_accuracy_hartree": .0015936014376405157,
            "molecules": [
                {"case_id": "a", "drop_percent": [0, 5*s, 50][:2 if s == 0 else 3],
                 "baseline_simulation_budget_hartree": .0005,
                 "energy_error_hartree": [.0001, -.0001*(s+1), .003][:2 if s == 0 else 3]},
                {"case_id": "b", "drop_percent": [0, 10*s, 90][:2 if s == 0 else 3],
                 "baseline_simulation_budget_hartree": .0006,
                 "energy_error_hartree": [-.0003, .0001*(s+1), .004][:2 if s == 0 else 3]},
            ],
        }} for s in range(5)]}


def test_five_seed_mean_and_sd_use_matched_rounds_not_future_values():
    data = module.aggregate(snapshot())
    assert data["common_round"] == 1
    assert data["iterations"] == [0, 1]
    assert data["molecule_mean_relative_t95"] == [[1, .9], [1, .8]]
    assert data["mean_relative_t95"] == pytest.approx([1, .85])
    assert data["std_relative_t95"] == pytest.approx([0, np.std([1, .925, .85, .775, .7], ddof=1)])
    assert data["molecule_std_relative_t95"][0][1] == pytest.approx(np.std([1, .95, .9, .85, .8], ddof=1))
    assert data["mean_energy_error_mha"] == pytest.approx([.2, .3])
    assert data["molecule_mean_energy_error_mha"][0] == pytest.approx([.1, .3])
    assert data["std_energy_error_mha"] == pytest.approx([0, np.std([.1, .2, .3, .4, .5], ddof=1)])
    assert data["chemical_accuracy_mha"] == pytest.approx(1.5936014376405157)
    assert data["molecule_baseline_simulation_budget_mha"] == [.5, .6]
    assert data["mean_baseline_simulation_budget_mha"] == pytest.approx(.55)
    altered = snapshot()
    altered["seeds"][4]["trajectory"]["molecules"][0]["drop_percent"][2] = 99
    assert module.aggregate(altered)["figure_key"] == data["figure_key"]


def test_incomplete_seed_or_changed_molecule_cohort_is_not_pooled():
    data = snapshot()
    data["seeds"][1]["trajectory"]["molecules"][0]["case_id"] = "other"
    with pytest.raises(ValueError, match="same molecules"):
        module.aggregate(data)
    data = snapshot()
    data["seeds"].pop()
    with pytest.raises(ValueError, match="Five distinct"):
        module.aggregate(data)


def test_mismatched_energy_budget_is_not_averaged_into_a_common_line():
    data = snapshot()
    data["seeds"][3]["trajectory"]["chemical_accuracy_hartree"] = .002
    with pytest.raises(ValueError, match="same chemical-accuracy"):
        module.aggregate(data)


def test_calibrated_qpe_budget_must_be_fixed_and_valid_across_seeds():
    data = snapshot()
    data["seeds"][3]["trajectory"]["molecules"][0]["baseline_simulation_budget_hartree"] = .0007
    with pytest.raises(ValueError, match="same per-molecule calibration budgets"):
        module.aggregate(data)
    data = snapshot()
    data["seeds"][0]["trajectory"]["molecules"][0]["baseline_simulation_budget_hartree"] = .002
    with pytest.raises(ValueError, match="Invalid baseline simulation budget"):
        module.aggregate(data)
