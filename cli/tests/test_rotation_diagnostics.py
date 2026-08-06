from __future__ import annotations

import math
from pathlib import Path

from hsopt.evaluation.rotation_diagnostics import comparison_histograms, rotation_samples


REPO = Path(__file__).resolve().parents[2]
ABSOLUTE = REPO / "cli" / "tests" / "fixtures" / "abs_energy_h2"
DIFFERENCE = REPO / "cli" / "tests" / "fixtures" / "diff_energy_h2_bond_stretch"


def test_absolute_rotation_samples_match_the_evaluator_schedule() -> None:
    samples = rotation_samples(
        ABSOLUTE / "initial_pf.py",
        ABSOLUTE / "hamiltonian.json",
    )

    assert len(samples.angles) == 27
    assert len(samples.tolerances) == 27
    assert len(samples.weights) == 27
    assert samples.scheduled_rotations == 405
    assert samples.components == (
        {
            "label": "HamLib H2 linear R=0.7 STO-6G Jordan-Wigner",
            "n_qubits": 4,
            "n_steps": 15,
            "rotations_per_step": 27,
            "weighted_scheduled_rotations": 405.0,
        },
    )
    assert all(angle > 0.0 for angle in samples.angles)
    assert all(0.0 < epsilon < 1.0 for epsilon in samples.tolerances)


def test_reaction_rotation_samples_include_every_species() -> None:
    samples = rotation_samples(
        DIFFERENCE / "initial_pf.py",
        DIFFERENCE / "reaction.json",
    )

    assert {row["label"] for row in samples.components} == {"h2_r0_7", "h2_r0_8"}
    assert samples.scheduled_rotations == sum(
        row["weighted_scheduled_rotations"] for row in samples.components
    )
    assert samples.scheduled_rotations > 0.0


def test_comparison_histograms_are_common_normalized_log_grids() -> None:
    samples = rotation_samples(
        ABSOLUTE / "initial_pf.py",
        ABSOLUTE / "hamiltonian.json",
    )

    comparison = comparison_histograms(samples, samples, bins=20)

    assert comparison["qpe_repetitions_included"] is False
    assert comparison["initial"]["weighted_scheduled_rotations"] == 405
    for name in ("angles", "tolerances"):
        histogram = comparison[name]
        assert len(histogram["log10_edges"]) == 21
        assert len(histogram["initial_density"]) == 20
        assert histogram["initial_density"] == histogram["current_density"]
        assert math.isclose(max(histogram["initial_density"]), 1.0)
