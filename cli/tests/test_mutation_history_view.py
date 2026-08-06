from __future__ import annotations

import json
from pathlib import Path

from hsopt.view.mutation_history import build_mutation_history, write_mutation_history


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_mutation_history_uses_lane_decisions_and_marks_current_best(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    lane = campaign / "lanes" / "abs_energy" / "h2" / "run_01"
    _write_json(
        campaign / "campaign.json",
        {
            "iterations": 3,
            "lanes": [
                {
                    "family": "abs_energy",
                    "instance": "h2",
                    "replicate": "run_01",
                    "lane_dir": "lanes/abs_energy/h2/run_01",
                }
            ],
        },
    )
    _write_json(lane / "manifest.json", {"iterations_target": 3})
    _write_json(
        lane / "state.json",
        {"iterations_completed": 2, "accepted_mutations": 1, "pending_iteration": 3},
    )
    _write_json(lane / "baseline_result.json", {"score": 100, "t_count": 80})
    _write_json(lane / "incumbent_result.json", {"score": 90, "t_count": 70})
    (lane / "initial_pf.py").write_text("BASELINE = True\n", encoding="utf-8")
    first = lane / "iterations" / "0001"
    second = lane / "iterations" / "0002"
    third = lane / "iterations" / "0003"
    _write_json(first / "decision.json", {"accepted": False, "candidate_score": 110})
    (first / "candidate.py").write_text("POLICY = 'rejected'\n", encoding="utf-8")
    _write_json(second / "decision.json", {"accepted": True, "candidate_score": 90})
    (second / "candidate.py").write_text("POLICY = 'best'\n", encoding="utf-8")
    (third / "candidate.py").parent.mkdir(parents=True, exist_ok=True)
    (third / "candidate.py").write_text("POLICY = 'awaiting evaluation'\n", encoding="utf-8")

    model = build_mutation_history(campaign)

    assert model["summary"] == {
        "lanes": 1,
        "iterations_target": 3,
        "proposals_archived": 3,
        "accepted": 1,
        "rejected": 1,
        "pending": 1,
    }
    viewed_lane = model["lanes"][0]
    assert [cell["status"] for cell in viewed_lane["iterations"]] == [
        "rejected",
        "accepted",
        "pending",
    ]
    assert viewed_lane["current_best_iteration"] == 2
    assert viewed_lane["iterations"][1]["is_current_best"] is True
    assert viewed_lane["iterations"][1]["candidate_source"] == "POLICY = 'best'\n"

    output = tmp_path / "viewer"
    write_mutation_history(campaign, output)
    assert (output / "mutation-history.json").is_file()
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "pale" not in page
    assert "Mutation history" in page
    assert "POLICY = 'best'" in page


def test_baseline_is_current_best_before_first_acceptance(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    lane = campaign / "lane"
    _write_json(
        campaign / "campaign.json",
        {"iterations": 1, "lanes": [{"lane_dir": "lane", "instance": "x"}]},
    )
    _write_json(lane / "manifest.json", {"iterations_target": 1})
    _write_json(lane / "state.json", {"iterations_completed": 0})
    (lane / "initial_pf.py").write_text("ORDER = 2\n", encoding="utf-8")

    viewed_lane = build_mutation_history(campaign)["lanes"][0]

    assert viewed_lane["baseline_is_current_best"] is True
    assert viewed_lane["current_best_iteration"] == 0
