from __future__ import annotations

import copy
import json

import pytest

from hsopt.sandbox import openshell
from hsopt.sandbox.briefing import compact_feedback, compact_history, mutation_prompt


def large_feedback():
    return {
        "status": "error",
        "score": None,
        "case_count": 115,
        "failed_case_ids": [f"case-{i:03d}" for i in range(115)],
        "case_metrics": [
            {
                "case_id": f"case-{i:03d}",
                "passes_chemical_accuracy": False,
                "error_hartree": 0.002 + i * 0.00001,
                "relative_t95": 0.5,
                "evaluation_error": "uncertified numerical result" if i == 0 else None,
            }
            for i in range(115)
        ],
    }


def test_brief_feedback_preserves_failure_counts_and_prioritizes_uncertified_cases():
    feedback = large_feedback()
    original = copy.deepcopy(feedback)
    brief = compact_feedback(feedback)
    assert feedback == original
    assert brief["case_count"] == brief["failed_case_count"] == 115
    assert brief["failed_case_ids_omitted"] == 110
    assert brief["attention_cases"][0]["evaluation_error"] == "uncertified numerical result"
    assert brief["attention_cases"][1]["case_id"] == "case-114"
    assert len(brief["attention_cases"]) == 5
    assert "case_metrics" not in brief


def test_shared_prompt_is_bounded_without_losing_failure_feedback_or_contract():
    context = {"shared_protocol": {"energy_target": "ground_branch"}}
    prompt = mutation_prompt(large_feedback(), context=context)
    assert len(prompt) < 6500
    assert "uncertified numerical result" in prompt
    assert '"failed_case_count": 115' in prompt
    assert "agent_feedback.json" in prompt and "trial_history.json" in prompt
    assert "ALL suite cases" in prompt and "EVERY case must pass" in prompt
    assert "E_compiled,0 - E_0" in prompt
    assert "not an exhaustive list" in prompt and "mixed-precision" in prompt
    assert "hsopt_pysr" not in prompt
    hybrid = mutation_prompt(context={**context, "pysr": {"tool": "hsopt_pysr"}})
    assert "hsopt_pysr" in hybrid and "All probes consume" in hybrid


def test_history_keeps_recent_decisions_and_marks_shortened_diffs():
    history = {
        "recent_proposals": [
            dict(large_feedback(), iteration=i, accepted=False, change="previous-diff\n" * 1000)
            for i in range(8)
        ],
        "recent_failures": [{"error": "duplicate policy"}],
    }
    original = copy.deepcopy(history)
    brief = compact_history(history)
    assert history == original
    assert len(brief["recent_proposals"]) == 8
    assert all(row["change_truncated"] for row in brief["recent_proposals"])
    assert brief["recent_failures"] == history["recent_failures"]
    assert brief["details_file"] == "trial_history.json"
    assert len(json.dumps(brief)) < len(json.dumps(history)) / 5


def test_pysr_requires_explicit_enable_even_when_endpoint_is_inherited(monkeypatch):
    monkeypatch.delenv("HSOPT_ENABLE_PYSR", raising=False)
    monkeypatch.setenv("HSOPT_PYSR_URL", "http://runtime.example:8787")
    monkeypatch.setenv("HSOPT_PYSR_TOKEN", "private-token")
    assert openshell._pysr_settings() is None
    assert not openshell.opencode_config().get("plugins")
    with openshell.pysr_capability(True):
        config = openshell.opencode_config()
        assert config["plugins"][-1]["package"] == "/sandbox/hsopt-pysr"
        with openshell.pysr_capability(False):
            assert openshell._pysr_settings() is None
        assert openshell._pysr_settings()
    assert openshell._pysr_settings() is None


def test_enabled_pysr_fails_clearly_without_service_config(monkeypatch):
    monkeypatch.delenv("HSOPT_PYSR_URL", raising=False)
    with openshell.pysr_capability(True), pytest.raises(ValueError, match="HSOPT_PYSR_URL"):
        openshell.opencode_config()
    assert openshell._pysr_settings() is None


@pytest.mark.parametrize("flag,enabled", [("--pysr", True), ("--no-pysr", False)])
def test_mutation_cli_exposes_both_pysr_choices(flag, enabled):
    args = openshell._parser().parse_args(
        [
            "mutate",
            "--policy-file",
            "candidate.py",
            "--feedback-file",
            "feedback.json",
            "--run-dir",
            "round",
            "--kind",
            "protocol",
            "--source",
            "suite.json",
            flag,
        ]
    )
    assert args.pysr is enabled
