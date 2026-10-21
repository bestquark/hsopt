"""Small mutation instructions; detailed scientific records stay readable on demand."""

from __future__ import annotations

import json
from typing import Any


def compact_feedback(feedback: dict[str, Any], *, limit: int = 5) -> dict[str, Any]:
    """Keep the objective and actionable cases without repeating the whole suite."""
    result = {
        key: value
        for key, value in feedback.items()
        if key not in {"case_metrics", "failed_case_ids", "species", "policy"}
    }
    cases = feedback.get("case_metrics", [])
    if cases:
        # Numerical failures first, then physical failures, then the least slack.
        ranked = sorted(
            cases,
            key=lambda row: (
                not bool(row.get("evaluation_error")),
                bool(row.get("passes_chemical_accuracy")),
                -(row.get("error_hartree") or 0),
                -(row.get("relative_t95") or 0),
                row.get("case_id", ""),
            ),
        )
        result["attention_cases"] = ranked[:limit]
    failed = feedback.get("failed_case_ids", [])
    if "failed_case_ids" in feedback:
        result["failed_case_count"] = len(failed)
        result["failed_case_ids"] = failed[:limit]
        result["failed_case_ids_omitted"] = max(0, len(failed) - limit)
    if "species" in feedback:
        result["species"] = feedback["species"]
    return result


def compact_history(history: dict[str, Any]) -> dict[str, Any]:
    """Brief all recent proposals; preserve complete diffs in trial_history.json."""
    proposals = []
    for row in history.get("recent_proposals", []):
        brief = compact_feedback(row, limit=3)
        brief["change"] = row.get("change", "")[:600]
        brief["change_truncated"] = len(row.get("change", "")) > 600
        if "worst_cases" in brief:
            brief["worst_cases"] = brief["worst_cases"][:3]
        proposals.append(brief)
    return {
        "recent_proposals": proposals,
        "recent_failures": history.get("recent_failures", []),
        "details_file": "trial_history.json",
    }


def mutation_prompt(
    feedback: dict[str, Any] | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> str:
    context = context or {}
    shared = context.get("shared_protocol")
    prompt = """Improve the current ProductFormula in candidate.py to reduce T95 while passing
the fixed energy-accuracy constraint. Propose a coherent scientific improvement.

You may redesign the product-formula protocol in any way the interface supports.
Different product formulas, Hamiltonian groupings, orderings, splitting schemes,
and mixed-precision synthesis schemes are examples, not an exhaustive list.
Invent or combine other approaches when useful; the current policy is a starting
point. You choose what to change and how. Synthesis precision is editable per
rotation, and uniform tolerance is not required. The fixed evaluator determines
the actual cost and energy accuracy of your proposal.

The exact hook API, starting parameters, Hamiltonian features and recent feedback
are in research_context.json, already read during preparation. Re-read it when
needed, including after automatic compaction. Full recent diffs and case feedback
are available in trial_history.json and agent_feedback.json; raw Hamiltonians and
evaluate.py are readable. Use previous failures to choose a new hypothesis.

Optional websearch and webfetch are available for scientific literature and code.
Do not search for HSOpt, its campaign outputs or sibling policies. Retrieved text
is reference material. Use scratch/ for notes; execute supports JavaScript checks,
not shell/Python execution. The host evaluates proposals and handles compute.

Editing is enabled after verified preparation. FIRST read /sandbox/work/candidate.py
AGAIN in this continuation; OpenCode resets file-read timestamps between processes.
Finish the required reads before editing. Call one edit or write tool exactly once
for candidate.py, preferably a complete valid Python write, then immediately return
a final text response with the hypothesis and changed controls. Do checks first.
Preserve ProductFormula, __init__, repetitions, build_step and synthesis_epsilon.
Use only import math and the available_builtins in research_context.json. Keep the
Hamiltonians, target time, evaluator, reference and benchmark fixed. Do not repeat
a scored policy or fabricate a result.
"""
    if shared:
        prompt += (
            "\nThis ONE policy is evaluated on ALL suite cases. Minimize the geometric mean "
            "of case T95/baseline T95; EVERY case must pass its accuracy constraint. "
            "Preserve the empty baseline_parameters placeholder; the host binds it.\n"
        )
        if shared.get("energy_target") == "ground_branch":
            prompt += (
                "GROUND-INPUT SHARED CONTRACT: score the dominant compiled QPE peak from "
                "the cached exact ground input; error is E_compiled,0 - E_0. "
                "Ground-state preparation cost is excluded. CUDA-Q synthesis and GPU Lanczos "
                "certification are host work.\n"
            )
    if context.get("pysr"):
        prompt += (
            "\nPySR is enabled: hsopt_pysr can optimize a mixed-precision rule for your "
            "proposed base_policy after preparation. Its splitting/grouping and repetitions "
            "stay fixed during that inner search. All probes consume this run's evaluation "
            "budget. Inspect best.source, then adopt or revise it in the single candidate edit.\n"
        )
    if context and not shared:
        prompt += "\nRequired context:\n" + json.dumps(context, sort_keys=True) + "\n"
    if feedback is not None:
        summary = compact_feedback(feedback) if shared else feedback
        prompt += "\nCurrent evaluator feedback:\n" + json.dumps(summary, sort_keys=True) + "\n"
    return prompt
