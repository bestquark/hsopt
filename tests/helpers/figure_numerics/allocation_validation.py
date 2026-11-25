"""Numerical validation extracted from archived validate_angle_allocation.py; no plotting frontend."""

from __future__ import annotations


import ast


import numpy as np

from hsopt.campaign.protocol import bind_policy

from hsopt.evaluation import compiled_unitary as ev


def derived_policy(source, parameters, tolerances, output):
    """Keep the frozen schedule/repetitions; replace only the tolerance lookup."""
    bind_policy(source, parameters, output)
    tree = ev.validate_policy_source(output)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    initializer = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    initializer.body.extend(ast.parse("self.synthesis_tolerances = " + repr(tolerances)).body)
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "synthesis_epsilon"
    )
    method.body = ast.parse("return self.synthesis_tolerances[term['label']]").body
    output.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    ev.validate_policy_source(output)


def summarize(results, baseline, agent, compilation):
    completed = {key: value for key, value in results.items() if value["status"] == "ok"}
    failures = [key for key, value in completed.items() if not value["passes_chemical_accuracy"]]
    uncertified = [key for key, value in results.items() if value["status"] != "ok"]
    all_pass = len(completed) == len(baseline) and not failures
    comparisons = []
    for case_id, original in baseline.items():
        current = results.get(case_id, {})
        ratio = None
        if current.get("status") == "ok" and current["passes_chemical_accuracy"]:
            ratio = current["score"] / original["score"]
        comparisons.append(
            {
                "case_id": case_id,
                "baseline_t_count": original["t_count"],
                "baseline_t95": original["score"],
                "agent_t_count": agent[case_id]["t_count"],
                "agent_t95": agent[case_id]["score"],
                "derived_t_count": current.get("t_count"),
                "derived_t95": current.get("score"),
                "derived_relative_t95": ratio,
                "baseline_energy_error_mha": original.get("absolute_energy_error_mha"),
                "agent_energy_error_mha": agent[case_id].get("absolute_energy_error_mha"),
                "derived_energy_error_mha": current.get("absolute_energy_error_mha"),
                "passes_chemical_accuracy": current.get("passes_chemical_accuracy"),
                "status": current.get("status", "pending"),
            }
        )
    baseline_t = sum(value["t_count"] for value in baseline.values())
    agent_ratio = float(
        np.exp(
            np.mean(
                np.log([agent[key]["score"] / value["score"] for key, value in baseline.items()])
            )
        )
    )
    derived_ratio = (
        float(np.exp(np.mean(np.log([row["derived_relative_t95"] for row in comparisons]))))
        if all_pass
        else None
    )
    return {
        "status": "complete" if len(results) == len(baseline) else "running",
        "molecules": len(baseline),
        "completed": len(results),
        "chemical_accuracy_passes": len(completed) - len(failures),
        "physical_accuracy_failures": failures,
        "uncertified_cases": uncertified,
        "passes_entire_suite": all_pass,
        "chemical_accuracy_hartree": ev.CHEMICAL_ACCURACY_HARTREE,
        "baseline_total_circuit_t_count": baseline_t,
        "agent_total_circuit_t_count": compilation["agent_total_compiled_evolution_t_count"],
        "derived_total_circuit_t_count": compilation["derived_total_compiled_evolution_t_count"],
        "agent_circuit_t_reduction_vs_baseline": 1
        - compilation["agent_total_compiled_evolution_t_count"] / baseline_t,
        "derived_circuit_t_reduction_vs_baseline": 1
        - compilation["derived_total_compiled_evolution_t_count"] / baseline_t,
        "agent_geomean_relative_t95": agent_ratio,
        "derived_geomean_relative_t95": derived_ratio,
        "derived_t95_reduction_vs_baseline": 1 - derived_ratio if all_pass else None,
        "derived_geomean_t95_over_agent": derived_ratio / agent_ratio if all_pass else None,
        "derived_t95_reduction_vs_agent": 1 - derived_ratio / agent_ratio if all_pass else None,
        "max_derived_error_mha": max(
            (v["absolute_energy_error_mha"] for v in completed.values()), default=None
        ),
        "cases": comparisons,
    }
