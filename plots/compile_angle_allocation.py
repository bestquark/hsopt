#!/usr/bin/env python3
"""Count actual CUDA-Q T gates for a bounded angle allocation and a frozen agent.

Compilation only: no GPU, energy evaluation, QPE success probability or T95.
Every Hamiltonian term is included, including near-zero angle responses.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_policy_sensitivity import sha
from hsopt.campaign.protocol import bind_policy
from hsopt.evaluation import compiled_unitary as ev


def bounded_allocation(weights, response, agent):
    """Minimize sum m log(1/eps) at the agent's linearized error budget.

    eps_j = min(cap, C/g_j), with cap=max(agent_eps). A zero response receives
    cap; it is neither discarded nor assigned unbounded synthesis tolerance.
    g_j is the mean absolute angle response over the term's occurrences.
    """
    weights, response, agent = (np.asarray(a, dtype=float) for a in (weights, response, agent))
    if not (weights.shape == response.shape == agent.shape) or weights.ndim != 1:
        raise ValueError("allocation vectors must be matching one-dimensional arrays")
    if not all(np.isfinite(a).all() for a in (weights, response, agent)):
        raise ValueError("allocation data must be finite")
    if np.any(weights <= 0) or np.any(response < 0) or np.any(agent <= 0) or np.any(agent >= 1):
        raise ValueError("invalid multiplicity, response, or tolerance")
    cap = float(agent.max())
    budget = float(np.sum(weights * response * agent))
    lower, upper = 0., cap * float(response.max())
    epsilon = np.full_like(response, cap)
    for _ in range(100):
        scale = .5 * (lower + upper)
        np.divide(scale, response, out=epsilon, where=response > 0)
        np.minimum(epsilon, cap, out=epsilon)
        used = float(np.sum(weights * response * epsilon))
        if used > budget:
            upper = scale
        else:
            lower = scale
    scale = lower
    np.divide(scale, response, out=epsilon, where=response > 0)
    np.minimum(epsilon, cap, out=epsilon)
    used = float(np.sum(weights * response * epsilon))
    if not np.isclose(used, budget, rtol=1e-11, atol=1e-28) or np.any(epsilon <= 0):
        raise ValueError("bounded allocation failed to match the first-order budget")
    old_cost = float(np.sum(weights * np.log(1 / agent)))
    new_cost = float(np.sum(weights * np.log(1 / epsilon)))
    if new_cost > old_cost + 1e-8 * max(1., old_cost):
        raise ValueError("derived allocation increased its own convex log-cost model")
    return epsilon.copy(), {
        "epsilon_cap": cap, "scale": scale, "agent_dimensionless_budget": budget,
        "derived_dimensionless_budget": used,
        "terms_at_cap": int(np.count_nonzero(epsilon >= cap * (1 - 1e-12))),
        "agent_log_cost": old_cost, "derived_log_cost": new_cost,
    }


def compiled_counts(schedule, formula):
    keys, compiled, execution = ev._compile_rotations(schedule, formula)
    by_term = Counter()
    for (term, _), key in zip(schedule, keys, strict=True):
        by_term[term["label"]] += compiled[key]["t_count"]
    for result in compiled.values():
        if result["t_count"] != result["gate_string"].count("T"):
            raise ValueError("CUDA-Q T count does not match the literal compiled gate string")
    return by_term, execution


def run_case(case, root, policy, samples, expected, output):
    problem_path = root / case["source"]
    if sha(problem_path) != expected["problem_sha256"]:
        raise ValueError("Hamiltonian differs from the frozen snapshot")
    problem = ev.load_problem(problem_path)
    by_label = {row["label"]: row for row in samples}
    if set(by_label) != {t["label"] for t in problem["terms"]}:
        raise ValueError("sensitivity table must include every Hamiltonian term")
    labels = sorted(by_label)
    weights = np.array([float(by_label[label]["occurrences"]) for label in labels])
    responses = np.array([float(by_label[label]["angle_weight"]) for label in labels])
    agent_epsilon = np.array([float(by_label[label]["epsilon"]) for label in labels])
    derived_epsilon, allocation = bounded_allocation(weights, responses, agent_epsilon)
    tolerances = dict(zip(labels, derived_epsilon, strict=True))

    class Derived:
        def synthesis_epsilon(self, term, angle):
            return float(tolerances[term["label"]])

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hsopt-angle-count-") as tmp:
        bound = Path(tmp) / "bound.py"
        bind_policy(policy, case["parameters"], bound)
        formula, schedule, metrics = ev._product_formula_schedule(problem, ev.load_policy(bound))
    multiplicities = Counter(term["label"] for term, _ in schedule)
    for label, weight in zip(labels, weights, strict=True):
        if multiplicities[label] != weight:
            raise ValueError("rotation multiplicity differs from sensitivity analysis")
    for term, angle in schedule:
        if not math.isclose(formula.synthesis_epsilon(term, angle),
                            float(by_label[term["label"]]["epsilon"]), rel_tol=1e-12):
            raise ValueError("agent tolerance differs from frozen table")
    agent_counts, agent_execution = compiled_counts(schedule, formula)
    derived_counts, derived_execution = compiled_counts(schedule, Derived())
    n_steps = int(metrics["n_steps"])
    agent_total, derived_total = (n_steps * sum(counts.values()) for counts in (agent_counts, derived_counts))
    gm_agent = float(np.exp(np.average(np.log(agent_epsilon), weights=weights)))
    gm_derived = float(np.exp(np.average(np.log(derived_epsilon), weights=weights)))
    terms = []
    for label, epsilon in zip(labels, derived_epsilon, strict=True):
        row = dict(by_label[label])
        row.update({
            "derived_epsilon": float(epsilon),
            "learned_shape": float(row["epsilon"]) / gm_agent,
            "derived_shape": float(epsilon) / gm_derived,
            "agent_term_step_t_count": agent_counts[label],
            "derived_term_step_t_count": derived_counts[label],
        })
        terms.append(row)
    result = {
        "case_id": case["id"], "qubits": problem["n_qubits"],
        "problem_sha256": sha(problem_path), "terms": len(labels),
        "n_steps": n_steps, "rotations_per_step": len(schedule),
        "rotation_schedule_sha256": metrics["rotation_schedule_sha256"],
        "allocation": allocation,
        "agent_compiled_evolution_t_count": agent_total,
        "derived_compiled_evolution_t_count": derived_total,
        "derived_over_agent": derived_total / agent_total,
        "agent_synthesis_execution": agent_execution,
        "derived_synthesis_execution": derived_execution,
        "wall_seconds": time.perf_counter() - started,
        "energy_validated_for_derived": False,
    }
    (output / "cases" / f"{case['id']}.json").write_text(json.dumps(result, indent=2) + "\n")
    ev._compile_rz.cache_clear()
    return result, terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sensitivities-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output_dir.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("write compilation diagnostics outside the campaign")
    snapshot = json.loads(args.snapshot.read_text())
    sensitivity = json.loads((args.sensitivities_dir / "sensitivity_manifest.json").read_text())
    if sha(args.policy) != snapshot["policy_sha256"] or sensitivity["policy_sha256"] != sha(args.policy):
        raise ValueError("policy and sensitivity snapshots must match")
    suite = json.loads((root / "suite.json").read_text())
    expected = {c["case_id"]: c for c in snapshot["cases"]}
    with gzip.open(args.sensitivities_dir / "term_sensitivities.csv.gz", "rt") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    if set(grouped) != set(expected) or set(expected) != {c["id"] for c in suite["cases"]}:
        raise ValueError("compile the complete frozen cohort")
    if len(rows) != sensitivity["terms"]:
        raise ValueError("sensitivity table row count changed")
    (output / "cases").mkdir(parents=True, exist_ok=True)
    synthesis = ev._synthesis_metadata()
    started = time.perf_counter()
    results, terms = [], []
    for i, case in enumerate(suite["cases"]):
        result, points = run_case(case, root, args.policy, grouped[case["id"]], expected[case["id"]], output)
        results.append(result)
        terms.extend(points)
        print(f"{i+1}/{len(expected)} {case['id']}: "
              f"agent={result['agent_compiled_evolution_t_count']} "
              f"derived={result['derived_compiled_evolution_t_count']} "
              f"ratio={result['derived_over_agent']:.4f} {result['wall_seconds']:.1f}s", flush=True)
    table = output / "compiled_term_comparison.csv.gz"
    with gzip.open(table, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(terms[0]))
        writer.writeheader()
        writer.writerows(terms)
    agent_total = sum(r["agent_compiled_evolution_t_count"] for r in results)
    derived_total = sum(r["derived_compiled_evolution_t_count"] for r in results)
    report = {
        "schema": 1, "status": "complete", "policy_sha256": sha(args.policy),
        "script_sha256": sha(__file__), "suite_sha256": sha(root / "suite.json"),
        "sensitivity_manifest_sha256": sha(args.sensitivities_dir / "sensitivity_manifest.json"),
        "compiled_table_sha256": sha(table), "molecules": len(results), "terms": len(terms),
        "rotation_synthesis": synthesis,
        "cost_definition": "T gates for one full physical evolution per molecule; includes PF repetitions",
        "error_budget": "match each molecule's agent first-order absolute angle-error budget",
        "epsilon_cap": "each molecule's maximum agent tolerance",
        "near_zero_terms": "included; bounded optimum uses cap, no sensitivity floor",
        "agent_total_compiled_evolution_t_count": agent_total,
        "derived_total_compiled_evolution_t_count": derived_total,
        "derived_over_agent_total": derived_total / agent_total,
        "geometric_mean_derived_over_agent": float(np.exp(np.mean(np.log([r["derived_over_agent"] for r in results])))),
        "molecules_fewer_t_gates": sum(r["derived_over_agent"] < 1 for r in results),
        "molecules_more_t_gates": sum(r["derived_over_agent"] > 1 for r in results),
        "molecules_equal_t_gates": sum(r["derived_over_agent"] == 1 for r in results),
        "energy_or_t95_evaluated": False, "gpu_used": False,
        "wall_seconds": time.perf_counter() - started, "cases": results,
    }
    (output / "synthesis_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}), flush=True)


if __name__ == "__main__":
    main()
