#!/usr/bin/env python3
"""Validate a frozen angle allocation with the unchanged ground-input evaluator.

Post-hoc results and bound policies live outside the source/campaign. Cached
iteration-0 and agent controls are reused after input/compiler/count checks.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from hsopt.campaign.protocol import bind_policy
from hsopt.evaluation import compiled_unitary as ev


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def derived_policy(source, parameters, tolerances, output):
    """Keep the frozen schedule/repetitions; replace only the tolerance lookup."""
    bind_policy(source, parameters, output)
    tree = ev.validate_policy_source(output)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    initializer = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    initializer.body.extend(ast.parse("self.angle_tolerances = " + repr(tolerances)).body)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "synthesis_epsilon")
    method.body = ast.parse("return self.angle_tolerances[term['label']]").body
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
        comparisons.append({
            "case_id": case_id,
            "baseline_t_count": original["t_count"], "baseline_t95": original["score"],
            "agent_t_count": agent[case_id]["t_count"], "agent_t95": agent[case_id]["score"],
            "derived_t_count": current.get("t_count"), "derived_t95": current.get("score"),
            "derived_relative_t95": ratio,
            "derived_energy_error_mha": current.get("absolute_energy_error_mha"),
            "passes_chemical_accuracy": current.get("passes_chemical_accuracy"),
            "status": current.get("status", "pending"),
        })
    baseline_t = sum(value["t_count"] for value in baseline.values())
    agent_ratio = float(np.exp(np.mean(np.log([
        agent[key]["score"] / value["score"] for key, value in baseline.items()
    ]))))
    derived_ratio = float(np.exp(np.mean(np.log([
        row["derived_relative_t95"] for row in comparisons
    ])))) if all_pass else None
    return {
        "status": "complete" if len(results) == len(baseline) else "running",
        "molecules": len(baseline), "completed": len(results),
        "chemical_accuracy_passes": len(completed) - len(failures),
        "physical_accuracy_failures": failures, "uncertified_cases": uncertified,
        "passes_entire_suite": all_pass,
        "chemical_accuracy_hartree": ev.CHEMICAL_ACCURACY_HARTREE,
        "baseline_total_circuit_t_count": baseline_t,
        "agent_total_circuit_t_count": compilation["agent_total_compiled_evolution_t_count"],
        "derived_total_circuit_t_count": compilation["derived_total_compiled_evolution_t_count"],
        "agent_circuit_t_reduction_vs_baseline": 1 - compilation["agent_total_compiled_evolution_t_count"] / baseline_t,
        "derived_circuit_t_reduction_vs_baseline": 1 - compilation["derived_total_compiled_evolution_t_count"] / baseline_t,
        "agent_geomean_relative_t95": agent_ratio,
        "derived_geomean_relative_t95": derived_ratio,
        "derived_t95_reduction_vs_baseline": 1 - derived_ratio if all_pass else None,
        "max_derived_error_mha": max((v["absolute_energy_error_mha"] for v in completed.values()), default=None),
        "cases": comparisons,
    }


def main(before_case=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--compilation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--agent-iteration", required=True, type=int)
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output_dir.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("write post-hoc results outside the campaign")
    suite = json.loads((root / "suite.json").read_text())
    compilation = json.loads((args.compilation_dir / "synthesis_comparison.json").read_text())
    if suite.get("energy_target") != "ground_branch" or suite.get("kind") != "absolute":
        raise ValueError("ground-input absolute suite required")
    if compilation["policy_sha256"] != sha(args.policy) or compilation["suite_sha256"] != sha(root / "suite.json"):
        raise ValueError("frozen policy/suite and compilation report disagree")
    controls = root / "iterations"
    agent_dir = controls / f"{args.agent_iteration:04d}"
    if sha(agent_dir / "candidate.py") != sha(args.policy):
        raise ValueError("selected agent iteration is not the frozen compared policy")
    baseline = json.loads((controls / "0000/case_results.json").read_text())
    agent = json.loads((agent_dir / "case_results.json").read_text())
    expected = {case["case_id"]: case for case in compilation["cases"]}
    if set(baseline) != set(expected) or set(agent) != set(expected):
        raise ValueError("controls must contain the complete same cohort")
    table = args.compilation_dir / "compiled_term_comparison.csv.gz"
    if sha(table) != compilation["compiled_table_sha256"]:
        raise ValueError("compiled tolerance table checksum mismatch")
    tolerances = defaultdict(dict)
    with gzip.open(table, "rt") as handle:
        for row in csv.DictReader(handle):
            tolerances[row["case_id"]][row["label"]] = float(row["derived_epsilon"])
    for case_id in expected:
        for control in (baseline[case_id], agent[case_id]):
            if not control["passes_chemical_accuracy"] or control["energy_target"] != "ground_branch":
                raise ValueError("a saved control failed the selected scientific contract")
            if control["exact_reference_problem_sha256"] != expected[case_id]["problem_sha256"]:
                raise ValueError("saved control Hamiltonian differs")
            if control["rotation_synthesis"] != compilation["rotation_synthesis"]:
                raise ValueError("compiler settings differ from saved controls")
        if agent[case_id]["t_count"] != expected[case_id]["agent_compiled_evolution_t_count"]:
            raise ValueError("fresh agent compilation differs from saved scored circuit")
    for name in ("policies", "results"):
        (output / name).mkdir(parents=True, exist_ok=True)
    import cupy as cp

    metadata = {
        "policy_sha256": sha(args.policy), "compiled_table_sha256": sha(table),
        "suite_sha256": sha(root / "suite.json"), "validation_script_sha256": sha(__file__),
        "source_hashes": {name: sha(Path(ev.__file__).parent / name) for name in (
            "compiled_unitary.py", "gpu_circuit.py", "gpu_spectrum.py", "ground_branch.py",
        )},
        "accelerator": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        "backend": "cupy-lanczos", "matrix_free": True, "full_diagonalization_permitted": False,
        "agent_iteration": args.agent_iteration,
    }
    results = {}
    started = time.monotonic()
    write(output / "validation_summary.json", {**summarize(results, baseline, agent, compilation), **metadata})
    for i, case in enumerate(suite["cases"]):
        case_id = case["id"]
        result_path = output / "results" / f"{case_id}.json"
        policy_path = output / "policies" / f"{case_id}.py"
        source = root / case["source"]
        if sha(source) != expected[case_id]["problem_sha256"]:
            raise ValueError("Hamiltonian hash changed")
        derived_policy(args.policy, case["parameters"], tolerances[case_id], policy_path)
        if result_path.exists():
            stored = json.loads(result_path.read_text())
            if stored["derived_policy_sha256"] != sha(policy_path) or stored["metadata"] != metadata:
                raise ValueError("existing validation result has different provenance")
            result = stored["result"]
        else:
            if before_case is not None:
                before_case()
            case_start = time.monotonic()
            try:
                result = ev.evaluate_molecule(
                    source, policy_path, reference_path=root / case["reference"],
                    backend="cupy-lanczos", energy_target="ground_branch", matrix_free=True,
                )
                if result["t_count"] != expected[case_id]["derived_compiled_evolution_t_count"]:
                    raise ValueError("scored circuit T count differs from compiled comparison")
                if result["rotation_schedule_sha256"] != expected[case_id]["rotation_schedule_sha256"]:
                    raise ValueError("derived evaluation changed the frozen rotation schedule")
                if result["matrix_construction"] != "matrix_free_cuda_graph_literal_circuit":
                    raise ValueError("matrix-free evaluator required")
            except Exception as exc:
                result = {"status": "uncertified", "error": str(exc),
                          "error_type": type(exc).__name__, "case_id": case_id,
                          "evaluation_wall_seconds": time.monotonic() - case_start}
            write(result_path, {"derived_policy_sha256": sha(policy_path), "metadata": metadata, "result": result})
        results[case_id] = result
        report = {**summarize(results, baseline, agent, compilation), **metadata,
                  "wall_seconds": time.monotonic() - started}
        write(output / "validation_summary.json", report)
        print(f"{i+1}/{len(expected)} {case_id}: status={result['status']} "
              f"pass={result.get('passes_chemical_accuracy')} "
              f"error_mHa={result.get('absolute_energy_error_mha')} "
              f"T95={result.get('score')} seconds={result['evaluation_wall_seconds']:.1f}", flush=True)
        ev._compile_rz.cache_clear()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
