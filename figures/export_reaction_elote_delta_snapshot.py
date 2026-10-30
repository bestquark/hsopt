"""Export the completed CUDA-Q reaction campaign for Figure 4.

Run this beside the HSOpt workdir with ``--workdir``. The checked-in JSON is a
small, auditable snapshot; plotting never needs access to a live cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs(result: dict) -> dict[str, int]:
    assert result["status"] == "ok"
    assert result["passes_chemical_accuracy"] is True
    assert not result["failed_case_ids"] and not result["unresolved_case_ids"]
    rows = result["case_metrics"]
    values = {row["case_id"]: int(row["t95"]) for row in rows}
    assert len(rows) == len(values) == 10
    assert all(row["passes_chemical_accuracy"] and row["t95"] > 0 for row in rows)
    return values


def versioned_cases(path: Path, expected_cases: set[str]) -> tuple[float, str, str]:
    rows = read(path)
    assert set(rows) == expected_cases
    assert all(row["status"] == "ok" and row["passes_chemical_accuracy"] for row in rows.values())
    contracts = {row["evaluation_contract_version"] for row in rows.values()}
    evaluators = {row["evaluator_sha256"] for row in rows.values()}
    compilers = {
        species["rotation_synthesis"]["version"]
        for row in rows.values()
        for species in row["species"].values()
    }
    assert len(contracts) == len(evaluators) == len(compilers) == 1
    assert all(row["backend"] == "cupy-lanczos" for row in rows.values())
    assert all(row["energy_target"] == "ground_branch" for row in rows.values())
    assert all(row["difference_error_contract"] == "error_cancellation" for row in rows.values())
    assert "CUDA-Q Version 0.16.0" in next(iter(compilers))
    return (
        max(row["absolute_difference_error_mha"] for row in rows.values()),
        next(iter(evaluators)),
        next(iter(compilers)),
    )


def export(work: Path) -> dict:
    runs = work / "runs"
    campaign = runs / "elote_delta_ground10_warm_a100_20261023"
    transfer = runs / "validation/elote_delta_ground10_transfer_20261023"
    transfer_manifest = read(transfer / "manifest.json")
    baseline_path = transfer / "iterations/0000/result.json"
    transferred_path = transfer / "iterations/0001/result.json"
    baseline = costs(read(baseline_path))
    transferred = costs(read(transferred_path))
    cases = set(baseline)
    assert set(transferred) == cases
    transferred_sha = sha(transfer / "iterations/0001/candidate.py")
    suite_sha = transfer_manifest["suite_sha256"]
    contract = transfer_manifest["evaluation_contract_version"]
    assert contract == "normalized_hamiltonian_ground_input_signed_difference_cudaq_elote_v1"

    baseline_error, baseline_evaluator, compiler = versioned_cases(
        transfer / "iterations/0000/case_results.json", cases
    )
    transfer_error, transfer_evaluator, transfer_compiler = versioned_cases(
        transfer / "iterations/0001/case_results.json", cases
    )
    assert baseline_evaluator == transfer_evaluator
    assert compiler == transfer_compiler

    seeds = sorted(campaign.glob("seed-*"))
    assert len(seeds) == 5
    ranked = []
    for seed in seeds:
        manifest = read(seed / "manifest.json")
        state = read(seed / "state.json")
        assert state["iterations_completed"] == manifest["iterations_target"] == 50
        assert state["pending_round"] is None
        assert manifest["suite_sha256"] == suite_sha
        assert manifest["evaluation_contract_version"] == contract
        assert manifest["initial_policy_sha256"] == transferred_sha
        assert manifest["policy_profile"] == "fermionic_nested"
        assert manifest["difference_error_contract"] == "error_cancellation"
        assert manifest["energy_target"] == "ground_branch"
        assert costs(read(seed / "iterations/0000/result.json")) == transferred
        accepted = [
            i
            for i in range(51)
            if read(seed / "iterations" / f"{i:04d}" / "decision.json")["accepted"]
        ]
        assert accepted and accepted[0] == 0
        iteration = max(accepted)
        result_path = seed / "iterations" / f"{iteration:04d}" / "result.json"
        result = read(result_path)
        incumbent_path = seed / "incumbent_result.json"
        assert result == read(incumbent_path)
        values = costs(result)
        assert set(values) == cases
        score = math.exp(sum(math.log(values[k] / transferred[k]) for k in cases) / len(cases))
        assert math.isclose(score, result["score"], rel_tol=1e-12)
        candidate_path = seed / "iterations" / f"{iteration:04d}" / "candidate.py"
        assert sha(candidate_path) == sha(seed / "incumbent.py")
        error, evaluator, candidate_compiler = versioned_cases(
            seed / "iterations" / f"{iteration:04d}" / "case_results.json", cases
        )
        assert evaluator == baseline_evaluator and candidate_compiler == compiler
        raw = read(seed / "iterations" / f"{iteration:04d}" / "case_results.json")
        assert all(row["execution_accelerator"] == "a100" for row in raw.values())
        threshold = min(row["chemical_accuracy_hartree"] * 1000 for row in raw.values())
        assert error < threshold
        ranked.append(
            {
                "seed": seed.name,
                "accepted_iteration": iteration,
                "score_relative_to_transferred_elote": score,
                "t95": values,
                "max_reaction_error_mha": error,
                "chemical_accuracy_mha": threshold,
                "candidate_sha256": sha(candidate_path),
                "result_sha256": sha(result_path),
            }
        )
    ranked.sort(key=lambda row: (row["score_relative_to_transferred_elote"], row["seed"]))

    suite = read(seeds[0] / "suite.json")
    widths = {}
    for row in suite["cases"]:
        source = seeds[0] / row["source"]
        reaction = read(source)
        widths[row["id"]] = max(
            int(read(source.parent / term["problem"])["n_qubits"]) for term in reaction["terms"]
        )
    assert set(widths) == cases
    return {
        "schema": "reaction-elote-delta-cudaq-a100-figure4-v1",
        "order": [row["id"] for row in suite["cases"]],
        "reaction_max_qubits": widths,
        "baseline_t95": baseline,
        "elote_t95": transferred,
        "ranked_policies": ranked,
        "compiler": compiler,
        "eigensolver": "cupy-lanczos",
        "evaluation_contract_version": contract,
        "evaluator_sha256": baseline_evaluator,
        "suite_sha256": suite_sha,
        "transferred_policy_sha256": transferred_sha,
        "baseline_max_reaction_error_mha": baseline_error,
        "transferred_max_reaction_error_mha": transfer_error,
        "baseline_result_sha256": sha(baseline_path),
        "transferred_result_sha256": sha(transferred_path),
        "baseline_accelerator": "l4",
        "elote_and_delta_accelerator": "a100",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.workdir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
