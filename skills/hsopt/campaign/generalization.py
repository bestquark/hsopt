"""Sealed random train/test partitions and one final, evaluation-only test.

Only the training selection enters the mutation sandbox. A final policy is
frozen after its training budget, then evaluated against its test baseline.
Test failures are reported, never used to choose another policy.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from . import protocol as p
from .interface import _lane_lock

SCHEMA = "hsopt-protocol-generalization-v1"


def split(selection: Path, output: Path, *, seed: int, train_size=80, test_size=20):
    selection = selection.resolve()
    rows = p.read(selection)["cases"]
    if min(train_size, test_size) < 1 or len(rows) != train_size + test_size:
        raise ValueError("dataset size must equal positive training and test sizes")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset contains repeated case IDs")
    fingerprints, canonical = [], []
    for row in sorted(rows, key=lambda row: row["id"]):
        bound = dict(row)
        files = {}
        for key in ("source", "reference", "calibration", "calibration_root"):
            if row.get(key):
                path = (selection.parent / row[key]).resolve()
                bound[key] = str(path)
                if key != "calibration_root":
                    files[key] = p.sha(path)
        fingerprints.append({"id": row["id"], "files": files})
        canonical.append(bound)
    if len({row["files"]["source"] for row in fingerprints}) != len(rows):
        raise ValueError("dataset contains duplicate Hamiltonian files")
    chosen = set(random.Random(seed).sample([r["id"] for r in canonical], train_size))
    train = {"cases": [r for r in canonical if r["id"] in chosen]}
    test = {"cases": [r for r in canonical if r["id"] not in chosen]}
    record = {
        "schema": SCHEMA,
        "seed": seed,
        "dataset_size": len(rows),
        "training_ids": [r["id"] for r in train["cases"]],
        "test_ids": [r["id"] for r in test["cases"]],
        "dataset_fingerprints": fingerprints,
        "sampling": "uniform_without_replacement_python_random_sample_sorted_case_ids",
        "test_use": "final_frozen_training_incumbent_only",
    }
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        verify_split(output)
        if (
            p.read(output / "split.json")["seed"] != seed
            or p.read(output / "train.json") != train
            or p.read(output / "test.json") != test
        ):
            raise ValueError("refuse to replace a sealed split")
        return p.read(output / "split.json")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("train.json", train), ("test.json", test)):
        p.write(output / name, value)
        record[name.removesuffix(".json") + "_sha256"] = p.sha(output / name)
        (output / name).chmod(0o444)
    p.write(output / "split.json", record)
    (output / "split.json").chmod(0o444)
    return record


def verify_split(directory: Path):
    record = p.read(directory / "split.json")
    if record.get("schema") != SCHEMA:
        raise ValueError("unknown split schema")
    by_id = {}
    partitions = []
    for name, field in (("train", "training_ids"), ("test", "test_ids")):
        path = directory / f"{name}.json"
        if p.sha(path) != record[name + "_sha256"]:
            raise ValueError("sealed split selection changed")
        rows = p.read(path)["cases"]
        ids = [r["id"] for r in rows]
        if ids != record[field] or len(ids) != len(set(ids)):
            raise ValueError("sealed partition membership changed")
        partitions.append(set(ids))
        by_id.update({r["id"]: r for r in rows})
    if partitions[0] & partitions[1] or len(by_id) != record["dataset_size"]:
        raise ValueError("training and test sets must be disjoint and exhaustive")
    for fingerprint in record["dataset_fingerprints"]:
        row = by_id[fingerprint["id"]]
        for key, digest in fingerprint["files"].items():
            if p.sha(Path(row[key])) != digest:
                raise ValueError("sealed dataset input changed")
    return record


def prepare_training(
    directory: Path,
    run: Path,
    *,
    iterations: int,
    replicate: str,
    baseline="tight",
    energy_target="ground_branch",
    policy_profile="fermionic_nested",
):
    record = verify_split(directory)
    result = p.prepare(
        directory / "train.json",
        run,
        family="abs_protocol",
        baseline=baseline,
        iterations=iterations,
        replicate=replicate,
        energy_target=energy_target,
        policy_profile=policy_profile,
    )
    manifest = p.read(run / "manifest.json")
    manifest["generalization"] = {
        "schema": SCHEMA,
        "role": "training",
        "split_sha256": p.sha(directory / "split.json"),
        "split_seed": record["seed"],
        "train_size": len(record["training_ids"]),
        "test_size": len(record["test_ids"]),
        "test_use": record["test_use"],
    }
    p.write(run / "manifest.json", manifest)
    return result


def prepare_test(directory: Path, training: Path, run: Path):
    """Freeze the final training incumbent before exposing any held-out result."""
    verify_split(directory)
    manifest, state = p.read(training / "manifest.json"), p.read(training / "state.json")
    split_hash = p.sha(directory / "split.json")
    if (
        manifest.get("generalization", {}).get("split_sha256") != split_hash
        or manifest["generalization"]["role"] != "training"
    ):
        raise ValueError("training run belongs to another split")
    if (
        state.get("pending_round") is not None
        or not state.get("initialized")
        or state["iterations_completed"] != manifest["iterations_target"]
    ):
        raise ValueError("training budget must finish before held-out evaluation")
    if not p.read(training / "incumbent_result.json")["passes_chemical_accuracy"]:
        raise ValueError("final training incumbent must be feasible")
    candidate = training / "incumbent.py"
    if (run / "manifest.json").exists():
        prior = p.read(run / "manifest.json").get("generalization", {})
        if prior.get("split_sha256") != split_hash or prior.get("frozen_candidate_sha256") != p.sha(
            candidate
        ):
            raise ValueError("held-out test already belongs to another split or policy")
        return {"status": "already_prepared"}
    result = p.prepare(
        directory / "test.json",
        run,
        family="abs_protocol",
        baseline=manifest["baseline"],
        iterations=1,
        replicate=manifest["replicate"],
        energy_target=manifest["energy_target"],
        policy_profile=manifest["policy_profile"],
    )
    shutil.copyfile(candidate, run / "frozen_protocol.py")
    (run / "frozen_protocol.py").chmod(0o444)
    test_manifest = p.read(run / "manifest.json")
    test_manifest["generalization"] = {
        "schema": SCHEMA,
        "role": "held_out_test",
        "split_sha256": split_hash,
        "training_manifest_sha256": p.sha(training / "manifest.json"),
        "frozen_candidate_sha256": p.sha(candidate),
        "training_rounds": state["iterations_completed"],
        "test_use": "one_evaluation_no_mutation_no_model_selection",
    }
    p.write(run / "manifest.json", test_manifest)
    return result


def advance_test(run: Path):
    """Resolve test baselines, then score the frozen policy once; never mutate."""
    manifest = p.read(run / "manifest.json")
    meta = manifest.get("generalization", {})
    candidate = run / "frozen_protocol.py"
    if meta.get("role") != "held_out_test" or p.sha(candidate) != meta.get(
        "frozen_candidate_sha256"
    ):
        raise ValueError("held-out test needs an unchanged frozen training policy")
    state = p.read(run / "state.json")
    if state.get("pending_round") == 1:
        folder, bundle = p._pending(run)
        if bundle["shared_candidate_sha256"] != meta["frozen_candidate_sha256"]:
            raise ValueError("test request does not evaluate the frozen training policy")
    result = p.advance(run, propose=False)
    if result["status"] == "ready_to_mutate":
        with _lane_lock(run):
            return p._enqueue_round(run, 1, candidate)
    if result["status"] == "complete":
        # The test suite may reject this policy; report its actual result,
        # never substitute the baseline or pick a different training iterate.
        evaluation = p.read(run / "iterations/0001/result.json")
        report = {
            "schema": SCHEMA,
            **meta,
            "result": evaluation,
            "test_case_count": evaluation["case_count"],
            "baseline_case_count": len(p.read(run / "iterations/0000/case_results.json")),
        }
        p.write(run / "generalization_report.json", report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    command = sub.add_parser("split")
    command.add_argument("--selection", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--seed", required=True, type=int)
    command.add_argument("--train-size", type=int, default=80)
    command.add_argument("--test-size", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            split(
                args.selection,
                args.output,
                seed=args.seed,
                train_size=args.train_size,
                test_size=args.test_size,
            ),
            indent=2,
        )
    )
