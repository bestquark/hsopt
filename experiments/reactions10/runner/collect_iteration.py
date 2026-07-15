#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from campaign_common import (
    SCHEMA_VERSION,
    campaign_lock,
    campaign_paths,
    candidate_path,
    copy_file_atomic,
    expected_lane_ids,
    ensure_campaign_runner_runtime,
    iteration_dir,
    lane_map,
    mutation_dir,
    now_iso,
    open_collect_transactions,
    read_json,
    relative_to_campaign,
    require_campaign,
    result_path,
    result_validation_error,
    sha256_file,
    validate_iteration,
    write_json,
    write_state,
)


def rsync_results(root: Path, cluster_root: str, iteration: int, sync_host: str) -> None:
    destination = iteration_dir(root, iteration) / "results"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-ah",
            "--partial",
            f"{sync_host}:{cluster_root.rstrip('/')}/iterations/iter_{iteration:04d}/results/",
            f"{destination}/",
        ],
        check=True,
    )


def prevalidate_iteration(
    *,
    root: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    iteration: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """Validate every lane without mutating campaign state or canonical conveniences.

    Returns (results, candidate_hashes, retryable_errors, fatal_errors).
    Missing/error evaluator results are retryable. Candidate, lineage, and arithmetic
    drift are fatal because another evaluation cannot repair them.
    """
    by_id = lane_map(manifest)
    results: dict[str, dict[str, Any]] = {}
    candidate_hashes: dict[str, str] = {}
    retryable: dict[str, str] = {}
    fatal: dict[str, str] = {}
    for lane_id in expected_lane_ids(manifest):
        lane = by_id[lane_id]
        lane_state = state["lanes"].get(lane_id)
        if not isinstance(lane_state, dict):
            fatal[lane_id] = "campaign state is missing the lane"
            continue
        candidate = candidate_path(root, iteration, lane_id)
        record_path = mutation_dir(root, iteration, lane_id) / "record.json"
        try:
            if not candidate.is_file() or not record_path.is_file():
                raise FileNotFoundError("candidate or mutation record is missing")
            record = read_json(record_path)
            candidate_hash = sha256_file(candidate)
            candidate_hashes[lane_id] = candidate_hash
            if (
                record.get("schema_version") != SCHEMA_VERSION
                or record.get("lane_id") != lane_id
                or record.get("iteration") != iteration
                or record.get("status") != "candidate_ready"
                or record.get("candidate_policy_sha256") != candidate_hash
            ):
                raise RuntimeError("candidate and immutable mutation record disagree")
            candidate_files = sorted(
                path.relative_to(candidate.parent).as_posix()
                for path in candidate.parent.rglob("*")
                if path.is_file()
            )
            if candidate_files != ["simple_pf.py"]:
                raise RuntimeError("candidate directory contains unexpected files")
            if iteration == 0:
                if (
                    record.get("kind") != "baseline"
                    or record.get("parent_iteration") is not None
                    or record.get("parent_policy_sha256") is not None
                    or candidate_hash != lane.get("seed_policy_sha256")
                ):
                    raise RuntimeError("baseline provenance/lineage mismatch")
                if int(lane_state.get("proposal_count", -1)) != 0:
                    raise RuntimeError("baseline lane proposal count is not zero")
            else:
                mutation_root = mutation_dir(root, iteration, lane_id)
                prompt_path = mutation_root / "prompt.md"
                intent_path = mutation_root / "intent.json"
                launch_intent_path = mutation_root / "launch_intent.json"
                launch_receipt_path = mutation_root / "launch.json"
                if not all(
                    path.is_file()
                    for path in (
                        prompt_path,
                        intent_path,
                        launch_intent_path,
                        launch_receipt_path,
                    )
                ):
                    raise RuntimeError("proposal lacks durable prompt/launch evidence")
                intent = read_json(intent_path)
                launch_intent = read_json(launch_intent_path)
                launch_receipt = read_json(launch_receipt_path)
                if (
                    record.get("kind") != "autoresearch_proposal"
                    or record.get("parent_iteration") != lane_state.get("incumbent_iteration")
                    or record.get("parent_policy_sha256")
                    != lane_state.get("incumbent_policy_sha256")
                    or int(lane_state.get("proposal_count", -1)) != iteration - 1
                    or record.get("prompt_sha256")
                    != hashlib.sha256(prompt_path.read_bytes()).hexdigest()
                    or intent.get("prompt_sha256") != record.get("prompt_sha256")
                    or record.get("intent_sha256") != sha256_file(intent_path)
                    or record.get("launch_intent_sha256") != sha256_file(launch_intent_path)
                    or record.get("launch_receipt_sha256") != sha256_file(launch_receipt_path)
                    or launch_intent.get("command_sha256") != record.get("launch_command_sha256")
                    or launch_receipt.get("command_sha256") != record.get("launch_command_sha256")
                    or launch_receipt.get("status") != "launched"
                ):
                    raise RuntimeError("proposal parent lineage/count mismatch")
                process_path = mutation_root / "process.json"
                if record.get("process_sha256") != (
                    sha256_file(process_path) if process_path.is_file() else None
                ):
                    raise RuntimeError("proposal process receipt hash mismatch")
                if process_path.is_file() and read_json(process_path).get(
                    "command_sha256"
                ) != record.get("launch_command_sha256"):
                    raise RuntimeError("proposal process command hash mismatch")
        except Exception as exc:
            fatal[lane_id] = f"{type(exc).__name__}: {exc}"
            continue

        result_file = result_path(root, iteration, lane_id)
        error = result_validation_error(
            result_file,
            lane=lane,
            iteration=iteration,
            candidate_hash=candidate_hashes[lane_id],
        )
        if error is not None:
            try:
                present_payload = read_json(result_file) if result_file.is_file() else None
            except Exception:
                present_payload = None
            if isinstance(present_payload, dict) and present_payload.get("status") == "ok":
                fatal[lane_id] = error
            else:
                retryable[lane_id] = error
            continue
        payload = read_json(result_file)
        execution = manifest.get("execution")
        if isinstance(execution, dict) and execution.get("cluster_root"):
            cluster_root = Path(str(execution["cluster_root"]))
            expected_cache_root = str((cluster_root / "problem_cache").resolve())
            expected_policy_path = str(
                (
                    cluster_root
                    / "iterations"
                    / f"iter_{iteration:04d}"
                    / "candidates"
                    / lane_id
                    / "simple_pf.py"
                ).resolve()
            )
            if (
                payload.get("cache_root") != expected_cache_root
                or payload.get("policy_path") != expected_policy_path
            ):
                fatal[lane_id] = (
                    "result was not evaluated from the immutable cluster candidate/cache"
                )
                continue
        if iteration == 0 and not bool(payload["passes_contract"]):
            fatal[lane_id] = "fixed baseline fails its lane contract"
            continue
        if iteration == 0:
            try:
                validate_baseline_seed(lane, payload)
            except Exception as exc:
                fatal[lane_id] = f"{type(exc).__name__}: {exc}"
                continue
        results[lane_id] = payload
    return results, candidate_hashes, retryable, fatal


def validate_baseline_seed(lane: dict[str, Any], result: dict[str, Any]) -> None:
    expectation = lane.get("seed_result_expectation")
    if not isinstance(expectation, dict):
        raise RuntimeError("lane has no fixed seed-result expectation")
    if result.get("total_t_gate_count") != expectation.get("total_t_gate_count"):
        raise RuntimeError("iteration-0 T count does not reproduce the preserved seed")
    for field in (
        "signed_reaction_error_hartree",
        "absolute_reaction_error_hartree",
        "triangle_bound_hartree",
        "max_species_absolute_error_hartree",
        "equal_share_error_limit_hartree",
    ):
        if not math.isclose(
            float(result[field]),
            float(expectation[field]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-10,
        ):
            raise RuntimeError(f"iteration-0 {field} does not reproduce the preserved seed")
    expected_species = expectation.get("species")
    if not isinstance(expected_species, dict) or set(expected_species) != set(lane["species"]):
        raise RuntimeError("fixed seed species expectation is malformed")
    for species, expected in expected_species.items():
        actual = result["species"][species]
        if actual.get("formula_order") != expectation.get("formula_order") or actual.get(
            "n_steps"
        ) != expectation.get("n_steps"):
            raise RuntimeError(f"iteration-0 PF order/step drift for species {species}")
        if actual.get("t_gate_count") != expected.get("t_gate_count"):
            raise RuntimeError(f"iteration-0 T count drift for species {species}")
        if not math.isclose(
            float(actual["signed_energy_error_hartree"]),
            float(expected["signed_energy_error_hartree"]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-10,
        ):
            raise RuntimeError(f"iteration-0 signed error drift for species {species}")


def build_decisions(
    *,
    root: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    iteration: int,
    results: dict[str, dict[str, Any]],
    candidate_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    if set(results) != set(expected_lane_ids(manifest)):
        raise RuntimeError("cannot decide an iteration until all 20 results validate")
    decisions: list[dict[str, Any]] = []
    for lane_id in expected_lane_ids(manifest):
        lane_state = state["lanes"][lane_id]
        result = results[lane_id]
        previous_t = lane_state.get("incumbent_t_gate_count")
        candidate_t = int(result["total_t_gate_count"])
        if iteration == 0:
            accepted = True
            reason = "initial_baseline"
        elif not bool(result["passes_contract"]):
            accepted = False
            reason = "contract_failed"
        elif previous_t is None:
            raise RuntimeError(f"{lane_id}: proposal has no accepted baseline")
        elif candidate_t < int(previous_t):
            accepted = True
            reason = "strict_t_gate_improvement"
        else:
            accepted = False
            reason = "no_t_gate_improvement"
        best_iteration = iteration if accepted else lane_state.get("incumbent_iteration")
        best_t_count = candidate_t if accepted else previous_t
        decisions.append(
            {
                "lane_id": lane_id,
                "iteration": iteration,
                "accepted": accepted,
                "reason": reason,
                "decision_reason": reason,
                "parent_iteration": None
                if iteration == 0
                else lane_state.get("incumbent_iteration"),
                "parent_policy_sha256": None
                if iteration == 0
                else lane_state.get("incumbent_policy_sha256"),
                "candidate_policy_sha256": candidate_hashes[lane_id],
                "result_sha256": sha256_file(result_path(root, iteration, lane_id)),
                "passes_contract": bool(result["passes_contract"]),
                "candidate_t_gate_count": candidate_t,
                "previous_best_t_gate_count": previous_t,
                "best_iteration": best_iteration,
                "best_t_gate_count": best_t_count,
            }
        )
    return decisions


def _batches_for_iteration(state: dict[str, Any], iteration: int) -> list[dict[str, Any]]:
    batches = [
        batch for batch in state["batches"].values() if int(batch.get("iteration", -1)) == iteration
    ]
    batches.sort(key=lambda item: str(item["batch_id"]))
    return batches


def validate_batch_partition(
    state: dict[str, Any], manifest: dict[str, Any], iteration: int
) -> None:
    batches = _batches_for_iteration(state, iteration)
    covered = [str(lane_id) for batch in batches for lane_id in batch.get("lane_ids", [])]
    expected = expected_lane_ids(manifest)
    if len(covered) != len(set(covered)) or set(covered) != set(expected):
        raise RuntimeError("persisted evaluation batches must cover every lane exactly once")
    for batch in batches:
        if batch.get("state") not in {"prepared", "submitted", "retry_ready", "collected"}:
            raise RuntimeError(f"invalid persisted batch state for {batch.get('batch_id')}")


def mark_retryable_batches(
    *,
    root: Path,
    state: dict[str, Any],
    iteration: int,
    errors: dict[str, str],
) -> None:
    error_ids = set(errors)
    covered: set[str] = set()
    fingerprint = hashlib.sha256(json.dumps(errors, sort_keys=True).encode("utf-8")).hexdigest()
    for batch in _batches_for_iteration(state, iteration):
        retry_ids = [lane_id for lane_id in batch["lane_ids"] if lane_id in error_ids]
        if not retry_ids:
            continue
        covered.update(retry_ids)
        if batch.get("retry_fingerprint") != fingerprint or batch.get("state") != "retry_ready":
            batch["retry_count"] = int(batch.get("retry_count", 0)) + 1
        batch["retry_fingerprint"] = fingerprint
        batch["retry_lane_ids"] = retry_ids
        batch["state"] = "retry_ready"
        batch["updated_at"] = now_iso()
        attempts = batch.get("attempts", [])
        if attempts and attempts[-1].get("state") in {"prepared", "submitted"}:
            attempts[-1]["state"] = "retry_ready"
            attempts[-1]["retry_reason"] = {lane_id: errors[lane_id] for lane_id in retry_ids}
        batch_path = (
            iteration_dir(root, iteration) / "batches" / str(batch["batch_id"]) / "batch.json"
        )
        write_json(batch_path, batch)
    if covered != error_ids:
        raise RuntimeError(
            f"retryable lanes are not covered by persisted batches: {sorted(error_ids - covered)}"
        )


def desired_state_after_collection(
    *,
    root: Path,
    state: dict[str, Any],
    iteration: int,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    desired = copy.deepcopy(state)
    by_decision = {str(item["lane_id"]): item for item in decisions}
    for lane_id, decision in by_decision.items():
        lane_state = desired["lanes"][lane_id]
        candidate = candidate_path(root, iteration, lane_id)
        result_file = result_path(root, iteration, lane_id)
        if decision["accepted"]:
            lane_state["incumbent_iteration"] = iteration
            lane_state["incumbent_policy_path"] = relative_to_campaign(root, candidate)
            lane_state["incumbent_policy_sha256"] = decision["candidate_policy_sha256"]
            lane_state["incumbent_t_gate_count"] = decision["candidate_t_gate_count"]
            lane_state["incumbent_result_path"] = relative_to_campaign(root, result_file)
            lane_state["status"] = "accepted"
        else:
            lane_state["status"] = "rejected"
        lane_state["last_collected_iteration"] = iteration
        lane_state["proposal_count"] = iteration
        lane_state["last_result_sha256"] = decision["result_sha256"]
    for batch in _batches_for_iteration(desired, iteration):
        batch["state"] = "collected"
        batch["retry_lane_ids"] = []
        batch["collected_at"] = now_iso()
        batch["updated_at"] = now_iso()
        attempts = batch.get("attempts", [])
        if attempts:
            attempts[-1]["state"] = "collected"
            attempts[-1]["collected_at"] = now_iso()
    desired["current_collected_iteration"] = iteration
    desired["next_iteration"] = iteration + 1 if iteration < 50 else None
    desired["last_validation_errors"] = {}
    desired["updated_at"] = now_iso()
    return desired


def replay_collect_transaction(root: Path, transaction_path: Path) -> None:
    transaction = read_json(transaction_path)
    if transaction.get("phase") == "committed":
        return
    if transaction.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported collection transaction: {transaction_path}")
    iteration = int(transaction["iteration"])
    decisions = list(transaction["decisions"])
    desired = copy.deepcopy(transaction["desired_state"])
    transaction["phase"] = "applying"
    transaction["replayed_at"] = now_iso()
    write_json(transaction_path, transaction)

    for decision in decisions:
        lane_id = str(decision["lane_id"])
        candidate = candidate_path(root, iteration, lane_id)
        result_file = result_path(root, iteration, lane_id)
        if sha256_file(candidate) != decision["candidate_policy_sha256"]:
            raise RuntimeError(f"transaction candidate hash drift for {lane_id}")
        if sha256_file(result_file) != decision["result_sha256"]:
            raise RuntimeError(f"transaction result hash drift for {lane_id}")
        lane_dir = root / "lanes" / lane_id
        copy_file_atomic(result_file, lane_dir / "last_result.json")
        incumbent_path = root / desired["lanes"][lane_id]["incumbent_policy_path"]
        if sha256_file(incumbent_path) != desired["lanes"][lane_id]["incumbent_policy_sha256"]:
            raise RuntimeError(f"transaction incumbent hash drift for {lane_id}")
        copy_file_atomic(incumbent_path, lane_dir / "simple_pf.py")
        if decision["accepted"]:
            copy_file_atomic(candidate, lane_dir / "best" / "simple_pf.py")
            copy_file_atomic(result_file, lane_dir / "best" / "result.json")

    decision_payload = {
        "schema_version": SCHEMA_VERSION,
        "iteration": iteration,
        "created_at": transaction["created_at"],
        "decisions": decisions,
    }
    write_json(iteration_dir(root, iteration) / "decisions.json", decision_payload)
    for batch in _batches_for_iteration(desired, iteration):
        write_json(
            iteration_dir(root, iteration) / "batches" / str(batch["batch_id"]) / "batch.json",
            batch,
        )
    write_state(root, desired)
    transaction["desired_state"] = desired
    transaction["phase"] = "committed"
    transaction["committed_at"] = now_iso()
    transaction["desired_state_sha256"] = hashlib.sha256(
        json.dumps(desired, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(transaction_path, transaction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prevalidate and atomically apply one reaction campaign iteration."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--sync-host")
    parser.add_argument("--skip-rsync", action="store_true")
    args = parser.parse_args()
    validate_iteration(args.iteration)
    if not args.skip_rsync and not args.sync_host:
        parser.error("--sync-host is required unless --skip-rsync is used")

    root = args.campaign_root.resolve()
    with campaign_lock(root, "orchestration"):
        manifest, state = require_campaign(root, allow_open_collect_transaction=True)
        ensure_campaign_runner_runtime(root, Path(__file__), manifest)
        open_transactions = open_collect_transactions(root)
        if open_transactions:
            if len(open_transactions) != 1:
                raise RuntimeError(f"multiple open collection transactions: {open_transactions}")
            transaction = read_json(open_transactions[0])
            if int(transaction.get("iteration", -1)) != args.iteration:
                raise RuntimeError(
                    f"must replay collection iteration {transaction.get('iteration')} before {args.iteration}"
                )
            replay_collect_transaction(root, open_transactions[0])
            print(f"replayed and committed collection transaction for iteration {args.iteration}")
            return
        if int(state["next_iteration"]) != args.iteration:
            if int(state["current_collected_iteration"]) == args.iteration:
                print(f"iteration {args.iteration} already collected")
                return
            raise RuntimeError(
                f"iteration sequence violation: state expects {state['next_iteration']}, got {args.iteration}"
            )
        if not args.skip_rsync:
            rsync_results(
                root,
                str(manifest["execution"]["cluster_root"]),
                args.iteration,
                str(args.sync_host),
            )
        validate_batch_partition(state, manifest, args.iteration)
        results, candidate_hashes, retryable, fatal = prevalidate_iteration(
            root=root,
            manifest=manifest,
            state=state,
            iteration=args.iteration,
        )
        validation_payload = {
            "schema_version": SCHEMA_VERSION,
            "iteration": args.iteration,
            "validated_lane_count": len(results),
            "retryable_errors": retryable,
            "fatal_errors": fatal,
            "validated_at": now_iso(),
        }
        write_json(iteration_dir(root, args.iteration) / "validation.json", validation_payload)
        if fatal:
            state["last_validation_errors"] = fatal
            write_state(root, state)
            raise RuntimeError(f"fatal iteration provenance/arithmetic errors: {fatal}")
        if retryable:
            mark_retryable_batches(
                root=root,
                state=state,
                iteration=args.iteration,
                errors=retryable,
            )
            state["last_validation_errors"] = retryable
            write_state(root, state)
            print(
                json.dumps({"iteration": args.iteration, "retry_ready": retryable}, sort_keys=True)
            )
            raise SystemExit(2)
        decisions = build_decisions(
            root=root,
            manifest=manifest,
            state=state,
            iteration=args.iteration,
            results=results,
            candidate_hashes=candidate_hashes,
        )
        desired = desired_state_after_collection(
            root=root,
            state=state,
            iteration=args.iteration,
            decisions=decisions,
        )
        transaction_path = (
            campaign_paths(root)["transactions"] / f"collect_iter_{args.iteration:04d}.json"
        )
        if transaction_path.exists():
            raise RuntimeError(f"collection transaction already exists: {transaction_path}")
        transaction = {
            "schema_version": SCHEMA_VERSION,
            "phase": "prepared",
            "iteration": args.iteration,
            "manifest_sha256": state["manifest_sha256"],
            "created_at": now_iso(),
            "decisions": decisions,
            "desired_state": desired,
        }
        write_json(transaction_path, transaction)
        replay_collect_transaction(root, transaction_path)
        accepted = sum(int(decision["accepted"]) for decision in decisions)
        print(
            f"collected iteration={args.iteration} accepted={accepted} "
            f"rejected={len(decisions) - accepted}"
        )


if __name__ == "__main__":
    main()
