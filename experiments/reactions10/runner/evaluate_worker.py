#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from campaign_common import (
    candidate_path,
    evaluate_candidate,
    ensure_campaign_runner_runtime,
    lane_map,
    mutation_dir,
    now_iso,
    read_json,
    require_campaign,
    result_path,
    result_validation_error,
    sha256_file,
    validate_iteration,
    write_json,
)


def evaluate_one(root_text: str, lane: dict[str, Any], iteration: int) -> dict[str, Any]:
    root = Path(root_text)
    try:
        return evaluate_candidate(root, lane, iteration)
    except Exception as exc:
        candidate = candidate_path(root, iteration, str(lane["lane_id"]))
        return {
            "lane_id": lane["lane_id"],
            "reaction_key": lane["reaction_key"],
            "contract": lane["contract"],
            "iteration": int(iteration),
            "status": "error",
            "policy_sha256": sha256_file(candidate) if candidate.is_file() else None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "evaluated_at": now_iso(),
        }


def lane_ids_from_file(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError("lane ID file is empty")
    if len(values) != len(set(values)):
        raise ValueError("lane ID file contains duplicates")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one immutable reaction batch shard.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--lane-ids-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    validate_iteration(args.iteration)

    root = args.campaign_root.resolve()
    manifest, _state = require_campaign(root)
    ensure_campaign_runner_runtime(root, Path(__file__), manifest)
    by_id = lane_map(manifest)
    lane_ids = lane_ids_from_file(args.lane_ids_file)
    unknown = sorted(set(lane_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown lane IDs: {unknown}")
    lanes = [by_id[lane_id] for lane_id in lane_ids]
    for lane in lanes:
        lane_id = str(lane["lane_id"])
        candidate = candidate_path(root, args.iteration, lane_id)
        record_path = mutation_dir(root, args.iteration, lane_id) / "record.json"
        if not candidate.is_file() or not record_path.is_file():
            raise FileNotFoundError(f"candidate/record missing for {lane_id}")
        record = read_json(record_path)
        if record.get("candidate_policy_sha256") != sha256_file(candidate):
            raise RuntimeError(f"candidate hash drift for {lane_id}")

    lanes.sort(
        key=lambda lane: (len(lane["species"]), int(lane["seed_total_t_gate_count"])),
        reverse=True,
    )
    workers = min(args.workers, len(lanes))
    print(
        f"pid={os.getpid()} iteration={args.iteration} lanes={len(lanes)} workers={workers}",
        flush=True,
    )
    errors = 0
    pending: list[dict[str, Any]] = []
    for lane in lanes:
        lane_id = str(lane["lane_id"])
        candidate_hash = sha256_file(candidate_path(root, args.iteration, lane_id))
        existing = result_path(root, args.iteration, lane_id)
        if (
            existing.is_file()
            and result_validation_error(
                existing,
                lane=lane,
                iteration=args.iteration,
                candidate_hash=candidate_hash,
            )
            is None
        ):
            print(f"{lane_id} status=already_valid", flush=True)
            continue
        pending.append(lane)

    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(pending) or 1))) as pool:
        futures = {
            pool.submit(evaluate_one, str(root), lane, args.iteration): lane["lane_id"]
            for lane in pending
        }
        for future in as_completed(futures):
            result = future.result()
            if result.get("status") != "ok":
                errors += 1
            destination = result_path(root, args.iteration, str(result["lane_id"]))
            write_json(destination, result)
            print(
                f"{result['lane_id']} status={result.get('status')} "
                f"pass={result.get('passes_contract')} T={result.get('total_t_gate_count')}",
                flush=True,
            )
    if errors:
        raise SystemExit(f"{errors} of {len(pending)} attempted lane evaluations failed")


if __name__ == "__main__":
    main()
