#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-hamlib121-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

from campaign_common import (  # noqa: E402
    sha256_file,
    validate_campaign_identity,
    validate_evaluation_result_provenance,
    validate_runner_provenance,
    write_json,
)


EXPECTED_LANES = 121


def closure_sha256(paths: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload_hash = sha256_file(path).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(payload_hash)
    return digest.hexdigest()


def target_result_payload(
    *,
    source_path: Path,
    source_manifest: dict,
    source_identity: dict,
    target: Path,
    target_manifest: dict,
    target_identity: dict,
) -> dict:
    result = json.loads(source_path.read_text(encoding="utf-8"))
    lane_id = str(result["lane_id"])
    source_result_sha256 = sha256_file(source_path)
    for key in (
        "accepted",
        "decision_reason",
        "relative_gain_percent",
        "incumbent_iteration",
        "incumbent_t_gates_qpe",
        "mutation_summary",
    ):
        result.pop(key, None)
    result["iteration"] = 0
    target_policy = target / "lanes" / lane_id / "initial_simple_pf.py"
    if target_policy.is_symlink() or not target_policy.is_file():
        raise RuntimeError(f"{lane_id}: target initial policy is missing or unsafe")
    if sha256_file(target_policy) != result.get("policy_sha256"):
        raise RuntimeError(f"{lane_id}: target initial policy differs from source baseline")
    result["policy_path"] = str(target_policy)
    result["policy_snapshot"] = "initial"
    result["campaign_id"] = target_manifest["campaign_id"]
    result["campaign_manifest_sha256"] = target_identity["campaign_manifest_sha256"]
    result["baseline_replication"] = {
        "schema_version": 1,
        "source_campaign_id": source_manifest["campaign_id"],
        "source_manifest_sha256": source_identity["campaign_manifest_sha256"],
        "source_result_sha256": source_result_sha256,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse one deterministic iteration-0 baseline evaluation in replicate campaigns."
    )
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--target-campaign", type=Path, action="append", required=True)
    args = parser.parse_args()

    source = args.source_campaign.resolve()
    source_manifest, source_identity = validate_campaign_identity(source)
    source_runner = validate_runner_provenance(source)
    source_lanes = {
        str(lane["lane_id"]): lane for lane in source_manifest["lanes"] if lane.get("exact_fit")
    }
    if len(source_lanes) != EXPECTED_LANES:
        raise RuntimeError(f"source campaign must contain exactly {EXPECTED_LANES} active lanes")
    source_results = source / "iterations" / "iter_0000" / "results"
    if source_results.is_symlink() or not source_results.is_dir():
        raise RuntimeError("source iteration-0 result directory is missing or unsafe")
    unsafe_result_entries = [
        path.name
        for path in source_results.iterdir()
        if path.is_symlink() or not path.is_file() or path.suffix != ".json"
    ]
    if unsafe_result_entries:
        raise RuntimeError(
            f"source result directory has undeclared entries: {unsafe_result_entries}"
        )
    result_paths = sorted(source_results.glob("*.json"))
    if len(result_paths) != EXPECTED_LANES or {path.stem for path in result_paths} != set(
        source_lanes
    ):
        raise RuntimeError(
            f"source iteration-0 result closure must contain exactly {EXPECTED_LANES} lane files"
        )
    source_baselines = source / "baselines"
    if source_baselines.is_symlink() or not source_baselines.is_dir():
        raise RuntimeError("source baseline cache directory is missing or unsafe")
    unsafe_baseline_entries = [
        path.name
        for path in source_baselines.iterdir()
        if path.is_symlink() or not path.is_file() or path.suffix != ".json"
    ]
    if unsafe_baseline_entries:
        raise RuntimeError(
            f"source baseline cache directory has undeclared entries: {unsafe_baseline_entries}"
        )
    baseline_paths = sorted(source_baselines.glob("*.json"))
    if len(baseline_paths) != EXPECTED_LANES or {path.stem for path in baseline_paths} != set(
        source_lanes
    ):
        raise RuntimeError(
            f"source baseline cache closure must contain exactly {EXPECTED_LANES} lane files"
        )
    for source_path in result_paths:
        lane_id = source_path.stem
        result = json.loads(source_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok" or result.get("accepted") is not True:
            raise RuntimeError(f"{lane_id}: source iteration-0 baseline is not accepted/ok")
        validate_evaluation_result_provenance(
            campaign_root=source,
            lane=source_lanes[lane_id],
            iteration=0,
            result=result,
            policy_path=source / "lanes" / lane_id / "initial_simple_pf.py",
        )

    # Validate the full target set before writing. Exact partial closures are allowed so a crash
    # during the 242-file copy can resume safely; mismatched or undeclared bytes fail closed.
    target_contexts: list[tuple[Path, dict, dict]] = []
    for target_arg in args.target_campaign:
        target = target_arg.resolve()
        target_manifest, target_identity = validate_campaign_identity(target)
        if target_manifest["campaign_id"] == source_manifest["campaign_id"]:
            raise RuntimeError("replicate campaigns must have independent random campaign IDs")
        if validate_runner_provenance(target) != source_runner:
            raise RuntimeError("target runner provenance differs from source campaign")
        if target_manifest.get("hamlib_archives") != source_manifest.get(
            "hamlib_archives"
        ) or target_manifest.get("hamlib_zip_checksums_sha256") != source_manifest.get(
            "hamlib_zip_checksums_sha256"
        ):
            raise RuntimeError("target HamLib ZIP input binding differs from source campaign")
        target_lanes = {
            str(lane["lane_id"]): lane for lane in target_manifest["lanes"] if lane.get("exact_fit")
        }
        if set(target_lanes) != set(source_lanes):
            raise RuntimeError("target active lane set differs from source campaign")
        target_results = target / "iterations" / "iter_0000" / "results"
        target_baselines = target / "baselines"
        allowed_names = set(source_lanes)
        for directory, label in (
            (target_results, "iteration-0 result"),
            (target_baselines, "baseline cache"),
        ):
            if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
                raise RuntimeError(f"target {label} directory is unsafe: {directory}")
            if directory.exists():
                undeclared = [
                    entry.name
                    for entry in directory.iterdir()
                    if entry.is_symlink()
                    or not entry.is_file()
                    or entry.suffix != ".json"
                    or entry.stem not in allowed_names
                ]
                if undeclared:
                    raise RuntimeError(
                        f"target {label} directory has undeclared entries: {undeclared}"
                    )
        for baseline_path in baseline_paths:
            existing = target_baselines / baseline_path.name
            if existing.exists() and sha256_file(existing) != sha256_file(baseline_path):
                raise RuntimeError(f"target baseline cache differs from source: {existing}")
        for source_path in result_paths:
            expected = target_result_payload(
                source_path=source_path,
                source_manifest=source_manifest,
                source_identity=source_identity,
                target=target,
                target_manifest=target_manifest,
                target_identity=target_identity,
            )
            existing = target_results / source_path.name
            if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != expected:
                raise RuntimeError(f"target replicated result differs from source: {existing}")
        target_contexts.append((target, target_manifest, target_identity))

    for target, target_manifest, target_identity in target_contexts:
        target_results = target / "iterations" / "iter_0000" / "results"
        target_baselines = target / "baselines"
        target_results.mkdir(parents=True, exist_ok=True)
        target_baselines.mkdir(parents=True, exist_ok=True)
        for baseline_path in baseline_paths:
            destination = target_baselines / baseline_path.name
            if destination.exists():
                continue
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                shutil.copy2(baseline_path, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        for source_path in result_paths:
            destination = target_results / source_path.name
            if destination.exists():
                continue
            write_json(
                destination,
                target_result_payload(
                    source_path=source_path,
                    source_manifest=source_manifest,
                    source_identity=source_identity,
                    target=target,
                    target_manifest=target_manifest,
                    target_identity=target_identity,
                ),
            )
        target_baseline_paths = sorted(target_baselines.glob("*.json"))
        target_result_paths = sorted(target_results.glob("*.json"))
        if (
            len(target_baseline_paths) != EXPECTED_LANES
            or len(target_result_paths) != EXPECTED_LANES
        ):
            raise RuntimeError(f"target baseline replication closure is incomplete: {target}")
        receipt = {
            "schema_version": 1,
            "source_campaign_id": source_manifest["campaign_id"],
            "source_manifest_sha256": source_identity["campaign_manifest_sha256"],
            "target_campaign_id": target_manifest["campaign_id"],
            "target_manifest_sha256": target_identity["campaign_manifest_sha256"],
            "lanes": EXPECTED_LANES,
            "baseline_closure_sha256": closure_sha256(target_baseline_paths, root=target),
            "result_closure_sha256": closure_sha256(target_result_paths, root=target),
        }
        receipt_path = target / "manifests" / "baseline_replication_receipt.json"
        if receipt_path.exists():
            if (
                receipt_path.is_symlink()
                or json.loads(receipt_path.read_text(encoding="utf-8")) != receipt
            ):
                raise RuntimeError(f"target baseline replication receipt differs: {receipt_path}")
        else:
            write_json(receipt_path, receipt)
        print(f"reused {len(result_paths)} initial baseline results in {target}")


if __name__ == "__main__":
    main()
